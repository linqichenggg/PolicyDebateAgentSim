import random
import os
import json
import networkx as nx
import numpy as np
from user_data import load_real_users
from prompt import *

def update_day(agent):
    """Update a single agent stance state from the latest belief value.

    State semantics:
    - Support: current belief=1 and unchanged from initial belief
    - Oppose: current belief=0 and unchanged from initial belief
    - Changed: current belief differs from initial belief
    """
    current_belief = agent.beliefs[-1]
    initial_belief = agent.initial_belief

    if current_belief != initial_belief:
        target_state = "Changed"
    elif current_belief == 1:
        target_state = "Support"
    else:
        target_state = "Oppose"

    old_state = agent.stance_state
    if old_state == target_state:
        return

    # Decrement old state counter.
    if old_state == "Support":
        agent.model.support -= 1
    elif old_state == "Oppose":
        agent.model.oppose -= 1
    elif old_state == "Changed":
        agent.model.changed -= 1

    # Increment new state counter and daily transition tracker.
    if target_state == "Support":
        agent.model.support += 1
        agent.model.daily_new_support_cases += 1
    elif target_state == "Oppose":
        agent.model.oppose += 1
        agent.model.daily_new_oppose_cases += 1
    else:
        agent.model.changed += 1
        agent.model.daily_new_changed_cases += 1

    agent.stance_state = target_state
    if hasattr(agent, "log_behavior"):
        reason = (
            "Current belief differs from initial stance."
            if target_state == "Changed"
            else f"Current belief aligns with initial {target_state.lower()} stance."
        )
        agent.log_behavior(
            "stance_state_change",
            {
                "before": old_state,
                "after": target_state,
                "current_belief": current_belief,
                "initial_belief": initial_belief,
                "reason": reason,
            },
        )

def clear_cache():
    """Remove files under the local cache directory."""
    cache_dir = ".cache"
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, file))
        print("Done")

def create_social_network(agents, connection_probability=0.2):
    G = nx.Graph()
    
    for agent in agents:
        G.add_node(agent.unique_id)
    
    for i, agent1 in enumerate(agents):
        max_connections = min(20, len(agents)//5)
        current_connections = 0
        
        for j, agent2 in enumerate(agents[i+1:], i+1):
            if current_connections >= max_connections:
                break
                
            p = connection_probability
            
            if agent1.qualification == agent2.qualification:
                p += 0.1 
            
            p = max(0.05, min(0.9, p))
            
            if random.random() < p:
                weight = random.uniform(0.5, 1.0)
                G.add_edge(agent1.unique_id, agent2.unique_id, weight=weight)
                current_connections += 1
    
    return G

class DialogueState:
    """Track dialogue-level state shared across both agents in one conversation."""
    def __init__(self, topic, agent1_id, agent2_id):
        self.topic = topic
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.turn_count = 0
        self.stance_strength = {
            agent1_id: 0,
            agent2_id: 0
        }
        self.common_ground = 0
        self.belief_shifts = {
            agent1_id: 0,
            agent2_id: 0
        }
        self.stop_reason = None
    
    def update_after_turn(self, agent_id, response_data):
        """Update dialogue state after one agent response."""
        if agent_id == self.agent1_id:
            self.turn_count += 0.5 
        
        if "stance_strength" in response_data:
            self.stance_strength[agent_id] = response_data["stance_strength"]
        elif "internal_thoughts" in response_data:
            thoughts = response_data["internal_thoughts"].lower()
            if "certain" in thoughts or "confident" in thoughts:
                self.stance_strength[agent_id] = 1.0
            elif "doubt" in thoughts or "uncertain" in thoughts:
                self.stance_strength[agent_id] = 0.5
            elif "disagree" in thoughts or "reject" in thoughts:
                self.stance_strength[agent_id] = -1.0
        
        if "common_ground" in response_data:
            self.common_ground = response_data["common_ground"]
        
        if "belief_shift" in response_data:
            self.belief_shifts[agent_id] += response_data["belief_shift"]

def should_stop_dialogue(dialogue_state, response1, response2, max_turns=3, convergence_threshold=0.1):
    """Return True when dialogue should stop by turn limit, convergence, or explicit ending."""
    if dialogue_state.turn_count >= max_turns:
        dialogue_state.stop_reason = "Maximum turns reached"
        return True
    
    recent_shift1 = abs(response1.get("belief_shift", 0))
    recent_shift2 = abs(response2.get("belief_shift", 0))
    
    if recent_shift1 < convergence_threshold and recent_shift2 < convergence_threshold:
        dialogue_state.stop_reason = "Dialogue converged"
        return True
    
    if "response" in response1 and ("end" in response1["response"].lower() or "goodbye" in response1["response"].lower()):
        dialogue_state.stop_reason = "Agent 1 explicitly ended"
        return True
    
    if "response" in response2 and ("end" in response2["response"].lower() or "goodbye" in response2["response"].lower()):
        dialogue_state.stop_reason = "Agent 2 explicitly ended"
        return True
    
    return False

def calculate_final_belief_change(agent, dialogue_state, conversation_history):
    """Compute final belief change for an agent from dialogue state and history."""
    agent_id = agent.unique_id
    
    belief_change = dialogue_state.belief_shifts.get(agent_id, 0)
    
    support_shift_phrases = [
        "i now support",
        "i support this",
        "you are right about this policy",
        "i can accept this position",
    ]
    oppose_shift_phrases = [
        "i now oppose",
        "i oppose this",
        "i cannot support this policy",
        "this argument does not hold",
    ]
    conversion_phrases = [
        "i changed my mind",
        "you convinced me",
    ]

    for turn in conversation_history:
        if turn["speaker"] == agent.name:
            content = turn["content"].lower()
            if any(phrase in content for phrase in conversion_phrases):
                if agent.beliefs[-1] == 1:
                    belief_change = -1.0
                else:
                    belief_change = 1.0
                break
            if any(phrase in content for phrase in support_shift_phrases):
                belief_change = max(belief_change, 0.6)
            elif any(phrase in content for phrase in oppose_shift_phrases):
                belief_change = min(belief_change, -0.6)
    
    if dialogue_state.turn_count < 1:
        belief_change *= 0.5
    
    return belief_change

def format_dialogue_history(conversation_history):
    """Format structured dialogue history into a plain text transcript."""
    if not conversation_history:
        return "(No dialogue history)"
    
    formatted = ""
    for turn in conversation_history:
        formatted += f"{turn['speaker']}: {turn['content']}\n"
    
    return formatted

def get_dialogue_summary(dialogue_content, topic):
    """Summarize dialogue content for memory updates."""
    user_msg = dialogue_summary_prompt.format(
        dialogue_content=dialogue_content,
        topic=topic
    )
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

def create_memory_from_policy_opinion(policy_opinion, name):
    """Create initial long-term memory text from user policy opinion."""
    if not policy_opinion:
        return ""
    
    memory = f"My name is {name}, and my policy view is: {policy_opinion}\n\n"
    memory += "This view comes from my values, experiences, and understanding of public issues."
    
    return memory
