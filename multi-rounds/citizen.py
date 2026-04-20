import mesa
from utils import get_completion_from_messages, get_completion_from_messages_json, format_dialogue_history
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
import json
from prompt import *
import random

def get_summary_long(long_mem, short_mem):
    if not long_mem:
        return short_mem
    
    user_msg = f"""
    Integrate the two pieces of content below into one coherent summary while keeping key information.

    Long-term memory: {long_mem}
    
    New information: {short_mem}
    """
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

def get_summary_short(opinions, topic):
    opinions_text = "\n".join(f"- {opinion}" for opinion in opinions) if opinions else "- No external opinions."
    user_msg = f"""Summarize the following public opinions about this topic:
Topic: {topic}

Opinions:
{opinions_text}
"""
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)

def get_dialogue_summary(dialogue_content, topic):
    user_msg = dialogue_summary_prompt.format(dialogue_content=dialogue_content, topic=topic)
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)

class Citizen(mesa.Agent):
    """Agent with profile traits, stance state, and evolving opinions."""

    def __init__(self, model, unique_id, name, age, traits, qualification, stance_state, opinion, topic, initial_memory=""):
        super().__init__(unique_id, model)
        self.name = name
        self.age = age
        self.opinion = opinion
        self.traits = traits
        self.qualification = qualification
        self.topic = topic
        self.opinions = []
        self.beliefs = []
        self.long_opinion_memory = []
        self.long_memory_full = []
        self.short_opinion_memory = []
        self.reasonings = []
        self.contact_ids = []
        self.dialogue_history = []
        self.interaction_history = []
        self.dialogue_partners = {}
        self.dialogue_summaries = {}
        self.self_description = ""

        self.stance_state = stance_state

        self.agent_interaction = []

        self.initial_belief = 1 if stance_state == 'Support' else 0
        self.initial_reasoning = 'initial_reasoning'
        self.opinions.append(self.opinion)
        self.beliefs.append(self.initial_belief)
        self.reasonings.append(self.initial_reasoning)

        self.behavior_log = []
        
        self.log_behavior("initialization", {
            "belief": 1 if stance_state == "Support" else 0,
            "stance_state": stance_state,
            "opinion": opinion,
            "date": model.current_date.strftime("%Y-%m-%d")
        })

    def initial_opinion_belief(self):
        if self.stance_state == 'Support':
            belief = 1
        else:  # Oppose / Changed
            belief = 0

        reasoning = 'initial_reasoning'

        return belief, reasoning

    def interact(self):
        """Run single-round interaction and update opinion/belief."""
        others_opinions = [agent.opinions[-1] for agent in self.agent_interaction]
        opinion_short_summary = get_summary_short(others_opinions, topic=self.topic)
        
        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)
        
        user_msg = update_opinion_prompt.format(
            agent_name=self.name,
            openness=self.traits["openness"],
            conscientiousness=self.traits["conscientiousness"],
            extraversion=self.traits["extraversion"],
            agreeableness=self.traits["agreeableness"],
            neuroticism=self.traits["neuroticism"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            opinion="[IMPORTANT] " + self.opinion,
            long_mem=long_mem,
            others_opinions=opinion_short_summary
        )
        
        self.opinion, self.belief, self.reasoning = self.response_and_belief(user_msg)
        self.opinions.append(self.opinion)
        self.beliefs.append(self.belief)
        self.reasonings.append(self.reasoning)
        
        print(f"ID: {self.unique_id}")
        print(f"Tweet: {self.opinion}")
        print(f"Belief: {self.belief}")
        print(f"Reasoning: {self.reasoning}")
        print("-" * 50)
        
        self.long_opinion_memory = long_mem
        
        self.agent_interaction = []
    
    def generate_dialogue_initiation(self, other_agent):
        """Generate the opening response in a dialogue."""
        user_msg = dialogue_initiation_prompt.format(
            agent_name=self.name,
            openness=self.traits["openness"],
            conscientiousness=self.traits["conscientiousness"],
            extraversion=self.traits["extraversion"],
            agreeableness=self.traits["agreeableness"],
            neuroticism=self.traits["neuroticism"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            current_opinion=self.opinions[-1],
            other_name=other_agent.name
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            response_data = json.loads(response_json)
            self.log_behavior("dialogue_start", {
                "partner_id": other_agent.unique_id,
                "partner_name": other_agent.name,
                "response": response_data["response"] if "response" in response_data else "",
                "topic": self.topic
            })
            return response_data
        except:
            return {
                "response": f"I am {self.name}. I want to discuss my view on {self.topic}.",
                "internal_thoughts": "I will share my perspective first.",
                "belief_shift": 0,
                "reasoning": "This is the opening turn."
            }
    
    def generate_dialogue_response(self, conversation_history, dialogue_state, other_agent):
        """Generate one response turn within an ongoing dialogue."""
        turn_number = int(dialogue_state.turn_count) + 1
        
        if conversation_history:
            other_response = conversation_history[-1]["content"]
        else:
            other_response = "(Dialogue begins)"
        
        formatted_history = format_dialogue_history(conversation_history)
        
        user_msg = multi_turn_dialogue_prompt.format(
            agent_name=self.name,
            openness=self.traits["openness"],
            conscientiousness=self.traits["conscientiousness"],
            extraversion=self.traits["extraversion"],
            agreeableness=self.traits["agreeableness"],
            neuroticism=self.traits["neuroticism"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            current_opinion=self.opinions[-1],
            other_name=other_agent.name,
            turn_number=turn_number,
            conversation_history=formatted_history,
            other_response=other_response
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            response_data = json.loads(response_json)
            self.log_behavior("dialogue_reply", {
                "partner_id": other_agent.unique_id,
                "partner_name": other_agent.name,
                "response": response_data["response"] if "response" in response_data else "",
                "turn_count": dialogue_state.turn_count,
                "common_ground": dialogue_state.common_ground
            })
            return response_data
        except:
            return {
                "response": "I understand your point, but I need more time to think.",
                "internal_thoughts": "I am not fully convinced by the other side yet.",
                "belief_shift": 0,
                "reasoning": "I need more information before making a judgment."
            }
    
    def update_belief_after_dialogue(self, belief_change, conversation_history):
        """Update belief and opinion after dialogue ends."""
        current_belief = self.beliefs[-1]
        
        new_belief_value = current_belief
        
        # Lower threshold helps capture realistic moderate persuasion in policy debates.
        if abs(belief_change) >= 0.2:
            if current_belief == 1 and belief_change < 0:
                new_belief_value = 0
            elif current_belief == 0 and belief_change > 0:
                new_belief_value = 1
        
        if new_belief_value != current_belief:
            if new_belief_value == 1:
                new_opinion = f"I now support this position: {random.choice(topic_to_sentences[self.topic]['support'])}"
            else:
                new_opinion = f"I now oppose this position: {random.choice(topic_to_sentences[self.topic]['oppose'])}"
            
            new_reasoning = "My view changed through dialogue."
            for turn in conversation_history:
                if turn["speaker"] == self.name and "reasoning" in turn:
                    new_reasoning = turn["reasoning"]
                    break
        else:
            if current_belief == 1:
                new_opinion = f"I still support this position: {random.choice(topic_to_sentences[self.topic]['support'])}"
            else:
                new_opinion = f"I still oppose this position: {random.choice(topic_to_sentences[self.topic]['oppose'])}"
            
            new_reasoning = self.reasonings[-1]
        
        self.beliefs.append(new_belief_value)
        self.opinions.append(new_opinion)
        self.reasonings.append(new_reasoning)
        
        print(f"Agent {self.unique_id} belief update: {current_belief} -> {new_belief_value}")
        print(f"New opinion: {new_opinion}")
        print(f"Reasoning: {new_reasoning}")
        
        dialogue_content = format_dialogue_history(conversation_history)
        try:
            dialogue_summary = get_dialogue_summary(dialogue_content, self.topic)
        except Exception:
            dialogue_summary = (
                f"Fallback dialogue summary for topic '{self.topic}'. "
                f"Latest stance={new_belief_value}, turns={len(conversation_history)}."
            )

        self.update_long_memory_with_dialogue(dialogue_summary)
        
        self.log_behavior("belief_update", {
            "belief_before_dialogue": current_belief,
            "belief_after_dialogue": new_belief_value,
            "belief_change": belief_change,
            "dialogue_partner": conversation_history[0]['speaker'] if conversation_history else "unknown",
            "final_opinion": new_opinion if new_opinion else "",
            "stance_state": self.stance_state
        })
        
        return new_belief_value
    
    def update_long_memory_with_dialogue(self, dialogue_summary):
        """Merge dialogue summary into long-term memory."""
        if not self.long_opinion_memory:
            self.long_opinion_memory = dialogue_summary
            return
        
        updated_memory = get_summary_long(self.long_opinion_memory, dialogue_summary)
        self.long_opinion_memory = updated_memory
        self.long_memory_full.append(updated_memory)

    def response_and_belief(self, user_msg):
        """Call the model and parse tweet/belief/reasoning fields."""
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=1)
        try:
            output = json.loads(response_json)
            tweet = output['tweet']
            belief = int(output['belief'])
            reasoning = output['reasoning']
            return tweet, belief, reasoning
        except:
            return "Unable to parse model response.", 0, "Parsing error"

    def step(self):
        """Agent step hook (managed externally by World)."""
        pass

    def update_long_memory(self):
        """Regenerate long-term memory summary from current memory content."""
        if not self.long_opinion_memory:
            return
        
        user_msg = long_memory_prompt.format(
            agent_name=self.name,
            openness=self.traits["openness"],
            conscientiousness=self.traits["conscientiousness"],
            extraversion=self.traits["extraversion"],
            agreeableness=self.traits["agreeableness"],
            neuroticism=self.traits["neuroticism"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            long_mem=self.long_opinion_memory
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response = get_completion_from_messages(msg, temperature=0.7)
        
        self.long_opinion_memory = response
        self.long_memory_full.append(response)

    def reflect_on_community(self, community_opinions):
        """Reflect on community opinions and optionally update belief."""
        user_msg = reflecting_prompt.format(
            agent_name=self.name,
            openness=self.traits["openness"],
            conscientiousness=self.traits["conscientiousness"],
            extraversion=self.traits["extraversion"],
            agreeableness=self.traits["agreeableness"],
            neuroticism=self.traits["neuroticism"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            opinion=self.opinions[-1],
            long_mem=self.long_opinion_memory,
            community_opinions=community_opinions
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            output = json.loads(response_json)
            reflection = output.get('reflection', "Unable to generate reflection.")
            updated_belief = int(output.get('updated_belief', self.beliefs[-1]))
            reasoning = output.get('reasoning', "Unable to explain reasoning.")
            
            if updated_belief != self.beliefs[-1]:
                if updated_belief == 1:
                    new_opinion = f"After reflection, I support this position: {random.choice(topic_to_sentences[self.topic]['support'])}"
                else:
                    new_opinion = f"After reflection, I oppose this position: {random.choice(topic_to_sentences[self.topic]['oppose'])}"
                
                self.opinions.append(new_opinion)
                self.beliefs.append(updated_belief)
                self.reasonings.append(reasoning)
                
                print(f"Agent {self.unique_id} updated belief via reflection: {self.beliefs[-2]} -> {updated_belief}")
                print(f"Reflection: {reflection}")
                print(f"New opinion: {new_opinion}")
                print(f"Reasoning: {reasoning}")
            
            return reflection, updated_belief, reasoning
        except:
            print(f"Agent {self.unique_id} reflection processing failed")
            return "Reflection processing failed", self.beliefs[-1], "Processing error"

    def get_long_term_memory(self):
        """Return long-term memory as a single string."""
        if not self.long_opinion_memory:
            return ""
        elif isinstance(self.long_opinion_memory, str):
            return self.long_opinion_memory
        else:
            return "\n\n".join(self.long_opinion_memory)

    def log_behavior(self, action_type, details):
        """Record an agent behavior event.

        Args:
            action_type: Behavior type label.
            details: Dictionary containing behavior details.
        """
        timestamp = self.model.current_date.strftime("%Y-%m-%d") + f" (step {self.model.schedule.steps})"
        
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.unique_id,
            "agent_name": self.name,
            "action_type": action_type,
            "details": details
        }
        
        self.behavior_log.append(log_entry)
