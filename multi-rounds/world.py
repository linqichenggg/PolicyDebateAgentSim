import json
import os
import pickle
import random
from datetime import datetime

import mesa
import psutil
from citizen import Citizen
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from tqdm import tqdm

from utils import (
    DialogueState,
    calculate_final_belief_change,
    create_memory_from_policy_opinion,
    create_social_network,
    load_real_users,
    should_stop_dialogue,
    update_day,
)
from prompt import *


def compute_num_support(model):
    return sum(1 for a in model.schedule.agents if a.stance_state == "Support")


def compute_num_oppose(model):
    return sum(1 for a in model.schedule.agents if a.stance_state == "Oppose")


def compute_num_changed(model):
    return sum(1 for a in model.schedule.agents if a.stance_state == "Changed")


class World(mesa.Model):
    """Simulation world where citizens debate policy topics and update stance states."""

    def __init__(self, args, initial_support=18, initial_oppose=2, contact_rate=3):
        self.initial_support = initial_support
        self.initial_oppose = initial_oppose
        self.population = initial_support + initial_oppose
        self.step_count = args.no_days
        self.offset = 0
        self.name = args.name
        self.topic = random.choice(debate_topics)

        self.support = initial_support
        self.oppose = initial_oppose
        self.changed = 0

        self.daily_new_support_cases = 0
        self.daily_new_oppose_cases = 0
        self.daily_new_changed_cases = 0

        self.track_contact_rate = [0]
        self.list_new_support_cases = [self.initial_support]
        self.list_new_oppose_cases = [self.initial_oppose]
        self.list_new_changed_cases = [0]

        self.current_date = datetime(2025, 6, 1)
        self.contact_rate = contact_rate
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            model_reporters={
                "Support": compute_num_support,
                "Oppose": compute_num_oppose,
                "Changed": compute_num_changed,
            }
        )

        try:
            real_users = load_real_users(args.user_data_file)
            if len(real_users) < self.population:
                raise ValueError(f"Insufficient user rows. Need {self.population}, got {len(real_users)}.")
        except Exception as e:
            print(f"Failed to load user data: {e}")
            raise

        for i in range(self.population):
            agent_id = i
            user_data = real_users[i]

            if i < self.initial_support:
                stance_state = "Support"
                opinion = random.choice(topic_to_sentences[self.topic]["support"])
                opinion = "I support this position: " + opinion
            else:
                stance_state = "Oppose"
                opinion = random.choice(topic_to_sentences[self.topic]["oppose"])
                opinion = "I oppose this position: " + opinion

            citizen = Citizen(
                model=self,
                unique_id=agent_id,
                name=user_data["name"],
                age=random.randrange(60, 90),
                traits=user_data["traits"],
                opinion=opinion,
                qualification=user_data["education"],
                stance_state=stance_state,
                topic=self.topic,
            )
            citizen.self_description = user_data["description"]
            citizen.party_affiliation = user_data.get("party_affiliation", "independent")
            citizen.ideology_score = user_data.get("ideology_score", 0.0)
            citizen.issue_interest = user_data.get("issue_interest", "medium")

            user_policy_opinion = user_data["policy_opinion"]
            if user_policy_opinion:
                if not hasattr(citizen, "long_opinion_memory") or citizen.long_opinion_memory is None:
                    citizen.long_opinion_memory = []
                elif isinstance(citizen.long_opinion_memory, str):
                    citizen.long_opinion_memory = [citizen.long_opinion_memory]

                long_memory = create_memory_from_policy_opinion(user_policy_opinion, citizen.name)
                citizen.long_opinion_memory.append(long_memory)

            self.schedule.add(citizen)

        self.datacollector.collect(self)
        self.social_network = create_social_network(self.schedule.agents)
        self.dialogue_pairs = []
        self.dialogue_records = []
        self.current_simulation_step = 0
        self.max_dialogue_turns = 3
        self.dialogue_convergence_threshold = 0.1

        self.check_consistency()
        print(
            f"Initialization complete: population={self.population}, "
            f"social_edges={self.social_network.number_of_edges()}, topic='{self.topic[:60]}...'"
        )

    def _trait_level(self, value):
        """Normalize trait label into low/medium/high."""
        text = str(value).strip().lower()
        if text in {"high", "h"}:
            return "high"
        if text in {"low", "l"}:
            return "low"
        return "medium"

    def _estimate_susceptibility(self, agent):
        """Estimate persuasion susceptibility from profile traits.

        Higher value => more likely to shift when facing opposite stance.
        """
        openness = self._trait_level(agent.traits.get("openness", "medium"))
        conscientiousness = self._trait_level(agent.traits.get("conscientiousness", "medium"))
        issue_interest = str(getattr(agent, "issue_interest", "medium")).strip().lower()
        ideology_score = abs(float(getattr(agent, "ideology_score", 0.0)))

        susceptibility = 0.18
        if openness == "high":
            susceptibility += 0.12
        elif openness == "low":
            susceptibility -= 0.04

        if conscientiousness == "high":
            susceptibility -= 0.03
        elif conscientiousness == "low":
            susceptibility += 0.03

        if issue_interest == "high":
            susceptibility -= 0.02
        elif issue_interest == "low":
            susceptibility += 0.04

        # Strong ideological prior reduces likelihood of flipping.
        susceptibility -= min(0.08, ideology_score * 0.06)
        return max(0.05, min(0.45, susceptibility))

    def _apply_cross_stance_nudge(self, agent, other_agent, belief_change):
        """Inject bounded stochastic nudge when two opposite stances interact.

        This prevents fully flat dynamics when model outputs conservative belief_shift values.
        """
        current_belief = agent.beliefs[-1]
        other_belief = other_agent.beliefs[-1]
        if current_belief == other_belief:
            return belief_change

        # Keep explicit model signal if it is already strong enough.
        if abs(belief_change) >= 0.25:
            return belief_change

        susceptibility = self._estimate_susceptibility(agent)
        if random.random() > susceptibility:
            return belief_change

        # Direction points toward counterpart stance.
        direction = 1 if other_belief == 1 else -1
        nudge_magnitude = random.uniform(0.22, 0.5)
        nudge = direction * nudge_magnitude

        if hasattr(agent, "log_behavior"):
            agent.log_behavior(
                "cross_stance_nudge",
                {
                    "partner_id": other_agent.unique_id,
                    "belief_before": current_belief,
                    "partner_belief": other_belief,
                    "base_belief_change": belief_change,
                    "applied_nudge": nudge,
                    "susceptibility": susceptibility,
                },
            )

        return belief_change + nudge

    def decide_dialogue_pairs(self):
        """Create dialogue pairs for this step based on the social graph."""
        self.dialogue_pairs = []

        available_agents = list(self.schedule.agents)
        random.shuffle(available_agents)

        while len(available_agents) >= 2:
            agent1 = available_agents.pop(0)

            potential_partners = []
            for agent2 in available_agents:
                if self.social_network.has_edge(agent1.unique_id, agent2.unique_id):
                    strength = self.social_network[agent1.unique_id][agent2.unique_id]["weight"]
                    potential_partners.append((agent2, strength))

            potential_partners.sort(key=lambda x: x[1], reverse=True)

            if potential_partners:
                total_strength = sum(strength for _, strength in potential_partners)
                r = random.uniform(0, total_strength)
                cumulative = 0
                selected_agent = None
                for agent, strength in potential_partners:
                    cumulative += strength
                    if r <= cumulative:
                        selected_agent = agent
                        break
                if not selected_agent:
                    selected_agent = potential_partners[0][0]
            else:
                selected_agent = random.choice(available_agents)

            available_agents.remove(selected_agent)
            self.dialogue_pairs.append((agent1, selected_agent))

            agent1.interaction_history.append(selected_agent.unique_id)
            selected_agent.interaction_history.append(agent1.unique_id)
            agent1.dialogue_partners[selected_agent.unique_id] = agent1.dialogue_partners.get(selected_agent.unique_id, 0) + 1
            selected_agent.dialogue_partners[agent1.unique_id] = selected_agent.dialogue_partners.get(agent1.unique_id, 0) + 1

    def conduct_dialogue(self, agent1, agent2):
        """Run a multi-turn dialogue between two agents and apply belief updates."""
        print(f"Start dialogue: {agent1.name}(ID:{agent1.unique_id}) with {agent2.name}(ID:{agent2.unique_id})")

        dialogue_state = DialogueState(topic=self.topic, agent1_id=agent1.unique_id, agent2_id=agent2.unique_id)
        conversation_history = []

        def append_turn(agent, response_data, turn_idx):
            """Append a fully structured utterance record for auditing."""
            conversation_history.append(
                {
                    "turn": turn_idx,
                    "speaker_id": agent.unique_id,
                    "speaker": agent.name,
                    "content": response_data.get("response", ""),
                    "reasoning": response_data.get("reasoning", ""),
                    "internal_thoughts": response_data.get("internal_thoughts", ""),
                    "belief_shift": response_data.get("belief_shift", 0),
                }
            )

        try:
            response1_data = agent1.generate_dialogue_initiation(agent2)
            if "response" not in response1_data:
                response1_data["response"] = f"I am {agent1.name}. I want to discuss {self.topic}."
        except Exception as e:
            print(f"Error: failed to generate initial dialogue from agent {agent1.name}: {e}")
            response1_data = {
                "response": f"I am {agent1.name}. I want to discuss {self.topic}.",
                "internal_thoughts": "Error when generating initial response.",
                "belief_shift": 0,
                "reasoning": "Error handling fallback.",
            }

        append_turn(agent1, response1_data, 0)
        dialogue_state.update_after_turn(agent1.unique_id, response1_data)

        for turn in range(1, self.max_dialogue_turns + 1):
            try:
                response2_data = agent2.generate_dialogue_response(
                    conversation_history=conversation_history,
                    dialogue_state=dialogue_state,
                    other_agent=agent1,
                )
                if "response" not in response2_data:
                    response2_data["response"] = (
                        f"I am {agent2.name}. Thanks for sharing your view. I am thinking about this."
                    )
            except Exception as e:
                print(f"Error: failed to generate dialogue response from agent {agent2.name}: {e}")
                response2_data = {
                    "response": f"I am {agent2.name}. Thanks for sharing your view. I am thinking about this.",
                    "internal_thoughts": "Error when generating response.",
                    "belief_shift": 0,
                    "reasoning": "Error handling fallback.",
                }

            append_turn(agent2, response2_data, turn * 2 - 1)
            dialogue_state.update_after_turn(agent2.unique_id, response2_data)

            try:
                response1_data = agent1.generate_dialogue_response(
                    conversation_history=conversation_history,
                    dialogue_state=dialogue_state,
                    other_agent=agent2,
                )
                if "response" not in response1_data:
                    response1_data["response"] = f"I am {agent1.name}. Thanks for your reply. Let me think a bit more."
            except Exception as e:
                print(f"Error: failed to generate dialogue response from agent {agent1.name}: {e}")
                response1_data = {
                    "response": f"I am {agent1.name}. Thanks for your reply. Let me think a bit more.",
                    "internal_thoughts": "Error when generating response.",
                    "belief_shift": 0,
                    "reasoning": "Error handling fallback.",
                }

            append_turn(agent1, response1_data, turn * 2)
            dialogue_state.update_after_turn(agent1.unique_id, response1_data)

            try:
                should_stop = should_stop_dialogue(
                    dialogue_state,
                    response1_data,
                    response2_data,
                    max_turns=self.max_dialogue_turns,
                    convergence_threshold=self.dialogue_convergence_threshold,
                )
                if should_stop:
                    print(f"Dialogue ended at round {turn}, reason: {dialogue_state.stop_reason}")
                    break
            except Exception as e:
                print(f"Error: failed to evaluate dialogue stop condition: {e}")
                dialogue_state.stop_reason = "Dialogue stopped due to error handling."
                break

        try:
            belief_change1 = calculate_final_belief_change(agent1, dialogue_state, conversation_history)
        except Exception as e:
            print(f"Error: failed to compute belief change for agent {agent1.name}: {e}")
            belief_change1 = 0

        try:
            belief_change2 = calculate_final_belief_change(agent2, dialogue_state, conversation_history)
        except Exception as e:
            print(f"Error: failed to compute belief change for agent {agent2.name}: {e}")
            belief_change2 = 0

        belief_change1 = self._apply_cross_stance_nudge(agent1, agent2, belief_change1)
        belief_change2 = self._apply_cross_stance_nudge(agent2, agent1, belief_change2)

        try:
            agent1.update_belief_after_dialogue(belief_change1, conversation_history)
        except Exception as e:
            print(f"Error: failed to update belief for agent {agent1.name}: {e}")

        try:
            agent2.update_belief_after_dialogue(belief_change2, conversation_history)
        except Exception as e:
            print(f"Error: failed to update belief for agent {agent2.name}: {e}")

        dialogue_result = {
            "simulation_step": self.current_simulation_step,
            "agents": (agent1.unique_id, agent2.unique_id),
            "agent_names": (agent1.name, agent2.name),
            "history": conversation_history,
            "belief_changes": (belief_change1, belief_change2),
            "final_beliefs": (agent1.beliefs[-1], agent2.beliefs[-1]),
            "stop_reason": dialogue_state.stop_reason,
            "turns": dialogue_state.turn_count,
            "topic": self.topic,
        }
        return conversation_history, belief_change1, belief_change2, dialogue_result

    def step(self):
        """Run one simulation step."""
        self.decide_dialogue_pairs()

        dialogue_results = []
        for agent1, agent2 in self.dialogue_pairs:
            conversation_history, _, _, dialogue_result = self.conduct_dialogue(agent1, agent2)
            print(f"Dialogue finished: {agent1.name} and {agent2.name}, rounds={len(conversation_history)}")
            dialogue_results.append(dialogue_result)

        self.dialogue_records.extend(dialogue_results)
        print(f"Step complete, total dialogue records: {len(self.dialogue_records)}")

        for agent in self.schedule.agents:
            update_day(agent)
            print(f"Agent {agent.unique_id}: {agent.stance_state} (belief={agent.beliefs[-1]})")

        self.check_consistency()
        self.track_contact_rate.append(len(self.dialogue_pairs) * 2)
        print(f"Step summary: dialogue_pairs={len(self.dialogue_pairs)}, population={self.population}")

    def run_model(self, checkpoint_path=None, offset=0):
        """Run the simulation model."""
        self.offset = offset
        expected_steps = self.offset + self.step_count

        for i in tqdm(range(self.offset, self.step_count)):
            self.current_simulation_step = i + 1
            self.step()
            self.datacollector.collect(self)
            if checkpoint_path and i % 10 == 0 and i > 0:
                self.save_checkpoint(checkpoint_path + f"/{self.name}-{i}.pkl")

        model_data = self.datacollector.get_model_vars_dataframe()
        if len(model_data) > expected_steps + 1:
            print(f"WARNING: Too many data points collected: {len(model_data)}, expected {expected_steps + 1}")

    def save_checkpoint(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_checkpoint(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def check_consistency(self):
        """Validate tracked counts against agent-level stance states."""
        total = self.support + self.oppose + self.changed
        if total != self.population:
            print(f"ERROR: Population mismatch! {total} != {self.population}")

        for agent in self.schedule.agents:
            current_belief = agent.beliefs[-1]
            expected_state = (
                "Changed"
                if current_belief != agent.initial_belief
                else ("Support" if current_belief == 1 else "Oppose")
            )
            if agent.stance_state != expected_state:
                print(
                    f"WARNING: Agent {agent.unique_id} inconsistent state: "
                    f"{agent.stance_state}, expected={expected_state}, belief={current_belief}"
                )

        support_count = sum(1 for a in self.schedule.agents if a.stance_state == "Support")
        oppose_count = sum(1 for a in self.schedule.agents if a.stance_state == "Oppose")
        changed_count = sum(1 for a in self.schedule.agents if a.stance_state == "Changed")

        if support_count != self.support:
            print(f"ERROR: Support count mismatch! Actual: {support_count}, Tracked: {self.support}")
        if oppose_count != self.oppose:
            print(f"ERROR: Oppose count mismatch! Actual: {oppose_count}, Tracked: {self.oppose}")
        if changed_count != self.changed:
            print(f"ERROR: Changed count mismatch! Actual: {changed_count}, Tracked: {self.changed}")

    def save_dialogue_data(self, file_path):
        """Save collected dialogue records to JSON."""
        if not self.dialogue_records:
            print("Warning: no dialogue records to save!")
            dialogue_data = {
                "topic": self.topic,
                "population": self.population,
                "initial_support": self.initial_support,
                "initial_oppose": self.initial_oppose,
                "total_dialogues": 0,
                "dialogues": [],
                "warning": "Dialogue records are empty. Check whether conduct_dialogue executed correctly.",
            }
        else:
            dialogue_data = {
                "topic": self.topic,
                "population": self.population,
                "initial_support": self.initial_support,
                "initial_oppose": self.initial_oppose,
                "total_dialogues": len(self.dialogue_records),
                "dialogues": self.dialogue_records,
            }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
        print(f"Dialogue data saved to {file_path}")

    def save_dialogue_transcript(self, file_path):
        """Save human-readable dialogue transcript text."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Topic: {self.topic}\n")
            f.write(f"Population: {self.population}\n")
            f.write(f"Initial Support: {self.initial_support}, Initial Oppose: {self.initial_oppose}\n")
            f.write(f"Total Dialogues: {len(self.dialogue_records)}\n\n")

            if not self.dialogue_records:
                f.write("No dialogue records.\n")
            else:
                for idx, dialogue in enumerate(self.dialogue_records, start=1):
                    agent_names = dialogue.get("agent_names", ["unknown", "unknown"])
                    f.write(f"=== Dialogue {idx} | step={dialogue.get('simulation_step', 0)} ===\n")
                    f.write(f"Agents: {agent_names[0]} <-> {agent_names[1]}\n")
                    f.write(f"Stop Reason: {dialogue.get('stop_reason', 'N/A')}\n")
                    f.write(f"Belief Changes: {dialogue.get('belief_changes', [0, 0])}\n")
                    f.write(f"Final Beliefs: {dialogue.get('final_beliefs', [None, None])}\n")
                    f.write("Transcript:\n")
                    for turn in dialogue.get("history", []):
                        f.write(
                            f"[t{turn.get('turn', '?')}] "
                            f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}\n"
                        )
                    f.write("\n")

        print(f"Dialogue transcript saved to {file_path}")

    def save_evaluation_pack(self, public_path, key_path):
        """Export blinded dialogue data for human-vs-agent evaluation."""
        public_samples = []
        key_items = []

        for idx, dialogue in enumerate(self.dialogue_records, start=1):
            sample_id = f"D{idx:06d}"
            names = dialogue.get("agent_names", ["SpeakerA", "SpeakerB"])
            name_to_alias = {
                names[0]: "SpeakerA",
                names[1]: "SpeakerB",
            }

            public_turns = []
            for turn in dialogue.get("history", []):
                original_speaker = turn.get("speaker", "Unknown")
                content = turn.get("content", "")
                for real_name, alias in name_to_alias.items():
                    content = content.replace(real_name, alias)
                public_turns.append(
                    {
                        "turn": turn.get("turn", 0),
                        "speaker": name_to_alias.get(original_speaker, "Unknown"),
                        "content": content,
                    }
                )

            public_samples.append(
                {
                    "sample_id": sample_id,
                    "topic": dialogue.get("topic", self.topic),
                    "dialogue": public_turns,
                }
            )

            key_items.append(
                {
                    "sample_id": sample_id,
                    "simulation_step": dialogue.get("simulation_step", 0),
                    "agent_ids": dialogue.get("agents", []),
                    "agent_names": names,
                    "belief_changes": dialogue.get("belief_changes", [0, 0]),
                    "final_beliefs": dialogue.get("final_beliefs", [None, None]),
                    "stop_reason": dialogue.get("stop_reason", "N/A"),
                }
            )

        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": self.topic,
                    "total_samples": len(public_samples),
                    "samples": public_samples,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(key_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": self.topic,
                    "total_samples": len(key_items),
                    "key": key_items,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Evaluation public set saved to {public_path}")
        print(f"Evaluation key saved to {key_path}")

    def save_agent_behavior_logs(self, file_path):
        """Save per-agent behavior logs to JSON."""
        behavior_data = {
            "simulation_info": {
                "topic": self.topic,
                "population": self.population,
                "initial_support": self.initial_support,
                "initial_oppose": self.initial_oppose,
                "total_steps": self.schedule.steps,
                "start_date": self.current_date.strftime("%Y-%m-%d"),
            },
            "agents": [],
        }

        for agent in self.schedule.agents:
            initial_state = "Support" if agent.initial_belief == 1 else "Oppose"
            agent_data = {
                "id": agent.unique_id,
                "name": agent.name,
                "traits": agent.traits,
                "education": agent.qualification,
                "party_affiliation": getattr(agent, "party_affiliation", "independent"),
                "ideology_score": getattr(agent, "ideology_score", 0.0),
                "issue_interest": getattr(agent, "issue_interest", "medium"),
                "self_description": agent.self_description if hasattr(agent, "self_description") else "",
                "initial_state": initial_state,
                "final_state": agent.stance_state,
                "behavior_log": agent.behavior_log,
            }
            behavior_data["agents"].append(agent_data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(behavior_data, f, ensure_ascii=False, indent=2)
        print(f"Agent behavior logs saved to {file_path}")

    def monitor_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        print(f"Current memory usage: {memory_usage_mb:.2f} MB")
        return memory_usage_mb
