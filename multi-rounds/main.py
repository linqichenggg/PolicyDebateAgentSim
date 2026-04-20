import argparse
import json
import os

from matplotlib import pyplot as plt

from world import World


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="political-debate-run", help="Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=3, type=int, help="Average contact count per agent per step.")
    parser.add_argument("--no_init_support", default=70, type=int, help="Initial number of Support agents.")
    parser.add_argument("--no_init_oppose", default=30, type=int, help="Initial number of Oppose agents.")
    parser.add_argument("--no_days", default=10, type=int, help="Total number of simulation days.")
    parser.add_argument("--no_of_runs", default=1, type=int, help="How many repeated runs to execute.")
    parser.add_argument("--offset", default=0, type=int, help="Starting step offset when resuming from checkpoints.")
    parser.add_argument("--max_dialogue_turns", default=3, type=int, help="Maximum turns in one dialogue.")
    parser.add_argument("--dialogue_convergence", default=0.2, type=float, help="Convergence threshold for dialogue stop.")
    parser.add_argument(
        "--save_dialogues",
        action="store_true",
        help="Deprecated: dialogue records are always saved to JSON and TXT.",
    )
    parser.add_argument(
        "--export_eval_pack",
        action="store_true",
        help="Export blinded dialogue set for human evaluation.",
    )
    parser.add_argument("--save_behaviors", action="store_true", help="Save detailed per-agent behavior logs.")
    parser.add_argument("--user_data_file", default="users.csv", help="Path to CSV/XLSX user profile data.")
    parser.add_argument("--checkpoint_interval", default=5, type=int, help="Checkpoint save interval in steps.")
    args = parser.parse_args()

    for i in range(args.no_of_runs):
        output_dir = f"output/run-{i + 1}"
        os.makedirs(output_dir, exist_ok=True)

        model = World(
            args=args,
            initial_support=args.no_init_support,
            initial_oppose=args.no_init_oppose,
            contact_rate=args.contact_rate,
        )
        model.max_dialogue_turns = args.max_dialogue_turns
        model.dialogue_convergence_threshold = args.dialogue_convergence
        model.run_model(checkpoint_path=output_dir, offset=args.offset)

        model_data = model.datacollector.get_model_vars_dataframe()
        expected_rows = args.no_days + 1
        if len(model_data) > expected_rows:
            print(f"WARNING: Row count ({len(model_data)}) exceeds expected ({expected_rows}), truncating...")
            model_data = model_data.iloc[:expected_rows]

        model_data["Step"] = range(len(model_data))
        data_path = f"{output_dir}/{args.name}-data.csv"
        model_data.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")

        final_state = model_data.iloc[-1]
        print("\nFinal state:")
        print(f"Support: {final_state['Support']}")
        print(f"Oppose: {final_state['Oppose']}")
        print(f"Changed: {final_state['Changed']}")
        print(f"Total population: {final_state['Support'] + final_state['Oppose'] + final_state['Changed']}")

        plt.figure(figsize=(10, 6))
        plt.plot(model_data["Step"], model_data["Support"], "b-", label="Support")
        plt.plot(model_data["Step"], model_data["Oppose"], "r-", label="Oppose")
        plt.plot(model_data["Step"], model_data["Changed"], "g-", label="Changed")
        plt.xlabel("step")
        plt.ylabel("number")
        plt.title("Stance Dynamics (Support/Oppose/Changed)")
        plt.legend()
        plt.grid(True)
        fig_path = f"{output_dir}/{args.name}-stance.png"
        plt.savefig(fig_path)
        plt.close()

        dialogue_json_path = f"{output_dir}/{args.name}-dialogues.json"
        dialogue_txt_path = f"{output_dir}/{args.name}-dialogues.txt"
        model.save_dialogue_data(dialogue_json_path)
        model.save_dialogue_transcript(dialogue_txt_path)
        if args.export_eval_pack:
            eval_public_path = f"{output_dir}/{args.name}-eval-public.json"
            eval_key_path = f"{output_dir}/{args.name}-eval-key.json"
            model.save_evaluation_pack(eval_public_path, eval_key_path)

        agent_beliefs = {"agents": []}
        for agent in model.schedule.agents:
            initial_state = "Support" if agent.initial_belief == 1 else "Oppose"
            agent_data = {
                "id": agent.unique_id,
                "name": agent.name,
                "age": agent.age,
                "initial_state": initial_state,
                "final_state": agent.stance_state,
                "belief_history": agent.beliefs,
                "opinion_history": agent.opinions,
                "dialogue_partners": agent.dialogue_partners,
            }
            agent_beliefs["agents"].append(agent_data)

        belief_path = f"{output_dir}/{args.name}-agent-beliefs.json"
        with open(belief_path, "w", encoding="utf-8") as f:
            json.dump(agent_beliefs, f, ensure_ascii=False, indent=2)
        print(f"Agent belief data saved to {belief_path}")

        if args.save_behaviors:
            behavior_path = f"{output_dir}/{args.name}-behaviors.json"
            model.save_agent_behavior_logs(behavior_path)
            print(f"Behavior logs saved to {behavior_path}")
