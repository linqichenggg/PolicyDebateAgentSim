# PolicyDebateAgentSim

This project simulates multi-round political stance discussions among agents and tracks stance dynamics with three states:
- `Support`
- `Oppose`
- `Changed`

It is an experimental framework for LLM-based agent social simulation. Agents represent policy discussants with profiles, Big Five traits, policy opinions, dialogue memory, and evolving stance states.

The current implementation focuses on policy debate simulation, dialogue recording, and opinion-change tracking.

## 1. Project Structure

- `multi-rounds/main.py`: Entry point.
- `multi-rounds/world.py`: Simulation loop, pairing, dialogue orchestration, data export.
- `multi-rounds/citizen.py`: Agent behavior and stance update logic.
- `multi-rounds/llm_service.py`: LLM API integration (`POST /v1/chat/completions`).
- `multi-rounds/topic_library.py`: Debate topics and stance statements.
- `multi-rounds/prompt_templates.py`: Prompt templates for dialogue and reflection.
- `multi-rounds/user_data.py`: User profile data loading.

## 2. Method Overview

The simulation follows this process:

1. Load agent profiles from `users.csv`.
2. Initialize agents with Support/Oppose stances, Big Five traits, policy opinions, and memory.
3. Build a weighted undirected social network with `networkx`.
4. Pair agents by social-network edge weights.
5. Run multi-round LLM dialogues through the DeepSeek chat-completions API.
6. Compute belief changes and update each agent's stance state.
7. Export dialogue records, agent trajectories, and daily Support/Oppose/Changed counts.

## 3. Environment Setup

From project root:

```bash
cd PolicyDebateAgentSim
```

Install dependencies:

```bash
python -m pip install -U pandas matplotlib networkx numpy tqdm psutil "mesa==1.2.1"
```

## 4. API Key Configuration (DeepSeek)

Supported priority order:
1. Environment variables
2. `multi-rounds/secrets.local.json` (recommended, gitignored)

### Option A: Environment Variables

```bash
export DEEPSEEK_API_KEY="YOUR_KEY"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/chat/completions"
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_MAX_TOKENS="512"
export DEEPSEEK_TOP_P="0.95"
```

### Option B: Local Secrets File (recommended)

Create `multi-rounds/secrets.local.json`:

```json
{
  "deepseek_api_key": "YOUR_KEY",
  "deepseek_base_url": "https://api.deepseek.com/chat/completions",
  "deepseek_model": "deepseek-chat",
  "deepseek_max_tokens": 512,
  "deepseek_top_p": 0.95
}
```

Reference: [DeepSeek API Docs](https://api-docs.deepseek.com/zh-cn/)

## 5. Input Data

Default user profile file: `users.csv` (project root).

Required columns (minimum):
- `username`
- `policy_opinion`

Common optional columns:
- `education`
- Big Five traits (`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`)
- `party_affiliation`
- `ideology_score`
- `issue_interest`

## 6. Run

### Quick Smoke Test

Use this command first to confirm that the environment, user data, and API configuration work correctly.

```bash
python -u multi-rounds/main.py \
  --user_data_file users.csv \
  --no_days 1 \
  --no_init_support 1 \
  --no_init_oppose 1 \
  --max_dialogue_turns 2 \
  --dialogue_convergence 0.1 \
  --name smoke
```

### Medium Development Run

```bash
python -u multi-rounds/main.py \
  --user_data_file users.csv \
  --no_days 3 \
  --no_init_support 6 \
  --no_init_oppose 6 \
  --max_dialogue_turns 2 \
  --dialogue_convergence 0.1 \
  --name mid-12x3-run1 \
  --export_eval_pack
```

### Larger Experimental Run

Increase `--no_init_support`, `--no_init_oppose`, and `--no_days` only after the smoke test and medium run finish correctly.

```bash
python -u multi-rounds/main.py \
  --user_data_file users.csv \
  --no_days 10 \
  --no_init_support 50 \
  --no_init_oppose 50 \
  --max_dialogue_turns 2 \
  --dialogue_convergence 0.1 \
  --name run-100x10 \
  --export_eval_pack \
  --save_behaviors
```

## 7. Outputs

Outputs are saved under `output/run-1/` (or run index).

Core outputs:
- `*-data.csv`: Step-level stance counts.
- `*-stance.png`: Stance curve figure.
- `*-agent-beliefs.json`: Agent-level initial/final state and histories.
- `*-dialogues.json`: Full structured dialogue records (includes reasoning fields).
- `*-dialogues.txt`: Readable transcript text.

Evaluation outputs (`--export_eval_pack`):
- `*-eval-public.json`: Blinded set for volunteers (`SpeakerA/SpeakerB`, content only).
- `*-eval-key.json`: Mapping and metadata for researcher-only use.

Behavior logs (`--save_behaviors`):
- `*-behaviors.json`: Per-agent behavior records and state-change logs.

## 8. Known Limitations

- `contact_rate` is currently stored as a parameter, but it does not yet scale the number of daily pairings.
- Pairing is based on weighted social-network edges and does not force Support/Oppose cross-stance matching.
- `checkpoint_interval` exists as a CLI parameter, but checkpoint saving is currently fixed in the model loop.
- API-based LLM outputs may not be perfectly reproducible across repeated runs.

## 9. Notes

- If the API key is missing, the system falls back to placeholder responses and still runs.
- This project is configured to use `deepseek-chat`.
- For formal experiments, always run with valid API key and keep `*-eval-key.json` private.
