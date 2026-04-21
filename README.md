# PolicyDebateAgentSim

This project simulates multi-round political stance discussions among agents and tracks stance dynamics with three states:
- `Support`
- `Oppose`
- `Changed`

It is adapted from an earlier multi-agent simulation baseline into a policy debate setting.

## 1. Project Structure

- `multi-rounds/main.py`: Entry point.
- `multi-rounds/world.py`: Simulation loop, pairing, dialogue orchestration, data export.
- `multi-rounds/citizen.py`: Agent behavior and stance update logic.
- `multi-rounds/llm_service.py`: LLM API integration (`POST /v1/chat/completions`).
- `multi-rounds/topic_library.py`: Debate topics and stance statements.
- `multi-rounds/prompt_templates.py`: Prompt templates for dialogue and reflection.
- `multi-rounds/user_data.py`: User profile data loading.

## 2. Environment Setup

From project root:

```bash
cd /Users/lqcmacmini/code/anything/PolicyDebateAgentSim
```

Install dependencies:

```bash
python -m pip install -U pandas matplotlib networkx numpy tqdm psutil "mesa==1.2.1"
```

## 3. API Key Configuration (DeepSeek)

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

## 4. Input Data

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

## 5. Run

### Quick Smoke Test

```bash
python -u /Users/lqcmacmini/code/anything/PolicyDebateAgentSim/multi-rounds/main.py \
  --user_data_file /Users/lqcmacmini/code/anything/PolicyDebateAgentSim/users.csv \
  --no_days 1 \
  --no_init_support 10 \
  --no_init_oppose 10 \
  --name smoke
```

### Standard Run

```bash
python -u /Users/lqcmacmini/code/anything/PolicyDebateAgentSim/multi-rounds/main.py \
  --user_data_file /Users/lqcmacmini/code/anything/PolicyDebateAgentSim/users.csv \
  --no_days 10 \
  --no_init_support 70 \
  --no_init_oppose 30 \
  --name run1 \
  --export_eval_pack
```

## 6. Outputs

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

## 7. Notes

- If the API key is missing, the system falls back to placeholder responses and still runs.
- This project is configured to use `deepseek-chat`.
- For formal experiments, always run with valid API key and keep `*-eval-key.json` private.
