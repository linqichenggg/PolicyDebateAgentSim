import json
import os
import time
import urllib.error
import urllib.request

_API_WARNING_SHOWN = False


def _is_placeholder_value(value):
    text = str(value).strip()
    if not text:
        return True
    upper = text.upper()
    return upper.startswith("YOUR_") or upper in {"CHANGE_ME", "REPLACE_ME"}


def _load_local_secrets():
    """Load secrets from local file only."""
    path = os.path.join(os.path.dirname(__file__), "secrets.local.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"Failed to read local secrets: {e}")
    return {}


def _resolve_deepseek_api_key():
    secrets = _load_local_secrets()
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key and not _is_placeholder_value(env_key):
        return env_key

    local_key = str(secrets.get("deepseek_api_key", "")).strip()
    if local_key and not _is_placeholder_value(local_key):
        return local_key
    return ""


def _resolve_deepseek_base_url():
    secrets = _load_local_secrets()
    value = os.getenv("DEEPSEEK_BASE_URL") or secrets.get(
        "deepseek_base_url",
        "https://api.deepseek.com/chat/completions",
    )
    return str(value).strip()


def _resolve_default_model():
    secrets = _load_local_secrets()
    value = os.getenv("DEEPSEEK_MODEL") or secrets.get("deepseek_model", "deepseek-chat")
    return str(value).strip()


def _resolve_generation_params():
    secrets = _load_local_secrets()
    max_tokens_raw = os.getenv("DEEPSEEK_MAX_TOKENS") or secrets.get("deepseek_max_tokens", 512)
    top_p_raw = os.getenv("DEEPSEEK_TOP_P") or secrets.get("deepseek_top_p", 0.95)
    try:
        max_tokens = int(max_tokens_raw)
    except Exception:
        max_tokens = 512
    try:
        top_p = float(top_p_raw)
    except Exception:
        top_p = 0.95
    max_tokens = max(32, min(4096, max_tokens))
    top_p = max(0.1, min(1.0, top_p))
    return max_tokens, top_p


def _resolve_endpoint():
    base_url = _resolve_deepseek_base_url().rstrip("/")
    if "/chat/completions" in base_url:
        return base_url
    return f"{base_url}/chat/completions"


def _extract_text_content(message_obj):
    content = message_obj.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def _is_dialogue_request(messages):
    if not messages:
        return False
    try:
        content = str(messages[0].get("content", "")).lower()
    except Exception:
        return False
    markers = ["response", "internal_thoughts", "belief_shift", "dialogue", "conversation"]
    return any(marker in content for marker in markers)


def _chat_completion(messages, model=None, temperature=0, max_retries=3):
    api_key = _resolve_deepseek_api_key()
    if not api_key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or deepseek_api_key in secrets.local.json.")

    endpoint = _resolve_endpoint()
    model_name = model or _resolve_default_model()
    max_tokens, top_p = _resolve_generation_params()

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    retry = 0
    last_error = "Unknown error"
    while retry < max_retries:
        try:
            request = urllib.request.Request(
                endpoint,
                data=payload_bytes,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                response_body = response.read().decode("utf-8")
            response_data = json.loads(response_body)
            return _extract_text_content(response_data["choices"][0]["message"])
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            last_error = f"HTTP {e.code}: {detail}"
            if e.code in {400, 401, 403, 404}:
                break
            retry += 1
            print(f"HTTP Error: {e.code} {detail}\nRetrying...")
            time.sleep(0.5)
        except Exception as e:
            last_error = str(e)
            retry += 1
            print(f"Error: {e}\nRetrying...")
            time.sleep(0.5)

    return f"Unable to get response. {last_error}"


def get_completion_from_messages(messages, model=None, temperature=0):
    global _API_WARNING_SHOWN
    try:
        return _chat_completion(messages=messages, model=model, temperature=temperature, max_retries=3)
    except RuntimeError as e:
        if not _API_WARNING_SHOWN:
            print(f"Warning: {e} Falling back to local placeholder responses.")
            _API_WARNING_SHOWN = True
        return "Model unavailable. Fallback summary generated locally."


def get_completion_from_messages_json(messages, model=None, temperature=0):
    formatted_messages = list(messages)
    formatted_messages.append(
        {
            "role": "system",
            "content": """Return valid JSON. Include one of the following schemas:
                1) Opinion update:
                   - "tweet": updated opinion
                   - "belief": stance value (1=Support, 0=Oppose)
                   - "reasoning": short reason

                2) Dialogue response:
                   - "response": dialogue text (required)
                   - "internal_thoughts": internal thoughts
                   - "belief_shift": belief shift
                   - "reasoning": response reason
                """,
        }
    )
    is_dialogue = _is_dialogue_request(messages)

    try:
        content = _chat_completion(
            messages=formatted_messages,
            model=model,
            temperature=temperature,
            max_retries=3,
        )
    except RuntimeError as e:
        global _API_WARNING_SHOWN
        if not _API_WARNING_SHOWN:
            print(f"Warning: {e} Falling back to local placeholder responses.")
            _API_WARNING_SHOWN = True
        content = ""

    try:
        response_data = json.loads(content)
        if is_dialogue and "response" not in response_data:
            response_data["response"] = "I need more time to think about this."
        elif not is_dialogue and "tweet" not in response_data:
            response_data["tweet"] = "Unable to form a clear opinion."
            response_data["belief"] = 0
            response_data["reasoning"] = "Failed to parse response."
        return json.dumps(response_data)
    except Exception:
        try:
            if "{" in content and "}" in content:
                json_part = content[content.find("{"):content.rfind("}") + 1]
                response_data = json.loads(json_part)
                if is_dialogue:
                    if "response" not in response_data:
                        response_data["response"] = "I am still thinking about this."
                else:
                    if "tweet" not in response_data:
                        response_data["tweet"] = "Unable to form a clear opinion."
                        response_data["belief"] = 0
                        response_data["reasoning"] = "Error while parsing response."
                return json.dumps(response_data)

            if is_dialogue:
                return json.dumps(
                    {
                        "response": "I need more time to think about this.",
                        "internal_thoughts": "I cannot form a clear thought yet.",
                        "belief_shift": 0,
                        "reasoning": "Model did not return valid JSON.",
                    }
                )
            return json.dumps(
                {
                    "tweet": "Unable to parse response. This is a fallback opinion.",
                    "belief": 0,
                    "reasoning": "Model did not return valid JSON.",
                }
            )
        except Exception:
            if is_dialogue:
                return json.dumps(
                    {
                        "response": "Unable to generate a valid response.",
                        "internal_thoughts": "Processing error.",
                        "belief_shift": 0,
                        "reasoning": "JSON parsing failed.",
                    }
                )
            return json.dumps(
                {
                    "tweet": "Unable to parse response. This is a fallback tweet.",
                    "belief": 0,
                    "reasoning": "Model did not return valid JSON.",
                }
            )


def get_summary_short(opinions, topic):
    if not opinions:
        return "No opinions were collected from others."

    user_msg = f"""
    Summarize the following opinions about "{topic}" and extract key points concisely:

    {opinions}
    """

    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)


def get_summary_long(long_mem, short_mem):
    if not long_mem:
        return short_mem

    user_msg = f"""
    Integrate the two texts below into one coherent summary while preserving key information:

    Long-term memory: {long_mem}

    New information: {short_mem}
    """

    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)
