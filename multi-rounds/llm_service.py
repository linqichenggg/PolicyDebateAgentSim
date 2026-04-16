import os
import json
import time
import random
from zhipuai import ZhipuAI  # 导入智谱AI客户端

def _load_local_secrets():
    """从本地私密配置文件读取密钥，不纳入版本控制。"""
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.local.json")
    if not os.path.exists(secrets_path):
        return {}

    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"读取私密配置失败: {e}")
    return {}


def _resolve_zhipu_api_key():
    # 优先环境变量，其次本地私密文件
    secrets = _load_local_secrets()
    return os.getenv("ZHIPUAI_API_KEY") or secrets.get("zhipuai_api_key", "")


# 初始化智谱AI客户端
zhipu_client = ZhipuAI(api_key=_resolve_zhipu_api_key())

# 使用智谱AI获取响应
def get_completion_from_messages(messages, model="glm-3-turbo", temperature=0):
    """使用智谱AI替代OpenAI API"""
    success = False
    retry = 0
    max_retries = 5
    while retry < max_retries and not success:
        try:
            # 转换消息格式以适应智谱AI API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 调用智谱AI API
            response = zhipu_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature
            )
            success = True
        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(0.5)

    if success:
        return response.choices[0].message.content
    else:
        return "无法获取响应，请检查API密钥或网络连接。"

# 使用智谱AI获取JSON格式的响应
def get_completion_from_messages_json(messages, model="glm-3-turbo", temperature=0):
    """使用智谱AI替代OpenAI API，并返回JSON格式的响应"""
    success = False
    retry = 0
    max_retries = 30
    while retry < max_retries and not success:
        try:
            # 转换消息格式以适应智谱AI API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 更明确的JSON格式要求，包含所有可能需要的字段
            formatted_messages.append({
                "role": "system",
                "content": """请以JSON格式返回响应，格式必须包含以下字段之一或多个：
                1. 对于观点更新：
                   - "tweet": 更新后的观点
                   - "belief": 信任程度 (0或1)
                   - "reasoning": 解释原因
                
                2. 对于对话响应：
                   - "response": 对话内容（必须字段）
                   - "internal_thoughts": 内心想法
                   - "belief_shift": 信念变化
                   - "reasoning": 响应原因
                
                确保返回的是有效的JSON格式。"""
            })
            
            # 调用智谱AI API
            response = zhipu_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature
            )
            success = True
        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(0.5)

    if success:
        content = response.choices[0].message.content
        # 尝试解析JSON，如果失败则格式化为JSON
        try:
            response_data = json.loads(content)
            
            # 验证包含必要字段
            is_dialogue = "response" in messages[0]["content"].lower()
            if is_dialogue and "response" not in response_data:
                print("警告：响应缺少'response'字段，添加默认值")
                response_data["response"] = "我需要进一步思考这个问题。"
            elif not is_dialogue and "tweet" not in response_data:
                print("警告：响应缺少'tweet'字段，添加默认值")
                response_data["tweet"] = "无法形成明确观点。"
                response_data["belief"] = 0
                response_data["reasoning"] = "无法解析响应"
                
            return json.dumps(response_data)
        except:
            # 如果返回的不是有效JSON，尝试提取并格式化
            try:
                # 查找可能的JSON部分
                if "{" in content and "}" in content:
                    json_part = content[content.find("{"):content.rfind("}")+1]
                    # 尝试解析提取的JSON
                    response_data = json.loads(json_part)
                    
                    # 验证所需字段
                    is_dialogue = "response" in messages[0]["content"].lower()
                    if is_dialogue:
                        if "response" not in response_data:
                            response_data["response"] = "我在思考这个问题。"
                    else:
                        if "tweet" not in response_data:
                            response_data["tweet"] = "无法形成明确观点。"
                            response_data["belief"] = 0
                            response_data["reasoning"] = "解析响应时出现问题"
                            
                    return json.dumps(response_data)
                else:
                    # 创建一个适合上下文的默认响应
                    is_dialogue = "dialogue" in messages[0]["content"].lower()
                    if is_dialogue:
                        return json.dumps({
                            "response": "我需要更多时间思考这个问题。",
                            "internal_thoughts": "无法形成清晰想法",
                            "belief_shift": 0,
                            "reasoning": "无法从模型获取有效的JSON响应"
                        })
                    else:
                        return json.dumps({
                            "tweet": "无法解析响应，这是一个模拟的观点。",
                            "belief": 0,
                            "reasoning": "无法从模型获取有效的JSON响应"
                        })
            except:
                # 创建一个适合上下文的默认响应
                is_dialogue = "dialogue" in messages[0]["content"].lower()
                if is_dialogue:
                    return json.dumps({
                        "response": "无法形成有效回应。",
                        "internal_thoughts": "处理错误",
                        "belief_shift": 0,
                        "reasoning": "JSON解析失败"
                    })
                else:
                    return json.dumps({
                        "tweet": "无法解析响应，这是一个模拟的推文。",
                        "belief": 0,
                        "reasoning": "无法从模型获取有效的JSON响应。"
                    })
    else:
        # API调用失败的默认响应
        return json.dumps({
            "response": "无法获取响应，请检查API密钥或网络连接。",
            "tweet": "无法获取响应，请检查API密钥或网络连接。",
            "belief": 0,
            "reasoning": "API调用失败。",
            "internal_thoughts": "连接问题",
            "belief_shift": 0
        })

# 获取短期记忆摘要
def get_summary_short(opinions, topic):
    if not opinions:
        return "没有收集到其他人的观点。"
    
    user_msg = f"""
    请总结以下关于"{topic}"的观点，简明扼要地提取关键信息：

    {opinions}
    """
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

# 获取长期记忆摘要
def get_summary_long(long_mem, short_mem):
    if not long_mem:
        return short_mem
    
    user_msg = f"""
    请将以下两段内容整合为一个连贯的摘要，保留关键信息：

    长期记忆：{long_mem}
    
    新信息：{short_mem}
    """
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

# 话题相关句子
