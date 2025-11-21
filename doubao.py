import os
import functools
import openai_proxy
import getpass

SUPPORTED_MODELS = [
    "doubao-1.5-thinking-vision-pro-250428",
    "doubao-seed-1.6-250615",
]

def generate_with_proxy(messages, model="doubao-seed-1.6-250615"):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model}")
    user = getpass.getuser()
    transaction_id = f"{user}-{model}"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 OPENAI_API_KEY")

    client = openai_proxy.GptProxy(api_key=api_key)
    rsp = client.generate(
        messages=messages,
        model=model,
        channel_code="doubao",
        transaction_id=transaction_id,
        # thinking={
        #     "type": "disabled" # 1.6 disable thinking
        # }
    )
    if rsp.ok:
        return True, rsp.json()
    else:
        return False, rsp.text


def test_all_models():
    """
    遍历所有支持的模型，发送相同消息并打印结果。
    """
    messages = [{"role": "user", "content": "你叫什么名字？"}]
    for model in SUPPORTED_MODELS:
        print(f"\n=== Testing model: {model} ===")
        success, result = generate_with_proxy(messages, model)
        if success:
            print("Response JSON:", result)
        else:
            print("Error response:", result)

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""
    test_all_models()