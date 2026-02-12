import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.cli_ui import print_startup_ui

def must_get_env(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def main():
    load_dotenv()

    base_url = must_get_env("LLM_BASE_URL")
    model = must_get_env("LLM_MODEL")
    api_key = os.getenv("LLM_API_KEY", "empty").strip() or "empty"

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
    )

    # 多輪對話歷史（含 system 提示）
    history = [
        SystemMessage(content="你是個助理，請用繁體中文回答，回答要清楚、簡潔。")
    ]

    print_startup_ui(
        model=model,
        base_url=base_url,
        version="1.0.0",
        app_name="SmartBI Chat CLI",
        framework="LangChain",
        clear_screen=True,
    )

    """
    Chat Loop
    1、
    """
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Input Error] :{e}. Exiting.")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("[Error] Good")

        history.append(HumanMessage(content=user_input))

        try:
            resp = llm.invoke(history)
            reply = getattr(resp, "content", str(resp)).strip()
        except Exception as e:
            # 不中斷 session，但提示錯誤
            print(f"[ERROR] LLM call failed: {e}")
            # 回滾本輪 user 訊息，避免污染 history
            history.pop()
            continue

        history.append(AIMessage(content=reply))
        print(f"AI> {reply}\n")


if __name__ == "__main__":
    main()
