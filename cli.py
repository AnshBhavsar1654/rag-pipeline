"""
RAG Pipeline — CLI Chatbot

Interactive terminal chatbot for the RAG pipeline.

Usage:
    python cli.py
    python cli.py --config path/to/config.yaml
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI chatbot for the RAG pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    from src.pipeline.rag_pipeline import RAGPipeline
    from src.chat.chatbot import RAGChatbot

    # Initialize
    try:
        pipeline = RAGPipeline.from_config(args.config)
        pipeline.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[TIP] Run `python ingest.py` first to build the vector store.")
        sys.exit(1)

    chatbot = RAGChatbot(pipeline)

    # Chat loop
    print("\n" + "=" * 60)
    print("  [CHAT] RAG Chatbot — CLI Mode")
    print("  Type your questions below. Commands:")
    print("    /clear  — Clear conversation history")
    print("    /quit   — Exit the chatbot")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[BYE] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("\n[BYE] Goodbye!")
            break

        if user_input.lower() in ("/clear", "clear"):
            chatbot.clear_history()
            print("[CLR]  Conversation history cleared.\n")
            continue

        # Get response
        print("\n[THINK] Thinking...\n")
        try:
            response = chatbot.chat(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"[ERROR] Error: {e}\n")


if __name__ == "__main__":
    main()
