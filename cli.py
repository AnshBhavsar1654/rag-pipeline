"""
RAG Pipeline — CLI Chatbot

Interactive terminal chatbot for the RAG pipeline. Each pipeline step
(query translation, routing, query construction, retrieval, generation)
is printed as it happens, then the answer streams in.

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

    try:
        pipeline = RAGPipeline.from_config(args.config)
        pipeline.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[TIP] Run `python ingest.py` first to build the index.")
        sys.exit(1)

    chatbot = RAGChatbot(pipeline)

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

        print()
        try:
            streaming = False
            for event in chatbot.events(user_input):
                if event.kind == "token":
                    if not streaming:
                        print("\nAssistant: ", end="", flush=True)
                        streaming = True
                    print(event.text, end="", flush=True)
                elif event.kind == "reset":
                    print("\n\n[Self-RAG] Discarding that answer and trying again...")
                    streaming = False
                elif event.kind == "done":
                    sources = pipeline.format_response("", event.documents)
                    print(f"\n{sources}\n" if sources else "\n")
                # step events are already printed by the pipeline
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")


if __name__ == "__main__":
    main()
