"""
RAG Pipeline — Gradio Web UI

Beautiful chat interface for the RAG pipeline.

Usage:
    python app.py
    python app.py --config path/to/config.yaml
    python app.py --share  (public link)
"""

import argparse

import gradio as gr

from src.pipeline.rag_pipeline import RAGPipeline
from src.chat.chatbot import RAGChatbot


def create_app(config_path: str | None = None, share: bool = False):
    """Create and launch the Gradio chatbot application."""

    # Initialize pipeline
    pipeline = RAGPipeline.from_config(config_path)
    pipeline.load()

    chatbot = RAGChatbot(pipeline)
    ui_config = pipeline.config.ui

    # ---- Gradio interface ----

    def respond(message: str, history: list[dict[str, str]]):
        """Process a user message and yield the response."""
        if not message.strip():
            return ""

        response = chatbot.chat(message)
        return response

    def clear_chat():
        """Clear chat history."""
        chatbot.clear_history()
        return [], ""

    # Build the UI
    theme_map = {
        "soft": gr.themes.Soft(),
        "default": gr.themes.Default(),
        "glass": gr.themes.Glass(),
    }
    theme = theme_map.get(ui_config.theme, gr.themes.Soft())

    with gr.Blocks(title=ui_config.title) as app:

        gr.Markdown(
            f"# {ui_config.title}\n\n"
            f"{ui_config.description}\n\n"
            f"**Retrieval Strategy:** {chatbot.pipeline._retriever.strategy_name} "
            f"| **LLM:** {pipeline.config.llm.model}"
        )

        chat_interface = gr.ChatInterface(
            fn=respond,
            examples=[
                "What is this document about?",
                "Summarize the key points.",
                "Explain the main concepts.",
            ],
        )

    # Launch
    app.launch(
        server_port=ui_config.port,
        share=share or ui_config.share,
        show_error=True,
        theme=theme,
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        footer { display: none !important; }
        """,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Launch the RAG Chatbot web UI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    args = parser.parse_args()

    create_app(config_path=args.config, share=args.share)


if __name__ == "__main__":
    main()
