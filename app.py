"""
RAG Pipeline — Gradio Web UI

Beautiful chat interface for the RAG pipeline with document upload.

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

    # Initialize pipeline (auto-ingest if vectorstore doesn't exist)
    pipeline = RAGPipeline.from_config(config_path)
    try:
        pipeline.load()
    except FileNotFoundError:
        print("[INFO] No vectorstore found. Ingesting documents from config...")
        try:
            pipeline.ingest()
            pipeline.load()
        except Exception as e:
            print(f"[ERROR] Ingestion failed: {e}")
            print("[INFO] Starting with empty vector store. Upload documents via the UI.")

    chatbot = RAGChatbot(pipeline)
    ui_config = pipeline.config.ui

    # ---- Handlers ----

    def upload_files(files):
        """Handle uploaded files: ingest into vector store."""
        if not files:
            return "No files selected."

        file_paths = [f for f in files]
        result = pipeline.ingest_files(file_paths)
        return result

    def respond(message: str, history: list[dict[str, str]]):
        """Process a user message and yield the response."""
        if not message.strip():
            return ""

        if chatbot.pipeline._retriever is None:
            return "No documents loaded yet. Upload and ingest documents first, then try again."

        response = chatbot.chat(message)
        return response

    def clear_chat():
        """Clear chat history."""
        chatbot.clear_history()
        return [], ""

    # ---- Build the UI ----

    theme_map = {
        "soft": gr.themes.Soft(),
        "default": gr.themes.Default(),
        "glass": gr.themes.Glass(),
    }
    theme = theme_map.get(ui_config.theme, gr.themes.Soft())

    with gr.Blocks(title=ui_config.title) as app:

        strategy_name = (
            chatbot.pipeline._retriever.strategy_name
            if chatbot.pipeline._retriever
            else "N/A (ingest documents to start)"
        )
        gr.Markdown(
            f"# {ui_config.title}\n\n"
            f"{ui_config.description}\n\n"
            f"**Retrieval Strategy:** {strategy_name} "
            f"| **LLM:** {pipeline.config.llm.model}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".csv", ".html", ".htm"],
                    type="filepath",
                )
                upload_btn = gr.Button("Ingest Documents", variant="primary", size="sm")
                upload_status = gr.Markdown("*Upload PDF, TXT, MD, DOCX, CSV, or HTML files.*")

        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[upload_status],
        )

        gr.Markdown("---")

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
