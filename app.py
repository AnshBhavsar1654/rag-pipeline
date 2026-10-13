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
from src.chat.chatbot import format_history, messages_from_gradio
from src.vectorstore.store import VectorStoreNotReadyError


def create_app(config_path: str | None = None, share: bool = False):
    """Create and launch the Gradio chatbot application."""

    # Initialize pipeline (auto-ingest if there is no usable vectorstore)
    pipeline = RAGPipeline.from_config(config_path)
    try:
        pipeline.load()
    except VectorStoreNotReadyError as e:
        print(f"[INFO] {e}")
        print("[INFO] Ingesting documents from config...")
        try:
            pipeline.ingest()
        except Exception as e:
            print(f"[ERROR] Ingestion failed: {e}")
        if not pipeline.is_ready:
            print("[INFO] Starting with empty vector store. Upload documents via the UI.")

    ui_config = pipeline.config.ui
    memory_window = pipeline.config.chat.memory_window

    def header() -> str:
        strategy = pipeline.strategy_name if pipeline.is_ready else "N/A (ingest documents to start)"
        return (
            f"# {ui_config.title}\n\n"
            f"{ui_config.description}\n\n"
            f"**Retrieval Strategy:** {strategy} "
            f"| **LLM:** {pipeline.config.llm.model}"
        )

    # ---- Handlers ----

    def upload_files(files):
        """Handle uploaded files: ingest into vector store."""
        if not files:
            return "No files selected.", header()
        try:
            result = pipeline.ingest_files(list(files))
        except Exception as e:
            result = f"Ingestion failed: {e}"
        return result, header()

    def respond(message: str, history: list[dict]):
        """Stream the answer to *message*, using this session's chat history."""
        if not message.strip():
            yield ""
            return

        if not pipeline.is_ready:
            yield "No documents loaded yet. Upload and ingest documents first, then try again."
            return

        # History comes from the browser session, so each user has their own
        # conversation and the UI's clear button really resets it.
        chat_history = format_history(messages_from_gradio(history), memory_window)

        answer = ""
        try:
            stream, documents = pipeline.stream_query(message, chat_history)
            for piece in stream:
                answer += piece
                yield answer
            yield pipeline.format_response(answer, documents)
        except Exception as e:
            yield f"{answer}\n\n**Error:** {e}".strip()

    # ---- Build the UI ----

    theme_map = {
        "soft": gr.themes.Soft(),
        "default": gr.themes.Default(),
        "glass": gr.themes.Glass(),
    }
    theme = theme_map.get(ui_config.theme, gr.themes.Soft())

    with gr.Blocks(title=ui_config.title) as app:

        header_md = gr.Markdown(header())

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
            outputs=[upload_status, header_md],
        )

        gr.Markdown("---")

        gr.ChatInterface(
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
