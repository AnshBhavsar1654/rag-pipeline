"""
RAG Pipeline — Document Ingestion

Run this script to load, chunk, embed, and index your documents.

Usage:
    python ingest.py
    python ingest.py --config path/to/config.yaml
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG pipeline vector store."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from src.pipeline.rag_pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline.from_config(args.config)
        pipeline.ingest()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
