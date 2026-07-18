#!/usr/bin/env python3
"""CLI entry point for embedding trajectory visualization.

Visualizes how a language model's embedding representation evolves as more
text context is accumulated, using LMStudio's OpenAI-compatible API.
"""

import argparse
import os
import re
import string
import sys

from embedding_client import LMStudioClient
from visualizer import plot_combined, plot_sentence_level, plot_word_level


def read_text_input(args: argparse.Namespace) -> str:
    """Read text from --text, --file, or stdin."""
    if args.text:
        return args.text
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    # Read from stdin
    return sys.stdin.read()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def split_words(text: str) -> list[str]:
    """Split text into words, preserving punctuation attached to words."""
    words = text.strip().split()
    return [w for w in words if w]


def build_cumulative_texts(parts: list[str]) -> list[str]:
    """Build cumulative texts: [p1, p1+p2, p1+p2+p3, ...]."""
    cumulative = []
    current = ""
    for part in parts:
        current = (current + " " + part).strip() if current else part
        cumulative.append(current)
    return cumulative


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how a language model's understanding evolves with context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reads text from --text, --file, or stdin.\n"
            "Requires LMStudio running locally with an embedding model loaded."
        ),
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Direct text input string",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to text file to read",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen-0.6b",
        help="Model name (default: qwen-0.6b)",
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:1234/v1",
        help="LMStudio base URL (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for graphs (default: results)",
    )
    args = parser.parse_args()

    # Read input text
    text = read_text_input(args)
    if not text or not text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize client
    client = LMStudioClient(base_url=args.url, model=args.model)

    # --- Sentence-level processing ---
    sentences = split_sentences(text)
    print(f"Embedding {len(sentences)} sentences...")
    cumulative_sentence_texts = build_cumulative_texts(sentences)
    sentence_embeddings = client.get_embeddings(cumulative_sentence_texts)
    print(f"  Computed {len(sentence_embeddings)} cumulative sentence embeddings.")

    # --- Word-level processing ---
    words = split_words(text)
    if len(words) < 2:
        print("Warning: Need at least 2 words for word-level graph. Skipping.", file=sys.stderr)
        word_embeddings = []
    else:
        print(f"Embedding {len(words)} words...")
        cumulative_word_texts = build_cumulative_texts(words)
        word_embeddings = client.get_embeddings(cumulative_word_texts)
        print(f"  Computed {len(word_embeddings)} cumulative word embeddings.")

    # --- Generate visualizations ---
    print("Generating graphs...")

    sentence_labels = [f"S{i+1}" for i in range(len(sentences))]
    word_labels = [f"W{i+1}" for i in range(len(words))] if word_embeddings else []

    combined_output = os.path.join(args.output_dir, "combined.png")
    plot_combined(
        sentence_embeddings,
        word_embeddings,
        sentence_labels,
        word_labels,
        output_path=combined_output,
    )

    sentence_output = os.path.join(args.output_dir, "sentence_level.png")
    plot_sentence_level(sentence_embeddings, sentence_labels, output_path=sentence_output)

    if word_embeddings:
        word_output = os.path.join(args.output_dir, "word_level.png")
        plot_word_level(word_embeddings, word_labels, output_path=word_output)

    print("\nDone! Output files:")
    print(f"  Combined:        {os.path.abspath(combined_output)}")
    print(f"  Sentence-level:  {os.path.abspath(sentence_output)}")
    if word_embeddings:
        print(f"  Word-level:      {os.path.abspath(word_output)}")


if __name__ == "__main__":
    main()
