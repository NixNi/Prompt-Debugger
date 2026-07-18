#!/usr/bin/env python3
"""CLI entry point for embedding trajectory visualization.

Visualizes how a language model's embedding representation evolves as more
text context is accumulated, using LMStudio's OpenAI-compatible API.
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden", module="umap")

from embedding_client import LMStudioClient
from visualizer import plot_combined, plot_sentence_level, plot_word_level, plot_common


def read_text_input(text_arg: str | None, file_arg: str | None) -> str | None:
    """Read text from --text/--text2 or --file/--file2."""
    if text_arg:
        return text_arg
    if file_arg:
        if not os.path.isfile(file_arg):
            print(f"Error: File not found: {file_arg}", file=sys.stderr)
            sys.exit(1)
        with open(file_arg, "r", encoding="utf-8") as f:
            return f.read()
    return None


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
        help="First prompt text input",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to first prompt text file",
    )
    parser.add_argument(
        "--text2",
        type=str,
        default=None,
        help="Second prompt text input (for comparison)",
    )
    parser.add_argument(
        "--file2",
        type=str,
        default=None,
        help="Path to second prompt text file (for comparison)",
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

    text1 = read_text_input(args.text, args.file)
    text2 = read_text_input(args.text2, args.file2)

    if not text1 or not text1.strip():
        if not sys.stdin.isatty():
            text1 = sys.stdin.read()
        else:
            print("Error: No input text provided. Use --text or --file.", file=sys.stderr)
            sys.exit(1)

    if not text1.strip():
        print("Error: Empty input text.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    client = LMStudioClient(base_url=args.url, model=args.model)

    def process_prompt(text: str, name: str) -> dict:
        print(f"\n--- Processing {name} ---")
        sentences = split_sentences(text)
        print(f"  Embedding {len(sentences)} sentences...")
        cumulative_sentence_texts = build_cumulative_texts(sentences)
        sentence_embeddings = client.get_embeddings(cumulative_sentence_texts)

        words = split_words(text)
        if len(words) < 2:
            print(f"  Warning: Need at least 2 words for word-level. Skipping.", file=sys.stderr)
            word_embeddings = []
        else:
            print(f"  Embedding {len(words)} words...")
            cumulative_word_texts = build_cumulative_texts(words)
            word_embeddings = client.get_embeddings(cumulative_word_texts)

        return {
            "sentences": sentences,
            "sentence_embeddings": sentence_embeddings,
            "words": words,
            "word_embeddings": word_embeddings,
        }

    result1 = process_prompt(text1, "Prompt 1")

    print(f"\nGenerating graphs for Prompt 1...")
    s_labels1 = [f"S{i+1}" for i in range(len(result1["sentences"]))]
    w_labels1 = [f"W{i+1}" for i in range(len(result1["words"]))] if result1["word_embeddings"] else []

    plot_combined(
        result1["sentence_embeddings"],
        result1["word_embeddings"],
        s_labels1,
        w_labels1,
        output_path=os.path.join(args.output_dir, "prompt1_combined.png"),
    )
    plot_sentence_level(
        result1["sentence_embeddings"],
        s_labels1,
        output_path=os.path.join(args.output_dir, "prompt1_sentence.png"),
    )
    if result1["word_embeddings"]:
        plot_word_level(
            result1["word_embeddings"],
            w_labels1,
            output_path=os.path.join(args.output_dir, "prompt1_word.png"),
        )

    result2 = None
    if text2 and text2.strip():
        result2 = process_prompt(text2, "Prompt 2")

        print(f"\nGenerating graphs for Prompt 2...")
        s_labels2 = [f"S{i+1}" for i in range(len(result2["sentences"]))]
        w_labels2 = [f"W{i+1}" for i in range(len(result2["words"]))] if result2["word_embeddings"] else []

        plot_combined(
            result2["sentence_embeddings"],
            result2["word_embeddings"],
            s_labels2,
            w_labels2,
            output_path=os.path.join(args.output_dir, "prompt2_combined.png"),
        )
        plot_sentence_level(
            result2["sentence_embeddings"],
            s_labels2,
            output_path=os.path.join(args.output_dir, "prompt2_sentence.png"),
        )
        if result2["word_embeddings"]:
            plot_word_level(
                result2["word_embeddings"],
                w_labels2,
                output_path=os.path.join(args.output_dir, "prompt2_word.png"),
            )

    if result2:
        print(f"\nGenerating common comparison plots...")

        all_s_labels = [f"P1-S{i+1}" for i in range(len(result1["sentences"]))] + \
                       [f"P2-S{i+1}" for i in range(len(result2["sentences"]))]
        all_w_labels = [f"P1-W{i+1}" for i in range(len(result1["words"]))] + \
                       [f"P2-W{i+1}" for i in range(len(result2["words"]))]

        plot_common(
            result1["sentence_embeddings"] + result2["sentence_embeddings"],
            result1["word_embeddings"] + result2["word_embeddings"],
            all_s_labels,
            all_w_labels,
            output_path=os.path.join(args.output_dir, "common.png"),
        )

        plot_sentence_level(
            result1["sentence_embeddings"] + result2["sentence_embeddings"],
            all_s_labels,
            output_path=os.path.join(args.output_dir, "common_sentence.png"),
        )

        if result1["word_embeddings"] and result2["word_embeddings"]:
            plot_word_level(
                result1["word_embeddings"] + result2["word_embeddings"],
                all_w_labels,
                output_path=os.path.join(args.output_dir, "common_word.png"),
            )

    print("\nDone! Output files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
