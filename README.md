# Prompt Debugger

**Visualize how language models understand text — word by word, sentence by sentence.**

Prompt Debugger is a tool that builds embedding trajectories from text input and renders interactive 3D visualizations showing how a model's semantic representation evolves as context accumulates. Built on top of [LMStudio](https://lmstudio.ai/)'s OpenAI-compatible API.

## What It Does

Feed a prompt (or two) into the tool. It will:

1. Split the text into sentences and words
2. Build cumulative embeddings: after each word/sentence, embed the entire accumulated text so far
3. Project the resulting trajectory into 3D using UMAP (or PCA for small datasets)
4. Render interactive HTML plots you can rotate, zoom, and hover over

You can also plot arbitrary reference points (tool calls, expected outputs, etc.) on the same 3D space to see where they fall relative to the prompt trajectory.

## Quick Start

### Prerequisites

- Python 3.10+
- [LMStudio](https://lmstudio.ai/) running locally with an embedding model loaded

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python3 main.py --text "Your prompt here" --model "your-embedding-model"
```

Or use the included example:

```bash
bash examples.sh
```

Open the generated HTML files in `results/` to explore the 3D visualizations.

## Requirements

| Package | Version |
|---|---|
| `openai` | >= 1.0.0 |
| `numpy` | >= 1.24.0 |
| `umap-learn` | >= 0.5.3 |
| `plotly` | >= 5.18.0 |
| `scikit-learn` | >= 1.3.0 |

## CLI Reference

```
python3 main.py [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `--text`, `-t` | First prompt text | — |
| `--file`, `-f` | Path to first prompt file | — |
| `--text2` | Second prompt text (for comparison) | — |
| `--file2` | Path to second prompt file | — |
| `--model`, `-m` | Embedding model name | `qwen-0.6b` |
| `--url`, `-u` | LMStudio base URL | `http://localhost:1234/v1` |
| `--output-dir`, `-o` | Output directory for plots | `results` |
| `--runs`, `-r` | Number of embedding runs to average | `1` |
| `--point`, `-p` | Reference point text (repeatable) | — |
| `--point-radius` | Radius for point markers on graphs | `0.5` |
| `--simplify-epsilon` | Merge points with cosine similarity above this threshold | `0.0` |
| `--timeout` | API request timeout (seconds) | `240` |

## Output Files

| File | Contents |
|---|---|
| `prompt1_combined.html` | Sentence + word trajectories on shared 3D space |
| `prompt1_sentence.html` | Sentence-level trajectory only |
| `prompt1_word.html` | Word-level trajectory only |
| `common.html` | Both prompts on shared 3D space (when `--text2` used) |
| `common_sentence.html` | Sentence-level comparison |
| `common_word.html` | Word-level comparison |

## License

MIT
