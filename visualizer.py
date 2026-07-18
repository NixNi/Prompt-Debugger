"""UMAP + matplotlib visualization for embedding trajectories."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import umap


def plot_sentence_level(
    cumulative_embeddings: list[list[float]],
    sentence_labels: list[str],
    output_path: str = "sentence_level.png",
) -> str:
    """Plot sentence-level cumulative embedding trajectory.

    Args:
        cumulative_embeddings: List of embeddings where each embedding represents
            the cumulative context up to that sentence.
        sentence_labels: Human-readable labels for each point.
        output_path: Path to save the output PNG.

    Returns:
        The output_path on success.
    """
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 sentences to create a trajectory plot.")
        return output_path

    if n_points < 5:
        print("Warning: UMAP results may be unstable with fewer than 5 points.")

    # For very small datasets, use PCA fallback instead of UMAP
    if n_points <= 5:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings_array)
    else:
        n_neighbors = min(15, n_points - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings_array)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by index to show progression
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_points))

    # Draw trajectory line first (behind points)
    ax.plot(coords[:, 0], coords[:, 1], color="gray", alpha=0.4, linewidth=1, zorder=1)

    # Scatter with gradient colors
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(n_points),
        cmap="viridis",
        s=80,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    # Annotate each point
    for i, (x, y, label) in enumerate(zip(coords[:, 0], coords[:, 1], sentence_labels)):
        ax.annotate(
            label,
            (x, y),
            fontsize=8,
            ha="center",
            va="bottom",
            xytext=(0, 8),
            textcoords="offset points",
            zorder=3,
        )

    ax.set_title("Sentence-level embedding trajectory", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    plt.colorbar(scatter, ax=ax, label="Context progression", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sentence-level plot to: {output_path}")
    return output_path


def plot_word_level(
    cumulative_embeddings: list[list[float]],
    word_labels: list[str],
    output_path: str = "word_level.png",
) -> str:
    """Plot word-level cumulative embedding trajectory.

    Args:
        cumulative_embeddings: List of embeddings where each embedding represents
            the cumulative context up to that word.
        word_labels: Human-readable labels for each point.
        output_path: Path to save the output PNG.

    Returns:
        The output_path on success.
    """
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 words to create a trajectory plot.")
        return output_path

    if n_points < 5:
        print("Warning: UMAP results may be unstable with fewer than 5 points.")

    if n_points <= 5:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings_array)
    else:
        n_neighbors = min(15, n_points - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings_array)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw trajectory line behind points
    ax.plot(coords[:, 0], coords[:, 1], color="gray", alpha=0.3, linewidth=0.8, zorder=1)

    # Scatter with gradient colors (smaller points for word level)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(n_points),
        cmap="viridis",
        s=30,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    # Auto-calculate label stride to avoid overlap
    label_every_n = max(1, n_points // 20)

    # Annotate every Nth word
    for i in range(0, n_points, label_every_n):
        x, y = coords[i, 0], coords[i, 1]
        ax.annotate(
            word_labels[i],
            (x, y),
            fontsize=6,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            zorder=3,
        )
    # Always label the last point
    if (n_points - 1) % label_every_n != 0:
        x, y = coords[-1, 0], coords[-1, 1]
        ax.annotate(
            word_labels[-1],
            (x, y),
            fontsize=6,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            zorder=3,
        )

    ax.set_title("Word-level embedding trajectory", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    plt.colorbar(scatter, ax=ax, label="Context progression", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved word-level plot to: {output_path}")
    return output_path


def plot_combined(
    sentence_embeddings: list[list[float]],
    word_embeddings: list[list[float]],
    sentence_labels: list[str],
    word_labels: list[str],
    output_path: str = "combined.png",
) -> str:
    """Plot both trajectories on the same UMAP space."""
    n_sent = len(sentence_embeddings)
    n_word = len(word_embeddings)

    if n_sent < 2 and n_word < 2:
        print("Warning: Need at least 2 points in either set.")
        return output_path

    all_embeddings = np.array(sentence_embeddings + word_embeddings)
    n_total = len(all_embeddings)
    n_neighbors = min(15, n_total - 1) if n_total > 2 else 2

    if n_total <= 5:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )

    coords = reducer.fit_transform(all_embeddings)
    sent_coords = coords[:n_sent]
    word_coords = coords[n_sent:]

    fig, ax = plt.subplots(figsize=(12, 9))

    if n_sent >= 2:
        ax.plot(sent_coords[:, 0], sent_coords[:, 1], color="#2196F3", alpha=0.6, linewidth=2, zorder=1)
        ax.scatter(sent_coords[:, 0], sent_coords[:, 1], c="#1976D2", s=100, edgecolors="white", linewidths=1, zorder=3, label="Sentences")
        for i, (x, y, label) in enumerate(zip(sent_coords[:, 0], sent_coords[:, 1], sentence_labels)):
            ax.annotate(label, (x, y), fontsize=9, fontweight="bold", ha="center", va="bottom", xytext=(0, 10), textcoords="offset points", zorder=4)

    if n_word >= 2:
        ax.plot(word_coords[:, 0], word_coords[:, 1], color="#FF9800", alpha=0.4, linewidth=1, zorder=1)
        ax.scatter(word_coords[:, 0], word_coords[:, 1], c="#F57C00", s=25, edgecolors="white", linewidths=0.3, zorder=2, label="Words")
        label_every_n = max(1, n_word // 15)
        for i in range(0, n_word, label_every_n):
            ax.annotate(word_labels[i], (word_coords[i, 0], word_coords[i, 1]), fontsize=6, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", zorder=4)
        if (n_word - 1) % label_every_n != 0:
            ax.annotate(word_labels[-1], (word_coords[-1, 0], word_coords[-1, 1]), fontsize=6, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", zorder=4)

    ax.set_title("Combined embedding trajectory (shared UMAP space)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to: {output_path}")
    return output_path
