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
    """Plot sentence-level cumulative embedding trajectory."""
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 sentences to create a trajectory plot.")
        return output_path

    unique_ratio = len(np.unique(embeddings_array, axis=0)) / n_points
    use_pca = n_points <= 10 or unique_ratio < 0.5

    if use_pca:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
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

    prompt_groups = {}
    for i, label in enumerate(sentence_labels):
        prefix = label.split("-")[0] if "-" in label else "P1"
        prompt_groups.setdefault(prefix, []).append(i)

    group_colors = {"P1": "#1976D2", "P2": "#388E3C", "P3": "#7B1FA2"}

    for group, indices in prompt_groups.items():
        if len(indices) < 2:
            continue
        pts = coords[indices]
        color = group_colors.get(group, "gray")
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5, linewidth=1.5, zorder=1)

    all_colors = [group_colors.get(label.split("-")[0] if "-" in label else "P1", "#1976D2") for label in sentence_labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=80, edgecolors="white", linewidths=0.5, zorder=2)

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

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=g)
                       for g, c in group_colors.items() if g in prompt_groups]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

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
    """Plot word-level cumulative embedding trajectory."""
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 words to create a trajectory plot.")
        return output_path

    unique_ratio = len(np.unique(embeddings_array, axis=0)) / n_points
    use_pca = n_points <= 10 or unique_ratio < 0.5

    if use_pca:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
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

    prompt_groups = {}
    for i, label in enumerate(word_labels):
        prefix = label.split("-")[0] if "-" in label else "P1"
        prompt_groups.setdefault(prefix, []).append(i)

    group_colors = {"P1": "#F57C00", "P2": "#7B1FA2", "P3": "#00897B"}

    for group, indices in prompt_groups.items():
        if len(indices) < 2:
            continue
        pts = coords[indices]
        color = group_colors.get(group, "gray")
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.4, linewidth=0.8, zorder=1)

    all_colors = [group_colors.get(label.split("-")[0] if "-" in label else "P1", "#F57C00") for label in word_labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=30, edgecolors="white", linewidths=0.3, zorder=2)

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

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6, label=g)
                       for g, c in group_colors.items() if g in prompt_groups]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

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


def plot_common(
    sentence_embeddings: list[list[float]],
    word_embeddings: list[list[float]],
    sentence_labels: list[str],
    word_labels: list[str],
    output_path: str = "common.png",
) -> str:
    """Plot all trajectories from multiple prompts on one shared UMAP space."""
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

    p1_sent = [i for i, l in enumerate(sentence_labels) if l.startswith("P1-")]
    p2_sent = [i for i, l in enumerate(sentence_labels) if l.startswith("P2-")]
    p1_word = [i for i, l in enumerate(word_labels) if l.startswith("P1-")]
    p2_word = [i for i, l in enumerate(word_labels) if l.startswith("P2-")]

    sent_coords = coords[:n_sent]
    word_coords = coords[n_sent:]

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = {"P1-sent": "#1976D2", "P1-word": "#F57C00", "P2-sent": "#388E3C", "P2-word": "#7B1FA2"}
    sizes = {"sent": 100, "word": 25}

    for idx, label_prefix, color, size, is_word in [
        (p1_sent, "P1-S", colors["P1-sent"], sizes["sent"], False),
        (p2_sent, "P2-S", colors["P2-sent"], sizes["sent"], False),
        (p1_word, "P1-W", colors["P1-word"], sizes["word"], True),
        (p2_word, "P2-W", colors["P2-word"], sizes["word"], True),
    ]:
        if len(idx) < 2:
            continue
        pts = sent_coords[idx] if not is_word else word_coords[idx]
        labels = [sentence_labels[i] if not is_word else word_labels[i] for i in idx]
        prompt_name = "Prompt 1" if "P1" in label_prefix else "Prompt 2"
        level = "Words" if is_word else "Sentences"
        legend_label = f"{prompt_name} {level}"

        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5, linewidth=1.5, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=size, edgecolors="white", linewidths=0.5, zorder=3, label=legend_label)

        if not is_word:
            for x, y, lbl in zip(pts[:, 0], pts[:, 1], labels):
                ax.annotate(lbl, (x, y), fontsize=8, fontweight="bold", ha="center", va="bottom", xytext=(0, 8), textcoords="offset points", zorder=4)
        else:
            step = max(1, len(idx) // 10)
            for j in range(0, len(idx), step):
                ax.annotate(labels[j], (pts[j, 0], pts[j, 1]), fontsize=5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", zorder=4)
            if len(idx) % step != 1:
                ax.annotate(labels[-1], (pts[-1, 0], pts[-1, 1]), fontsize=5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", zorder=4)

    ax.set_title("Common comparison — all trajectories (shared UMAP)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved common plot to: {output_path}")
    return output_path
