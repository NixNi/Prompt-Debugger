"""UMAP + plotly 3D interactive visualization for embedding trajectories."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


POINT_COLORS = ["#E53935", "#D81B60", "#00897B", "#FF8F00"]


def _make_reducer(n_points: int, n_components: int = 3):
    """Create a PCA or UMAP reducer based on data characteristics."""
    unique_ratio = len(np.unique(np.random.rand(n_points, 3), axis=0)) / max(n_points, 1)
    use_pca = n_points <= 10 or unique_ratio < 0.5

    if use_pca:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42)
    else:
        import umap
        n_neighbors = min(15, n_points - 1)
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )


def _make_reducer_from_embeddings(embeddings: list[list[float]], n_components: int = 3):
    """Create a reducer fitted to the given embeddings."""
    embeddings_array = np.array(embeddings)
    n_points = len(embeddings_array)
    unique_ratio = len(np.unique(embeddings_array, axis=0)) / n_points
    use_pca = n_points <= 10 or unique_ratio < 0.5

    if use_pca:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        import umap
        n_neighbors = min(15, n_points - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
    reducer.fit(embeddings_array)
    return reducer


def _add_point_overlay(
    fig: go.Figure,
    reducer,
    point_embeddings: list[list[float]],
    point_labels: list[str],
    point_radius: float,
) -> None:
    """Add point overlays as labeled stars with transparent radius spheres."""
    point_coords = reducer.transform(np.array(point_embeddings))
    for i, (x, y, z, label) in enumerate(
        zip(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2], point_labels)
    ):
        color = POINT_COLORS[i % len(POINT_COLORS)]

        # Transparent sphere for radius
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 16)
        sx = x + point_radius * np.outer(np.cos(u), np.sin(v))
        sy = y + point_radius * np.outer(np.sin(u), np.sin(v))
        sz = z + point_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            colorscale=[[0, color], [1, color]],
            opacity=0.15,
            showscale=False,
            name=label,
            showlegend=True,
            hoverinfo="skip",
        ))

        # Star marker
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="diamond", line=dict(width=1, color="white")),
            text=[label],
            textposition="top center",
            name=f"Point: {label}",
            showlegend=True,
        ))


def _build_figure(
    coords_3d: np.ndarray,
    labels: list[str],
    group_color_map: dict[str, str],
    title: str,
    prompt_groups: dict[str, list[int]],
) -> go.Figure:
    """Build a 3D scatter figure with trajectory lines for each group."""
    fig = go.Figure()

    for group, indices in prompt_groups.items():
        if len(indices) < 2:
            continue
        pts = coords_3d[indices]
        color = group_color_map.get(group, "gray")
        group_label = group

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color=color, width=4),
            name=group_label,
            showlegend=True,
            hoverinfo="skip",
        ))

        # Scatter points
        group_labels = [labels[i] for i in indices]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=5, color=color, line=dict(width=0.5, color="white")),
            text=group_labels,
            hovertemplate="%{text}<extra></extra>",
            name=f"{group_label} points",
            showlegend=False,
        ))

        # Annotations for each point
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="text",
            text=group_labels,
            textposition="top center",
            textfont=dict(size=9, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _get_group_prefix(label: str) -> str:
    return label.split("-")[0] if "-" in label else "P1"


def plot_sentence_level(
    cumulative_embeddings: list[list[float]],
    sentence_labels: list[str],
    output_path: str = "sentence_level.html",
    point_embeddings: list[list[float]] | None = None,
    point_labels: list[str] | None = None,
    point_radius: float = 0.5,
) -> str:
    """Plot sentence-level cumulative embedding trajectory in 3D."""
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 sentences to create a trajectory plot.")
        return output_path

    reducer = _make_reducer_from_embeddings(cumulative_embeddings, n_components=3)
    coords = reducer.transform(embeddings_array)

    prompt_groups: dict[str, list[int]] = {}
    for i, label in enumerate(sentence_labels):
        prefix = _get_group_prefix(label)
        prompt_groups.setdefault(prefix, []).append(i)

    group_colors = {"P1": "#1976D2", "P2": "#388E3C", "P3": "#7B1FA2"}

    fig = _build_figure(coords, sentence_labels, group_colors, "Sentence-level embedding trajectory (3D)", prompt_groups)

    if point_embeddings is not None and point_labels is not None:
        _add_point_overlay(fig, reducer, point_embeddings, point_labels, point_radius)

    fig.write_html(output_path)
    print(f"Saved sentence-level plot to: {output_path}")
    return output_path


def plot_word_level(
    cumulative_embeddings: list[list[float]],
    word_labels: list[str],
    output_path: str = "word_level.html",
    point_embeddings: list[list[float]] | None = None,
    point_labels: list[str] | None = None,
    point_radius: float = 0.5,
) -> str:
    """Plot word-level cumulative embedding trajectory in 3D."""
    embeddings_array = np.array(cumulative_embeddings)
    n_points = len(embeddings_array)

    if n_points < 2:
        print("Warning: Need at least 2 words to create a trajectory plot.")
        return output_path

    reducer = _make_reducer_from_embeddings(cumulative_embeddings, n_components=3)
    coords = reducer.transform(embeddings_array)

    prompt_groups: dict[str, list[int]] = {}
    for i, label in enumerate(word_labels):
        prefix = _get_group_prefix(label)
        prompt_groups.setdefault(prefix, []).append(i)

    group_colors = {"P1": "#F57C00", "P2": "#7B1FA2", "P3": "#00897B"}

    fig = _build_figure(coords, word_labels, group_colors, "Word-level embedding trajectory (3D)", prompt_groups)

    if point_embeddings is not None and point_labels is not None:
        _add_point_overlay(fig, reducer, point_embeddings, point_labels, point_radius)

    fig.write_html(output_path)
    print(f"Saved word-level plot to: {output_path}")
    return output_path


def plot_combined(
    sentence_embeddings: list[list[float]],
    word_embeddings: list[list[float]],
    sentence_labels: list[str],
    word_labels: list[str],
    output_path: str = "combined.html",
    point_embeddings: list[list[float]] | None = None,
    point_labels: list[str] | None = None,
    point_radius: float = 0.5,
) -> str:
    """Plot both trajectories on the same shared 3D UMAP space."""
    n_sent = len(sentence_embeddings)
    n_word = len(word_embeddings)

    if n_sent < 2 and n_word < 2:
        print("Warning: Need at least 2 points in either set.")
        return output_path

    all_embeddings = np.array(sentence_embeddings + word_embeddings)
    reducer = _make_reducer_from_embeddings(all_embeddings, n_components=3)
    coords = reducer.fit_transform(all_embeddings)
    sent_coords = coords[:n_sent]
    word_coords = coords[n_sent:]

    fig = go.Figure()

    # Sentence trajectory
    if n_sent >= 2:
        fig.add_trace(go.Scatter3d(
            x=sent_coords[:, 0], y=sent_coords[:, 1], z=sent_coords[:, 2],
            mode="lines+markers+text",
            line=dict(color="#2196F3", width=4),
            marker=dict(size=6, color="#1976D2", line=dict(width=0.5, color="white")),
            text=sentence_labels,
            textposition="top center",
            textfont=dict(size=9, color="#1976D2"),
            name="Sentences",
            hovertemplate="%{text}<extra>Sentence</extra>",
        ))

    # Word trajectory
    if n_word >= 2:
        label_every_n = max(1, n_word // 15)
        display_labels = [word_labels[i] if i % label_every_n == 0 or i == n_word - 1 else "" for i in range(n_word)]

        fig.add_trace(go.Scatter3d(
            x=word_coords[:, 0], y=word_coords[:, 1], z=word_coords[:, 2],
            mode="lines+markers+text",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=3, color="#F57C00", line=dict(width=0.3, color="white")),
            text=display_labels,
            textposition="top center",
            textfont=dict(size=7, color="#F57C00"),
            name="Words",
            hovertemplate="%{text}<extra>Word</extra>",
        ))

    fig.update_layout(
        title=dict(text="Combined embedding trajectory — shared 3D UMAP space", font=dict(size=16)),
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if point_embeddings is not None and point_labels is not None:
        _add_point_overlay(fig, reducer, point_embeddings, point_labels, point_radius)

    fig.write_html(output_path)
    print(f"Saved combined plot to: {output_path}")
    return output_path


def plot_common(
    sentence_embeddings: list[list[float]],
    word_embeddings: list[list[float]],
    sentence_labels: list[str],
    word_labels: list[str],
    output_path: str = "common.html",
    point_embeddings: list[list[float]] | None = None,
    point_labels: list[str] | None = None,
    point_radius: float = 0.5,
) -> str:
    """Plot all trajectories from multiple prompts on one shared 3D UMAP space."""
    n_sent = len(sentence_embeddings)
    n_word = len(word_embeddings)

    if n_sent < 2 and n_word < 2:
        print("Warning: Need at least 2 points in either set.")
        return output_path

    all_embeddings = np.array(sentence_embeddings + word_embeddings)
    reducer = _make_reducer_from_embeddings(all_embeddings, n_components=3)
    coords = reducer.fit_transform(all_embeddings)

    sent_coords = coords[:n_sent]
    word_coords = coords[n_sent:]

    p1_sent = [i for i, l in enumerate(sentence_labels) if l.startswith("P1-")]
    p2_sent = [i for i, l in enumerate(sentence_labels) if l.startswith("P2-")]
    p1_word = [i for i, l in enumerate(word_labels) if l.startswith("P1-")]
    p2_word = [i for i, l in enumerate(word_labels) if l.startswith("P2-")]

    fig = go.Figure()

    configs = [
        (p1_sent, sent_coords, "P1-Sentences", "#1976D2", 4, 6, sentence_labels),
        (p2_sent, sent_coords, "P2-Sentences", "#388E3C", 4, 6, sentence_labels),
        (p1_word, word_coords, "P1-Words", "#F57C00", 2, 3, word_labels),
        (p2_word, word_coords, "P2-Words", "#7B1FA2", 2, 3, word_labels),
    ]

    for indices, all_coords, name, color, line_w, marker_size, labels in configs:
        if len(indices) < 2:
            continue
        pts = all_coords[indices]
        group_labels = [labels[i] for i in indices]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color=color, width=line_w),
            name=name,
            showlegend=True,
            hoverinfo="skip",
        ))

        # Markers
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color, line=dict(width=0.5, color="white")),
            text=group_labels,
            hovertemplate="%{text}<extra></extra>",
            name=f"{name} pts",
            showlegend=False,
        ))

        # Labels
        is_word = "Words" in name
        step = max(1, len(indices) // 10) if is_word else 1
        display_labels = [group_labels[j] if j % step == 0 or j == len(indices) - 1 else "" for j in range(len(indices))]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="text",
            text=display_labels,
            textposition="top center",
            textfont=dict(size=7 if is_word else 9, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text="Common comparison — all trajectories (shared 3D UMAP)", font=dict(size=16)),
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if point_embeddings is not None and point_labels is not None:
        _add_point_overlay(fig, reducer, point_embeddings, point_labels, point_radius)

    fig.write_html(output_path)
    print(f"Saved common plot to: {output_path}")
    return output_path
