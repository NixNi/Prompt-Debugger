# Prompt Debugger — Reading the Diagram

## What You're Looking At

An interactive 3D scatter plot. Each point is an embedding vector — a numerical snapshot of how the model "understands" all the text accumulated up to that moment. Points are connected by lines to form a **trajectory**: the path the model's understanding takes as words are added one by one.

You can rotate, zoom, and hover over any point to see the exact text it represents.

## The Two Trajectories

Every prompt generates two trajectories plotted on the same 3D space:

- **Sentence trajectory** (blue line, larger dots) — one point per sentence. Each point embeds everything from the first sentence up to that sentence. Labels: S1, S2, S3, ...
- **Word trajectory** (orange line, smaller dots) — one point per word. Each point embeds everything from the first word up to that word. Labels: W1, W2, W3, ...

The sentence trajectory is a coarse view (big jumps). The word trajectory is a fine-grained view (step by step).

## What a Trajectory Point Means

Point W5 doesn't mean "word 5". It means "the entire text from word 1 through word 5, embedded as a single vector." The trajectory shows how the model's holistic representation shifts with each addition.

## How to Read the Shape

### Direction changes = semantic pivots

When the trajectory turns sharply, a word or sentence significantly altered the model's understanding. A flat segment means the model's view barely changed — those words added redundant or expected information.

**Example**: If you add "Paris" after "the weather in", you might see a sharp turn because the model now knows you're asking about a specific location. Adding "today" after that might barely move the point — the temporal framing is a minor detail compared to the location.

### Distance between points = magnitude of semantic shift

Far apart = the new word/sentence changed a lot. Close together = the model already "expected" that content. Points clustering tightly means the text is semantically homogeneous — the model's understanding stabilizes.

### Clusters = thematic regions

If the trajectory loops back or clusters, the text covers overlapping themes. The model keeps returning to the same semantic neighborhood because the content reinforces the same meaning.

### Trajectory length = complexity of the prompt

A long, winding trajectory means the prompt covers diverse topics or shifts meaning. A short, straight trajectory means the prompt is focused and semantically consistent.

## Comparing Two Prompts

When you provide `--text` and `--text2`, a shared comparison plot appears. Look for:

### Overlapping regions = shared semantics

If the two trajectories pass through the same area of 3D space, those portions of the prompts are semantically similar — even if the exact words differ completely. This is the distributional hypothesis in action: different words that appear in similar contexts end up near each other.

### Divergence points = where meanings part ways

Find where the two trajectories split. That's where the prompts stop being about the same thing. Trace backwards from the divergence to find the exact word or sentence that pulled them apart.

### Relative position = thematic relationship

If Prompt 1's endpoint is near Prompt 2's midpoint, Prompt 1 "covers" what Prompt 2 builds toward. Prompt 2 takes longer to reach the same semantic destination.

## Reference Points (--point)

Reference points appear as colored diamonds with transparent spheres. They are separate embeddings — not part of any trajectory. They show where specific phrases live in the same 3D space.

### What reference points tell you

**Proximity to trajectory start**: The phrase is thematically related to the beginning of the prompt. The model was already "thinking about" this from the first words.

**Proximity to trajectory end**: The phrase is thematically related to where the prompt leads. The model's understanding converges toward this meaning by the end.

**Far from the trajectory**: The phrase is unrelated to the prompt. The model sees no semantic connection.

**Between two trajectories (comparison mode)**: The phrase relates to one prompt more than the other. Whichever trajectory it's closer to, that prompt is semantically more aligned with the phrase.

### The tool-call observation

A common finding: if you embed a tool call (e.g. `get_weather('Moscow')`) and its description (e.g. "Fetches current weather data for a given location"), they land near each other — and near the end of a prompt that logically requires that tool. This is not a coincidence. It's the model encoding **intent** before **execution**.

If you truncate the prompt right before the tool call, the trajectory endpoint still sits near the tool's reference point. The model "knows" where the conversation is heading based on accumulated context, not on the literal output tokens.

**What this means practically**: The embedding captures what the prompt is *about to do*, not just what it *says*. This is useful for debugging prompts — if your reference point (expected output) is far from the trajectory endpoint, the prompt isn't leading the model toward the right conclusion.

## Trajectory Simplification (--simplify-epsilon)

When enabled, consecutive points with cosine similarity above the threshold are merged. The trajectory keeps only meaningfully different snapshots. Use this when:

- The word trajectory is too dense to read (hundreds of words)
- You want to see only the major semantic shifts
- Points are so close they overlap visually

Threshold guide:
- `0.95` — aggressive, keeps only very distinct points
- `0.99` — moderate, merges near-identical consecutive points
- `0.0` — disabled, shows everything

## Practical Debugging Patterns

### "My prompt is too vague"

The trajectory stays in one tight cluster. Adding words barely moves the point. The model can't distinguish where the prompt is heading because every word is equally generic. **Fix**: add specific nouns, domain terms, or constraints to pull the trajectory toward a distinct region.

### "The model misunderstood my prompt"

Compare your prompt's trajectory endpoint with a reference point representing what you actually wanted. If they're far apart, the prompt's accumulated semantics don't lead where you intended. **Fix**: rewrite the prompt so its trajectory naturally arrives at the reference point's location.

### "Two prompts produce the same output but I wrote them differently"

Plot both. If the trajectories overlap heavily, the model genuinely sees them as semantically equivalent — your rewording didn't change the meaning. If they diverge, the model treats them differently despite similar surface text.

### "Which of my prompts is better for a specific task?"

Embed the task description as a reference point. The prompt whose trajectory endpoint is closer to that point is semantically better aligned with the task — regardless of which one "sounds better" to you.

## Color Coding

| Element | Color |
|---|---|
| Prompt 1 sentences | Blue (`#1976D2`) |
| Prompt 1 words | Orange (`#F57C00`) |
| Prompt 2 sentences | Green (`#388E3C`) |
| Prompt 2 words | Purple (`#7B1FA2`) |
| Reference points | Cycled from `POINT_COLORS` palette |

## Limitations

- **3D is a projection**: UMAP preserves local structure (nearby points stay nearby) but global distances can be misleading. Two points that look close might be farther apart in the original high-dimensional space than two points that look far apart.
- **Model-dependent**: Results vary between embedding models. A trajectory that looks meaningful with one model may look different with another. The tool is diagnostic, not absolute.
- **Cosine similarity ≠ meaning**: Two points can be close in cosine space and still mean different things in edge cases. Use the visualizations as a guide, not proof.
