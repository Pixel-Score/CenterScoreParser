# SHS Story Map

A static viewer for episode JSON produced by `shs_decoder.py`. It draws every segment as a card
(speaker, emotion and line for each node) and connects them with typed links: plain continuation,
choice options (with any variable effects), condition checks (true / otherwise), minigame win / lose,
and jumps to other scenes. No build step; it runs as-is on GitHub Pages.

## Publish on GitHub Pages
1. Copy this folder into a repo (root or `/docs`).
2. Put decoded episodes in `episodes/`, then run `python3 tools/make_manifest.py` to list them.
3. Repo Settings, Pages: deploy from the branch and folder you used.

Opening `index.html` straight from disk works too, but browsers block `fetch` on `file://`, so the
picker stays empty; use **Open JSON** or drag a file onto the page instead
(or run `python3 -m http.server` in this folder).

## Views
- **Scene map**: one card per scene, with the jumps between scenes and counts of lines, choices,
  minigames and condition checks.
- **Scene flow**: every segment in one scene. Links that leave the scene end in a dashed portal card;
  click it to follow.
- **Everything**: the whole episode graph in one layout.

Side panel: scene list, a **Variables** index (every write, change, check and read, each linking to
its segment) and full-text **Search**. Click a card for the inspector: full script with bytecode
offsets and image ids, outgoing and incoming links, and the raw JSON.

Deep links: the URL hash holds the episode, view, scene and selected segment, e.g.
`#ep=episodes/dough.json&view=scene&scene=3&seg=s3_02`.

## Accepted JSON
- **Segment graph** (current decoder): `entry`, `scene_entries`, `segments{ id: { nodes, next | gate |
  choice | minigame | goto_scene | end } }`, including terminals written as the last node. Any other
  field whose value names a segment is still drawn, as a grey "other reference" link, so new
  decoder fields show up without viewer changes.
- **Scene list** (older decoder / overlay output): `scenes[ { scene, script, nodes, gate?, control_flow? } ]`.
  Links the file does not state explicitly (flat choice regions, outcomes) are drawn dotted and
  marked inferred.

The viewer never adds routing of its own; every solid link corresponds to a field in the JSON.
`examples/` holds two small synthetic files (not real episode content) that exercise both formats.

Layout uses [dagre](https://github.com/dagrejs/dagre) (MIT), vendored in `vendor/`.
