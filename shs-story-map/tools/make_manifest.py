#!/usr/bin/env python3
"""Rebuild episodes/index.json from every decoded episode JSON in episodes/.

Usage (from the site root):  python3 tools/make_manifest.py
The picker label is the file's "episode" (or "title") field, falling back to the file name.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "episodes"
items = []
for p in sorted(root.glob("*.json")):
    if p.name == "index.json":
        continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        title = d.get("episode") or d.get("title") or p.stem
    except Exception as e:
        print(f"skip {p.name}: {e}")
        continue
    items.append({"file": p.name, "title": str(title)})
(root / "index.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {len(items)} episode(s) to {root / 'index.json'}")
