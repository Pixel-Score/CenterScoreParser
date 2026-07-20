# Surviving High School — Decoder (`shs_decoder.py`)

A reverse-engineering pipeline that turns *Surviving High School* episode files
(EA/Pixelberry, the "kiwi"/Blast engine) into a single, runtime-ready JSON
description of the episode: its scenes, dialogue, choices, branches, minigames,
and score logic. The JSON is consumed by the Godot 4 runner (`story_runner.gd`).

- **Input:** a `.exp` container (a "CSPUD" archive) or a raw `.kiw` script.
- **Output:** a segmented JSON story graph (default), a Markdown/plain-text
  transcript, or an annotated bytecode disassembly.

---

## 1. Quick start

```bash
# Decode an episode to the story-graph JSON (what the Godot runner consumes)
python3 shs_decoder.py 1_Making_Some_Dough.exp --format json -o dough.json

# Human-readable transcript instead
python3 shs_decoder.py 1_Making_Some_Dough.exp --format md  -o dough.md
python3 shs_decoder.py 1_Making_Some_Dough.exp --format txt -o dough.txt

# Annotated bytecode listing (diagnostic; see §4)
python3 shs_decoder.py 1_Making_Some_Dough.exp --disasm -o dough_disasm.txt
```

---

## 2. Command-line options

| Option | Purpose |
|---|---|
| `input` | Path to a `.exp` container or a `.kiw` script (required). |
| `-o, --output PATH` | Write output here (default: stdout). |
| `--format {md,txt,json}` | Output format. `json` is the story graph; `md`/`txt` are transcripts. Default `md`. |
| `--overlay PATH` | Per-episode overlay JSON of observed gameplay logic, merged onto the decode by bytecode offset. |
| `--extract-dir DIR` | For `.exp` input: also write every extracted chunk (script `.kiw` + `.png` art) to `DIR`. |
| `--cast "A,B,C"` | Override the cast/name table with an explicit comma-separated list. |
| `--cast-from FILE.kiw` | Borrow the cast table from another script (for continuation scenes with no table of their own). |
| `--no-backgrounds` | Omit background-change markers from a transcript. |
| `--branches` | In Markdown, label each choice's option-branches separately. |
| `--story-dir DIR` | Also write the episode as a folder of small per-segment JSON files. |
| `--disasm` | Print an annotated bytecode listing instead of a transcript (see §4). |
| `--disasm-range LO:HI` | With `--disasm`, restrict output to a byte-offset window, per scene. |

---

## 3. The story-graph JSON (`--format json`)

### 3.1 Top-level shape

```json
{
  "title": "1_Making_Some_Dough",
  "episode": "01: Making Some Dough",
  "cast": ["Kim", "Kim", "Ms. Rose", "Mr. Lumbar", "..."],
  "main_characters": ["Kim"],
  "genders": { "...": "..." },
  "name_vars": { "...": "..." },
  "entry": "s1",
  "variable_defaults": {
    "values": { "1000": 0, "2000": 0, "2001": 0, "2004": 0, "...": 0 },
    "set_only": ["1000", "1001"]
  },
  "scene_entries": { "...": "..." },
  "segments": {
    "s1":            { "nodes": [ /* ... */ ] },
    "s1_04_var2002_lt3": { "nodes": [ /* ... */ ] },
    "...":           { "nodes": [ /* ... */ ] }
  }
}
```

| Key | Meaning |
|---|---|
| `title` | The source file's stem. |
| `episode` | The episode's display title, read from the archive metadata. |
| `cast` | The name table. A line's `speaker` is one of these names; the index also lines up with the bytecode's speaker indices. |
| `main_characters` | The playable protagonist(s). |
| `genders`, `name_vars` | Player-customisable name/gender substitutions the engine fills at runtime. |
| `entry` | The segment where play begins (the first scene). |
| `variable_defaults.values` | Every game variable and its starting value. `2000`–`2011` are gameplay counters (money, day, score, flags); `1000`/`1001` are engine/scene selectors. |
| `variable_defaults.set_only` | Variables the engine only ever *sets* (never reads back for branching), e.g. scene selectors. |
| `segments` | The story graph. Each key is a segment id; each value is `{ "nodes": [ ... ] }`. |

### 3.2 Segments and flow

A **segment** is a straight run of nodes ending in a control node (a `choice`,
`gate`, `dispatch`, `goto_scene`, `next`, `minigame`, or `end`) that decides where
play goes next. Segments reference each other by id (with a `.json` suffix in the
raw links, stripped when loaded). Node order within a segment is play order.

Every content node carries an `offset` — its position in the original bytecode.
Offsets are how a fix or a disassembly line ties back to a specific instruction.

---

### 3.3 Node type reference

Below is every node type the decoder emits, with a real example, what each field
means, and what the node is for.

---

#### `dialogue` — a spoken line

```json
{
  "type": "dialogue",
  "speaker": "Kim",
  "emotion_code": "2",
  "image": "2252",
  "text": "Let's just see what I've made here... Snickerdoodles!",
  "offset": "11608"
}
```

| Field | Meaning |
|---|---|
| `speaker` | Cast name of who is talking. |
| `emotion_code` | Raw expression code (1=angry/annoyed, 2=happy/neutral, 3=sad/serious, 4=surprised). |
| `image` | Portrait sprite id (sprite base + emotion). Absent/`null` for the player avatar, which the engine renders dynamically. |
| `text` | The line. |
| `offset` | Bytecode offset that emitted this line. |

Used for: every character speech line.

---

#### `narration` — narration / stage direction (no speaker)

```json
{
  "type": "narration",
  "text": "It's a pleasant Monday evening, and Kim is busy working her late shift.",
  "offset": "11596"
}
```

Used for: scene-setting text and stage directions shown with no speaker.

---

#### `title_card` — episode/scene title card

```json
{
  "type": "title_card",
  "title": "Making Some Dough",
  "subtitle": "Can you earn enough money to save the bakery?"
}
```

Used for: the intro card shown before the first spoken line. `subtitle` is the
goal/hook line.

---

#### `end_card` — closing card

```json
{
  "type": "end_card",
  "text": "Thanks for playing! See you in another episode!",
  "offset": "13456"
}
```

Used for: credit/sign-off text shown after an ending.

---

#### `status` — an on-screen HUD / status line (may be dynamic)

Static form:

```json
{ "type": "status", "text": "Poll results:", "dynamic": false, "offset": "14332" }
```

Dynamic form (a `%d` filled from a variable at runtime):

```json
{
  "type": "status",
  "text": "You have earned %d out of 100 Points for this chapter.",
  "dynamic": true,
  "offset": "28992",
  "value_from": { "var": "2000", "multiply": "5" }
}
```

| Field | Meaning |
|---|---|
| `text` | The template. A `%d` is filled at runtime. |
| `dynamic` | `true` if the text has a runtime-filled value. |
| `value_from` | How to compute the `%d` (see below). Absent when `dynamic` is `false`. |

**`value_from` transforms** — the engine reads `var`, then applies any transforms
**in this order**: `minus_from` first, then `multiply`.

| `value_from` | Formula | Example |
|---|---|---|
| `{ "var": "2000" }` | `var2000` | "Kim has %d dollars." |
| `{ "var": "2001", "minus_from": "10" }` | `10 - var2001` | "Kim has %d days left." |
| `{ "var": "2000", "multiply": "5" }` | `var2000 * 5` | "You have earned %d out of 100 Points!" (score scaled to 100) |
| `{ "var": "2000", "minus_from": "16", "multiply": "5" }` | `(16 - var2000) * 5` | "You're %d points short of unlocking the bonus." |

Used for: money/day/score HUD lines and end-of-episode score readouts. The score
is stored raw (max 20) and displayed ×5 so it reads out of 100.

---

#### `choice` — a branching player choice

```json
{
  "type": "choice",
  "prompt": "What should I do?",
  "options": [
    { "index": "0", "label": "Try the academic challenge.", "next": "s1_05_opt0" },
    { "index": "1", "label": "Look at the Help Ads.",        "next": "s1_06_opt1" },
    { "index": "2", "label": "Work at the bakery.",          "next": "s1_07_opt2" }
  ]
}
```

A *timed* choice adds a countdown (`after` is where it goes on expiry):

```json
{
  "type": "choice",
  "prompt": "...",
  "options": [ { "index": "0", "label": "...", "next": "..." } ],
  "timed": true,
  "timer_ms": "5000",
  "after": "s2_03_after"
}
```

| Field | Meaning |
|---|---|
| `prompt` | The question shown above the options (may be absent). |
| `options[].index` | Option number. |
| `options[].label` | Button text. |
| `options[].next` | Segment to go to if chosen. |
| `options[].effects` | Optional list of variable effects (e.g. `[{ "var": "money", "delta": -10 }]`), when supplied by an overlay. |
| `timed`, `timer_ms`, `after` | For timed choices: countdown length and the fallback segment on expiry. |

Used for: every player decision point.

---

#### `gate` — a conditional branch on a variable

```json
{
  "type": "gate",
  "var": "2011",
  "equals": "1",
  "then": "s1_12_var2011_eq1",
  "else": "s1_13_else"
}
```

| Field | Meaning |
|---|---|
| `var` | Variable to test. |
| `equals` | Value to compare against. |
| `op` | Comparison: absent/`""` = equality, `"gte"` = `>=`, `"lt"` = `<`. |
| `then` | Segment when the comparison is **true**. |
| `else` | Segment when **false** (may be absent if there is no false branch). |

Used for: score/flag-driven branching (e.g. the bonus-scene gate `var2000 >= 16`,
the loan-vs-non-loan ending gate, the sophomore-chapter variant gate).

---

#### `dispatch` — a multi-way switch on a variable

```json
{
  "type": "dispatch",
  "state_var": "2002",
  "arms": [
    { "equals": "0", "scene": "s2_alt" },
    { "equals": "1", "scene": "s2_07" }
  ],
  "default": "s2_15"
}
```

| Field | Meaning |
|---|---|
| `state_var` | Variable that selects the arm. |
| `arms[].equals` / `arms[].scene` | If `state_var` equals this value, go to this segment. |
| `default` | Segment used when no arm matches. |

Used for: the bakery day-cascade (`var2004` → each day's minigame), chapter
selectors, and scene hubs. This is the multi-branch cousin of `gate`.

---

#### `random` — a random selection with optional exhaustion tracking

```json
{
  "type": "random",
  "state_var": "2003",
  "options": [
    { "scene": "s3_06", "done_when": { "var": "2005", "op": "bitmask", "equals": "1" } },
    { "scene": "s3_08", "done_when": { "var": "2006", "op": "bitmask", "equals": "1" } },
    { "scene": "s3_14", "done_when": { "var": "2005", "op": "bitmask", "equals": "2" } }
  ],
  "exhausted": "s3_alt",
  "max_count": "6"
}
```

| Field | Meaning |
|---|---|
| `state_var` | Optional counter variable the engine bumps per pick. |
| `options[].scene` | A candidate segment to pick. |
| `options[].done_when` | Marks this option "already played". `op` is `bitmask` (played iff `(var & equals) != 0`), `gte`, or `eq`. |
| `exhausted` | Where to go once every option is done. |
| `max_count` | Safety stop: emitted only when every option is guarded. |

Used for: the dough "Help Ads" — six ads picked without repeats. The done-flags
are additive bitmasks (two ads share one flag var, values 1 and 2), so an ad is
done when `(flag & value) != 0`.

---

#### `minigame` — a skill minigame

```json
{
  "type": "minigame",
  "minigame_type": "pick_word",
  "kind": "bake",
  "win": "s4_17_m",
  "lose": "s4_lose",
  "win_threshold": "7",
  "timed": true,
  "timer_ms": "20000",
  "offset": "11588",
  "setup": {
    "bank": "pick_word",
    "device": "mobile",
    "timer_ms": "20000",
    "rounds": [
      {
        "offset": "783",
        "subtitle": "Bake cookies! Pick the right ingredients!",
        "correct": ["Peanut butter.", "Cranberry raisin."],
        "decoys":  ["Poenut botter.", "Cronberry raison."]
      }
    ]
  }
}
```

| Field | Meaning |
|---|---|
| `minigame_type` | `pick_word`, `build_word`, `prompt`, `word-match`, `action`, etc. |
| `kind` | Role in the flow: `bake`, `outcome_fork`, `content_fork`, `prompt`, `action`. |
| `win` / `lose` | Segments taken on pass / fail. |
| `win_threshold` | Correct picks needed to win. |
| `timed`, `timer_ms` | The game runs on a countdown (e.g. 20 s). |
| `setup.bank` | Which word bank this game draws from. |
| `setup.device` | `mobile` (pick) or `tablet` (build). |
| `setup.rounds` | A **single** word-set: two `correct` words and two `decoys`. |
| `setup_alt` | An alternate device variant (e.g. the tablet build version of the same game). |

**How a pick-word game plays:** one word-set is shown; the player taps a word;
the list reshuffles and they pick again; this repeats until the timer ends,
tallying correct picks against `win_threshold`. It is **one** game (2 correct +
2 decoy), *not* a sequence of different-word rounds.

In the bakery, each day is a different word-set that gets harder; winning a day
advances to the next word-set, losing repeats the same one. The day → word-set
order is Cinnamon → Coconut → Peanut butter → Peanut butter (repeat) →
Snickerdoodles, with win thresholds 6, 6, 6, 7, 8.

---

#### `background` — change the on-screen background

Standard (global asset library):

```json
{ "type": "background", "style": "standard", "asset_id": "1094", "offset": "11582" }
```

Custom (a PNG packaged in this `.exp`):

```json
{ "type": "background", "style": "custom", "asset_id": "26185",
  "name": "school swimming pool", "offset": "..." }
```

Used for: setting the scene backdrop. `name` is filled for known custom backgrounds.

---

#### `music` / `sfx` — audio cues

```json
{ "type": "music", "track_id": "8217", "offset": "11589" }
{ "type": "sfx",   "sfx_id":  "8006", "offset": "11644" }
```

Used for: `music` changes the background track; `sfx` plays a one-off sound cue.

---

#### `pov_change` — switch the player-controlled character

```json
{ "type": "pov_change", "character": "Kim", "index": "1", "offset": "11556" }
```

Used for: episodes where you play as more than one character; marks who you
control from here on.

---

#### `var_set` / `var_add` — write a game variable

```json
{ "type": "var_add", "var": "2011", "value": 1, "offset": "12421" }

{ "type": "var_set", "var": "1000", "value": 1, "offset": "11269",
  "when": { "var": "1000", "equals": "0" } }
```

| Field | Meaning |
|---|---|
| `var` | Variable to write. |
| `value` | Amount to set (`var_set`) or add (`var_add`). |
| `when` | Optional guard: only apply if the condition holds (e.g. first-visit init). |

Used for: scoring (`var_add var2000`), progress counters (`var_add var2004` per
bake day), flags, and one-time initialisers. Score flags are written additively
so they act as bitmasks.

---

#### `next` — unconditional continue

```json
{ "type": "next", "next": "s1_15_choice" }
```

Used for: a segment that simply flows into the next one.

---

#### `goto_scene` — jump to another scene

```json
{
  "type": "goto_scene",
  "file": "s5",
  "section_register": { "var": "1001", "value": "25" },
  "offset": "12144"
}
```

| Field | Meaning |
|---|---|
| `file` / `to_scene` | Target segment/scene. |
| `section_register` | Optional selector the engine sets before jumping (which sub-section of the target to enter). |

Used for: transfers between scenes (intro → day hub → task scenes, etc.).

---

#### `checkpoint_replay` — return to a checkpoint

```json
{ "type": "checkpoint_replay", "next": "s2_03_after.json" }
```

Used for: "play again from here" flow — sends the player back to a saved
checkpoint segment.

---

#### `end` — terminal

```json
{ "type": "end" }
```

Used for: the end of a path (the episode/branch stops here).

---

## 4. Bytecode disassembly (`--disasm`)

`--disasm` prints an annotated, one-line-per-instruction view of the raw VM
bytecode. It is a **diagnostic view**: it reports what each instruction *is* and,
where possible, resolves jump targets and annotates arithmetic — but it makes no
routing decisions (it does not build segments or decide where the story flows).
Use it to check the decoder's work against the actual bytes.

```bash
# The whole script
python3 shs_decoder.py 2_1_The_Tutors.exp --disasm

# Just a window of offsets (per scene)
python3 shs_decoder.py 2_1_The_Tutors.exp --disasm --disasm-range 18450:18475
```

Example — the score-scaling formula for the "points short of the bonus" line:

```
@18452  push 16
@18455  push 2000
@18458  1f2d VAR_READ   ; read var2000
@18462  SUB (0x51)   ; -> 16 - var2000
@18463  push 5
@18466  MUL (0x52)   ; -> (16 - var2000) * 5
@18467  SET_GOAL (0x3e)   ; display value = ((16 - var2000) * 5)
@18472  push 188   ; "You're %d points short of unlocking the bonus"
```

Example — a score gate and the branch it guards:

```
@31300  1f2d VAR_READ   ; read var2000
@31304  push 16
@31307  >= (0x0d)   ; -> var2000 >= 16
@31308  jump_if_false (0x2b)   ; -> @31332 "Achieving the rank of 'Honor Roll..."
@31314  push 9673   ; "You have unlocked the bonus scene!"
@31325  SEP op=1677   ; -> @31365 "A while later, $ZOE wanders out of her..."
```

Each line is `@offset  mnemonic   ; note`. The note resolves text pushes to their
string, names variable reads/writes, shows the running arithmetic expression,
gives a comparison's condition, and resolves a jump/SEP to its target offset plus
the first line of text found there. The offsets match the `offset` fields in the
JSON, so you can line a disassembly up against the story graph.

> A conditional jump (`0x2b` / `jump_if_false`) fires when the comparison is
> **false** — it skips the following block. So `var2000 >= 16` with
> `jump_if_false -> @31332` means: high score falls through to the bonus block;
> low score jumps to `@31332`.

---

## 5. Overlays

Some structure can't be derived from the bytes with confidence (exact per-option
branch boundaries, a scene that must be carved out of another, declared effects
like "spending $10", gating conditions). These live in a per-episode **overlay**
JSON, merged onto the decode purely by matching bytecode offsets. The overlay is
**data, not code**: the parser stays generic and applies any overlay to any
episode by offset. Content (the lines) always comes from the decode; the overlay
only supplies attribution, effects, and structure the bytecode can't yield.

Pass one with `--overlay FILE.json`. Anything sourced from an overlay rather than
the bytes is marked `"source": "observed"` in the output.

---

## 6. Notes on the engine side

The JSON is designed so the Godot runner evaluates control flow **live** against a
variable store, rather than the decoder resolving every branch statically. A few
runner responsibilities the JSON assumes:

- **`gate`**: compare `var` to `equals` using `op` (`""`=equality, `gte`, `lt`);
  follow `then`/`else`.
- **`random`**: respect each option's `done_when.op` — for `bitmask`, an option is
  done when `(variables[var] & equals) != 0`; enforce `max_count`; write the
  done-flags additively.
- **`status` `value_from`**: read `var`, then apply `minus_from` (as
  `minus_from - value`) **before** `multiply` (`value * multiply`).
- **`minigame`**: show the single `setup.rounds[0]` word-set, reshuffle on each
  pick, tally correct picks over the timer, and compare to `win_threshold`.

---

## 7. File formats (for reference)

- **`.exp` — CSPUD archive.** A big-endian indexed-blob container: `b"CSPUD"`,
  a `uint32` entry count, a directory of `(uint16 id, uint32 offset)` records,
  then chunks (each `comp_size`, `uncomp_size`, `flags`, data; LZMA-alone when
  `flags & 1`). Chunk payloads are identified by magic: `b"kiwi"` (script) or
  `b"\x89PNG"` (art).
- **`.kiw` — compiled dialogue script.** `b"kiwi"` + header; NUL-terminated
  strings; then a stack-VM bytecode that ties each line to a speaker/emotion and
  encodes choices, jumps, variable reads/writes, arithmetic, and minigames. A
  line's text lives at `ref * 2 + 15`.
