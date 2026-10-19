# CenterScoreParser — Surviving High School episode decoder / encoder

Two Python files that turn *Surviving High School* episode files
(EA / Pixelberry, KiWi script engine) into a JSON description of the episode,
and compile that JSON back into a playable `.exp`.

| file | what it does |
|---|---|
| `decoder.py` | `.exp` / `.kiw` → story-graph JSON (or a transcript, or an annotated bytecode listing). Verifies its own output against the original before writing it. |
| `encoder.py` | JSON → `.exp`. It takes either a decoded episode or a small hand-written authoring spec. |

Nothing in either file is specific to one episode; every episode goes through
the same code. Needs Python 3 (tested on 3.11) and nothing outside the
standard library.

> Keep both files in the same folder. Each one imports the other: the decoder
> uses the encoder to check its own output, and the encoder borrows the
> opcode table from the decoder. Either the repository names
> (`decoder.py` / `encoder.py`) or the original names
> (`shs_decoder.py` / `shs_encoder.py`) work.

---

## 1. Quick start

```bash
# Decode an episode to JSON (what a game runtime plays)
python3 decoder.py 1_Making_Some_Dough.exp --format json -o dough.json

# Compile it back to a playable .exp
python3 encoder.py dough.json -o dough_rebuilt.exp

# Read it as a transcript
python3 decoder.py 1_Making_Some_Dough.exp --format md  -o dough.md

# Annotated bytecode, one line per VM instruction
python3 decoder.py 1_Making_Some_Dough.exp --disasm -o dough_disasm.txt
```

Decoding a whole season (as `runner.ipynb` does):

```python
import subprocess, pathlib
for exp in sorted(pathlib.Path("Season_1").glob("*.exp")):
    out = pathlib.Path("Season1_JSON") / (exp.stem + ".json")
    out.parent.mkdir(exist_ok=True)
    subprocess.run(["python3", "decoder.py", str(exp), "--format", "json", "-o", str(out)], check=True)
```

A successful decode prints:

```
[+] extracted 21 chunks to Season_1/2_1_The_Tutors (2 script scene(s))
[+] verification: decode matches the original under 4 policies
[+] wrote 813 entries to Season1_JSON/2_1_The_Tutors.json
```

The `verification` line is the decoder's guarantee that the JSON plays like the
original (§3). If it reports `verification skipped`, the encoder isn't next to
the decoder.

---

## 2. Decoder options

| option | purpose |
|---|---|
| `input` | A `.exp` container or a single `.kiw` script. |
| `-o, --output PATH` | Where to write (default: stdout). |
| `--format {md,txt,json}` | `json` is the story graph a runtime plays; `md` / `txt` are readable transcripts. Default `md`. |
| `--extract-dir DIR` | Where the `.exp`'s chunks (scripts `.kiw`, art `.png`, audio) are written. Default: a folder named after the episode, next to the `.exp`. |
| `--exact` | Emit every scene as exact code (IR, §4.5) instead of the story graph. Always faithful; less readable. |
| `--no-verify` | Skip self-verification (faster; no guarantee). |
| `--story-dir DIR` | Also write the episode as a folder of small per-segment JSON files. |
| `--cast "A,B,C"` | Override the cast table. |
| `--cast-from FILE.kiw` | Borrow the cast table from another script (a continuation scene with none of its own). |
| `--no-backgrounds` | Transcripts: omit background changes. |
| `--branches` | Markdown: label each choice option's branch. |
| `--overlay PATH` | Optional per-episode overlay (§7). Not needed for any of the tested episodes. |
| `--disasm` | Annotated bytecode listing instead of a transcript (§6). |
| `--disasm-range LO:HI` | With `--disasm`, only that byte-offset window (per scene). |

---

## 3. How a decode is produced

1. **Story graph.** The bytecode is read into readable segments: dialogue,
   narration, choices, gates on variables, variable writes, scene jumps,
   minigames, backgrounds, music.
2. **Exact code where needed.** Every KiWi routine is also lifted to a small
   intermediate representation (IR). A routine is kept as IR when it computes
   what it shows at run time, or uses engine services the story graph has no
   node for. The IR is specialised against the script's constant data, so it
   holds no addresses or memory images. Loops over constant tables unroll, and
   table reads become constants or inline tables.
3. **Self-verification.** The decoder re-encodes its own JSON with the
   encoder. It then plays the original and the re-encoded scripts on a built-in
   VM and engine stand-in, under four choice policies: first option, last
   option, and two seeded random runs. Any scene where the two first differ is
   rebuilt as exact IR, and the check repeats until they agree.
4. **Static text check.** Playthroughs can't cover every branch. So each story
   graph scene is also checked for every text its code can display. If any is
   missing from that scene's reachable nodes, the scene is rebuilt as exact IR.
5. **Tidy-up.** Segments nothing can reach, whose text is shown elsewhere
   anyway, are dropped.

The result is a mix: most scenes stay readable story graph, and the rest are
exact IR that a runtime interprets. Both are described below.

**The JSON says what should happen. It does not store VM state.** No memory
image, no variable table, no frame sizes. The runtime owns every store: story
variables, routine locals, string bindings and scratch cells. The JSON only
says which to read or write, and when.

---

## 4. The JSON

### 4.1 Top level

```json
{
  "title": "1_Making_Some_Dough",
  "episode": "01: Making Some Dough",
  "localized_titles": ["01: Making Some Dough", "01 - La main à la pâte", "..."],
  "pack_id": "101",
  "episode_id": "1",
  "cast": ["Kim", "Kim", "Ms. Rose", "Mr. Lumbar", "..."],
  "main_characters": ["Kim"],
  "genders": { "Kim": "F", "Mr. Lumbar": "M" },
  "entry": "s1_ir",
  "scene_entries": { "1": "s1_ir", "2": "s2_ir", "5": "s5" },
  "scene_setup": { "1": { "characters": [ { "id": "2", "name": "Ms. Rose", "art": "2195", "format": 1 } ], "ui_defaults": { "74": "1", "75": "23" } } },
  "section_dispatch": { "1": { "register": "1001", "entry_by_value": { "0": "s1_enter_v0", "26": "s1_03_after" } } },
  "scene_minigame_banks": { "1": { "question_options": [ { "question": "How would you rate this pack?", "options": ["Loved it!", "..."] } ] } },
  "routines": { "s1_r54": { "entry": "s1_r54_b1", "params": 2, "offset": "11381" } },
  "ir_scenes": ["1", "2", "3", "4"],
  "variable_defaults": { "values": { "2000": 0, "2001": 0 }, "set_only": ["1000"], "owner_values": { "1:629": 0 } },
  "name_vars": { "$Travis": { "set_to": ["Travis"] } },
  "segments": { "s1": { "nodes": [ ] } }
}
```

| key | meaning |
|---|---|
| `title` / `episode` / `localized_titles` | File stem, display title, and the title in each language from the archive. |
| `pack_id` / `episode_id` | Archive ids. The encoder writes them back. |
| `cast` | The script's name table. |
| `main_characters`, `genders` | Playable character(s) and each cast member's gender. |
| `entry` | First segment of the episode. |
| `scene_entries` | Scene number → its entry segment. `s<N>_ir` means the scene runs as exact IR. |
| `scene_setup` | Per scene, what it registers with the engine before showing anything: `characters` (engine id, display name, portrait art base, format; id 0 is the player) and `ui_defaults` (`74` primary speaker, `75` narrator id). Empty for scenes that reuse earlier registrations. |
| `section_dispatch` | Per scene, the section register (usually `var1001`) a `goto_scene` sets, and which segment each value enters (`entry_by_value`). |
| `scene_minigame_banks` | Index of minigame / survey content found in each scene. For reading only; the nodes carry what a runtime needs. |
| `routines` | Lowered routines: `entry` (first IR segment), `params` (argument count), `offset` (source bytecode offset). |
| `ir_scenes` | Scenes whose whole program runs as IR (`s<N>_ir` → `s<N>_main`). |
| `variable_defaults` | A read-only index built from the nodes. `values` lists every story variable the episode touches (all start at `0`; the engine zero-initialises). `set_only` lists variables that are only ever assigned. `owner_values` lists character-owned numbers as `"owner:key"`. |
| `name_vars` | A read-only index of `$Token` name placeholders. `default` comes from `string_default`, `set_to` lists names `set_string` stores, and `from_player_input` marks a token bound from typed text. |
| `segments` | The episode itself: `{ id: { "nodes": [...] } }`. |

The two indexes (`variable_defaults`, `name_vars`) describe the episode. The
nodes act. A runtime that executes the nodes gets the same state without
reading the indexes.

### 4.2 Segments and flow

A segment is a list of nodes played in order. It ends in a node that decides
where play goes: `next`, `choice`, `gate`, `dispatch`, `goto_scene`,
`minigame`, `end` or `return`. Links are segment ids. Strip a trailing
`.json` if one is present.

Most nodes carry `offset`, the bytecode offset they came from. Use it to line
the JSON up with a `--disasm` listing. Numbers taken from the bytecode are
written as strings (`"2000"`). IR expressions use native JSON ints.

### 4.3 Story-graph nodes

**Lines**

```json
{ "type": "dialogue", "speaker": "Kim", "text": "Let's see... Snickerdoodles!",
  "character_id": "1", "expression": "2", "emotion_code": "2", "image": "2252", "offset": "11608" }
{ "type": "narration", "text": "It's a pleasant Monday evening...", "offset": "11596" }
{ "type": "status", "text": "Kim has %d dollars.", "dynamic": true, "character_id": "23",
  "format": { "pattern": "Kim has %d dollars.", "args": [ { "var": 2000 } ] } }
{ "type": "notification", "kind": "score", "text": "Score Up!" }
```

| field | meaning |
|---|---|
| `speaker` | Display name. `$TOKEN` names are substituted at display time (§4.6). |
| `character_id` | The engine character id the script passes. The name shown is whatever that id was registered as. |
| `expression` / `emotion_code` / `image` | Portrait expression, and the portrait sprite id when it is fixed. |
| `mode` | `-2` a thought (shown in parentheses), `-3` emphasised. |
| `via` | Which engine call produced the line, kept so it re-encodes exactly. `15` is a one-off named speaker (`argc` is kept too). `13` is a status line shown as dialogue. |
| `format` | Text with runtime values: fill each `%d` in `pattern` from `args`, which are expressions over story variables (§4.5). |
| `text_bbcode` | The same text with the script's colour/emphasis markup as BBCode, when it has any. |

**Choices**

```json
{ "type": "choice", "prompt": "What do you say?",
  "options": [ { "index": "0", "label": "Textbooks!", "next": "s1_41_opt0" },
               { "index": "1", "label": "New clothes!", "next": "s1_43_opt2" } ],
  "menu": { "title": "Hurry!", "text": "What do you say?", "timeout_ms": "3000",
            "timeout_result": "1000", "character_id": "6", "reserved": "4", "portrait_mode": "1" },
  "timed": true, "timer_ms": "3000", "on_timeout": "s1_44_timeout", "after": "s1_02_after" }
```

Each option has a `next` segment. `menu` holds the menu call's exact arguments,
used for presentation and re-encoding. A timed choice goes to `on_timeout` when
the timer runs out. `after` is where the option branches rejoin.

**Branching**

```json
{ "type": "gate", "var": "2000", "op": "gte", "equals": "16", "then": "s3_bonus", "else": "s3_end" }
{ "type": "dispatch", "state_var": "2000", "arms": [ { "equals": "0", "scene": "s4_v5_20000" } ], "default": "s4_x" }
{ "type": "goto_scene", "file": "s2_ir", "section_register": { "var": "1001", "value": "1" } }
{ "type": "call", "segment": "s1_rt30090" }
{ "type": "return" }
{ "type": "next", "next": "s1_15_choice" }
{ "type": "end" }
```

| node | meaning |
|---|---|
| `gate` | Compare `var` with `equals`. `op` is one of `eq` (also when absent), `ne`, `gt`, `gte`, `lt`, `lte`. Go to `then` or `else`. |
| `dispatch` | Multi-way switch on `state_var`: the first arm whose `equals` matches, else `default`. |
| `goto_scene` | Set `section_register.var` to `value`, then enter scene `file`. The target scene's `section_dispatch` picks the section. |
| `call` with `segment` | Play that segment until it reaches `return`, then continue after the call. |
| `call` with `routine` | Run a lowered routine (§4.5): `{"routine":"s3_r26","args":[6,20,10,0],"result_var":"3902"}`. If `result_var` is given, register B is stored there afterwards. |
| `end` | This scene is over. Load the next scheduled scene, or finish the episode. |

**Variables and names**

```json
{ "type": "var_set", "var": "1000", "value": 1, "when": { "var": "1000", "equals": "0" } }
{ "type": "var_set", "var": "2000", "expr": { "var": 2001 } }
{ "type": "var_add", "var": "2005", "value": 7 }
{ "type": "set_string", "key": "$Travis", "value": "Travis" }
{ "type": "string_default", "key": "$ZOE", "default": "Zoe", "slot": "-1" }
{ "type": "set_scene_value", "value": "6" }
{ "type": "set_ui_default", "value": "30" }
```

`when` applies the write only if the condition holds. `expr` is an expression
over story variables (§4.5). `string_default` gives a `$TOKEN` the name used
until something is stored under it.

**Presentation**

```json
{ "type": "title_card", "title": "The Fashionista", "subtitle": "Can $ALLISON get the Troublemakers through an impossible task?", "asset_id": "26063", "flag": "-1" }
{ "type": "end_card", "text": "Thanks for playing!" }
{ "type": "background", "style": "standard", "asset_id": "1111", "variant": "0" }
{ "type": "background", "style": "custom", "asset_id": "26084", "variant": "-1", "character_id": "-1", "expression": "-1", "via": "35" }
{ "type": "music", "track_id": "8217", "loop": true }
{ "type": "music", "action": "stop", "fade_ms": "4500" }
{ "type": "sfx", "sfx_id": "26064" }
{ "type": "set_expression", "character_id": "0", "expression": "2" }
{ "type": "pov_change", "character": "Matt", "index": "0" }
{ "type": "wobble" }
{ "type": "vibrate" }
```

A `custom` background is art packaged inside the `.exp`. A `standard` one
comes from the game's shared library. `set_expression`
sets the expression a character's later lines show unless a line overrides it.

**Minigames**

```json
{ "type": "minigame", "minigame_type": "pick_word", "kind": "content_fork",
  "title": "Make your choice!", "prompt": "Concentrate!",
  "options": [ ["Learn!", "Study!", "Concentrate!"], ["Daydream!", "Flirt!", "Get distracted!"] ],
  "duration_ms": "20000", "win_threshold": "6", "win": "s2_03_win", "lose": "s2_04_lose" }
{ "type": "minigame", "minigame_type": "content_fork", "minigame_id": "1000", "via": "by_id",
  "win": "s5_03_win", "lose": "s5_06_lose" }
```

`pick_word` carries the word game's exact arguments: `options[0]` are the
correct words and `options[1]` the decoys. Picks are tallied over
`duration_ms` against `win_threshold`. `by_id` is a built-in engine minigame
selected by `minigame_id`. Both branch to `win` or `lose`. Minigames that need
more than this run inside exact IR as engine yields (§4.5).

### 4.4 Exact IR segments

Scenes listed in `ir_scenes`, and every `routine`, are made of these nodes:

| node | meaning |
|---|---|
| `{"type":"let","to":LV,"value":E}` | Assignment. `LV` is `{"local":k}`, `{"reg":"A"\|"B"}`, `{"scratch":i}` or `{"deref":E}`. |
| `{"type":"yield","yield":id,"name":"dialogue","args":[E…],"to":k}` | Engine call with evaluated arguments; the result goes into local `k` when `to` is given. Dialogue (13) and narration (65) yields with a constant character also carry a `speaker` hint. |
| `{"type":"if","cond":E,"then":seg,"else":seg}` | Branch. |
| `{"type":"next","next":seg}` | Jump. |
| `{"type":"call","routine":name,"args":[E…]}` | Call another routine. |
| `{"type":"ret"}` | Return; register B holds the result. |
| `{"type":"end"}` | Halt: the scene is over. |
| `{"type":"pause"}` | VM pause. |

### 4.5 Expressions and pointers

* `int` — a number. All arithmetic is 16-bit signed; `/` and `%` truncate toward zero.
* `{"text": "…"}` — a text constant.
* `{"var": k}` — story variable *k* (story-graph expressions).
* `{"local": k}` — frame cell *k* of the running routine. Arguments are negative: with *n* arguments, argument *i* is `local -(2+n-i)`.
* `{"reg": "A"|"B"}` — VM registers.
* `{"op": o, "a": E, "b": E}`, where `o` is one of `+ - * / % == != > >= < <= or and`; also `{"op": "neg"|"not", "a": E}`.
* `{"scratch": i}` — the value in VM scratch cell *i* (the VM's own working words, such as survey answers or a copied-input buffer).

Pointers name a cell, and `{"deref": P}` reads through one:
`{"addr": k}` (a frame cell), `{"scratch_addr": i}` (a scratch cell), and
`{"table": [e0, e1, …]}` (a constant read-only array of ints or `{"text"}`).
For example, `{"deref":{"op":"+","a":{"table":["You're going down!", "..."]},"b":{"local":9}}}`
reads a random taunt.

### 4.6 Engine services (yield ids)

The names come from the native engine (`libshs09.so`, Android 1.0.9),
checked with REA + Ghidra.

| id | name | id | name |
|---|---|---|---|
| 0 | format (`%d` text → dynamic slot) | 44 / 54 | set_number (story variable) |
| 1 | choice | 45 / 55 | get_number |
| 2 / 3 / 4 | menu_begin / menu_option / menu_show | 46 / 47 | set_string / get_string_default |
| 5 | set_character_expression | 49 | set_character_name |
| 7 / 63 | **end_episode** | 52 / 53 | set / get owner number |
| 9 | submit_poll | 65 | narration |
| 10 | goto_scene | 70 | platform_query (shows nothing) |
| 11 / 35 | set_background | 72 / 73 | set / get character art base |
| 13 | dialogue | 74 / 75 | set_ui_default |
| 15 / 76 | named_line (one-off speaker) | 79 / 80 / 81 | request audio / stop audio |
| 16 | set_scene_value | 82 | vibrate |
| 18 / 99 | noop | 88 | dialogue_notice |
| 27 | random_below | 89 | wobble |
| 33 | show_card | 91 | loading |

Ids this table doesn't name appear as `yield_<id>`. Text shown to the player
goes through `$Token` substitution. Dynamic text results live in slots
starting at `0x7FF5`.

---

## 5. Runtime notes

Things a runtime has to do that aren't spelled out by a single node:

* **End of episode (yield 7 / 63).** The native engine clears the scene
  schedule, resets the running script and clears the resume save. Nothing
  after the call runs.
* **Survey prompt.** When a choice's text contains *"Help us improve the game
  by taking this optional survey"*, the native engine marks the episode
  finished as the menu appears. Every tested episode ends with this menu.
* **Promotional narration.** A narration containing *"Episodes 1-20 for 50%
  off"* or *"September 27, 2011"* is never shown.
* **Engine state persists across scenes.** Registered characters, names and
  variables carry over, so a scene with an empty `scene_setup` relies on the
  scenes before it.
* **Exact IR.** The Godot project's `kiwi_ir.gd` (class `KiwiIR`) interprets
  routines and `s<N>_ir` scenes. The story runner hands it every `call` with a
  `routine` and implements `kiwi_yield(id, args, node)` for the engine services.

---

## 6. Bytecode listing (`--disasm`)

`--disasm` prints one line per VM instruction: offset, mnemonic, and a note
that resolves text pushes, variable reads and writes, arithmetic, comparisons
and jump targets. It's a diagnostic view, and offsets match the JSON's
`offset` fields.

```
@18452  push 16
@18455  push 2000
@18458  1f2d VAR_READ   ; read var2000
@18462  SUB (0x51)   ; -> 16 - var2000
@18463  push 5
@18466  MUL (0x52)   ; -> (16 - var2000) * 5
@18472  push 188   ; "You're %d points short of unlocking the bonus"
```

A conditional jump (`0x2b`) is taken when the comparison is **false**.

---

## 7. Encoder

```bash
python3 encoder.py episode.json -o episode.exp
```

It accepts two kinds of input:

* **A decoded episode** (has `segments`). Everything in §4 is compiled back:
  story graph, routines, IR scenes, tables, scene setup, archive ids and
  titles. The decoder uses exactly this path to verify itself, so a decoded
  episode always re-encodes to something that plays the same.
* **An authoring spec** (has `scenes`), for writing new content by hand:

```json
{
  "title": "My Episode",
  "cast": ["", "Zoe", "Sam"],
  "scenes": [
    { "id": 1, "nodes": [
      { "type": "narrate", "text": "It was a dark night." },
      { "type": "say", "speaker": "Zoe", "text": "Hi!" },
      { "type": "choice", "prompt": "What do you do?", "options": [
        { "label": "Be nice", "body": [ { "type": "add", "var": "2000", "value": 1 } ] },
        { "label": "Be mean", "body": [ { "type": "set", "var": "2001", "value": 1 } ] } ] },
      { "type": "gate", "var": "2000", "op": "gte", "value": 1,
        "then": [ { "type": "say", "speaker": "Sam", "text": "Nice!" } ], "else": [] },
      { "type": "background", "id": 1030 },
      { "type": "music", "id": 8225 },
      { "type": "goto", "scene": 2 } ] },
    { "id": 2, "nodes": [ { "type": "narrate", "text": "The end." } ] }
  ]
}
```

`--tree` selects the older tree-folding compiler instead of the graph
compiler. To check any output, decode it again with `decoder.py`.

---

## 8. Overlays (optional)

`--overlay FILE.json` merges hand-observed structure onto a decode by bytecode
offset: scene carving, option boundaries, effects, background names. It is
data, not code, and anything it supplies is marked `"source": "observed"`.
Self-verification makes overlays unnecessary for the tested episodes. They
remain for annotating episodes by hand.

---

## 9. File formats

* **`.exp` (CSPUD archive).** `b"CSPUD"`, a big-endian `uint32` entry count, a
  directory of `(uint16 id, uint32 offset)`, then chunks (`comp_size`,
  `uncomp_size`, `flags`, data; LZMA when `flags & 1`). Payloads are
  identified by magic: `kiwi` (script), PNG (art), or audio.
* **`.kiw` (KiWi script).** `kiwi`, version 2, library and external flags,
  four counts, then the NUL-terminated strings, constant data words and a
  stack-VM bytecode. Engine services are `1f <selector> <argc>` calls, and a
  scene's program starts by reading the section register and dispatching.

---

## 10. Status

Checked on 16 episodes (Making Some Dough, As Time Goes By, A Float Is Born,
From Europe With Love, The Wrong Side of Town, Halloween, A Fallon Family
Christmas, Magic School, The New Girl, The Breakup, The Tutors, Swim Team
Retreat, The Halloween Dance pt 2, The Pumpkin Farmer's Daughter, The
Fashionista, Spencer's Story):

* Self-verification passes on all 16, and the static text check finds nothing missing.
* Six episodes decode entirely to story graph. The rest keep some scenes as exact IR.
* Original vs decode → encode in the reference runtime: 8,487 transcript lines
  identical, including speaker, expression and presentation.
* An independent KiWi interpreter (codkiwi) shows every displayed event identical.
* The Godot IR interpreter matches the Python reference: 138 runs, identical.

Known limits:

* **The New Girl.** Answering "call Keith" forever overflows the VM stack a
  little sooner in the re-encode (about 1,190 lines instead of about 1,310).
* **The 0x80–0x9F text encoding** is not handled. None of the tested episodes
  uses it.
* **After the survey menu.** What the native engine does once the menu is
  answered isn't fully settled. The decode reproduces the script's calls
  exactly either way.
