#!/usr/bin/env python3
"""
shs_encoder.py -- compile a simple story spec (JSON) into a playable .exp file.

This is a COMPILER, not the inverse of shs_decoder: it generates correct KiWi
bytecode from a high-level authoring spec. Scope (v1): linear dialogue/narration,
choices, variable set/add and equality/threshold gates, scene transitions, and
background/music/sfx cues. Minigames and checkpoint/save-state flows are out of
scope for now.

Authoring schema (JSON):
{
  "title": "My Episode",
  "cast": ["", "Zoe", "Sam", ...],        # index 0 usually the player/blank
  "variables": {"2000": 0, "2001": 0},    # initial values (optional)
  "scenes": [
    { "id": 1, "nodes": [ <node>, ... ] },
    ...
  ]
}

Nodes:
  {"type":"say", "speaker":"Zoe", "text":"Hi!"}          # dialogue
  {"type":"narrate", "text":"It was a dark night."}      # narration
  {"type":"set", "var":"2000", "value":5}                # var = value
  {"type":"add", "var":"2000", "value":1}                # var += value
  {"type":"background", "id":1030}                        # set background asset
  {"type":"music", "id":8225}                            # music track
  {"type":"sfx", "id":8003}                              # sound effect
  {"type":"goto", "scene":2}                             # jump to scene 2
  {"type":"choice", "prompt":"What do you do?",
   "options":[
     {"label":"Be nice", "body":[ <node>,... ]},
     {"label":"Be mean", "body":[ <node>,... ]}
   ]}
  {"type":"gate", "var":"2000", "op":"gte", "value":3,   # if var op value:
   "then":[ <node>... ], "else":[ <node>... ]}           #   then else

The compiler resolves branch targets in logical-PC space (the VM's real model),
packs strings on word-aligned boundaries, builds the KiWi v2 header, and wraps
everything in a CSPUD .exp container. Validate output by decoding it back with
shs_decoder and comparing the story graph.
"""
import json
import re
import struct
import lzma
from pathlib import Path


# --------------------------------------------------------------------------- #
# Opcode / selector constants (verified against decoded episodes)
# --------------------------------------------------------------------------- #
PUSH = 0x1A            # push uint16 operand
# NOTE: 0x41 is a frame-relative LOAD in the VM, not a literal push --
# use PUSH (0x1a) for literal/sentinel values (it applies signed16).
PAIR = 0x1B            # push (hi,lo) pair
PUSH0 = 0x5A           # push literal 0
PUSH1 = 0x5B           # push literal 1
CMP_EQ = 0x0A          # ==
CMP_NE = 0x0B          # !=
CMP_GT = 0x0C          # >
CMP_GE = 0x0D          # >=
CMP_LT = 0x0E          # <
CMP_LE = 0x0F          # <=
JMPF = 0x2B            # pop; jump (pc+operand) if zero/false
JMP = 0x28             # jump pc+operand
SYS = 0x1F             # call builtin: operand = (selector<<8)|argc
ADD = 0x50             # arithmetic add (stack)
SUB = 0x51             # arithmetic subtract (left - right)
MUL = 0x52             # arithmetic multiply
PUSH_RET = 0x42        # push return address (pc+1); frame prologue
PUSH_FP = 0x48         # push current frame pointer
SET_FP = 0x4A          # fp = sp (establish this scene's frame)
POP_FRAME = 0x22       # tear down frame (needs a matching prologue first!)
RET = 0x43             # return: pc = pop() + 1 (needs return addr on stack)
HALT = 0x33            # stop the VM (scene end with no caller to return to)

# builtin selectors (high byte of the SYS operand)
S_CHOICE = 0x01
S_DIALOGUE = 0x0D
S_NARRATION = 0x41
S_GOTO = 0x0A
S_VAR_WRITE = 0x2C
S_VAR_READ = 0x2D
S_BACKGROUND = 0x0B
S_MUSIC = 0x50
S_SFX = 0x4F
S_SET_NAME = 0x31      # set_character_name (yield 49): [char_id, name_ref, fmt]
S_SET_ART = 0x48       # set_character_art_base (yield 72): [char_id, asset_id]

STRING_BASE = 15
SCENE_CHUNK_BASE = 0x61A9   # scene N -> chunk id base+N-1 (25001 = 0x61A9)


class _Asm:
    """Instruction assembler with symbolic labels resolved in logical-PC space."""

    def __init__(self):
        self.ins = []          # list of (opcode, operand_or_None, label_or_None)
        self.labels = {}       # name -> pc index

    def emit(self, opcode, operand=None, target_label=None):
        self.ins.append([opcode, operand, target_label])

    def emit_dispatch(self, match, target_label):
        """op5c section dispatch: if register A == match, PC += dist. The operand
        is (dist<<8)|match; dist is resolved from the target label at resolve()."""
        self.ins.append([0x5C, ("dispatch", match & 0xFF), target_label])

    def label(self, name):
        self.labels[name] = len(self.ins)

    def new_label(self, prefix="L"):
        return "%s%d" % (prefix, len(self.ins) + len(self.labels))

    def resolve(self):
        """Return the final list of (opcode, operand) with labels turned into
        pc-relative (JMP/JMPF) or dispatch (0x5c) operands."""
        out = []
        for pc, (op, operand, tgt) in enumerate(self.ins):
            if tgt is not None:
                dest = self.labels[tgt]
                if op in (JMP, JMPF):
                    operand = (dest - pc) & 0xFFFF     # pc + operand = dest
                elif op == 0x5C:
                    match = operand[1] if isinstance(operand, tuple) else 0
                    dist = (dest - pc) & 0xFF
                    operand = (dist << 8) | match      # (dist<<8)|match
                else:
                    raise ValueError("label on non-jump opcode 0x%02x" % op)
            out.append((op, operand))
        return out


class _StringTable:
    """Packs strings into the word-table byte region and hands out text refs.

    A text ref V addresses the string at byte V*2 + STRING_BASE, so referenced
    strings must begin at an offset where (offset - STRING_BASE) is even. We pack
    NUL-terminated strings and pad to keep every new string on that boundary.
    """

    def __init__(self):
        self.buf = bytearray()
        self.refs = {}         # text -> ref

    def intern(self, text):
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        if text in self.refs:
            return self.refs[text]
        # current byte position within the word-table region
        pos = len(self.buf)
        # referenced string must start where (STRING_BASE+pos - STRING_BASE)=pos is even
        if pos % 2 != 0:
            self.buf.append(0)         # pad byte
            pos += 1
        ref = pos // 2
        self.buf.extend(text.encode("latin1", "replace"))
        self.buf.append(0)             # NUL terminator
        self.refs[text] = ref
        return ref

    def pad_byte(self):
        """Append a single NUL. Used to reserve the blank index-0 cast slot so the
        decoder restores it and the first real name reads clean (no glued header
        byte)."""
        self.buf.append(0)

    def append_words(self, words):
        """Append raw 16-bit words at the current position and return the word
        index (data address) of the first one. Pads to a word boundary first so
        the address is a whole word index -- used for minigame record tables
        (word_grid problems/symbols) that instruction args address directly."""
        if len(self.buf) % 2 != 0:
            self.buf.append(0)
        start_word = len(self.buf) // 2
        for w in words:
            self.buf.append((w >> 8) & 0xFF)
            self.buf.append(w & 0xFF)
        return start_word

    def word_bytes(self):
        """The packed region, padded to an even length (whole 16-bit words)."""
        b = bytearray(self.buf)
        if len(b) % 2 != 0:
            b.append(0)
        return bytes(b)


# --------------------------------------------------------------------------- #
# Per-node code generation
# --------------------------------------------------------------------------- #
def _cast_index(cast, speaker):
    if speaker is None:
        return None
    if speaker in cast:
        return cast.index(speaker)
    return None


# Decoder-output node type -> encoder-internal type, so a spec written in the
# decoder's own JSON schema compiles without translation. The encoder's short
# authoring aliases (say/narrate/set/add/goto) are also accepted.
_TYPE_ALIASES = {
    "dialogue": "say", "say": "say",
    "narration": "narrate", "narrate": "narrate",
    "var_set": "set", "set": "set",
    "var_add": "add", "add": "add",
    "goto_scene": "goto", "goto": "goto",
    "background": "background", "music": "music", "sfx": "sfx",
    "choice": "choice", "gate": "gate", "next": "next", "end": "end",
    "title_card": "title_card", "end_card": "end_card",
}


def _int(v, default=0):
    """Coerce a decoder value to int. Handles ints, numeric strings, and the
    float-like ids the decoder sometimes emits (e.g. '1.142'). Non-numeric ->
    default."""
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            digits = "".join(c for c in s if c.isdigit())
            return int(digits) if digits else default


def _norm_field(node, *names, default=None):
    """First present field among `names` (supports decoder + authoring names)."""
    for k in names:
        if k in node and node[k] is not None:
            return node[k]
    return default


def _gen_node(node, asm, st, cast):
    """Emit instructions for one story node.

    Accepts BOTH the decoder's output schema (type=dialogue/narration/var_set/...
    with fields like track_id, asset_id, sfx_id) and the short authoring aliases
    (say/narrate/set/add/goto). Graph-style control nodes (choice/gate options
    that point to other segments by name) are handled by the segment-graph
    compiler, which inlines them before calling this.
    """
    if not isinstance(node, dict):
        return                              # skip stray non-node entries
    t = _TYPE_ALIASES.get(node.get("type"), node.get("type"))

    if t == "narrate":
        # Narration (yield 0x41) takes 3 args: [character_id, text_ref, override].
        # The text is the SECOND arg; character 0 and override -1 mean "no speaker,
        # no expression override" (engine reads text at index 1).
        ref = st.intern(node["text"])
        asm.emit(PUSH0)                     # character_id (unused for narration)
        asm.emit(PUSH, ref)                 # text ref (read at index 1)
        asm.emit(PUSH, 0xFFFF)              # override = -1 (0x1a pushes signed16)
        asm.emit(SYS, (S_NARRATION << 8) | 3)

    elif t == "say":
        # Dialogue (yield 0x0d): [text_ref, character_id]; text at index 0.
        ref = st.intern(node["text"])
        speaker = node.get("speaker")
        spk = _cast_index(cast, speaker)
        if spk is None and speaker:
            # A one-off named speaker not in the cast (e.g. "Voice") uses the
            # custom-speaker op 1f 0f, which carries the name as a string:
            # push name_ref; push text_ref; push pair; 1f 0f 04.
            asm.emit(PUSH, st.intern(speaker))
            asm.emit(PUSH, ref)
            asm.emit(PAIR, 0xFF00)
            asm.emit(SYS, (0x0F << 8) | 4)
        else:
            asm.emit(PUSH, ref)                 # text ref
            if spk is None:
                asm.emit(PUSH0)
            else:
                asm.emit(PUSH, spk)             # speaker (cast index)
            asm.emit(SYS, (S_DIALOGUE << 8) | 2)

    elif t == "set":
        var = int(node["var"])
        val = int(_norm_field(node, "value", default=0))
        asm.emit(PUSH, var)
        asm.emit(PUSH, val)
        asm.emit(SYS, (S_VAR_WRITE << 8) | 2)

    elif t == "add":
        var = int(node["var"])
        val = int(node.get("value", 1))
        # Compute (var + K), then write with args in [key, value] order. The write
        # yield reads args as (key, value), so push the var id FIRST, then the new
        # value on top: push var ; push var ; read ; push_result ; push K ; ADD ; write.
        asm.emit(PUSH, var)                 # args[0] = key (var id), stays on stack
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)                      # push_result (current value)
        asm.emit(PUSH, val)
        asm.emit(ADD)                       # args[1] = value (var + K) on top
        asm.emit(SYS, (S_VAR_WRITE << 8) | 2)

    elif t in ("var_sub", "sub"):
        # var = var - from_var (or var - K). Uses SUB (0x51).
        var = int(node["var"])
        asm.emit(PUSH, var)                 # args[0] = key
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)                      # current value of var
        if node.get("from_var") is not None:
            asm.emit(PUSH, int(node["from_var"]))
            asm.emit(SYS, (S_VAR_READ << 8) | 1)
            asm.emit(0x21)                  # value of from_var
        else:
            asm.emit(PUSH, int(node.get("value", 0)))
        asm.emit(SUB)                       # var - operand
        asm.emit(SYS, (S_VAR_WRITE << 8) | 2)

    elif t in ("var_random", "random_var"):
        # var = random(min..max). RANDOM (1f 1b) pushes rand()%operand in [0,operand);
        # to get [min, max] we roll (max-min+1) and add min.
        var = int(node["var"])
        lo = int(node.get("min", 0))
        hi = int(node.get("max", 0))
        span = max(1, hi - lo + 1)
        asm.emit(PUSH, var)                 # args[0] = key
        asm.emit(PUSH, span)                # roll modulus
        asm.emit(SYS, (0x1B << 8) | 1)      # RANDOM: result in [0, span)
        asm.emit(0x21)                      # push_result (the roll)
        if lo:
            asm.emit(PUSH, lo)
            asm.emit(ADD)                   # + min
        asm.emit(SYS, (S_VAR_WRITE << 8) | 2)

    elif t == "dispatch":
        _gen_dispatch(node, asm, st, cast)

    elif t == "random":
        _gen_random(node, asm, st, cast)

    elif t == "score_tier":
        _gen_score_tier(node, asm, st, cast)

    elif t == "minigame":
        _gen_minigame(node, asm, st, cast)

    elif t == "zone_background":
        _gen_zone_background(node, asm, st, cast)

    elif t == "background":
        bg = _int(_norm_field(node, "asset_id", "id"), 0)
        if node.get("style") == "custom":
            # Packaged PNG background (asset >= ~26000) via 1f 23 04.
            asm.emit(PUSH, bg & 0xFFFF)
            asm.emit(PUSH, 0xFFFF)
            asm.emit(PAIR, 0xFFFF)
            asm.emit(SYS, (0x23 << 8) | 4)
        else:
            asm.emit(PUSH0)
            asm.emit(PUSH, bg & 0xFFFF)
            asm.emit(PUSH, 0xFFFF)              # -1 sentinel
            asm.emit(SYS, (S_BACKGROUND << 8) | 3)

    elif t == "music":
        if node.get("action") == "stop" or _norm_field(node, "track_id", "id") == "stop":
            asm.emit(PUSH0)                      # stop/fade current music (1f 51)
            asm.emit(SYS, (0x51 << 8) | 1)
        else:
            mid = _int(_norm_field(node, "track_id", "id"), 0)
            asm.emit(PUSH, mid & 0xFFFF)
            asm.emit(PUSH0)
            asm.emit(SYS, (S_MUSIC << 8) | 2)

    elif t == "sfx":
        sid = _int(_norm_field(node, "sfx_id", "id"), 0)
        asm.emit(PUSH, sid & 0xFFFF)
        asm.emit(SYS, (S_SFX << 8) | 1)

    elif t == "stop_music" or t == "stop_audio":
        # yield 81, need(1): stop/fade the current track.
        asm.emit(PUSH0)
        asm.emit(SYS, (0x51 << 8) | 1)

    elif t == "vibrate":
        # yield 82, need(0).
        asm.emit(SYS, (0x52 << 8) | 0)

    elif t == "dialogue_notice" or t == "notification":
        # yield 88, need(1): a one-line notice above the next dialogue.
        ref = st.intern(_norm_field(node, "text", default=""))
        asm.emit(PUSH, ref)
        asm.emit(SYS, (0x58 << 8) | 1)

    elif t == "wobble":
        # yield 89, need(0): wobble the next dialogue line.
        asm.emit(SYS, (0x59 << 8) | 0)

    elif t == "scene_badge":
        # yield 90, need(2): [badge_id, text_ref] (badge_id -1 clears).
        ref = st.intern(_norm_field(node, "text", default="")) \
            if node.get("text") else 0xFFFF
        asm.emit(PUSH, int(node.get("badge_id", 0)) & 0xFFFF)
        asm.emit(PUSH, ref & 0xFFFF)
        asm.emit(SYS, (0x5A << 8) | 2)

    elif t == "loading":
        # yield 91, need(1): show/hide the loading screen (arg is blocking flag).
        asm.emit(PUSH, 1 if node.get("blocking", True) else 0)
        asm.emit(SYS, (0x5B << 8) | 1)

    elif t == "set_string":
        # yield 46, need(2): [key_ref, value_ref]. String variable assignment.
        k = st.intern(str(_norm_field(node, "key", "var", default="")))
        v = st.intern(str(_norm_field(node, "value", "text", default="")))
        asm.emit(PUSH, k)
        asm.emit(PUSH, v)
        asm.emit(SYS, (0x2E << 8) | 2)

    elif t == "presentation":
        # yield 8, need(4): [title, subtitle, style, param]. Framing card.
        title = _norm_field(node, "title", "text", default="")
        subtitle = node.get("subtitle", "")
        asm.emit(PUSH, st.intern(title))
        asm.emit(PUSH, st.intern(subtitle))
        asm.emit(PUSH, int(node.get("style", 0)) & 0xFFFF)
        asm.emit(PUSH, int(node.get("param", 0)) & 0xFFFF)
        asm.emit(SYS, (0x08 << 8) | 4)

    elif t == "vibrate_noop" or t == "native_noop":
        asm.emit(SYS, (0x12 << 8) | 0)     # yield 18, need(0)

    elif t == "append_text":
        # yield 25, need(2): strings[key] += read_text(value). Both refs.
        k = st.intern(str(_norm_field(node, "key", "var", default="")))
        v = st.intern(str(_norm_field(node, "value", "text", default="")))
        asm.emit(PUSH, k)
        asm.emit(PUSH, v)
        asm.emit(SYS, (0x19 << 8) | 2)

    elif t == "random_below":
        # yield 27, need(1): pushes random(0..n-1) into the result register, which
        # we store into `var`. bound n from "max" (exclusive).
        var = int(node["var"])
        n = int(_norm_field(node, "max", "bound", default=2))
        asm.emit(PUSH, var)                 # args[0]=key for the write
        asm.emit(PUSH, n)
        asm.emit(SYS, (0x1B << 8) | 1)      # random_below -> result register
        asm.emit(0x21)                      # push_result
        asm.emit(SYS, (S_VAR_WRITE << 8) | 2)

    elif t == "set_number_bit" or t == "get_number_bit":
        # yield 50/51, need(4/3): bit-addressed number ops. Emit as a plain var op
        # is not equivalent; expose the raw form for round-trip completeness.
        owner = int(node.get("owner", 0))
        key = int(node.get("var", node.get("key", 0)))
        bit = int(node.get("bit", 0))
        sel = 0x32 if t == "set_number_bit" else 0x33
        argc = 4 if t == "set_number_bit" else 3
        asm.emit(PUSH, owner)
        asm.emit(PUSH, key)
        asm.emit(PUSH, bit)
        if argc == 4:
            asm.emit(PUSH, int(node.get("value", 0)) & 0xFFFF)
        asm.emit(SYS, (sel << 8) | argc)

    elif t == "set_scene_value":
        # yield 16 (0x10), need(1): sets the scene value register.
        asm.emit(PUSH, int(_norm_field(node, "value", default=0)) & 0xFFFF)
        asm.emit(SYS, (0x10 << 8) | 1)

    elif t == "set_ui_default":
        # yield 75 (0x4b), need(1): sets a UI default slot value.
        asm.emit(PUSH, int(_norm_field(node, "value", default=0)) & 0xFFFF)
        asm.emit(SYS, (0x4B << 8) | 1)

    elif t == "text_input":
        # yield 40 (0x28), need(3): [title, prompt, destination string slot]. The
        # destination is a word address where the typed text is stored; reserve a
        # small run of words in the data region and point at it.
        title = st.intern(_norm_field(node, "title", default=""))
        prompt = st.intern(_norm_field(node, "prompt", "text", default=""))
        dest = st.append_words([0] * 16)   # reserve 16 words for the input buffer
        asm.emit(PUSH, title)
        asm.emit(PUSH, prompt)
        asm.emit(PUSH, dest & 0xFFFF)
        asm.emit(SYS, (0x28 << 8) | 3)

    elif t == "character_picker":
        # yield 78 (0x4e), need(3): [prompt, count, address]. The address points at
        # a table of `count` character ids (1..5) in the word region.
        prompt = st.intern(_norm_field(node, "prompt", "text", default=""))
        ids = [int(x) for x in (node.get("character_ids") or node.get("ids") or [])]
        if not 1 <= len(ids) <= 5:
            ids = ids[:5] or [1]
        addr = st.append_words(ids)
        asm.emit(PUSH, prompt)
        asm.emit(PUSH, len(ids) & 0xFFFF)
        asm.emit(PUSH, addr & 0xFFFF)
        asm.emit(SYS, (0x4E << 8) | 3)

    elif t == "goto":
        # authoring: {scene: 2}; decoder: {file: "s2"} or to_scene number.
        if "scene" in node:
            scene_num = int(node["scene"])
        else:
            fname = _norm_field(node, "file", "to_scene")
            scene_num = _scene_id_from_seg(fname) if isinstance(fname, str) \
                else int(fname or 1)
        chunk = SCENE_CHUNK_BASE + (scene_num - 1)
        asm.emit(PUSH, chunk)
        asm.emit(PUSH0)
        asm.emit(SYS, (S_GOTO << 8) | 2)

    elif t == "goto_scene":
        _gen_node({**node, "type": "goto"}, asm, st, cast)

    elif t == "title_card":
        # Title card via 1f 08 (episode title + optional subtitle).
        text = _norm_field(node, "text", "title", default="")
        sub = node.get("subtitle", "")
        asm.emit(PUSH, st.intern(text))
        asm.emit(PUSH, st.intern(sub))
        asm.emit(PUSH, int(node.get("style", 0)) & 0xFFFF)
        asm.emit(PUSH, int(node.get("param", 0)) & 0xFFFF)
        asm.emit(SYS, (0x08 << 8) | 4)

    elif t == "end_card":
        # End-of-episode credits card. The engine renders these via a text
        # heuristic at a section separator; emitting a raw separator here risks
        # disrupting scene flow, so emit the text as narration -- the content is
        # preserved and shown, which is what matters for authoring.
        text = _norm_field(node, "text", default="")
        if text:
            _gen_node({"type": "narrate", "text": text}, asm, st, cast)

    elif t == "notification":
        # A score/feedback banner (decoded from 0x58). Carries text, so emit it
        # via the same score-feedback op rather than dropping it.
        txt = _norm_field(node, "text", default="")
        if txt:
            asm.emit(PUSH, st.intern(txt))
            asm.emit(SYS, (0x58 << 8) | 1)

    elif t == "status":
        # On-screen status/HUD line via 1f 00. Carries a printf-style template the
        # engine fills at runtime; optionally a value_from var supplies the %d.
        txt = _norm_field(node, "text", default="")
        if txt:
            vf = node.get("value_from")
            if isinstance(vf, dict) and vf.get("var") is not None:
                asm.emit(PUSH, _int(vf["var"]))
                asm.emit(SYS, (S_VAR_READ << 8) | 1)
                asm.emit(0x21)                  # push_result (the value)
                asm.emit(PUSH, st.intern(txt))
                asm.emit(SYS, (0x00 << 8) | 2)
            else:
                asm.emit(PUSH, st.intern(txt))
                asm.emit(SYS, (0x00 << 8) | 1)

    elif t == "pov_change":
        # "Now playing as X" -- set the POV character via 1f 4a by cast index.
        idx = node.get("index")
        if idx is None:
            idx = _cast_index(cast, node.get("character"))
        if idx is not None:
            asm.emit(PUSH, _int(idx) & 0xFFFF)
            asm.emit(SYS, (0x4A << 8) | 1)

    elif t == "_goto_seg":
        pass                                    # already-emitted merge marker

    elif t == "gate":
        _gen_gate(node, asm, st, cast)

    elif t == "choice":
        _gen_choice(node, asm, st, cast)

    elif t in ("random_encounter", "checkpoint_replay", "dispatch_combat"):
        # Combat/checkpoint constructs the semantic encoder cannot fully
        # regenerate (their turn-loop/save-state can't be statically flattened --
        # use shs_faithful for byte-exact round-trip of these). Preserve what is
        # expressible: roll a random into the time/roll var so downstream reads
        # have a value, and run any inline win/first-option body so the episode
        # stays playable rather than failing to compile.
        rv = node.get("time_var") or node.get("var")
        rmax = node.get("roll_max") or node.get("time_total")
        if rv is not None and rmax is not None:
            _gen_node({"type": "var_random", "var": _int(rv),
                       "min": 0, "max": _int(rmax)}, asm, st, cast)
        opts = node.get("options")
        if isinstance(opts, list) and opts and isinstance(opts[0], dict):
            for sub in (opts[0].get("body") or []):
                _gen_node(sub, asm, st, cast)

    else:
        # Unknown/unsupported node: don't fail the whole episode. Emit any text it
        # carries as narration so nothing visible is silently dropped.
        txt = node.get("text") or node.get("prompt")
        if txt:
            _gen_node({"type": "narrate", "text": txt}, asm, st, cast)


_CMP = {"eq": CMP_EQ, "ne": CMP_NE, "gt": CMP_GT,
        "gte": CMP_GE, "lt": CMP_LT, "lte": CMP_LE}


def _flatten_covered_segments(segments, entry):
    """Which segment ids the flattener visits from `entry` (following next/then/
    else/choice pointers). Used to find same-scene segments reached only via
    dispatch/goto, so their content isn't dropped."""
    from collections import deque
    seen, dq = set(), deque([entry])
    while dq:
        s = dq.popleft()
        if s in seen or s not in segments:
            continue
        seen.add(s)
        for n in segments[s].get("nodes", []):
            if not isinstance(n, dict):
                continue
            for k in ("next", "then", "else", "win", "lose"):
                if isinstance(n.get(k), str):
                    dq.append(n[k])
            if n.get("type") == "choice":
                for o in n.get("options", []) or []:
                    if isinstance(o, dict) and isinstance(o.get("next"), str):
                        dq.append(o["next"])
            elif n.get("type") == "random_encounter":
                for o in n.get("options", []) or []:
                    if isinstance(o, dict) and isinstance(o.get("fight"), str):
                        dq.append(o["fight"])
    return seen


def _flatten_segment_graph(segments, entry):
    """Turn a decoder-style scene (flat {seg_id: {nodes:[...]}} with next/then/
    else/options[].next pointers) into the nested authoring node list the code
    generators consume.

    Strategy: walk from `entry`, inlining each segment's content nodes. At a
    control node (choice/gate) whose arms point to other segments, inline each
    arm up to the point where the arms reconverge (a shared "merge" segment),
    then continue from the merge once. This reproduces the original nested shape
    (choice.options[].body, gate.then/else) that the bytecode was built from.
    A segment is emitted at most once; a back-reference to an already-emitted or
    shared segment becomes a `goto`-free fallthrough (the linear next).
    """
    def succ(seg_id):
        seg = segments.get(seg_id, {})
        out = []
        for n in seg.get("nodes", []):
            if not isinstance(n, dict):
                continue
            for k in ("next", "then", "else", "win", "lose"):
                if isinstance(n.get(k), str):
                    out.append(n[k])
            # A choice's options point at segments via `next`; a random_encounter's
            # options point at combat segments via `fight`.
            if n.get("type") == "choice":
                for o in n.get("options", []) or []:
                    if isinstance(o, dict) and isinstance(o.get("next"), str):
                        out.append(o["next"])
            elif n.get("type") == "random_encounter":
                for o in n.get("options", []) or []:
                    if isinstance(o, dict) and isinstance(o.get("fight"), str):
                        out.append(o["fight"])
        return out

    # Reference counts: a segment reached by >1 path is a merge point where arms
    # reconverge; it is emitted once, after the branching node.
    from collections import defaultdict, deque
    refs = defaultdict(int)
    seen = set()
    dq = deque([entry])
    while dq:
        s = dq.popleft()
        if s in seen:
            continue
        seen.add(s)
        for t in succ(s):
            refs[t] += 1
            dq.append(t)

    emitted = set()

    def norm_target(name):
        return name.replace(".json", "") if isinstance(name, str) else name

    def inline(seg_id, stop_at):
        """Emit seg_id's nodes as a flat list, stopping (returning the id) when we
        reach a segment in `stop_at` (a shared merge) rather than descending."""
        out = []
        cur = seg_id
        while cur is not None and cur not in stop_at:
            if cur in emitted:
                # already emitted elsewhere -> jump to it via a synthetic marker
                out.append({"type": "_goto_seg", "seg": cur})
                return out, None
            emitted.add(cur)
            seg = segments.get(cur, {})
            nxt = None
            for n in seg.get("nodes", []):
                if not isinstance(n, dict):
                    continue
                nt = n.get("type")
                if nt == "next":
                    nxt = norm_target(n.get("next"))
                elif nt == "gate":
                    then_id = norm_target(n.get("then"))
                    else_id = norm_target(n.get("else"))
                    merge = _find_merge(then_id, else_id, succ, refs)
                    sub_stop = stop_at | ({merge} if merge else set())
                    then_body, _ = inline(then_id, sub_stop) if then_id else ([], None)
                    else_body, _ = inline(else_id, sub_stop) if else_id else ([], None)
                    gnode = {"type": "gate", "var": n.get("var"),
                             "op": n.get("op", "eq"),
                             "value": int(n.get("equals", n.get("value", 0))),
                             "then": then_body, "else": else_body}
                    out.append(gnode)
                    nxt = merge
                    break
                elif nt == "choice":
                    opt_ids = [norm_target(o.get("next")) for o in n.get("options", [])]
                    merge = _find_merge_many(opt_ids, succ, refs)
                    sub_stop = stop_at | ({merge} if merge else set())
                    cnode = {"type": "choice",
                             "prompt": n.get("prompt", "Make your choice!"),
                             "options": []}
                    for o in n.get("options", []):
                        oid = norm_target(o.get("next"))
                        body, _ = inline(oid, sub_stop) if oid else ([], None)
                        cnode["options"].append({"label": o.get("label", ""),
                                                 "body": body})
                    out.append(cnode)
                    nxt = merge
                    break
                elif nt == "goto_scene":
                    out.append(n)
                    nxt = None
                    break
                elif nt == "end":
                    nxt = None
                    break
                else:
                    out.append(n)      # plain content node (dialogue, var, media...)
            cur = nxt
        return out, cur

    body, _ = inline(entry, set())
    # Safety net: the nested-inline model can orphan content when several gates
    # share a merge point (cascading var-gates), because a shared merge is emitted
    # once from the outermost gate and its siblings' pre-merge content can fall
    # through -- and a segment reached again is emitted as a _goto_seg marker with
    # its content skipped. Append the content of any covered segment whose nodes
    # never actually made it into the output, so nothing is lost.
    def _emitted_text_ids(nlist, acc):
        for n in nlist:
            if not isinstance(n, dict):
                continue
            if n.get("type") in ("dialogue", "narration") and n.get("text"):
                acc.add(n["text"])
            for k in ("then", "else"):
                if isinstance(n.get(k), list):
                    _emitted_text_ids(n[k], acc)
            for o in n.get("options", []) or []:
                if isinstance(o, dict) and isinstance(o.get("body"), list):
                    _emitted_text_ids(o["body"], acc)
        return acc

    emitted_text = _emitted_text_ids(body, set())
    covered = _flatten_covered_segments(segments, entry)
    for sid in covered:
        seg = segments.get(sid, {})
        # does this segment have text not present anywhere in the output?
        seg_text = {n["text"] for n in seg.get("nodes", [])
                    if isinstance(n, dict)
                    and n.get("type") in ("dialogue", "narration") and n.get("text")}
        if not seg_text or seg_text <= emitted_text:
            continue
        for n in seg.get("nodes", []):
            if isinstance(n, dict) and n.get("type") not in ("next", "_goto_seg"):
                nn = {k: v for k, v in n.items()
                      if k not in ("then", "else") or not isinstance(v, str)}
                body.append(nn)
        emitted_text |= seg_text
    return body


def _find_merge(a, b, succ, refs):
    """The nearest segment both branches reach (a shared continuation), or None."""
    if not a or not b:
        return None
    seen_a = _reachable(a, succ)
    seen_b = _reachable(b, succ)
    common = seen_a & seen_b
    # the merge is a common node with >1 incoming ref (a real join)
    joins = [c for c in common if refs.get(c, 0) > 1]
    if not joins:
        return None
    # pick the one closest to both (smallest combined distance)
    return min(joins, key=lambda c: _dist(a, c, succ) + _dist(b, c, succ))


def _find_merge_many(ids, succ, refs):
    ids = [i for i in ids if i]
    if len(ids) < 2:
        return None
    reach = [_reachable(i, succ) for i in ids]
    common = set.intersection(*reach) if reach else set()
    joins = [c for c in common if refs.get(c, 0) > 1]
    if not joins:
        return None
    return min(joins, key=lambda c: sum(_dist(i, c, succ) for i in ids))


def _reachable(start, succ):
    from collections import deque
    seen, dq = set(), deque([start])
    while dq:
        s = dq.popleft()
        if s in seen:
            continue
        seen.add(s)
        for t in succ(s):
            dq.append(t)
    return seen


def _dist(start, target, succ):
    from collections import deque
    dq = deque([(start, 0)])
    seen = set()
    while dq:
        s, d = dq.popleft()
        if s == target:
            return d
        if s in seen:
            continue
        seen.add(s)
        for t in succ(s):
            dq.append((t, d + 1))
    return 999


def _seg_scene_num(name):
    """'s2_05' -> 2 ; used to route dispatch/random arms to scene chunks."""
    m = re.match(r"s(\d+)", name or "")
    return int(m.group(1)) if m else None


def _emit_goto_scene_num(asm, scene_num):
    chunk = SCENE_CHUNK_BASE + (int(scene_num) - 1)
    asm.emit(PUSH, chunk)
    asm.emit(PUSH0)
    asm.emit(SYS, (S_GOTO << 8) | 2)


def _gen_dispatch(node, asm, st, cast):
    """Section-register dispatch: load state_var into register A, then a cascade
    of op5c (if A == match: jump to that arm's goto-scene), falling through to the
    default. Mirrors the real scene-hub prologue.

    Arms/default reference scenes by segment name (e.g. 's2_01'); we route to the
    scene number that name belongs to. Arms whose target is in the SAME scene (a
    mid-scene section, not a chunk) can't be a cross-scene goto -- those are left
    to fall through to the default, since a new episode expresses branching with
    choices/gates rather than the decoder's section-resume hubs.
    """
    var = int(node["state_var"])
    arms = node.get("arms", [])
    default = node.get("default")
    # load state var -> register A
    asm.emit(PUSH, var)
    asm.emit(SYS, (S_VAR_READ << 8) | 1)
    asm.emit(0x21)                          # push_result
    asm.emit(0x03)                          # A = pop
    end_lbl = asm.new_label("dend")
    arm_lbls = []
    for k, arm in enumerate(arms):
        scene_num = _seg_scene_num(arm.get("scene"))
        if scene_num is None:
            continue
        lbl = asm.new_label("darm")
        arm_lbls.append((lbl, scene_num))
        asm.emit_dispatch(int(arm["equals"]), lbl)
    # fall-through: default
    def_num = _seg_scene_num(default) if isinstance(default, str) else None
    if def_num is not None:
        _emit_goto_scene_num(asm, def_num)
    asm.emit(JMP, target_label=end_lbl)
    # per-arm goto stubs
    for lbl, scene_num in arm_lbls:
        asm.label(lbl)
        _emit_goto_scene_num(asm, scene_num)
        asm.emit(JMP, target_label=end_lbl)
    asm.label(end_lbl)


def _gen_random(node, asm, st, cast):
    """Random scene selection: roll RANDOM(0..n-1), then dispatch the roll to one
    of n goto-scene arms. Uses the same op5c cascade keyed on the roll value."""
    options = node.get("options", [])
    n = len(options)
    if n == 0:
        return
    asm.emit(PUSH, n)
    asm.emit(SYS, (0x1B << 8) | 1)          # RANDOM in [0, n)
    asm.emit(0x21)                          # push_result (roll)
    asm.emit(0x03)                          # A = roll
    end_lbl = asm.new_label("rend")
    arm_lbls = []
    for k, opt in enumerate(options):
        scene_num = _seg_scene_num(opt.get("scene"))
        if scene_num is None:
            continue
        lbl = asm.new_label("rarm")
        arm_lbls.append((lbl, scene_num))
        asm.emit_dispatch(k, lbl)
    asm.emit(JMP, target_label=end_lbl)
    for lbl, scene_num in arm_lbls:
        asm.label(lbl)
        _emit_goto_scene_num(asm, scene_num)
        asm.emit(JMP, target_label=end_lbl)
    asm.label(end_lbl)


def _gen_score_tier(node, asm, st, cast):
    """End-of-episode rank cascade: for each tier (var <= threshold) show its rank
    banner; the default (highest) tier shows its banner+rank. Emitted as a chain
    of gates on the score var, each printing the tier's rank as narration."""
    var = int(node["var"])
    tiers = node.get("tiers", [])
    default = node.get("default", {})
    end_lbl = asm.new_label("stend")
    for tier in tiers:
        op = _CMP.get(tier.get("op", "lte"), CMP_LE)
        thr = int(tier["threshold"])
        skip = asm.new_label("stskip")
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)
        asm.emit(PUSH, thr)
        asm.emit(op)                        # var <op> threshold
        asm.emit(JMPF, target_label=skip)   # not this tier -> next
        _gen_node({"type": "narrate", "text": tier.get("rank", "")}, asm, st, cast)
        asm.emit(JMP, target_label=end_lbl)
        asm.label(skip)
    # default (top rank)
    if isinstance(default, dict):
        if default.get("banner"):
            _gen_node({"type": "narrate", "text": default["banner"]}, asm, st, cast)
        if default.get("rank"):
            _gen_node({"type": "narrate", "text": default["rank"]}, asm, st, cast)
    asm.label(end_lbl)


def _gen_minigame(node, asm, st, cast):
    """Emit a pick-word minigame (yield 71 / 0x47) and its win/lose score gate.

    Byte-derived from Tutors/Float: push the result-slot address (op5f), then 8
    args [title, prompt, correct, decoys, duration_ms, character_id, portrait_mode,
    timing2], call `1f 47 08`, store the returned score to the slot, then compare
    it against win_threshold and branch to the win body (>= threshold) or lose
    body (<). The engine requires >=2 correct words, >=3 decoys, and duration > 0.

    Authoring schema:
      {"type":"minigame","minigame_type":"pick_word",
       "prompt":"Pick words that mean 'pretty!'",
       "correct":["Radiant!","Beauteous!","Fetching!"],
       "decoys":["Homely!","Hideous!","Monstrous!"],
       "win_threshold":6, "duration_ms":20000,
       "speaker":"Zoe",
       "win":[<nodes>...], "lose":[<nodes>...]}
    win/lose may be inline node lists (authoring form). Segment-ref targets (the
    decoder's win/lose seg names) are handled by the graph flattener upstream.
    """
    mtype = node.get("minigame_type", "pick_word")
    if mtype == "build_word" or mtype == "word_grid":
        _gen_build_word(node, asm, st, cast)
        return
    if mtype not in ("pick_word", None):
        # Only pick_word is generatable so far; other types (build_word, football,
        # word_grid, action_tap) need their own generators. Emit the prompt as a
        # narration so nothing is silently dropped, then run the win body.
        prompt = node.get("prompt") or "(minigame)"
        _gen_node({"type": "narrate", "text": prompt}, asm, st, cast)
        for n in (node.get("win") or []):
            _gen_node(n, asm, st, cast)
        return

    correct = node.get("correct") or []
    decoys = node.get("decoys") or []
    # The decoder emits pick_word word-groups as `options: [[correct...],
    # [decoys...]]` rather than named keys; accept that shape too.
    if (not correct or not decoys) and isinstance(node.get("options"), list):
        groups = [g for g in node["options"] if isinstance(g, list)]
        if len(groups) >= 2:
            correct = correct or groups[0]
            decoys = decoys or groups[1]
    if len(correct) < 2 or len(decoys) < 3:
        # Not enough word data to build a valid game (e.g. a decoded minigame whose
        # word bank lived in a separate data block). Degrade to a prompt + win body
        # rather than fail the whole episode, so the round-trip still produces a
        # playable file.
        prompt = node.get("prompt") or "(minigame)"
        _gen_node({"type": "narrate", "text": prompt}, asm, st, cast)
        for n in (node.get("win") or []):
            if isinstance(n, dict):
                _gen_node(n, asm, st, cast)
        return
    title = node.get("title", "Make your choice!")
    prompt = node.get("prompt", "")
    duration = int(node.get("duration_ms", 20000))
    if duration <= 0:
        duration = 20000
    threshold = int(node.get("win_threshold", len(correct)))
    timing2 = int(node.get("timing2", 3000))
    spk = _cast_index(cast, node.get("speaker"))
    portrait_mode = int(node.get("portrait_mode", 13))
    char_id = spk if spk is not None else 1

    title_ref = st.intern(title)
    prompt_ref = st.intern(prompt)
    correct_ref = st.intern("|".join(correct))
    decoy_ref = st.intern("|".join(decoys))

    # push result-slot address, then the 8 args, then the minigame yield
    asm.emit(0x5F)                              # @fp0 result slot address
    asm.emit(PUSH, title_ref)                   # args[0] title
    asm.emit(PUSH, prompt_ref)                  # args[1] prompt/subtitle
    asm.emit(PUSH, correct_ref)                 # args[2] correct words
    asm.emit(PUSH, decoy_ref)                   # args[3] decoy words
    asm.emit(PUSH, duration & 0xFFFF)           # args[4] duration ms
    asm.emit(PAIR, (char_id << 8) | (portrait_mode & 0xFF))  # args[5],[6]
    asm.emit(PUSH, timing2 & 0xFFFF)            # args[7] timing2
    asm.emit(SYS, (0x47 << 8) | 8)              # 1f 47 08 -> word minigame
    # store the returned score into the frame slot, then gate on it
    asm.emit(0x21)                              # push_result (score)
    asm.emit(0x3E)                              # STORE to @fp0
    asm.emit(0x15)                              # drop
    lose_lbl = asm.new_label("mglose")
    end_lbl = asm.new_label("mgend")
    asm.emit(0x5F)                              # @fp0 addr
    asm.emit(0x3F)                              # load score
    asm.emit(PUSH, threshold)
    asm.emit(CMP_GE)                            # score >= threshold ?
    asm.emit(JMPF, target_label=lose_lbl)       # below -> lose
    for n in (node.get("win") or []):
        _gen_node(n, asm, st, cast)
    asm.emit(JMP, target_label=end_lbl)
    asm.label(lose_lbl)
    for n in (node.get("lose") or []):
        _gen_node(n, asm, st, cast)
    asm.label(end_lbl)


def _gen_zone_background(node, asm, st, cast):
    """Set a background by a variable's threshold (highest threshold met wins):
    a cascade of gates, highest threshold first, each setting its background then
    jumping to the end."""
    var = _int(node.get("var"), 0)
    zones = sorted(node.get("zones", []),
                   key=lambda z: _int(z.get("threshold"), 0), reverse=True)
    end_lbl = asm.new_label("zbend")
    for z in zones:
        thr = _int(z.get("threshold"), 0)
        bg = _int(z.get("bg"), 0)
        skip = asm.new_label("zbskip")
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)
        asm.emit(PUSH, thr)
        asm.emit(CMP_GE)                    # var >= threshold ?
        asm.emit(JMPF, target_label=skip)
        _gen_node({"type": "background", "id": bg}, asm, st, cast)
        asm.emit(JMP, target_label=end_lbl)
        asm.label(skip)
    asm.label(end_lbl)


def _gen_build_word(node, asm, st, cast):
    """Emit a build-word / word-grid minigame (yield 96 / 0x60, 20 args).

    Byte/engine-derived (read_grid): the 20 args carry timing, a target score, and
    the ADDRESS+COUNT of a problem record table placed in the word region. Each
    problem is 10 words: [width, height, heading_ref, prompt_ref, words_ref(pipe),
    alphabet_ref, highlight, advance_after_one, minimum_starts, identifier]. We
    support a single problem (the common case) plus win/lose routing on the score.

    Authoring schema:
      {"type":"minigame","minigame_type":"build_word",
       "heading":"Find the words","prompt":"Spell them out!",
       "words":["CAT","DOG"], "alphabet":"CATDOGX",
       "width":3, "height":2, "min_starts":1,
       "target":2, "duration_sec":30,
       "win":[<nodes>], "lose":[<nodes>]}
    The alphabet must contain every letter of every word, and each word must fit
    in width*height cells (the engine validates this).
    """
    words = node.get("words") or []
    alphabet = node.get("alphabet") or ""
    if not words or not alphabet:
        # A decoded by_id build_word has its word bank in a separate block, so no
        # direct words/alphabet here. Degrade to a prompt + win body rather than
        # failing the whole episode.
        prompt = node.get("prompt") or node.get("heading") or "(word game)"
        _gen_node({"type": "narrate", "text": prompt}, asm, st, cast)
        for n in (node.get("win") or []):
            if isinstance(n, dict):
                _gen_node(n, asm, st, cast)
        return
    width = int(node.get("width", 3))
    height = int(node.get("height", 2))
    if not (1 <= width <= 5 and 1 <= height <= 5):
        raise ValueError("build_word grid must be 1..5 in each dimension")
    for w in words:
        if any(c not in alphabet for c in w):
            raise ValueError("alphabet %r cannot form word %r" % (alphabet, w))
        if len(w) > width * height:
            raise ValueError("word %r does not fit a %dx%d grid" % (w, width, height))
    min_starts = int(node.get("min_starts", 1))
    target = int(node.get("target", len(words)))
    duration_sec = int(node.get("duration_sec", 30))
    heading = node.get("heading", "")
    prompt = node.get("prompt", "")

    heading_ref = st.intern(heading)
    prompt_ref = st.intern(prompt)
    words_ref = st.intern("|".join(words))
    alpha_ref = st.intern(alphabet)
    # problem record: 10 words. highlight/advance = 0; id 1001 (native default).
    problem = [width, height, heading_ref, prompt_ref, words_ref, alpha_ref,
               0, 0, min_starts, 1001]
    prob_addr = st.append_words(problem)

    # 20 args. Layout from read_grid: [0]=duration_sec(*1000), [1]=target,
    # [2]=round_limit, [3]=bonus, [4..9]=char/expr (0), [10]=shuffle, [11]=count,
    # [12]=address, [13]=feedback, [14]=success_text, [15]=failure_text,
    # [16]=sym_count, [17]=sym_addr, [18]=tut_count, [19]=tut_addr.
    args = [0] * 20
    args[0] = duration_sec
    args[1] = target
    args[2] = int(node.get("round_limit_ms", 20000))   # must be > 0, <= 32767
    args[3] = int(node.get("bonus_ms", 5000))          # 0 <= bonus <= 32767
    args[11] = 1
    args[12] = prob_addr
    asm.emit(0x5F)                              # result-slot address
    for a in args:
        asm.emit(PUSH, a & 0xFFFF)
    asm.emit(SYS, (0x60 << 8) | 20)             # 1f 60 14 -> word_grid
    # store score and gate on it (score >= target -> win)
    asm.emit(0x21)
    asm.emit(0x3E)
    asm.emit(0x15)
    lose_lbl = asm.new_label("bwlose")
    end_lbl = asm.new_label("bwend")
    asm.emit(0x5F)
    asm.emit(0x3F)
    asm.emit(PUSH, target)
    asm.emit(CMP_GE)
    asm.emit(JMPF, target_label=lose_lbl)
    for n in (node.get("win") or []):
        _gen_node(n, asm, st, cast)
    asm.emit(JMP, target_label=end_lbl)
    asm.label(lose_lbl)
    for n in (node.get("lose") or []):
        _gen_node(n, asm, st, cast)
    asm.label(end_lbl)


def _gen_gate(node, asm, st, cast):
    """if (var OP value) { then } else { else }.

    Emits: read var; push value; CMP; JMPF else; <then>; JMP end; else: <else>; end:
    JMPF jumps when the comparison is FALSE (0), so the fall-through is the THEN
    arm -- matching how the decoder reads gates.
    """
    val = _int(_norm_field(node, "value", "equals"), 0)
    op = _CMP[node.get("op") or "eq"]
    else_lbl = asm.new_label("gelse")
    end_lbl = asm.new_label("gend")
    if str(node.get("var")) == "score":
        # A score-gate tests the minigame score accumulator, not a numbered var.
        # The score sits in frame slot fp+0 (where minigames store their result),
        # so read it via op5f/op3f rather than a var-read yield.
        asm.emit(0x5F)                      # @fp0 (score slot address)
        asm.emit(0x3F)                      # load score
    else:
        var = _int(node["var"], 0)
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)                      # push_result (the var's value)
    asm.emit(PUSH, val)
    asm.emit(op)
    asm.emit(JMPF, target_label=else_lbl)
    for n in node.get("then", []):
        _gen_node(n, asm, st, cast)
    asm.emit(JMP, target_label=end_lbl)
    asm.label(else_lbl)
    for n in node.get("else", []):
        _gen_node(n, asm, st, cast)
    asm.label(end_lbl)


def _gen_choice(node, asm, st, cast):
    """Present a menu and run the chosen option's body.

    Setup:  push prompt; push "opt1|opt2|..."; pair sentinels; push1; SYS choice.
    Routing per option i: push SCORE-answer; push i; CMP==; JMPF next; <body>; JMP end.
    The engine leaves the selected index in the answer slot; we test it against
    each option index in turn.
    """
    options = node["options"]
    prompt = node.get("prompt", "Make your choice!")
    opt_str = "|".join(o["label"] for o in options)
    lead_in = node.get("text", "")            # optional lead-in line under the title
    title_ref = st.intern(prompt)
    opts_ref = st.intern(opt_str)
    text_ref = st.intern(lead_in) if lead_in else None
    # 8 args in engine order: [title, options, text, timeout, timeout_result,
    # character, reserved, portrait_mode]. Matches real episodes:
    #   push title; push options; push text; pair 0xffff; pair 0xfffe; push1.
    asm.emit(PUSH, title_ref)                  # args[0] title
    asm.emit(PUSH, opts_ref)                   # args[1] options ("a|b|c")
    if text_ref is not None:
        asm.emit(PUSH, text_ref)               # args[2] lead-in text
    else:
        asm.emit(PUSH, 0xFFFF)                 # args[2] = -1 (no lead-in)
    # args[3]=timeout_ms, args[4]=timeout_result. A timed choice counts down and
    # auto-selects timeout_result when it expires. Default is no timeout (-1/-1).
    timeout_ms = int(node.get("timeout_ms", 0))
    if timeout_ms > 0:
        timeout_result = int(node.get("timeout_result", 1000))
        asm.emit(PUSH, timeout_ms & 0xFFFF)    # args[3] timeout
        asm.emit(PUSH, timeout_result & 0xFFFF)  # args[4] result
    else:
        asm.emit(PAIR, 0xFFFF)                 # args[3]=-1 timeout, args[4]=-1 result
    asm.emit(PAIR, 0xFFFE)                     # args[5]=-2 character, args[6]=-1
    asm.emit(PUSH1)                            # args[7] portrait_mode = 1
    asm.emit(SYS, (S_CHOICE << 8) | 8)
    # Answer routing (mirrors real episodes). 0x3e STORE pops (value, address), so
    # push the destination slot ADDRESS first, then push_result (the chosen index),
    # then STORE writes it and leaves the value; sp-- drops it. Each option then
    # re-reads the slot via @fp0/LOAD and compares.
    asm.emit(0x5F)                       # push address of frame slot fp+0
    asm.emit(0x21)                       # push_result (the chosen index)
    asm.emit(0x3E)                       # STORE: write result -> slot, push value
    asm.emit(0x15)                       # sp -= 1 (drop the pushed value)
    end_lbl = asm.new_label("cend")
    for i, opt in enumerate(options):
        next_lbl = asm.new_label("copt")
        asm.emit(0x5F)                   # push address of frame slot fp+0
        asm.emit(0x3F)                   # load the stored answer
        if i == 0:
            asm.emit(PUSH0)
        elif i == 1:
            asm.emit(PUSH1)
        else:
            asm.emit(PUSH, i)
        asm.emit(CMP_EQ)                 # answer == i ?
        asm.emit(JMPF, target_label=next_lbl)
        for n in opt.get("body", []):
            _gen_node(n, asm, st, cast)
        asm.emit(JMP, target_label=end_lbl)
        asm.label(next_lbl)
    asm.label(end_lbl)


# --------------------------------------------------------------------------- #
# Scene -> .kiw chunk, and episode -> .exp container
# --------------------------------------------------------------------------- #
def _serialize_instructions(instrs):
    """(opcode, operand) list -> bytes, operand present iff opcode is 3-byte."""
    from shs_decoder import _KIWI_OPERAND_OPCODES
    out = bytearray()
    for op, operand in instrs:
        out.append(op)
        if op in _KIWI_OPERAND_OPCODES:
            if operand is None:
                operand = 0
            out += struct.pack(">H", operand & 0xFFFF)
    return bytes(out)


def _build_scene_chunk(scene, cast, characters=None):
    """Compile one scene into a complete .kiw byte string.

    `characters` maps a cast name to its config, e.g. {"Zoe": {"art": 2001}}.
    Every scene registers the cast's display names (and art bases, when given) at
    entry so the engine shows speaker names and portraits instead of falling back
    to the nameless narrator presentation.
    """
    characters = characters or {}
    asm = _Asm()
    st = _StringTable()
    # Cast/name table FIRST: the decoder reads NUL-terminated names starting at
    # byte 15, up to an "Event" marker. Byte 14 is the instruction-count low byte
    # of the header, which can be printable and would otherwise glue onto the
    # first name. Emit a single leading NUL so the first real name always starts
    # clean at an even offset, then the cast names, then "Event". This mirrors the
    # blank leading-slot form the decoder already restores (cast index 0 = the
    # player). Names are interned before any dialogue text so they sit at the
    # front of the word table where read_cast looks.
    if cast:
        st.pad_byte()                          # leading NUL: index-0 blank slot
        for i, name in enumerate(cast):
            if i == 0 and not name:
                continue                       # blank player slot already padded
            st.intern(name)
        st.intern("Event")
    # Frame prologue: a scene runs as a function frame. PUSH_RET/PUSH_FP/SET_FP
    # establish it so the stack pointer and frame pointer are valid before the
    # body executes any pops or yields. (Without this the VM crashes on the very
    # first frame-relative or stack op -- "stack address -1 out of range".)
    asm.emit(PUSH_RET)
    asm.emit(PUSH_FP)
    asm.emit(SET_FP)
    asm.emit(PUSH0)            # reserve frame slot fp+0 (choice answer is stored here)
    # Register each named cast member so the UI shows their name (and portrait).
    # The engine only shows a speaker name when the character has registered art
    # variants, so set_character_art_base is what makes BOTH the name and image
    # appear; set_character_name supplies the label.
    for idx, name in enumerate(cast or []):
        if idx == 0 or not name:
            continue                       # index 0 is the blank/player slot
        name_ref = st.intern(name)
        asm.emit(PUSH, idx)                # char_id
        asm.emit(PUSH, name_ref)           # name text ref
        asm.emit(PUSH0)                    # format flag (0 = use as-is)
        asm.emit(SYS, (S_SET_NAME << 8) | 3)
        art = characters.get(name, {}).get("art")
        if art is not None:
            asm.emit(PUSH, idx)            # char_id
            asm.emit(PUSH, int(art) & 0xFFFF)   # art base asset id
            asm.emit(SYS, (S_SET_ART << 8) | 2)
    # Token renames ($Antagonist -> a display name) via 1f 2e, so $tokens in the
    # dialogue resolve. The decoder captures these in runtime_text_vars; we emit
    # the first binding at scene start so the token has a valid value. (Renames
    # that change mid-scene -- e.g. per combat enemy -- keep their first value;
    # the exact per-encounter rebinding lives in the combat structure.)
    for token, name in (scene.get("_renames") or {}).items():
        asm.emit(PUSH, st.intern(token))
        asm.emit(PUSH, st.intern(name))
        asm.emit(SYS, (0x2E << 8) | 2)
    # Scene body.
    for node in scene.get("nodes", []):
        _gen_node(node, asm, st, cast)
    # Terminal: HALT. A scene compiled standalone has no caller to RET to, and a
    # `goto` node (if present) already transferred control before we get here.
    asm.emit(HALT)

    instrs = asm.resolve()
    code = _serialize_instructions(instrs)
    words = bytearray(st.word_bytes())
    # Init resource table: 8-byte rows [type, asset, name_ref, index] appended to
    # the word region (between the strings and the code). This is decoder-facing
    # metadata -- the engine gets art from the set_character_art_base yields above,
    # but our decoder reads this table to resolve each line's portrait `image`.
    # Terminated by a type-3 / asset-0xffff sentinel row.
    res_rows = []
    for idx, name in enumerate(cast or []):
        if not name and idx != 0:
            continue
        cfg = characters.get(name, {}) if name else {}
        art = cfg.get("art")
        # The decoder anchors the table scan on a first row with index 0, so always
        # emit an index-0 row (the player/lead slot) even when it has no art.
        if art is None and idx != 0:
            continue
        typ = {"male": 1, "female": 2, "object": 3}.get(cfg.get("gender"), 2)
        name_ref = st.intern(name) if name else 0
        asset = int(art) & 0xFFFF if art is not None else 0
        res_rows.append((typ, asset, name_ref, idx))
    if any(idx != 0 for _, _, _, idx in res_rows):
        words = bytearray(st.word_bytes())  # re-fetch (intern above may have grown it)
        for typ, asset, nref, idx in res_rows:
            words += struct.pack(">HHHH", typ, asset, nref, idx)
        words += struct.pack(">HHHH", 3, 0xFFFF, 0, 0)   # sentinel terminator
        if len(words) % 2:
            words.append(0)
    words = bytes(words)

    # KiWi v2 header: 'kiwi' + version(2) + previous_flag(0) + flag(0)
    #   + counts(main, gap, extra, instr) ; then word table ; then code.
    main_count = len(words) // 2
    header = bytearray()
    header += b"kiwi"
    header.append(2)                 # version
    header.append(0)                 # previous_flag
    header.append(0)                 # flag
    header += struct.pack(">H", main_count)   # main_words
    header += struct.pack(">H", 0)            # gap_count
    header += struct.pack(">H", 0)            # extra_words
    header += struct.pack(">H", len(instrs))  # instruction_count
    return bytes(header) + words + code


def _pack_exp(chunks, title, localized_titles=None, pack_id=0, episode_id=0):
    """Pack (id, payload) chunks into a CSPUD .exp container.

    Directory: 'CSPUD' + uint32be count + [uint16be id, uint32be offset]*count.
    Each chunk: uint32be comp_size, uint32be uncomp_size, uint32be flags, data.
    Stored raw (flags=0); the decoder reads raw and LZMA alike.
    """
    # Metadata chunk (id 1), in the engine's exact format (content.py metadata()):
    #   >HH  pack_id, episode_id     (the pack/season this episode belongs to and
    #                                 its number within that pack; 0,0 = standalone)
    #   then EXACTLY 5 localized titles, each  >H length  + UTF-8 bytes
    #   with no trailing bytes (the parser requires pos == len at the end).
    titles = list(localized_titles) if localized_titles else []
    if not titles:
        titles = [title] * 5
    titles = (titles + [title] * 5)[:5]            # pad/truncate to exactly 5
    meta = bytearray(struct.pack(">HH", pack_id & 0xFFFF, episode_id & 0xFFFF))
    for t in titles:
        tb = t.encode("utf-8")
        if len(tb) > 65535:
            raise ValueError("episode title too long")
        meta += struct.pack(">H", len(tb)) + tb
    entries = [(1, bytes(meta))] + list(chunks)

    directory_size = 5 + 4 + len(entries) * 6
    body = bytearray()
    offsets = []
    for eid, payload in entries:
        offsets.append(directory_size + len(body))
        body += struct.pack(">III", len(payload), len(payload), 0)  # raw store
        body += payload

    out = bytearray()
    out += b"CSPUD"
    out += struct.pack(">I", len(entries))
    for (eid, _), off in zip(entries, offsets):
        out += struct.pack(">H", eid)
        out += struct.pack(">I", off)
    out += body
    return bytes(out)


def _scene_id_from_seg(seg_name):
    """'s3' / 's3_04_after' -> scene number 3."""
    m = re.match(r"s(\d+)", seg_name or "")
    return int(m.group(1)) if m else None


def compile_episode(spec):
    """Compile an authoring spec (dict) into .exp bytes.

    Accepts two shapes:
      * Authoring form: {scenes: [{id, nodes:[...]}]}  -- nested bodies inline.
      * Decoder-output form: {segments: {seg_id: {nodes}}, scene_entries: {...}}
        -- the flat segment graph shs_decoder emits. It is flattened per scene
        back into nested nodes before compiling, so a decoded episode can be
        edited and re-encoded with the same schema.
    """
    cast = spec.get("cast", [])
    characters = dict(spec.get("characters", {}))

    if spec.get("segments") is not None and "scenes" not in spec:
        # Decoder-output form: group segments by scene, flatten each scene's graph.
        segments = spec["segments"]
        # Reconstruct character art bases from decoded dialogue `image` fields when
        # an explicit `characters` map wasn't supplied, so re-encoding a decoded
        # episode keeps speaker names and portraits. The base is the smallest
        # image id seen for a speaker (emotion frames are base+offset).
        if not characters:
            seen = {}
            for seg in segments.values():
                for n in seg.get("nodes", []):
                    if isinstance(n, dict) and n.get("type") == "dialogue" \
                            and n.get("speaker") and n.get("image"):
                        sp = n["speaker"]
                        img = int(n["image"])
                        seen[sp] = min(seen.get(sp, img), img)
            characters = {sp: {"art": base} for sp, base in seen.items()}
        scene_entries = spec.get("scene_entries") or {}
        # entry segment per scene number
        entries = {}
        for num, seg in scene_entries.items():
            entries[int(num)] = seg.replace(".json", "") if isinstance(seg, str) else seg
        # infer any scenes not in scene_entries from segment id prefixes
        scene_nums = sorted({_scene_id_from_seg(s) for s in segments
                             if _scene_id_from_seg(s) is not None})
        scenes = []
        for num in scene_nums:
            entry = entries.get(num) or ("s%d" % num)
            if entry not in segments:
                continue
            nodes = _flatten_segment_graph(segments, entry)
            # The flattener follows next/then/else/choice pointers but not
            # goto_scene/dispatch edges, so segments in this scene reached only via
            # section-dispatch (e.g. a day-hub) would be dropped. Append any such
            # not-yet-emitted segments of this scene so their content survives the
            # round-trip (order within them is preserved; cross-links become linear).
            emitted_txt = {id(n) for n in nodes}
            covered = _flatten_covered_segments(segments, entry)
            for sid in segments:
                if _scene_id_from_seg(sid) == num and sid not in covered:
                    for n in segments[sid].get("nodes", []):
                        if isinstance(n, dict) and n.get("type") not in ("next",):
                            nodes.append(n)
            scene_node = {"id": num, "nodes": nodes}
            # Attach token renames for this scene (first binding per token) from
            # runtime_text_vars so $tokens in dialogue resolve.
            rtv = ((spec.get("runtime_text_vars") or {}).get("scenes") or {}).get(str(num))
            if rtv:
                ren = {}
                for token, binds in rtv.items():
                    if isinstance(binds, list) and binds:
                        ren[token] = binds[0].get("name", "")
                if ren:
                    scene_node["_renames"] = ren
            scenes.append(scene_node)
    else:
        scenes = spec.get("scenes", [])

    if not scenes:
        raise ValueError("episode has no scenes")
    chunks = []
    for i, scene in enumerate(scenes):
        sid = scene.get("id", i + 1)
        chunk_id = SCENE_CHUNK_BASE + (int(sid) - 1)
        chunks.append((chunk_id, _build_scene_chunk(scene, cast, characters)))
    # Prefer the real display title. When re-encoding a decoded episode, the JSON
    # carries the metadata title in `episode` and the source filename in `title`,
    # so `episode` is the byte-faithful choice; fall back to `title` for
    # hand-authored specs that only set `title`.
    ep_title = spec.get("episode") or spec.get("title") or "Untitled"
    return _pack_exp(chunks, ep_title, spec.get("localized_titles"),
                     pack_id=int(spec.get("pack_id", 0)),
                     episode_id=int(spec.get("episode_id", 0)))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Compile a story spec JSON into a .exp")
    ap.add_argument("input", help="authoring spec .json")
    ap.add_argument("-o", "--output", required=True, help="output .exp path")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.input).read_text())
    data = compile_episode(spec)
    Path(args.output).write_bytes(data)
    if spec.get("scenes"):
        n_scenes = len(spec["scenes"])
    else:
        n_scenes = len({_scene_id_from_seg(s) for s in (spec.get("segments") or {})
                        if _scene_id_from_seg(s) is not None})
    print("[+] wrote %s (%d bytes, %d scenes)"
          % (args.output, len(data), n_scenes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
