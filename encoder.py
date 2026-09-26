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
        self.verbatim = set()  # instruction indexes reproduced from the original
                               # bytecode (lowered IR yields): exempt from the
                               # generated-code arity guard

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
        self.base = 0          # data address of buf[0]

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
        ref = self.base + pos // 2
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
        start_word = self.base + len(self.buf) // 2
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


_VE_BIN = {"==": 0x0A, "!=": 0x0B, ">": 0x0C, ">=": 0x0D, "<": 0x0E, "<=": 0x0F,
           "or": 0x11, "and": 0x12, "+": 0x50, "-": 0x51, "*": 0x52, "/": 0x53, "%": 0x54}


def _emit_var_expr(asm, e):
    """Story-level expression over variables: int | {"var": key} |
    {"op": o, "a": e, "b": e} | {"op": "neg"|"not", "a": e}."""
    if isinstance(e, bool):
        e = int(e)
    if isinstance(e, (int, str)):
        asm.emit(PUSH, int(str(e), 0) & 0xFFFF)
    elif "var" in e:
        asm.emit(PUSH, int(str(e["var"]), 0) & 0xFFFF)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)
    elif e.get("op") == "neg":
        _emit_var_expr(asm, e["a"])
        asm.emit(0x24)
    elif e.get("op") == "not":
        _emit_var_expr(asm, e["a"])
        asm.emit(0x13)
    else:
        _emit_var_expr(asm, e["a"])
        _emit_var_expr(asm, e["b"])
        asm.emit(_VE_BIN[e["op"]])


def _emit_text(node, asm, st, text):
    """Push the text reference a display yield shows: the interned text, or --
    when the node carries `format` {pattern, args} -- the engine's formatted
    string (yield 0: pattern + values, result = dynamic string ref)."""
    fm = node.get("format")
    if isinstance(fm, dict) and fm.get("pattern") is not None:
        args = fm.get("args") or []
        asm.emit(PUSH, st.intern(fm["pattern"]))
        for a in args:
            if isinstance(a, dict) and "text" in a:
                asm.emit(PUSH, st.intern(a["text"]))
            else:
                _emit_var_expr(asm, a)
        asm.emit(SYS, (0x00 << 8) | (1 + len(args)))
        asm.emit(0x21)
    else:
        asm.emit(PUSH, st.intern(text))


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
    if t == "_jump":
        # a link to another segment inside a graph-compiled scene
        if _GCTX is not None:
            _g_jump(_GCTX, node.get("to"))
        else:
            asm.emit(HALT)
        return

    if t == "narrate":
        # Narration (yield 0x41) takes 3 args: [character_id, text_ref, override].
        # The text is the SECOND arg; character 0 and override -1 mean "no speaker,
        # no expression override" (engine reads text at index 1).
        if node.get("character_id") is not None:
            asm.emit(PUSH, _int(node["character_id"]) & 0xFFFF)
        elif _NARRATOR[0] is not None:
            asm.emit(PUSH, _NARRATOR[0] & 0xFFFF)   # the scene's narrator (yield 75)
        else:
            asm.emit(PUSH0)
        _emit_text(node, asm, st, node["text"])   # text ref (read at index 1)
        asm.emit(PUSH, (_int(node["expression"]) if node.get("expression") is not None else -1) & 0xFFFF)
        asm.emit(SYS, (S_NARRATION << 8) | 3)

    elif t == "say":
        # Dialogue (yield 0x0d): [text_ref, character_id]; text at index 0.
        ref = st.intern(node["text"])
        speaker = node.get("speaker")
        spk = _cast_index(cast, speaker)
        if _int(node.get("via"), 0) == 15 and speaker:
            spk = None                          # decoded from a named-speaker line (yield 15)
        if spk is None and speaker:
            # A one-off named speaker not in the cast (e.g. "Voice") uses the
            # custom-speaker op 1f 0f, which carries the name as a string:
            # push name_ref; push text_ref; push pair; 1f 0f 04.
            asm.emit(PUSH, st.intern(speaker))
            _emit_text(node, asm, st, node["text"])
            asm.emit(PUSH, _int(node.get("character_id"), 0) & 0xFFFF)
            asm.emit(PUSH, (_int(node["expression"]) if node.get("expression") is not None else -1) & 0xFFFF)
            asm.emit(SYS, (0x0F << 8) | 4)
        else:
            # [mode?, text, character, expression?] exactly as decoded
            # (character_id / mode / expression), else text + cast index
            n_args = 2
            if node.get("mode") is not None:
                asm.emit(PUSH, _int(node["mode"]) & 0xFFFF)
                n_args += 1
            _emit_text(node, asm, st, node["text"])   # text ref
            if node.get("character_id") is not None:
                asm.emit(PUSH, _int(node["character_id"]) & 0xFFFF)
            elif spk is None:
                asm.emit(PUSH0)
            else:
                asm.emit(PUSH, spk)             # speaker (cast index)
            if node.get("expression") is not None:
                asm.emit(PUSH, _int(node["expression"]) & 0xFFFF)
                n_args += 1
            asm.emit(SYS, (S_DIALOGUE << 8) | n_args)

    elif t == "set":
        var = int(node["var"])
        asm.emit(PUSH, var)
        if node.get("expr") is not None:
            # value computed from other variables (decoder: var_set.expr)
            _emit_var_expr(asm, node["expr"])
        else:
            asm.emit(PUSH, int(_norm_field(node, "value", default=0)))
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
        variant = _int(node.get("variant"), 0)
        if node.get("style") == "custom" or _int(node.get("via"), 0) == 35:
            # yield 35 [character, variant, base, expression]: background (and
            # the panel's character / expression), as the reference engine
            # reads it
            asm.emit(PUSH, _int(node.get("character_id"), -1) & 0xFFFF)
            asm.emit(PUSH, variant & 0xFFFF)
            asm.emit(PUSH, bg & 0xFFFF)
            asm.emit(PUSH, _int(node.get("expression"), -1) & 0xFFFF)
            asm.emit(SYS, (0x23 << 8) | 4)
        else:
            # yield 11 [variant, base, -1]
            asm.emit(PUSH, variant & 0xFFFF)
            asm.emit(PUSH, bg & 0xFFFF)
            asm.emit(PUSH, 0xFFFF)              # -1 sentinel
            asm.emit(SYS, (S_BACKGROUND << 8) | 3)

    elif t == "music":
        if node.get("action") == "stop" or _norm_field(node, "track_id", "id") == "stop":
            asm.emit(PUSH, _int(node.get("fade_ms"), 0) & 0xFFFF)   # fade duration (1f 51)
            asm.emit(SYS, (0x51 << 8) | 1)
        else:
            mid = _int(_norm_field(node, "track_id", "id"), 0)
            asm.emit(PUSH, mid & 0xFFFF)
            asm.emit(PUSH1 if node.get("loop") else PUSH0)     # loop flag
            asm.emit(SYS, (S_MUSIC << 8) | 2)

    elif t == "set_expression":
        # yield 5 [character, expression]: the expression the character's later
        # lines show unless they override it
        asm.emit(PUSH, _int(node.get("character_id"), 0) & 0xFFFF)
        asm.emit(PUSH, _int(node.get("expression"), 0) & 0xFFFF)
        asm.emit(SYS, (0x05 << 8) | 2)

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

    elif t == "string_default":
        # yield 47, need(3): [key_ref, default_ref, ~slot]; stores the default
        # under the key when the key is unset (a $TOKEN substitution binding)
        asm.emit(PUSH, st.intern(str(node.get("key", ""))))
        asm.emit(PUSH, st.intern(str(node.get("default", ""))))
        asm.emit(PUSH, _int(node.get("slot"), -1) & 0xFFFF)
        asm.emit(SYS, (0x2F << 8) | 3)

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
        asm.emit(PUSH1 if _int(node.get("flag"), 0) else PUSH0)
        asm.emit(SYS, (S_GOTO << 8) | 2)

    elif t == "goto_scene":
        _gen_node({**node, "type": "goto"}, asm, st, cast)

    elif t == "title_card":
        # Title card via 1f 08 (episode title + optional subtitle).
        text = _norm_field(node, "text", "title", default="")
        sub = node.get("subtitle", "")
        asm.emit(PUSH, st.intern(text))
        asm.emit(PUSH, st.intern(sub))
        # yield 8 [title, subtitle, asset, flag]
        asm.emit(PUSH, _int(node.get("asset_id", node.get("style")), 0) & 0xFFFF)
        asm.emit(PUSH, _int(node.get("flag", node.get("param")), 0) & 0xFFFF)
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
        # On-screen status/HUD line. Reference engine: yield 0 is FORMAT_STRING
        # (args: pattern, values...), returning a dynamic-string reference; it
        # displays nothing. The original formats and then shows the result as
        # narration (yield 65 [character, text, expr]) or dialogue (yield 13):
        #   push <pattern> ; push <value> ; 1f 00 ; push_res ; ... ; 1f 41
        txt = _norm_field(node, "text", default="")
        if not txt:
            return
        _nid = _int(node["character_id"]) if node.get("character_id") is not None \
            else (_NARRATOR[0] if _NARRATOR[0] is not None else 0)
        _expr = _int(node["expression"]) if node.get("expression") is not None else -1

        def push_text():
            if isinstance(node.get("format"), dict):
                _emit_text(node, asm, st, txt)
                return
            vf = node.get("value_from")
            if isinstance(vf, dict) and vf.get("var") is not None:
                asm.emit(PUSH, st.intern(txt))  # pattern
                # the %d value: var, or (minus_from - var), optionally * multiply
                # (Making Some Dough's "Kim has %d days left." = 10 - var2001)
                if vf.get("minus_from") is not None:
                    asm.emit(PUSH, _int(vf["minus_from"]) & 0xFFFF)
                asm.emit(PUSH, _int(vf["var"]))
                asm.emit(SYS, (S_VAR_READ << 8) | 1)
                asm.emit(0x21)                  # the var's value
                if vf.get("minus_from") is not None:
                    asm.emit(SUB)               # minus_from - var
                if vf.get("multiply") is not None:
                    asm.emit(PUSH, _int(vf["multiply"]) & 0xFFFF)
                    asm.emit(MUL)
                asm.emit(SYS, (0x00 << 8) | 2)  # format -> dynamic string ref
                asm.emit(0x21)                  # push the formatted ref
            else:
                asm.emit(PUSH, st.intern(txt))
        if _int(node.get("via"), 65) == 13:
            n_args = 2
            if node.get("mode") is not None:
                asm.emit(PUSH, _int(node["mode"]) & 0xFFFF)
                n_args += 1
            push_text()
            asm.emit(PUSH, _nid & 0xFFFF)
            if node.get("expression") is not None:
                asm.emit(PUSH, _expr & 0xFFFF)
                n_args += 1
            asm.emit(SYS, (S_DIALOGUE << 8) | n_args)
        else:
            asm.emit(PUSH, _nid & 0xFFFF)       # narration character (the narrator)
            push_text()
            asm.emit(PUSH, _expr & 0xFFFF)      # expression override
            asm.emit(SYS, (S_NARRATION << 8) | 3)

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
                if isinstance(n.get("on_timeout"), str):
                    dq.append(n["on_timeout"])
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
                if isinstance(n.get("on_timeout"), str):
                    out.append(n["on_timeout"])
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

    def _arm_body(v, sub_stop):
        """Body for a gate arm that may be a segment name, an inline node list, or
        a chained gate dict (nest_gates output)."""
        if isinstance(v, str):
            return inline(norm_target(v), sub_stop)[0]
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, list):
            return [(_inline_gate(x, sub_stop) if x.get("type") == "gate" else x)
                    for x in v if isinstance(x, dict)]
        return []

    def _inline_gate(n, sub_stop):
        return {"type": "gate", "var": n.get("var"), "op": n.get("op", "eq"),
                "value": _int(n.get("equals", n.get("value", 0))),
                "then": _arm_body(n.get("then"), sub_stop),
                "else": _arm_body(n.get("else"), sub_stop)}

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
                    t_raw, e_raw = n.get("then"), n.get("else")
                    if not isinstance(t_raw, str) and not isinstance(e_raw, str):
                        # INLINE gate (nested var-gate chain): both arms carry
                        # their nodes inline (a list, or a chained gate dict) and
                        # the gate sits MID-segment -- the segment's following
                        # nodes are the shared continuation, so keep reading them
                        # rather than stopping at the gate.
                        out.append(_inline_gate(n, stop_at))
                        continue
                    then_id = norm_target(t_raw) if isinstance(t_raw, str) else None
                    else_id = norm_target(e_raw) if isinstance(e_raw, str) else None
                    merge = _find_merge(then_id, else_id, succ, refs)
                    sub_stop = stop_at | ({merge} if merge else set())
                    then_body = _arm_body(t_raw, sub_stop)
                    else_body = _arm_body(e_raw, sub_stop)
                    gnode = {"type": "gate", "var": n.get("var"),
                             "op": n.get("op", "eq"),
                             "value": int(n.get("equals", n.get("value", 0))),
                             "then": then_body, "else": else_body}
                    out.append(gnode)
                    nxt = merge
                    break
                elif nt == "choice":
                    opt_ids = [norm_target(o.get("next")) for o in n.get("options", [])]
                    tmo_id = norm_target(n.get("on_timeout")) \
                        if isinstance(n.get("on_timeout"), str) else None
                    after_id = norm_target(n.get("after")) \
                        if isinstance(n.get("after"), str) else None
                    if not any(opt_ids) and not tmo_id and not after_id:
                        # LINEAR choice: no option leads anywhere of its own -- every
                        # pick continues with the segment's following nodes. Emit the
                        # menu and keep reading instead of ending the segment here.
                        _lin = {"type": "choice",
                                "prompt": n.get("prompt", "Make your choice!"),
                                "options": [{"label": o.get("label", ""), "body": []}
                                            for o in n.get("options", [])]}
                        for _tk in ("timeout_ms", "timer_ms", "timeout_result", "text", "menu"):
                            if n.get(_tk) is not None:
                                _lin[_tk] = n[_tk]
                        out.append(_lin)
                        continue
                    # The decoder records each choice's convergence point in
                    # `after`; trust it rather than re-deriving the merge from the
                    # graph (inference can miss it, dropping the continuation).
                    merge = after_id or _find_merge_many(
                        opt_ids + ([tmo_id] if tmo_id else []), succ, refs)
                    sub_stop = stop_at | ({merge} if merge else set())
                    cnode = {"type": "choice",
                             "prompt": n.get("prompt", "Make your choice!"),
                             "options": []}
                    for _tk in ("timeout_ms", "timer_ms", "timeout_result", "text", "menu"):
                        if n.get(_tk) is not None:
                            cnode[_tk] = n[_tk]
                    if tmo_id:
                        cnode["timeout_body"], _ = inline(tmo_id, sub_stop)
                    for o in n.get("options", []):
                        oid = norm_target(o.get("next"))
                        body, _ = inline(oid, sub_stop) if oid else ([], None)
                        cnode["options"].append({"label": o.get("label", ""),
                                                 "body": body})
                    if cnode.get("timeout_body"):
                        # The original can ALSO route the timeout through an
                        # option's path via a `score == 1000` gate (the explicit
                        # form). With the timeout body now in the fall-through,
                        # that gate is unreachable inside an `answer == i` branch;
                        # keep only its real (else) content so it doesn't linger as
                        # dead code (and mislead re-decoding).
                        for _o in cnode["options"]:
                            _b = _o.get("body") or []
                            if _b and isinstance(_b[0], dict) \
                                    and _b[0].get("type") == "gate" \
                                    and str(_b[0].get("var")) == "score" \
                                    and str(_b[0].get("equals", _b[0].get("value"))) == "1000":
                                _else = _b[0].get("else")
                                _o["body"] = (list(_else) if isinstance(_else, list)
                                              else []) + _b[1:]
                    out.append(cnode)
                    nxt = merge
                    break
                elif nt == "minigame" and (isinstance(n.get("win"), str)
                                           or isinstance(n.get("lose"), str)):
                    # A minigame BRANCHES on its result (win / lose): inline both
                    # outcomes like a gate's arms instead of treating the node as
                    # plain content (which left everything behind it unreached).
                    w_id = norm_target(n["win"]) if isinstance(n.get("win"), str) else None
                    l_id = norm_target(n["lose"]) if isinstance(n.get("lose"), str) else None
                    merge = _find_merge(w_id, l_id, succ, refs) if (w_id and l_id) else None
                    sub_stop = stop_at | ({merge} if merge else set())
                    m = dict(n)
                    m["win"] = inline(w_id, sub_stop)[0] if w_id else (n.get("win") or [])
                    m["lose"] = inline(l_id, sub_stop)[0] if l_id else (n.get("lose") or [])
                    out.append(m)
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
                elif isinstance(n.get(k), dict):
                    _emitted_text_ids([n[k]], acc)
            for o in n.get("options", []) or []:
                if isinstance(o, dict) and isinstance(o.get("body"), list):
                    _emitted_text_ids(o["body"], acc)
            if isinstance(n.get("timeout_body"), list):
                _emitted_text_ids(n["timeout_body"], acc)
        return acc

    emitted_text = _emitted_text_ids(body, set())
    covered = _flatten_covered_segments(segments, entry)
    for sid in covered:
        seg = segments.get(sid, {})
        # does this segment have text not present anywhere in the output?
        seg_text = {n["text"] for n in seg.get("nodes", [])
                    if isinstance(n, dict)
                    and n.get("type") in ("dialogue", "narration") and n.get("text")}
        if sid in emitted and (not seg_text or seg_text <= emitted_text):
            continue
        if sid not in emitted and not seg_text and not any(
                isinstance(x, dict) and x.get("type") not in ("next", "end")
                for x in seg.get("nodes", [])):
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


def _gen_outcome_fork(node, asm, st, cast):
    """A minigame whose game can't be regenerated (its word bank lives in a data
    block we don't rebuild): keep the ORIGINAL branch structure -- show the prompt,
    then branch on the minigame result slot (score >= win_threshold -> win, else
    lose) -- so both outcomes stay reachable instead of silently keeping only the
    win path."""
    prompt = node.get("prompt") or node.get("heading")
    if prompt:
        _gen_node({"type": "narrate", "text": prompt}, asm, st, cast)
    thr = _int(node.get("win_threshold"), 1)
    lose_lbl = asm.new_label("oflose")
    end_lbl = asm.new_label("ofend")
    asm.emit(0x5F)                              # result slot @fp0
    asm.emit(0x3F)                              # load
    asm.emit(PUSH, thr & 0xFFFF)
    asm.emit(CMP_GE)
    asm.emit(JMPF, target_label=lose_lbl)
    for n in (node.get("win") or []):
        if isinstance(n, dict):
            _gen_node(n, asm, st, cast)
    asm.emit(JMP, target_label=end_lbl)
    asm.label(lose_lbl)
    for n in (node.get("lose") or []):
        if isinstance(n, dict):
            _gen_node(n, asm, st, cast)
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
        _gen_outcome_fork(node, asm, st, cast)
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
        _gen_outcome_fork(node, asm, st, cast)
        return
    title = node.get("title", "Make your choice!")
    prompt = node.get("prompt", "")
    duration = int(node.get("duration_ms", 20000))
    if duration <= 0:
        duration = 20000
    threshold = int(node.get("win_threshold", len(correct)))
    timing2 = int(node.get("timing2", 3000))
    spk = _cast_index(cast, node.get("speaker"))
    portrait_mode = int(node.get("portrait_mode", 1))
    if node.get("character_id") is not None:
        char_id = int(node["character_id"])
    else:
        char_id = spk if spk is not None else 13

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
    asm.emit(PUSH, char_id & 0xFFFF)            # args[5] character id
    asm.emit(PUSH, portrait_mode & 0xFFFF)      # args[6] portrait mode
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
        _gen_outcome_fork(node, asm, st, cast)
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


# --------------------------------------------------------------------------- #
# Graph compiler (decoder-format specs)
# --------------------------------------------------------------------------- #
# A decoded episode is a GRAPH of segments. Compiling it by first folding it into
# a nested tree loses anything a tree can't express (sections reached by in-scene
# dispatch, shared continuations) -- those end up present but unreachable. So a
# decoder-format scene is compiled AS a graph: every segment becomes one labeled
# code block, emitted exactly once, and every link (next, gate arms, choice
# options / after / on_timeout, minigame win/lose, dispatch & random arms, combat
# fights) is a jump to that label. A target not yet emitted is placed inline at
# its first reference -- reproducing the original layout (option bodies right
# after their test, a timeout body in the fall-through) -- otherwise it is jumped
# to. Cross-scene links set the section register and goto the other scene.

import sys as _sys
_sys.setrecursionlimit(max(_sys.getrecursionlimit(), 20000))

_GCTX = None          # active graph context while compiling a graph scene


def _seg_links(n):
    """Every segment name a node links to (strings only; recurses into inline
    gate arms)."""
    out = []
    if not isinstance(n, dict):
        return out
    t = n.get("type")
    for k in ("next", "then", "else", "after", "on_timeout", "win", "lose",
              "default", "exhausted"):
        v = n.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            out += _seg_links(v)
        elif isinstance(v, list):
            for x in v:
                out += _seg_links(x)
    for o in n.get("options", []) or []:
        if isinstance(o, dict):
            for k in ("next", "scene", "fight"):
                if isinstance(o.get(k), str):
                    out.append(o[k])
    for a in n.get("arms", []) or []:
        if isinstance(a, dict) and isinstance(a.get("scene"), str):
            out.append(a["scene"])
    if t == "goto_scene" and isinstance(n.get("file"), str):
        out.append(n["file"])
    return [x.replace(".json", "") for x in out]


class _GraphCtx:
    def __init__(self, segments, members, scene_of, asm, st, cast):
        self.segments, self.members, self.scene_of = segments, members, scene_of
        self.asm, self.st, self.cast = asm, st, cast
        self.labels, self.emitted, self.pending = {}, set(), []
        self.routines = _ROUTINES
        self.routines_emitted = set()

    def label(self, sid):
        if sid not in self.labels:
            self.labels[sid] = "seg:%s:%d" % (sid, len(self.labels))
        return self.labels[sid]


def _g_goto_other(ctx, target, register=None):
    """Leave this scene for the scene that owns `target`."""
    num = ctx.scene_of.get(target) or _seg_scene_num(target)
    if register and register.get("var") is not None:
        ctx.asm.emit(PUSH, _int(register["var"]) & 0xFFFF)
        ctx.asm.emit(PUSH, _int(register.get("value")) & 0xFFFF)
        ctx.asm.emit(SYS, (S_VAR_WRITE << 8) | 2)
    if num:
        _emit_goto_scene_num(ctx.asm, num)
    ctx.asm.emit(HALT)


def _g_jump(ctx, target, allow_inline=True):
    t = target.replace(".json", "") if isinstance(target, str) else None
    if not t or t not in ctx.segments:
        ctx.asm.emit(HALT)
        return
    if t in ctx.members:
        if t not in ctx.emitted and allow_inline:
            _g_emit_segment(ctx, t)
        else:
            if t not in ctx.emitted:
                ctx.pending.append(t)
            ctx.asm.emit(JMP, target_label=ctx.label(t))
    else:
        _g_goto_other(ctx, t)


def _g_arm(v):
    """A gate/minigame arm as an inline node list: a segment name becomes a
    `_jump`; inline lists / chained gate dicts are kept (recursively)."""
    if isinstance(v, str):
        return [{"type": "_jump", "to": v}]
    if isinstance(v, dict):
        v = [v]
    if isinstance(v, list):
        return [_g_conv(x) for x in v if isinstance(x, dict)]
    return []


def _g_conv(n):
    if n.get("type") == "gate":
        g = dict(n)
        g["then"], g["else"] = _g_arm(n.get("then")), _g_arm(n.get("else"))
        return g
    return n


def _g_dispatch_to(ctx, pairs, default):
    """Register A already holds the selector. `pairs` = [(match, target)]. Emit
    the op5c cascade to local stubs (never inlined, so the 8-bit dispatch
    distances stay in range), the default, then the stubs."""
    asm = ctx.asm
    stubs = []
    for match, tgt in pairs:
        lbl = asm.new_label("gdarm")
        stubs.append((lbl, tgt))
        asm.emit_dispatch(int(match) & 0xFF, lbl)
    if isinstance(default, str):
        _g_jump(ctx, default, allow_inline=False)
    else:
        asm.emit(HALT)
    for lbl, tgt in stubs:
        asm.label(lbl)
        _g_jump(ctx, tgt, allow_inline=False)


# Reserved variable tracking call depth for decoder `call` / `return` nodes
# (outside the ranges any shipped episode uses).
CALL_DEPTH_VAR = 3999



# --------------------------------------------------------------------------- #
# Lowered routines (decoder `routines` + IR nodes: let / yield / if / call with
# args / ret). Frame layout matches the KiWi VM: a caller pushes the arguments
# and PUSH_RET; the routine runs `0x48 0x4a`, reserves its locals, and returns
# with `0x22 0x43`; locals are frame cells fp+k (k < 0: arguments).
# --------------------------------------------------------------------------- #
_ROUTINES = {}
_NARRATOR = [None]      # the scene's narrator id (setup ui_defaults 75), per scene
_IR_BIN = {"==": CMP_EQ, "!=": CMP_NE, ">": CMP_GT, ">=": CMP_GE, "<": CMP_LT,
           "<=": CMP_LE, "or": 0x11, "and": 0x12, "+": ADD, "-": SUB, "*": MUL,
           "/": 0x53, "%": 0x54}


def _ir_int(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    return int(str(v), 0)


def _g_addr(asm, k):
    k = _ir_int(k)
    if 0 <= k <= 3:
        asm.emit(0x5F + k)
    else:
        asm.emit(0x23, k & 0xFFFF)


def _g_expr(ctx, e):
    asm = ctx.asm
    if isinstance(e, (int, str)) and not isinstance(e, bool) or isinstance(e, bool):
        v = _ir_int(e)
        if v in (0, 1):
            asm.emit(0x5A + v)
        else:
            asm.emit(PUSH, v & 0xFFFF)
        return
    if "text" in e:
        asm.emit(PUSH, ctx.st.intern(e["text"]))
    elif "local" in e:
        asm.emit(0x41, _ir_int(e["local"]) & 0xFFFF)
    elif "addr" in e:
        _g_addr(asm, e["addr"])
    elif "reg" in e:
        asm.emit(0x07 if e["reg"] == "A" else 0x08)
    elif "deref" in e:
        _g_expr(ctx, e["deref"])
        asm.emit(0x3F)
    elif "table" in e:
        # a constant word array: laid out in the scene's data, its address pushed
        asm.emit(PUSH, _g_table(ctx, e["table"]))
    elif "scratch_addr" in e:
        _g_scratch_addr(ctx, e["scratch_addr"])
    elif "scratch" in e:
        _g_scratch_addr(ctx, e["scratch"])
        asm.emit(0x3F)
    elif "mem" in e:
        asm.emit(0x40, _ir_int(e["mem"]) & 0xFFFF)
    elif "op" in e:
        o = e["op"]
        _g_expr(ctx, e["a"])
        if o == "neg":
            asm.emit(0x24)
        elif o == "not":
            asm.emit(0x13)
        else:
            _g_expr(ctx, e["b"])
            asm.emit(_IR_BIN[o])
    else:
        raise ValueError("unsupported IR expression %r" % (e,))


def _g_table(ctx, entries):
    words = tuple(ctx.st.intern(x["text"]) if isinstance(x, dict) and "text" in x
                  else _ir_int(x) & 0xFFFF for x in entries)
    cache = ctx.asm.__dict__.setdefault("tables", {})
    if words not in cache:
        cache[words] = ctx.st.append_words(list(words))
    return cache[words]


def _g_scratch_addr(ctx, i):
    """VM scratch cell i lives in the scene's uninitialised data words (the
    gap after the string table); its address is bound when the table is final"""
    asm = ctx.asm
    if isinstance(i, int) and not isinstance(i, bool):
        asm.scratch = max(getattr(asm, "scratch", 0), i + 1)
        asm.emit(PUSH, ("scratch", i))
        return
    asm.scratch = max(getattr(asm, "scratch", 0), 1)
    asm.emit(PUSH, ("scratch", 0))
    _g_expr(ctx, i)
    asm.emit(ADD)


def _g_drop(asm, n):
    if n <= 0:
        return
    if n <= 4:
        asm.emit(0x14 + n)
    else:
        asm.emit(0x19, n)


def _g_ir_node(ctx, n):
    """IR nodes of lowered routines. Returns (handled, terminal)."""
    asm, t = ctx.asm, n.get("type")
    if t == "let":
        to = n["to"]
        if "reg" in to:
            _g_expr(ctx, n["value"])
            asm.emit(0x03 if to["reg"] == "A" else 0x04)
            return True, False
        if "local" in to:
            _g_addr(asm, to["local"])
        elif "scratch" in to:
            _g_scratch_addr(ctx, to["scratch"])
        else:
            _g_expr(ctx, to["deref"])
        _g_expr(ctx, n["value"])
        asm.emit(0x3E)
        asm.emit(0x15)
        return True, False
    if t == "yield":
        args = n.get("args") or []
        if n.get("to") is not None:
            _g_addr(asm, n["to"])
        for a in args:
            _g_expr(ctx, a)
        asm.verbatim.add(len(asm.ins))
        asm.emit(SYS, (_ir_int(n["yield"]) << 8) | len(args))
        if n.get("to") is not None:
            asm.emit(0x21)
            asm.emit(0x3E)
            asm.emit(0x15)
        return True, False
    if t == "if":
        els = n["else"]
        if els not in ctx.emitted:
            ctx.pending.append(els)
        _g_expr(ctx, n["cond"])
        asm.emit(JMPF, target_label=ctx.label(els))
        _g_jump(ctx, n["then"])
        return True, True
    if t == "ret":
        asm.emit(POP_FRAME)
        asm.emit(RET)
        return True, True
    if t == "pause":
        asm.emit(0x32)
        return True, False
    if t == "call" and n.get("routine"):
        name = n["routine"]
        r = ctx.routines.get(name)
        if r is None:
            return True, False
        args = n.get("args") or []
        for a in args:
            _g_expr(ctx, a)
        asm.emit(PUSH_RET)
        if name not in ctx.routines_emitted:
            ctx.routines_emitted.add(name)
            ctx.pending.append(("__routine__", name))
        asm.emit(JMP, target_label="routine:" + name)
        _g_drop(asm, len(args))
        if n.get("result_var") is not None:
            asm.emit(PUSH, _ir_int(n["result_var"]) & 0xFFFF)
            asm.emit(0x08)
            asm.emit(SYS, (S_VAR_WRITE << 8) | 2)
        return True, False
    return False, False


def _routine_frame(ctx, entry):
    """frame cells a routine uses (the highest local / address / yield result
    index in its IR segments, +1)"""
    seen, work, mx = set(), [entry], -1

    def w(e):
        nonlocal mx
        if isinstance(e, dict):
            for k in ("local", "addr"):
                if isinstance(e.get(k), int) and not isinstance(e.get(k), bool):
                    mx = max(mx, e[k])
            for v in e.values():
                w(v)
        elif isinstance(e, list):
            for v in e:
                w(v)
    while work:
        sid = work.pop()
        if sid in seen or sid not in ctx.segments:
            continue
        seen.add(sid)
        for n in ctx.segments[sid].get("nodes") or []:
            if n.get("type") in ("let", "yield", "if", "call", "ret", "pause", "next", "end"):
                w({k: v for k, v in n.items() if k in ("to", "value", "args", "cond")})
                if n.get("type") == "yield" and isinstance(n.get("to"), int):
                    mx = max(mx, n["to"])
            for k in ("next", "then", "else"):
                if isinstance(n.get(k), str):
                    work.append(n[k])
    return mx + 1


def _g_emit_routine(ctx, name):
    r = ctx.routines[name]
    asm = ctx.asm
    asm.label("routine:" + name)
    asm.emit(PUSH_FP)
    asm.emit(SET_FP)
    for _ in range(_routine_frame(ctx, r["entry"])):
        asm.emit(0x5A)
    _g_jump(ctx, r["entry"], allow_inline=True)


def _g_node(ctx, n):
    """Emit one node of a graph scene. Returns True if control never falls
    through past it."""
    asm, st, cast = ctx.asm, ctx.st, ctx.cast
    t = n.get("type")
    _h, _term = _g_ir_node(ctx, n)
    if _h:
        return _term
    if t == "call":
        # Subroutine call: PUSH_RET pushes its own pc+1 (the jump right after it)
        # and RET resumes one past that, exactly the original `0x42 ; 0x29` pair.
        # The callee is its own labelled segment (never inlined).
        tgt = n.get("segment")
        tgt = tgt.replace(".json", "") if isinstance(tgt, str) else None
        if tgt and tgt in ctx.members:
            _gen_node({"type": "var_add", "var": CALL_DEPTH_VAR, "value": 1}, asm, st, cast)
            asm.emit(PUSH_RET)
            if tgt not in ctx.emitted and tgt not in ctx.pending:
                ctx.pending.append(tgt)
            asm.emit(JMP, target_label=ctx.label(tgt))
        return False
    if t == "return":
        # A `return` reached with no active call ends the scene (decoder contract),
        # so guard the RET with the call-depth counter instead of underflowing.
        ok = asm.new_label("gret")
        asm.emit(PUSH, CALL_DEPTH_VAR)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)
        asm.emit(PUSH, 0)
        asm.emit(CMP_EQ)
        asm.emit(JMPF, target_label=ok)
        if isinstance(n.get("top_level_next"), str):
            # no active call: the scene's own return, which (decoder contract)
            # goes back to the host schedule loop at `top_level_next`
            _g_jump(ctx, n["top_level_next"], allow_inline=False)
        else:
            asm.emit(HALT)
        asm.label(ok)
        _gen_node({"type": "var_add", "var": CALL_DEPTH_VAR, "value": -1}, asm, st, cast)
        asm.emit(RET)
        return True
    if t == "next":
        _g_jump(ctx, n.get("next"))
        return True
    if t == "end":
        asm.emit(HALT)
        return True
    if t == "checkpoint_replay":
        if isinstance(n.get("next"), str):
            _g_jump(ctx, n["next"])
            return True
        return False
    if t == "goto_scene":
        f = n.get("file") if isinstance(n.get("file"), str) else None
        f = f.replace(".json", "") if f else None
        if f and f in ctx.members:
            _g_jump(ctx, f)
        elif f:
            _g_goto_other(ctx, f, n.get("section_register"))
        else:
            _gen_node(n, asm, st, cast)
            asm.emit(HALT)
        return True
    if t == "gate":
        both_links = isinstance(n.get("then"), str) and isinstance(n.get("else"), str)
        _gen_node(_g_conv(n), asm, st, cast)
        return both_links
    if t == "choice":
        c = {"type": "choice", "prompt": n.get("prompt", "Make your choice!"),
             "options": []}
        for k in ("timeout_ms", "timer_ms", "timeout_result", "text", "menu"):
            if n.get(k) is not None:
                c[k] = n[k]
        for o in n.get("options", []):
            nx = o.get("next") if isinstance(o, dict) else None
            c["options"].append({"label": (o or {}).get("label", ""),
                                 "body": [{"type": "_jump", "to": nx}]
                                 if isinstance(nx, str) else [],
                                 **({"when": o["when"]} if isinstance(o, dict)
                                    and o.get("when") else {})})
        if isinstance(n.get("on_timeout"), str):
            c["timeout_body"] = [{"type": "_jump", "to": n["on_timeout"]}]
        _gen_node(c, asm, st, cast)
        return False
    if t == "minigame":
        m = dict(n)
        for k in ("win", "lose"):
            if isinstance(n.get(k), str):
                m[k] = [{"type": "_jump", "to": n[k]}]
                if n[k].replace(".json", "") in ctx.members:
                    ctx.pending.append(n[k].replace(".json", ""))
        _gen_node(m, asm, st, cast)
        return isinstance(n.get("win"), str)
    if t == "dispatch":
        var = _int(n.get("state_var"), 0)
        asm.emit(PUSH, var)
        asm.emit(SYS, (S_VAR_READ << 8) | 1)
        asm.emit(0x21)
        asm.emit(0x03)                          # A = register
        _g_dispatch_to(ctx, [(_int(a.get("equals")), a.get("scene"))
                             for a in n.get("arms", []) or []
                             if isinstance(a, dict) and isinstance(a.get("scene"), str)],
                       n.get("default"))
        return True
    if t in ("random", "random_encounter"):
        key = "scene" if t == "random" else "fight"
        opts = [o for o in n.get("options", []) or []
                if isinstance(o, dict) and isinstance(o.get(key), str)]
        if not opts:
            _gen_node(n, asm, st, cast)
            return False
        if t == "random_encounter" and n.get("time_var") is not None:
            _gen_node({"type": "var_random", "var": _int(n["time_var"]),
                       "min": 0, "max": _int(n.get("roll_max") or n.get("time_total"))},
                      asm, st, cast)
        asm.emit(PUSH, len(opts))
        asm.emit(SYS, (0x1B << 8) | 1)          # RANDOM in [0, n)
        asm.emit(0x21)
        asm.emit(0x03)                          # A = roll
        _g_dispatch_to(ctx, [(i, o[key]) for i, o in enumerate(opts)],
                       n.get("exhausted"))
        return True
    _gen_node(n, asm, st, cast)
    return False


def _g_emit_segment(ctx, sid):
    ctx.emitted.add(sid)
    ctx.asm.label(ctx.label(sid))
    nodes = [n for n in ctx.segments[sid].get("nodes", []) if isinstance(n, dict)]
    done = False
    for i, n in enumerate(nodes):
        done = _g_node(ctx, n)
        if done:
            break
    if not done:
        # A choice that ends the segment continues at its convergence point.
        last = nodes[-1] if nodes else {}
        if last.get("type") == "choice" and isinstance(last.get("after"), str):
            _g_jump(ctx, last["after"])
        else:
            ctx.asm.emit(HALT)


def _emit_graph_scene(graph, asm, st, cast):
    """Compile one scene of a decoder-format spec as a segment graph."""
    global _GCTX
    segments, entry, members, scene_of = graph[:4]
    section_entries = graph[4] if len(graph) > 4 else []
    ctx = _GraphCtx(segments, members, scene_of, asm, st, cast)
    prev, _GCTX = _GCTX, ctx
    try:
        # Section entry. Other scenes enter this one mid-way by setting a section
        # register and going to this scene (goto_scene file=<section> with
        # section_register); the original scene prologue reads that register and
        # dispatches to the section. Reproduce it: read each register, dispatch
        # value -> section, and fall through to the normal entry otherwise.
        by_var = {}
        for var, value, target in section_entries:
            by_var.setdefault(var, [])
            if (value, target) not in by_var[var]:
                by_var[var].append((value, target))
        if by_var:
            start = asm.new_label("gentry")
            for var, pairs in sorted(by_var.items()):
                asm.emit(PUSH, var & 0xFFFF)
                asm.emit(SYS, (S_VAR_READ << 8) | 1)
                asm.emit(0x21)
                asm.emit(0x03)                  # A = section register
                stubs = []
                for value, target in pairs:
                    lbl = asm.new_label("gsec")
                    stubs.append((lbl, target))
                    asm.emit_dispatch(value & 0xFF, lbl)
                nxt = asm.new_label("gsecn")
                asm.emit(JMP, target_label=nxt)
                for lbl, target in stubs:
                    asm.label(lbl)
                    _g_jump(ctx, target, allow_inline=False)
                asm.label(nxt)
            asm.label(start)
        _g_emit_segment(ctx, entry)

        def drain():
            while ctx.pending:
                p = ctx.pending.pop(0)
                if isinstance(p, tuple):
                    _g_emit_routine(ctx, p[1])
                elif p not in ctx.emitted:
                    _g_emit_segment(ctx, p)
        drain()
        # Members nothing links to (dead in the original too) keep their content.
        for s in sorted(members):
            if s not in ctx.emitted:
                _g_emit_segment(ctx, s)
                drain()
    finally:
        _GCTX = prev


def _assign_scenes(segments, entries):
    """Map segment -> scene number: BFS from each scene entry over in-scene links
    (never through another scene's entry, never along a cross-scene goto)."""
    entry_set = set(entries.values())
    scene_of = {}
    for num in sorted(entries):
        e = entries[num]
        if e not in segments:
            continue
        queue = [e]
        while queue:
            s = queue.pop(0)
            if s in scene_of or s not in segments:
                continue
            if s in entry_set and s != e:
                continue
            scene_of[s] = num
            for n in segments[s].get("nodes", []):
                if isinstance(n, dict) and n.get("type") == "goto_scene":
                    continue
                queue += [x for x in _seg_links(n) if x not in scene_of]
    for s in segments:
        if s not in scene_of:
            num = _seg_scene_num(s)
            scene_of[s] = num if num in entries else min(entries)
    return scene_of


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
    # A timed choice (authored `timeout_ms`, or decoded `timed`/`timer_ms`) counts
    # down; in the original bytecode its title slot is always "Hurry!" with the
    # question moved to the text slot -- the marker the decoder keys on to
    # recognise a timed choice -- so reproduce that layout.
    timeout_ms = _int(node.get("timeout_ms") or node.get("timer_ms"), 0)
    menu = node.get("menu") if isinstance(node.get("menu"), dict) else None
    if menu:
        # the decoded menu's exact arguments
        prompt = menu.get("title") or ""
        lead_in = menu.get("text") or ""
        timeout_ms = _int(menu.get("timeout_ms"), -1)
    elif timeout_ms > 0:
        if not lead_in:
            lead_in = prompt
        prompt = "Hurry!"
    if any(isinstance(o, dict) and o.get("when") for o in options):
        # CONDITIONAL options: the engine's builder menu (yields 2/3/4), each
        # option added only when its `when` conditions all hold -- as the
        # original New Girl class / phone menus do. The chosen option's VALUE
        # (its position here) comes back as the result, like a plain choice.
        asm.emit(PUSH, st.intern(prompt))      # args[0] title
        asm.emit(PUSH, st.intern(lead_in) if lead_in else 0xFFFF)  # args[1] text
        if menu:
            asm.emit(PUSH, _int(menu.get("timeout_ms"), -1) & 0xFFFF)      # args[2] timeout
            asm.emit(PUSH, _int(menu.get("timeout_result"), -1) & 0xFFFF)  # args[3] default
            asm.emit(PUSH, _int(menu.get("character_id"), -2) & 0xFFFF)    # args[4] character
            asm.emit(PUSH, _int(menu.get("portrait_mode"), 1) & 0xFFFF)    # args[5] portrait
        else:
            asm.emit(PAIR, 0xFFFF)             # args[2]=-1 timeout, args[3]=-1
            asm.emit(PUSH, 0xFFFE)             # args[4] character = -2 (default)
            asm.emit(PUSH1)                    # args[5] portrait_mode = 1
        asm.emit(SYS, (0x02 << 8) | 6)
        for i, opt in enumerate(options):
            skip = asm.new_label("bskip")
            conds = opt.get("when") or []
            for k, c in enumerate(conds):
                asm.emit(PUSH, _int(c.get("var"), 0))
                asm.emit(SYS, (S_VAR_READ << 8) | 1)
                asm.emit(0x21)
                asm.emit(PUSH, _int(c.get("value"), 0) & 0xFFFF)
                asm.emit(_CMP[c.get("op") or "eq"])
                if k:
                    asm.emit(0x12)             # AND
            if conds:
                asm.emit(JMPF, target_label=skip)
            asm.emit(PUSH, st.intern(opt["label"]))
            asm.emit(PUSH, i)                  # value
            asm.emit(PUSH1)                    # enabled
            asm.emit(SYS, (0x03 << 8) | 3)
            asm.label(skip)
        asm.emit(PUSH0)                        # displayed in order
        asm.emit(SYS, (0x04 << 8) | 1)
    else:
        _gen_plain_choice_call(node, asm, st, prompt, opt_str, lead_in, timeout_ms)
    _gen_choice_routing(node, options, asm, st, cast)


def _gen_plain_choice_call(node, asm, st, prompt, opt_str, lead_in, timeout_ms):
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
    menu = node.get("menu") if isinstance(node.get("menu"), dict) else None
    if menu:
        asm.emit(PUSH, _int(menu.get("timeout_ms"), -1) & 0xFFFF)          # args[3]
        asm.emit(PUSH, _int(menu.get("timeout_result"), -1) & 0xFFFF)      # args[4]
        asm.emit(PUSH, _int(menu.get("character_id"), -2) & 0xFFFF)        # args[5]
        asm.emit(PUSH, _int(menu.get("reserved"), -1) & 0xFFFF)            # args[6]
        asm.emit(PUSH, _int(menu.get("portrait_mode"), 1) & 0xFFFF)        # args[7]
    elif timeout_ms > 0:
        timeout_result = _int(node.get("timeout_result"), 1000)
        asm.emit(PUSH, timeout_ms & 0xFFFF)    # args[3] timeout
        asm.emit(PUSH, timeout_result & 0xFFFF)  # args[4] result
        asm.emit(PAIR, 0xFFFE)                 # args[5]=-2 character, args[6]=-1
        asm.emit(PUSH1)                        # args[7] portrait_mode = 1
    else:
        asm.emit(PAIR, 0xFFFF)                 # args[3]=-1 timeout, args[4]=-1 result
        asm.emit(PAIR, 0xFFFE)                 # args[5]=-2 character, args[6]=-1
        asm.emit(PUSH1)                        # args[7] portrait_mode = 1
    asm.emit(SYS, (S_CHOICE << 8) | 8)


def _gen_choice_routing(node, options, asm, st, cast):
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
    # Timeout path: when the clock expires the result is the sentinel (1000),
    # which matches no option index, so control falls through past the last test.
    # A distinct timeout body lives here, exactly as in the original bytecode
    # (Spencer's "Uh... I'll skip this for now...", Magic's frozen-with-fear beat).
    for n in node.get("timeout_body") or []:
        _gen_node(n, asm, st, cast)
    asm.label(end_lbl)


# --------------------------------------------------------------------------- #
# Scene -> .kiw chunk, and episode -> .exp container
# --------------------------------------------------------------------------- #
# Argument counts the reference runtime enforces for each yield (engine.py
# dispatch: need(n) = exactly n, at_least(n) = n or more). Extracted from
# shs-runtime's EngineState.dispatch and cross-checked against all 21,109 yields
# in the 17 original episodes (only yield 91 differs there: the shipped game calls
# the loading screen with 0 args, the runtime requires 1 -- we emit the 1-arg form).
YIELD_ARITY = {
    0: ("at_least", 1), 1: ("need", 8), 2: ("need", 6), 3: ("need", 3),
    4: ("need", 1), 5: ("need", 2), 8: ("need", 4), 10: ("need", 2),
    11: ("at_least", 2), 13: ("at_least", 2), 16: ("need", 1), 17: ("need", 3),
    18: ("need", 0), 24: ("need", 2), 25: ("need", 2), 27: ("need", 1),
    28: ("need", 1), 29: ("need", 1), 34: ("at_least", 2), 35: ("at_least", 4),
    40: ("need", 3), 44: ("need", 2), 45: ("need", 1), 46: ("need", 2),
    47: ("need", 3), 49: ("need", 3), 50: ("need", 4), 51: ("need", 3),
    52: ("need", 3), 53: ("need", 2), 54: ("need", 2), 55: ("need", 1),
    64: ("need", 2), 65: ("need", 3), 71: ("need", 8), 72: ("need", 2),
    73: ("need", 1), 74: ("need", 1), 75: ("need", 1), 78: ("need", 3),
    79: ("need", 1), 80: ("need", 2), 81: ("need", 1), 82: ("need", 0),
    86: ("at_least", 3), 88: ("need", 1), 89: ("need", 0), 90: ("need", 2),
    91: ("need", 1), 94: ("need", 22), 95: ("need", 1), 96: ("need", 20),
    99: ("need", 1),
}


class EncoderVerifyError(ValueError):
    pass


def _verify_instructions(instrs, where="", verbatim=()):
    """Reject bytecode the reference runtime would stop on ("Playback paused"):
    a yield with the wrong argument count, a scene that does not open with the
    frame prologue (PUSH_RET; PUSH_FP; SET_FP -- without it the first stack op
    faults with "stack address -1 out of range"). (0x41 frame loads are legal:
    lowered routines read their locals and arguments with them.)"""
    problems = []
    ops = [i[0] for i in instrs]
    if ops[:3] != [PUSH_RET, PUSH_FP, SET_FP]:
        problems.append("scene does not start with the frame prologue 42 48 4a")
    for k, ins in enumerate(instrs):
        op, operand = ins[0], ins[1]
        if op == SYS and isinstance(operand, int) and k not in verbatim:
            y, argc = operand >> 8, operand & 0xFF
            rule = YIELD_ARITY.get(y)
            if rule and ((rule[0] == "need" and argc != rule[1])
                         or (rule[0] == "at_least" and argc < rule[1])):
                problems.append("instruction %d: yield %d with %d args (runtime needs %s %d)"
                                % (k, y, argc, "exactly" if rule[0] == "need" else "at least",
                                   rule[1]))
    if problems:
        raise EncoderVerifyError("%s: %d problem(s): %s"
                                 % (where or "scene", len(problems), "; ".join(problems[:6])))


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
    cast_for_table = cast
    # Cast/name table FIRST: the decoder reads NUL-terminated names starting at
    # byte 15, up to an "Event" marker. Byte 14 is the instruction-count low byte
    # of the header, which can be printable and would otherwise glue onto the
    # first name. Emit a single leading NUL so the first real name always starts
    # clean at an even offset, then the cast names, then "Event". This mirrors the
    # blank leading-slot form the decoder already restores (cast index 0 = the
    # player). Names are interned before any dialogue text so they sit at the
    # front of the word table where read_cast looks.
    if cast_for_table:
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
    setup = scene.get("_setup")
    _NARRATOR[0] = _int((setup or {}).get("ui_defaults", {}).get("75")) \
        if (setup or {}).get("ui_defaults", {}).get("75") is not None else None
    if scene.get("_in_code"):
        pass                               # the scene's exact code registers its cast
    elif setup is not None:
        # the registration the original scene performs, recorded by execution
        # (decoder scene_setup): same ids, art bases, names and format flags
        for c in setup.get("characters") or []:
            cid = _int(c.get("id"))
            if c.get("art") is not None:
                asm.emit(PUSH, cid & 0xFFFF)
                asm.emit(PUSH, _int(c["art"]) & 0xFFFF)
                asm.emit(SYS, (S_SET_ART << 8) | 2)
            if c.get("name") is not None:
                asm.emit(PUSH, cid & 0xFFFF)
                asm.emit(PUSH, st.intern(c["name"]))
                if "format" in c:
                    asm.emit(PUSH, _int(c["format"]) & 0xFFFF)
                    asm.emit(SYS, (S_SET_NAME << 8) | 3)
                else:
                    asm.emit(SYS, (0x40 << 8) | 2)          # yield 64: name as-is
        for slot, val in sorted((setup.get("ui_defaults") or {}).items()):
            asm.emit(PUSH, _int(val) & 0xFFFF)
            asm.emit(SYS, (_int(slot) << 8) | 1)           # 74 / 75
    else:
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
    # Scene body.
    if scene.get("_graph"):
        _emit_graph_scene(scene["_graph"], asm, st, cast)
    else:
        for node in scene.get("nodes", []):
            _gen_node(node, asm, st, cast)
    # Terminal: HALT. A scene compiled standalone has no caller to RET to, and a
    # `goto` node (if present) already transferred control before we get here.
    asm.emit(HALT)

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
    main_count = len(words) // 2
    # VM scratch cells (lowered routines' scratch words, engine text buffers)
    # are the uninitialised data words right after the main words
    scratch = getattr(asm, "scratch", 0)
    gap_count = scratch + 128 if scratch else 0
    for ins in asm.ins:
        if isinstance(ins[1], tuple) and ins[1][0] == "scratch":
            ins[1] = (main_count + ins[1][1]) & 0xFFFF
    instrs = asm.resolve()
    _verify_instructions(instrs, "scene %s" % scene.get("id"), asm.verbatim)
    code = _serialize_instructions(instrs)

    # KiWi v2 header: 'kiwi' + version(2) + previous_flag(0) + flag(0)
    #   + counts(main, gap, extra, instr) ; then word table ; then code.
    header = bytearray()
    header += b"kiwi"
    header.append(2)                 # version
    header.append(0)                 # previous_flag
    header.append(0)                 # flag
    header += struct.pack(">H", main_count)   # main_words
    header += struct.pack(">H", gap_count)    # gap_count
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


def compile_episode(spec, graph=True):
    """Compile an authoring spec (dict) into .exp bytes.

    Accepts two shapes:
      * Authoring form: {scenes: [{id, nodes:[...]}]}  -- nested bodies inline.
      * Decoder-output form: {segments: {seg_id: {nodes}}, scene_entries: {...}}
        -- the flat segment graph shs_decoder emits. It is flattened per scene
        back into nested nodes before compiling, so a decoded episode can be
        edited and re-encoded with the same schema.
    """
    global _ROUTINES
    _ROUTINES = dict(spec.get("routines") or {})
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
        ents = {}
        for num in scene_nums:
            e = entries.get(num) or ("s%d" % num)
            if e in segments:
                ents[num] = e
        scene_of = _assign_scenes(segments, ents)
        # Cross-scene gotos that land MID-scene (a section entry): collect
        # (register, value, target) per target scene for its entry dispatch.
        sec_entries = {}
        for _sid, _g in segments.items():
            for _n in _g.get("nodes", []):
                if not (isinstance(_n, dict) and _n.get("type") == "goto_scene"):
                    continue
                _f = _n.get("file")
                _reg = _n.get("section_register") or {}
                if not isinstance(_f, str) or _reg.get("var") is None:
                    continue
                _f = _f.replace(".json", "")
                _tn = scene_of.get(_f)
                if _tn is None or _f == ents.get(_tn) or scene_of.get(_sid) == _tn:
                    continue
                sec_entries.setdefault(_tn, []).append(
                    (_int(_reg["var"]), _int(_reg.get("value")), _f))
        use_graph = bool(spec.get("_graph_compile")) or graph
        # The scene prologue's own dispatch table (section_dispatch.bindings:
        # register value -> section byte offset) -- the original op5c cascade run
        # on scene entry. Map each offset to the segment starting (or sitting
        # just after) there and add it to that scene's section-entry dispatch.
        _ranges = {}
        for _sid, _g in segments.items():
            _o = [int(x["offset"]) for x in _g.get("nodes", [])
                  if isinstance(x, dict) and str(x.get("offset", "")).lstrip("-").isdigit()]
            if _o:
                _ranges[_sid] = (min(_o), max(_o))

        def _seg_at_offset(off, scn):
            cont = after = al = None
            for _nm, (_lo, _hi) in _ranges.items():
                if scene_of.get(_nm) != scn:
                    continue
                if _lo <= off <= _hi:
                    cont = _nm
                elif _lo >= off and (al is None or _lo < al):
                    after, al = _nm, _lo
            return cont or (after if after and al - off <= 64 else None)

        for _scn, _sd in (spec.get("section_dispatch") or {}).items():
            if not isinstance(_sd, dict) or _sd.get("register") is None:
                continue
            _tn = _int(_scn, 0)
            _ebv = _sd.get("entry_by_value")
            if isinstance(_ebv, dict) and _ebv:
                # engine-accurate (decoder ran the prologue per register value)
                for _v, _seg in _ebv.items():
                    _seg = str(_seg).replace(".json", "")
                    if _seg in segments and _seg != ents.get(_tn):
                        sec_entries.setdefault(_tn, []).append(
                            (_int(_sd["register"]), _int(_v), _seg))
                continue
            for _b in _sd.get("bindings") or []:
                if _b.get("value") is None or _b.get("section_offset") is None:
                    continue
                _seg = _seg_at_offset(_int(_b["section_offset"]), _tn)
                if _seg and _seg != ents.get(_tn):
                    sec_entries.setdefault(_tn, []).append(
                        (_int(_sd["register"]), _int(_b["value"]), _seg))
        for num in sorted(ents):
            entry = ents[num]
            members = {s for s, sn in scene_of.items() if sn == num}
            if use_graph:
                # Graph compiler (DEFAULT): every segment a labelled block, links as
                # jumps, calls as PUSH_RET+JMP / returns as guarded RET, and the
                # section-entry dispatch. Across the 16 decoded episodes it keeps
                # all but 29 of 11056 runtime-reachable lines, and every re-encoded
                # episode replays its choices correctly in the reference engine
                # (except Wrong Side's synthesized fights). The tree fold below
                # (graph=False / --tree) loses ~566 and is kept for comparison.
                scene_node = {"id": num, "nodes": [],
                              "_graph": (segments, entry, members, scene_of,
                                         sec_entries.get(num, []))}
            else:
                # Default: fold the scene graph into a nested tree, then append
                # any same-scene segment the fold did not reach so its content is
                # kept (verified runtime-reachable across the corpus).
                nodes = _flatten_segment_graph(segments, entry)
                covered = _flatten_covered_segments(segments, entry)
                for sid in segments:
                    if _scene_id_from_seg(sid) == num and sid not in covered:
                        for n in segments[sid].get("nodes", []):
                            if isinstance(n, dict) and n.get("type") not in ("next",):
                                nodes.append(n)
                scene_node = {"id": num, "nodes": nodes}
            _su = (spec.get("scene_setup") or {}).get(str(num))
            if _su is not None:
                scene_node["_setup"] = _su
            if str(num) in (spec.get("ir_scenes") or []):
                scene_node["_in_code"] = True
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
    ap.add_argument("--tree", action="store_true",
                    help="use the legacy tree fold instead of the graph compiler")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.input).read_text())
    data = compile_episode(spec, graph=not args.tree)
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
