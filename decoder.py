#!/usr/bin/env python3
"""
shs_decoder.py — Extract & decode "Surviving High School" episode files.

Handles two file types (reverse-engineered formats; see notes below):

  1. .exp  — a "CSPUD" archive containing a compiled dialogue script plus
             image assets (character portraits, backgrounds).
  2. .kiw  — a compiled dialogue script ("kiwi" magic). Contains the text,
             a cast/name table, and a stack-VM bytecode that ties each line
             to a speaker and an emotion.

Typical use:

    # Decode a script straight to a transcript:
    python3 shs_decoder.py script.kiw -o transcript.md

    # Unpack a container (writes the .kiw + .png assets) and decode it:
    python3 shs_decoder.py Episode.exp --extract-dir out/ -o transcript.md

    # A continuation scene whose .kiw has no cast table of its own — borrow
    # the cast table from scene 1 of the same episode:
    python3 shs_decoder.py scene3.kiw --cast-from scene1.kiw -o scene3.md

------------------------------------------------------------------------------
FORMAT NOTES (what the bytes mean)
------------------------------------------------------------------------------
.exp  (CSPUD archive)
  off 0 : b"CSPUD"
  off 5 : uint32be  entry_count
  off 9 : directory of `entry_count` records, each:
            uint16be id, uint32be offset
  each chunk @offset:
            uint32be comp_size, uint32be uncomp_size, uint32be flags, then data
            flags & 1  -> data is an LZMA-alone stream (13-byte header)
            flags == 0 -> data is stored raw (comp_size == uncomp_size)
  chunk payloads are identified by magic: b"kiwi" (script) or b"\\x89PNG".

.kiw  (compiled dialogue script)
  off 0  : b"kiwi" + 10 header bytes; string content begins at off 14
  strings: NUL-terminated; the first one carries a leading 'L' byte.
  cast   : the run of strings from off 14 up to the "Event" marker
           (index 0 = first speaker, etc.). Continuation scenes may omit it.
  bytecode: everything after the last long string. Relevant opcodes:
            0x1a / 0x41  push uint16be                       (3 bytes)
            0x1b         push (emotion, speaker) byte pair    (3 bytes)
            0x1f <sel>   call builtin <sel>; consumes operands (2 bytes)
                           sel 0x0d -> show dialogue line (has a speaker)
                           sel 0x41 -> show narration / stage direction (no speaker)
                           other sel -> set sprite / background / etc. (no line)
  A line's TEXT is the pushed value V whose string lives at  off = V*2 + 15
  (text offsets are stored as 16-bit word indices). The SPEAKER/EMOTION is the
  push immediately after the text: a 0x1b pair (emotion, speaker) or a small
  bare value (speaker only). speaker indexes the cast table; out-of-range means
  the line is narration / a system card.

Emotion codes are emitted raw (emotion_code); the engine maps a code to a
sprite frame (portrait = sprite_base + code). No verbal labels are assigned.
"""

import argparse
import json
import lzma
import re
import struct
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# KiWi v2 instruction core (authoritative, verified against libshs09.so)
# --------------------------------------------------------------------------- #
# Ground-truth decode of the KiWi v2 script format, folded in from the reference
# runtime (decode verified against FUN_00057068, jump model against FUN_00055ce0).
# This replaced an earlier heuristic model that guessed the code-section start
# (max-string-end), guessed which opcodes take operands, and computed jump
# targets in byte space -- all three were systematically wrong. The format has a
# real header with explicit counts, an authoritative operand table, and jump
# targets in logical-PC space; the code below reads exactly that.

# int32 table at libshs09.so 0x002593a8: only these opcodes consume a 2-byte
# operand. Everything else is a lone opcode byte.
_KIWI_OPERAND_OPCODES = frozenset({
    0x01, 0x19, 0x1A, 0x1B, 0x1E, 0x1F, 0x20, 0x23,
    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x40, 0x41, 0x5C,
})


class _KiwiFormatError(ValueError):
    """Not a decodable KiWi v2 program (bad signature, version, or truncated)."""


class _KiwiInstr:
    """One decoded instruction: logical pc, byte offset, opcode, operand."""
    __slots__ = ("pc", "byte_offset", "opcode", "operand")

    def __init__(self, pc, byte_offset, opcode, operand):
        self.pc = pc
        self.byte_offset = byte_offset
        self.opcode = opcode
        self.operand = operand

    def branch_target(self):
        """Statically encoded branch target in logical instruction indices, or
        None. Native PC arithmetic wraps at 16 bits; the dynamic-return opcode
        0x43 is deliberately not assigned a guessed target."""
        op = self.opcode
        if op == 0x29:
            return self.operand
        if op in (0x28, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E):
            return (self.pc + self.operand) & 0xFFFF
        if op == 0x5C:
            return (self.pc + (self.operand >> 8)) & 0xFFFF
        return None


class _KiwiProgram:
    """Decoded header counts plus the instruction stream. Only the fields the
    decoder actually consumes are kept (no re-encoder)."""
    __slots__ = ("main_words", "extra_words", "code_start", "instructions")

    def __init__(self, main_words, extra_words, code_start, instructions):
        self.main_words = main_words
        self.extra_words = extra_words
        self.code_start = code_start
        self.instructions = instructions


def _kiwi_decode(data):
    """Decode a KiWi v2 script strictly. Header: 'kiwi' + version(=2) +
    previous_flag + [prev_data,prev_code if flag] + flag + 4 counts
    (main,gap,extra,instr); then the word table (main+extra 16-bit words); then
    `instr` instructions, each an opcode with a 2-byte operand iff the opcode is
    in _KIWI_OPERAND_OPCODES. Raises _KiwiFormatError on anything unexpected."""
    pos = 0
    n = len(data)

    def take(size):
        nonlocal pos
        if pos + size > n:
            raise _KiwiFormatError("truncated KiWi at byte 0x%x" % pos)
        chunk = data[pos:pos + size]
        pos += size
        return chunk

    def u16():
        return struct.unpack(">H", take(2))[0]

    if take(4) != b"kiwi":
        raise _KiwiFormatError("invalid KiWi signature at byte 0x0")
    version = take(1)[0]
    if version != 2:
        raise _KiwiFormatError("unsupported KiWi version %d" % version)
    previous_flag = take(1)[0]
    _prev_data = u16() if previous_flag else None
    previous_code = u16() if previous_flag else None
    _flag = take(1)[0]
    main, _gap, extra, count = (u16() for _ in range(4))
    main_words = tuple(u16() for _ in range(main))
    extra_words = tuple(u16() for _ in range(extra))
    code_start = pos
    base_pc = previous_code or 0
    instructions = []
    for index in range(count):
        offset = pos
        opcode = take(1)[0]
        operand = u16() if opcode in _KIWI_OPERAND_OPCODES else None
        instructions.append(_KiwiInstr(base_pc + index, offset, opcode, operand))
    if pos != n:
        raise _KiwiFormatError("%d trailing bytes at byte 0x%x" % (n - pos, pos))
    return _KiwiProgram(main_words, extra_words, code_start, tuple(instructions))


_kiwi_cache = {}


def _kiwi_program(data):
    """Cached _kiwi_decode keyed by object identity (script payloads are reused
    within a run)."""
    hit = _kiwi_cache.get(id(data))
    if hit is not None and hit[0] is data:
        return hit[1]
    prog = _kiwi_decode(data)
    _kiwi_cache[id(data)] = (data, prog)
    return prog

STRING_BASE = 15
EPISODE_ASSET_BASE = 20000  # asset ids at/above this are PNG chunks packaged in
                            # the .exp itself (episode-specific art); below are
                            # ids into the shared, external sprite library.          # text offset = ref * 2 + STRING_BASE
SFX_ID_MIN, SFX_ID_MAX = 8001, 8113      # engine-validated bands, read from
MUSIC_ID_MIN, MUSIC_ID_MAX = 8201, 8232  # libshs09.so's own range checks
                                         # (`id - 0x1f41 < 0x71`, `id - 0x2009 < 0x20`)
CAST_START = 14           # first cast string begins here ('L' + name)

# In-dialogue emphasis: the script marks emphasised words/titles with backticks
# (`like this`) but carries NO colour. The engine colours emphasis by the
# SPEAKER's gender (the `type` field in the name table): male -> one colour,
# female -> the other. Narration (no speaker) and per-term exceptions are set in
# the overlay. These are the fallback colours if the overlay omits them.
EMPHASIS_MALE = "#e23b3b"     # red
EMPHASIS_FEMALE = "#3b78e2"   # blue
EMPHASIS_NARRATION = "#e23b3b"


def emphasis_to_bbcode(text, base_color, term_colors=None):
    """Return `text` with emphasis wrapped as Godot BBCode (italic + colour), or
    None if nothing was wrapped. Backtick spans use term_colors[span] if listed,
    else base_color (the speaker's gender colour). Listed terms that are NOT
    backtick-marked are also wrapped, so colour can be forced where the script
    has no marker (e.g. "Ten days!")."""
    term_colors = term_colors or {}

    def repl(m):
        inner = m.group(1)
        color = term_colors.get(inner, base_color)
        if not color:
            return "[i]%s[/i]" % inner          # unknown gender -> italic, no colour
        return "[color=%s][i]%s[/i][/color]" % (color, inner)

    out = re.sub(r'`([^`]+)`', repl, text)
    for term, color in term_colors.items():
        if ("`%s`" % term) in text:          # already handled by the backtick pass
            continue
        if term and term in out:
            out = out.replace(term, "[color=%s][i]%s[/i][/color]" % (color, term))
    return out if out != text else None

DISPLAY_DIALOGUE = 0x0d   # 1f 0d -> character speech line (carries a speaker)
SET_SPEAKER = 0x05        # 1f 05 -> stage setup: the pushed pair names two resource rows
                          #          (the characters shown on screen). Speaker-less 0x0d
                          #          lines are instead resolved by row markers: a 0x5a byte
                          #          before the call = row 0 speaks, 0x5b = row 1 speaks.
CUSTOM_SPEAKER = 0x0f     # 1f 0f -> speech by a one-off named speaker (name is a
                          #          string, not a cast index): "Other Voice", etc.
TITLE_CARD = 0x08         # 1f 08 -> title / intro card (episode title + subtitle)
DISPLAY_NARRATION = 0x41  # 1f 41 -> narration / stage-direction card (no speaker)
DISPLAY_STATUS = 0x00     # 1f 00 -> on-screen status / HUD line, often a printf-style
                          #          template the engine fills at runtime (e.g.
                          #          "Kim has %d dollars." -> the live money counter).
SET_BG_CUSTOM = 0x23      # 1f 23 -> set a packaged background, by PNG asset # (this file)
SET_BACKGROUND = 0x0b     # 1f 0b -> set the displayed background image, by *global* asset
                          #          id (~1000-1111, a shared library not bundled in the .exp);
                          #          this is the actual on-screen background. Operands:
                          #          [asset_id, 0xffff].
SET_MUSIC = 0x50          # 1f 50 -> change background music, by track id (0x20xx ~ 8201-8230,
                          #          a separate audio band from the 0x4f SFX cues at ~8003-8007).
                          #          Reused across scenes/moods; this is the "sound change" cue.
STOP_MUSIC = 0x51         # 1f 51 -> stop/fade the current music (yield 81, need 1;
                          #          engine sets audio_stopped, music_id=-1). Appears
                          #          ~158x; distinct from 0x52. (Corrected: 0x52 is
                          #          vibrate, not stop-music -- verified vs the
                          #          reference engine's yield handlers.)
VIBRATE = 0x52            # 1f 52 -> haptic buzz (yield 82, need 0). No audio effect.
PRESENT_CHOICE = 0x01     # 1f 01 -> present a branching choice menu ("opt|opt|...")
TCHOICE_SETUP = 0x02      # 1f 02 -> begin a TIMED text choice: pushes the prompt, the
                          #          setup line, and a countdown timer (ms, e.g. 8000),
                          #          plus the first option. Followed by 1f 03 options and
                          #          a 1f 04 resolve. Distinct from 1f 01 (untimed) and
                          #          from the word/action minigame that also reuses 0x02
                          #          but with a 0xffff sentinel and no 1f 03 options.
TCHOICE_OPTION = 0x03     # 1f 03 -> add one option to the timed choice being built.
TCHOICE_RESOLVE = 0x04    # 1f 04 -> resolve the timed choice; the picked index is then
                          #          tested (push K; 0x0a EQ; 0x2b) to branch win/lose.
PLAY_SFX = 0x4f           # 1f 4f -> play a sound effect, by SFX id. Only a few ids recur
                          #          (e.g. 8003 / 8006 / 8007), reused throughout an episode
                          #          as cue sounds between lines -- NOT story/decision flags.
SCENE_GOTO = 0x0a         # 1f 0a -> transfer control to another scene, by its chunk id
                          #          (25001..=0x61a9..). Scenes form a graph via these jumps;
                          #          chunk/file order is NOT play order (e.g. an intro scene
                          #          jumps to a "day hub" scene that dispatches to task scenes).
PLACE_SPRITE = 0x34       # 1f 34 -> position a character's portrait as an on-screen sprite
                          #          (operands: cast row + a fixed slot value). The POV/player
                          #          character is the camera and is never placed this way, so a
                          #          frequent speaker that is never placed is a playable character.
ACTION_MINIGAME = 0x47    # 1f 47 -> action/word minigame round: prompt + one or more
                          #          pipe-delimited option groups (e.g. tutoring rounds)
SCORE_FEEDBACK = 0x58     # 1f 58 -> minigame score / feedback banner ("Score Up!")
SET_EXPRESSION = 0x05     # (handled via SET_SPEAKER path; expression 0x05 shares selector)
SET_SCENE_VALUE = 0x10    # 1f 10 -> set the scene 'value' register (yield 16, need 1)
TEXT_EQUAL = 0x18         # 1f 18 -> compare two strings, push bool (yield 24, need 2)
COPY_LAST_INPUT = 0x1c    # 1f 1c -> copy the last text input into a string slot (yield 28)
TEXT_INPUT = 0x28         # 1f 28 -> prompt the player to type text (yield 40, need 3:
                          #          title, prompt, destination string slot)
SET_STRING = 0x2e         # 1f 2e -> string variable write [key_ref, value_ref] (yield 46).
                          #          NOTE: 0x2e is also the $token rename; a write with two
                          #          NON-$ string refs is a set_string, a $Token first ref is
                          #          a rename (handled via _name_renames).
CHARACTER_PICKER = 0x4e   # 1f 4e -> player picks 1..5 characters (yield 78, need 3:
                          #          prompt, count, address-of-id-table in the word region)
SET_UI_DEFAULT = 0x4b     # 1f 4b -> set a UI default slot (yield 75, need 1)
WOBBLE = 0x59             # 1f 59 -> wobble the next dialogue line (yield 89, need 0)
LOADING = 0x5b            # 1f 5b -> show/hide the loading screen (yield 91, need 1 flag)
SET_POV = 0x4a            # 1f 4a -> set the player-controlled (POV) character, by cast
                          #          index. The engine's authoritative "you now play as X"
                          #          state set; the on-screen "You are now playing as X"
                          #          caption (when present) is separate narration a few
                          #          instructions later. Operand is the cast index, pushed
                          #          as a 1a value or as the first byte of a 1b pair. Holds
                          #          until the next 1f4a, so co-present main characters who
                          #          merely speak do not change the POV. Drives Matt<->Sir
                          #          Smoothness in ATGB and Emily/Cameron/Hannah/Ben in Swim.
VAR_READ = 0x2d           # 1f 2d -> read a game variable (operand: var id). The read
                          #          value feeds gates and read-modify-write updates.
VAR_WRITE = 0x2c          # 1f 2c -> write a game variable. Two operand forms:
                          #          [var, value] = set var; [delta] = add the signed
                          #          delta to the variable just read by 1f 2d (the
                          #          read-modify-write form, e.g. money -10 / +1).

# Custom-background PNG names. The PNG asset id is byte-literal; what the image
# DEPICTS is observed, so names come from the per-episode overlay (key
# `bg_names`: {"0x6596": "...", ...}) and the parser ships none of its own.
KNOWN_BG = {}


# --------------------------------------------------------------------------- #
# VM opcode reference: builtins the interpreter handles but the decoder does not
# yet emit a node for.
#
# These were read directly from the engine's bytecode interpreter (FUN_0009fe3c
# in libshs09.so), which dispatches on the raw opcode byte; a `1f` byte is an
# escape meaning "the next byte is the opcode". Everything below is a `1f XX`
# builtin whose XX is listed. None of them appears in the ~9 episodes decoded so
# far, so they are documented rather than handled -- when an episode that uses
# one turns up, this is the spec for adding it. Operand fetch is the usual
# 16-bit word; a name-slot operand (>= 0x7ff5) resolves through the runtime
# name table, i.e. it is a $Player/$Antagonist-style substitution.
#
# TEXT-EMITTING builtins (all call the same display routine as SAY/NARR, but wrap
# it differently -- these DO produce on-screen lines and would need transcript
# nodes):
#   1f 22  SAY_WITH_PORTRAIT   spoken line that also (re)positions the speaker's
#                              portrait sprite; falls back to a plain line if no
#                              portrait is active.
#   1f 24  SAY_INTERP1         line with ONE runtime value interpolated into the
#                              text (a name-slot or number spliced in at render).
#   1f 25  SAY_INTERP1_MODAL   as 1f24, but shown as a blocking/await card (sets
#                              the "awaiting input" flag; execution pauses).
#   1f 26  SAY_POSITIONED      line drawn at explicit screen coordinates (operand
#                              selects the anchor), rather than the default box.
#   1f 27  SAY_STYLED          line with a style/format variant applied (no extra
#                              operand; a preset look).
#   1f 4c  SAY_INTERP2_MODAL   line with TWO runtime values interpolated, modal.
#   1f 56  SAY_AUTO            line that auto-advances (no tap): if its slot is 0
#                              it triggers an auto-advance before positioning.
#   1f 5e  CHOICE_BUILD        emits the line then constructs a full choice-menu
#                              object -- a menu-building card distinct from the
#                              1f01 choice path already handled.
#
# NON-CONTENT builtins (state, frame, audio, widget -- correctly produce NO
# transcript node; listed so their bytes are not mistaken for missing content):
#   1f 10  scene/transition style set (cosmetic; operand is 20 or 6). Investigated
#          separately: not gender/character, tracks episode+scene presentation.
#   1f 11  jump/continue (shares the 1f28 jump landing).
#   1f 16/18/19/1a  frame control helpers (advance/query/reset frame state).
#   1f 1b  RANDOM: pushes abs(rand() % operand) in [0, operand); operand is the
#          range (a 0 operand yields 0). Consumed on the stack by the following
#          arithmetic -- e.g. combat damage is RANDOM(0..attack)+1 = [1, attack].
#          Already reconstructed inside combat_decode.py, not emitted standalone.
#   1f 1c  frame string fetch into a scratch buffer.
#   1f 1d  load a frame-local slot value onto the stack.
#   1f 1e  virtual call through the scene object's vtable (engine-internal).
#   1f 21  build a UI widget object (operator_new; no story text).
#   1f 2e  RENAME: bind a $token to a display name (two name-slot operands).
#          Handled at decode time via _name_renames / runtime_text_vars, so no
#          node is emitted here.
#   1f 2f/31  frame-driven widget construction (FUN_0009f420 value fetch + build).
#   1f 48  frame variable WRITE (array-indexed store into scene state).
#   1f 49  frame variable READ (array-indexed load from scene state).
#   1f 4b  companion to 1f4a (POV): sets a secondary control/most-recent slot.
#   1f 51  MINIGAME_BY_ID / sound trigger: operand is a minigame id (1000/2000/
#          3000/5000). Used as the content_fork play-point marker; consumed by
#          the minigame resolution pass, not emitted as its own node.
#   1f 53/54/55  audio transport (pause/resume/stop-all on the sound engine).
#   1f 57  read a scene status flag onto the stack.
#   1f 59  set a scene completion flag.
#   1f 5b/60  allocate/enter a sub-scene or overlay object (frame push).
#   1f 5f  conditional scene-state query (branches on operand == 1).
#   1f 61  toggle a boolean scene flag (operand != 0).
#   1f 62  no-op in this interpreter.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _printable(b: int) -> bool:
    return 32 <= b < 127


def find_strings(data: bytes):
    """Return list of (offset, text) for every NUL-terminated ASCII run."""
    out, i, n = [], 0, len(data)
    while i < n:
        if _printable(data[i]):
            j = i
            while j < n and _printable(data[j]):
                j += 1
            out.append((i, data[i:j].decode("latin1")))
            i = j
        else:
            i += 1
    return out


def lzma_alone_decompress(stream: bytes, uncomp_size: int) -> bytes:
    """Decompress an LZMA-alone stream that uses an in-file 13-byte header.

    The container stores the true output size in the chunk header, so we
    rebuild a clean ALONE header (props + dict + correct size) and feed the
    compressed payload that follows the original 13-byte header.
    """
    if len(stream) < 13:
        return b""                       # empty / placeholder chunk
    props = stream[0]
    dict_size = stream[1:5]
    header = bytes([props]) + dict_size + struct.pack("<Q", uncomp_size)
    return lzma.decompress(header + stream[13:], format=lzma.FORMAT_ALONE)


# --------------------------------------------------------------------------- #
# .exp (CSPUD) archive
# --------------------------------------------------------------------------- #
class ExpArchive:
    MAGIC = b"CSPUD"

    def __init__(self, data: bytes):
        if data[:5] != self.MAGIC:
            raise ValueError("Not a CSPUD (.exp) archive")
        self.data = data
        self.entries = self._read_directory()

    def _read_directory(self):
        data = self.data
        count = struct.unpack(">I", data[5:9])[0]
        entries, p = [], 9
        for _ in range(count):
            eid = struct.unpack(">H", data[p:p + 2])[0]
            off = struct.unpack(">I", data[p + 2:p + 6])[0]
            entries.append((eid, off))
            p += 6
        return entries

    def chunks(self):
        """Yield (id, payload_bytes, kind) for every chunk.

        kind is one of: 'script', 'image', 'data'.
        """
        data = self.data
        for eid, off in self.entries:
            comp = struct.unpack(">I", data[off:off + 4])[0]
            uncomp = struct.unpack(">I", data[off + 4:off + 8])[0]
            flags = struct.unpack(">I", data[off + 8:off + 12])[0]
            # Bound the body by the chunk's own compressed size, not the next
            # directory entry: some archives alias several ids to one offset
            # (duplicate entries), which would make a next-offset span empty.
            body = data[off + 12:off + 12 + comp]
            if flags & 1:
                payload = lzma_alone_decompress(body, uncomp)
            else:
                payload = body[:uncomp] if uncomp else body
            if payload[:4] == b"kiwi":
                kind = "script"
            elif payload[:4] == b"\x89PNG":
                kind = "image"
            else:
                kind = "data"
            yield eid, payload, kind

    def episode_title(self):
        """Return the episode title from the metadata chunk, if present."""
        for _eid, payload, kind in self.chunks():
            if kind == "data":
                for off, text in find_strings(payload):
                    if len(text) >= 4:
                        return text
        return None

    def episode_meta(self):
        """Parse the metadata chunk (id 1): {pack_id, episode_id, titles}. The
        chunk is `>HH pack_id, episode_id` then five `>H`-length UTF-8 titles.
        pack_id identifies the pack/season; episode_id the number within it."""
        for eid, payload, kind in self.chunks():
            if eid == 1 or (kind == "data" and len(payload) >= 4):
                try:
                    pack_id, episode_id = struct.unpack(">H", payload[0:2])[0], \
                        struct.unpack(">H", payload[2:4])[0]
                    titles, pos = [], 4
                    for _ in range(5):
                        if pos + 2 > len(payload):
                            break
                        ln = struct.unpack(">H", payload[pos:pos + 2])[0]
                        pos += 2
                        titles.append(payload[pos:pos + ln].decode("utf-8", "replace"))
                        pos += ln
                    return {"pack_id": pack_id, "episode_id": episode_id,
                            "titles": titles}
                except Exception:
                    return None
        return None

    def extract(self, out_dir: Path):
        """Write every chunk to out_dir; return path to the script chunk."""
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = None
        for eid, payload, kind in self.chunks():
            ext = {"script": "kiw", "image": "png"}.get(kind, "bin")
            path = out_dir / f"asset_{eid:04x}.{ext}"
            path.write_bytes(payload)
            if kind == "script" and script_path is None:
                script_path = path
        return script_path


# --------------------------------------------------------------------------- #
# .kiw (compiled dialogue script)
# --------------------------------------------------------------------------- #
class Line:
    __slots__ = ("text", "speaker", "emotion", "sprite", "off", "bc_off",
                 "section_start", "section_jump", "threshold")

    def __init__(self, text, speaker, emotion, sprite=None, off=None, bc_off=None,
                 section_start=False, section_jump=None, threshold=None):
        self.text = text          # the line of text
        self.speaker = speaker    # cast name, None=narration, or a __TAG__ sentinel
        self.emotion = emotion    # raw emotion code, or card kind for sentinels
        self.sprite = sprite      # portrait image id (base+emotion), or None
        self.off = off            # source string offset (for locating cards)
        self.bc_off = bc_off      # bytecode offset of the instruction that emitted it
        self.section_start = section_start  # title card opening a dispatch-section
                                  # body (immediately preceded by a 0x42 SEP): the
                                  # prior section ends here and does NOT fall through
        self.section_jump = section_jump  # on such a title card: the section-body
                                  # offset the PRIOR section's terminating SEP jumps
                                  # to (so the prior section gotos there, not ends)
        self.threshold = threshold  # for an action minigame node: the per-round
                                  # score threshold the runtime compares against
                                  # (extracted from the immediate 5f 3f push N 0d 2b
                                  # gate that follows the 1f47 call, when present)


def read_cast(data: bytes):
    """Read the cast/name table from a script, or [] if it has none.

    The table is a run of NUL-terminated names ending at the "Event" marker.
    Names begin at offset 15 (string references resolve as ref*2 + 15, and
    name_ref 0 in the resource table lands on 15 in every episode checked);
    the byte at offset 14 is the last header byte. When that byte happens to
    be printable ('L', 'N', '$', ...), find_strings glues it onto the first
    name -- so a string that STARTS at 14 has its first character dropped.
    When it is NUL, the lead may instead appear as an *empty* slot (the
    player's name is supplied by the engine); the empty slot is kept so
    speaker indices in the bytecode line up with the cast. A leading '$' on
    a name (at 15) marks a variable/player-customisable name
    (e.g. "$ALLISON" -> "Allison").
    """
    cast = []
    first_off = None
    for off, text in find_strings(data):
        if off < CAST_START:
            continue
        if text == "Event":
            break
        if first_off is None:
            first_off = off
        name = text[1:] if off == CAST_START else text   # drop glued header byte
        name = _clean_name(name)
        cast.append(name)
        if len(cast) > 64:        # safety: a real cast table is short
            return []
    if cast and max(len(c) for c in cast) > 40:
        return []                 # "table" is actually dialogue -> no cast table
    # Empty leading slots: NUL bytes between CAST_START and the first real name
    # are blank cast entries (index 0...) that find_strings skipped. Restore them
    # so cast indices match the bytecode's speaker indices.
    if first_off is not None and first_off > CAST_START \
            and all(b == 0 for b in data[CAST_START:first_off]):
        cast = [""] * (first_off - CAST_START) + cast
    return cast


def lead_of(cast):
    """The lead/player: first non-empty cast entry (index 0 may be a blank slot)."""
    return next((c for c in cast if c), None)


def normalize_strings(strings):
    """Content strings begin at offset 15; the byte at 14 is the last header
    byte (verified: resource-table name_ref 0 resolves to 15 in every script).
    When that byte is printable it gets glued onto the first string -- this
    applies to EVERY script, not just cast tables (e.g. a continuation scene
    whose first string is 'XSophomore year' = header 'X' + 'Sophomore year').
    Returns the list with any string starting at 14 re-based to 15."""
    return [(15, t[1:]) if o == 14 and len(t) > 1 else (o, t) for o, t in strings]


def _clean_name(name):
    """'$NAME' marks a player-customisable name slot: '$ALLISON' -> 'Allison'."""
    return name[1:].capitalize() if name[:1] == "$" else name


def _resource_rows(data: bytes):
    """Parse the init resource table into rows of {typ, asset, name, idx}.

    Each 8-byte entry is: uint16 type, uint16 asset_id, uint16 name_ref,
    uint16 index. `name_ref` is a string word-reference (off = ref*2 + 15),
    which resolves each entry's name directly from the bytes -- the authoritative
    index<->name mapping. Types seen across episodes: 1 = male character,
    2 = female character, 3 = non-gendered (objects/props, e.g. a Computer).
    The table ends at the `type 3 / asset 0xffff` sentinel row (a type-3 row
    with a real asset id is an ordinary entry, NOT a terminator).
    """
    n = len(data)
    strings = find_strings(data)
    smap = {o: t for o, t in strings}
    # Names start at off 15; the byte at 14 is the last header byte. When it is
    # printable, find_strings glues it onto the first name -- register the
    # de-glued form at 15 so name_ref 0 resolves correctly.
    if 14 in smap and 15 not in smap:
        smap[15] = smap[14][1:]
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    # The init resource table sits between the strings and the code section. The
    # max-string heuristic can land short of it when a scene has many option/
    # choice strings, so also try scanning backward from the authoritative code
    # start (header-derived) where the table's sentinel/rows end.
    code_start = _kiwi_code_start(data)
    scan_offsets = list(range(bc_start, bc_start + 12))
    if code_start is not None:
        # The table ends just before the code, as a run of 8-byte rows terminated
        # by a type-3/asset-0xffff sentinel. Walk backward to the FIRST index-0
        # row (the head), preferring earlier anchors so we don't lock onto the
        # sentinel row (which also reports index 0).
        back = [code_start - 8 * r for r in range(1, 40) if code_start - 8 * r >= 0]
        scan_offsets += sorted(back)
    seen_anchor = set()
    for off in scan_offsets:
        if off in seen_anchor:
            continue
        seen_anchor.add(off)
        if off + 8 > n:
            break
        idx0 = (data[off + 6] << 8) | data[off + 7]
        typ0 = (data[off] << 8) | data[off + 1]
        if idx0 != 0 or typ0 not in (1, 2, 3):
            continue
        rows, k = [], off
        while k + 8 <= n and len(rows) < 256:
            typ = (data[k] << 8) | data[k + 1]
            asset = (data[k + 2] << 8) | data[k + 3]
            nref = (data[k + 4] << 8) | data[k + 5]
            idx = (data[k + 6] << 8) | data[k + 7]
            if asset == 0xffff or typ not in (1, 2, 3):
                break
            name = smap.get(nref * 2 + STRING_BASE)
            rows.append({"typ": typ, "asset": asset, "idx": idx,
                         "name": _clean_name(name) if name else None})
            k += 8
        return rows
    return []


def read_resources(data: bytes, smap: dict, bc_start: int, cast):
    """{character_name: sprite_base_asset_id}, names resolved via the resource
    table's own name_ref field (byte-literal; independent of cast alignment).
    A line's displayed portrait is sprite_base + emotion.

    A character may appear TWICE in the table: once with a shared/global sprite
    base (~1000-2500, from the library that is not bundled in the .exp) and once
    with an EPISODE-PACKAGED base (>= EPISODE_ASSET_BASE, a PNG chunk shipped in
    this .exp -- e.g. Halloween's costumed Kay/Kel at 26000/26005). The packaged
    art is the episode-specific one and takes precedence.
    """
    out = {}
    for r in _resource_rows(data):
        nm = r["name"]
        if not nm:
            continue
        cur = out.get(nm)
        if cur is None:
            out[nm] = r["asset"]
        elif r["asset"] >= EPISODE_ASSET_BASE > cur:
            out[nm] = r["asset"]      # episode-packaged art wins over the global
    return out


def _text_substitutions(data: bytes, rows=None):
    """Detect `1f 2e` renames used as TEXT-TEMPLATE substitutions and classify
    them as STABLE (bound once) or REBOUND (a mutable runtime variable).

    Distinct from `_name_renames` (which relabels a speaker): here a `$Token`
    appears literally inside narration/status/dialogue strings and is filled with
    a display value at runtime. Two very different cases occur:

      * STABLE -- a token bound to exactly one display value in the whole scene
        (e.g. `$MAN1 -> "Ice Cream Jim"`, `$Travis -> "Travis"`, and the
        customisable `$Player` -> protagonist name). These can be substituted
        into the text directly, everywhere.

      * REBOUND -- a token rebound to MANY values as the story progresses. Wrong
        Side of Town's battle system rebinds `$Antagonist` before each fight
        (`-> Travis`, then `-> Skazz`, `-> Alexei`, ... 14 opponents in all) and
        the combat HUD template "$Antagonist still has %d life left!" is SHARED,
        jumped into by every fight. Baking one name into it would be wrong, so
        `$Antagonist` stays a live variable: instead of substituting, we return
        its binding SCHEDULE -- the ordered (bytecode-offset, name) pairs -- so
        the runtime can show the name bound most recently before the line it is
        rendering.

    Returns (stable, schedule):
      stable   = {"$Token": "Display"}                 apply directly to text
      schedule = {"$Token": [(offset, "Display"), ...]} ordered; runtime-resolved
    Only tokens that literally occur in some string are considered, so speaker-
    only 1f2e aliases in other episodes are never touched.
    """
    strings = find_strings(data)
    smap = {o: t for o, t in strings}
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    present = set()
    for _o, _t in strings:
        for _m in re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", _t):
            present.add(_m)
    _protagonist = None
    if rows:
        for _r in rows:
            if _r.get("idx") == 0 and _r.get("name"):
                _protagonist = _r["name"]
                break
    # ordered list of (offset, token, display) bindings in bytecode order
    binds = []
    for q in fo:
        if not (data[q] == 0x1f and data[q + 1] == 0x2e):
            continue
        strs = []
        for j in range(fi[q] - 1, max(fi[q] - 5, -1), -1):
            qq = fo[j]
            if data[qq] == 0x1a:
                v = (data[qq + 1] << 8) | data[qq + 2]
                t = smap.get(v * 2 + STRING_BASE)
                if t and len(t) > 0:
                    strs.append(t)
                    if len(strs) == 2:
                        break
                else:
                    break
            else:
                break              # stop at any non-push (e.g. a prior 1f2e)
        strs.reverse()
        for _k, _s in enumerate(strs):
            if not (_s.startswith("$") and _s in present):
                continue
            _disp = strs[_k + 1] if _k + 1 < len(strs) else None
            if _disp is not None and not _disp.startswith("$"):
                binds.append((q, _s, _disp))
            elif _s == "$Player" and _protagonist:
                binds.append((q, _s, _protagonist))
    # group by token, preserving order
    by_token = {}
    for off, tok, disp in binds:
        by_token.setdefault(tok, []).append((off, disp))
    stable, schedule = {}, {}
    for tok, seq in by_token.items():
        distinct = {d for _o, d in seq}
        if len(distinct) == 1:
            stable[tok] = seq[0][1]            # bound once -> direct substitution
        else:
            schedule[tok] = seq                # rebound -> runtime variable
    return stable, schedule


def _apply_text_subs(text, subs):
    """Replace every `$Token` in `text` with its bound display value."""
    if not text or not subs or "$" not in text:
        return text
    for _tok in sorted(subs, key=len, reverse=True):
        if _tok in text:
            text = text.replace(_tok, subs[_tok])
    return text


def _name_renames(data: bytes, rows):
    """Detect dynamic character renames set by the `1f 2e` opcode.

    Shape: `push "$VarName" ; push "DisplayName" ; 1f2e`. It rebinds the
    character whose resource name is VarName (the `$` prefix marks a
    customisable name slot; the resource table stores it as the cleaned form)
    to a new on-screen DisplayName. Example: europe binds `$Man2 -> "French
    Man"`, so the French suitor's lines should be labelled "French Man" rather
    than the raw slot name "Man2".

    Only applied when the display name is NOT already a distinct resource row of
    its own: e.g. As Time Goes By's `$Travis -> "Mr. Hotpants"` is a narrative
    alias where BOTH names have their own rows and the bytecode already speaks
    each line under the correct index, so no relabelling is needed (and forcing
    one would wrongly merge two separate characters). Returns {clean_var_name:
    display_name}.
    """
    if not rows:
        return {}
    existing = {(r["name"] or "").lower() for r in rows if r["name"]}
    strings = find_strings(data)
    smap = {o: t for o, t in strings}
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    out = {}
    for q in fo:
        if not (data[q] == 0x1f and data[q + 1] == 0x2e):
            continue
        strs = []
        for j in range(fi[q] - 1, max(fi[q] - 5, -1), -1):
            qq = fo[j]
            if data[qq] == 0x1a:
                v = (data[qq + 1] << 8) | data[qq + 2]
                t = smap.get(v * 2 + STRING_BASE)
                if t and len(t) > 2:
                    strs.append(t)
                    if len(strs) == 2:
                        break
                else:
                    break          # a non-string push ends this rename's operands
            else:
                break              # any other opcode (e.g. a prior 1f2e) is a boundary
        strs.reverse()
        # Need a `$var` source followed by a plain display name.
        if len(strs) >= 2 and strs[0].startswith("$") \
                and not strs[1].startswith("$"):
            var_clean = _clean_name(strs[0])
            display = strs[1]
            # skip narrative aliases where the target is its own character row
            if display.lower() in existing:
                continue
            out[var_clean.lower()] = display
    return out


def _costume_spans(data: bytes, rows):
    """Detect sprite-costume overrides for a scene.

    A character can have a second resource row with the SAME name but a
    DIFFERENT sprite asset -- a costume (e.g. Dinah's nightclub disguise
    "DINAH" = 26005 vs "Dinah" = 2105; Matt's bad-haircut "Matt" = 26014 vs
    2285). The script arms the costume with a bare `1b(emo, costume_idx)` push
    placed immediately before a 0x42 SEP (a section boundary) -- not consumed
    by any display call. From that point until the next section marker, the
    character's dialogue portraits use the costume asset even though the lines
    still carry the base speaker index.

    `rows` are the shared resource rows (from scene 1; later scenes reuse them).
    Returns a list of {start, end, name, base_asset, costume_asset}. Only real
    swaps are reported (costume asset differs from the base), so duplicate rows
    that merely repeat the same asset are ignored.
    """
    if not rows:
        return []
    by_name = {}
    for r in rows:
        by_name.setdefault((r["name"] or "").lower(), []).append(
            (r["idx"], r["asset"]))
    costume = {}                                   # costume_idx -> (name, c, base)
    for name, entries in by_name.items():
        if len(entries) < 2 or not name:
            continue
        srt = sorted(entries)
        base_asset = srt[0][1]
        for idx, asset in srt[1:]:
            if asset != base_asset:
                costume[idx] = (name, asset, base_asset)
    if not costume:
        return []
    bc = max((o + len(t) for o, t in find_strings(data) if len(t) >= 12),
             default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    markers = [q for q in fo if data[q] == 0x22 and q + 3 < len(data)
               and data[q + 1] == 0x43 and data[q + 2] == 0x48
               and data[q + 3] == 0x4a]
    spans = []
    for q in fo:
        if data[q] == 0x1b and data[q + 2] in costume:
            nxt = fo[fi[q] + 1] if fi[q] + 1 < len(fo) else None
            if nxt is not None and data[nxt] == 0x42:      # bare 1b before a SEP
                name, c_asset, b_asset = costume[data[q + 2]]
                after = [m for m in markers if m > q]
                end = after[0] if after else len(data)
                spans.append({"start": q, "end": end, "name": name,
                              "base_asset": b_asset, "costume_asset": c_asset})
    return spans


def read_genders(data: bytes, cast):
    """{character_name: "M"|"F"|"O"} from the resource-table `type` field
    (1 = male, 2 = female, 3 = non-gendered object / non-human noise -- e.g.
    a Computer, an Offscreen Group). Names are resolved via name_ref. The
    engine uses this field to colour the speaker plate: male = blue,
    female = pink, object = a distinct colour reserved for non-human sounds.
    Narration (0x41 op / the "Narrator" custom speaker) is coloured yellow by
    the engine regardless of any table row -- it is a speaker-less card, not
    an entry in the resource table."""
    out = {}
    for r in _resource_rows(data):
        if r["name"] and r["typ"] in (1, 2, 3):
            out.setdefault(r["name"],
                           "M" if r["typ"] == 1 else
                           "F" if r["typ"] == 2 else "O")
    return out


def find_cards(strings, displayed_offsets, bc_start, cast, episode_title=None):
    """Locate the script's non-dialogue framing text.

    Returns (title_cards, minigame_words, end_cards, intro_narr, instr_prompts):
      * title_cards   - the title card pair: the head string that the episode's
                        metadata title ends with (e.g. metadata '02: As Time
                        Goes By' -> head string 'As Time Goes By'), plus the
                        string immediately after it (the on-screen subtitle).
                        Anchored by that cross-reference only -- no guessing.
      * minigame_words- a word-match minigame data block (questions + options)
                        that sits at the top of the script, read by the minigame
                        loop rather than by a display op
      * end_cards     - credit/end strings shown after the last spoken line
      * intro_narr    - scene-intro narration sentences before the first line
      * instr_prompts - minigame / task instruction cards ('...!')
    NOTE: the caller drops any of these whose text is ALSO displayed by a
    bytecode op in the same script (such strings are real in-flow lines, and
    synthesising them again would duplicate the content).
    """
    if not displayed_offsets:
        return [], [], [], [], []
    lo, hi = min(displayed_offsets), max(displayed_offsets)
    castset = set(cast)

    def is_system(t):
        return _is_system(t)

    # Content strings (everything past the cast/name table), minus system banks.
    content = [(off, t) for off, t in strings
               if 14 <= off and t not in castset and t != "Event"
               and not _NAME_VAR_RE.match(t) and not is_system(t)]

    # Everything before the first spoken line is framing: either a small set of
    # title cards, or a minigame/quiz data bank (questions + answer options +
    # feedback) the minigame loop reads directly. A block with question strings
    # and more than a handful of entries is the latter.
    pre = [(off, t) for off, t in content if off < lo]
    end = [t for off, t in content if off > hi and off < bc_start]
    n_questions = sum(1 for _, t in pre if t.rstrip().endswith("?"))
    if n_questions >= 5 and len(pre) >= 15:
        # A large bank of questions + answer options + feedback = a quiz / word-match
        # minigame the loop reads directly (not spoken lines).
        minigame = [t for _, t in pre]
        return [], minigame, [t for t in end if t not in set(minigame)], [], []

    # Title card: the pre string the metadata episode title ends with, plus the
    # string immediately after it as the on-screen subtitle (the same
    # title-then-subtitle pair layout the 0x08 title op uses).
    title = []
    if episode_title:
        for k, (_, t) in enumerate(pre):
            if len(t) >= 4 and episode_title.endswith(t):
                title = [t] + ([pre[k + 1][1]] if k + 1 < len(pre) else [])
                break

    intro, prompts = [], []
    for _, t in pre:
        if "|" in t or t[:1].islower() or len(t) < 6 or t in title:
            continue
        ts = t.rstrip()
        if ts.endswith("!"):
            prompts.append(t)            # minigame / task instruction card
        elif ts.endswith((".", "...")) or len(t) > 40:
            intro.append(t)              # scene-intro narration sentence
    skip = set(title) | set(intro) | set(prompts)
    return title, [], [t for t in end if t not in skip], intro, prompts


def _instr_len(data, p):
    """Length in bytes of the VM instruction at offset p.

    AUTHORITATIVE: an instruction consumes a 2-byte operand iff its opcode is in
    the OPERAND_OPCODES table decompiled from libshs09.so (int32 table at
    0x002593a8); otherwise it is a lone opcode byte. This replaces the old
    heuristic (which special-cased 0x1a/0x41/0x1b/0x2b/0x28=3, 0x1f=2, 0x42=4 and
    guessed everything else as 1) -- that model mis-sized many opcodes (e.g.
    0x29, 0x5c) and desynced inside data words. See the KiWi v2 core above.
    """
    return 3 if data[p] in _KIWI_OPERAND_OPCODES else 1


def _kiwi_code_start(data):
    """Byte offset where the instruction section begins, from the KiWi v2 header
    (not the old max-string-end guess). Header: 'kiwi' + version + previous_flag
    + [prev_data,prev_code if flag] + flag + 4 counts(main,gap,extra,instr); then
    the word table (main+extra 16-bit words). Returns None if not a v2 program."""
    if data[:4] != b"kiwi" or len(data) < 6 or data[4] != 2:
        return None
    prev_flag = data[5]
    pos = 6 + (4 if prev_flag else 0) + 1        # past prev-block + flag byte
    if pos + 8 > len(data):
        return None
    main = (data[pos] << 8) | data[pos + 1]
    extra = (data[pos + 4] << 8) | data[pos + 5]
    return pos + 8 + 2 * (main + extra)


def _instruction_offsets(data, bc_start=None):
    """Authoritative instruction start offsets for the whole code section.

    Locates the code section from the KiWi v2 header counts and walks it with the
    real operand table. `bc_start` is accepted for call-site compatibility but
    ignored: the true code start comes from the header. Falls back to a header-
    less walk only if the data is not a decodable KiWi program.
    """
    try:
        return [ins.byte_offset for ins in _kiwi_program(data).instructions]
    except _KiwiFormatError:
        start = _kiwi_code_start(data)
        if start is None:
            start = bc_start or 0
        offs, p, n = [], start, len(data)
        while p < n:
            offs.append(p)
            p += _instr_len(data, p)
        return offs


def _fold_instruction_offsets(data, bc):
    """Instruction offsets under the SECTION-DISPATCH operand model.

    The 16-bit operand carried by a 0x42 SEP (and listed in a scene's dispatch
    table) is an instruction count from `bc` to a section body's first
    instruction -- but counted in a slightly coarser unit than the base walk:
    the 1-byte argc that follows a builtin call (0x1f sel) is folded INTO the
    call, and a 0x5c switch row is 3 bytes (key+case). Reverse-engineered and
    byte-verified: every dispatch SEP operand resolves onto a real section body
    (a few instructions past its 22-marker) under this model.
    """
    offs, p, n = [], bc, len(data)
    while p < n:
        offs.append(p)
        op = data[p]
        if op in (0x1a, 0x41, 0x1b, 0x2b, 0x28, 0x5c):
            p += 3
        elif op == 0x42:
            p += 4
        elif op == 0x1f:
            p += 2
            if p < n and data[p] <= 0x08:      # fold the argc byte into the call
                p += 1
        else:
            p += 1
    return offs


def _section_markers(data, bc):
    """Offsets of every section-entry marker (22 43 48 4a) from bc onward."""
    n = len(data)
    return [p for p in range(bc, n - 4)
            if data[p] == 0x22 and data[p + 1] == 0x43
            and data[p + 2] == 0x48 and data[p + 3] == 0x4a]


def resolve_sep_target(data, bc, operand, fold_offs=None, fold_idx=None,
                       markers=None):
    """Resolve a 0x42 SEP / dispatch 16-bit operand to the section body it jumps
    to. The operand is an instruction count (fold model) from bc; the landing is
    a section body's first instruction, a few instructions past its 22-marker.

    Returns (target_offset, marker_offset) when the landing sits just after a
    section marker -- i.e. the SEP is a SECTION JUMP -- else None (an ordinary
    local branch separator whose operand is a near offset, not a section).
    """
    if fold_offs is None:
        fold_offs = _fold_instruction_offsets(data, bc)
    if fold_idx is None:
        fold_idx = {o: i for i, o in enumerate(fold_offs)}
    ti = fold_idx.get(bc, 0) + operand
    if not (0 <= ti < len(fold_offs)):
        return None
    target = fold_offs[ti]
    if markers is None:
        markers = _section_markers(data, bc)
    near = [m for m in markers if 0 <= target - m <= 16]
    if near:
        return target, max(near)
    return None


def _sim_walk_offsets(data, bc):
    """Instruction offsets for forward simulation. Same widths as _instr_len,
    except op5c is a full 3-byte instruction (opcode + 16-bit operand word) --
    the VM (FUN_00055ce0) consumes op5c's operand word, so it must advance 3
    bytes. Returns (offsets, index_map)."""
    offs, p, n = [], bc, len(data)
    while p < n:
        offs.append(p)
        if data[p] == 0x5c:
            p += 3
        else:
            p += _instr_len(data, p)
    return offs, {o: i for i, o in enumerate(offs)}


def _sim_jump_target(data, o, offs, idx):
    """Fold-free jump target of a 0x28/0x2b at o, as an offset in `offs`.
    Operand is a signed instruction count from the instruction after the jump."""
    operand = (data[o + 1] << 8) | data[o + 2]
    if operand >= 0x8000:
        operand -= 0x10000
    i = idx.get(o)
    if i is None:
        return None
    ti = i + 1 + operand
    return offs[ti] if 0 <= ti < len(offs) else None


def simulate_reachable(data, bc, start_off=None):
    """Forward-execute the bytecode from `start_off` (chunk entry when None),
    returning the set of reachable instruction offsets. Faithful to the VM's
    control flow (FUN_00055ce0):
      - 0x28 JMP: unconditional relative jump by instruction count.
      - 0x2b/0x2c/0x2d conditional branches: BOTH arms explored (the predicate
        is runtime data, so statically both are reachable).
      - 0x5c section dispatch: `if reg==(operand&0xff): PC=(PC-1)+(operand>>8)`.
        The register is data-dependent, so BOTH the matched-case jump (distance
        = high operand byte, in instructions) and the fall-through are explored.
      - 1f0a goto-scene: leaves the chunk; the path ends.
    This is the byte-derived equivalent of the engine's section-resume: a
    section body is reachable iff execution can land its PC there."""
    offs, idx = _sim_walk_offsets(data, bc)
    n = len(offs)
    if start_off is None:
        start = 0
    else:
        start = idx.get(start_off)
        if start is None:
            start = next((i for i, o in enumerate(offs) if o >= start_off), 0)
    seen, stack = set(), [start]
    while stack:
        i = stack.pop()
        if i is None or i < 0 or i >= n or i in seen:
            continue
        seen.add(i)
        o = offs[i]
        b = data[o]
        if b == 0x28:                                   # JMP
            t = _sim_jump_target(data, o, offs, idx)
            if t is not None:
                stack.append(idx[t])
            continue
        if b in (0x2b, 0x2c, 0x2d):                     # conditional: both arms
            t = _sim_jump_target(data, o, offs, idx)
            if t is not None:
                stack.append(idx[t])
            stack.append(i + 1)
            continue
        if b == 0x5c:                                   # section dispatch
            ti = i + data[o + 1]                        # (PC-1)+(operand>>8)
            if 0 <= ti < n:
                stack.append(ti)
            stack.append(i + 1)
            continue
        if b == 0x1f and o + 1 < len(data) and data[o + 1] == 0x0a:
            continue                                    # goto-scene: leaves chunk
        stack.append(i + 1)
    return {offs[i] for i in seen}


def resolve_scene_resume_sections(data, bc):
    """Byte-derive section bodies that are ONLY reachable as a section-register
    RESUME after a `goto scene` round-trip (the engine re-enters the chunk at a
    section head, restoring the saved PC). These are exactly the section markers
    (`22 43 48 4a`) that: (a) forward simulation from the chunk entry cannot
    reach, but (b) sit immediately after a `goto-scene` (1f0a) -- so control left
    the chunk right before them and only a resume can land there.

    Returns a list of {"marker": marker_off, "resume_at": body_off,
    "after_goto": goto_off} -- the goto whose return resumes at that section."""
    entry_reachable = simulate_reachable(data, bc)
    markers = _section_markers(data, bc)
    io = _instruction_offsets(data, bc)
    io_set = set(io)
    out = []
    for m in markers:
        # section body begins a few instructions past the 22-marker
        body = m + 4
        while body < len(data) and body not in io_set:
            body += 1
        if body in entry_reachable:
            continue                        # already reachable by normal flow
        # find the nearest preceding goto-scene (1f0a); the marker must sit just
        # after it (control left the chunk, so only a resume reaches this body)
        prev_goto = None
        for o in io:
            if o >= m:
                break
            if data[o] == 0x1f and o + 1 < len(data) and data[o + 1] == 0x0a:
                prev_goto = o
        if prev_goto is None:
            continue
        # the goto must be close (this marker is its resume landing, not a distant one)
        gap = sum(1 for o in io if prev_goto < o < body)
        if gap > 12:
            continue
        out.append({"marker": m, "resume_at": body, "after_goto": prev_goto})
    return out


def _resolve_degenerate_exits(data, bc):
    """Byte-derive sentinel-SEP section terminators -> chapter convergence.

    A chapter that dispatches its sections through a 1f12 reads a DISPATCH REGISTER
    just before the dispatch (`push <var_id> ; 1f2d ; ... ; 1f12`) and routes via a
    table of 0x5c rows + 0x42 SEPs. The final SEP -- the one sitting immediately
    before the 1f12 -- carries the chapter EXIT SENTINEL operand. Section bodies
    then end with the repeated terminator `SEP op=<sentinel> ; 28 A ; 28 B`: a
    register switch where the SEP is the convergence/exit case and the two 0x28
    jumps are alternate handlers for other register values.

    A terminator's 28 pair is dead code when the section body containing it does
    NOT write the dispatch register: the register holds the same value at the SEP
    that it had when the chapter entered the section's case, which is precisely the
    value that selects the SEP itself (so neither 28 fires). In that case the
    operative exit is the SEP, and the section converges. Two empirical sub-shapes
    confirm this:
      * degenerate pair (A == B) -- the 28s would jump to the same place anyway, a
        compiler no-op (e.g. ATGB junior's loyal-to-Sophie branch).
      * malformed pair -- one or both 28s land mid-instruction (e.g. ATGB junior's
        Paula/Taylor 28 alt-targets fall on the second byte of a `push 2005`).
    Both reduce to the same rule: if the body doesn't touch the dispatch register,
    the SEP is the operative exit. A terminator in a body that DOES write the
    dispatch register is a real conditional fork and is left to the jump resolver.

    Returns {terminator_offset: convergence_offset}; empty for scenes without this
    dispatch shape (so it is a no-op on episodes/scenes that don't use it).
    """
    n = len(data)
    disp = next((p for p in range(bc, n - 1)
                 if data[p] == 0x1f and data[p + 1] == 0x12), None)
    if disp is None:
        return {}

    # Dispatch register: the variable id pushed and then read by 1f2d immediately
    # before the dispatch's case-table. Walk backward from the 1f12 looking for the
    # last `push <id> ; 1f2d` pair before any 0x42 SEP rows of the table itself.
    dispatch_var = None
    for q in range(disp - 1, bc, -1):
        if data[q] == 0x1a and q + 5 < n \
                and data[q + 3] == 0x1f and data[q + 4] == 0x2d:
            dispatch_var = (data[q + 1] << 8) | data[q + 2]
            break

    sentinel = disp_sep = None
    for q in range(disp - 4, bc - 1, -1):            # the SEP just before 1f12
        if data[q] == 0x42 and q + 4 <= disp:
            sentinel = (data[q + 2] << 8) | data[q + 3]
            disp_sep = q
            break
    if sentinel is None:
        return {}
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    markers = _section_markers(data, bc)

    def jtgt(joff):
        op = (data[joff + 1] << 8) | data[joff + 2]
        i = fi.get(joff)
        return fo[i + op] if i is not None and 0 <= i + op < len(fo) else None

    # Convergence = the section body that performs the chapter's outgoing scene
    # goto (1f 0a). Take the marker preceding the last such goto.
    conv = None
    for p in range(bc, n - 1):
        if data[p] == 0x1f and data[p + 1] == 0x0a:
            pre = [m for m in markers if m <= p]
            if pre:
                conv = max(pre)
    if conv is None:
        return {}

    def section_of(off):
        """[start, end) of the section marker pair containing offset off, or None."""
        pre = [m for m in markers if m <= off]
        if not pre:
            return None
        start = max(pre)
        post = [m for m in markers if m > off]
        return (start, min(post) if post else n)

    def body_writes_dispatch_var(sec_start, term_off):
        """True if any `push <dispatch_var> ; ... ; 1f 2c` write appears in the
        section body from sec_start up to the terminator. Scans for the push then
        looks forward (within a short window, not crossing a 0x42 SEP) for the
        write. Conservative: false negatives just mean we apply the exit rule.
        """
        if dispatch_var is None:
            return False
        for p in range(sec_start, term_off):
            if data[p] != 0x1a:
                continue
            if ((data[p + 1] << 8) | data[p + 2]) != dispatch_var:
                continue
            # Scan forward for a 1f2c write; stop at a 0x42 SEP (control boundary).
            for q in range(p + 3, min(p + 24, term_off)):
                if data[q] == 0x42:
                    break
                if data[q] == 0x1f and data[q + 1] == 0x2c:
                    return True
        return False

    out = {}
    for p in range(bc, n - 10):
        if data[p] != 0x42 or data[p + 4] != 0x28 or data[p + 7] != 0x28:
            continue
        sop = (data[p + 2] << 8) | data[p + 3]
        if sop != sentinel or p == disp_sep:
            continue                                  # not a sentinel terminator
        sec = section_of(p)
        if sec is None or sec[0] == conv:
            continue                                  # terminator IN convergence: skip
        if body_writes_dispatch_var(sec[0], p):
            continue                                  # real conditional fork; leave alone
        out[p] = conv
    return out


def resolve_linear_sep_chains(data, bc):
    """Byte-derive consecutive sections that play LINEARLY via a bare SEP.

    Most section boundaries are a `42 <op> hi lo` SEP followed by a `22 43 48 4a`
    section marker. When that SEP does NOT encode a dispatch/section jump (its
    operand doesn't resolve onto another section body) and the section it closes
    contains no `goto_scene` (1f 0a) and no forward jump that leaves the section,
    the VM's SEP handler simply advances the PC (FUN_00055ce0 case 0x42: PC+1),
    so execution falls straight through into the NEXT section. These sequential
    story beats -- e.g. Magic School's intro flowing into "Chapter 2 / First
    Days" and the spell-casting chapters -- have no explicit jump wiring them, so
    the segmenter leaves the following section orphaned.

    Returns a list of (section_body_offset, next_section_body_offset): the offset
    of the section that ends in the bare SEP, paired with the body offset of the
    section it falls through into. The caller resolves each to a segment and adds
    a `next` edge -- but only when the target is otherwise orphaned, so episodes
    whose sections are already wired by dispatch/branch edges are untouched.
    """
    if data[:4] != b"kiwi":
        return []
    io = _instruction_offsets(data, bc)
    io_set = set(io)
    markers = sorted(_section_markers(data, bc))
    if len(markers) < 2:
        return []
    fold_offs = _fold_instruction_offsets(data, bc)
    fold_idx = {o: i for i, o in enumerate(fold_offs)}
    n = len(data)

    def body_of(marker):
        # a section body begins a few instructions past its 22-marker (the
        # 22 43 48 4a run + any title/background setup); use the marker+4 as the
        # nominal body start, which the segment mapper resolves to the real
        # first emitted node.
        return marker + 4

    out = []
    for i in range(len(markers) - 1):
        m, nm = markers[i], markers[i + 1]
        seg_ins = [o for o in io if m < o < nm]
        if not seg_ins:
            continue
        # a goto_scene leaves the chunk -> not a linear fall-through
        if any(data[o] == 0x1f and o + 1 < n and data[o + 1] == 0x0a
               for o in seg_ins):
            continue
        # a forward jump whose target is at/after the next marker leaves the
        # section (its own edge already carries the flow)
        leaves = False
        for o in seg_ins:
            if data[o] in (0x28, 0x2b):
                t = resolve_jump_target(data, o, bc_start=bc)
                if t is not None and t >= nm:
                    leaves = True
                    break
        if leaves:
            continue
        # the section must terminate in a SEP sitting just before the marker,
        # and that SEP must be BARE (not a dispatch/section jump)
        sep = None
        for o in reversed(seg_ins):
            if data[o] == 0x42:
                sep = o
                break
            # allow only trailing display/return cleanup ops after the SEP-free
            # tail; a control op other than SEP means this isn't a clean linear end
            if data[o] in (0x28, 0x2b, 0x5c):
                break
        if sep is None:
            continue
        operand = (data[sep + 2] << 8) | data[sep + 3] if sep + 3 < n else 0
        if resolve_sep_target(data, bc, operand,
                              fold_offs=fold_offs, fold_idx=fold_idx,
                              markers=markers) is not None:
            continue                # SEP is a real section jump, already wired
        out.append((body_of(m), body_of(nm)))
    return out


def _resolve_section_transitions(data, bc):
    """Section-end SEP transitions: a LONE `42 29 hi lo` SEP that sits
    immediately before a section marker (the 4-byte `22 43 48 4a` pattern, or
    the end of bytecode) is a one-way jump to another section's body. Its
    operand is a CASE LABEL, not a relative jump count -- it matches one of
    the dispatch-table SEP operands, and the engine routes execution to the
    case body that the dispatch table assigns to that label.

    So all SEPs with operand N converge on the same target regardless of
    where the SEP physically sits: the target is the dispatch-table SEP's
    own resolved jump destination for operand N. For sentinel operands (the
    dispatch's exit case, whose dispatch-SEP relative-jump runs off the end),
    the target is the chapter convergence section.

    This pattern shows up at the end of a section body that doesn't end with
    a `SEP;28;28` switch -- e.g. ATGB sophomore's Sophie-press section ends
    with `SEP op=848` -> the chapter's case-3 body; junior's lose-Business
    arm ends with `SEP op=817` -> the Flying-solo (lonely Travis) section.
    Returns {term_off: target_off}.
    """
    n = len(data)
    disp = next((p for p in range(bc, n - 1)
                 if data[p] == 0x1f and data[p + 1] == 0x12), None)
    if disp is None:
        return {}
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    markers = _section_markers(data, bc)

    def jtgt(joff, op):
        i = fi.get(joff)
        if i is None or i + op >= len(fo) or i + op < 0:
            return None
        return fo[i + op]

    # Walk back from the 1f12 to collect dispatch-table SEPs and their case
    # body targets. The dispatch table is the run of SEPs ending at the 1f12;
    # SEPs whose relative jump runs off the end use the sentinel operand and
    # represent the chapter exit case. The table contains variable-length
    # instructions (SEPs are 4 bytes, jumps are 3 bytes), so walk by fold-
    # instruction index rather than by fixed stride.
    case_targets = {}                                   # operand -> case body offset
    sentinel = None
    disp_sep_offsets = set()
    disp_i = fi.get(disp)
    if disp_i is None:
        return {}
    j = disp_i - 1
    while j >= 0:
        q = fo[j]
        op0 = data[q]
        if op0 == 0x42 and q + 1 < n and data[q + 1] == 0x29:
            op = (data[q + 2] << 8) | data[q + 3]
            disp_sep_offsets.add(q)
            t = jtgt(q, op)
            if t is None:
                if sentinel is None:
                    sentinel = op
            else:
                case_targets[op] = t
        elif op0 == 0x28:
            pass                                        # case-jump slot, keep walking
        else:
            break                                       # left the dispatch table
        j -= 1
    if not case_targets and sentinel is None:
        return {}

    # Convergence = section marker preceding the chapter's last 1f0a outgoing
    # goto. Used as the target for sentinel-operand transitions.
    conv = None
    for p in range(bc, n - 1):
        if data[p] == 0x1f and data[p + 1] == 0x0a:
            pre = [m for m in markers if m <= p]
            if pre:
                conv = max(pre)

    out = {}
    for q in fo:
        if data[q] != 0x42 or q + 8 > n or data[q + 1] != 0x29:
            continue
        if q in disp_sep_offsets:
            continue                                    # the dispatch table itself
        # Lone: NOT followed by `28 ... 28 ...` (the SEP;28;28 switch shape).
        if data[q + 4] == 0x28 and data[q + 7] == 0x28:
            continue
        # Section-end: immediately followed by the 4-byte section marker
        # `22 43 48 4a`. (If a future format ever lets a section-end SEP sit
        # at the very end of bytecode with nothing after, that case would need
        # an explicit handler -- not seen in any episode in the validation set.)
        if not (data[q + 4] == 0x22 and data[q + 5] == 0x43
                and data[q + 6] == 0x48 and data[q + 7] == 0x4a):
            continue
        op = (data[q + 2] << 8) | data[q + 3]
        if op == sentinel and conv is not None:
            out[q] = conv
        elif op in case_targets:
            out[q] = case_targets[op]
    return out


def resolve_jump_target(data, jump_off, instr_offsets=None, bc_start=None):
    """Resolve a branch's destination BYTE offset (authoritative).

    The operand is interpreted in the VM's logical-PC space and mapped back to a
    byte offset: 0x28/0x2a/0x2b/0x2c/0x2d/0x2e -> pc+operand; 0x29 -> the operand
    as an absolute PC; 0x5c -> pc+(operand>>8). A target exactly one past the
    last instruction maps to end-of-code. This replaced an earlier rule that
    treated the operand as a signed instruction count stepped through a mis-sized
    instruction walk, which landed off-boundary on essentially every branch.
    """
    try:
        prog = _kiwi_program(data)
    except _KiwiFormatError:
        prog = None
    if prog is not None:
        by_off = {ins.byte_offset: ins for ins in prog.instructions}
        ins = by_off.get(jump_off)
        if ins is None:
            return None
        target_pc = ins.branch_target()
        if target_pc is None:
            return None
        pc2b = {i.pc: i.byte_offset for i in prog.instructions}
        if target_pc in pc2b:
            return pc2b[target_pc]
        maxpc = max((i.pc for i in prog.instructions), default=-1)
        return len(data) if target_pc == maxpc + 1 else None

    # Header-less fallback: same logical-PC model computed from the raw walk.
    offs = _instruction_offsets(data, bc_start)
    pc_of = {o: i for i, o in enumerate(offs)}
    end_pc = len(offs)
    if jump_off + 2 >= len(data) or jump_off not in pc_of:
        return None
    op = data[jump_off]
    operand = (data[jump_off + 1] << 8) | data[jump_off + 2]
    pc = pc_of[jump_off]
    if op == 0x29:
        target_pc = operand
    elif op in (0x28, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e):
        target_pc = (pc + operand) & 0xFFFF
    elif op == 0x5c:
        target_pc = (pc + (operand >> 8)) & 0xFFFF
    else:
        return None
    if 0 <= target_pc < end_pc:
        return offs[target_pc]
    return len(data) if target_pc == end_pc else None


def disassemble_script(data, cast=None, resolve_targets=True):
    """Produce an annotated, human-readable bytecode listing for a .kiw script.

    Walks the bytecode the same way the decoder does and emits one line per
    instruction: its offset, the raw opcode, a mnemonic, decoded operands, and --
    where it can be inferred -- a plain-language note (the string a text push
    resolves to, the variable a read/write touches, a jump's resolved target and
    the first line of text there, an arithmetic op's running expression, etc.).

    Returns a list of text lines. This is a *diagnostic view*: it interprets the
    bytes with the same opcode vocabulary the decoder uses, but makes no routing
    decisions -- it's meant for inspecting exactly what the VM sees (e.g. the
    `push 16 ; read var2000 ; SUB ; push 5 ; MUL` score-scaling sequence).
    """
    if data[:4] != b"kiwi":
        return ["(not a kiwi script)"]
    strings = find_strings(data)
    ss = {o: t for o, t in strings}
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    n = len(data)
    cast = cast or []

    # builtin selector names (the 0x1f <sel> calls)
    SELN = {
        0x00: "STATUS", 0x01: "CHOICE", 0x05: "SET_SPEAKER", 0x08: "TITLE",
        0x0a: "SCENE_GOTO", 0x0b: "SET_BG", 0x0d: "SAY", 0x0f: "CUSTOM_SPEAKER",
        0x12: "DISPATCH", 0x1b: "RANDOM", 0x23: "SET_BG_PNG", 0x2c: "VAR_WRITE",
        0x2d: "VAR_READ", 0x34: "PLACE_SPRITE", 0x41: "NARRATION",
        0x46: "BUILD_WORD", 0x47: "PICK_WORD", 0x4a: "SET_POV", 0x4f: "PLAY_SFX",
        0x50: "SET_MUSIC", 0x51: "MINIGAME_BYID", 0x52: "STOP_MUSIC",
        0x58: "SCORE_FEEDBACK", 0x63: "CHECKPOINT",
    }
    # standalone (non-0x1f) arithmetic / compare opcodes
    ARITH = {0x50: "ADD", 0x51: "SUB", 0x52: "MUL", 0x53: "DIV"}
    # Comparison mnemonics for the diagnostic listing. Operand order is
    # `var CMP const` (var pushed first, const on top). Verified against the
    # engine's frame VM (FUN_00055ce0 in libshs09.so): each op computes a
    # relation between the two stack words --
    #   0x0a var == const   0x0b var != const   0x0c var > const
    #   0x0d var >= const    0x0e var < const     0x0f var <= const
    CMP = {0x0a: "==", 0x0b: "!=", 0x0c: ">", 0x0d: ">=", 0x0e: "<", 0x0f: "<="}

    def txt_of(ref):
        return ss.get(ref * 2 + STRING_BASE)

    def first_text(off, span=90):
        for q in range(off, min(off + span, n)):
            if q in fi and data[q] == 0x1a:
                v = (data[q + 1] << 8) | data[q + 2]
                t = txt_of(v)
                if t and len(t) > 8:
                    return t[:52]
        return None

    out = []
    # a small symbolic operand stack, so reads and arithmetic can be annotated
    # with the running expression they build (purely cosmetic). Each entry is a
    # (display_string, is_literal_number) pair.
    expr = []

    def vname(v):
        # variables live at >= 2000 in this VM; below that a pushed number is a
        # literal constant, a string ref, or a small selector arg
        return "var%d" % v if v >= 2000 else str(v)

    for idx, p in enumerate(fo):
        op = data[p]
        note = ""
        if op in (0x1a, 0x41):                       # push uint16be
            v = (data[p + 1] << 8) | data[p + 2]
            t = txt_of(v)
            mnem = "push %d" % v
            if t and _displayable(t) and len(t) >= 3:
                note = '"%s"' % (t[:46])
            expr.append(str(v))
        elif op == 0x1b:                             # push (emotion, speaker) pair
            emo, spk = data[p + 1], data[p + 2]
            mnem = "push_pair (emo=%d, spk=%d)" % (emo, spk)
            if 0 <= spk < len(cast) and cast[spk]:
                note = "speaker=%s" % cast[spk]
            expr.append(None)
        elif op == 0x1f:                             # builtin call
            sel = data[p + 1]
            name = SELN.get(sel, "sel_0x%02x" % sel)
            mnem = "1f%02x %s" % (sel, name)
            if sel == VAR_READ and expr:
                # the just-pushed number is a variable id; reflect the read in the
                # expression stack so following arithmetic reads correctly
                top = expr[-1]
                if top is not None and top.isdigit():
                    vn = vname(int(top))
                    expr[-1] = vn
                    note = "read %s" % vn
            elif sel == VAR_WRITE and len(expr) >= 2:
                a, b = expr[-2], expr[-1]
                if a is not None and a.isdigit():
                    note = "write %s = %s" % (vname(int(a)), b)
                expr = []
            elif sel == 0x0a and expr:               # scene goto
                note = "-> scene %s" % (expr[-1])
                expr = []
            else:
                expr = []                            # a call consumes operands
        elif op in ARITH:                            # standalone arithmetic
            mnem = "%s (0x%02x)" % (ARITH[op], op)
            if len(expr) >= 2:
                b = expr.pop()
                a = expr.pop()
                sym = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}[ARITH[op]]
                combined = "(%s %s %s)" % (a, sym, b)
                expr.append(combined)
                note = "-> %s %s %s" % (a, sym, b)
        elif op in CMP:                              # comparison
            mnem = "%s (0x%02x)" % (CMP[op], op)
            if len(expr) >= 2:
                note = "-> %s %s %s" % (expr[-2], CMP[op], expr[-1])
        elif op in (0x2b, 0x28):                     # conditional / uncond jump
            tgt = resolve_jump_target(data, p, bc_start=bc) if resolve_targets \
                else None
            kind = "jump_if_false" if op == 0x2b else "jump"
            mnem = "%s (0x%02x)" % (kind, op)
            if tgt is not None:
                ft = first_text(tgt)
                note = "-> @%d%s" % (tgt, ' "%s"' % ft if ft else "")
            expr = []
        elif op == 0x42:                             # SEP / section jump
            spop = (data[p + 2] << 8) | data[p + 3]
            mnem = "SEP op=%d" % spop
            _res = resolve_sep_target(data, bc, spop, fold_offs=fo,
                                      fold_idx=fi) if resolve_targets else None
            tgt = _res[0] if isinstance(_res, tuple) else _res
            if tgt is not None:
                ft = first_text(tgt)
                note = "-> @%d%s" % (tgt, ' "%s"' % ft if ft else "")
            expr = []
        elif op in (0x5a, 0x5b):                     # resource-row / small push
            mnem = "row%d (0x%02x)" % (op - 0x5a, op)
            expr.append(str(op - 0x5a))
        elif op == 0x3e:
            mnem = "SET_GOAL (0x3e)"
            if expr:
                note = "display value = %s" % expr[-1]
        elif op == 0x21:
            mnem = "fmt (0x21)"
        elif op == 0x11:
            mnem = "NOT (0x11)"
        elif op == 0x22 and p + 3 < n and data[p + 1] == 0x43 \
                and data[p + 2] == 0x48 and data[p + 3] == 0x4a:
            mnem = "-- SECTION MARKER --"
        else:
            mnem = "op 0x%02x" % op

        line = "@%-6d %s" % (p, mnem)
        if note:
            line += "   ; %s" % note
        out.append(line)
    return out


def _scan_sep63_terminals(data):
    """Return the offsets of `SEP op=63` "return to caller" jumps in a script.

    Making Some Dough's several alternate endings each conclude with a 0x42 SEP
    whose 16-bit operand is 63 -- a return-to-the-section-dispatcher jump whose
    target lands in a data region and does not resolve to an instruction. The
    decoder therefore cannot follow it and instead falls the ending through into
    the next section in file order (wrong: the endings are independent, separated
    by `22` markers). Surfacing these offsets lets segmentation re-point each such
    ending at the shared end-of-episode survey rather than leaking into a
    different ending. Byte-derived; empty for scripts without this pattern.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    out = []
    for q in fo:
        if data[q] == 0x42 and q + 3 < len(data):
            operand = (data[q + 2] << 8) | data[q + 3]
            if operand == 63:
                out.append(q)
    return out


def scan_control_flow(data):
    """Re-walk a .kiw script's bytecode and record its raw branch skeleton.

    This does NOT interpret the VM's compare semantics (which are unverified); it
    only surfaces, in bytecode order, the observable control instructions so an
    engine can reconstruct ordering:
      * branches  -- each conditional/relative jump (0x2b / 0x28): its offset, the
                     opcode, the raw 16-bit operand, and the immediate values still
                     pending on the operand stack (the literals being tested, e.g.
                     the day thresholds 3/6/9/10).
      * var_reads -- each `push <id> ; 1f 2d` (a probable variable read): offset and
                     the id read (e.g. 2000/2001/2002 -- candidate dollars/day/state
                     counters). Labelled "probable" because 0x2d is not confirmed.
    Offsets match each node's `offset` field, so a branch can be tied to the nodes
    around it. Each branch now also carries a resolved `target` offset: the VM's
    jump operand is a signed instruction count (see resolve_jump_target), which
    lands cleanly on instruction boundaries for every jump in this episode.
    """
    if data[:4] != b"kiwi":
        return {"branches": [], "var_reads": []}
    strings = find_strings(data)
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    instr_offsets = _instruction_offsets(data, bc_start)
    p, n = bc_start, len(data)
    pushes = []
    branches, var_reads = [], []
    while p < n - 1:
        op = data[p]
        if op in (0x1a, 0x41) and p + 2 < n:
            pushes.append((data[p + 1] << 8) | data[p + 2]); p += 3; continue
        if op == 0x1b and p + 2 < n:
            p += 3; continue            # pair bytes are not stack immediates here
        if op == 0x1f and p + 1 < n:
            sel = data[p + 1]
            if sel == 0x2d and pushes:   # probable variable read
                var_reads.append({"offset": p, "value": pushes[-1]})
            pushes = []; p += 2; continue
        if op in (0x2b, 0x28) and p + 2 < n:
            branches.append({"offset": p, "op": "0x%02x" % op,
                             "operand": (data[p + 1] << 8) | data[p + 2],
                             "target": resolve_jump_target(data, p, instr_offsets),
                             "compared": list(pushes)})
            pushes = []; p += 3; continue
        if op == 0x42 and p + 3 < n:
            pushes = []; p += 4; continue
        p += 1
    # The "exits" map is a unified `term_off -> target_off` table; the consuming
    # pass routes each source segment to its target the same way regardless of
    # whether the SEP is a sentinel terminator (SEP;28;28 at section end going
    # to convergence) or a section-end transition (lone SEP at section end going
    # to another section's case body).
    exits = dict(_resolve_degenerate_exits(data, bc_start))
    exits.update(_resolve_section_transitions(data, bc_start))
    return {"branches": branches, "var_reads": var_reads,
            "degenerate_exits": exits}


def decode_script(data: bytes, cast=None, episode_title=None,
                  script_ids=None, image_ids=None, audio_ids=None,
                  sprites_by_idx=None,
                  costume_rows=None, name_renames=None, text_subs=None):
    """Decode a .kiw script into a list of Line objects.

    `script_ids` / `image_ids` are the archive directory's real chunk ids:
    a 0x0a goto operand must be one of the archive's script chunk ids, and a
    0x23 custom-background operand must be one of its image chunk ids -- the
    archive itself defines the valid values, so no numeric range is assumed.
    """
    if data[:4] != b"kiwi":
        raise ValueError("Not a kiwi (.kiw) script")
    strings = normalize_strings(find_strings(data))
    smap = {o: t for o, t in strings}
    if cast is None:
        cast = read_cast(data)
    # Display text never resolves into the cast/name-table region: those
    # strings are names. (A stray operand resolving there produced junk
    # narration lines like a lone cast name.)
    _evt = data.find(b"Event\x00")
    content_min = (_evt + 6) if _evt >= 0 else CAST_START

    # Bytecode starts after the last sizeable string. This boundary is a
    # heuristic: the 10 kiwi header bytes were tested against every script of
    # four episodes and no field consistently encodes this offset, so the
    # string/bytecode split cannot (yet) be read from the header.
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    resources = read_resources(data, smap, bc_start, cast)  # name -> sprite base asset

    # RELATIONSHIP CAPTIONS: a section subtitle of the form "Dating <Name>..."
    # (where Name is a cast member) marks who the player is dating in that section.
    # Reminder/monologue lines push this caption's ref where a speaker index would
    # sit; the implicit speaker of such a line is the named partner. Map ref->name.
    caption_partner = {}
    for o, t in strings:
        m = re.match(r"^\s*Dating\s+([A-Za-z][\w'\-]*)", t)
        if m:
            nm = next((c for c in cast if c and c.split()[0] == m.group(1)), None)
            if nm is None and m.group(1) in cast:
                nm = m.group(1)
            if nm:
                caption_partner[(o - STRING_BASE) // 2] = nm

    lines = []
    emitted_titles = set()   # title strings shown via the 0x08 opcode
    p, n = bc_start, len(data)
    # A title card immediately preceded by a 0x42 SEP (with no content line
    # between) opens a new dispatch-section body: the SEP is the section
    # boundary, so the prior section ENDS there and does not fall through into
    # this one. Tracked only for scenes that actually have a 0x12 section
    # dispatch (otherwise title cards are ordinary in-flow framing).
    has_dispatch = b"\x1f\x12" in data[bc_start:]
    # For resolving a 0x42 SEP's 16-bit operand to the section body it jumps to
    # (the breakup section's terminator jumps to the loser section this way).
    _fold_offs = _fold_instruction_offsets(data, bc_start)
    _fold_idx = {o: i for i, o in enumerate(_fold_offs)}
    _markers = _section_markers(data, bc_start)
    sep_section_target = None   # if the last SEP (post-dispatch) was a section
                                # jump, the section-body offset it targets
    sep_pending = False
    seen_dispatch = False   # set once the 0x12 section dispatch is passed; only
                            # title cards AFTER it open dispatch-section bodies
                            # (the pre-dispatch episode/intro title is not one)
    pushes = []  # operands accumulated since the previous display instruction
    last_var_read = None  # the variable id read by the most recent 1f 2d call
    saw_add = False       # a 0x50 add byte appeared since the last call (the
                          # read-modify-write marker: read var, push delta, add)
    computed_write = False  # a LOAD/SLOT op fed the value about to be written by a
                            # 1f2c: the value is runtime-computed (a score-table
                            # lookup), not a static constant, so no var node is emitted
    slot_display = False    # a 0x15 STORE consumed operands into a quiz/array slot;
                            # a following 1f00/1f41 renders a runtime "%s" from the
                            # slot, so its stored-string operand is data, not text
    last_spk = None  # most recent speaker index (from any 1b pair); a speaker-less
                     # 0x0d line is a continuation spoken by this "active" speaker
    marker_row = None  # 0x5a / 0x5b before a display call select the speaker by
                       # RESOURCE-TABLE ROW: 0x5a -> row 0, 0x5b -> row 1.
                       # (Byte-verified across episodes: in Making Some Dough
                       # rows 0 and 1 are both Kim, which is why the two markers
                       # were indistinguishable there; As Time Goes By's 0x5a
                       # lines are all Matt = row 0, The Tutors' 0x5b lines are
                       # Ben = row 1.)
    tchoice = None     # accumulator for a timed text choice being built across
                       # 1f02 (setup) / 1f03 (options) / 1f04 (resolve).
    while p < n - 1:
        op = data[p]
        if op in (0x5a, 0x5b):                               # push 0 / push 1
            # One opcode, two contexts: before a display call the pushed 0/1
            # selects the SPEAKER ROW; as a plain operand it is the value 0/1
            # (e.g. `push var ; 0x5b ; 1f 2c` = set var to 1). Tagged "m" so
            # text-candidate scans ignore it.
            marker_row = 0 if op == 0x5a else 1
            pushes.append(("m", marker_row))
            p += 1
            continue
        if op == 0x50:                                       # arithmetic ADD byte
            saw_add = True                                   # (NOT the 1f 50 music call)
            p += 1
            continue
        if op in (0x1a, 0x41) and p + 2 < n:                 # push uint16be
            pushes.append(("v", (data[p + 1] << 8) | data[p + 2]))
            p += 3
            continue
        if op == 0x1b and p + 2 < n:                          # push (emo, spk)
            pushes.append(("pair", data[p + 1], data[p + 2]))
            # The 2nd byte is normally a speaker index, but in COMPACT-FORM display
            # lines it is instead a TEXT reference (1b <emotion> <textref> ; 1f 0d,
            # with no separate speaker push). Using a text ref as the active
            # speaker corrupts the next speaker-less line -- and if the ref happens
            # to be below the cast size it silently mis-attributes the current line
            # too (e.g. text ref 20 colliding with cast index 20). Only skip the
            # speaker update when this pair is ACTUALLY a compact display line: it
            # must be immediately followed by a display call (1f 0d / 1f 41) and
            # its 2nd byte must map to a real displayable content string. A pair
            # that is merely operands for some other call (e.g. 1f 08) keeps the
            # normal speaker-tracking behaviour, since its bytes are not text.
            _b2 = data[p + 2]
            _nxt = data[p + 3] if p + 3 < n else None
            _nxt2 = data[p + 4] if p + 4 < n else None
            _is_display = (_nxt == 0x1f and _nxt2 in (0x0d, 0x41))
            _b2_txt = smap.get(_b2 * 2 + STRING_BASE)
            _is_textref = _is_display and bool(_b2_txt) \
                and _b2 * 2 + STRING_BASE >= content_min \
                and _displayable(_b2_txt)
            if not _is_textref:
                last_spk = _b2
            p += 3
            continue
        if op == 0x1f and p + 1 < n:                          # call builtin <sel>
            sel = data[p + 1]
            if sel == 0x12:                                   # section dispatch passed
                seen_dispatch = True
            _before = len(lines)
            if sel == DISPLAY_DIALOGUE:                       # 0x0d -> spoken line
                lines.append(_emit(pushes, smap, cast, resources,
                                   narration=False, last_spk=last_spk,
                                   speaker_row=marker_row,
                                   sprites=sprites_by_idx,
                                   content_min=content_min,
                                   caption_partner=caption_partner))
                sep_pending = False   # content between a SEP and a later title
                                      # card means that title is not a boundary
            elif sel == SET_SPEAKER:                          # 0x05 -> stage setup
                # The operand pair names two resource rows (the characters put
                # on screen, e.g. (Kim, Andy)); it does NOT set the speaker --
                # speaker-less lines are resolved by the 0x5a/0x5b row markers.
                pass
            elif sel == CUSTOM_SPEAKER:                       # 0x0f -> one-off named speaker
                sp = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                      if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap]
                if len(sp) >= 2:
                    emo = next((pk[1] for pk in pushes if pk[0] == "pair"), None)
                    emo = None if emo in (None, 0xff) else emo
                    lines.append(Line(sp[-1], sp[0], emo))
                elif sp:
                    lines.append(Line(sp[0], None, None))
                else:
                    # PAIR-ENCODED FORM: a one-off named line whose text and speaker
                    # NAME are both packed into a 1b pair: 1b <textref> <nameref> ;
                    # [1b <flag> <..>] ; 1f 0f. The 1st byte is the text ref, the 2nd
                    # the speaker-name ref (e.g. "Narrator"). Used for the scene-intro
                    # voiceover lines. A "Narrator" name renders as narration.
                    for pk in pushes:
                        if pk[0] != "pair":
                            continue
                        toff = pk[1] * 2 + STRING_BASE
                        noff = pk[2] * 2 + STRING_BASE
                        txt = smap.get(toff)
                        nam = smap.get(noff)
                        if txt and len(txt) > 3 and _displayable(txt) and txt != nam:
                            spk = None if (not nam or nam == "Narrator") else nam
                            lines.append(Line(txt, spk, None, off=toff))
                            break
            elif sel == TITLE_CARD:                           # 0x08 -> title / intro card
                strs = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                        if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap
                        and len(smap[pk[1] * 2 + STRING_BASE]) >= 4]
                if strs:
                    # a title card pushes the title then (optionally) a subtitle;
                    # emit ONE node carrying both (subtitle in the sprite slot).
                    subtitle = strs[1] if len(strs) > 1 else None
                    _ss = has_dispatch and sep_pending and seen_dispatch
                    lines.append(Line(strs[0], "__CARD__", "title", sprite=subtitle,
                                      bc_off=p, section_start=_ss,
                                      section_jump=(sep_section_target if _ss
                                                    else None)))
                    for t in strs:
                        emitted_titles.add(t)
                sep_pending = False
                sep_section_target = None
            elif sel == DISPLAY_NARRATION:                    # 0x41 -> narration card
                lines.append(_emit(pushes, smap, cast, resources, narration=True,
                                   sprites=sprites_by_idx,
                                   content_min=content_min))
                sep_pending = False
            elif sel == DISPLAY_STATUS:                       # 0x00 -> status / HUD line
                txts = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                        if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap
                        and smap[pk[1] * 2 + STRING_BASE]]
                txts = [t for t in txts if t not in cast and len(t) > 6]
                if slot_display and txts and all(
                        t.strip().startswith("%s") for t in txts):
                    # a runtime quiz display: the visible text is a "%s" template
                    # filled from the array slot at play time; the stored strings
                    # (question/answer/feedback) were already consumed by STORE and
                    # are data, not a static status line. Emit nothing.
                    pass
                elif txts:
                    lines.append(Line(txts[-1], "__STATUS__", None))
                else:
                    ln = _emit(pushes, smap, cast, resources, narration=True,
                               last_spk=last_spk, sprites=sprites_by_idx,
                               content_min=content_min)
                    if ln and ln.text:            # end-screen results line (ref in 1b 1st byte)
                        lines.append(ln)
            elif sel == SET_BACKGROUND:                       # 0x0b -> background image
                # Operands are [layer_flag(0/1), base_asset(1000-1200), 0xffff].
                # The 0x5a/0x5b (0 or 1) before the asset is a LAYER/SLOT flag, not
                # an increment: the displayed background is the base asset id
                # itself. (Byte-confirmed: bases are clean library ids like 1111 /
                # 1036; the flag co-occurs with the same base under both values, and
                # ground-truth checks show base+flag over-counts by the flag.)
                base_i = None
                for i, pk in enumerate(pushes):
                    if pk[0] == "v" and 1000 <= pk[1] <= 1200:
                        base_i = i
                if base_i is not None:
                    lines.append(Line(None, "__BG__", ("standard", pushes[base_i][1])))
            elif sel == SET_BG_CUSTOM:                        # 0x23 -> packaged PNG background
                ids = [pk[1] for pk in pushes if pk[0] == "v" and pk[1] != 0xffff
                       and (pk[1] in image_ids if image_ids else True)]
                if ids:
                    # with archive validation there is one candidate; without,
                    # fall back to the largest (chunk ids sit above operands)
                    lines.append(Line(None, "__BG__", ("custom", ids[-1] if image_ids
                                                       else max(ids))))
            elif sel == SET_MUSIC:                            # 0x50 -> background music change
                # A track is either a GLOBAL library id -- the native engine
                # validates these against a fixed band (`param_2 - 0x2009U < 0x20`
                # in libshs09.so -> 8201..8232, and the base APK ships /8201.mp3
                # ../8225.mp3) -- or an EPISODE-PACKAGED MP3 chunk shipped in this
                # .exp (e.g. Swim Team Retreat 2's 26044..26050, Halloween Dance
                # Part 2's 26000..26005), exactly like packaged sprites/backgrounds.
                # The old `>= 0x2000` test was too loose (it let packaged PNG ids
                # through as music); band-only was too strict (it dropped the
                # packaged tracks).
                codes = [pk[1] for pk in pushes if pk[0] == "v"
                         and (MUSIC_ID_MIN <= pk[1] <= MUSIC_ID_MAX
                              or (audio_ids and pk[1] in audio_ids))]
                if codes:
                    lines.append(Line(None, "__MUSIC__", codes[-1]))
            elif sel == STOP_MUSIC:                           # 0x51 -> stop / fade out music
                lines.append(Line(None, "__MUSIC__", "stop"))
            elif sel == VIBRATE:                              # 0x52 -> haptic buzz
                lines.append(Line(None, "__VIBRATE__", None))
            elif sel == PRESENT_CHOICE:                       # 0x01 -> choice menu / QTE
                texts = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                         if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap]
                if not any("|" in t for t in texts):
                    # Some choices (e.g. the day-hub task menu) pack the prompt and the
                    # "a|b|c" option string into the bytes of 1b pairs instead of 1a
                    # pushes: 1b <optionsref> <..> ; 1b <..> <promptref> ; 1f 01.
                    for pk in pushes:
                        if pk[0] == "pair":
                            for b in (pk[1], pk[2]):
                                t = smap.get(b * 2 + STRING_BASE)
                                if t and len(t) > 3 and t not in texts:
                                    texts.append(t)
                opts = next((t for t in texts if "|" in t), None)
                if opts:                                      # story choice (a|b|c)
                    # Choice prompts like "What do you do?" / "What do you say?" are
                    # flagged by _is_system (so they don't leak into spoken dialogue),
                    # but on a choice they ARE the player-facing question. Prefer a
                    # non-system "?" line; otherwise fall back to a decision-prompt
                    # phrase, and only then to any non-system line. The "Make your
                    # choice!" banner stays excluded.
                    _prompt_phrases = ("what do you", "how do you respond",
                                       "what will you", "what should you")
                    nonopt = [t for t in texts if "|" not in t and not _is_system(t)]
                    decision = [t for t in texts if "|" not in t
                                and any(p in t.lower() for p in _prompt_phrases)]
                    prompt = (next((t for t in nonopt if t.rstrip().endswith("?")), None)
                              or (next((t for t in decision
                                        if t.rstrip().endswith("?")), None))
                              or (decision[0] if decision else None)
                              or (nonopt[0] if nonopt else None))
                    lines.append(Line(opts, "__CHOICE__", prompt))
                elif len(texts) >= 2 and all(t.rstrip().endswith(("!", "?")) for t in texts):
                    prompt = " ".join(t for t in texts if t.rstrip().endswith("?"))
                    options = [t for t in texts if not t.rstrip().endswith("?")]
                    packed = ([prompt] if prompt else []) + (["|".join(options)] if options else [])
                    lines.append(Line("§".join(packed), "__MINIGAME__", "action"))
                elif texts:                                   # lone prompt
                    lines.append(Line(texts[0], "__NOTIFY__", "score"))
            elif sel == TCHOICE_SETUP:                        # 0x02 -> timed choice begin
                # Text refs pushed: [prompt ("Make your choice!"), setup line,
                # first option]. A timer (ms) is pushed too. Distinguish the real
                # timed TEXT choice (has a countdown + will be followed by 1f03
                # options) from the word/action minigame reuse (0xffff sentinel,
                # no 1f03 options) by requiring a plausible timer value.
                txts = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                        if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap
                        and smap[pk[1] * 2 + STRING_BASE]]
                timer = next((pk[1] for pk in pushes
                              if pk[0] == "v" and 1000 <= pk[1] <= 60000
                              and (pk[1] * 2 + STRING_BASE) not in smap), None)
                if timer is not None and txts:
                    tchoice = {"timer_ms": timer, "prompt": None,
                               "setup": None, "options": []}
                    # prompt is a system banner ("Make your choice!"); the other
                    # text is the setup / lead-in line spoken by the wooer. The
                    # options come exclusively from the 1f03 calls that follow;
                    # 1f02 itself carries NO option.
                    sys_t = [t for t in txts if _is_system(t)]
                    body = [t for t in txts if not _is_system(t)]
                    if sys_t:
                        tchoice["prompt"] = sys_t[0]
                    if body:
                        tchoice["setup"] = body[-1]
            elif sel == TCHOICE_OPTION:                       # 0x03 -> add option
                if tchoice is not None:
                    ot = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                          if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap
                          and smap[pk[1] * 2 + STRING_BASE]]
                    if ot:
                        tchoice["options"].append(ot[-1])
            elif sel == TCHOICE_RESOLVE:                      # 0x04 -> resolve choice
                if tchoice is not None and tchoice["options"]:
                    if tchoice["setup"]:                      # emit the lead-in line
                        lines.append(_emit([("v", None)], smap, cast, resources,
                                           narration=False, last_spk=last_spk)
                                     if False else Line(tchoice["setup"], None, None))
                    packed = "§".join(
                        ([tchoice["prompt"]] if tchoice["prompt"] else [])
                        + ["|".join(tchoice["options"])])
                    lines.append(Line(packed, "__TCHOICE__", tchoice["timer_ms"]))
                    tchoice = None
            elif sel == SCORE_FEEDBACK:                       # 0x58 -> score / feedback banner
                texts = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                         if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap]
                if texts:
                    lines.append(Line(texts[-1], "__NOTIFY__", "score"))
            elif sel == VAR_READ:                             # 0x2d -> read variable
                vals = [pk[1] for pk in pushes if pk[0] in ("v", "m")]
                if vals:
                    last_var_read = vals[-1]
            elif sel == VAR_WRITE:                            # 0x2c -> write variable
                vals = [pk[1] for pk in pushes if pk[0] in ("v", "m")]
                if computed_write:
                    # value came from a LOAD/SLOT (score-table lookup); it is not a
                    # static constant, so emit no var node rather than a spurious
                    # set/add built from the leftover index / var-id operands.
                    pass
                elif saw_add and vals and last_var_read is not None:
                    # read-modify-write: read var, push delta, 0x50 add, write
                    delta = vals[-1] - 0x10000 if vals[-1] >= 0x8000 else vals[-1]
                    lines.append(Line(None, "__VAR__", ("add", last_var_read, delta)))
                elif len(vals) >= 2:
                    lines.append(Line(None, "__VAR__", ("set", vals[-2], vals[-1])))
            elif sel == PLAY_SFX:                             # 0x4f -> play sound effect
                # Engine-validated SFX band (`param_2 - 0x1f41U < 0x71`) = 8001..8113.
                vals = [pk[1] for pk in pushes
                        if pk[0] == "v" and SFX_ID_MIN <= pk[1] <= SFX_ID_MAX]
                if vals:
                    lines.append(Line(None, "__SFX__", vals[-1]))
            elif sel == SCENE_GOTO:                           # 0x0a -> jump to another scene
                tgt = [pk[1] for pk in pushes if pk[0] == "v"
                       and (pk[1] in script_ids if script_ids
                            else 25000 <= pk[1] < 26000)]
                if tgt:
                    lines.append(Line(None, "__GOTO__", tgt[-1]))
            elif sel == ACTION_MINIGAME:                      # 0x47 -> action / word minigame
                texts = [smap[pk[1] * 2 + STRING_BASE] for pk in pushes
                         if pk[0] == "v" and (pk[1] * 2 + STRING_BASE) in smap]
                groups = [t for t in texts if "|" in t]
                if groups:
                    prompts = [t for t in texts if "|" not in t]
                    # Per-round WIN THRESHOLD: some episodes emit a score test
                    # immediately after the 1f47 call in the form
                    #     [<opaque bytes>...] 5f 3f 1a <lo> <hi> 0d 2b <skip>
                    # where the pushed 16-bit value is the threshold the runtime
                    # compares this round's score against. Used by Making Some
                    # Dough's bakery chain -- each round has its own N. Scan the
                    # next few instructions past the call for this shape and
                    # attach the threshold to the emitted node. Other episodes'
                    # action minigames don't emit this pattern (their success/
                    # fail is resolved separately by resolve_minigame_gates), so
                    # they get None here, matching prior behavior.
                    wt = None
                    scan = p + 2
                    if scan < n and data[scan] <= 0x08:  # fold-argc byte
                        scan += 1
                    steps = 0
                    while scan < n - 6 and steps < 8:
                        b = data[scan]
                        if b == 0x5f and scan + 6 < n \
                                and data[scan + 1] == 0x3f \
                                and data[scan + 2] in (0x1a, 0x41) \
                                and data[scan + 5] == 0x0d \
                                and data[scan + 6] == 0x2b:
                            wt = (data[scan + 3] << 8) | data[scan + 4]
                            break
                        # skip past this instruction using the fold rules
                        if b in (0x1a, 0x41, 0x1b, 0x2b, 0x28, 0x5c):
                            scan += 3
                        elif b == 0x42:
                            scan += 4
                        elif b == 0x1f:
                            scan += 2
                            if scan < n and data[scan] <= 0x08:
                                scan += 1
                        else:
                            scan += 1
                        steps += 1
                    # text packs the prompt/label(s) and option groups, separated by §
                    lines.append(Line("§".join(prompts + groups),
                                      "__MINIGAME__", "action", threshold=wt))
                # A 0x47 with NO inline option groups (only the 20000.../3000 tail and
                # coordinate pairs) is an engine-rendered sprite/animation effect, not a
                # playable round -- e.g. the eight such calls woven through the ATGB
                # freshman intro. Real action minigames always carry their prompt and
                # pipe-delimited options inline (and the win/lose split is recovered
                # separately by the 0x47 gate scan), so a bare 0x47 emits no node.
            elif sel == SET_POV:                              # 0x4a -> set POV character
                # Operand = cast index of the now-controlled character, taken as the
                # most recent operand before the call. It can arrive as a bare value
                # (1a), as the 0/1 row markers (0x5a/0x5b -> e.g. Matt=0, Ben=1), or
                # as the first byte of a 1b pair (e.g. Hannah: 1b 04 00).
                idx = None
                for pk in reversed(pushes):
                    cand = pk[1] if pk[0] in ("v", "m", "pair") else None
                    if cand is not None and 0 <= cand < len(cast):
                        idx = cand
                        break
                if idx is not None and cast[idx]:
                    lines.append(Line(cast[idx], "__POVSET__", idx))
            elif sel == WOBBLE:                               # 0x59 -> wobble next line
                lines.append(Line(None, "__WOBBLE__", None))
            elif sel == LOADING:                              # 0x5b -> loading screen
                flag = next((pk[1] for pk in reversed(pushes)
                             if pk[0] in ("v", "m")), 1)
                lines.append(Line(None, "__LOADING__", flag))
            elif sel == SET_SCENE_VALUE:                      # 0x10 -> scene value reg
                val = next((pk[1] for pk in reversed(pushes)
                            if pk[0] in ("v", "m")), None)
                if val is not None:
                    lines.append(Line(None, "__SCENEVAL__", val))
            elif sel == SET_UI_DEFAULT:                       # 0x4b -> UI default slot
                val = next((pk[1] for pk in reversed(pushes)
                            if pk[0] in ("v", "m")), None)
                if val is not None:
                    lines.append(Line(None, "__UIDEFAULT__", val))
            elif sel == SET_STRING:                           # 0x2e -> string var write
                # [key_ref, value_ref]. A $Token first ref is a rename (handled by
                # _name_renames), so only emit set_string when the key is a plain
                # (non-$) string.
                refs = [pk[1] for pk in pushes if pk[0] == "v"]
                if len(refs) >= 2:
                    ktext = smap.get(refs[-2] * 2 + STRING_BASE, "")
                    vtext = smap.get(refs[-1] * 2 + STRING_BASE, "")
                    if ktext and not ktext.startswith("$"):
                        lines.append(Line("%s\x1f%s" % (ktext, vtext),
                                          "__SETSTR__", None))
            elif sel == TEXT_INPUT:                           # 0x28 -> player types text
                texts = [smap.get(pk[1] * 2 + STRING_BASE, "") for pk in pushes
                         if pk[0] == "v"]
                texts = [t for t in texts if t]
                title = texts[0] if texts else ""
                prompt = texts[1] if len(texts) > 1 else ""
                lines.append(Line("%s\x1f%s" % (title, prompt),
                                  "__TEXTINPUT__", None))
            elif sel == CHARACTER_PICKER:                     # 0x4e -> pick character(s)
                texts = [smap.get(pk[1] * 2 + STRING_BASE, "") for pk in pushes
                         if pk[0] == "v"]
                prompt = next((t for t in texts if t), "")
                lines.append(Line(prompt, "__PICKCHAR__", None))
            # any other selector (set-sprite, etc.) produces no transcript line.
            for _ln in lines[_before:]:
                if _ln is not None and _ln.bc_off is None:
                    _ln.bc_off = p
            pushes = []      # a call always consumes the operands pushed for it
            marker_row = None   # a row marker only applies to the call it preceded
            saw_add = False
            computed_write = False
            slot_display = False
            p += 2
            continue
        if op in (0x2b, 0x28) and p + 2 < n:  # conditional jump (3 bytes); a
            pushes = []                         # control boundary, so reset operands
            saw_add = False
            computed_write = False
            slot_display = False
            p += 3
            continue
        if op == 0x42 and p + 3 < n:  # branch separator (42 29 00 NN) — used by
            # The episode's final credits card (episode title + "Thanks for playing
            # ...! See you in another episode") is pushed just before this end-screen
            # separator and rendered by the host UI, NOT by a 1f display op -- so it
            # never reaches the dialogue/narration path and the thank-you screen is
            # lost. Surface it as an end card. Matched on the credits text (byte-
            # present), so ordinary mid-scene separators are unaffected.
            for _pk in pushes:
                if _pk[0] == "v":
                    _t = smap.get(_pk[1] * 2 + STRING_BASE)
                    if _t and ("thanks for playing" in _t.lower()
                               or "see you in another" in _t.lower()):
                        lines.append(Line(_t, "__CARD__", "end", bc_off=p))
            lines.append(Line(None, "__SEP__", None))   # branches mode only
            sep_pending = True   # a following title card (with no content between)
                                 # opens a new dispatch-section body
            # The SEP carries a 16-bit operand. Past the scene's 0x12 dispatch, a
            # SEP whose operand resolves onto another section body is that
            # section's EXIT JUMP (e.g. the Sophie-breakup section jumps to the
            # loser section), not a plain separator -- record the target so the
            # prior section gotos there instead of dead-ending.
            sep_section_target = None
            if seen_dispatch:
                _op = (data[p + 2] << 8) | data[p + 3]
                _hit = resolve_sep_target(data, bc_start, _op, _fold_offs,
                                          _fold_idx, _markers)
                if _hit is not None:
                    sep_section_target = _hit[0]
            pushes = []
            p += 4
            continue
        if op in (0x3e, 0x15, 0x3f, 0x60, 0x61, 0x62):
            # INDEX / STORE / LOAD / SLOT0-2: array & slot-table machinery. A 0x50
            # ADD that feeds one of these is computing an ARRAY INDEX (e.g. the
            # extra-credit score table `slot[var2000 + N]`), not a var delta, so a
            # pending saw_add here is stale -- clear it. A VWRITE fed by LOAD/SLOT
            # writes a looked-up/computed value that is not a static constant, so
            # flag it: the write is real but its value is runtime-computed, and
            # emitting a spurious `set`/`add` with the leftover operand bytes (The
            # Tutors scene 2's bogus `var2000 += 2000` / `var3 = 2000`) is wrong.
            saw_add = False
            if op in (0x3f, 0x60, 0x61, 0x62):
                computed_write = True
            if op == 0x15:
                # 0x15 STORE just wrote the pushed operand(s) into a quiz/array
                # slot -- they are DATA, not display content. A following 1f00 /
                # 1f41 that renders a "%s" template filled from that slot would
                # otherwise pick the stored string as its text (The Tutors scene 2
                # leaked "A beast...", "Correct!", "Good answer!" as status lines).
                # Drop the consumed operands so the display sees only its template.
                pushes = []
                slot_display = True
            p += 1
            continue
        p += 1  # any other opcode (e.g. 0x5a marker): ignore, keep operands

    lines = [ln for ln in lines if ln is not None]

    # CONVERSATIONAL-PARTNER ATTRIBUTION. Compact-form lines (1b <emotion>
    # <textref> ; 1f 0d) carry no speaker index -- the engine shows whichever
    # portrait is already on screen. When such a line resolves to no speaker but
    # DID carry an emotion (so it is spoken dialogue, not narration), attribute it
    # to the scene's other party in the two-person exchange: the nearest
    # explicitly-named non-lead speaker (searching the surrounding lines, nearest
    # first). This fixes e.g. Ms. Lee's classroom lines, which alternate with
    # Kim's explicitly-attributed lines but are themselves speaker-less. Only
    # applied when exactly one non-lead speaker is nearby, so a genuine
    # multi-party scene is left untouched.
    _lead = lead_of(cast) if cast else None
    _CTRL = ("__CARD__", "__CHOICE__", "__SCENE__", "__MINIGAME__", "__BG__",
             "__MUSIC__", "__SFX__", "__GOTO__", "__STATUS__", "__SEP__",
             "__POV__", "__VAR__", "__TCHOICE__", "__NOTIFY__", "__VIBRATE__",
             "__WOBBLE__", "__LOADING__", "__SCENEVAL__", "__UIDEFAULT__",
             "__SETSTR__", "__TEXTINPUT__", "__PICKCHAR__")
    # Iterate to a fixed point: a compact-form line adjacent only to other
    # compact-form lines can be resolved once its neighbour is, so repeat until no
    # further line changes (bounded by the number of lines).
    for _ in range(len(lines)):
        _explicit = [ln.speaker for ln in lines]
        _changed = False
        for _i, _ln in enumerate(lines):
            if _ln.speaker is not None or _ln.emotion is None or not _ln.text:
                continue
            _near = []
            for _d in range(1, 8):
                for _j in (_i - _d, _i + _d):
                    if 0 <= _j < len(lines):
                        _s = _explicit[_j]
                        if _s and _s != _lead and _s not in _CTRL:
                            _near.append(_s)
                if _near:
                    break
            _uniq = list(dict.fromkeys(_near))
            if len(_uniq) == 1:
                _spk = _uniq[0]
                _ln.speaker = _spk
                _base = None
                if sprites_by_idx and _spk in cast:
                    _ci = cast.index(_spk)
                    if 0 <= _ci < len(sprites_by_idx):
                        _base = sprites_by_idx[_ci]
                if _base is None and _spk in resources:
                    _base = resources[_spk]
                _ln.sprite = _base + (_ln.emotion or 0) if _base is not None else None
                _changed = True
        if not _changed:
            break

    # last) are shown via the engine's title path, not the dialogue op. Any
    # candidate whose text is ALSO displayed by a bytecode op in this script is
    # a real in-flow line, not framing -- synthesising it would duplicate
    # content, so it is dropped here (data-driven dedup, byte-for-byte).
    displayed = {ln.off for ln in lines if ln.off is not None}
    op_texts = {ln.text for ln in lines if ln.text}
    # Bank subtitles are already structured in the scene's `minigames` data;
    # re-synthesising them as prompt nodes would duplicate them too.
    bank_subs = {r.get("subtitle")
                 for recs in extract_minigame_banks(data).values() for r in recs}
    # Strings consumed as operands by a minigame call (0x60 build-word / 0x47
    # action) are the minigame's own UI banners ("Time's up!", "Next round",
    # round prompts/words), NOT standalone instruction cards. Collect their text
    # so the head-card scan does not surface them as phantom "prompt" minigames.
    # Byte-derived: the operand window pushed before each 1f60 / 1f47 call.
    mg_operand_text = set()
    _q, _N, _refs = bc_start, len(data), []
    while _q < _N:
        _op = data[_q]
        if _op in (0x1a, 0x41) and _q + 2 < _N:
            _refs.append((data[_q + 1] << 8) | data[_q + 2]); _q += 3
        elif _op == 0x1b and _q + 2 < _N:
            _refs.append(data[_q + 1]); _refs.append(data[_q + 2]); _q += 3
        elif _op == 0x1f and _q + 1 < _N:
            if data[_q + 1] in (0x60, 0x47):
                for _r in _refs:
                    _t = smap.get(_r * 2 + STRING_BASE)
                    if _t:
                        mg_operand_text.add(_t)
            _refs = []; _q += 2
        elif _op in (0x2b, 0x28):
            _refs = []; _q += 3
        elif _op == 0x42:
            _refs = []; _q += 4
        else:
            _q += 1
    title_cards, minigame_words, end_cards, intro_narr, instr_prompts = find_cards(
        strings, displayed, bc_start, cast, episode_title=episode_title)
    # Drop phantom scene-intro narration. A head-card sentence whose string ref is
    # never pushed by any instruction is a dead, unreferenced string (leftover
    # content), not a line the engine ever shows -- a displayed string is always
    # pushed. Surfacing one fabricates narration that downstream gate-nesting then
    # sweeps into a branch, producing a bogus replay loop. Purely byte-derived:
    # the test is membership in this script's set of pushed string refs.
    if intro_narr:
        _pushed, _q, _N = set(), bc_start, len(data)
        while _q < _N:
            _op = data[_q]
            if _op in (0x1a, 0x41) and _q + 2 < _N:
                _pushed.add((data[_q + 1] << 8) | data[_q + 2]); _q += 3
            elif _op == 0x1f:
                _q += 2
            elif _op in (0x1b, 0x2b, 0x28):
                _q += 3
            elif _op == 0x42:
                _q += 4
            else:
                _q += 1
        _off_by_text = {}
        for _o, _t in strings:
            _off_by_text.setdefault(_t, []).append(_o)

        def _referenced(t):
            return any((_o - STRING_BASE) % 2 == 0
                       and ((_o - STRING_BASE) // 2) in _pushed
                       for _o in _off_by_text.get(t, []))
        intro_narr = [t for t in intro_narr if _referenced(t)]
    _tc = [t for t in title_cards if t not in emitted_titles and t not in op_texts]
    head = []
    if _tc:
        # episode-intro title + (optional) subtitle/goal as one card
        head.append(Line(_tc[0], "__CARD__", "title",
                         sprite=(_tc[1] if len(_tc) > 1 else None)))
    head += [Line(t, None, None) for t in intro_narr
             if t not in op_texts]                              # scene-intro narration
    head += [Line(t, "__MINIGAME__", "prompt") for t in instr_prompts
             if t not in op_texts and t not in bank_subs
             and t not in mg_operand_text]
    if minigame_words:
        head.append(Line(" | ".join(minigame_words), "__MINIGAME__", "word-match"))
    tail = [Line(t, "__CARD__", "end") for t in end_cards if t not in op_texts]
    all_cards = head + lines + tail

    # Year-banner inheritance. A year's sections share one banner ("Sophomore
    # year", "Junior year", ...) that the engine shows once and keeps on screen;
    # each section's own title-card op then carries only its subtitle (which ends
    # in "..."). So a subtitle-only card reads e.g. "Dating Holly..." with no
    # banner. Re-attach the scene's banner as the main title of every such card.
    banner = next((ln.text for ln in all_cards
                   if ln.speaker == "__CARD__" and ln.emotion == "title" and ln.text
                   and re.search(r"\byear$", ln.text.strip(), re.I)), None)
    if banner is None and 15 in smap:        # banner may be the chunk's header title
        head_t = smap[15].strip()            # normalize_strings already de-glued it
        if re.search(r"\byear$", head_t, re.I):
            banner = head_t
    if banner:
        for ln in all_cards:
            if (ln.speaker == "__CARD__" and ln.emotion == "title" and ln.text
                    and ln.text.rstrip().endswith("...") and ln.text != banner
                    and not ln.sprite):
                ln.sprite = ln.text          # the section string is the subtitle
                ln.text = banner             # the year banner is the main title
    # Apply sprite-costume overrides. Within a costume span (armed by a bare
    # `1b(*,costume_idx)` before a section SEP), a character's dialogue portrait
    # uses the costume asset (base+emotion -> costume+emotion) even though the
    # line still carries the base speaker index. This is how a disguise / restyle
    # is shown: Dinah's nightclub sequence and Matt's bad-haircut scene both keep
    # their normal speaker index but should render the exp-bundled costume sprite.
    _spans = _costume_spans(data, costume_rows) if costume_rows else []
    if _spans:
        for ln in all_cards:
            if ln.bc_off is None or not ln.speaker \
                    or not isinstance(ln.speaker, str) \
                    or ln.speaker.startswith("__"):
                continue
            for sp in _spans:
                if sp["start"] <= ln.bc_off < sp["end"] \
                        and ln.speaker.lower() == sp["name"] \
                        and ln.sprite is not None \
                        and sp["base_asset"] <= ln.sprite \
                        <= sp["base_asset"] + 15:
                    # rebase the portrait onto the costume asset, preserving the
                    # emotion offset (sprite = base + emotion)
                    ln.sprite = sp["costume_asset"] + (ln.sprite - sp["base_asset"])
                    break

    # Apply dynamic character renames (1f2e): relabel a speaker whose resource
    # name is a `$`-variable slot to the display name bound in the bytecode
    # (e.g. Man2 -> "French Man"). Speaker-label only; sprites/routing untouched.
    # Renames are episode-wide: a rebind armed in one scene applies to that
    # character's lines in every scene, so the map is collected across all scenes
    # and passed in (a scene may speak the renamed character without arming it).
    if name_renames:
        for ln in all_cards:
            if not ln.speaker or not isinstance(ln.speaker, str) \
                    or ln.speaker.startswith("__"):
                continue
            new = name_renames.get(ln.speaker.lower())
            if new:
                ln.speaker = new
    # Substitute $Token text placeholders (e.g. "$Antagonist ..." -> "Travis ...")
    # in every line's visible text, using the episode-wide 1f2e text bindings.
    if text_subs:
        for ln in all_cards:
            if ln.text and "$" in ln.text:
                ln.text = _apply_text_subs(ln.text, text_subs)
    return all_cards, cast


def _emit(pushes, smap, cast, resources, narration=False, last_spk=None,
          speaker_row=None, sprites=None, content_min=0, caption_partner=None):
    """Build one Line from the operands pushed before a display instruction.

    `narration=True` (the 0x41 display op) forces the line to be narration with
    no speaker, regardless of any value in the operand buffer.

    A speaker-less spoken line marked 0x5a / 0x5b is spoken by resource-table
    row 0 / row 1 respectively (`speaker_row`); an unmarked one falls back to
    the most recent pair speaker (`last_spk`).

    `caption_partner` maps a string ref -> the relationship partner named in that
    caption (e.g. ref of "Dating Sophie..." -> "Sophie"). A line that pushes such
    a caption ref is spoken by the partner: the caption ref occupies the slot a
    speaker index normally would, so without this the caption ref is misread as a
    cast index. When a caption is present the speaker is the partner and the
    emotion is taken from the separate small operand.
    """
    def sprite_for(idx, speaker, emotion):
        """Portrait base by resource-table row index (the byte-true mapping);
        fall back to the name-keyed map for speakers resolved without an index."""
        base = None
        if idx is not None and sprites and 0 <= idx < len(sprites):
            base = sprites[idx]
        if base is None and speaker in resources:
            base = resources[speaker]
        return base + (emotion or 0) if base is not None else None

    cont_spk = speaker_row if speaker_row is not None else last_spk

    # RELATIONSHIP-CAPTION SPEAKER: if any operand is a caption ref ("Dating X..."),
    # the line is spoken by partner X. The caption value sits where a speaker index
    # would, so it must not be read as a cast index; the emotion is the separate
    # small operand (a bare push, or the 1st byte of the pair whose 2nd byte is the
    # caption). Byte-derived: caption string + cast membership.
    caption_partner = caption_partner or {}
    cap_partner, cap_emotion, caption_vals = None, None, set()
    if caption_partner:
        for pk in pushes:
            if pk[0] == "v":
                vals = (pk[1],)
            elif pk[0] == "pair":
                vals = (pk[1], pk[2])
            else:
                continue
            for v in vals:
                if v in caption_partner:
                    caption_vals.add(v)
                    if cap_partner is None:
                        cap_partner = caption_partner[v]
        if cap_partner is not None:
            text_ref = max((pk[1] for pk in pushes if pk[0] == "v"
                            and (pk[1] * 2 + STRING_BASE) in smap), default=-1)
            for pk in pushes:
                if pk[0] == "pair" and pk[2] in caption_vals:
                    cap_emotion = pk[1]; break
                if pk[0] == "v" and pk[1] not in caption_vals \
                        and pk[1] != text_ref and 0 < pk[1] < 16:
                    cap_emotion = pk[1]; break


    # TEXT = the pushed value that resolves to a string. Dialogue lives deep in
    # the file, so among colliding candidates pick the largest reference.
    text_idx, text_val = None, -1
    for idx, pk in enumerate(pushes):
        if pk[0] == "v":
            off = pk[1] * 2 + STRING_BASE
            if off in smap and off >= content_min and smap[off] and pk[1] > text_val:
                text_idx, text_val = idx, pk[1]

    # COMPACT FORM: scene-opening lines with a small text reference encode it as
    # the 2nd byte of the 1b pair (no separate 1a text push):
    #     1b <emotion> <textref> ; [push <speaker>] ; 1f 0d
    # When no normal text push is present, fall back to the pair's 2nd byte.
    if text_idx is None:
        compact = next((pk for pk in pushes if pk[0] == "pair"
                        and (pk[2] * 2 + STRING_BASE) in smap
                        and pk[2] * 2 + STRING_BASE >= content_min
                        and smap[pk[2] * 2 + STRING_BASE]
                        and _displayable(smap[pk[2] * 2 + STRING_BASE])),
                       None)
        if compact is None:
            # END-SCREEN FORM: results-screen lines (e.g. the end-of-episode money/day
            # tallies) encode the text ref in the *1st* byte of the 1b pair and render
            # it as narration: 1b <textref> <flag> ; 1f 00 ; 1f 41. Only used for
            # narration, and only when the 1st byte maps to a real, displayable string.
            if narration:
                endcard = next((pk for pk in pushes if pk[0] == "pair"
                                and (pk[1] * 2 + STRING_BASE) in smap
                                and pk[1] * 2 + STRING_BASE >= content_min
                                and len(smap[pk[1] * 2 + STRING_BASE]) > 5), None)
                if endcard is not None:
                    off = endcard[1] * 2 + STRING_BASE
                    return Line(smap[off], None, None, off=off)
            else:
                # COMPACT SPEECH FORM: some spoken lines (e.g. the day-hub's
                # "What should I do?") also carry the text ref in the 1st byte of the
                # 1b pair, with the 2nd byte a speaker sentinel (>= len(cast)) standing
                # for the active speaker. Emit as the active/lead speaker's line, only
                # when the 1st byte maps to a real displayable string.
                firstb = next((pk for pk in pushes if pk[0] == "pair"
                               and (pk[1] * 2 + STRING_BASE) in smap
                               and pk[1] * 2 + STRING_BASE >= content_min
                               and len(smap[pk[1] * 2 + STRING_BASE]) > 5
                               and _displayable(smap[pk[1] * 2 + STRING_BASE])), None)
                if firstb is not None:
                    off = firstb[1] * 2 + STRING_BASE
                    text = smap[off]
                    if text.startswith("'") and cast:
                        speaker = lead_of(cast)
                    elif cont_spk is not None and 0 <= cont_spk < len(cast) and cast[cont_spk]:
                        speaker = cast[cont_spk]
                    else:
                        speaker = lead_of(cast) if cast else None
                    idx = cont_spk if (cont_spk is not None and 0 <= cont_spk < len(cast)
                                       and cast[cont_spk] == speaker) else None
                    return Line(text, speaker, None, sprite_for(idx, speaker, None), off=off)
            return None
        text_off = compact[2] * 2 + STRING_BASE
        text = smap[text_off]
        emotion = compact[1]
        if narration:
            return Line(text, None, None, off=text_off)
        if cap_partner is not None:
            pidx = cast.index(cap_partner) if cap_partner in cast else None
            return Line(text, cap_partner, cap_emotion,
                        sprite_for(pidx, cap_partner, cap_emotion), off=text_off)
        speaker_idx = next((pk[1] for pk in pushes
                            if pk[0] == "v" and pk[1] < max(len(cast), 1)), None)
        if speaker_idx is not None and 0 <= speaker_idx < len(cast):
            speaker = cast[speaker_idx]
        elif text.startswith("'") and cast:
            speaker = lead_of(cast)
        elif cont_spk is not None and 0 <= cont_spk < len(cast) and cast[cont_spk]:
            speaker = cast[cont_spk]          # continuation by the active speaker
        else:
            speaker = None
        idx = speaker_idx if speaker_idx is not None else (
            cont_spk if (cont_spk is not None and 0 <= cont_spk < len(cast)
                         and cast[cont_spk] == speaker) else None)
        return Line(text, speaker, emotion, sprite_for(idx, speaker, emotion),
                    off=text_off)

    text_off = pushes[text_idx][1] * 2 + STRING_BASE
    text = smap[text_off]

    if narration:
        return Line(text, None, None, off=text_off)

    if cap_partner is not None:
        pidx = cast.index(cap_partner) if cap_partner in cast else None
        return Line(text, cap_partner, cap_emotion,
                    sprite_for(pidx, cap_partner, cap_emotion), off=text_off)

    # SPEAKER / EMOTION = the push immediately following the text reference.
    speaker_idx, emotion = None, None
    for pk in pushes[text_idx + 1:]:
        if pk[0] == "pair":
            emotion, speaker_idx = pk[1], pk[2]
            break
        if pk[0] == "v" and pk[1] < max(len(cast), 1):
            speaker_idx = pk[1]
            break

    if speaker_idx is not None and 0 <= speaker_idx < len(cast):
        speaker = cast[speaker_idx]
    elif not narration and text.startswith("'") and cast:
        # A speaker-less dialogue-op line whose text is wrapped in single quotes is
        # the POV character's diary / inner monologue (e.g. Dinah's "'Diary Entry...'").
        # The narration op (0x41) is unaffected; only the 0x0d speech op reaches here.
        speaker = lead_of(cast)
    elif cont_spk is not None and 0 <= cont_spk < len(cast) and cast[cont_spk]:
        # A 0x0d line with no inline speaker is a continuation: by the explicit active
        # speaker (1f 05) if this line was marked 0x5b, else the most recent speaker.
        speaker = cast[cont_spk]
    else:
        speaker = None  # narration / system card / on-screen sprite only

    # PORTRAIT IMAGE = the speaker's sprite-sheet base + the emotion frame.
    # (Characters absent from the resource table -- e.g. the player character --
    # are rendered from a dynamic avatar and have no fixed asset id here.)
    idx = speaker_idx if speaker_idx is not None else (
        cont_spk if (cont_spk is not None and 0 <= cont_spk < len(cast)
                     and cast[cont_spk] == speaker) else None)
    sprite = sprite_for(idx, speaker, emotion) if speaker is not None else None
    return Line(text, speaker, emotion, sprite, off=text_off)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
# Engine-level UI phrases that end a non-ideal outcome (the game prompts a
# replay). Used only to ANNOTATE outcome boundaries: the lines themselves are
# always kept in order, so an unrecognised phrasing degrades to an unsplit
# branch_dialogue (with a note), never to lost or invented content.
RETRY_BANNERS = ("try again", "play this one again", "play again", "start over",
                 "from the beginning", "better ending", "bonus scene",
                 "earn a better", "play again")

# In-flow minigame triggers come from the per-episode OVERLAY (key
# `minigame_triggers`: [{phrase, kind, label?}, ...]) -- they are observed
# gameplay (which line hands control to the player), not recoverable from the
# bytes. "word-match" triggers reposition the word-list data block (stored at
# the script head) to the trigger line; other kinds emit a standalone
# minigame-prompt node there, labelled with the overlay-supplied (observed)
# label. The parser itself ships NO trigger phrases: they are episode data.


def reposition_minigames(lines, triggers=None):
    """Move the word-match word-list to its in-flow trigger and emit prompt nodes.

    A word-match data block is stored at the top of a script, so the decoder
    surfaces it at the scene head. Given overlay-declared triggers, we hold it
    until the line that actually starts the game and drop it in there, and emit
    minigame nodes at action / play-as hand-off lines. With no triggers (no
    overlay, or none declared) this is a no-op apart from restoring any held
    block at its scene boundary. If a held word-list never finds a trigger in
    its scene, it is restored at the scene boundary.
    """
    triggers = [(t["phrase"].lower(), t.get("kind"), t.get("label"))
                for t in (triggers or [])]
    result, pending_wm = [], None
    for ln in lines:
        if ln.speaker == "__SCENE__":
            if pending_wm is not None:
                result.append(pending_wm)
                pending_wm = None
            result.append(ln)
            continue
        if ln.speaker == "__MINIGAME__" and ln.emotion == "word-match":
            pending_wm = ln          # hold; place at its trigger
            continue
        result.append(ln)
        low = (ln.text or "").lower()
        for phrase, kind, label in triggers:
            if phrase in low:
                if kind == "word-match":
                    if pending_wm is not None:
                        result.append(pending_wm)
                        pending_wm = None
                else:
                    result.append(Line(label, "__MINIGAME__", kind))
                break
    if pending_wm is not None:
        result.append(pending_wm)
    return result


def split_outcomes(branch_dialogue):
    """Split a choice's following dialogue into distinct outcomes.

    The engine ends each non-ideal path with a meta "banner" line that prompts
    the player to replay (e.g. "Try again...", "start over from the beginning").
    These reliably delimit the alternate endings, so we cut a new segment after
    each banner. Returns a list of segments, or None if fewer than two are found
    (i.e. an ordinary mid-scene choice with no replay prompts).
    """
    segs, cur = [], {"ends_with_retry_prompt": False, "lines": []}
    found = False
    for e in branch_dialogue:
        is_banner = e["type"] == "narration" and any(m in e["text"].lower() for m in RETRY_BANNERS)
        # A banner ends an outcome, but several banner lines can run together; only
        # start the next outcome when real (non-banner) content appears after one.
        if cur["ends_with_retry_prompt"] and not is_banner:
            segs.append(cur)
            cur = {"ends_with_retry_prompt": False, "lines": []}
        cur["lines"].append(e)
        if is_banner:
            cur["ends_with_retry_prompt"] = True
            found = True
    if cur["lines"]:
        segs.append(cur)
    if not found or len(segs) < 2:
        return None
    for idx, s in enumerate(segs):
        s["index"] = idx
    return segs


def tag_pov(lines, cast):
    """Flag who the player controls (the POV / main character) as it switches.

    Multi-protagonist episodes alternate control between main characters. The
    reliable signals, all read from the script's own lines:
      * the intro line "play as both X and Y" -> the set of playable mains;
      * explicit hand-off lines ("Help <main>...", "Play as her...") -> who you
        control for the next stretch;
      * the diary/monologue owner (already attributed) -> that character's POV.
    A __POV__ marker is inserted at each hand-off. Scenes with no explicit cue
    are left unmarked rather than guessed at.
    """
    mains = []
    for ln in lines:
        if ln.text and "play as both" in ln.text.lower():
            low = ln.text.lower().replace("`", "")
            mains = [c for c in cast if c.lower() in low]
            break
    if not mains:
        mains = [lead_of(cast)] if lead_of(cast) else []

    def controlled(text):
        low = text.lower().replace("`", "")
        if any(b in low for b in RETRY_BANNERS):       # retry banners aren't hand-offs
            return None
        named = [m for m in mains if m.lower() in low]
        if "play as both" in low:
            return mains
        if "play as" in low:
            return named[:1] or (mains[:1] if "play as her" in low or "play as him" in low else None)
        if "help " in low and named and any(v in low for v in
                                            ("cheer", "sneak", "talk", "find out", "win")):
            return named[:1]
        return None

    handoff_speakers = {None} | set(mains)   # narration or a main's own line
    out, current = [], None
    for ln in lines:
        out.append(ln)
        if ln.text and ln.speaker in handoff_speakers and ln.speaker != "__POV__":
            who = controlled(ln.text)
            if who:
                label = " & ".join(who)
                if label != current:
                    out.append(Line(label, "__POV__", "playable"))
                    current = label
    return out, mains


_HUD_RE = re.compile(
    r'\bhas\s+(?:%d|\d+)\s+\w*\s*dollars'      # "...has 200 dollars" / "...has %d dollars"
    r'|(?:%d|\d+)\s+days?\s+left'              # "4 days left" / "%d days left"
    r'|\b(?:earned|hands\s+\w+)\s+(?:%d|\d+)\s+dollars'   # reward banners
    r'|days?\s+are\s+up', re.I)
_FMT_RE = re.compile(r'%[ds]')


def is_hud_line(text):
    """True if a line is a money/days/reward HUD counter (shown via narration op)."""
    return bool(_HUD_RE.search(text))


def _is_system(t):
    """Score/rank/quiz/menu data the engine reads directly (printf templates, option
    lists, scoring/menu prompts) -- not a spoken line or a title card."""
    if "%" in t or "|" in t:
        return True
    low = t.lower()
    kw = ("points", "rank for", "grade for", "questions correct", "bonus scene",
          "unlocking", "new score", "new rank", "checkpoint", "restart episode",
          "replay", "perfect` score", "make your choice", "what do you do",
          "what do you want to do", "you answered", "out of 100", "extra credit",
          "earned %", "let's begin")
    return any(k in low for k in kw)


def _displayable(text):
    """Reject binary/format-fragment junk (e.g. ',a', '%s', '...') as on-screen text:
    require at least two real letters once %d/%s placeholders are removed."""
    bare = _FMT_RE.sub("", text)
    return sum(c.isalpha() for c in bare) >= 2


def line_to_dict(ln):
    """Convert a content Line to a JSON-friendly dict (None for control markers)."""
    if ln.speaker == "__POV__":
        return {"type": "control", "playable": ln.text.split(" & ")}
    if ln.speaker == "__POVSET__":
        return {"type": "pov_change", "character": ln.text, "index": ln.emotion}
    if ln.speaker in ("__SEP__", "__SCENE__", "__CHOICE__"):
        return None
    if ln.speaker == "__CARD__":
        if ln.emotion == "title":
            d = {"type": "title_card", "title": ln.text}
            if ln.sprite:                # subtitle carried in the sprite slot
                d["subtitle"] = ln.sprite
            if getattr(ln, "section_start", False):
                d["section_start"] = True
                if getattr(ln, "section_jump", None) is not None:
                    # the prior section's terminating SEP jumps to this body offset
                    d["_prev_section_jump"] = ln.section_jump
            return d
        return {"type": "end_card", "text": ln.text}
    if ln.speaker == "__MINIGAME_REF__":
        return {"type": "minigame", "kind": ln.emotion, "from_scene_bank": True}
    if ln.speaker == "__NOTIFY__":
        return {"type": "notification", "kind": ln.emotion, "text": ln.text}
    if ln.speaker == "__MINIGAME__":
        if ln.emotion == "word-match":
            return {"type": "minigame", "kind": "word-match", "words": ln.text.split(" | ")}
        if ln.emotion == "action-tap":
            return {"type": "minigame", "kind": "action", "inline_content": False,
                    "note": ("Engine-rendered action/tap round; its on-screen targets "
                             "and labels are driven by numeric parameters, not encoded "
                             "as script text.")}
        if ln.emotion == "action":
            parts = ln.text.split("§")
            prompts = [p for p in parts if "|" not in p]
            groups = [p.split("|") for p in parts if "|" in p]
            node = {"type": "minigame", "kind": "action",
                    "prompt": " ".join(prompts), "options": groups}
            if ln.threshold is not None:
                node["win_threshold"] = ln.threshold
            return node
        return {"type": "minigame", "kind": ln.emotion, "prompt": ln.text}
    if ln.speaker == "__TCHOICE__":
        parts = ln.text.split("§")
        prompt = next((p for p in parts if "|" not in p), None)
        opts = next((p.split("|") for p in parts if "|" in p), [])
        return {"type": "choice", "timed": True, "timer_ms": ln.emotion,
                "prompt": prompt,
                "options": [{"index": i, "label": o} for i, o in enumerate(opts)],
                "note": ("Timed choice (1f02/1f03/1f04): the player must pick "
                         "before the timer expires. The picked index is tested "
                         "against the correct answer to branch win/lose.")}
    if ln.speaker == "__VAR__":
        kind, var, val = ln.emotion
        if kind == "set":
            return {"type": "var_set", "var": var, "value": val}
        return {"type": "var_add", "var": var, "value": val}
    if ln.speaker == "__SFX__":
        return {"type": "sfx", "sfx_id": ln.emotion}
    if ln.speaker == "__MUSIC__":
        if ln.emotion == "stop":
            return {"type": "music", "action": "stop"}
        return {"type": "music", "track_id": ln.emotion}
    if ln.speaker == "__VIBRATE__":
        return {"type": "vibrate"}
    if ln.speaker == "__WOBBLE__":
        return {"type": "wobble"}
    if ln.speaker == "__LOADING__":
        return {"type": "loading", "blocking": bool(ln.emotion)}
    if ln.speaker == "__SCENEVAL__":
        return {"type": "set_scene_value", "value": ln.emotion}
    if ln.speaker == "__UIDEFAULT__":
        return {"type": "set_ui_default", "value": ln.emotion}
    if ln.speaker == "__SETSTR__":
        _k, _, _v = (ln.text or "").partition("\x1f")
        return {"type": "set_string", "key": _k, "value": _v}
    if ln.speaker == "__TEXTINPUT__":
        _t, _, _p = (ln.text or "").partition("\x1f")
        return {"type": "text_input", "title": _t, "prompt": _p}
    if ln.speaker == "__PICKCHAR__":
        return {"type": "character_picker", "prompt": ln.text or ""}
    if ln.speaker == "__GOTO__":
        return {"type": "goto_scene", "script": "0x%04x" % ln.emotion}
    if ln.speaker == "__STATUS__":
        if not _displayable(ln.text):
            return None
        return {"type": "status", "text": ln.text, "dynamic": bool(_FMT_RE.search(ln.text))}
    if ln.speaker == "__BG__":
        kind, code = ln.emotion
        d = {"type": "background", "style": kind, "asset_id": code}
        if kind == "custom" and code in KNOWN_BG:
            d["name"] = KNOWN_BG[code]       # observed label, from the overlay
        return d
    if ln.speaker is None:
        if not _displayable(ln.text):          # drop binary/format-fragment junk
            return None
        if is_hud_line(ln.text) and _FMT_RE.search(ln.text):
            # A genuine live HUD counter has a `%d` the engine fills at runtime
            # ("Kim has %d dollars." / "Kim has %d days left."). Emit those as
            # `status` so the engine substitutes the value. A reward/announcement
            # banner with the amount already baked into the text ("Kim just earned
            # 100 dollars!") is displayed via the plain narration op and carries no
            # placeholder or value_from, so it is ordinary narration -- typing it
            # as status would send the engine looking for a value_from that isn't
            # there.
            return {"type": "status", "text": ln.text, "dynamic": True}
        return {"type": "narration", "text": ln.text}
    return {"type": "dialogue", "speaker": ln.speaker,
            "emotion_code": ln.emotion, "image": ln.sprite, "text": ln.text}


def strip_notes(obj):
    """Recursively drop decoder-developer commentary fields from the runtime
    document, keeping only what the engine needs to run the episode. Removes
    `note`/`_note` explanation strings and `source` provenance markers wherever
    they appear; all routing, content, and structural fields are preserved.
    """
    DROP = {"note", "_note", "source"}
    if isinstance(obj, dict):
        return {k: strip_notes(v) for k, v in obj.items() if k not in DROP}
    if isinstance(obj, list):
        return [strip_notes(v) for v in obj]
    return obj


def stringify_numbers(obj):
    """Recursively convert every int/float in a JSON-ish structure to a string.

    Booleans are left as-is (they are not "numbers" here), as are None and
    existing strings. Dict keys are already strings in this document.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: stringify_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_numbers(v) for v in obj]
    return obj


def _to_int(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def restore_int_values(obj):
    """Run after stringify_numbers to give the fields that hold actual variable
    VALUES (not ids/offsets/labels) real integer typing: the `value` of a
    var_set/var_add node, and every entry of variable_defaults `values`.
    Everything else stays a string."""
    if isinstance(obj, dict):
        if obj.get("type") in ("var_set", "var_add") and "value" in obj:
            obj["value"] = _to_int(obj["value"])
        vd = obj.get("variable_defaults")
        if isinstance(vd, dict) and isinstance(vd.get("values"), dict):
            vd["values"] = {k: _to_int(v) for k, v in vd["values"].items()}
        for v in obj.values():
            restore_int_values(v)
    elif isinstance(obj, list):
        for x in obj:
            restore_int_values(x)
    return obj


_NAME_VAR_RE = re.compile(r'\$[A-Za-z][A-Za-z0-9_]{0,15}$')


_BARE_NAME_RE = re.compile(r"[A-Z][A-Za-z.'-]{0,18}$")   # a name-var default


def extract_name_vars(data: bytes):
    """Parse the name-variable defaults from a script's head.

    Dialogue text can contain '$VAR' placeholders the engine substitutes with a
    (player-customisable) name. The defaults are stored as literal strings in
    the zone between the cast table's 'Event' marker and the first prose line,
    in two byte-verified forms:
      * an explicit pair: the '$VAR' string immediately followed by its default
        (e.g. '$USR' then 'John', '$ZOE' then 'Zoe');
      * a bare name exactly equal to a cast '$'-variable's cleaned name
        (e.g. cast '$Travis' -> a lone 'Travis'; '$ALLISON' -> 'Allison').
    A '$VAR' must be a single token (no spaces/punctuation): dialogue lines that
    merely START with a variable ("$ALLISON! Wait!") are text, not declarations.
    Returns {'$VAR': default}. Cast '$'-vars with no stored default fall back
    to their own cleaned name (which is what every stored default equals).
    Scripts with no cast table (continuation scenes) contribute nothing.
    """
    if data[:4] != b"kiwi":
        return {}
    strings = find_strings(data)
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    evt_off = next((o for o, t in strings if t == "Event"), None)
    if evt_off is None:
        return {}
    cast_vars = {}            # cleaned name -> '$VAR' as written
    for off, t in strings:
        if off >= evt_off:
            break
        if off < CAST_START:
            continue
        raw = t[1:] if off == CAST_START else t       # de-glue the header byte
        if _NAME_VAR_RE.match(raw):
            cast_vars[_clean_name(raw)] = raw
    # The defaults zone: after 'Event', before the bytecode, up to the first
    # prose line (a real sentence ends the head data).
    zone = []
    for o, t in strings:
        if o <= evt_off or o >= bc_start:
            continue
        if len(t) > 45 and " " in t and "|" not in t:
            # A recap/prose line. Some episodes (e.g. Swim Team Retreat 2) open
            # the head with a "Previously on..." recap BEFORE the name-variable
            # declarations, so this must not end the zone -- skip it and keep
            # scanning. The pair test below is tight enough that prose cannot be
            # mistaken for a declaration.
            continue
        zone.append(t)
    out = {}
    for i, t in enumerate(zone):
        if _NAME_VAR_RE.match(t) and i + 1 < len(zone):
            nxt = zone[i + 1]
            # a default is a BARE name: single token, capitalised, no spaces
            if nxt and _BARE_NAME_RE.match(nxt):
                out.setdefault(t, nxt)
        elif t in cast_vars:
            out.setdefault(cast_vars[t], t)
    for clean, var in cast_vars.items():
        out.setdefault(var, clean)
    return out


def resolve_buildword_scoring(data):
    """Extract build-word minigame SCORING parameters from the bytecode.

    A build-word round (the 0x60 minigame call) carries a parameter block:

        0x60 ; push <p0> ; push <score> ; push <cap> ; push <mid> ; ... ; 1f 60

    where <cap> is a large ceiling (~10000) and <score> the round's score value
    (observed climbing 1500 -> 2000 across the cookie-baking rounds; ATGB's
    confidence round is 1600). The cap distinguishes a real score block from
    other 0x60 uses. Returns [{offset, score, cap, midpoint, param0}].

    The exact role of each field (whether <score> is the round target, the par,
    or the points awarded) is not asserted from the bytes; they are reported as
    the raw scoring parameters the engine feeds the minigame.
    """
    if data[:4] != b"kiwi":
        return []
    n = len(data)
    out, p = [], 0
    while p < n - 14:
        if data[p] == 0x60 and all(data[p + 1 + 3 * i] in (0x1a, 0x41)
                                   for i in range(4)):
            vals = [(data[p + 1 + 3 * i + 1] << 8) | data[p + 1 + 3 * i + 2]
                    for i in range(4)]
            if 8000 <= vals[2] <= 12000 and 100 <= vals[1] <= 9999:
                out.append({"offset": p, "param0": vals[0], "score": vals[1],
                            "cap": vals[2], "midpoint": vals[3]})
                p += 12
                continue
        p += 1
    return out


def resolve_minigame_gates(data):
    """Resolve MINIGAME WIN/LOSE branches that are statically present in the bytecode.

    Unlike the section dispatch (whose target is computed by the host VM), a
    minigame's outcome branch is an ordinary conditional jump on the engine's
    pass/fail flag, with BOTH targets static:

        <minigame trigger: 0x47 action, or 0x5e word-bank read> ...
        push <src> ; push <threshold> ; <cmp> ; 0x2b <to-fail-arm>

    The 0x2b jumps to the FAIL arm when the player misses the threshold; the
    fall-through is the PASS arm. Only the boolean pass/fail is runtime; the two
    arm offsets are in the bytes.

    To avoid mistaking a quiz/round LOOP gate (which reconverges) for a real
    win/lose FORK, a gate is only reported when its two arms DIVERGE -- i.e. they
    reach different terminal endings (different register-write values or
    different goto-scene targets). Loop gates, whose arms reconverge to the same
    continuation, are skipped.

    Returns [{"trigger": off, "gate": off, "pass_at": off, "fail_at": off}].
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    ins, p = [], bc
    while p < n:
        op = data[p]
        if op in (0x1a, 0x41, 0x1b, 0x28, 0x2b, 0x42, 0x5c):
            ins.append(p); p += 3
        elif op == 0x1f:
            ins.append(p); p += 3
        else:
            ins.append(p); p += 1
    index = {q: i for i, q in enumerate(ins)}

    def jtarget(q):
        v = (data[q + 1] << 8) | data[q + 2]
        ti = index[q] + v
        return ins[ti] if 0 <= ti < len(ins) else None

    smap = {o: t for o, t in normalize_strings(strings)}

    def terminal(start, maxhop=2000):
        # follow forward to the first register-write or goto-scene (the ending)
        q, steps = start, 0
        while q < n - 8 and steps < maxhop:
            op = data[q]
            if op in (0x1a, 0x41):
                v = (data[q + 1] << 8) | data[q + 2]
                if data[q + 3] in (0x1a, 0x41) and data[q + 6] == 0x1f \
                        and data[q + 7] == VAR_WRITE:
                    return ("var", (data[q + 4] << 8) | data[q + 5])
                if 25000 <= v < 26000:
                    return ("goto", v)
                q += 3
            elif op in (0x1b, 0x28, 0x2b, 0x42, 0x5c, 0x1f):
                q += 3
            else:
                q += 1
            steps += 1
        return ("none", None)

    def first_line(start, span=80):
        # the first displayable string on an arm (to compare arm content)
        q = start
        while q < min(start + span, n - 2):
            if data[q] in (0x1a, 0x41):
                v = (data[q + 1] << 8) | data[q + 2]
                t = smap.get(v * 2 + STRING_BASE)
                if t and len(t) > 6 and "|" not in t:
                    return t
                q += 3
            elif data[q] in (0x1b, 0x28, 0x2b, 0x42, 0x5c, 0x1f):
                q += 3
            else:
                q += 1
        return None

    out, seen_gates = [], set()
    _bakery_forks = []
    for i, q in enumerate(ins):
        trig = None
        if data[q] == 0x1f and data[q + 1] == ACTION_MINIGAME:
            trig = q
        elif data[q] == 0x1f and data[q + 1] == 0x46:
            # 0x46 is the "make/choose" minigame (e.g. Making Some Dough's bakery
            # bake-the-cookies round): like 0x47 it is followed by a static
            # pass/fail result gate. Same divergence guard below prevents loop
            # gates from being mistaken for real win/lose forks, so this only adds
            # branches where the two arms genuinely reach different endings.
            trig = q
        elif data[q] == 0x1f and data[q + 1] == 0x51:   # play-minigame-by-id
            # real minigame launch is preceded by push <id ~1000>; the end-of-
            # episode replay/survey screen also uses 0x51 but without an id push.
            # 0x51 (FUN_000a3260) is ALSO used as a plain asset/sound cue -- it is
            # not itself a minigame launcher (the real launcher is 0x47, which
            # allocates the minigame object). A genuine by-id minigame FORK gates
            # on the minigame SCORE (a slot/threshold compare with no variable read
            # between the launch and the compare). A 0x51 followed by an ordinary
            # variable-read gate (`push <var>; 1f2d VREAD; cmp`) is a plain content
            # branch on that variable -- NOT a minigame -- and must not be carved
            # into win/lose arms, because the lose arm then swallows the section's
            # real continuation (e.g. Halloween scene 2's `var2001 == 0` gate, whose
            # fall-through carries the goto to scene 3). So accept the by-id fork
            # only when the following result gate does NOT read a variable (1f2d)
            # before its comparison.
            prev = ins[i - 1] if i > 0 else None
            if prev is not None and data[prev] in (0x1a, 0x41) \
                    and 1000 <= ((data[prev + 1] << 8) | data[prev + 2]) <= 1010:
                # scan forward to the result gate (0x2b). A genuine minigame result
                # gate either tests the score directly or via a score-table lookup
                # (VREAD used as an INDEX, then a slot LOAD is compared). Reject only
                # a gate that reads a variable (1f2d VREAD) with NO slot operation
                # (0x3e INDEX / 0x15 STORE / 0x3f LOAD) anywhere before the compare --
                # that is a raw variable content branch, not a minigame (Halloween
                # scene 2's `push 2001; 1f2d; ==0`). Wrong Side of Town's scene-5 fork
                # reads var2006 but then INDEX/STORE/LOADs a score slot, so it is kept.
                _has_vread = False
                _has_slot = False
                for _k in range(i + 1, min(i + 16, len(ins))):
                    _o = ins[_k]
                    _b = data[_o]
                    if _b in (0x3e, 0x15, 0x3f):
                        _has_slot = True
                    if _b == 0x1f and _o + 1 < n and data[_o + 1] == 0x2d:
                        _has_vread = True
                    if _b == 0x2b:
                        break
                _choice_fed = False
                for _k in range(i - 1, max(i - 8, -1), -1):
                    _o = ins[_k]
                    if data[_o] == 0x1f and _o + 1 < n and data[_o + 1] == 0x01:
                        _choice_fed = True
                        break
                    if data[_o] == 0x2b:
                        break
                if not (_has_vread and not _has_slot) and not _choice_fed:
                    trig = q
        elif data[q] == 0x5e:
            trig = q
        if trig is None:
            continue
        is_bakery = data[q] == 0x1f and data[q + 1] == 0x46
        # find a result gate (0x2b, far target) within a short window
        for j in range(i + 1, min(i + 16, len(ins))):
            if data[ins[j]] != 0x2b:
                continue
            t = jtarget(ins[j])
            # The bakery (0x46) win/lose fork sits tighter than an action
            # minigame's (its arms rejoin a shared bakery flow), so allow a
            # smaller instruction gap for it; other triggers keep the >15 rule
            # that screens out local control jumps.
            min_gap = 8 if is_bakery else 15
            if t is None or abs(index[t] - j) <= min_gap:
                continue
            fall = ins[j + 1] if j + 1 < len(ins) else None
            if fall is None:
                break
            gp = ins[j]
            if gp in seen_gates:        # same gate via two triggers -> dedup
                break
            # CLASSIFY: a real fork has DIFFERENT content on its two arms. Quiz/round
            # loops have IDENTICAL first lines -> skip those. The bakery fork is
            # exempt from the first-line test: both arms briefly share the "These
            # cookies look great!" result banner before diverging, so it is gated
            # instead by the stronger terminal-divergence test below (its arms
            # must reach different endings to be reported).
            pf, ff = first_line(fall), first_line(t)
            if not is_bakery and (pf is None or ff is None or pf == ff):
                break
            if is_bakery:
                # The bakery arms each begin with a SEP that jumps to the real
                # win / lose body, so the linear `terminal()` (which does not
                # follow jumps) sees a shared convergence and cannot tell them
                # apart. Follow each arm's first jump, then compare terminals.
                def _hop(a):
                    qa, hops = a, 0
                    while qa < n - 4 and hops < 6:
                        if data[qa] == 0x42 and data[qa + 1] == 0x29:
                            op2 = (data[qa + 2] << 8) | data[qa + 3]
                            nxt = ins[index[qa] + op2] if qa in index \
                                and 0 <= index[qa] + op2 < len(ins) else None
                            if nxt is None:
                                break
                            return nxt
                        if data[qa] in (0x1a, 0x41, 0x1b, 0x28,
                                        0x2b, 0x5c, 0x1f):
                            qa += 3
                        else:
                            qa += 1
                        hops += 1
                    return a
                pa, fa = _hop(fall), _hop(t)
                # The bakery win/lose arms both eventually return to the hub
                # (var1001 := 25), so equal terminals do NOT mean "no fork" here
                # -- the win arm shows a new visitor and advances the day while the
                # lose arm repeats the same day. Distinguish by CONTENT: the arms
                # are a real fork when their first displayed lines differ. Only
                # when the content is identical too is it a genuine loop gate.
                pa_line, fa_line = first_line(pa), first_line(fa)
                same_terminal = terminal(pa) == terminal(fa)
                same_content = (pa_line or "") == (fa_line or "")
                if same_terminal and same_content:
                    break                # arms truly reconverge -> loop gate
                fall, t = pa, fa
            # skip end-of-episode replay/survey flow and other system prompts:
            # those use the same 0x51/jump shape but their arms are system text
            # ("Make your choice!", "Take Survey", retry banners), not gameplay.
            low = ((pf or "") + " " + (ff or "")).lower()
            if _is_system(pf or "") or _is_system(ff or "") \
                    or any(b in low for b in RETRY_BANNERS) \
                    or "another shot" in low or "survey" in low:
                break
            # outcome_fork = arms reach different endings; content_fork = same ending,
            # different content (score changes what you see, not where you end up).
            kind = "outcome_fork" if terminal(fall) != terminal(t) else "content_fork"
            # The win_threshold is the score the player must reach, set into the
            # minigame by a `push N ; 0x3e` (set-goal) just before/around the score
            # test. This is the difficulty-scaled bar (Making Some Dough's bakery
            # runs 6/6/6/7/8 across its five days; the fight minigame is 6). Earlier
            # code anchored on the nearest constant to the result gate, which is the
            # engine's `== 2` status code (2 = "played"), NOT the score bar -- that
            # made every bakery day read as 2 and understated every skill game.
            # Prefer the `push N ; 0x3e` goal; fall back to a `push N ; GTE`
            # comparison, then to the old nearest-constant walk.
            thr = None
            _scan_lo = max(i - 2, 0)
            _scan_hi = min(j + 20, len(ins))
            for jj in range(_scan_lo, _scan_hi):     # push N ; 0x3e (set goal)
                if data[ins[jj]] == 0x3e and jj > 0:
                    q3 = ins[jj - 1]
                    if data[q3] in (0x1a, 0x41):
                        cand = (data[q3 + 1] << 8) | data[q3 + 2]
                        if 0 < cand < 1000:
                            thr = cand
                            break
                    elif data[q3] in (0x5a, 0x5b):
                        thr = data[q3] - 0x5a
                        break
            if thr is None:
                for jj in range(j, min(j + 18, len(ins))):   # push N ; GTE
                    if data[ins[jj]] == 0x0d:
                        for kk in range(jj - 1, max(jj - 4, -1), -1):
                            q3 = ins[kk]
                            if data[q3] in (0x1a, 0x41):
                                cand = (data[q3 + 1] << 8) | data[q3 + 2]
                                if cand < 1000:
                                    thr = cand
                                break
                            if data[q3] in (0x5a, 0x5b):
                                thr = data[q3] - 0x5a
                                break
                        if thr is not None:
                            break
            if thr is None:
                for jj in range(j - 1, max(j - 5, -1), -1):
                    q2 = ins[jj]
                    if data[q2] in (0x1a, 0x41):
                        thr = (data[q2 + 1] << 8) | data[q2 + 2]
                        break
                    if data[q2] in (0x5a, 0x5b):
                        thr = data[q2] - 0x5a
                        break
            seen_gates.add(gp)
            # How the minigame is launched, and its engine id when played by id
            # (push <1000..1010> ; 1f 51). The fight reads a word bank (0x5e) and
            # then launches engine minigame 1000.
            via = ("action" if data[trig] == 0x1f and data[trig + 1] == ACTION_MINIGAME
                   else "by_id" if data[trig] == 0x1f and data[trig + 1] == 0x51
                   else "word_bank")
            mg_id = None
            for jj in range(i, j + 1):
                qj = ins[jj]
                if data[qj] == 0x1f and data[qj + 1] == 0x51 and jj > 0:
                    pv = ins[jj - 1]
                    if data[pv] in (0x1a, 0x41):
                        cand = (data[pv + 1] << 8) | data[pv + 2]
                        if 1000 <= cand <= 1010:
                            mg_id = cand
                            break
            _g = {"trigger": trig, "gate": gp, "pass_at": fall,
                  "fail_at": t, "win_threshold": thr, "kind": kind,
                  "via": via, "minigame_id": mg_id}
            if is_bakery:
                # A 0x46 round only counts as a real fork here through the relaxed
                # content-based divergence test above (its arms both return to the
                # hub). That relaxation is only warranted for the bakery, which is
                # a *series* of such rounds (repeated bake-the-cookies days, win ->
                # new day, lose -> repeat). Collect these separately and only fold
                # them into the result if several share the scene; a lone 0x46
                # content-fork elsewhere (e.g. As Time Goes By) is not a bakery and
                # is left undetected, exactly as before.
                _bakery_forks.append(_g)
            else:
                out.append(_g)
            break
    # Fold in the 0x46 bakery rounds only when several share a scene (the
    # repeated-day signature). This keeps a single 0x46 content-fork in another
    # episode from being detected at all, preserving prior output there.
    if len(_bakery_forks) >= 3:
        for g in _bakery_forks:
            g["bakery"] = True
            out.append(g)
        out.sort(key=lambda g: int(g["trigger"]))
    return out


def resolve_score_tier_cascade(data, start_off=None, window=220):
    """Execute a score-tier RANK cascade the way the VM would, returning the tiers.

    The end-of-episode rank screen is a chain of score comparisons, each guarding
    one rank narration, in bytecode:

        op5f op3f            ; load the running score
        push <N>             ; a tier threshold
        <cmp>  (0x0e..0x0f)  ; score <cmp> N   (lte in the rank screen)
        2b <rel=7>           ; JMPF: if false, skip this rank's narration
        1b <rankref> <..>    ; the rank text ref (1st byte of the pair)
        1f00 .. 1f41         ; show that one rank

    A LINEAR scan emits every narration; the real VM takes the JMPF and shows
    exactly ONE. This reads the chain statically and returns the mutually-
    exclusive tiers [{op, threshold, text}], plus the fall-through `else` rank,
    so the flattened narrations can be collapsed into a single gated node.

    When start_off is None the whole scene is scanned for the first cascade of
    3+ tiers; otherwise the scan begins at start_off.

    Returns {"tiers": [...], "else_text": <str|None>} or None.
    """
    if data[:4] != b"kiwi":
        return None
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    smap = {o: t for o, t in normalize_strings(strings)}

    def s(r):
        return smap.get(r * 2 + STRING_BASE, "")

    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}

    def _scan_from(i0, end):
        tiers, offsets = [], []
        i = i0
        _last_i = None
        while i < len(fo) - 3 and fo[i] < end:
            o = fo[i]
            if data[o] == 0x1a:
                thr = (data[o + 1] << 8) | data[o + 2]
                cmp_b = data[fo[i + 1]]
                if cmp_b in (0x0e, 0x0f, 0x0c, 0x0d) and data[fo[i + 2]] == 0x2b:
                    op = {0x0e: "lt", 0x0f: "lte", 0x0c: "gt", 0x0d: "gte"}[cmp_b]
                    txt = None
                    for j in range(i + 3, i + 7):
                        if j >= len(fo):
                            break
                        if data[fo[j]] == 0x1b:
                            txt = s(data[fo[j] + 1])
                            break
                        if data[fo[j]] == 0x1a:
                            rr = (data[fo[j] + 1] << 8) | data[fo[j] + 2]
                            if s(rr):
                                txt = s(rr)
                                break
                    if txt and ("rank for this episode" in txt.lower()
                                or "grade for this episode" in txt.lower()):
                        # Tiers of one cascade are tightly packed (~23 instrs
                        # apart). A large gap means we have run past this cascade
                        # into a later one (e.g. the extra-credit rescore) -- stop
                        # so each cascade is resolved separately.
                        if _last_i is not None and (i - _last_i) > 20:
                            break
                        tiers.append({"op": op, "threshold": thr, "text": txt})
                        offsets.append(fo[i + 3] if i + 3 < len(fo) else o)
                        _last_i = i
            i += 1
        return tiers, offsets

    if start_off is not None and start_off in fi:
        tiers, offsets = _scan_from(fi[start_off], start_off + window)
    else:
        # scan the whole scene; take the first run of >=3 rank tiers
        tiers, offsets = [], []
        k = 0
        while k < len(fo):
            _t, _o = _scan_from(k, len(data))
            if len(_t) >= 3:
                tiers, offsets = _t, _o
                break
            if _t:
                # skip past this short run and keep looking
                k = fi.get(_o[-1], k + 1) + 1
            else:
                k += 1
    if len(tiers) < 3:
        return None
    # The fall-through (else) rank is the pushed rank string right after the last
    # tier -- the "perfect score" / top grade shown when no threshold matched.
    else_text = None
    if offsets:
        li = fi.get(offsets[-1], 0)
        for j in range(li, min(li + 24, len(fo))):
            if data[fo[j]] == 0x1a:
                v = (data[fo[j] + 1] << 8) | data[fo[j] + 2]
                t = s(v)
                if t and ("grade for this episode" in t.lower()
                          or "rank for this episode" in t.lower()) and t not in \
                        [ti["text"] for ti in tiers]:
                    else_text = t
                    break
    # Also collect every rank-threshold binding anywhere in the scene (a rescore
    # screen repeats the ranks with == tests), so a caller can map ANY rank text
    # to its score threshold, not just the first cascade's.
    all_thr = {}
    kk = 0
    while kk < len(fo) - 3:
        o = fo[kk]
        if data[o] == 0x1a:
            thr = (data[o + 1] << 8) | data[o + 2]
            cmp_b = data[fo[kk + 1]]
            if cmp_b in (0x0e, 0x0f, 0x0c, 0x0d, 0x0a) and data[fo[kk + 2]] == 0x2b:
                op = {0x0e: "lt", 0x0f: "lte", 0x0c: "gt", 0x0d: "gte",
                      0x0a: "eq"}[cmp_b]
                for j in range(kk + 3, kk + 7):
                    if j >= len(fo):
                        break
                    rr = None
                    if data[fo[j]] == 0x1b:
                        rr = data[fo[j] + 1]
                    elif data[fo[j]] == 0x1a:
                        rr = (data[fo[j] + 1] << 8) | data[fo[j] + 2]
                    if rr is not None:
                        t = s(rr)
                        if t and ("rank for this episode" in t.lower()
                                  or "grade for this episode" in t.lower()):
                            all_thr.setdefault(t, {"op": op, "threshold": thr})
                            break
        kk += 1
    return {"tiers": tiers, "else_text": else_text, "all_thresholds": all_thr}


def resolve_section_dispatch(data):
    """Resolve a scene's SECTION DISPATCH header, derived from the bytecode +
    the native VM (libshs09.so, FUN_0009fe3c).

    A continuation scene that hosts several mutually-exclusive sections opens
    with a single read of a register (var 1001 here) followed by a switch:

        push <reg> ; 1f 2d ; 0x21 <k> ; (0x5c <key> <case_id>)* ; 0x28 ...

    VM facts established from the decompile:
      * 0x5c / 0x12 / 0x21 are NOT executable opcodes -- they fall through the
        VM's default case. They are switch-table markers consumed by the host.
      * Each 0x5c row pairs an internal KEY with a global CASE_ID. The keys are
        local slots (key = 3*slot + 5 -> slot 0,1,2,..), and
        case_id = slot + chapter_base, where chapter_base is the first row's
        case_id.
      * The register VALUE equals the CASE_ID it selects (byte-confirmed across
        chapters): section_for(value) == the case whose case_id == value.

    What is NOT statically recoverable is the case_id -> body byte-offset binding
    (the 0x12 opcode computes the jump target at runtime; it is not consistently
    derivable from offset order). So this returns the dispatch STRUCTURE -- the
    register, the valid values (== case_ids), and the candidate section bodies --
    and leaves the value->offset binding to an overlay (a handful of observed
    numbers), rather than guessing it.

    Returns {register, transform_operand, chapter_base, cases:[{value,key}],
             candidate_sections:[{offset, first_line, exit}]} or None.
    """
    if data[:4] != b"kiwi":
        return None
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    smap = {o: t for o, t in normalize_strings(strings)}
    n = len(data)
    # locate: push <reg> ; 1f 2d ; 0x21 <op> ; 0x5c rows -- within the scene head
    reg = transform = None
    rows = []
    q = bc
    while q < min(bc + 500, n - 4):
        # The dispatch header normally sits at the scene head, but a scene with
        # a large data preamble before it (e.g. Making Some Dough's bakery, whose
        # baking-minigame option tables push the var1001 switch ~400 bytes past
        # the bytecode start) needs a wider window than the typical ~260. First
        # match still wins, so scenes whose table is early are unaffected.
        if data[q] in (0x1a, 0x41) and data[q + 3] == 0x1f and data[q + 4] == VAR_READ:
            cand = (data[q + 1] << 8) | data[q + 2]
            r = q + 6                      # after push(3) + 1f 2d argc(3)
            if r < n and data[r] == 0x21:
                reg, transform = cand, data[r + 1]
                r += 2
                while r + 2 < n and data[r] == 0x5c:
                    rows.append({"key": data[r + 1], "case_id": data[r + 2]})
                    r += 3
                if rows:
                    break
        q += 1
    if reg is None or not rows:
        return None
    base = rows[0]["case_id"]
    valid = [rw["case_id"] for rw in rows]

    # BYTE-DERIVED SEP-TABLE BINDINGS -- the dispatch table sitting just before
    # the 0x12 opcode is a run of paired `28` fall-through skips + `42 29` SEP
    # case targets. Each SEP's operand resolves via the fold-instruction jump
    # rule to that case body's entry offset; the SEPs appear in bytecode order
    # matching valid_values in order (rows[i].case_id -> SEPs[i]'s target).
    # This gives us every case, not just the natural landings' second body.
    # ATGB's later ATGB cases already have overlay bindings; the overlay merge
    # in apply_overlay preserves the overlay value when both are present, so
    # this only ADDS coverage for cases lacking any prior binding.
    bindings = []
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    disp_i = fi.get(disp_off) if (disp_off := next(
        (p for p in range(bc, n - 1) if data[p] == 0x1f and data[p + 1] == 0x12),
        None)) is not None else None
    sep_targets = []
    if disp_i is not None:
        j = disp_i - 1
        while j >= 0:
            q = fo[j]
            op0 = data[q]
            if op0 == 0x42 and q + 1 < n and data[q + 1] == 0x29:
                sep_op = (data[q + 2] << 8) | data[q + 3]
                tgt = fo[fi[q] + sep_op] if (fi[q] + sep_op) < len(fo) else None
                sep_targets.append(tgt)
            elif op0 == 0x28:
                pass                          # fall-through skip: not a case body
            else:
                break                         # left the dispatch table
            j -= 1
        sep_targets.reverse()
    # Bind SEPs to case_ids one-to-one when the counts match, but only when
    # the dispatch has 2+ cases: a single-case dispatch's one body IS the
    # scene entry (no runtime lookup needed), and a SEP binding would split
    # the scene into two segments (entry + case body) with nothing routing
    # to the entry (regressing Tutors scene 1/2 for example).
    # Snap a mid-scene case body back to the title card that opens it. A SEP
    # operand points at the section's first *runtime* instruction, which is often
    # a few bytes PAST the chapter title card (the compiler emits `push title;
    # 1f08 TITLE; push bg; 1f0b` then the SEP lands on the line after). Entering at
    # the SEP offset would skip the title card, orphaning it (e.g. ATGB's
    # "Sophomore year / Dating Holly" card). If, walking back from the SEP target
    # to the nearest section marker, the section opens with a title card (1f08),
    # move the binding to that section start so the title shows. Only applied to
    # non-entry cases (idx>0): the first case is the scene's own entry, already a
    # reachable root, and must not be shifted.
    def _snap_to_title(tgt):
        markers = _section_markers(data, bc)
        before = [m for m in markers if m < tgt]
        if not before:
            return tgt
        m = before[-1]
        if not (4 <= tgt - m <= 400):
            return tgt
        c = m + 4
        while c < n and c not in fi:
            c += 1
        cc = c
        for _ in range(8):
            if cc >= n or cc not in fi or cc >= tgt:
                break
            if data[cc] == 0x1f and data[cc + 1] == 0x08:   # title card
                return c                                    # section's first instr
            i = fi.get(cc)
            cc = fo[i + 1] if i is not None and i + 1 < len(fo) else n
        return tgt

    # The dispatch instruction offset, used to measure the gap to value-0's body.
    _disp_off = next((p for p in range(bc, n - 1)
                      if data[p] == 0x1f and data[p + 1] == 0x12), None)

    def _says_between(lo, hi):
        # count spoken lines (1f0d) strictly between two offsets -- a proxy for
        # "a real section of content sits here", not just var/bg setup.
        return sum(1 for q in fo if lo < q < hi
                   and data[q] == 0x1f and data[q + 1] == 0x0d)

    def _gate_before(tgt):
        # If a variable-gate (`push VAR ; 1f2d ; ... ; CMP ; 0x2b JMPF`) sits
        # immediately before a section body such that entering AT the body would
        # skip the gate (landing inside its then-arm), return the gate's JUMP
        # offset so the section binds to the gate segment (whose own node carries
        # that offset) and the gate evaluates -- keeping both arms reachable.
        # Only triggers when the JMPF's else-target diverts OUTSIDE the section
        # body region (the gate genuinely branches to other content, e.g.
        # Halloween scene 4's `var2002 == 1` gate whose else-arm is the twins
        # reunion). Byte-derived: gate, var-read, and jump target all from bytes.
        ti = fi.get(tgt)
        if ti is None:
            return None
        for k in range(ti - 1, max(ti - 12, -1), -1):
            o = fo[k]
            if data[o] == 0x2b:                       # JMPF (the gate branch)
                jt = resolve_jump_target(data, o, bc_start=bc)
                if jt is None or jt <= tgt:           # else-arm not diverting away
                    return None
                # The else-arm must be a SUBSTANTIAL diverting section, not a small
                # in-line conditional skip. A section-entry gate whose else-arm is a
                # whole alternate scene (Halloween s4's var2002 gate: else-arm 452
                # bytes away, the twins-reunion) is worth entering at; a local
                # `skip a couple of lines` gate (e.g. Making Some Dough scene 2's
                # value-1 arm, whose gate skips ~18 bytes) is not -- backing the
                # section binding onto it would displace the dispatch arm. Require
                # the else-target to sit well past the gate AND to contain spoken
                # lines (a real scene), both byte-derived.
                if jt - o < 100:
                    return None
                _spoken = sum(1 for _q in fo if o < _q < jt
                              and data[_q] == 0x1f and data[_q + 1] == 0x0d)
                if _spoken < 3:
                    return None
                # confirm a var-read feeds this gate
                has_vread = False
                for m in range(k - 1, max(k - 8, -1), -1):
                    om = fo[m]
                    if data[om] == 0x1f and om + 1 < n and data[om + 1] == VAR_READ:
                        has_vread = True
                        break
                    if data[om] == 0x2b:
                        break
                return o if has_vread else None
            if data[o] == 0x1f and data[o + 1] in (0x0d, 0x41):
                return None                            # spoken line: not gate prologue
        return None

    if len(sep_targets) == len(rows) and len(rows) >= 2:
        for idx, (row, tgt) in enumerate(zip(rows, sep_targets)):
            if tgt is None:
                continue                      # sentinel case (game-end / no body)
            if idx > 0:
                tgt = _snap_to_title(tgt)
            elif idx == 0:
                _g = _gate_before(tgt)
                if _g is not None:
                    tgt = _g
            b = {"value": row["case_id"], "section_offset": tgt,
                 "source": "bytecode",
                 "_note": ("Byte-derived from the dispatch table's "
                           "0x42 SEP case target (fold-instruction "
                           "jump); the SEP's operand is the case's "
                           "body entry offset, snapped back to the "
                           "section's title card when it opens with one. "
                           "SEPs run in bytecode order matching "
                           "valid_values in order.")}
            # Split the scene at each subsequent case body so a `goto_scene`
            # from a choice can target the right case directly. The FIRST
            # case (rows[0]) is normally the natural fall-through of the scene
            # entry and needs no cut; adding one would leave the scene entry
            # empty (e.g. ATGB scene 4 with only 2 cases would orphan s4).
            #
            # EXCEPTION: when substantial spoken content sits between the
            # dispatch and value-0's body, that content is a SEPARATE section
            # (e.g. Wrong Side of Town's opening combat minigame) reached by
            # dispatch fall-through, not by the scene entry. Entering at the
            # bytecode top would start the episode mid-minigame and skip the
            # title sequence. Detect this (many 1f0d lines in the gap) and make
            # value 0 an enter_at_section landing too, so the scene entry lands
            # on the real opening. Normal scenes have 0-1 setup lines in this gap.
            if idx > 0:
                b["enter_at_section"] = True
            elif _disp_off is not None and _says_between(_disp_off, tgt) >= 4:
                b["enter_at_section"] = True
                b["_note"] += (" Scene entry lands here (not the bytecode top): "
                               "the intervening spoken section is reached by "
                               "dispatch fall-through, not at entry.")
            bindings.append(b)
    elif len(rows) >= 2:
        # Fallback for scenes where the SEP-table shape isn't found (dispatch
        # was located but the table walk didn't recover it cleanly): use the
        # legacy natural-landings binding for just the second case body.
        landings = _natural_section_landings(data, bc)
        if len(landings) >= 2:
            bindings.append({"value": rows[1]["case_id"],
                             "section_offset": landings[1], "source": "bytecode",
                             "_note": ("Byte-derived natural dispatch landing "
                                       "(second section body in bytecode order); "
                                       "enters at the section's true first "
                                       "instruction rather than skipping it.")})

    out = {"register": reg, "transform_operand": transform,
           "chapter_base": base, "valid_values": valid,
           "cases": [{"value": rw["case_id"], "key": rw["key"]} for rw in rows],
           "note": ("Section switch read from the scene header. The register "
                    "VALUE equals the CASE_ID it selects (value==case_id, "
                    "byte-confirmed across chapters). The set of valid_values "
                    "is the register values that route here. value->section "
                    "bindings are byte-derived from the dispatch table's SEP "
                    "case targets (in bytecode order); observed overrides may "
                    "be supplied by an overlay when the byte-derived entry "
                    "point differs from the segment's title-card offset.")}
    if bindings:
        out["bindings"] = bindings
    return out


def _natural_section_landings(data, bc):
    """Section bodies in bytecode order for a section-dispatch scene.

    body #0 opens right after the 0x12 dispatch (the first 1f08/1f0b after it);
    each later body is the dead-code block immediately after a `goto_scene`
    (1f 0a) -- i.e. a block with no inbound jump that the host VM's 0x12 reaches
    by its computed jump. Returns offsets in order.
    """
    n = len(data)
    io = _instruction_offsets(data, bc)
    inb = set()
    p = bc
    while p < n:
        op = data[p]
        if op in (0x2b, 0x28):
            t = resolve_jump_target(data, p, io)
            if t:
                inb.add(t)
        p += _instr_len(data, p)
    landings = []
    # first body: first title/background op after the 0x12 dispatch
    p = bc
    while p < n - 1:
        if data[p] == 0x1f and data[p + 1] == 0x12:
            q = p + 2
            while q < n and not (data[q] == 0x1f and data[q + 1] in (0x08, 0x0b)):
                q += 1
            if q < n:
                landings.append(q)
            break
        p += 1
    # subsequent bodies: each dispatch-section body opens with the 4-byte
    # section-entry marker `22 43 48 4a`, placed right after the previous
    # section's terminator -- a `goto_scene` (1f 0a) or a 0x42 SEP. (The orphan
    # block heuristic used to pick the dead code after ANY goto, which wrongly
    # caught choice-branch outcomes that share a goto's tail -- e.g. the Sophie
    # BREAKUP, which has no entry marker and is a choice branch, not a section.)
    # The marker after the dispatch's own jump table is body #0, already found
    # above, so only markers following a goto/SEP are taken here.
    p = bc
    last_term = None
    while p < n - 4:
        op = data[p]
        if op == 0x1f and p + 1 < n and data[p + 1] in (0x0a, 0x12):
            last_term = p + 2          # goto_scene / dispatch: a section boundary
            p += 2
            continue
        if op == 0x42:
            last_term = p + 4          # SEP: a section boundary
            p += 4
            continue
        if (data[p] == 0x22 and data[p + 1] == 0x43
                and data[p + 2] == 0x48 and data[p + 3] == 0x4a):
            if last_term is not None and 0 <= p - last_term <= 4:
                landings.append(p)
            last_term = None
            p += 4
            continue
        p += 1
    return landings


def _score_win_flows_to_fail(data, jmp, block_lo, block_hi, smap, bc_start):
    """True when an equality score-gate's WIN block flows straight into a fail
    screen -- the mis-merge signature that distinguishes a genuine win/fail gate
    from an ordinary `score == 0` choice-answer test.

    Walks the win block's execution from block_lo, following linear flow and
    unconditional 0x28 jumps, and stops at the first choice (1f 01), goto-scene
    (1f 0a), or the block's natural exit. If a "you have failed" / "has failed"
    string is displayed along that path, the win narration was concatenated with
    the fail screen (Magic School's enrollment and beast-QTE dodges) and the gate
    should be split. Choice-test `== 0` gates branch away before any fail line, so
    they return False and are left untouched."""
    n = len(data)
    fail_refs = set()
    for o, t in smap.items():
        lo = t.lower()
        if "have failed" in lo or "has failed" in lo:
            fail_refs.add((o - STRING_BASE) // 2)
    pos, steps, seen = block_lo, 0, set()
    while steps < 500:
        steps += 1
        if pos in seen or pos >= n or pos < 0:
            break
        seen.add(pos)
        b = data[pos]
        if b == 0x1a:
            r = (data[pos + 1] << 8) | data[pos + 2]
            if r in fail_refs:
                return True
            pos += 3
        elif b == 0x28:                         # unconditional jump: follow it
            t = resolve_jump_target(data, pos, bc_start=bc_start)
            pos = t if t is not None else pos + 3
        elif b == 0x1f and pos + 1 < n and data[pos + 1] == 0x01:
            return False                        # choice prompt: branch point
        elif b == 0x1f and pos + 1 < n and data[pos + 1] == 0x0a:
            return False                        # goto-scene: leaves the chunk
        elif b in (0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
            pos += 3
        else:
            pos += 1
    return False


def resolve_score_gates(data, minigame_gate_offsets=None):
    """Resolve mid-scene SCORE-THRESHOLD gates: `op5f op3f ; push N ; <cmp> ; 2b`.

    Some scenes decide a win/fail outcome on the accumulated minigame SCORE (the
    `op5f`/`op3f` running-score load) rather than a stored variable -- e.g. Fallon
    Family Christmas's intruder fight, which gates the "police training kicks in,
    Mal wins" narration on `score >= 2`, jumping to the "He has failed" checkpoint
    branch otherwise. A linear scan emits both the win and fail narration back to
    back (making the win text run straight into the defeat text); this reads the
    gate statically so the two outcomes become the `then`/`else` arms of a gate
    node, exactly like `resolve_var_gates`.

    Shape (byte-verified, in the same 3-byte-call instruction model the other
    resolvers use):
        op5f            ; push accumulated score
        op3f            ; load it
        push <N> | 5a|5b; the threshold
        <cmp>           ; 0x0c gt / 0x0d gte / 0x0e lt / 0x0f lte
        2b <skip>       ; JMPF: skip the fall-through block when the test is false

    Only threshold comparisons (gt/gte/lt/lte) are treated as score gates; an
    equality (0x0a) after `op5f op3f` is a choice test block (option-index match)
    handled by resolve_choice_branches, and is left alone. `minigame_gate_offsets`
    (from resolve_minigame_gates) name gates the minigame-gate machinery already
    resolves -- those are skipped so this resolver only picks up the win/fail
    gates nothing else handles. Returns the same {"at", "var": "score",
    "equals": N, "op", "then": [lo, hi]} shape as resolve_var_gates so downstream
    gate-nesting consumes it identically."""
    if data[:4] != b"kiwi":
        return []
    _skip = set(minigame_gate_offsets or ())
    strings = find_strings(data)
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    ins, p = [], bc_start
    while p < n:
        op = data[p]
        if op in (0x1a, 0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
            if p + 3 > n:
                break
            ins.append(p)
            p += 3
        else:
            ins.append(p)
            p += 1
    index = {q: i for i, q in enumerate(ins)}
    out = []
    i = 0
    while i < len(ins) - 4:
        q = ins[i]
        # score load: op5f then op3f
        if not (data[q] == 0x5f and data[ins[i + 1]] == 0x3f):
            i += 1
            continue
        const, jmp, cmp_op = None, None, None
        for j in range(i + 2, min(i + 6, len(ins))):
            b = data[ins[j]]
            if b in (0x1a, 0x41):
                const = (data[ins[j] + 1] << 8) | data[ins[j] + 2]
            elif b == 0x5a:
                const = 0
            elif b == 0x5b:
                const = 1
            elif b in (0x0c, 0x0d, 0x0e, 0x0f):     # threshold comparisons
                cmp_op = b
            elif b == 0x0a:                          # equality: only a win/fail
                cmp_op = b                           # gate if it flows to a fail
                                                     # screen (checked below)
            elif b == 0x0b:                          # != : choice inversion, skip
                break
            elif b == 0x2b:
                jmp = ins[j]
                break
            elif b == 0x1f:
                break
        if const is None or jmp is None or cmp_op is None:
            i += 1
            continue
        # skip gates the minigame-gate machinery already resolves (its `gate`
        # offset is the 0x2b of this same test), so we don't double-split them
        if jmp in _skip or q in _skip:
            i += 1
            continue
        v = (data[jmp + 1] << 8) | data[jmp + 2]
        ti = index[jmp] + v
        if 0 <= ti < len(ins) and ins[ti] > jmp:
            block_lo, block_hi = ins[index[jmp] + 1], ins[ti]
            # Only treat this as a win/fail narrative gate when the fall-through
            # block carries real, non-rank narration. The end-of-episode RANK
            # cascade (`score <= 19/39/59...` -> "Your rank for this episode is
            # ...") uses the same op5f/op3f/threshold shape but is handled by
            # resolve_score_tier_cascade; its tier bodies are tiny and hold only
            # rank/points text. Requiring a substantive narrative body keeps this
            # resolver from double-claiming those tiers.
            if not hasattr(resolve_score_gates, "_smap_cache") \
                    or resolve_score_gates._smap_cache[0] is not data:
                resolve_score_gates._smap_cache = (
                    data, {o: t for o, t in strings})
            _smap = resolve_score_gates._smap_cache[1]
            narrative = []
            has_choice = False
            bo = block_lo
            while bo < block_hi:
                if data[bo] == 0x1f and bo + 1 < n and data[bo + 1] == 0x01:
                    has_choice = True        # a choice prompt lives in this block
                if data[bo] == 0x1a:
                    r = (data[bo + 1] << 8) | data[bo + 2]
                    t = _smap.get(r * 2 + STRING_BASE, "")
                    if t and len(t) > 8:
                        narrative.append(t)
                    bo += 3
                elif data[bo] in (0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
                    bo += 3
                else:
                    bo += 1
            is_rank = any("rank for this episode" in t.lower()
                          or "grade for this episode" in t.lower()
                          for t in narrative)
            # Require the fail arm (the jump target and just past it) to carry the
            # checkpoint-replay structure: a `1f63` CHECKPOINT syscall and/or a
            # "has failed" / "you have earned ... points" fail screen. This is what
            # distinguishes a genuine minigame WIN/FAIL gate (Fallon's intruder
            # fight) from an ordinary `op5f`-counter availability gate (Making Some
            # Dough reuses op5f/op3f as a day counter, `< 3`, with no fail screen),
            # so the split never fires on those and leaves every other episode's
            # segmentation byte-identical.
            fail_lo = block_hi
            fail_hi = min(n, block_hi + 900)
            has_ckpt = False
            fail_text = []
            fo = fail_lo
            while fo < fail_hi:
                b = data[fo]
                if b == 0x1f and fo + 1 < n and data[fo + 1] == 0x63:
                    has_ckpt = True
                if b == 0x1a:
                    r = (data[fo + 1] << 8) | data[fo + 2]
                    t = _smap.get(r * 2 + STRING_BASE, "")
                    if t:
                        fail_text.append(t)
                    fo += 3
                elif b in (0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
                    fo += 3
                else:
                    fo += 1
            has_fail_screen = any("has failed" in t.lower()
                                  or "you have earned" in t.lower()
                                  for t in fail_text)
            # For a THRESHOLD gate (Fallon's `score >= 2` fight), the checkpoint/
            # fail signature is enough. For an EQUALITY gate (`score == 0`), that
            # shape is far more common -- most choice-answer tests use it -- so a
            # much stricter test is required: the WIN block's own linear execution
            # (following unconditional jumps, stopping at any choice/goto/branch)
            # must actually reach a "you have failed" line. That only happens when
            # the win narration was mis-concatenated with the fail screen in one
            # segment (Magic School's "you sign your name / ...you have failed"
            # enrollment, and its beast-QTE dodges). Choice-test `== 0` gates in
            # other episodes branch away before any fail line, so they never match
            # and every other episode stays byte-identical.
            is_eq = cmp_op == 0x0a
            accept = (narrative and not is_rank and not has_choice)
            if is_eq:
                accept = accept and _score_win_flows_to_fail(
                    data, jmp, block_lo, block_hi, _smap, bc_start)
            else:
                accept = accept and (has_ckpt or has_fail_screen)
            if accept:
                out.append({"at": jmp, "var": "score", "equals": const,
                            "op": {0x0a: "eq", 0x0c: "gt", 0x0d: "gte",
                                   0x0e: "lt", 0x0f: "lte"}[cmp_op],
                            "then": [block_lo, block_hi]})
                i = index[jmp] + 1
                continue
        i += 1
    return out


def resolve_var_gates(data):
    """Resolve in-script VARIABLE GATES: `1f 2d` (read var) ; 0x21 ;
    <const push / 0x5a / 0x5b> ; 0x0a (compare) ; 0x2b <skip>.
    Semantics (verified against init guards and observed gameplay): the block
    after the 0x2b runs when var == const; the jump skips it otherwise.

    Returns [{"at": jump_off, "var": v, "equals": c,
              "then": [block_lo, block_hi]}] with block_hi = the jump target.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    ins, p = [], bc_start
    while p < n:
        op = data[p]
        if op in (0x1a, 0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
            if p + 3 > n:
                break
            ins.append(p)
            p += 3
        else:
            ins.append(p)
            p += 1
    index = {q: i for i, q in enumerate(ins)}
    out = []
    i = 0
    while i < len(ins) - 4:
        q = ins[i]
        if not (data[q] == 0x1f and data[q + 1] == VAR_READ):
            i += 1
            continue
        # the variable id: last push before the read
        var = None
        for j in range(i - 1, max(0, i - 4), -1):
            if data[ins[j]] in (0x1a, 0x41):
                var = (data[ins[j] + 1] << 8) | data[ins[j] + 2]
                break
        # scan forward a short window: const push, compare (0x0a == or 0x0e <),
        # 0x2b jump. The compare op decides the gate's semantics: 0x0a tests
        # equality (fall-through when var == const), 0x0e tests less-than
        # (fall-through when var < const). Both jump over the fall-through block
        # when the test is false. Distinguishing them is essential: e.g. Making
        # Some Dough's hub gates the academic-challenge option on `var2002 < 3`
        # (0x0e), not `== 3` -- reading it as == inverts day-1 availability.
        const, jmp, cmp_op = None, None, None
        for j in range(i + 1, min(i + 6, len(ins))):
            b = data[ins[j]]
            if b in (0x1a, 0x41):
                const = (data[ins[j] + 1] << 8) | data[ins[j] + 2]
            elif b == 0x5a:
                const = 0
            elif b == 0x5b:
                const = 1
            elif b in (0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f):
                cmp_op = b
            elif b == 0x2b:
                jmp = ins[j]
                break
            elif b == 0x1f:        # another call intervenes: not a gate
                break
        if var is None or const is None or jmp is None:
            i += 1
            continue
        v = (data[jmp + 1] << 8) | data[jmp + 2]
        ti = index[jmp] + v
        if 0 <= ti < len(ins) and ins[ti] > jmp:
            g = {"at": jmp, "var": var, "equals": const,
                 "then": [ins[index[jmp] + 1], ins[ti]]}
            if cmp_op == 0x0e:
                # less-than gate: the fall-through block plays when var < const.
                g["op"] = "lt"
            elif cmp_op == 0x0d:
                # greater-than-or-equal gate: the fall-through block plays when
                # var >= const (e.g. Making Some Dough's ending money check
                # `var2000 >= 200` -> the success path). The jump skips it when
                # the test fails (money < 200 -> failure setup).
                g["op"] = "gte"
            elif cmp_op == 0x0c:
                # greater-than: fall-through when var > const. (Wrong Side of
                # Town's walk time-tier boundary `var2007 > 6`.)
                g["op"] = "gt"
            elif cmp_op == 0x0f:
                # less-than-or-equal: fall-through when var <= const (e.g. a
                # health check `var2001 <= 0` -> defeat).
                g["op"] = "lte"
            elif cmp_op == 0x0b:
                # not-equal: fall-through when var != const.
                g["op"] = "ne"
            # cmp_op 0x0a (==) is the default gate shape; leaving op unset keeps
            # the existing equality semantics that downstream code assumes.
            out.append(g)
            i = index[jmp] + 1
            continue
        i += 1
    return out


def resolve_sep_cascade_gates(data):
    """Resolve NESTED section dispatches that use a `2b`/SEP if-elif-else chain
    instead of the top-level `0x12` switch.

    Some scenes route sub-sections with a secondary dispatch nested inside a
    top-level case body -- e.g. Making Some Dough's quiz, whose first case
    (the intro) is followed by a var2002 cascade selecting which of three
    daily question-sets to play. Its byte shape (verified in scene 2/4):

        push VAR ; 1f2d (read) ; 0x21 <t> ; 0x0a (compare) ; 0x2b op=SKIP
            0x42 SEP op=T0            <- arm 0 body target (var == 0)
            0x28 ; 0x28              <- fall-through skip pair
        push VAR ; 1f2d ; 0x21 <t> ; 0x0a ; 0x2b op=SKIP
            0x42 SEP op=T1            <- arm 1 body target (var == 1)
            0x28 ; 0x28
            0x42 SEP op=T2            <- final arm body target (var == 2), no 2b
        0x22 0x43 0x48 0x4a          <- section marker terminates the cascade

    Each SEP operand resolves (fold-instruction jump) to that arm's body; the
    arm ordinal IS the compared value (0, 1, 2, ...). Unlike resolve_var_gates
    (which handles `2b` guarding an INLINE then-block), here the `2b` guards a
    SEP jump to a separate body, so the arms are addressable section targets.

    Returns [{"at": read_off, "var": v, "cases": [{"equals": k,
              "section_offset": off}, ...]}] -- all byte-derived, no overlay.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}

    def jtgt(joff, op):
        i = fi.get(joff)
        if i is None or i + op >= len(fo) or i + op < 0:
            return None
        return fo[i + op]

    out = []
    consumed = set()
    for idx, q in enumerate(fo):
        if q in consumed:
            continue
        # Anchor: a var read (push VAR ; 1f2d) that is immediately followed by
        # a compare + 2b + SEP (the cascade's first arm). Require the var push
        # right before the read.
        if not (data[q] == 0x1f and data[q + 1] == VAR_READ):
            continue
        if idx == 0:
            continue
        pv = fo[idx - 1]
        if data[pv] not in (0x1a, 0x41):
            continue
        var = (data[pv + 1] << 8) | data[pv + 2]
        # Walk the cascade arms. Each arm: [21 t] 0a 2b op ; SEP op ; 28 ; 28
        cases = []
        j = idx + 1
        arm_val = 0
        ok = True
        anchor_read = q
        while j < len(fo):
            # optional transform 0x21 <t>, then an optional 0x5a/0x5b value
            # marker (the compared constant: 5a=0, 5b=1, or a bare push), then
            # the 0x0a/0x0e compare, then the 0x2b guard.
            k = j
            if k < len(fo) and data[fo[k]] == 0x21:
                k += 1
            if k < len(fo) and data[fo[k]] in (0x5a, 0x5b, 0x1a, 0x41):
                k += 1
            if k < len(fo) and data[fo[k]] in (0x0a, 0x0e):
                k += 1
            if k >= len(fo) or data[fo[k]] != 0x2b:
                # No guard here -> this is the FINAL (else) arm: expect a lone SEP.
                kk = j
                while kk < len(fo) and data[fo[kk]] != 0x42 and data[fo[kk]] != 0x22:
                    kk += 1
                if kk < len(fo) and data[fo[kk]] == 0x42:
                    tgt = jtgt(fo[kk], (data[fo[kk] + 2] << 8) | data[fo[kk] + 3])
                    if tgt is not None:
                        cases.append({"equals": arm_val, "section_offset": tgt})
                    consumed.add(fo[kk])
                break
            guard = fo[k]                       # the 0x2b
            sep_i = k + 1
            if sep_i >= len(fo) or data[fo[sep_i]] != 0x42:
                ok = False
                break
            sep_off = fo[sep_i]
            tgt = jtgt(sep_off, (data[sep_off + 2] << 8) | data[sep_off + 3])
            if tgt is None:
                ok = False
                break
            cases.append({"equals": arm_val, "section_offset": tgt})
            consumed.add(sep_off)
            arm_val += 1
            # After the SEP: a `28 ; 28` skip pair, then either the next arm's
            # `push VAR ; 1f2d` or a final lone SEP / marker.
            m = sep_i + 1
            while m < len(fo) and data[fo[m]] == 0x28:
                m += 1
            # Next arm must re-read the SAME var, else the cascade ends.
            if (m + 1 < len(fo) and data[fo[m]] in (0x1a, 0x41)
                    and ((data[fo[m] + 1] << 8) | data[fo[m] + 2]) == var
                    and data[fo[m + 1]] == 0x1f and data[fo[m + 1] + 1] == VAR_READ):
                j = m + 2
                continue
            # Or a final lone SEP arm (the else), then stop.
            if m < len(fo) and data[fo[m]] == 0x42:
                tgt2 = jtgt(fo[m], (data[fo[m] + 2] << 8) | data[fo[m] + 3])
                if tgt2 is not None:
                    cases.append({"equals": arm_val, "section_offset": tgt2})
                consumed.add(fo[m])
            break
        # Only accept a real cascade: 2+ distinct arms with distinct targets.
        if ok and len(cases) >= 2:
            seen_t = set()
            uniq = []
            for c in cases:
                if c["section_offset"] in seen_t:
                    continue
                seen_t.add(c["section_offset"])
                uniq.append(c)
            if len(uniq) >= 2:
                out.append({"at": anchor_read, "var": var, "cases": uniq})
    # A cascade's 2nd..Nth arms each re-read the var, so the scan re-anchors on
    # them and produces sub-cascades that are strict suffixes of the first. Keep
    # only maximal cascades: drop any whose target set is a subset of an earlier
    # (longer) one for the same var.
    out.sort(key=lambda c: (-len(c["cases"]), c["at"]))
    kept = []
    for c in out:
        tset = {x["section_offset"] for x in c["cases"]}
        if any(c["var"] == k["var"]
               and tset <= {x["section_offset"] for x in k["cases"]}
               for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c["at"])
    return kept


def _scan_timed_choices(data):
    """Return the timed (quick-time) choices in a scene's bytecode, by offset.

    Action/reflex choices (Swim's "Slow!|Rush!" ledge crossing, Halloween's
    "Scream!|Duck!") are presented on a countdown: the player has a fixed time to
    pick, and letting it expire drops to the non-ideal ("wrong") branch. In the
    bytecode a timed choice is the normal `1f 01` present-choice preceded by a
    distinctive setup:

        0x5f ; push "Hurry!" ; push <options "a|b"> ; push <prompt> ;
        push <TIMER_MS> ; push 1000 ; 1b <emo> <spk> ; const1 ; 1f 01

    The `"Hurry!"` urgency string is the reliable marker (a plain choice uses
    "Make your choice!" and has no bare timer push), and the bare number pushed
    just after the prompt is the countdown in milliseconds (2000-6000 observed;
    the trailing 1000 is a fixed secondary parameter, constant across all timed
    choices, not the duration). Returns {choice_offset: {timer_ms}} for each timed
    `1f 01`; byte-derived, empty for scenes with none. The correct-vs-timeout
    routing is the choice's own compare/branch (already emitted); a timeout takes
    the same non-ideal branch a wrong pick does.
    """
    if data[:4] != b"kiwi":
        return {}
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    smap = {o: t for o, t in strings}
    fo = _fold_instruction_offsets(data, bc)
    out = {}
    for idx, q in enumerate(fo):
        if not (data[q] == 0x1f and data[q + 1] == PRESENT_CHOICE):
            continue
        # gather the string/number pushes in the setup window before the choice
        pushed = []
        for j in range(max(0, idx - 12), idx):
            p = fo[j]
            if data[p] == 0x1a:
                pushed.append((data[p + 1] << 8) | data[p + 2])
        has_hurry = any(smap.get(v * 2 + STRING_BASE) == "Hurry!" for v in pushed)
        if not has_hurry:
            continue
        # timer = the number pushed immediately before the trailing constant 1000
        # (uniform layout: ...<options> <prompt> <TIMER_MS> <1000> 1b const1 1f01).
        # The slot POSITION defines the timer, so trust its value even when it also
        # happens to resolve as a valid string reference (a coincidental collision:
        # e.g. Swim's "We're a team!" carries 4000 there, which is both the 4000 ms
        # countdown and, by chance, a string offset). Only reject it when the slot
        # value is out of the plausible countdown range.
        timer = None
        for k in range(len(pushed) - 1, 0, -1):
            if pushed[k] == 1000:
                prev = pushed[k - 1]
                if 1000 < prev <= 20000:
                    timer = prev
                break
        # the option labels (from the "a|b|c" options string) for matching an
        # offset-less choice node back to this timed choice
        labels = None
        for v in pushed:
            t = smap.get(v * 2 + STRING_BASE)
            if t and "|" in t:
                labels = t.split("|")
                break
        # Distinct timeout branch: some timed choices test the choice result
        # against the sentinel 1000 (the value returned when the clock expires)
        # with `push 1000 ; 0x0a (EQ) ; 0x2b`, giving the timeout its OWN outcome
        # distinct from any deliberate pick (e.g. Swim's "We're a team!" times out
        # into Emily fumbling "Because, uh..." rather than the pointed "I'm better
        # than you!" line). Locate that test in the choice's outcome region and
        # record the byte offset the timeout path begins at (the fall-through just
        # past the test), so segmentation can point on_timeout at the right node.
        # When no such test exists, the timeout is equivalent to the non-ideal
        # pick and on_timeout stays the choice's default branch.
        timeout_offset = None
        for j in range(idx + 1, min(idx + 90, len(fo))):
            p = fo[j]
            if data[p] == 0x1a and ((data[p + 1] << 8) | data[p + 2]) == 1000 \
                    and j + 1 < len(fo) and data[fo[j + 1]] == 0x0a \
                    and j + 2 < len(fo) and data[fo[j + 2]] in (0x28, 0x2b):
                # the EQ+branch tests result==1000; the timeout path is the
                # fall-through right after the branch instruction
                if j + 3 < len(fo):
                    timeout_offset = fo[j + 3]
                break
            # stop if we run into the next choice/section marker
            if data[p] == 0x1f and data[p + 1] == PRESENT_CHOICE and p != q:
                break
        out[q] = {"timer_ms": timer, "labels": labels,
                  "timeout_offset": timeout_offset}
    return out


def _scan_checkpoint_replays(data):
    """Return the checkpoint SET and REPLAY records for a scene's bytecode.

    The two-scene episodes (Tutors, Swim, Halloween) carry a checkpoint system on
    opcode 0x63 (`1f 63`). Two shapes matter:

      * SET: `push 2001 ; const0 ; 1f2c (write) ; [more resets] ; 1f63 ; SEP`
        commits a checkpoint and zeroes the run counters (var2001, often var2000).
        The resets are the `push VAR ; const0 ; 1f2c` writes just before the 1f63.
      * REPLAY: `1f63 ; SEP op=N` reached from a "Replay from checkpoint" choice.
        The SEP operand N is an instruction count from the bytecode start (same
        encoding resolve_sep_target uses): target = fold[fold_index(bc) + N]. This
        is the story offset the replay jumps back to (the last passed checkpoint,
        e.g. Swim's "Emily reaches the far side").

    Returns {"sets": [{offset, target, resets:[var,...]}],
             "replays": [{offset, target}]}. Byte-derived; empty for scripts with
    no 0x63 checkpoints. A REPLAY is distinguished from a SET by the absence of a
    var2001 reset in the few instructions preceding the 1f63.
    """
    if data[:4] != b"kiwi":
        return {"sets": [], "replays": []}
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    bc_idx = fi.get(bc, 0)
    sets, replays = [], []
    for idx, q in enumerate(fo):
        if not (data[q] == 0x1f and data[q + 1] == 0x63):
            continue
        if idx + 1 >= len(fo) or data[fo[idx + 1]] != 0x42:
            continue                          # inline marker, no jump target
        aq = fo[idx + 1]
        op = (data[aq + 2] << 8) | data[aq + 3]
        ti = bc_idx + op
        target = fo[ti] if 0 <= ti < len(fo) else None
        if target is None:
            continue
        # collect `push VAR ; const0 ; 1f2c` resets in the window before the 1f63
        resets = []
        for j in range(max(0, idx - 6), idx):
            p = fo[j]
            if data[p] == 0x1a and j + 2 < len(fo) \
                    and data[fo[j + 1]] in (0x5a, 0x5b) \
                    and data[fo[j + 2]] == 0x1f and data[fo[j + 2] + 1] == 0x2c:
                resets.append((data[p + 1] << 8) | data[p + 2])
        if 2001 in resets:
            sets.append({"offset": q, "target": target, "resets": resets})
        else:
            replays.append({"offset": q, "target": target})
    # Skill-check FAIL sections. A skill-check minigame (Swim's ledge crossing)
    # commits a checkpoint, then on SUCCESS sets the section register and jumps to
    # the next scene; the very next section body in bytecode order is the FAILURE
    # screen ("She has failed."), reached only when the host VM's skill loop is
    # exhausted rather than passing. That failure landing is not a static jump
    # target -- the minigame VM routes to it -- so it surfaces as an orphaned
    # section. Record the pattern (a section marker immediately following a
    # `1f63 CHECKPOINT ; ... ; goto_scene`) so segmentation can mark that section
    # as a runtime-reachable entry IF it is otherwise orphaned. Byte-derived; the
    # orphan test (applied later, with the full graph) keeps this from firing on
    # checkpoints whose following section is already reached normally.
    markers = _section_markers(data, bc)
    marker_set = set(markers)
    fail_sections = []
    for idx, q in enumerate(fo):
        if not (data[q] == 0x1f and data[q + 1] == 0x63):
            continue
        goto_off = None
        j = idx + 1
        while j < len(fo) and fo[j] < q + 80:
            qq = fo[j]
            if data[qq] == 0x1f and data[qq + 1] == 0x0a:   # goto_scene
                goto_off = qq
            if qq in marker_set:
                if goto_off is not None and qq - goto_off < 8:
                    # section body starts a few instrs past the marker
                    c = qq + 4
                    while c < len(data) and c not in fi:
                        c += 1
                    fail_sections.append(c)
                break
            j += 1
    # General case (byte-derived, no 1f63 required): any section body that the
    # forward simulator cannot reach from the chunk entry but which sits just
    # after a `goto scene` is a section-register RESUME target -- the engine
    # re-enters the chunk at that section head when the scene round-trips back.
    # This recovers score-gated fail screens and post-round-trip bonus sections
    # (e.g. Swim's "She has failed", Dough's Sublime Snickerdoodle) that the
    # narrow 1f63 shape above misses. The downstream orphan guard still applies,
    # so already-reachable sections are never touched.
    for _rs in resolve_scene_resume_sections(data, bc):
        if _rs["resume_at"] not in fail_sections:
            fail_sections.append(_rs["resume_at"])
    return {"sets": sets, "replays": replays,
            "fail_sections": fail_sections}


def _scan_ad_flag_writes(data):
    """Return the per-ad done-flag writes (var2005/2006/2007) in bytecode order.

    Each Help Ad marks itself done, on entry, by setting a per-ad-pair flag var
    (2005, 2006, or 2007) to a distinguishing value (two ads share each var, one
    using value 1 and the other value 2). The write is a small increment/const
    store: `push VAR ; 1f2d (read) ; ... ; push V ; (0x50 ADD | const) ; 1f2c
    (write)`. This surfaces them, in order, so segmentation can pair each with the
    ad whose listing immediately follows it and emit a `done_when` guard. All
    byte-derived; returns [] for scripts with no such flags.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    out = []
    for idx, q in enumerate(fo):
        if data[q] != 0x1a:
            continue
        var = (data[q + 1] << 8) | data[q + 2]
        if var not in (2005, 2006, 2007):
            continue
        if idx + 1 >= len(fo) or data[fo[idx + 1]] != 0x1f \
                or data[fo[idx + 1] + 1] != 0x2d:
            continue                         # must be a read of the flag var
        # the added/stored value and confirmation of a following write
        val, saw_write = None, False
        for j in range(idx + 2, min(idx + 9, len(fo))):
            qq = fo[j]
            if data[qq] == 0x1a and val is None:
                val = (data[qq + 1] << 8) | data[qq + 2]
            elif data[qq] == 0x5b and val is None:
                val = 1
            elif data[qq] == 0x5a and val is None:
                val = 0
            elif data[qq] == 0x1f and data[qq + 1] == 0x2c:
                saw_write = True
                break
        if saw_write and val is not None and val < 100:
            out.append({"offset": q, "var": str(var), "value": int(val)})
    return out


def resolve_random_selector(data):
    """Detect a RANDOM section selector and return its option set.

    Some scenes route to one of several sections by a runtime *random draw*
    rather than a deterministic dispatch -- e.g. Making Some Dough's Help Ads,
    which on each visit shows a randomly-chosen not-yet-completed helpee ad, and
    falls through to a "no more ads" section once every ad is done. This can't be
    represented as a single static `next` (there is no one correct target), so it
    is emitted as a `random` node: the engine performs the draw at runtime.

    Byte shape (verified in scene 3): a run of near-identical blocks, each

        push STATE ; 1f2d (read) ; 0x21 <t> ; push K ; 0x0a ; 0x11   <- guard on
                                                                        STATE (which
                                                                        ads remain)
        0x2b op=SKIP                                     <- guard fails -> next block
        ... push m ; 0x1f 0x1b (RANDOM roll) ...        <- 1f1b = random draw
        0x0a ; 0x2b op=SKIP2                             <- roll picks an arm
            0x42 SEP -> option A body
            0x28 ; 0x28
            0x42 SEP -> option B body
        (fall through to the next block)

    After the last block, control falls through to the EXHAUSTED section (the
    "no more ads" body). Everything the node carries is byte-derived: the state
    var (the read var id), each option's target (the SEP fold-jump), and the
    exhausted fall-through (the terminal 28/2b landing past the last block).

    Returns a list of selectors:
        [{"at": first_block_off, "state_var": v,
          "options": [{"section_offset": off}, ...],
          "exhausted_offset": off_or_None}]
    or [] when no random selector is present.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}

    def jtgt(joff, op):
        i = fi.get(joff)
        if i is None or i + op >= len(fo) or i + op < 0:
            return None
        return fo[i + op]

    # Locate every RANDOM roll (1f 1b) that is followed, within a few folded
    # instructions, by at least one SEP -- that pairing is the selector's
    # per-block "roll then jump to a random arm" core.
    roll_blocks = []
    for idx, q in enumerate(fo):
        if not (data[q] == 0x1f and data[q + 1] == 0x1b):
            continue
        seps = []
        state_var = None
        # SEP targets shortly after the roll (each block offers 2 random arms,
        # the second sitting ~12 folded instructions out past a 28/28 skip pair).
        for j in range(idx + 1, min(idx + 16, len(fo))):
            qq = fo[j]
            if data[qq] == 0x42:
                t = jtgt(qq, (data[qq + 2] << 8) | data[qq + 3])
                if t is not None:
                    seps.append(t)
            elif data[qq] == 0x1f and data[qq + 1] == 0x1b:
                break                          # next roll block starts
        # the STATE var guarding this block: the nearest preceding `push V;1f2d`
        for j in range(idx - 1, max(0, idx - 12), -1):
            qq = fo[j]
            if data[qq] == 0x1f and data[qq + 1] == VAR_READ and j >= 1 \
                    and data[fo[j - 1]] in (0x1a, 0x41):
                pv = fo[j - 1]
                state_var = (data[pv + 1] << 8) | data[pv + 2]
                break
        if seps:
            roll_blocks.append({"roll_at": q, "state_var": state_var,
                                "seps": seps})
    if not roll_blocks:
        return []

    # A single selector is a contiguous run of roll blocks sharing one state var.
    # Group by state var and adjacency (each block's roll is within a small
    # instruction span of the previous).
    roll_blocks.sort(key=lambda b: b["roll_at"])
    selectors = []
    cur = None
    for b in roll_blocks:
        if cur and b["state_var"] == cur["state_var"] \
                and fi[b["roll_at"]] - fi[cur["_last_roll"]] <= 40:
            cur["options"].extend(b["seps"])
            cur["_last_roll"] = b["roll_at"]
        else:
            if cur:
                selectors.append(cur)
            cur = {"at": b["roll_at"], "state_var": b["state_var"],
                   "options": list(b["seps"]), "_last_roll": b["roll_at"]}
    if cur:
        selectors.append(cur)

    out = []
    for sel in selectors:
        opts = []
        seen = set()
        for off in sel["options"]:
            if off in seen:
                continue
            seen.add(off)
            opts.append({"section_offset": off})
        if len(opts) < 2:
            continue                           # a real random draw needs 2+ arms
        # Exhausted fall-through: when the LAST block's availability guard fails
        # (no un-done ad remains for that arm), control skips past the roll to
        # the "no more ads" body. That landing is the target of the guard `2b`
        # immediately preceding the last roll -- a backward-independent, byte-
        # derived pointer (e.g. scene 3's @12671 2b -> @12717, the s3_alt head).
        last_roll = sel["_last_roll"]
        li = fi[last_roll]
        exhausted = None
        for j in range(li - 1, max(0, li - 10), -1):
            qq = fo[j]
            if data[qq] == 0x2b:
                exhausted = jtgt(qq, (data[qq + 1] << 8) | data[qq + 2])
                break
        # Max draws are determined at node-build time from the number of guarded
        # options (see build_story_segments): every option plays once, so the pool
        # empties after len(options) draws. The counter-var equality test in the
        # bytecode counts from a staged sub-structure and reads short of the true
        # total, so it is NOT used as the cap.
        out.append({"at": sel["at"], "state_var": sel["state_var"],
                    "options": opts, "exhausted_offset": exhausted})
    return out


def resolve_dynamic_status_vars(data):
    """Resolve which variable fills the `%d` in each dynamic status line.

    HUD status lines like "Kim has %d dollars." / "Kim has %d days left." are
    formatted at runtime from a variable the bytecode reads just before building
    the string (push VAR ; 1f2d read ; 0x21 <fmt> ; ... display). Without the
    variable id the engine can't know whether `%d` is money or the day counter.
    Recover it per line: the format may also apply a constant (e.g. days-left is
    `10 - var2001`, pushed as `push 10 ; push VAR ; read`), which is returned too.

    Returns {display_offset: {"var": id, "consts": [k, ...]}} for each %d line.
    """
    if data[:4] != b"kiwi":
        return {}
    strings = find_strings(data)
    ss = {o: t for o, t in strings}
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    out = {}
    for idx, q in enumerate(fo):
        # a display/narration call that renders a formatted string
        if not (data[q] == 0x1f and data[q + 1] in (0x00, 0x0d, 0x41)):
            continue
        # walk back for the nearest variable read; a %d line reads its value just
        # before formatting, within a short window of the display call
        var, consts, multiply = None, [], None
        for j in range(idx - 1, max(idx - 14, -1), -1):
            qq = fo[j]
            if data[qq] == 0x1f and data[qq + 1] == VAR_READ \
                    and j >= 1 and data[fo[j - 1]] in (0x1a, 0x41):
                var = (data[fo[j - 1] + 1] << 8) | data[fo[j - 1] + 2]
                for k in range(j - 2, max(j - 5, -1), -1):
                    if data[fo[k]] in (0x1a, 0x41):
                        cv = (data[fo[k] + 1] << 8) | data[fo[k] + 2]
                        if cv < 2000:
                            consts.append(cv)
                break
            # a scale factor applied to the value: `push K ; 0x52` (standalone
            # multiply, not the `1f 52` stop-music builtin) between the read and
            # the display. Halloween Dance shows the score as `var2000 * 5` so it
            # reads out of 100; capture the factor so the engine scales too.
            if data[qq] == 0x52 and (qq == 0 or data[qq - 1] != 0x1f) \
                    and j >= 1 and data[fo[j - 1]] in (0x1a, 0x41):
                mk = (data[fo[j - 1] + 1] << 8) | data[fo[j - 1] + 2]
                if 0 < mk < 2000:
                    multiply = mk
            # stop if we cross another display call (previous line's scope)
            if data[qq] == 0x1f and data[qq + 1] in (0x00, 0x0d, 0x41):
                break
        if var is not None and var >= 2000:
            rec = {"var": var, "consts": consts}
            if multiply is not None:
                rec["multiply"] = multiply
            out[q] = rec
    return out


def resolve_quiz_random(data):
    """Detect the academic-quiz question rolls and return their question pools.

    Unlike the Help Ads' random selector (which guards on a "which items remain"
    state var so it never repeats an item), the quiz's per-section roll is a
    plain 1-of-N draw with no availability tracking: each section (History,
    Science, English) rolls once and shows one of a small pool of questions. The
    byte shape (verified in Making Some Dough scene 2):

        push N ; 1f 1b (RANDOM roll of N)
        push 0 ; 0x0a ; 0x2b -> skip      <- roll == 0 ? fall through to question 0
        <question 0 body ...>
        push 1 ; 0x0a ; 0x2b -> skip      <- roll == 1 ? fall through to question 1
        <question 1 body ...>
        ...                               <- the last skip lands on the roll == N-1
                                             question (the final fall-through arm)

    Each roll-compare is distinguished from a question's own answer-check by
    sitting directly on the roll chain before any 0x01 CHOICE. Returns a list of
        [{"at": roll_off, "options": [question_body_off, ...]}]
    one per section, or [] when the scene has no such rolls.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}

    def jtgt(joff):
        i = fi.get(joff)
        if i is None:
            return None
        v = (data[joff + 1] << 8) | data[joff + 2]
        ti = i + v
        return fo[ti] if 0 <= ti < len(fo) else None

    out = []
    for idx, q in enumerate(fo):
        if not (data[q] == 0x1f and data[q + 1] == 0x1b):
            continue
        options, last_skip = [], None
        j, steps = idx + 1, 0
        while j < len(fo) and steps < 160:
            qq = fo[j]
            # a roll-compare arm: <const> ; 0x0a ; 0x2b
            if (data[qq] in (0x5a, 0x5b) or data[qq] == 0x1a) \
                    and j + 2 < len(fo) and data[fo[j + 1]] == 0x0a \
                    and data[fo[j + 2]] == 0x2b:
                skip = jtgt(fo[j + 2])
                fall = fo[j + 3] if j + 3 < len(fo) else None
                if fall is not None:
                    options.append(fall)
                last_skip = skip
                if skip is not None and fi.get(skip, j) > j:
                    j = fi[skip]
                    steps += 1
                    continue
            # a CHOICE without a preceding roll-compare -> we've left the roll chain
            if data[qq] == 0x1f and data[qq + 1] == 0x01:
                break
            if data[qq] == 0x1f and data[qq + 1] == 0x1b:
                break                              # next section's roll
            j += 1
            steps += 1
        # the final skip lands on the last question (roll == N-1)
        if last_skip is not None and last_skip not in options:
            options.append(last_skip)
        # A genuine quiz question roll is a SMALL pool (a handful of questions),
        # and each arm leads to a player CHOICE (the answer options). Reject rolls
        # that don't match: large fan-outs and choice-less arms are other 1f1b
        # uses (e.g. The Tutors' 25-way visual/shuffle roll), not question draws.
        if not (2 <= len(options) <= 6):
            continue

        def _arm_has_choice(start):
            # a 0x01 CHOICE appears within a short span of the arm head, before the
            # next arm's roll-compare
            for k in range(fi.get(start, 0),
                           min(fi.get(start, 0) + 60, len(fo))):
                oq = fo[k]
                if data[oq] == 0x1f and data[oq + 1] == 0x01:
                    return True
                if data[oq] == 0x1f and data[oq + 1] == 0x1b:
                    break
            return False

        if not all(_arm_has_choice(o) for o in options):
            continue
        out.append({"at": q, "options": options})
    return out


def resolve_choice_branches(data):
    """Resolve a script's choice DISPATCH structure from the bytecode.

    The jump encoding (verified against known branch boundaries in several
    episodes): a conditional/unconditional jump's 16-bit operand is a distance
    in INSTRUCTIONS, with target = instruction_index(jump) + operand.
    Instruction sizes: 0x1a/0x41 push, 0x1b pair, 0x28/0x2b jump and 0x42 are
    3 bytes; 0x1f <sel> <argc> calls are 3 bytes; everything else is 1 byte.

    A choice (call 0x01) is followed by one TEST BLOCK per option:
        <0x5f|0x60> 0x3f <index> 0x0a ; 0x2b <skip>
    (0x5f, 0x60 and 0x61 all occur as the test-block prefix, interchangeably and in
    every episode -- byte-verified to have the identical block shape.)
    where <index> is 0x5a (push 0), 0x5b (push 1) or an ordinary pushed
    constant, and the 0x2b jumps to the NEXT test block (or to the merge)
    when the player picked a different option. Each option's branch body runs
    from after its test to that jump's target; a trailing 0x28 in the body
    jumps to the common MERGE point and is trimmed from the body.

    Returns [{"choice": off, "options": [{"index": i, "region": [lo, hi]}],
              "merge": off}] -- all byte offsets, fully derived from the bytes.
    """
    if data[:4] != b"kiwi":
        return []
    strings = find_strings(data)
    bc_start = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    ins, p = [], bc_start
    while p < n:
        op = data[p]
        if op in (0x1a, 0x41, 0x1b, 0x28, 0x2b, 0x42, 0x1f):
            if p + 3 > n:
                break
            ins.append(p)
            p += 3
        else:
            ins.append(p)
            p += 1
    index = {p2: i for i, p2 in enumerate(ins)}

    def jump_target(p2):
        v = (data[p2 + 1] << 8) | data[p2 + 2]
        ti = index[p2] + v
        return ins[ti] if 0 <= ti < len(ins) else None

    out = []
    for i, p2 in enumerate(ins):
        if data[p2] != 0x1f or data[p2 + 1] != PRESENT_CHOICE:
            continue
        options, merge = [], None
        j = i + 1
        while j < len(ins) - 5:
            q = ins[j]
            # Two shapes for a choice test block, both terminating in the same
            # `<marker> 0a ; 2b <skip>` sequence:
            #   STANDARD (5-instr):  <5f|60|61> 3f <marker> 0a ; 2b <skip>
            #   SHORT    (3-instr):                 <marker> 0a ; 2b <skip>
            # The prefix on the standard form (0x5f/0x60/0x61 + 0x3f) appears to
            # gate on a re-comparable value; without it the short form still
            # branches genuinely -- verified across ATGB scene 2 ("Pick a
            # strategy... Go for a sneak attack / Get Spud to help") and Tutors
            # ("What do you say? Academics / Girls"). The short form's marker
            # sits at instruction j; the standard form's marker sits at j+2.
            is_std = (data[q] in (0x5f, 0x60, 0x61)
                      and data[ins[j + 1]] == 0x3f)
            is_short = (not is_std
                        and data[q] in (0x5a, 0x5b, 0x1a, 0x41)
                        and data[ins[j + 1]] == 0x0a
                        and data[ins[j + 2]] == 0x2b)
            if not (is_std or is_short):
                if options or ins[j] - p2 > 24:
                    break       # tests sit right after the call / prior jump
                j += 1
                continue
            k = j + 2 if is_std else j     # the option-index constant
            b = data[ins[k]]
            if b == 0x5a:
                const = 0
            elif b == 0x5b:
                const = 1
            elif b in (0x1a, 0x41):
                const = (data[ins[k] + 1] << 8) | data[ins[k] + 2]
                if const > 15:
                    break       # a large constant is a variable gate inside a
                                # branch (e.g. a money check), not an option test
            else:
                break
            # Comparison op sits right after the marker: 0x0a (==) is the usual
            # test, but some choices encode an option with 0x0b (!=), which
            # INVERTS the branch -- the JMPF then fires when the answer DOES match
            # (jumping to that option's body) and falls through for the others.
            # (Byte-verified: Fallon Family Christmas scene 2's intruder QTE
            # "Duck and dive! / Jump to the side!" encodes option 1 as `push1 !=`,
            # so its body is the jump target, not the fall-through.) Track it so
            # the body region is taken from the correct arm.
            _cmp_off = ins[k + 1] if k + 1 < len(ins) else None
            is_ne = _cmp_off is not None and data[_cmp_off] == 0x0b
            jmp = next((ins[m] for m in range(k + 1, min(k + 5, len(ins)))
                        if data[ins[m]] == 0x2b), None)
            if jmp is None:
                break
            fail = jump_target(jmp)
            if fail is None or fail <= jmp:
                break
            if is_ne:
                # `!= ` inverts: the option body is the JUMP TARGET (taken when
                # the answer matches), and control falls through to the rest.
                body_lo = fail
                nxt_ins = ins[index[jmp] + 1]
                # the fall-through arm (nxt_ins .. body_lo) is the other option/
                # else; it usually ends in a 0x28 to the merge -- that jump's
                # target is the merge point bounding this option's body.
                merge_guess = None
                for m in ins[index[nxt_ins]:index[fail]]:
                    if data[m] == 0x28:
                        t = jump_target(m)
                        if t is not None and t >= fail:
                            merge_guess = t
                if merge is None and merge_guess is not None:
                    merge = merge_guess
                body_hi = merge or fail
                # trim a trailing exit 0x28 inside this option's body
                last28 = max((m for m in ins[index[fail]:index[body_hi]]
                              if data[m] == 0x28), default=None) \
                    if body_hi in index else None
                if last28 is not None:
                    t = jump_target(last28)
                    if t is not None and t >= body_hi \
                            and index[body_hi] - index[last28] <= 2:
                        body_hi = last28
                options.append({"index": const, "region": [body_lo, body_hi]})
                # The fall-through arm here is the impossible "answer is neither
                # option" case (option 0 was already matched by its own earlier
                # test, and this test catches option 1 via `!=`), so it is dead
                # code -- a freeze/miss branch the engine can never execute. It is
                # NOT emitted as an option: doing so would orphan real-looking
                # text that no choice path reaches. (Byte-verified across Fallon
                # scene 2 and Halloween Part 2's "Scream!/Duck!" QTE.)
                if merge is None:
                    merge = body_hi
                break
            body_lo = ins[index[jmp] + 1]
            body_hi = fail
            last28 = max((m for m in ins[index[jmp] + 1:index[fail]]
                          if data[m] == 0x28), default=None)
            if last28 is not None:
                tgt = jump_target(last28)
                if tgt is not None and tgt >= body_hi:
                    if merge is None:
                        merge = tgt
                    # trim the exit jump only when nothing follows it (interior
                    # blocks may each jump to the merge with content after)
                    if index[fail] - index[last28] <= 2:
                        body_hi = last28
            options.append({"index": const, "region": [body_lo, body_hi]})
            if data[fail] in (0x5f, 0x60, 0x61):  # chained: next option's test block
                j = index[fail]
                continue
            # No further test: the remaining option(s) fall through here as an
            # ELSE branch, running from the fail target to the merge. (A 0x28
            # at the very start means the branch is empty -- it adds no lines.)
            if merge is None:
                merge = fail
            elif fail < merge:
                lo2 = fail
                if data[lo2] == 0x28 and (jump_target(lo2) or 0) >= merge:
                    lo2 = ins[index[lo2] + 1]
                last28 = max((m for m in ins[index[fail]:index[merge]]
                              if data[m] == 0x28 and m >= lo2), default=None)
                hi2 = merge
                if last28 is not None:
                    tgt = jump_target(last28)
                    if tgt is not None and tgt >= merge \
                            and index[merge] - index[last28] <= 2:
                        hi2 = last28
                if hi2 > lo2:
                    # PHANTOM-ELSE GUARD: when the "else" region abuts the last
                    # explicit option's region (else.lo == last_opt.hi), the
                    # else is a bytecode artifact -- a chained test that always
                    # falls through once the earlier options have been ruled out
                    # (e.g. Halloween's "Hurry!" QTE: after opt1's exit jump to
                    # the merge, there's still a redundant test block for opt0
                    # followed by more opt0 content). Merging both regions into
                    # the last explicit option keeps that option's real body
                    # intact instead of orphaning the tail as a nameless "else"
                    # branch (which has no cast option to attach it to).
                    if (options and isinstance(options[-1].get("index"), int)
                            and options[-1]["region"][1] == lo2):
                        options[-1]["region"][1] = hi2
                    else:
                        options.append({"index": "else", "region": [lo2, hi2]})
            break
        if options:
            out.append({"choice": p2, "options": options, "merge": merge})
    return out


def extract_minigame_banks(data):
    """Parse the minigame DATA BANKS stored at a script's head.

    These banks are read by the engine's minigame loop, not by a display
    opcode, so they don't surface as ordinary nodes -- but every field is a
    literal string in the .kiw; nothing here is inferred. Two formats:

      build_word -- 6-string records:
          [subtitle_a, subtitle_b, Word1, Word2, 'w1|w2', letter_pool]
        e.g. ['Stir and mix','it all up!','Stir','Mix','stir|mix','stirmxuf'].
        The player spells each target word from the scrambled letter pool.

      pick_word  -- 3-string records:
          [subtitle (contains 'Pick'), 'right1|right2', 'wrong1|wrong2|...']
        e.g. ["Dodge the attacks! Pick 'duck' and 'dive'!", 'Duck!|Dive!',
              'Dang!|Drat!|Dang!|Drat!']. The player picks the correct words.

    Anchored on each record's pipe-delimited string, with the surrounding
    fields read by fixed position. Returns {} for scripts with no bank.
    """
    if data[:4] != b"kiwi":
        return {}
    # The bank sits at the head; stop at the first prose line (a real
    # narration/dialogue sentence never matches a bank record's pattern).
    strings = []
    for o, t in normalize_strings(find_strings(data)):
        if len(t) > 45 and " " in t and "|" not in t and "pick" not in t.lower():
            break
        strings.append((o, t))

    build = []
    for i, (o, t) in enumerate(strings):
        if not re.fullmatch(r'[a-z]+\|[a-z]+', t):
            continue
        if i + 1 >= len(strings) or not re.fullmatch(r'[a-z]+', strings[i + 1][1]):
            continue
        if i < 4:
            continue
        subtitle = (strings[i - 4][1].lstrip("@") + " " + strings[i - 3][1]).strip()
        build.append({"offset": strings[i - 4][0], "subtitle": subtitle,
                      "words": [strings[i - 2][1], strings[i - 1][1]],
                      "answers": t.split("|"), "letters": strings[i + 1][1]})

    # SINGLE-word build rounds use the same 6-string record but carry a question
    # instead of a second word, and so have no `w1|w2` string to anchor on:
    #     [action_a, action_b, question_a, question_b, word, letter_pool]
    #   e.g. ['Paint the','coulds!','What color are','clouds?','white','whitexk']
    # The letter pool is the target word plus a few decoy letters, which is what
    # identifies the pair (pool starts with the word and is a little longer).
    if not build:
        for i, (o, t) in enumerate(strings):
            if i < 4 or i + 1 >= len(strings):
                continue
            nxt = strings[i + 1][1]
            if not (re.fullmatch(r'[a-z]{3,}', t) and re.fullmatch(r'[a-z]{4,}', nxt)):
                continue
            if not (nxt.startswith(t) and 1 <= len(nxt) - len(t) <= 4):
                continue
            action = (strings[i - 4][1].lstrip("@") + " " + strings[i - 3][1]).strip()
            question = (strings[i - 2][1] + " " + strings[i - 1][1]).strip()
            build.append({"offset": strings[i - 4][0], "subtitle": action,
                          "question": question, "words": [t.capitalize()],
                          "answers": [t], "letters": nxt})

    pick = []
    i = 0
    while i < len(strings):
        # Pick-word records are runs of CONSECUTIVE pipe-strings, alternating
        # (correct, decoys) per round: a one-round game is 2 strings ('Duck!|
        # Dive!' then 'Dang!|Drat!|Dang!|Drat!'); the 3-round ingredients game
        # is 6. In every record checked the decoy string lists each decoy MORE
        # THAN ONCE -- the game fills the board with duplicate wrong answers.
        # Requiring a repeated decoy item rejects the other adjacent pipe-pairs
        # in scripts: choice-menu option strings (e.g. the day hub's task
        # menus), which never repeat an item. The nearest non-pipe string
        # before the run is the on-screen subtitle, shared by its rounds.
        if "|" not in strings[i][1]:
            i += 1
            continue
        run_start = i
        while i < len(strings) and "|" in strings[i][1]:
            i += 1
        run = strings[run_start:i]
        if len(run) < 2:
            continue
        subtitle = next((s for _, s in reversed(strings[:run_start])
                         if "|" not in s), None)
        for j in range(0, len(run) - 1, 2):
            (o, c), (_, d) = run[j], run[j + 1]
            decoy_items = d.split("|")
            if len(decoy_items) == len(set(decoy_items)):
                continue                    # no repeats -> a menu, not a pick bank
            correct = c.split("|")
            decoys = []
            for x in decoy_items:
                if x not in decoys and x not in correct:
                    decoys.append(x)
            pick.append({"offset": o, "subtitle": subtitle,
                         "correct": correct, "decoys": decoys})

    banks = {}
    if build:
        banks["build_word"] = build
    if pick:
        banks["pick_word"] = pick

    # QUESTION BANKS -- two further engine-loop record shapes, scanned over the
    # whole string region (not just the head):
    #   quiz     -- [question_long?, question_short?, option, option, ...]:
    #               two ADJACENT '?'-strings followed by the answer options
    #               (e.g. "What does the word 'abhor' mean?" / 'What does abhor
    #               mean?' / 'To hate.' / 'To love.' / ...). The options'
    #               on-screen order/correctness is NOT asserted here.
    #   question_options -- a '?'-string followed by ONE pipe-string of options
    #               (e.g. the end-of-episode survey), included only when the
    #               question string is never referenced by a bytecode push --
    #               push-referenced question+options pairs are ordinary choice
    #               prompts and already surface as choice nodes.
    all_strings = normalize_strings(find_strings(data))
    bc_start = max((o + len(t) for o, t in all_strings if len(t) >= 12), default=0)
    refs = set()
    p2 = bc_start
    n2 = len(data)
    while p2 < n2 - 2:
        op2 = data[p2]
        if op2 in (0x1a, 0x41):
            refs.add(((data[p2 + 1] << 8) | data[p2 + 2]) * 2 + STRING_BASE)
            p2 += 3
        elif op2 == 0x1b:
            refs.add(data[p2 + 1] * 2 + STRING_BASE)
            refs.add(data[p2 + 2] * 2 + STRING_BASE)
            p2 += 3
        elif op2 in (0x28, 0x2b, 0x42, 0x1f):
            p2 += 3
        else:
            p2 += 1
    region = [(o, t) for o, t in all_strings if 14 < o < bc_start and len(t) >= 2]

    def is_q(t):
        return t.rstrip().endswith("?")

    def _is_quiz_feedback(t):
        # Strings the engine shows AFTER an answer is picked -- they terminate a
        # quiz record's option list (the last record in a bank has no following
        # question to bound it, so without this the loop runs on into the feedback
        # banners and then ordinary story dialogue: The Tutors' "mitigate" record
        # was absorbing "Good answer!", "The next morning...", etc.). Matched as
        # anchored phrases, not loose "right"/"wrong" substrings, so a genuine
        # answer option like "To get a question wrong in a quiz." is NOT treated
        # as feedback.
        low = t.strip().lower()
        if low in ("correct!", "choose wisely!", "good answer!",
                   "that's right!", "sorry, that's wrong.",
                   "sorry, that's incorrect.", "sorry, that's not right.",
                   "that's definitely not right..."):
            return True
        return low.startswith(("%s the correct answer",
                               "the correct answer was"))

    def _rephrasing(a, b):
        """A quiz record's short question is a REPHRASING of the long one --
        its tokens are (nearly) a subset of the long form's (e.g. "What does
        the word 'abhor' mean?" -> 'What does abhor mean?'; 'Which of the
        following is a factor of the sum of 15 + 45?' -> 'A factor of the sum
        of 15 + 45?'). Two adjacent DIALOGUE questions share almost nothing."""
        tok = lambda x: {w.strip("'`\".,!?()").lower() for w in x.split()
                         if w.strip("'`\".,!?()")}
        wa, wb = tok(a), tok(b)
        if not wb:
            return False
        return len(wa & wb) / len(wb) >= 0.6

    quiz = []
    i2 = 0
    while i2 < len(region) - 2:
        o, t = region[i2]
        if is_q(t) and is_q(region[i2 + 1][1]) and "|" not in t \
                and "|" not in region[i2 + 1][1] \
                and _rephrasing(t, region[i2 + 1][1]):
            opts2, j2 = [], i2 + 2
            while j2 < len(region) \
                    and "|" not in region[j2][1] and len(region[j2][1]) < 60 \
                    and not _is_quiz_feedback(region[j2][1]):
                # A "?"-string normally marks the next question -- but a quiz answer
                # can itself be phrased as a question (e.g. the joke option
                # "Norway?"). The genuine next record is a question immediately
                # followed by its short rephrasing; a lone "?" answer is not. Stop
                # only at a real question pair, so joke options are kept.
                if is_q(region[j2][1]):
                    _nxt = region[j2 + 1][1] if j2 + 1 < len(region) else ""
                    if is_q(_nxt) and _rephrasing(region[j2][1], _nxt):
                        break
                opts2.append(region[j2][1])
                j2 += 1
            if len(opts2) >= 2:
                quiz.append({"offset": o, "question": t,
                             "question_short": region[i2 + 1][1],
                             "options": opts2})
                i2 = j2
                continue
        i2 += 1
    if len(quiz) >= 2:                  # a real bank is a series of records
        banks["quiz"] = quiz

    qopts = []
    for i2 in range(len(region) - 1):
        o, t = region[i2]
        o2, t2 = region[i2 + 1]
        if is_q(t) and "|" not in t and "|" in t2 and o not in refs:
            items = t2.split("|")
            if len(items) == len(set(items)) and len(items) >= 2:
                qopts.append({"offset": o, "question": t, "options": items})
    if qopts:
        banks["question_options"] = qopts
    return banks


def _branch_alt_sections(data):
    """Find conditional-branch ALTERNATE sections that would otherwise orphan.

    Pattern (Wrong Side of Town's scene-3 journey gate): a 0x2b conditional
    jump whose FALL-THROUGH arm exits the scene via `goto_scene` (the "leave"
    path), while the JUMP-TARGET arm flows a few instructions later into a
    section marker that begins a distinct section body (the "continue" path,
    e.g. the timed journey / multi-round fight minigame). Because the
    fall-through ends in a scene goto, the segmenter treats that goto as the
    segment's only exit and never reaches the jump-target section, orphaning
    the whole alternate branch.

    This surfaces the byte pattern; a later repair pass registers the alternate
    section as a runtime-reachable entry ONLY when it is actually orphaned (so
    it never touches branches already wired by normal segmentation). Returns a
    list of section-body offsets (the alternate arm's first instruction).
    """
    if data[:4] != b"kiwi":
        return []
    bc = max((o + len(t) for o, t in find_strings(data) if len(t) >= 12),
             default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    markers = sorted(set(_section_markers(data, bc)))
    out = []
    for q in fo:
        if data[q] != 0x2b:
            continue
        i = fi[q]
        rel = (data[q + 1] << 8) | data[q + 2]
        tgt = fo[i + rel] if 0 <= i + rel < len(fo) else None
        if tgt is None:
            continue
        # the fall-through arm reaches a scene goto before the jump target
        ft_goto = False
        j = i + 1
        while j < len(fo) and fo[j] < tgt and fo[j] < q + 120:
            if data[fo[j]] == 0x1f and data[fo[j] + 1] == 0x0a:
                ft_goto = True
                break
            j += 1
        if not ft_goto:
            continue
        # the jump target flows into a section marker shortly after
        near = [m for m in markers if 0 <= m - tgt <= 40]
        if near:
            m = min(near)
            c = m + 4
            while c < len(data) and c not in fi:
                c += 1
            out.append(c)
    return out


def _computed_gate_targets(data):
    """Find forward `0x2b` gate targets whose condition is COMPUTED on the stack
    (no plain `1f2d` variable read), which `resolve_var_gates` therefore misses.

    Shape: `... <computed value> ; <const push / 0x5a / 0x5b> ; 0x0a|0x0d|0x0e
    compare ; 0x2b <forward skip>`. In Wrong Side of Town's opening fight, the
    win/lose branch reads an indexed combat slot (a `0x3e` read, not `1f2d`),
    compares it, and the 0x2b jumps to the LOSS narration ("Brendan goes
    down!"); the fall-through is the WIN path. Because the condition isn't a
    bare var read, the gate isn't recognized and the jump-target block orphans.

    These gates are the VM's ordinary conditional-branch mechanism (dozens per
    scene), so the vast majority land WITHIN an already-reachable segment and
    are handled fine by linear threading. This only surfaces the raw pairs; the
    repair pass wires a target ONLY when it lands on a segment that is genuinely
    orphaned (per full reachability) AND distinct from the gate's own segment,
    which is what keeps every other episode byte-identical.

    Returns candidate (jump_offset, target_offset) pairs.
    """
    if data[:4] != b"kiwi":
        return []
    bc = max((o + len(t) for o, t in find_strings(data) if len(t) >= 12),
             default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    out = []
    for q in fo:
        if data[q] != 0x2b:
            continue
        i = fi[q]
        if not (i >= 1 and data[fo[i - 1]] in (0x0a, 0x0d, 0x0e, 0x0f)):
            continue
        if any(data[fo[j]] == 0x1f and data[fo[j] + 1] == 0x2d
               for j in range(max(i - 6, 0), i)):
            continue
        rel = (data[q + 1] << 8) | data[q + 2]
        tgt = fo[i + rel] if 0 <= i + rel < len(fo) else None
        if tgt is not None and tgt > q:      # forward skip only
            out.append((q, tgt))
    return out


def _dispatch_default_arm(data):
    """Find a section-dispatch's DEFAULT (fall-through) arm offset.

    A `1f12` dispatch runs the section right after it (the `33 48 4a`
    post-dispatch marker) when the register matches no case; the dispatch table
    also contains a plain `0x28` jump to that same offset (the default arm).
    Normally this default arm is the scene's main content and is reached by
    ordinary fall-through, so it is NOT special. But when the scene entry has
    been redirected to a value-N section that sits PAST the default arm (e.g.
    Wrong Side of Town, where value 0 jumps forward to the title and the default
    arm is an opening combat minigame that flows into it), the default arm can
    end up orphaned. This surfaces the arm offset; an orphan-gated repair pass
    registers it as a runtime entry only when nothing else reaches it, so every
    scene whose default arm is already reachable is left untouched.
    """
    if data[:4] != b"kiwi":
        return None
    bc = max((o + len(t) for o, t in find_strings(data) if len(t) >= 12),
             default=0)
    fo = _fold_instruction_offsets(data, bc)
    fi = {o: i for i, o in enumerate(fo)}
    disp = next((p for p in fo if data[p] == 0x1f and data[p + 1] == 0x12), None)
    if disp is None:
        return None
    after = fo[fi[disp] + 1] if fi[disp] + 1 < len(fo) else None
    if after is None:
        return None
    for j in range(max(fi[disp] - 12, 0), fi[disp]):
        q = fo[j]
        if data[q] == 0x28:
            rel = (data[q + 1] << 8) | data[q + 2]
            t = fo[fi[q] + rel] if 0 <= fi[q] + rel < len(fo) else None
            if t is not None and abs(t - after) <= 4:
                # body starts at the first fold instruction past the 33/48/4a bytes
                c = t
                while c < len(data) and c not in fi:
                    c += 1
                return c
    return None


def resolve_unconditional_jumps(data):
    """Map each unconditional jump (0x28) to its target instruction's byte
    offset. Flattened switch/elif bytecode lays each arm's block out in
    sequence, ending with a 0x28 that skips the sibling arms to the shared
    merge/END. Linear next-threading mistakes the next sibling arm for the
    continuation; this lets the threader follow the jump to the real merge."""
    if data[:4] != b"kiwi":
        return {}
    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    ins, p = [], bc
    while p < n:
        op = data[p]
        ins.append(p)
        p += 3 if op in (0x1a, 0x41, 0x1b, 0x28, 0x2b, 0x42, 0x5c, 0x1f) else 1
    index = {q: i for i, q in enumerate(ins)}
    out = {}
    for q in ins:
        if data[q] == 0x28:
            v = (data[q + 1] << 8) | data[q + 2]
            ti = index[q] + v
            if 0 <= ti < len(ins):
                out[q] = ins[ti]
    return out


def resolve_choice_forks(data):
    """Offsets of choice ops (1f 01) that are immediately followed by a 0x2b
    option-selector -- i.e. the choice genuinely branches one arm per option.

    After presenting a choice the engine stores the pick and, for a real
    branch, emits a 0x2b that routes to the chosen arm before any line is
    displayed. An *inline* choice (both options share the following content)
    has no such selector: a display op (a spoken/narration/title line, another
    choice, a minigame, ...) comes first. Scanning the few instructions after
    each 1f 01 for a leading 0x2b distinguishes the two cheaply and reliably.
    """
    if data[:4] != b"kiwi":
        return set()

    def real_len(p):
        op = data[p]
        if op == 0x1f:
            return 2
        if op == 0x42:
            return 4
        if op in (0x1a, 0x41, 0x1b, 0x2b, 0x28, 0x5c):
            return 3
        return 1

    strings = find_strings(data)
    bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
    n = len(data)
    forks = set()
    p = bc
    while p < n - 1:
        if data[p] == 0x1f and data[p + 1] == 0x01:      # present-choice op
            q, r, steps = p, p + 2, 0
            while r < n - 1 and steps < 16:
                op = data[r]
                if op == 0x2b:                            # selector -> real branch
                    forks.add(q)
                    break
                if op == 0x1f and data[r + 1] in (
                        0x0d, 0x41, 0x08, 0x01, 0x00, 0x47, 0x0f, 0x58):
                    break                                 # a line first -> inline
                r += real_len(r)
                steps += 1
            p += 2
            continue
        p += real_len(p)
    return forks


def build_episode_json(lines, cast, episode, title, main_characters=None,
                       scene_controls=None, genders=None, minigame_banks=None,
                       name_vars=None, choice_dispatch=None, chapter_heads=None,
                       var_gates=None, section_dispatch=None,
                       minigame_gates=None, buildword_scoring=None,
                       merge_jumps=None, choice_forks=None,
                       score_tier_thresholds=None):
    """Build a branch-aware episode JSON: scenes -> nodes, with choices nesting
    their option branches.

    Each scene becomes a list of nodes. A choice node carries its options and the
    dialogue that follows it (up to the next choice/scene) in script order. Exact
    per-option line boundaries are not recoverable from the bytecode, so they are
    not asserted. Sound-effect cues (1f 4f) appear inline as `sfx` nodes.

    Every node carries its bytecode `offset`. Nodes are listed in bytecode order,
    which is NOT the engine's execution order for branching scenes (e.g. the day
    hub) -- the offsets, plus each scene's raw `control_flow` skeleton (branch
    instructions and probable variable reads, uninterpreted), are how an engine
    reconstructs the real ordering.

    Scenes also carry a `protagonist` hint: who the player controls, taken from
    the in-scene control hand-off / diary signals (null where the game gives no
    explicit cue).
    """
    scene_controls = scene_controls or {}
    minigame_banks = minigame_banks or {}
    scenes, cur_nodes = [], []
    scene_meta = {"scene": 1, "script": None, "protagonist": None}
    pov_state = {"active": None}      # current 1f4a POV, persists across scenes

    def to_node(ln):
        d = line_to_dict(ln)
        if d is not None and getattr(ln, "bc_off", None) is not None:
            d["offset"] = ln.bc_off
        return d

    def flush_scene():
        if cur_nodes or scene_meta["script"] is not None:
            nodes = list(cur_nodes)
            # CHAPTER CARD: head title string + the scene's opening line as
            # subtitle (the pair renders as a title card in-game). Fires only
            # when the head candidate's second string IS the first displayed
            # content line of this scene.
            ch = (chapter_heads or {}).get(scene_meta["script"])
            if ch:
                t0, t1 = ch
                fi = next((k for k, nd in enumerate(nodes)
                           if nd.get("type") in ("narration", "dialogue")), None)
                if fi is not None and (nodes[fi].get("text") or "") == t1 \
                        and not any(nd.get("type") == "title_card" for nd in nodes[:fi]):
                    # one title_card node: t0 = title, t1 (the first content line)
                    # = subtitle. Preserve the subtitle node's offset/id/next.
                    merged = {k: v for k, v in nodes[fi].items()
                              if k in ("offset", "id", "next")}
                    merged = {"type": "title_card", "title": t0,
                              "subtitle": t1, **merged}
                    nodes[fi] = merged
            scene = {**scene_meta, "nodes": nodes}
            banks = minigame_banks.get(scene_meta["script"])
            if banks:
                scene["minigames"] = {
                    "note": ("Minigame data banks read straight from this script's "
                             "head strings (build_word: spell each target from the "
                             "letter pool; pick_word: choose the correct words). "
                             "Every field is a literal .kiw string."),
                    **banks,
                }
            cf = scene_controls.get(scene_meta["script"])
            if cf and cf.get("_sep63_terminals"):
                scene["_sep63_terminals"] = cf["_sep63_terminals"]
            if cf and cf.get("_linear_sep_chains"):
                scene["_linear_sep_chains"] = cf["_linear_sep_chains"]
            if cf and cf.get("branches"):
                scene["control_flow"] = {
                    "note": ("Branch skeleton in bytecode order (not execution "
                             "order); this scene branches on a counter (see "
                             "var_reads). Each branch's `target` is the resolved "
                             "jump destination: the jump operand is a signed "
                             "instruction count, which lands on an instruction "
                             "boundary for every jump in this archive."),
                    **cf,
                }
            sd = (section_dispatch or {}).get(scene_meta["script"])
            if sd:
                scene["section_dispatch"] = dict(sd)
            mjp = (merge_jumps or {}).get(scene_meta["script"])
            if mjp:
                scene["_merge_jumps"] = mjp
            cfk = (choice_forks or {}).get(scene_meta["script"])
            if cfk:
                scene["_choice_forks"] = set(cfk)
            mg = (minigame_gates or {}).get(scene_meta["script"])
            if mg:
                scene["minigame_results"] = {
                    "note": ("Minigame score branches. Both arm offsets are static; "
                             "only pass/fail is runtime (the engine's minigame "
                             "result vs win_threshold). On pass, continue at "
                             "pass_at; on fail (score below win_threshold), the "
                             "engine takes the 0x2b jump to fail_at. 'kind' is "
                             "outcome_fork (the arms reach different endings -- a "
                             "real story branch, e.g. win/lose) or content_fork "
                             "(the arms show different content but reconverge -- the "
                             "score changes what the player sees, e.g. a polished vs "
                             "fumbled performance, not where they end up). "
                             "win_threshold is the raw comparison constant; its "
                             "scale is not encoded. Quiz/round loops (identical arm "
                             "content) and end-of-episode replay/survey flow are "
                             "filtered out."),
                    "gates": mg,
                }
            bws = (buildword_scoring or {}).get(scene_meta["script"])
            if bws:
                scene["minigame_scoring"] = {
                    "note": ("Build-word minigame scoring parameters from each "
                             "0x60 round call: score (observed ~1500-2000), cap "
                             "(~10000 ceiling), midpoint, and param0. The exact "
                             "role of each field is not asserted; these are the raw "
                             "scoring values the engine feeds the round."),
                    "rounds": bws,
                }
            scenes.append(scene)

    def note_pov(node):
        # The 1f4a POV set is authoritative: it both updates the running POV state
        # (which persists across scenes) and pins this scene's protagonist.
        if node.get("type") == "pov_change":
            pov_state["active"] = node.get("character")
            scene_meta["protagonist"] = node.get("character")
            return
        # Otherwise fall back to softer hints, first definite signal wins.
        if scene_meta["protagonist"] is not None:
            return
        lead = lead_of(cast)
        if node.get("type") == "control":
            scene_meta["protagonist"] = " & ".join(node["playable"])
        elif (node.get("type") == "dialogue" and node.get("text", "").startswith("'")
              and node.get("speaker") == lead):
            scene_meta["protagonist"] = node["speaker"]


    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.speaker == "__SCENE__":
            flush_scene()
            cur_nodes = []
            num = len(scenes) + 1
            label = ln.text.split("(")[-1].rstrip(")") if "(" in ln.text else None
            scene_meta = {"scene": num, "script": label,
                          "protagonist": pov_state["active"]}
            i += 1
            continue
        if ln.speaker == "__CHOICE__":
            options = ln.text.split("|")
            j = i + 1
            while j < n and lines[j].speaker not in ("__CHOICE__", "__SCENE__"):
                j += 1
            disp = (choice_dispatch or {}).get(scene_meta["script"], {}) \
                .get(getattr(ln, "bc_off", None))
            gates = (var_gates or {}).get(scene_meta["script"], [])

            region = lines[i + 1:j]
            # A dispatch option whose body lies entirely PAST the next sibling
            # choice is a non-contiguous branch that jumps to a later section (e.g.
            # the Sophie "too high maintenance" breakup, which the bytecode routes
            # to a self-contained section after the following choice). We don't pull
            # those lines in (the section has its own nested choices and exit goto);
            # instead we record the jump offset so the option can be linked to that
            # section's segment after segmentation.
            next_choice_off = None
            if j < n and lines[j].speaker == "__CHOICE__":
                next_choice_off = getattr(lines[j], "bc_off", None)
            far_targets = {}
            if disp and next_choice_off is not None:
                explicit = {o2["index"] for o2 in disp.get("options", [])
                            if o2["index"] != "else"}
                remaining = [k for k in range(len(options)) if k not in explicit]
                for o2 in disp.get("options", []):
                    lo2, _hi2 = o2["region"]
                    if lo2 >= next_choice_off:
                        idx2 = o2["index"]
                        if idx2 == "else":
                            if len(remaining) != 1:
                                continue
                            idx2 = remaining[0]
                        far_targets[str(idx2)] = lo2
            branch_dialogue = []
            for r in region:
                d = to_node(r)
                if d is not None:
                    branch_dialogue.append(d)
                    note_pov(d)
            node = {
                "type": "choice",
                "prompt": ln.emotion,
                "options": [{"index": idx, "label": o} for idx, o in enumerate(options)],
            }
            if getattr(ln, "bc_off", None) is not None:
                node["offset"] = ln.bc_off
            if far_targets:
                # option index -> bytecode offset of a later section it jumps to
                node["_far_targets"] = far_targets



            def nest_gates(sel_lines, lo3, hi3):
                """Nest var-gated blocks: a gate node wraps the lines of its
                `then` region; siblings after it are the fall-through (else)
                content. Recursive for gates inside a then-block."""
                inner = [g for g in gates if lo3 <= g["at"] < hi3]
                if not inner:
                    return sel_lines
                inner.sort(key=lambda g: g["at"])
                out3, used3 = [], set()
                cursor = lo3
                for g in inner:
                    if g["at"] < cursor:        # nested inside a handled gate
                        continue
                    glo, ghi = g["then"]
                    pre = [x for x in sel_lines
                           if cursor <= int(x["offset"]) < g["at"]
                           and id(x) not in used3]
                    body = [x for x in sel_lines
                            if glo <= int(x["offset"]) < ghi
                            and id(x) not in used3]
                    for x in pre + body:
                        used3.add(id(x))
                    out3.extend(pre)
                    _gate_nd = {"type": "gate", "var": g["var"],
                                 "equals": g["equals"],
                                 "then": nest_gates(body, glo, ghi),
                                 "offset": g["at"]}
                    if g.get("op"):
                        _gate_nd["op"] = g["op"]
                    out3.append(_gate_nd)
                    cursor = ghi
                out3.extend(x for x in sel_lines
                            if id(x) not in used3 and int(x["offset"]) >= cursor)
                return out3
            if disp:
                # BYTECODE-RESOLVED branches: the choice's dispatch (test blocks
                # + resolved jumps) gives each option's byte region and the
                # common merge point. Lines at/after the merge are shared
                # continuation, kept in an "after the choice" segment.
                merge = disp.get("merge")
                explicit = {o2["index"] for o2 in disp["options"]
                            if o2["index"] != "else"}
                remaining = [k for k in range(len(options)) if k not in explicit]
                branches, used = [], set()
                for o2 in disp["options"]:
                    lo2, hi2 = o2["region"]
                    sel = [x for x in branch_dialogue
                           if x.get("offset") is not None
                           and lo2 <= int(x["offset"]) < hi2]
                    sel = [x for x in sel if x.get("offset") is not None]
                    used.update(id(x) for x in sel)
                    idx2 = o2["index"]
                    if idx2 == "else":
                        idx2 = remaining[0] if len(remaining) == 1 else remaining
                    branches.append({"index": idx2,
                                     "lines": nest_gates(sel, lo2, hi2)})
                after = [x for x in branch_dialogue if id(x) not in used]
                node["branch_dialogue"] = branches
                if merge is not None:
                    node["merge_offset"] = merge
                if after:
                    node["after_choice"] = after
                node["note"] = ("Per-option branches resolved from the bytecode "
                                "dispatch (option tests + jump targets); "
                                "after_choice is the shared continuation past "
                                "the merge point.")
                cur_nodes.append(node)
                i = j
                continue
            outcomes = split_outcomes(branch_dialogue)
            if outcomes:
                node["outcomes"] = outcomes
                node["note"] = ("Following dialogue is split into distinct outcomes at "
                                "the game's replay/ending banners. A segment whose "
                                "ends_with_retry_prompt is true is an alternate (non-ideal) "
                                "ending the player is told to replay.")
            else:
                node["branch_dialogue"] = branch_dialogue
                node["note"] = ("Dialogue for all options plays out in script order; "
                                "exact per-option line boundaries are not recoverable "
                                "from the bytecode.")
            cur_nodes.append(node)
            i = j
            continue
        d = to_node(ln)
        if d is not None:
            cur_nodes.append(d)
            note_pov(d)
        i += 1
    flush_scene()
    # Single-protagonist episodes: every scene is that character's POV.
    if main_characters and len(main_characters) == 1:
        for sc in scenes:
            if sc["protagonist"] is None:
                sc["protagonist"] = main_characters[0]

    # Resolve goto_scene targets (chunk ids) to the destination scene's index, so a
    # jump can be matched to a scene block by either its "scene" number or "script".
    label_to_index = {sc["script"]: sc["scene"] for sc in scenes}

    def link_gotos(nodes):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if n.get("type") == "goto_scene":
                n["to_scene"] = label_to_index.get(n["script"])
            elif n.get("type") == "gate":
                link_gotos(n.get("then", []))
            elif n.get("type") == "choice":
                for o in n.get("outcomes", []):
                    link_gotos(o["lines"])
                for x in n.get("branch_dialogue", []) or []:
                    if isinstance(x, dict) and "lines" in x:
                        link_gotos(x["lines"])
                    elif isinstance(x, dict):
                        link_gotos([x])
                link_gotos(n.get("after_choice", []) or [])
    for sc in scenes:
        link_gotos(sc["nodes"])

    # main_characters: prefer the authoritative set of POV characters named by 1f4a
    # (handles co-presence and catches every playable, e.g. Ben in Swim). Fall back
    # to the heuristic tag_pov list only when no 1f4a POV sets appear.
    pov_chars = []
    def _collect_pov(nodes):
        for nd in nodes:
            if not isinstance(nd, dict):
                continue
            if nd.get("type") == "pov_change":
                c = nd.get("character")
                if c and c not in pov_chars:
                    pov_chars.append(c)
            for key in ("then", "after_choice", "lines"):
                if isinstance(nd.get(key), list):
                    _collect_pov(nd[key])
            if nd.get("type") == "choice":
                for o in nd.get("outcomes", []) or []:
                    _collect_pov(o.get("lines", []))
                _collect_pov(nd.get("branch_dialogue", []) or [])
    for sc in scenes:
        _collect_pov(sc["nodes"])
    mains_out = pov_chars or main_characters or []

    return {"title": title, "episode": episode,
            "main_characters": mains_out, "cast": cast,
            "genders": genders or {}, "name_vars": name_vars or {},
            "scenes": scenes}


def _split_choice(nodes, spec):
    """Split a choice's flat branch_dialogue into per-option branches, by offset
    range, and attach declared effects. `spec` = {at, options:[{index, lines:[lo,hi],
    effects?}]}. Lines outside every option's range are dropped (they belong after
    the merge point)."""
    at = str(spec["at"])
    for nd in nodes:
        if nd.get("type") == "choice" and str(nd.get("offset")) == at:
            flat = nd.get("branch_dialogue") or []
            if any(isinstance(x, dict) and "lines" in x for x in flat):
                # already split from the bytecode dispatch; the overlay only
                # contributes the per-option effects
                for opt in spec["options"]:
                    if "effects" in opt:
                        for o in nd.get("options", []):
                            if str(o["index"]) == str(opt["index"]):
                                o["effects"] = opt["effects"]
                return True
            branches = []
            for opt in spec["options"]:
                lo, hi = opt["lines"]
                lines = [x for x in flat
                         if x.get("offset") is not None and lo <= int(x["offset"]) <= hi]
                branches.append({"index": str(opt["index"]), "lines": lines})
                if "effects" in opt:
                    for o in nd.get("options", []):
                        if str(o["index"]) == str(opt["index"]):
                            o["effects"] = opt["effects"]
            nd["branch_dialogue"] = branches
            nd.pop("note", None)          # the boundaries are now known, from the overlay
            return True
    return False


def apply_overlay(doc, overlay):
    """Merge a per-episode overlay (observed gameplay logic) onto a decoded episode.

    The overlay is DATA, not code: the parser stays generic and this applies any
    overlay to any episode purely by matching bytecode offsets. Supported directives
    (all keyed by the `offset` fields the decoder already emits):

      carve_scene  -- lift a [start,end] offset region out of a source scene into a
                      new logical scene (name/protagonist/gate from the overlay).
      choices      -- split a carved choice's flat branch_dialogue into per-option
                      branches by offset range, and attach effects (e.g. money -10).
      append       -- extra nodes to add to the new scene (e.g. a goto_scene).

    Content (the lines themselves) always comes from the decode; the overlay only
    supplies attribution/effects/structure that the bytecode can't yield reliably.
    """
    scenes = doc["scenes"]
    pending = []
    for directive in overlay.get("scenes", []):
        carve = directive.get("carve_scene")
        if not carve:
            continue
        src = next((s for s in scenes if s["script"] == carve["from_script"]), None)
        if src is None:
            continue
        lo, hi = carve["region"]
        carved, kept = [], []
        for nd in src["nodes"]:
            off = nd.get("offset")
            (carved if off is not None and lo <= int(off) <= hi else kept).append(nd)
        src["nodes"] = kept
        for ch in directive.get("choices", []):
            _split_choice(carved, ch)
        carved.extend(directive.get("append", []))
        new_scene = {"scene": carve.get("scene"), "script": carve["from_script"],
                     "name": carve.get("name"), "protagonist": carve.get("protagonist"),
                     "source": "observed", "region": list(carve["region"])}
        if "gate" in carve:
            new_scene["gate"] = carve["gate"]
        new_scene["nodes"] = carved
        pending.append((src, new_scene))
    for src, new_scene in pending:
        scenes.insert(scenes.index(src) + 1, new_scene)
    # Section-dispatch bindings: the value->section offset map. The decoder now
    # byte-derives the first two bindings (natural landings); the overlay supplies
    # the remaining ones (observed, since the host VM resolves them at runtime).
    # Merge per-value so the byte-derived bindings are preserved and the overlay
    # only adds/overrides the values it names.
    for binding in overlay.get("section_bindings", []):
        sc = next((s for s in scenes if s["script"] == binding.get("script")), None)
        if sc is not None and "section_dispatch" in sc:
            existing = {str(b["value"]): b
                        for b in sc["section_dispatch"].get("bindings", [])}
            for b in binding.get("bindings", []):
                existing[str(b["value"])] = b
            sc["section_dispatch"]["bindings"] = [existing[k] for k in
                                                  sorted(existing, key=lambda x: int(x))]
    # Gate fall-through directives: an `if var==N` guard whose then-branch (extra
    # content shown only when the guard holds) falls through into the shared
    # continuation. The bytecode allows this generically, but threading every such
    # gate proved unsafe (it can mis-merge unrelated blocks), so it is opt-in:
    # the overlay lists the then-branch offsets that should thread to the else.
    for ft in overlay.get("gate_fallthroughs", []):
        sc = next((s for s in scenes if s["script"] == ft.get("script")), None)
        if sc is not None:
            sc.setdefault("_gate_fallthroughs", []).extend(
                int(o) for o in ft.get("then_offsets", []))
    # Choice-exit directives: a choice that ENDS its section and transitions to
    # another scene, rather than falling through into the *next* (different)
    # section that merely follows it in the bytecode. Each option keeps its own
    # variable write; after the choice the section sets the dispatch register and
    # jumps to the named scene. Opt-in (observed) because sibling sections share a
    # single exit goto in the bytecode via relative jumps the decoder can't resolve.
    for ce in overlay.get("choice_exits", []):
        sc = next((s for s in scenes if s["script"] == ce.get("script")), None)
        if sc is not None:
            sc.setdefault("_choice_exits", {})[str(ce["choice_offset"])] = {
                "to_scene": ce.get("to_scene"),
                "set_register": ce.get("set_register")}
    # Router directives: counter-gated dispatcher routing the bytecode can't
    # yield. Stash raw (offset-based) here; normalize_graph resolves to node ids.
    for r in overlay.get("routers", []):
        sc = next((s for s in scenes if s["script"] == r.get("script")), None)
        if sc is not None:
            sc["_router_raw"] = {"on": r.get("on"), "rules": r.get("rules", []),
                                 "default_at": r.get("default")}
    if overlay.get("option_limits"):
        doc["_option_limits_raw"] = overlay["option_limits"]
    if overlay.get("sequences"):
        doc["_sequences_raw"] = overlay["sequences"]
    if overlay.get("emphasis"):
        doc["_emphasis"] = overlay["emphasis"]
    _apply_split_scenes(doc, overlay)
    return doc


def _flatten_scene_nodes(nodes):
    """Return a flat, offset-sorted list of leaf nodes (deep-ish copies) from a
    scene, un-nesting choice branch_dialogue / outcomes. Choices are kept as
    choice nodes but without their nested dialogue (which becomes flat siblings)."""
    flat = []

    def walk(ns):
        for nd in ns:
            base = {k: v for k, v in nd.items() if k not in ("branch_dialogue", "outcomes")}
            if base.get("offset") is not None:
                flat.append(base)
            bd = nd.get("branch_dialogue")
            if isinstance(bd, list):
                for x in bd:
                    if isinstance(x, dict) and "lines" in x:
                        walk(x["lines"])
                    elif isinstance(x, dict):
                        walk([x])
            for oc in nd.get("outcomes", []) or []:
                walk(oc.get("lines", []))
            walk(nd.get("after_choice", []) or [])
            if nd.get("type") == "gate":
                walk(nd.get("then", []) or [])

    walk(nodes)
    flat.sort(key=lambda n: int(n["offset"]))
    return flat


def _thread_flat(lines, tail):
    """Flatten a list of branch/outcome/then lines into a flat, next-threaded list
    ending at `tail`. Recurses into nested gates (then-branch) and choices. Each
    node's `next` points to the following sibling; the last points to `tail`."""
    out = []
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        nxt = tail
        for k in range(i + 1, len(lines)):
            if isinstance(lines[k], dict):
                nxt = lines[k].get("id")
                break
        if ln.get("type") == "gate" and isinstance(ln.get("then"), list):
            then = ln.pop("then")
            ln["then_at"] = then[0].get("id") if (then and isinstance(then[0], dict)) else nxt
            ln["next"] = nxt
            out.append(ln)
            out.extend(_thread_flat(then, nxt))
        elif ln.get("type") == "choice":
            out.extend(_flatten_choice_node(ln, nxt))
        else:
            ln["next"] = nxt
            out.append(ln)
    return out


def _flatten_choice_node(ch, tail):
    """Return [choice] + its hoisted, threaded branch/outcome nodes. The choice keeps
    options (each gaining `goto` -> its branch entry id) and `next`/`continues_at` ->
    the post-merge node; nested dialogue becomes flat siblings."""
    merge = ch.get("continues_at") or ch.get("next") or tail
    hoisted = []
    bd, oc = ch.get("branch_dialogue"), ch.get("outcomes")
    if isinstance(bd, list) and bd and isinstance(bd[0], dict) and "lines" in bd[0]:
        by_index = {str(b.get("index")): b for b in bd}
        for opt in ch.get("options", []):
            b = by_index.get(str(opt.get("index")))
            lines = b.get("lines", []) if b else []
            if lines:
                fl = _thread_flat(lines, merge)
                opt["goto"] = fl[0].get("id") if fl else merge
                hoisted.extend(fl)
            else:
                opt["goto"] = merge
        # branches with no matching option (defensive): hoist + thread to merge
        used = {str(b.get("index")) for b in bd}
        opt_idx = {str(o.get("index")) for o in ch.get("options", [])}
        for b in bd:
            if str(b.get("index")) not in opt_idx:
                hoisted.extend(_thread_flat(b.get("lines", []), merge))
    elif isinstance(bd, list):
        lines = [x for x in bd if isinstance(x, dict)]
        fl = _thread_flat(lines, merge)
        entry = fl[0].get("id") if fl else merge
        for opt in ch.get("options", []):
            opt["goto"] = entry
        ch["next"] = entry
        hoisted.extend(fl)
    elif isinstance(oc, list):
        gotos = []
        for k, seg in enumerate(oc):
            fl = _thread_flat(seg.get("lines", []), merge)
            if fl:
                gotos.append({"index": k, "goto": fl[0].get("id"),
                              "ends_with_retry_prompt": seg.get("ends_with_retry_prompt")})
                hoisted.extend(fl)
        ch["outcome_gotos"] = gotos
    else:
        for opt in ch.get("options", []):
            opt.setdefault("goto", merge)
    ch.pop("branch_dialogue", None)
    ch.pop("outcomes", None)
    if not ch.get("next"):
        ch["next"] = merge
    return [ch] + hoisted


def _apply_split_scenes(doc, overlay):
    """Carve a scene into per-actor sub-scenes by offset region (e.g. the Help
    Ads people), wiring each internal 2-way choice to its paid/unpaid outcomes
    and routing the hub to the right sub-scene by a counter. Content always comes
    from the decode; the overlay only supplies the regions and counter mapping."""
    scenes = doc["scenes"]
    for split in overlay.get("split_scenes", []):
        src = next((s for s in scenes if s["script"] == split["from_script"]), None)
        if src is None:
            continue
        flat = _flatten_scene_nodes(src["nodes"])
        people = split["people"]
        hub_hi = min(p["region"][0] for p in people) - 1
        src["nodes"] = [n for n in flat if int(n["offset"]) <= hub_hi]

        new_scenes = []
        for p in people:
            lo, hi = p["region"]
            pnodes = [dict(n) for n in flat if lo <= int(n["offset"]) <= hi]
            _wire_person_choice(pnodes)
            new_scenes.append({
                "scene": "%s_%s" % (src["scene"], p["name"]),
                "script": split["from_script"],
                "name": "help_ads_%s" % p["name"],
                "source": "observed", "region": [lo, hi],
                "counter_value": p["use"], "nodes": pnodes,
            })
        idx = scenes.index(src)
        for k, ns in enumerate(new_scenes, 1):
            scenes.insert(idx + k, ns)
        # Hub router: counter value -> the person sub-scene (resolved by name).
        src["_ad_router_raw"] = {
            "counter": split["counter"],
            "by_name": [{"eq": p["use"], "name": "help_ads_%s" % p["name"]} for p in people]}


def _wire_person_choice(pnodes):
    """If a person's block has a single 2-way choice, split the content after it
    into outcomes at each `goto_scene` boundary and point each option at its
    outcome's first node (by offset). The outcome containing a money effect is
    the paid one. Positional: option i -> outcome i."""
    ci = next((i for i, n in enumerate(pnodes) if n.get("type") == "choice"), None)
    if ci is None:
        return
    choice = pnodes[ci]
    # Outcome runs: each ends at a goto_scene (inclusive).
    runs, cur = [], []
    for n in pnodes[ci + 1:]:
        cur.append(n)
        if n.get("type") == "goto_scene":
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    opts = choice.get("options", [])
    if not runs or len(opts) != len(runs):
        return                                # leave as-is if it doesn't line up
    for opt, run in zip(opts, runs):
        opt["leads_to"] = {"node_at": int(run[0]["offset"])}
        opt["paid"] = any(re.search(r'earned\s+\d+[^.!?]*?(?:dollars|bucks)',
                                    (n.get("text") or ""), re.I) for n in run)
    choice["_intra_routed"] = True


# --------------------------------------------------------------------------- #
# Graph normalization: turn the decoded/overlaid document into a walkable graph.
# --------------------------------------------------------------------------- #
def _gotos_in(region):
    """Yield goto_scene nodes found directly in a choice's following region.

    `region` may be a flat branch_dialogue list, a list of per-option branches
    ({index, lines}), or a list of outcome segments ({lines}). Only top-level
    goto_scene nodes are collected (in document order)."""
    out = []
    for item in region or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "goto_scene":
            out.append(item)
        elif "lines" in item:                       # per-option branch / outcome
            out.extend(x for x in item["lines"]
                       if isinstance(x, dict) and x.get("type") == "goto_scene")
    return out


def normalize_graph(doc):
    """Add a reliable navigation layer on top of the decoded document.

    What this asserts (all derivable without guessing the VM's jump encoding):
      * every node gets a stable `id` ("<scene>.<n>"); each scene gets `entry`
        (the id of its first node).
      * `goto_scene` nodes get `to_node` = the entry id of the destination scene.
      * `choice` nodes are classified into a `branch_type`:
          - "menu"     : the choice is followed by one goto_scene per option, in
                         order -> each option gets `leads_to` {scene, entry}.
          - "branched" : the overlay split it into per-option branches -> each
                         option that contains a goto_scene gets `leads_to`.
          - "linear"   : no per-option destination (e.g. a scored quiz question);
                         flow falls through to the choice's `next`.
      * a default fall-through `next` (the following node in bytecode order) is
        added to ordinary nodes. A node that hands off control (goto_scene, or a
        "menu" choice) gets no `next`.
      * each scene is flagged `counter_gated` when its control_flow shows
        conditional branches/variable reads; for those scenes `next` is only the
        literal fall-through and counter-driven diversions (day thresholds,
        end-of-run screens) are NOT encoded here -- see `control_flow`.

    What this does NOT do: resolve raw within-scene conditional jump targets.
    Those remain in `control_flow` as unverified provenance.
    """
    scenes = doc.get("scenes", [])
    by_num = {str(s.get("scene")): s for s in scenes}

    # Pass 1: ids + scene entry.
    for s in scenes:
        sc = str(s.get("scene"))
        nodes = s.get("nodes", [])
        for i, nd in enumerate(nodes):
            nd["id"] = f"{sc}.{i}"
        s["entry"] = nodes[0]["id"] if nodes else None
        cf = s.get("control_flow")
        s["counter_gated"] = bool(
            cf and (any(b.get("compared") for b in cf.get("branches", []))
                    or cf.get("var_reads")))

    # Pass 1b: stable sub-ids for nested lines, so EVERY node is addressable:
    #   <choice_id>.d<j>          flat branch_dialogue line
    #   <choice_id>.opt<k>.<j>    line inside a per-option branch
    #   <choice_id>.out<k>.<j>    line inside an outcome segment
    def _sub_ids(nd):
        cid = nd.get("id")
        if not cid or nd.get("type") != "choice":
            return
        bd = nd.get("branch_dialogue")
        if isinstance(bd, list):
            j = 0
            for x in bd:
                if not isinstance(x, dict):
                    continue
                if "lines" in x:                       # per-option branch
                    k = x.get("index", j)
                    if isinstance(k, list):
                        k = "-".join(str(v) for v in k)
                    for jj, ln in enumerate(x["lines"]):
                        if isinstance(ln, dict) and "id" not in ln:
                            ln["id"] = f"{cid}.opt{k}.{jj}"
                        _sub_ids(ln)
                else:                                  # flat line
                    if "id" not in x:
                        x["id"] = f"{cid}.d{j}"
                    _sub_ids(x)
                j += 1
        for k, oc in enumerate(nd.get("outcomes", []) or []):
            for jj, ln in enumerate(oc.get("lines", [])):
                if isinstance(ln, dict) and "id" not in ln:
                    ln["id"] = f"{cid}.out{k}.{jj}"
                _sub_ids(ln)
        for jj, ln in enumerate(nd.get("after_choice", []) or []):
            if isinstance(ln, dict) and "id" not in ln:
                ln["id"] = f"{cid}.after.{jj}"
            _sub_ids(ln)
        if nd.get("type") == "gate":
            gid = nd.get("id", "g")
            for jj, ln in enumerate(nd.get("then", []) or []):
                if isinstance(ln, dict) and "id" not in ln:
                    ln["id"] = f"{gid}.t{jj}"
                _sub_ids(ln)

    for s in scenes:
        for nd in s.get("nodes", []):
            _sub_ids(nd)

    def entry_of(to_scene):
        tgt = by_num.get(str(to_scene))
        return tgt.get("entry") if tgt else None

    def off_to_id_in(scene, off):
        for nd in scene.get("nodes", []):
            if nd.get("offset") is not None and int(nd["offset"]) == int(off):
                return nd["id"]
        return None

    def _node_at_or_after(scene, off):
        best = None
        for nd in scene.get("nodes", []):
            o = nd.get("offset")
            if o is not None and int(o) >= int(off) and "id" in nd:
                if best is None or int(o) < best[0]:
                    best = (int(o), nd["id"])
        return best[1] if best else None

    # Pass 2: resolve gotos + classify choices + fall-through next.
    for s in scenes:
        nodes = s.get("nodes", [])
        choice_forks = s.pop("_choice_forks", None) or set()
        for i, nd in enumerate(nodes):
            typ = nd.get("type")
            hands_off = False

            if typ == "goto_scene":
                nd["to_node"] = entry_of(nd.get("to_scene"))
                hands_off = True

            elif typ == "choice":
                region = (nd.get("branch_dialogue")
                          if isinstance(nd.get("branch_dialogue"), list) else None)
                outcomes = nd.get("outcomes")
                opts = nd.get("options", [])
                # Per-option branches from the overlay: {index, lines}.
                per_option = bool(region and region and isinstance(region[0], dict)
                                  and "index" in region[0] and "lines" in region[0])

                if nd.get("_intra_routed"):
                    # Carved within-scene choice: options already point at outcome
                    # nodes (by offset); resolve them and route, don't fall through.
                    nd["branch_type"] = "outcomes"
                    nd.pop("_intra_routed", None)
                    for o in opts:
                        lt = o.get("leads_to")
                        if lt and "node_at" in lt:
                            o["leads_to"] = {"node": off_to_id_in(s, lt["node_at"])}
                    hands_off = True
                elif per_option:
                    nd["branch_type"] = "branched"
                    for b in region:
                        g = next((x for x in b.get("lines", [])
                                  if isinstance(x, dict) and x.get("type") == "goto_scene"),
                                 None)
                        if g is not None:
                            for o in opts:
                                if str(o.get("index")) == str(b.get("index")):
                                    o["leads_to"] = {"scene": str(g.get("to_scene")),
                                                     "entry": entry_of(g.get("to_scene"))}
                else:
                    gotos = _gotos_in(region) + _gotos_in(outcomes)
                    distinct = len({str(g.get("to_scene")) for g in gotos})
                    runs = []
                    if isinstance(region, list) and region and not per_option:
                        cur = []
                        for x in region:
                            cur.append(x)
                            if isinstance(x, dict) and x.get("type") == "goto_scene":
                                runs.append(cur)
                                cur = []
                        if cur:
                            runs.append(cur)
                    if (gotos and len(gotos) == len(opts) and len(opts) > 1
                            and distinct == len(opts)):
                        # A real menu: every option exits to a DISTINCT scene (e.g.
                        # a day-hub task list).
                        nd["branch_type"] = "menu"
                        for o, g in zip(opts, gotos):
                            o["leads_to"] = {"scene": str(g.get("to_scene")),
                                             "entry": entry_of(g.get("to_scene"))}
                        hands_off = True            # every option leaves the scene
                    elif (nd.get("offset") is not None
                          and int(nd["offset"]) in (choice_forks or set())
                          and len(opts) > 1 and len(runs) >= len(opts)
                          and all(any(isinstance(x, dict)
                                      and x.get("type") == "goto_scene" for x in r)
                                  for r in runs[:len(opts)])):
                        # Converging per-option branch. Only when the choice has a
                        # 0x2b option-selector right after it (choice_forks) does the
                        # post-choice content genuinely split one arm per option,
                        # each ending in its own goto_scene (e.g. two fight strategies
                        # that then both lead on to the next chapter). Give each
                        # option its own arm; content past the arms (a separately
                        # dispatched section the region swept in) is shared after-content.
                        arms = runs[:len(opts)]
                        after = [x for r in runs[len(opts):] for x in r]
                        nd["branch_type"] = "branched"
                        nd["branch_dialogue"] = [
                            {"index": str(o.get("index")), "lines": arm}
                            for o, arm in zip(opts, arms)]
                        if after:
                            nd["after_choice"] = (nd.get("after_choice") or []) + after
                        hands_off = True
                    else:
                        nd["branch_type"] = "linear"

            # Default fall-through (bytecode order) for nodes that don't hand off.
            if not hands_off and i + 1 < len(nodes):
                nxt = nodes[i + 1]["id"]
                # If an unconditional jump (0x28) sits between this node and the
                # next, and it skips PAST that next node to a further merge/END,
                # this node ends a switch/elif arm: follow the jump, not the
                # linearly-adjacent sibling arm.
                mj = s.get("_merge_jumps") or {}
                o = nd.get("offset")
                o2 = nodes[i + 1].get("offset")
                if mj and o is not None and o2 is not None:
                    o, o2 = int(o), int(o2)
                    js = [j for j in mj if o < int(j) < o2]
                    if js:
                        tgt = mj[min(js)]
                        if int(tgt) > o2:          # jumps past the sibling arm
                            tid = _node_at_or_after(s, tgt)
                            if tid:
                                nxt = tid
                nd["next"] = nxt

    # Tighten the control_flow note so it is read as provenance, not navigation.
    for s in scenes:
        # NB: keep s["_merge_jumps"] -- the segment builder uses it to repair
        # dead-end blocks whose final instruction is an unconditional jump.
        cf = s.get("control_flow")
        if cf:
            cf["note"] = ("Raw, unverified branch skeleton kept for provenance only. "
                          "Navigation comes from `entry`, each node's `next`, "
                          "`goto_scene.to_node`, and each choice option's `leads_to`. "
                          "These raw jump targets are NOT resolved (the relative-target "
                          "encoding is unverified); do not navigate from them.")

    # Resolve any overlay router (offset-based) to node ids now that ids exist.
    for s in scenes:
        raw = s.pop("_router_raw", None)
        if not raw:
            continue
        off_to_id = {int(nd["offset"]): nd["id"]
                     for nd in s.get("nodes", []) if "offset" in nd}
        rules = []
        for rule in raw.get("rules", []):
            nid = off_to_id.get(int(rule.get("at"))) if rule.get("at") is not None else None
            if nid is None:
                continue
            r = {k: v for k, v in rule.items() if k != "at"}
            r["node"] = nid
            rules.append(r)
        router = {"on": raw.get("on"), "rules": rules, "source": "observed"}
        if raw.get("default_at") is not None:
            router["default"] = off_to_id.get(int(raw["default_at"]))
        s["router"] = router

    # Resolve actor-routers (split_scene hubs): counter value -> sub-scene entry,
    # matched by the sub-scene's name.
    name_to_entry = {s.get("name"): s.get("entry") for s in scenes if s.get("name")}
    for s in scenes:
        raw = s.pop("_ad_router_raw", None)
        if not raw:
            continue
        rules = [{"eq": r["eq"], "node": name_to_entry.get(r["name"])}
                 for r in raw.get("by_name", []) if name_to_entry.get(r["name"])]
        s["router"] = {"on": raw.get("counter"), "rules": rules, "source": "observed"}

    # Flag the artifact the user sees: in a counter_gated scene the nodes are the
    # flattened arms of a counter switch (see control_flow), NOT a sequence. Make
    # that legible so a goto like "scene 5 -> scene 1" is not mistaken for the
    # real, unconditional flow.
    #
    # 1) scene-level dispatch note + router-driven entry.
    # 2) goto_scene nodes that *close a cycle* (their target scene can reach this
    #    one) are loop-back edges that the counter actually guards -> mark them.
    num_to_scene = {str(s.get("scene")): s for s in scenes}

    def scene_succs(sc):
        out = []
        for nd in sc.get("nodes", []):
            if nd.get("type") == "goto_scene" and nd.get("to_scene") is not None:
                out.append(str(nd["to_scene"]))
            if nd.get("type") == "choice":
                for o in nd.get("options", []):
                    lt = o.get("leads_to")
                    if lt and lt.get("scene") is not None:
                        out.append(str(lt["scene"]))
        return out

    # Classify scene->scene edges with a DFS from the episode start: an edge to a
    # scene currently on the DFS stack (an ancestor) is a back-edge, i.e. a real
    # loop closer (dispatcher -> day intro), as opposed to the forward edge that
    # enters the dispatcher in the first place.
    back_edges = set()
    on_stack, done = set(), set()

    def dfs(num):
        on_stack.add(num)
        sc = num_to_scene.get(num)
        if sc:
            for nx in scene_succs(sc):
                if nx in on_stack:
                    back_edges.add((num, nx))
                elif nx not in done:
                    dfs(nx)
        on_stack.discard(num)
        done.add(num)

    if scenes:
        dfs(str(scenes[0].get("scene")))

    for s in scenes:
        if not s.get("counter_gated"):
            continue
        this_num = str(s.get("scene"))
        if s.get("router"):
            s["entry_via"] = "router"
        s["dispatch_note"] = (
            "This scene's nodes are the flattened arms of a counter-gated switch "
            "(see control_flow, which reads the day/state counter and branches on "
            "it). They are NOT a linear sequence: a node's `next` is only bytecode "
            "fall-through, the literal `entry` is just the first arm, and any goto "
            "marked `loop_back` is one conditional arm, not the real flow. Use "
            "`router` (if present) for the real entry; the host advances the counter "
            "and re-enters via the router each cycle.")
        for nd in s.get("nodes", []):
            if nd.get("type") == "goto_scene" and nd.get("to_scene") is not None:
                if (this_num, str(nd["to_scene"])) in back_edges:
                    nd["loop_back"] = True
                    nd["note"] = ("Conditional loop-back arm: guarded by the scene's "
                                  "counter (see control_flow / router); not an "
                                  "unconditional jump.")

    # Attach observed per-option usage limits (e.g. Help Ads capped at 6 uses).
    # Resolve the "exhausted" line offset (in its own scene) to a node id, then
    # tag every menu option that leads to the limited task.
    for lim in doc.pop("_option_limits_raw", []):
        ex_scene = num_to_scene.get(str(lim.get("exhausted_scene")))
        ex_id = None
        if ex_scene and lim.get("exhausted_at") is not None:
            off_to_id = {int(nd["offset"]): nd["id"]
                         for nd in ex_scene.get("nodes", []) if "offset" in nd}
            ex_id = off_to_id.get(int(lim["exhausted_at"]))
        for s in scenes:
            for nd in s.get("nodes", []):
                if nd.get("type") != "choice":
                    continue
                for o in nd.get("options", []):
                    lt = o.get("leads_to")
                    if lt and str(lt.get("scene")) == str(lim.get("to_scene")):
                        o["limit"] = {"counter": lim.get("counter"),
                                      "max": lim.get("max"),
                                      "exhausted_line": ex_id,
                                      "source": "observed"}

    # Observed sequences (e.g. the Help Ads people, one per counter value). The
    # targets are nested inside the scene's choice blocks, so they are recorded
    # as offset references with a verifying text snippet rather than walkable node
    # ids (carving them into addressable sub-scenes is a separate step).
    def _deep_text_at(scene, want_off):
        found = [None]
        def walk(nodes):
            for nd in nodes:
                if nd.get("offset") is not None and int(nd["offset"]) == want_off:
                    found[0] = nd.get("text") or ""
                bd = nd.get("branch_dialogue")
                if isinstance(bd, list):
                    for x in bd:
                        if isinstance(x, dict) and "lines" in x:
                            walk(x["lines"])
                        elif isinstance(x, dict):
                            walk([x])
                for oc in nd.get("outcomes", []) or []:
                    walk(oc.get("lines", []))
                walk(nd.get("after_choice", []) or [])
                if nd.get("type") == "gate":
                    walk(nd.get("then", []) or [])
        walk(scene.get("nodes", []))
        return found[0]

    for seq in doc.pop("_sequences_raw", []):
        sc = next((s for s in scenes if s.get("script") == seq.get("script")), None)
        if sc is None:
            continue
        order = []
        for item in seq.get("order", []):
            snippet = _deep_text_at(sc, int(item["at"])) if item.get("at") is not None else None
            order.append({"use": item.get("use"), "person": item.get("person"),
                          "offset": item.get("at"),
                          "line": (snippet[:60] if snippet else None)})
        sc["ad_sequence"] = {
            "counter": seq.get("counter"), "order": order,
            "exhausted_offset": seq.get("exhausted_at"),
            "source": "observed",
            "note": ("Each counter value shows a different person, in this play order "
                     "(byte order differs). Targets are offsets nested inside this "
                     "scene's choice blocks; not yet walkable node ids — needs a carve "
                     "into per-person sub-scenes to route at runtime.")}

    # NOTE: money rewards are NOT derived from the reward text. Each reward is a
    # real, byte-derived `var_add` on var2000 (from the bytecode's
    # `read var2000 ; push N ; ADD ; write` increment), already emitted as its own
    # node. An earlier approach also attached a text-derived `effects` entry to the
    # "Kim earned N dollars" line, but that duplicated the var_add and made the
    # engine apply each payout twice. The var_add nodes are the single source of
    # truth for money, so no text-derived effects are added.

    for s in scenes:
        pass

    # Emphasis -> BBCode, by context:
    #   * dialogue            -> italic + the SPEAKER's gender colour (lead included)
    #   * narrator (no speaker)-> bold only, no colour
    #   * quiz / minigame question -> no emphasis (backtick markers stripped)
    # A quiz question shows both as a narration and as a choice prompt, so a
    # narration whose text mirrors a choice prompt is treated as a question.
    # Non-destructive: raw `text`/`prompt` are left as-is; display goes in
    # `text_bbcode` / `prompt_bbcode`.
    emph = doc.pop("_emphasis", None) or {}
    male_color = emph.get("male_color", EMPHASIS_MALE)
    female_color = emph.get("female_color", EMPHASIS_FEMALE)
    emph_terms = emph.get("term_colors", {})
    genders = doc.get("genders", {})

    def base_color_for(node):
        g = genders.get(node.get("speaker")) if node.get("speaker") else None
        return male_color if g == "M" else female_color if g == "F" else None

    def norm_q(s):
        return (s or "").strip().strip("'").replace("`", "").strip()

    question_set = set()

    def gather(nodes):
        for nd in nodes:
            if nd.get("type") == "choice" and nd.get("prompt"):
                question_set.add(norm_q(nd["prompt"]))
            bd = nd.get("branch_dialogue")
            if isinstance(bd, list):
                for x in bd:
                    if isinstance(x, dict) and "lines" in x:
                        gather(x["lines"])
                    elif isinstance(x, dict):
                        gather([x])
            for oc in nd.get("outcomes", []) or []:
                gather(oc.get("lines", []))
            gather(nd.get("after_choice", []) or [])
            if nd.get("type") == "gate":
                gather(nd.get("then", []) or [])
    for s in scenes:
        gather(s.get("nodes", []))

    def strip_ticks(s):
        return s.replace("`", "")

    def bold_emphasis(s):
        return re.sub(r'`([^`]+)`', lambda m: "[b]%s[/b]" % m.group(1), s)

    def add_bbcode(nodes):
        for nd in nodes:
            typ = nd.get("type")
            txt = nd.get("text")
            if isinstance(txt, str):
                bb = None
                if typ == "dialogue":
                    bb = emphasis_to_bbcode(txt, base_color_for(nd), emph_terms)
                elif typ == "minigame" and "`" in txt:
                    bb = strip_ticks(txt)
                elif typ in ("narration", "status"):
                    if "`" in txt:
                        bb = strip_ticks(txt) if norm_q(txt) in question_set else bold_emphasis(txt)
                if bb is not None and bb != txt:
                    nd["text_bbcode"] = bb
            if typ == "choice" and isinstance(nd.get("prompt"), str) and "`" in nd["prompt"]:
                nd["prompt_bbcode"] = strip_ticks(nd["prompt"])   # questions: no emphasis
            bd = nd.get("branch_dialogue")
            if isinstance(bd, list):
                for x in bd:
                    if isinstance(x, dict) and "lines" in x:
                        add_bbcode(x["lines"])
                    elif isinstance(x, dict):
                        add_bbcode([x])
            for oc in nd.get("outcomes", []) or []:
                add_bbcode(oc.get("lines", []))
            add_bbcode(nd.get("after_choice", []) or [])
            if nd.get("type") == "gate":
                add_bbcode(nd.get("then", []) or [])

    for s in scenes:
        add_bbcode(s.get("nodes", []))

    # Field provenance, so a consumer knows exactly what each value is:
    #   byte_literal      -- read directly from the .kiw bytes; never invented
    #   derived_from_text -- computed from byte-literal text by a generic rule
    #   structural        -- graph bookkeeping over byte-anchored positions
    #   observed          -- per-episode overlay data (real gameplay, not bytes);
    #                        such values also carry `source: observed` in place
    doc["provenance"] = {
        "byte_literal": ["text", "prompt", "speaker", "emotion_code", "options[].label",
                         "asset_id", "track_id", "sfx_id", "image", "offset", "script",
                         "cast", "genders (name-table type field)", "name_vars",
                         "minigames.* (subtitle/words/answers/letters/correct/decoys)",
                         "var_set/var_add (raw variable ids and values from the "
                         "0x2c write op; no variable is given a name)"],
        "derived_from_text": ["effects where source=derived_from_text "
                              "(the 'earned N dollars' line)",
                              "options[].paid", "node type classification "
                              "(narration vs status vs minigame prompt)",
                              "text_bbcode / prompt_bbcode (emphasis markers are "
                              "byte-literal; colour-by-gender is the engine rule)"],
        "structural": ["id", "entry", "next", "to_scene", "to_node", "leads_to",
                       "branch_type", "counter_gated", "outcomes segmentation "
                       "(split at the replay banners)"],
        "observed": ["everything marked source=observed: gates, routers, "
                     "option limits, carve regions, ad sequence, bg names, "
                     "emphasis colours, minigame triggers"],
        "engine_constants": ["emphasis default colours (overlay-overridable), "
                             "standard-background id band (1000-1200), music/sfx "
                             "id bands, retry-banner phrases used to segment "
                             "outcomes -- engine-level conventions, not "
                             "per-episode data"],
    }
    # Flatten the post-merge continuation: a choice's `after_choice` is the shared
    # dialogue that plays once the option branches reconverge at merge_offset. Keeping
    # it nested inside the choice makes the scene hard to walk, so hoist it to be normal
    # sibling nodes right after the choice. The choice keeps its divergent parts
    # (options / branch_dialogue / merge_offset) and gains `continues_at`, pointing to
    # the first continuation node. (The story-export already treats after_choice and the
    # following siblings identically, so this is transparent there.)
    for s in scenes:
        flat = []
        for nd in s.get("nodes", []):
            if isinstance(nd, dict) and nd.get("type") == "choice" \
                    and nd.get("after_choice"):
                cont = nd.pop("after_choice")
                if cont:
                    nd["continues_at"] = cont[0].get("id")
                    nd["note"] = ("Per-option branches resolved from the bytecode "
                                  "dispatch. The shared continuation past the merge "
                                  "point now follows as sibling nodes (see "
                                  "continues_at); merge_offset marks where the "
                                  "branches reconverge.")
                flat.append(nd)
                flat.extend(cont)
            else:
                flat.append(nd)
        s["nodes"] = flat

    # Variable defaults: every state variable this episode references, each starting
    # at 0. The engine zero-inits the whole variable store at episode load (no .kiw
    # script writes a nonzero initial value -- the first touch of nearly every var is
    # an `add`, which only makes sense from a known 0). At runtime they are changed by
    # var_set (replace) and var_add (accumulate) nodes. The var ids are byte-literal
    # (from the 0x2c write / 0x2d read ops); the 0 default is the engine's zero-init.
    set_vars, add_vars, read_vars = set(), set(), set()

    def collect_vars(nodes):
        for nd in nodes:
            if not isinstance(nd, dict):
                continue
            t = nd.get("type")
            if t in ("var_set", "var_add") and nd.get("var") is not None:
                (set_vars if t == "var_set" else add_vars).add(str(nd["var"]))
            g = nd.get("gate")
            if isinstance(g, dict) and g.get("var") is not None:
                read_vars.add(str(g["var"]))
            sd = nd.get("section_dispatch")
            if isinstance(sd, dict) and sd.get("register") is not None:
                read_vars.add(str(sd["register"]))
            for b in (nd.get("branch_dialogue") or []):
                if isinstance(b, dict) and "lines" in b:
                    collect_vars(b["lines"])
            for oc in (nd.get("outcomes") or []):
                if isinstance(oc, dict) and "lines" in oc:
                    collect_vars(oc["lines"])

    for s in scenes:
        collect_vars(s.get("nodes", []))
        sd = s.get("section_dispatch")
        if isinstance(sd, dict) and sd.get("register") is not None:
            read_vars.add(str(sd["register"]))
        for vg in (s.get("var_gates") or []):
            if isinstance(vg, dict) and vg.get("var") is not None:
                read_vars.add(str(vg["var"]))

    # "score" is the engine's running minigame score (op5f/op3f), not a slot in
    # the numeric variable store, so it must not become a variable_defaults entry
    # (and would break the numeric sort below).
    set_vars = {v for v in set_vars if v.isdigit()}
    add_vars = {v for v in add_vars if v.isdigit()}
    read_vars = {v for v in read_vars if v.isdigit()}

    all_vars = set_vars | add_vars | read_vars
    if all_vars:
        doc["variable_defaults"] = {
            "note": ("Every state variable this episode references. The engine "
                     "zero-inits the variable store at load, so each starts at 0; "
                     "no script sets a nonzero initial value. At runtime they are "
                     "changed by var_set (replace) and var_add (accumulate) nodes. "
                     "Variables in `set_only` are only ever assigned, never "
                     "accumulated (1001 is the section/dispatch register)."),
            "values": {v: 0 for v in sorted(all_vars, key=int)},
            "set_only": sorted(set_vars - add_vars, key=int),
        }

    return doc


def _render_branches_md(lines, md_line):
    """Render markdown with choice branches labelled per option.

    After a choice, the option branches appear in order in the bytecode,
    separated by 0x42 jumps (emitted as __SEP__). When the number of separators
    in a choice's region matches (#options - 1), each branch is labelled with its
    option; otherwise the dialogue is shown flat with a note (the engine uses a
    jump-table / minigame-loop layout that can't be split reliably here).
    """
    out = []
    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i]
        if ln.speaker == "__CHOICE__":
            out.append(md_line(ln))
            options = ln.text.split("|")
            # Region = up to the next choice / scene change / end.
            j = i + 1
            while j < n and lines[j].speaker not in ("__CHOICE__", "__SCENE__"):
                j += 1
            region = lines[i + 1:j]
            seps = [k for k, r in enumerate(region) if r.speaker == "__SEP__"]
            if options and len(seps) == len(options) - 1 and len(options) > 1:
                # Clean split: label each segment with its option.
                bounds = [-1] + seps + [len(region)]
                for opt_idx in range(len(options)):
                    seg = region[bounds[opt_idx] + 1:bounds[opt_idx + 1]]
                    out.append(f"> &nbsp;&nbsp;↳ **If «{options[opt_idx]}»:**\n")
                    for r in seg:
                        s = md_line(r, indent="&nbsp;&nbsp;&nbsp;&nbsp;")
                        if s is not None:
                            out.append(s)
                i = j
                continue
            else:
                if len(options) > 1:
                    out.append("> _(branch boundaries not cleanly separable here — "
                               "dialogue for all options follows in script order)_\n")
            i += 1
            continue
        s = md_line(ln)
        if s is not None:
            out.append(s)
        i += 1
    return "\n".join(out)


def render(lines, cast, title="Decoded transcript", fmt="md", backgrounds=True,
           episode=None, branches=False, main_characters=None, scene_controls=None,
           overlay=None, genders=None, minigame_banks=None, name_vars=None,
           choice_dispatch=None, chapter_heads=None, var_gates=None,
           section_dispatch=None, minigame_gates=None, buildword_scoring=None,
           merge_jumps=None, choice_forks=None,
           score_tier_thresholds=None):
    def bg_text(emotion):
        kind, code = emotion
        if kind == "custom":
            name = KNOWN_BG.get(code, "background")
            return f"Background -> {name} (PNG {code})"
        return f"Background -> standard location {code}"

    out = []
    if fmt == "json":
        doc = build_episode_json(lines, cast, episode, title, main_characters,
                                 scene_controls=scene_controls, genders=genders,
                                 minigame_banks=minigame_banks, name_vars=name_vars,
                                 choice_dispatch=choice_dispatch,
                                 chapter_heads=chapter_heads, var_gates=var_gates,
                                 section_dispatch=section_dispatch,
                                 minigame_gates=minigame_gates,
                                 buildword_scoring=buildword_scoring,
                                 merge_jumps=merge_jumps, choice_forks=choice_forks,
                  score_tier_thresholds=score_tier_thresholds)
        if overlay:
            doc = apply_overlay(doc, overlay)
        doc = normalize_graph(doc)
        return json.dumps(stringify_numbers(doc), indent=2, ensure_ascii=False)

    def md_line(ln, indent=""):
        if ln.speaker == "__SCENE__":
            return f"\n## {ln.text}\n"
        if ln.speaker == "__POV__":
            return f"\n> 🎮 **Now playing as: {ln.text}**\n"
        if ln.speaker == "__POVSET__":
            return f"\n> 🎮 **Now playing as: {ln.text}**\n"
        if ln.speaker == "__CARD__":
            kind = "Title card" if ln.emotion == "title" else "End card"
            return f"> 🎬 **{kind}:** {ln.text}\n"
        if ln.speaker == "__CHOICE__":
            opts = " / ".join(ln.text.split("|"))
            pr = f" _{ln.emotion}_" if ln.emotion else ""
            return f"> ❓ **Choice:**{pr} {opts}\n"
        if ln.speaker == "__MINIGAME_REF__":
            return f"> 🎮 **Minigame ({ln.emotion})** — words from scene bank\n"
        if ln.speaker == "__MINIGAME__":
            if ln.emotion == "action":
                parts = ln.text.split("§")
                pr = " ".join(p for p in parts if "|" not in p)
                gr = " // ".join(p for p in parts if "|" in p)
                return f"> 🎮 **Minigame (action):** {pr} — {gr}\n"
            return f"> 🎮 **Minigame ({ln.emotion}):** {ln.text}\n"
        if ln.speaker == "__VAR__":
            kind, var, val = ln.emotion
            sym = "=" if kind == "set" else "+="
            return f"> 🧮 _var {var} {sym} {val}_\n"
        if ln.speaker == "__SFX__":
            return f"> 🔊 _sound effect {ln.emotion}_\n"
        if ln.speaker == "__MUSIC__":
            if ln.emotion == "stop":
                return "> 🎵 _music → stop_\n"
            return f"> 🎵 _music → track {ln.emotion}_\n"
        if ln.speaker == "__VIBRATE__":
            return "> 📳 _vibrate_\n"
        if ln.speaker == "__WOBBLE__":
            return "> 〰️ _wobble next line_\n"
        if ln.speaker == "__LOADING__":
            return "> ⏳ _loading_\n"
        if ln.speaker in ("__SCENEVAL__", "__UIDEFAULT__"):
            return ""
        if ln.speaker == "__SETSTR__":
            _k, _, _v = (ln.text or "").partition("\x1f")
            return f"> 🔤 _{_k} = {_v}_\n"
        if ln.speaker == "__TEXTINPUT__":
            _t, _, _p = (ln.text or "").partition("\x1f")
            return f"> ⌨️ _text input: {_p or _t}_\n"
        if ln.speaker == "__PICKCHAR__":
            return f"> 👥 _pick character: {ln.text}_\n"
        if ln.speaker == "__GOTO__":
            return f"> ➡️ _go to scene 0x{ln.emotion:04x}_\n"
        if ln.speaker == "__STATUS__":
            return f"> 📊 _{ln.text}_\n"
        if ln.speaker == "__BG__":
            return f"> ▶ {bg_text(ln.emotion)}\n" if backgrounds else None
        if ln.speaker == "__SEP__":
            return None
        if ln.speaker is None:
            return f"{indent}*[narration / system]* — {ln.text}\n"
        tag = f" *(emotion {ln.emotion})*" if ln.emotion else ""
        img = f" `[img {ln.sprite}]`" if ln.sprite is not None else " `[img: player avatar]`"
        return f"{indent}**{ln.speaker}**{tag}{img} — {ln.text}\n"

    if fmt == "md":
        out.append(f"# {title}\n")
        if episode:
            out.append(f"**Episode:** {episode}\n")
        if cast:
            out.append("Cast: " + ", ".join(f"{i}={n}" for i, n in enumerate(cast)) + "\n")
        out.append("")
        if branches:
            return "\n".join(out) + "\n" + _render_branches_md(lines, md_line)
        for ln in lines:
            s = md_line(ln)
            if s is not None:
                out.append(s)
    else:  # plain text
        if episode:
            out.append(f"[EPISODE] {episode}")
        for ln in lines:
            if ln.speaker == "__SCENE__":
                out.append(f"\n===== {ln.text} =====")
            elif ln.speaker == "__CARD__":
                kind = "TITLE CARD" if ln.emotion == "title" else "END CARD"
                out.append(f"[{kind}] {ln.text}")
            elif ln.speaker == "__CHOICE__":
                out.append(f"[CHOICE] {' / '.join(ln.text.split('|'))}")
            elif ln.speaker == "__MINIGAME_REF__":
                out.append(f"[MINIGAME {ln.emotion} <- scene bank]")
            elif ln.speaker == "__MINIGAME__":
                out.append(f"[MINIGAME {ln.emotion}] {ln.text}")
            elif ln.speaker == "__POV__":
                out.append(f"\n[NOW PLAYING AS: {ln.text}]")
            elif ln.speaker == "__POVSET__":
                out.append(f"\n[NOW PLAYING AS: {ln.text}]")
            elif ln.speaker == "__VAR__":
                kind, var, val = ln.emotion
                out.append(f"[VAR {var} {'=' if kind=='set' else '+='} {val}]")
            elif ln.speaker == "__SFX__":
                out.append(f"[SFX -> {ln.emotion}]")
            elif ln.speaker == "__MUSIC__":
                out.append(f"[MUSIC -> {ln.emotion}]")
            elif ln.speaker == "__VIBRATE__":
                out.append("[VIBRATE]")
            elif ln.speaker == "__GOTO__":
                out.append(f"[GOTO SCENE 0x{ln.emotion:04x}]")
            elif ln.speaker == "__STATUS__":
                out.append(f"[STATUS: {ln.text}]")
            elif ln.speaker == "__SEP__":
                continue
            elif ln.speaker == "__BG__":
                if backgrounds:
                    out.append(f"=== {bg_text(ln.emotion)} ===")
            elif ln.speaker is None:
                out.append(f"(narration): {ln.text}")
            else:
                emo = f" [emotion {ln.emotion}]" if ln.emotion else ""
                img = f" (img {ln.sprite})" if ln.sprite is not None else " (img: avatar)"
                out.append(f"{ln.speaker}{emo}{img}: {ln.text}")
    return "\n".join(out)



# --------------------------------------------------------------------------- #
# Story export: one JSON file per linear segment
# --------------------------------------------------------------------------- #
def build_story_segments(doc, var_gates=None):
    """Build the episode as a set of small LINEAR story segments (the in-memory form
    behind both the story-export folder and the single-JSON `segments` map). Returns
    (files, index): `files` maps a segment name -> its body, and each body ends in
    exactly one terminal naming the next segment(s):

      {"next": "<segment>"}                       plain continuation
      {"choice": {prompt, options:[{index,label,next}], after}}
      {"gate": {"var","equals","then","else","op"?}}  play `then` when the test
                                                  holds, else `else`. The test is
                                                  var==equals by default, or
                                                  var<equals when op=="lt" (the
                                                  0x0e less-than compare; e.g. the
                                                  academic-challenge option shows
                                                  while var2002 < 3).
      {"goto_scene": {"scene", "file", "section_register"?}}
      {"end": true}

    `section_register` carries the value written to the dispatch register
    (e.g. var 1001) just before the jump -- the destination scene selects its
    section from it. Scene-level variable gates (var_gates) are nested before
    segmentation so girl-variant blocks etc. become separate files.
    """
    import json as _json
    files = {}
    scene_entry = {}
    counters = {}

    def newname(scene, tag):
        counters[scene] = counters.get(scene, 0) + 1
        base = "s%s_%02d" % (scene, counters[scene])
        return base + (("_" + tag) if tag else "")

    def title_of(nodes):
        for nd in nodes:
            if nd.get("type") in ("dialogue", "narration", "title_card"):
                t = (nd.get("text") or "")[:48]
                if t:
                    return t
        return ""

    def nest_scene_gates(nodes, gates):
        gates = sorted(gates, key=lambda g: g["at"])
        def offs_of(nd):
            o = nd.get("offset")
            return int(o) if o is not None else None
        def apply(lst, lo, hi):
            inner = [g for g in gates if lo <= g["at"] < hi]
            if not inner:
                return lst
            outl, used, cursor = [], set(), lo
            for g in inner:
                if g["at"] < cursor:
                    continue
                glo, ghi = g["then"]
                pre, body = [], []
                for x in lst:
                    if id(x) in used:
                        continue
                    o = offs_of(x)
                    if o is None:
                        continue
                    if cursor <= o < g["at"]:
                        pre.append(x); used.add(id(x))
                    elif glo <= o < ghi:
                        body.append(x); used.add(id(x))
                outl.extend(pre)
                _gnd = {"type": "gate", "var": g["var"],
                             "equals": g["equals"],
                             "then": apply(body, glo, ghi),
                             "offset": g["at"]}
                if g.get("op"):
                    _gnd["op"] = g["op"]
                outl.append(_gnd)
                cursor = ghi
            for x in lst:
                if id(x) not in used:
                    o = offs_of(x)
                    if o is None or o >= cursor or o < lo:
                        outl.append(x)
            return outl
        offs = [offs_of(x) for x in nodes if offs_of(x) is not None]
        if not offs:
            return nodes
        # Leading no-offset nodes (head cards: the scene-intro title/narration that
        # find_cards surfaces) belong at the scene head. If left in place they fall
        # into the trailing leftover pass below and get appended after the last
        # gate -- landing inside that gate's else-branch and displacing the real
        # continuation (a bogus replay loop). Pin them at the front; nest only the
        # offset-bearing body.
        lead, body, seen_off = [], [], False
        for x in nodes:
            if offs_of(x) is None and not seen_off:
                lead.append(x)
            else:
                seen_off = True
                body.append(x)
        return lead + apply(body, 0, max(offs) + 1)

    def split_minigame_forks(nodes, mgr):
        """Replace each `outcome_fork` minigame region with one fork node that
        carries its win[]/lose[] arms, so each arm later segments (and nests
        its own variable gates) independently instead of the two arms' switches
        chaining into one elif ladder. Returns (new_nodes, spans) where spans is
        a list of (lo, hi) offset ranges now owned by fork arms.

        Win arm = nodes in [pass_at, fail_at); lose arm = nodes from fail_at to
        the end of the contiguous strictly-increasing offset run (the next
        section is reached by a dispatch jump, which shows up as the offset
        sequence breaking)."""
        if not mgr:
            return nodes, []
        gates = [g for g in (mgr.get("gates") or [])
                 if g.get("kind") == "outcome_fork"
                 or (g.get("kind") == "content_fork" and g.get("via") == "by_id")
                 or (g.get("via") == "action"
                     and g.get("pass_at") != g.get("fail_at"))]
        if not gates:
            return nodes, []

        def offof(nd):
            o = nd.get("offset")
            return int(o) if o is not None else None

        out = list(nodes)
        spans = []
        for g in sorted(gates, key=lambda x: int(x["pass_at"])):
            pass_at, fail_at = int(g["pass_at"]), int(g["fail_at"])
            win_start = next((i for i, n in enumerate(out)
                              if offof(n) is not None and offof(n) >= pass_at),
                             None)
            if win_start is None:
                # The gate's pass_at falls past the current node list -- the
                # nodes between the trigger and the pass point were already
                # consumed by an earlier fork (e.g. Making Some Dough scene 3
                # has multiple by_id minigames; the first fork's run swallows
                # the intermediate content and the second gate's pass_at now
                # sits inside the folded-away range). Splice a marker at the
                # trigger's insertion point so the runtime still sees this
                # gate's threshold, id, and setup.
                trig = int(g.get("trigger") or 0)
                ins_at = next((i for i, n in enumerate(out)
                               if offof(n) is not None and offof(n) >= trig),
                              len(out))
                marker = {"type": "minigame", "fork": True,
                          "kind": g.get("kind"),
                          "minigame_id": (str(g["minigame_id"])
                                          if g.get("minigame_id") is not None
                                          else None),
                          "via": g.get("via"),
                          "trigger": str(g.get("trigger")),
                          "win_threshold": str(g.get("win_threshold")),
                          "offset": str(g.get("gate"))}
                if g.get("via") == "by_id":
                    for k in range(ins_at - 1, -1, -1):
                        nd = out[k]
                        if not isinstance(nd, dict) or nd.get("fork"):
                            continue
                        if nd.get("type") == "minigame" and nd.get("offset") \
                                and int(nd["offset"]) < trig:
                            if nd.get("setup"):
                                marker["setup"] = nd["setup"]
                            if nd.get("minigame_type"):
                                marker["minigame_type"] = nd["minigame_type"]
                            out.pop(k)
                            ins_at -= 1
                            break
                out = out[:ins_at] + [marker] + out[ins_at:]
                continue
            run_end, prev = len(out), None
            for j in range(win_start, len(out)):
                o = offof(out[j])
                if o is None:
                    continue
                if prev is not None and o <= prev:
                    run_end = j
                    break
                prev = o
            lose_start = run_end
            for j in range(win_start, run_end):
                o = offof(out[j])
                if o is not None and o >= fail_at:
                    lose_start = j
                    break
            win_nodes = out[win_start:lose_start]
            lose_nodes = out[lose_start:run_end]
            if not win_nodes or not lose_nodes:
                # The fork's lose arm isn't in this segment's node list -- the
                # bytecode for the fail path lives past a goto_scene that jumps
                # the runtime elsewhere (e.g. Making Some Dough's Andrew-writer
                # Help Ads job: on a good pick the win arm ends with `goto s5`
                # so the fail-arm bytes at fail_at sit orphaned). Rather than
                # dropping the gate on the floor, splice a lightweight marker
                # BEFORE the win-arm run so the runtime still sees the trigger,
                # threshold, and (for by_id) the round setup at the play point.
                # Fold in the section-header data-def node just like the full
                # fork path does.
                marker = {"type": "minigame", "fork": True,
                          "kind": g.get("kind"),
                          "minigame_id": (str(g["minigame_id"])
                                          if g.get("minigame_id") is not None
                                          else None),
                          "via": g.get("via"),
                          "trigger": str(g.get("trigger")),
                          "win_threshold": str(g.get("win_threshold")),
                          "offset": str(g.get("gate"))}
                if g.get("via") == "by_id":
                    trig = int(g.get("trigger") or 0)
                    for k in range(win_start - 1, -1, -1):
                        nd = out[k]
                        if not isinstance(nd, dict) or nd.get("fork"):
                            continue
                        if nd.get("type") == "minigame" and nd.get("offset") \
                                and int(nd["offset"]) < trig:
                            if nd.get("setup"):
                                marker["setup"] = nd["setup"]
                            if nd.get("minigame_type"):
                                marker["minigame_type"] = nd["minigame_type"]
                            out.pop(k)
                            win_start -= 1     # index shifted after the pop
                            break
                out = out[:win_start] + [marker] + out[win_start:]
                continue
            arm_offs = [offof(n) for n in (win_nodes + lose_nodes)
                        if offof(n) is not None]
            spans.append((min(arm_offs), max(arm_offs) + 1))
            fork = {"type": "minigame", "fork": True,
                    "kind": g.get("kind"),
                    "minigame_id": (str(g["minigame_id"])
                                    if g.get("minigame_id") is not None else None),
                    "via": g.get("via"),
                    "trigger": str(g.get("trigger")),
                    "win_threshold": str(g.get("win_threshold")),
                    "offset": str(g.get("gate")),
                    "win": win_nodes, "lose": lose_nodes}
            out = out[:win_start] + [fork] + out[run_end:]
            # A `by_id` minigame is DEFINED by a data-def node (the pick_word/build
            # bank) in the section header, then PLAYED later by id (1f51). Fold that
            # standalone data-def node into the fork so the one minigame node lands
            # at the play point (after the lead-in), carrying its round setup, and
            # drop the now-duplicated header node.
            if g.get("via") == "by_id":
                trig = int(g.get("trigger") or 0)
                for k in range(win_start - 1, -1, -1):
                    nd = out[k]
                    if not isinstance(nd, dict) or nd.get("fork"):
                        continue
                    if nd.get("type") == "minigame" and nd.get("offset") \
                            and int(nd["offset"]) < trig:
                        if nd.get("setup") and "setup" not in fork:
                            fork["setup"] = nd["setup"]
                        if nd.get("minigame_type"):
                            fork["minigame_type"] = nd["minigame_type"]
                        out.pop(k)
                        break
            elif g.get("via") == "action":
                # The action minigame is emitted as a flat play-point node (the
                # 0x47 prompt/options) just before its score gate. Fold that node
                # into the fork so the single node carries both the game (prompt,
                # options, platform variants) and its win/lose arms -- matching
                # the shape of the by_id / word-bank minigames.
                trig = int(g.get("trigger") or 0)
                for k in range(win_start - 1, -1, -1):
                    nd = out[k]
                    if not isinstance(nd, dict) or nd.get("fork"):
                        continue
                    if nd.get("type") == "minigame" and nd.get("offset") \
                            and int(nd["offset"]) <= trig:
                        for _key in ("prompt", "options", "setup", "variants",
                                     "minigame_type"):
                            if nd.get(_key) is not None and _key not in fork:
                                fork[_key] = nd[_key]
                        out.pop(k)
                        break
        return out, spans

    def nest_into_forks(lst, gates):
        """Recurse into fork arms (and gate `then` bodies) so each arm nests its
        own variable gates. Each arm's offset span naturally selects only the
        gates that fall inside it."""
        for nd in lst:
            if nd.get("type") == "minigame" and nd.get("fork"):
                nd["win"] = nest_scene_gates(nd.get("win") or [], gates)
                nd["lose"] = nest_scene_gates(nd.get("lose") or [], gates)
                nest_into_forks(nd["win"], gates)
                nest_into_forks(nd["lose"], gates)
            elif nd.get("type") == "gate":
                nest_into_forks(nd.get("then") or [], gates)
        return lst

    def last_section_write(nodes):
        v = None
        for nd in nodes:
            if nd.get("type") == "var_set" and str(nd.get("var")) == "1001":
                v = nd.get("value")
        return v

    # Offsets at which to force a segment boundary, so a section-dispatch jump
    # enters exactly there instead of mid-segment. Populated per scene ONLY from
    # overlay bindings flagged `enter_at_section` (observed: that section is
    # entered directly, skipping a block the decoder bundled before it). Cutting
    # generically is unsafe -- it can orphan a section's own lead-in (e.g. the
    # "stood up to Travis" confrontation before Dating Holly) -- so it is opt-in.
    cut_offsets = set()

    # Then-branch offsets (overlay opt-in) whose gate should fall through to its
    # else/continuation rather than end. See _gate_fallthroughs in apply_overlay.
    gate_thread_offs = set()

    # Choice offsets (overlay opt-in) whose section exits to another scene after
    # the choice. {offset: {to_scene, set_register}}. See _choice_exits.
    choice_exits = {}

    # The archive chunk id of the scene currently being segmented, so a scene-goto
    # can be recognized as a SELF-scene re-entry (goto to the scene's own chunk).
    # Set by the per-scene loop before each emit() call.
    _cur_script = [None]

    def emit(name, scene, nodes):
        """Segment a node list into files; returns the entry file name."""
        seg, terminal, k = [], None, 0
        first_name = None

        def flush(term):
            nonlocal seg, k, first_name
            fname = name if k == 0 else newname(scene, "")
            body = [nd for nd in seg
                    if not (nd.get("type") == "gate")]
            files[fname] = {"file": fname + ".json", "scene": scene,
                            "title": title_of(seg), "nodes": list(seg),
                            **term}
            if first_name is None:
                first_name = fname
            seg = []
            k += 1
            return fname

        # iterate; on a control node, the REMAINDER becomes the continuation
        i = 0
        nl = list(nodes)
        while i < len(nl):
            nd = nl[i]
            # Opt-in section boundary (see cut_offsets): flush what we have and
            # continue the rest as its own segment, so the dispatch jump enters
            # here rather than mid-segment. The pre-cut block keeps the original
            # fall-through (next) so it stays consistent if reached another way.
            if (cut_offsets and seg and nd.get("offset") is not None
                    and str(nd["offset"]) in cut_offsets):
                cont = emit(newname(scene, ""), scene, nl[i:])
                flush({"next": cont + ".json"})
                return first_name
            # Hard dispatch-section boundary: a title card that opens a new
            # section body (immediately preceded by a 0x42 SEP, after the scene's
            # 0x12 dispatch). The host VM enters each body via its dispatch, never
            # by running off the end of the previous one -- so the PRIOR section
            # ENDS here and does not fall through into this body. Terminate the
            # current segment and emit this body as its own (dispatch-reached)
            # segment. Guarded on the current segment already holding real content
            # so we never cut at a section's own opening card.
            if (nd.get("type") == "title_card" and nd.get("section_start")
                    and any(x.get("type") in ("dialogue", "narration", "choice",
                                              "minigame")
                            for x in seg)):
                emit(newname(scene, ""), scene, nl[i:])
                # If the prior section's terminating SEP jumped to another section
                # body (e.g. the Sophie-breakup section jumps to the loser arc),
                # the prior section GOTOs there rather than dead-ending. Resolved
                # to a segment name in the post-segmentation offset pass.
                sj = nd.get("_prev_section_jump")
                if sj is not None:
                    flush({"_section_goto_far": int(sj)})
                else:
                    flush({"end": True})
                return first_name
            t = nd.get("type")
            if t == "choice" and isinstance(nd.get("branch_dialogue"), list) \
                    and nd["branch_dialogue"] \
                    and isinstance(nd["branch_dialogue"][0], dict) \
                    and "lines" in nd["branch_dialogue"][0]:
                after = nd.get("after_choice") or []
                rest = after + nl[i + 1:]
                ce = choice_exits.get(str(nd.get("offset")))
                if ce:
                    # This choice ends its section and exits to another scene. Emit
                    # the natural continuation so sibling sections that dispatch into
                    # it still reach it, but THIS choice's options exit (each keeping
                    # its own variable write) to the named scene -- they must not fall
                    # through into the unrelated section that follows in the bytecode.
                    if rest:
                        emit(newname(scene, ""), scene, rest)
                    exit_nodes = [
                        {"type": "var_set", "var": "1001",
                         "value": ce["set_register"], "offset": nd.get("offset")},
                        {"type": "goto_scene", "to_scene": ce["to_scene"],
                         "offset": nd.get("offset")},
                    ]
                    after_file = emit(newname(scene, "exit"), scene, exit_nodes)
                else:
                    # QTE-shaped after_choice ("Hurry!" with one success body and
                    # three fail labels sharing fail content): the bytes between
                    # the success option's region and the merge are FAIL CONTENT
                    # for the other options, not shared continuation. The success
                    # option's body ends with a 0x28 → merge that skips past it;
                    # the fail options reach it via the option-test 2b jumps and
                    # then linearly fall through to the merge. So split `rest`
                    # at the merge offset: pre-merge becomes the fail-content
                    # segment (linear fall-through for non-skip options) and
                    # post-merge becomes the real shared continuation -- to which
                    # the skip-style options route directly.
                    merge_off = nd.get("merge_offset")
                    post_merge_file = None
                    after_file = None
                    if merge_off is not None and rest:
                        def _off_int(r):
                            o = r.get("offset")
                            if o is None or not str(o).lstrip("-").isdigit():
                                return None
                            return int(o)
                        post_rest = [r for r in rest
                                     if _off_int(r) is not None
                                     and _off_int(r) >= int(merge_off)]
                        pre_rest = [r for r in rest if r not in post_rest]
                        # Only treat this as a QTE split when pre_rest holds
                        # ACTUAL displayed content (dialogue / narration / minigame /
                        # title_card). For ordinary choices, `rest` may contain a
                        # stray non-displayable node (e.g. an sfx or music cue
                        # straddling the merge) that lands "before" the merge but
                        # isn't fail content -- splitting on those produces empty
                        # forwarder segments that add no information.
                        displayable_types = {"dialogue", "narration", "minigame",
                                             "title_card", "end_card"}
                        pre_has_content = any(
                            r.get("type") in displayable_types for r in pre_rest)
                        # And do NOT split when pre_rest contains nested control
                        # flow of its own (a nested choice or a scene exit). In
                        # that case the pre-merge bytes are the outer choice's
                        # OPTION-continuation content (e.g. ATGB sophomore's
                        # nested "What to do?" choice then goto → junior year on
                        # the Sophie arm), NOT fail content shared by other
                        # options -- and it must reach the outer choice's per-
                        # option branches through the normal segment chain rather
                        # than being carved off as an "after" arm.
                        pre_has_nested = any(
                            r.get("type") in ("choice", "goto_scene")
                            for r in pre_rest)
                        if (pre_rest and post_rest
                                and pre_has_content and not pre_has_nested):
                            post_merge_file = emit(
                                newname(scene, "after"), scene, post_rest)
                            after_file = emit(
                                newname(scene, "after"), scene, pre_rest)
                            # Chain pre-merge's terminal to point to post-merge,
                            # rewriting the file-level `end` (mirrors the per-
                            # option tail rewrite below). This way the fail-content
                            # arm naturally continues into the shared merge segment.
                            tail = files[sorted(f for f in files
                                                if f.startswith(after_file))[-1]] \
                                if after_file in files else files.get(after_file)
                            if tail is not None and "end" in tail:
                                tail.pop("end", None)
                                tail["next"] = post_merge_file + ".json"
                    if after_file is None:
                        after_file = emit(newname(scene, "after"), scene, rest) \
                            if rest else None
                opts = []
                branch_by_idx = {}
                for br in nd["branch_dialogue"]:
                    keyi = br["index"]
                    keys = keyi if isinstance(keyi, list) else [keyi]
                    bf = emit(newname(scene, "opt%s" %
                              "_".join(str(x) for x in keys)), scene,
                              br["lines"] or [])
                    if br["lines"] == [] and after_file:
                        bf = after_file
                    for kk in keys:
                        branch_by_idx[str(kk)] = bf
                    # branch falls through to the merge unless it ends in goto
                    last = (br["lines"] or [{}])[-1]
                    if last.get("type") != "goto_scene" and after_file \
                            and br["lines"]:
                        # If this option's last offset is well before the choice's
                        # merge, its body ended with a 0x28 → merge that skips
                        # fail content; route directly to the post-merge segment.
                        # But NOT when the option's body contains a nested choice
                        # or scene exit -- in those cases the "gap" between last
                        # decoded offset and merge is the option's OWN nested
                        # content (e.g. a nested "What to do?" choice + a
                        # post-nested goto_scene → junior year in the ATGB
                        # sophomore Sophie branch), not fail content for siblings.
                        # Redirecting the option's per-branch tail to post-merge
                        # would skip that nested content and land it in the next
                        # scene's material.
                        target_file = after_file
                        merge_off2 = nd.get("merge_offset")
                        has_nested = any(
                            x.get("type") in ("choice", "goto_scene")
                            for x in br["lines"])
                        if (post_merge_file is not None
                                and merge_off2 is not None
                                and not has_nested):
                            last_off = None
                            for x in reversed(br["lines"]):
                                o = x.get("offset")
                                if o is not None and str(o).lstrip("-").isdigit():
                                    last_off = int(o)
                                    break
                            if last_off is not None \
                                    and last_off < int(merge_off2):
                                target_file = post_merge_file
                        tail = files[sorted(f for f in files
                                            if f.startswith(bf))[-1]] \
                            if bf in files else files.get(bf)
                        if tail is not None and "end" in tail:
                            tail.pop("end", None)
                            tail["next"] = target_file + ".json"
                far_map = nd.get("_far_targets") or {}
                for o in nd.get("options", []):
                    nf = branch_by_idx.get(str(o["index"]), after_file)
                    far = far_map.get(str(o["index"]))
                    opt = {"index": o["index"], "label": o["label"],
                           **({"effects": o["effects"]} if o.get("effects") else {})}
                    if far is not None:
                        # non-contiguous branch: link to the jumped-to section in the
                        # post-segmentation pass (offset -> segment) rather than the
                        # fall-through merge.
                        opt["_far_offset"] = int(far)
                    else:
                        opt["next"] = (nf + ".json") if nf else None
                    opts.append(opt)
                term = {"choice": {"id": nd.get("id"),
                                   "prompt": nd.get("prompt"),
                                   "options": opts,
                                   **({"after": after_file + ".json"}
                                      if after_file else {})}}
                flush(term)
                return first_name
            if t == "choice" and isinstance(nd.get("branch_dialogue"), list) \
                    and nd["branch_dialogue"]:
                # UNRESOLVED choice: per-option boundaries were not recovered from the
                # bytecode, so branch_dialogue is still a flat node run (everything that
                # follows the choice in script order, up to the next choice/scene). Emit
                # it as a single continuation rather than dropping it -- otherwise any
                # terminal inside it (e.g. a goto_scene to the next scene) is lost and
                # the options dead-end at null. All options lead into that continuation.
                cont = (nd["branch_dialogue"] + (nd.get("after_choice") or [])
                        + nl[i + 1:])
                cont_file = emit(newname(scene, "after"), scene, cont) if cont else None
                opts = [{"index": o["index"], "label": o["label"],
                         **({"effects": o["effects"]} if o.get("effects") else {}),
                         "next": (cont_file + ".json") if cont_file else None}
                        for o in nd.get("options", [])]
                term = {"choice": {"id": nd.get("id"), "prompt": nd.get("prompt"),
                                   "options": opts,
                                   "note": ("Option branches not separable from the "
                                            "bytecode; all options continue into the "
                                            "shared following block in script order."),
                                   **({"after": cont_file + ".json"} if cont_file else {})}}
                flush(term)
                return first_name
            if t == "minigame" and nd.get("fork"):
                # an outcome_fork: the segment so far ends in the minigame; its
                # score routes to a win arm or a lose arm (each its own segment)
                win_f = emit(newname(scene, "win"), scene, nd.get("win") or [])
                lose_f = emit(newname(scene, "lose"), scene,
                              nd.get("lose") or [])
                mgt = {"kind": nd.get("kind"),
                       "minigame_id": nd.get("minigame_id"),
                       "via": nd.get("via"),
                       "trigger": nd.get("trigger"),
                       "win_threshold": nd.get("win_threshold"),
                       "offset": nd.get("offset"),
                       "note": ("Engine minigame; routes to win/lose by score "
                                "vs win_threshold. `setup` is the word bank the "
                                "player picks from (correct vs decoys)."),
                       "win": win_f + ".json",
                       "lose": lose_f + ".json"}
                # A by_id fork is played from a bank defined in the section header;
                # carry that bank as the fork's pick setup so the node names its
                # actual minigame (pick_word/build_word) instead of the fork kind.
                if nd.get("via") == "by_id" and "setup" not in mgt:
                    _b = scene_banks.get(str(scene), {})
                    for _k in ("pick_word", "build_word", "quiz"):
                        if _b.get(_k):
                            mgt["setup"] = {"bank": _k, "rounds": _b[_k]}
                            # pick_word / build_word skill games are time-bound:
                            # the game draws prompts from the bank until a ~20s
                            # timer expires ("Time's up!"), needing win_threshold
                            # correct to pass. Flag the timed nature so the runtime
                            # doesn't treat `rounds` as a fixed sequence.
                            if _k in ("pick_word", "build_word"):
                                mgt["timed"] = True
                                mgt["timer_ms"] = "20000"
                                mgt["setup"]["timer_ms"] = "20000"
                            break
                # An action (0x47) minigame carries its own play-point content --
                # the prompt and the pipe-delimited option groups folded in when
                # the fork absorbed the flat node. Carry those onto the emitted
                # node so it names the actual game, not just the fork skeleton.
                # Platform variants (mobile pick_word / tablet build_word) are
                # attached by the later scene-bank variant pass, which runs on
                # every minigame node that has a playable minigame_type.
                if nd.get("via") == "action":
                    for _k in ("prompt", "options", "minigame_type", "variants"):
                        if nd.get(_k) is not None:
                            mgt[_k] = nd[_k]
                    # The 0x47 action minigame's inline option groups (correct
                    # vs decoys) ARE the mobile pick_word form, so name it that
                    # when the fork skeleton left it generic. The later scene-bank
                    # variant pass adds the tablet build_word form when a bank for
                    # this scene exists.
                    if mgt.get("minigame_type") in (None, "content_fork") \
                            and mgt.get("options"):
                        mgt["minigame_type"] = "pick_word"
                term = {"minigame": mgt}
                flush(term)
                # bytecode after the fork is reached via dispatch, not fallthrough
                rest = nl[i + 1:]
                if rest:
                    emit(newname(scene, ""), scene, rest)
                return first_name
            if t == "gate":
                then_nodes = nd.get("then") or []
                if not any(x.get("type") in ("dialogue", "narration", "choice",
                                             "title_card", "minigame",
                                             "status", "goto_scene", "next")
                           for x in then_nodes):
                    # a guard around pure state writes (e.g. first-run init):
                    # keep inline, annotated, instead of splitting files.
                    # NOTE: status / goto_scene / next are excluded from this
                    # optimization because runtimes read them as sequential
                    # actions (display a HUD line, jump to another scene) --
                    # inlining them with a `when` field puts the guard in the
                    # payload where it can be missed and the action fired
                    # unconditionally (e.g. Making Some Dough's s5 day-end,
                    # where the day-6 / day-10 loop-back arms would otherwise
                    # short-circuit the "next day, after school" hub).
                    for x in then_nodes:
                        x2 = dict(x)
                        x2["when"] = {"var": nd["var"], "equals": nd["equals"],
                                      **({"op": nd["op"]} if nd.get("op") else {})}
                        seg.append(x2)
                    i += 1
                    continue
                _optag = "lt" if nd.get("op") == "lt" else "eq"
                then_f = emit(newname(scene, "var%s_%s%s" % (nd["var"],
                              _optag, nd["equals"])), scene, then_nodes)
                rest = nl[i + 1:]
                else_f = emit(newname(scene, "else"), scene, rest) \
                    if rest else None
                # Opt-in fall-through (see gate_thread_offs): this guard's
                # then-branch is extra content shown when var==N, after which the
                # engine continues into the shared continuation (the else). Thread
                # the then-branch's dead-ending tail into that continuation.
                then_off = then_nodes[0].get("offset") if then_nodes else None
                if (else_f and then_off is not None
                        and str(then_off) in gate_thread_offs):
                    tnames = sorted(f for f in files if f == then_f
                                    or f.startswith(then_f + "_"))
                    tail = files.get(tnames[-1]) if tnames else files.get(then_f)
                    if tail is not None and "end" in tail:
                        tail.pop("end", None)
                        tail["next"] = else_f + ".json"
                term = {"gate": {"var": nd["var"], "equals": nd["equals"],
                                 **({"op": nd["op"]} if nd.get("op") else {}),
                                 **({"offset": nd["offset"]}
                                    if nd.get("offset") is not None else {}),
                                 "then": then_f + ".json",
                                 **({"else": else_f + ".json"}
                                    if else_f else {})}}
                flush(term)
                return first_name
            seg.append(nd)
            if t == "goto_scene":
                sec = last_section_write(seg)
                rest = nl[i + 1:]
                # A SELF-scene goto (jump to this scene's own chunk) that sets a
                # section register and is immediately followed by more content is
                # the "arm the re-entry point, then keep playing" idiom: on the
                # first pass execution falls straight through into the content
                # after the goto; the register only matters if the scene is later
                # re-entered from elsewhere. Treat it as a plain continuation into
                # that content (recording the register as a var_set so re-entry
                # still routes correctly) rather than a jump that skips the tail.
                # Without this, the fall-through content is orphaned -- e.g. A
                # Float Is Born's "Later that evening... / That day after school"
                # beats after Linda commits to the float. Cross-scene gotos and
                # self-gotos with no trailing content keep the jump semantics.
                _self = (_cur_script[0] is not None
                         and nd.get("script") == _cur_script[0])
                if _self and rest and sec is not None:
                    # the goto itself becomes a no-op fall-through: drop it from
                    # the segment (the register write is already a decoded var_set
                    # node earlier in `seg`, which last_section_write just read),
                    # and continue straight into the trailing content.
                    if seg and seg[-1] is nd:
                        seg.pop()
                    cont = emit(newname(scene, ""), scene, rest)
                    flush({"next": cont + ".json"})
                    return first_name
                term = {"goto_scene": {"scene": nd.get("to_scene"),
                                       "script": nd.get("script"),
                                       **({"section_register":
                                           {"var": "1001", "value": sec}}
                                          if sec is not None else {})}}
                flush(term)
                # bytecode after an unconditional goto in the same list is
                # reached via other paths; keep emitting it as a fresh file
                if rest:
                    emit(newname(scene, ""), scene, rest)
                return first_name
            i += 1
        flush({"end": True})
        return first_name

    gates_by_script = var_gates or {}
    scene_banks = {}
    for sc in doc["scenes"]:
        label = sc.get("script")
        bank = sc.get("minigames") or {}
        scene_banks[str(sc["scene"])] = {k: v for k, v in bank.items()
                                         if k != "note"}
        sgates = gates_by_script.get(label, [])
        raw, fork_spans = split_minigame_forks(sc["nodes"],
                                               sc.get("minigame_results"))
        # gates inside a fork arm nest within that arm; the rest nest top-level
        outer = [g for g in sgates
                 if not any(lo <= g["at"] < hi for lo, hi in fork_spans)]
        nodes = nest_scene_gates(raw, outer)
        nest_into_forks(nodes, sgates)
        sd = sc.get("section_dispatch") or {}
        gate_thread_offs = set(str(x) for x in
                               (sc.get("_gate_fallthroughs") or []))
        choice_exits = {str(k): v for k, v in (sc.get("_choice_exits") or {}).items()}
        flagged = [int(b["section_offset"]) for b in sd.get("bindings", [])
                   if b.get("section_offset") is not None
                   and b.get("enter_at_section")]
        cut_offsets = set()
        if flagged:
            def _all_offs(lst, acc):
                for x in lst:
                    if not isinstance(x, dict):
                        continue
                    if x.get("offset") is not None:
                        acc.append((int(x["offset"]), x.get("type")))
                    for key in ("then", "win", "lose", "lines",
                                "branch_dialogue", "after_choice"):
                        if isinstance(x.get(key), list):
                            _all_offs(x[key], acc)
                return acc
            _offs = sorted(set(_all_offs(nodes, [])))
            for tgt in flagged:
                # Prefer cutting at a section's title card (its year/scene
                # banner): the decoded node opening the section sits a few
                # bytes past the dispatcher's raw target. When no title card
                # sits near the target (as in Making Some Dough's per-day
                # quiz/job/bakery revisits, which don't emit banners on entry),
                # fall back to the first content-bearing node at or after the
                # target so byte-derived bindings still create the segment cut.
                nxt = next((o for o, ty in _offs
                            if o >= tgt and ty == "title_card"), None)
                if nxt is not None and (nxt - tgt) <= 64:
                    cut_offsets.add(str(nxt))
                    continue
                fallback = next((o for o, ty in _offs
                                 if o >= tgt and ty in (
                                     "dialogue", "narration", "background",
                                     "music", "status", "choice", "minigame",
                                     "var_set", "var_add")), None)
                if fallback is not None and (fallback - tgt) <= 64:
                    cut_offsets.add(str(fallback))
        _cur_script[0] = sc.get("script")
        entry = emit("s%s" % sc["scene"], sc["scene"], nodes)
        # If value-0 of the section dispatch is an enter_at_section landing (the
        # scene's true opening sits past an intervening fall-through section, e.g.
        # Wrong Side of Town's opening combat minigame), the scene ENTRY is that
        # value-0 section, not the physical-first segment. Find the segment whose
        # first node is at/after value-0's target and use it as the entry, so the
        # episode starts on its title sequence rather than mid-minigame. The
        # skipped section stays present (reached by dispatch fall-through / goto).
        _v0_entry = next((b for b in sd.get("bindings", [])
                          if str(b.get("value")) == "0"
                          and b.get("enter_at_section")
                          and b.get("section_offset") is not None), None)
        if _v0_entry is not None:
            _v0t = int(_v0_entry["section_offset"])
            _cand = []
            for _fn, _bd in files.items():
                if _bd.get("scene") != sc["scene"]:
                    continue
                _os = [int(x["offset"]) for x in _bd.get("nodes", [])
                       if isinstance(x, dict) and x.get("offset") is not None
                       and str(x["offset"]).lstrip("-").isdigit()]
                if _os and min(_os) >= _v0t - 4:
                    _cand.append((min(_os), _fn))
            if _cand:
                entry = min(_cand)[1]
        # Combat-walk scenes (e.g. Wrong Side of Town's town gauntlet) can have the
        # shared combat/encounter code physically first, so the entry defaults to
        # that combat dump instead of the walk. When the chosen entry segment is a
        # combat dump AND the scene has a walk-hub segment (a choice whose options
        # include walking, plus a time/distance status), start at the walk hub
        # instead. General: only redirects when both conditions hold; other scenes
        # are untouched.
        def _is_combat_dump(bd):
            txt = " ".join((n.get("text") or "") for n in bd.get("nodes", [])
                           if isinstance(n, dict)).lower()
            return ("still has %d life left" in txt or "goes down!" in txt) \
                and "does %d damage" in txt

        def _is_walk_hub(bd):
            # The walk hub shows the countdown time/distance status. (The walk
            # choice may be split into a following segment during emit, so the
            # time status alone identifies the hub.)
            for n in bd.get("nodes", []):
                if not isinstance(n, dict):
                    continue
                t = (n.get("text") or "").lower()
                if n.get("type") == "status" and ("before sunset" in t
                                                  or "left before" in t):
                    return True
            return False

        _entry_bd = files.get(entry)
        if _entry_bd is not None and _is_combat_dump(_entry_bd):
            # Prefer a walk hub (town-walk scenes). Otherwise, when the scene has a
            # section dispatch whose first valid-value section sits past the combat
            # dump, that section is the true opening (e.g. Wrong Side scene 5's
            # arrival + Ryan confrontation). Use whichever applies.
            _hub = next((_fn for _fn, _bd in files.items()
                         if _bd.get("scene") == sc["scene"] and _is_walk_hub(_bd)),
                        None)
            if _hub is not None:
                entry = _hub
            else:
                # dispatch's first valid-value section target
                _valid = sd.get("valid_values") or []
                _sect = None
                for _b in sd.get("bindings", []):
                    if _b.get("section_offset") is not None \
                            and (not _valid or _b.get("value") in _valid):
                        _sect = int(_b["section_offset"])
                        break
                if _sect is not None:
                    _cand = []
                    for _fn, _bd in files.items():
                        if _bd.get("scene") != sc["scene"]:
                            continue
                        _os = [int(x["offset"]) for x in _bd.get("nodes", [])
                               if isinstance(x, dict) and x.get("offset") is not None
                               and str(x["offset"]).lstrip("-").isdigit()]
                        if _os and min(_os) >= _sect - 4:
                            _cand.append((min(_os), _fn))
                    if _cand:
                        entry = min(_cand)[1]
        # END-OF-EPISODE SCORE SCREEN AT ENTRY.
        # A scene whose header dispatches on the section register (var1001) is
        # entered from the prior scene with the register set to the STORY case:
        # byte-confirmed, the prior scene writes var1001=<case> immediately before
        # its goto here, and a restart path inside this scene resets it. The
        # bytecode-first section is the rank/score screen (the dispatch default),
        # so the physical-first segment -- the default scene entry -- is the score
        # screen, which would wrongly play before the story begins. When the chosen
        # entry opens with the end screen (a score_tier) and its own continuation
        # is a different story segment, replace the entry with a GATE that mirrors
        # the bytecode dispatch: register==<case> -> story, else -> score screen.
        # This keeps the score screen reachable (the else arm) -- no orphan -- and
        # is exactly what the engine does, so it needs no per-episode special-case.
        _entry_bd = files.get(entry)

        def _opens_with_score_tier(bd):
            # The score_tier collapse runs later (in build_segmented_doc), so at
            # this point the entry may still carry the FLATTENED rank narrations.
            # Accept either: a score_tier node, or a run of rank/grade narration
            # lines, possibly preceded by framing narration/status.
            _rank = 0
            for nd in (bd.get("nodes") if bd else []):
                if not isinstance(nd, dict):
                    continue
                if nd.get("type") == "score_tier":
                    return True
                t = (nd.get("text") or "").lower()
                if "rank for this episode" in t or "grade for this episode" in t:
                    _rank += 1
                    if _rank >= 2:
                        return True
                    continue
                if nd.get("type") in ("narration", "status", "notification"):
                    continue
                return _rank >= 2
            return _rank >= 2

        _case = None
        if sd and sd.get("register") is not None:
            _cases = sd.get("cases") or []
            _vals = [c.get("value") for c in _cases if c.get("value") not in (None, 0)]
            _case = _vals[0] if _vals else None

        if (_case is not None and _entry_bd is not None
                and _opens_with_score_tier(_entry_bd)):
            # the story target is the earliest same-scene segment (by bytecode
            # offset) that is not itself an end-screen fragment (the score screen /
            # rank-tier block). Leading bookkeeping nodes (var writes, bg/music/
            # sfx, control) are ignored when judging what a segment "opens" with.
            def _first_content(bd):
                for nd in bd.get("nodes", []):
                    if not isinstance(nd, dict):
                        continue
                    if nd.get("type") in ("var_add", "var_set", "control",
                                          "background", "music", "sfx", "sprite"):
                        continue
                    return nd.get("type")
                return None

            def _is_score_framing(bd):
                # score screen, extra-credit rescore, or quiz-result recap -- all
                # end-of-episode scoring UI, not the story opening.
                if _opens_with_score_tier(bd):
                    return True
                _txt = " ".join((n.get("text") or "") for n in bd.get("nodes", [])
                                if isinstance(n, dict))[:400].lower()
                for _kw in ("you answered", "out of 100 points",
                            "rank for this episode", "grade for this episode",
                            "extra credit", "you got the `perfect` score",
                            "points short of unlocking"):
                    if _kw in _txt:
                        return True
                return False

            _story, _best = None, None
            for _fn, _bd in files.items():
                if _bd.get("scene") != sc["scene"] or _fn == entry:
                    continue
                if _is_score_framing(_bd):
                    continue
                # skip choice-OPTION branch fragments (…_optN): these are the arms
                # of a mid-scene choice and can sit at an earlier bytecode offset
                # than the story spine, but entering on one drops the player into
                # the middle of a conversation. The spine segments (…_after / _win
                # / _lose / the scene head) are the real openings.
                if re.search(r"_opt\d+$", _fn):
                    continue
                # must carry real playable content (dialogue or narration), not be
                # a pure control/notification stub
                if _first_content(_bd) not in ("dialogue", "narration", "status"):
                    continue
                _os = [int(x["offset"]) for x in _bd.get("nodes", [])
                       if isinstance(x, dict) and x.get("offset") is not None
                       and str(x["offset"]).lstrip("-").isdigit()]
                if _os and (_best is None or min(_os) < _best):
                    _best, _story = min(_os), _fn
            _story_bd = files.get(_story) if _story else None
            if (_story_bd is not None and _story != entry
                    and _story_bd.get("scene") == sc["scene"]):
                _reg = sd.get("register")
                _gate_id = "s%s_dispatch" % sc["scene"]
                files[_gate_id] = {
                    "scene": sc["scene"],
                    "nodes": [{
                        "type": "gate",
                        "var": str(_reg),
                        "op": "eq",
                        "equals": _case,
                        "then": _story + ".json",
                        "else": entry + ".json",
                        "note": ("Scene-header section dispatch (byte-derived): the "
                                 "prior scene sets var%s=%s before entering, so the "
                                 "story plays first; the score screen (else) is the "
                                 "dispatch fall-through reached at episode end when "
                                 "the register is reset." % (_reg, _case)),
                    }],
                }
                entry = _gate_id
        scene_entry[str(sc["scene"])] = entry + ".json"

    # resolve word-select minigame refs to the scene's pick/build bank, and
    # attach the bank to any file that references it
    def pick_bank(scene):
        b = scene_banks.get(str(scene), {})
        for key in ("pick_word", "build_word", "quiz"):
            if b.get(key):
                return key, b[key]
        return None, None
    for fname, body in files.items():
        for nd in body["nodes"]:
            if nd.get("type") == "minigame":
                nd.pop("from_scene_bank", None)
        # the fork minigame is a terminal here; give it its word-pick setup
        mg = body.get("minigame")
        if mg and mg.get("via") == "word_bank" and "setup" not in mg:
            key, recs = pick_bank(body["scene"])
            if recs:
                mg["setup"] = {"bank": key, "rounds": recs}

    # Give every playable content minigame the same {bank, rounds} setup as the
    # fork: attach each scene's word/quiz banks to its minigame nodes, one bank
    # type per node in offset order. (Which trigger plays which exact round is
    # not byte-recoverable, so this is a scene-level, by-type attach; a scene's
    # full set of rounds for a type rides on that type's node.)
    # bank types already consumed by a fork in a scene must not be re-attached
    # to that scene's other (engine-rendered, bankless) minigame triggers
    fork_used = {}
    for body in files.values():
        mg = body.get("minigame")
        if mg and mg.get("setup"):
            fork_used.setdefault(str(body.get("scene")), set()).add(
                mg["setup"]["bank"])
    scene_play = {}
    for body in files.values():
        for nd in body["nodes"]:
            if (nd.get("type") == "minigame" and "setup" not in nd
                    and "words" not in nd and not nd.get("win")
                    and nd.get("kind") in ("action", "word-match", "word")):
                scene_play.setdefault(str(body.get("scene")), []).append(nd)
    for sc, nds in scene_play.items():
        b = scene_banks.get(sc, {})
        used = fork_used.get(sc, set())
        typed = sorted(((t, b[t]) for t in ("build_word", "pick_word", "quiz")
                        if b.get(t) and t not in used),
                       key=lambda tr: min((int(r.get("offset", 0))
                                           for r in tr[1]), default=0))
        nds.sort(key=lambda n: int(n["offset"]) if n.get("offset") else 0)
        for i, (t, rounds) in enumerate(typed):
            if i < len(nds):
                nds[i]["setup"] = {"bank": t, "rounds": rounds}
                nds[i].pop("inline_content", None)
                nds[i].pop("note", None)

    # A scene with several distinct pick-word boards (e.g. Making Some Dough's Help
    # Ads: the martial-arts ad's "Dodge the attacks!" board and the writer ad's
    # "Pick the real genres!" board) can end up with the whole bank attached to one
    # minigame node, so the engine can't tell which board that ad plays. Narrow
    # each such node to the single round whose subtitle matches the ad's premise,
    # matched by keyword against the ad's own dialogue. Byte-derived: rounds and
    # subtitles come from the scene word bank; this only routes each board to the
    # ad that plays it. No-op unless a node carries more than one round.
    _ROUND_KEYS = (("dodge", "duck", "attack"), ("genre", "brainstorm"))
    for body in files.values():
        _bank = scene_banks.get(str(body.get("scene")), {})
        if len(_bank.get("pick_word") or []) < 2:
            continue
        _ad_text = " ".join((x.get("text") or "")
                            for x in body.get("nodes") or []
                            if isinstance(x, dict)).lower()
        # a by_id fork minigame rides on body["minigame"], not in body["nodes"];
        # narrow both shapes
        _cands = [nd for nd in (body.get("nodes") or [])
                  if isinstance(nd, dict) and nd.get("type") == "minigame"]
        if isinstance(body.get("minigame"), dict):
            _cands.append(body["minigame"])
        for nd in _cands:
            _rounds = (nd.get("setup") or {}).get("rounds") or []
            if len(_rounds) <= 1:
                continue
            _chosen = None
            for _keys in _ROUND_KEYS:
                if not any(k in _ad_text for k in _keys):
                    continue
                for _r in _rounds:
                    if any(k in (_r.get("subtitle") or "").lower()
                           for k in _keys):
                        _chosen = _r
                        break
                if _chosen:
                    break
            if _chosen is not None:
                nd["setup"] = {"bank": "pick_word", "rounds": [_chosen]}

    # resolve goto_scene terminals to scene entry files
    for f in files.values():
        g = f.get("goto_scene")
        if g and g.get("scene") is not None:
            g["file"] = scene_entry.get(str(g["scene"]))

    # Re-link cross-segment node `next` edges into clean terminals. After
    # threading, a block's last node may point at a merge/continuation that the
    # flattened bytecode placed inside another segment (e.g. several switch/elif
    # arms converging on a shared END). Split segments at those edges so the
    # merge becomes an addressable entry and the jump becomes a `next` terminal.
    _TERMS = ("next", "gate", "goto_scene", "end", "choice", "minigame")

    def _relink_segments():
        def uniq(base):
            nm, k = base, 1
            while nm in files:
                k += 1
                nm = "%s%d" % (base, k)
            return nm
        for _ in range(20000):
            id2seg = {}
            for nm, s in files.items():
                for n in s.get("nodes", []):
                    if "id" in n:
                        id2seg[n["id"]] = nm
            acted = False
            for nm in list(files.keys()):
                s = files[nm]
                nodes = s.get("nodes", [])
                ids = {n["id"] for n in nodes if "id" in n}
                for idx, n in enumerate(nodes):
                    nx = n.get("next")
                    if not nx or nx in ids:
                        continue
                    tgt = id2seg.get(nx)
                    if tgt is None:
                        # points at a terminal-choice id or nothing addressable
                        continue
                    # (A) jump-out from the middle of a segment: split after it
                    if idx < len(nodes) - 1:
                        tail = nodes[idx + 1:]
                        tname = uniq(nm + "_cont")
                        files[tname] = {**{k: v for k, v in s.items()
                                           if k != "nodes"}, "nodes": tail}
                        for tk in _TERMS:
                            s.pop(tk, None)
                        s["nodes"] = nodes[:idx + 1]
                        acted = True
                        break
                    # (B) target sits mid-segment: split it so nx is an entry
                    tn = files[tgt]["nodes"]
                    if not (tn and tn[0].get("id") == nx):
                        pos = next((i for i, x in enumerate(tn)
                                    if x.get("id") == nx), None)
                        if pos is None or pos == 0:
                            continue
                        tname = uniq(tgt + "_cont")
                        files[tname] = {**{k: v for k, v in files[tgt].items()
                                           if k != "nodes"}, "nodes": tn[pos:]}
                        for tk in _TERMS:
                            files[tgt].pop(tk, None)
                        files[tgt]["nodes"] = tn[:pos]
                        files[tgt]["next"] = tname + ".json"
                        acted = True
                        break
                    # (C) last node, target is a clean entry: make it a terminal
                    cur = [k for k in _TERMS if k in s]
                    if not cur or cur == ["end"]:
                        s.pop("end", None)
                        s["next"] = tgt + ".json"
                    # else an explicit terminal already routes; the stray next
                    # is redundant -- just drop it
                    n.pop("next", None)
                    acted = True
                    break
                if acted:
                    break
            if not acted:
                break
        # safety: any segment with no terminal and no live next ends here
        for s in files.values():
            if not any(k in s for k in _TERMS):
                s["end"] = True

    _relink_segments()

    # Repair dead-end blocks. A segment that ends in `end` while its final
    # bytecode instruction is an unconditional jump (e.g. a gated title card that
    # skips an alternate branch to land on a shared continuation) must continue to
    # that jump's target, not stop. This works at the byte level (the jump map),
    # independent of the node-id rewrites the choice/gate nesting performs.
    mj_by_scene = {str(sc["scene"]): (sc.get("_merge_jumps") or {})
                   for sc in doc["scenes"]}

    def _seg_offsets(b):
        return [int(n["offset"]) for n in b["nodes"]
                if isinstance(n, dict) and n.get("offset") is not None]

    for _ in range(5000):
        acted = False
        for nm in list(files.keys()):
            body = files[nm]
            if not body.get("end"):
                continue
            scn = str(body.get("scene"))
            mj = mj_by_scene.get(scn) or {}
            offs = _seg_offsets(body)
            if not (offs and mj):
                continue
            last_off = max(offs)
            cand = [(int(j), j) for j in mj if last_off < int(j) <= last_off + 8]
            if not cand:
                continue                      # no trailing unconditional jump
            cand.sort()
            target = int(mj[cand[0][1]])
            best = None                       # the node at/after target in-scene
            for onm, ob in files.items():
                if str(ob.get("scene")) != scn:
                    continue
                for pos, n in enumerate(ob["nodes"]):
                    o = n.get("offset")
                    if o is not None and int(o) >= target and "id" in n:
                        if best is None or int(o) < best[0]:
                            best = (int(o), onm, pos)
            if best is None or best[1] == nm:
                continue
            _, tnm, pos = best
            tb = files[tnm]
            if pos > 0:                       # split so the merge is an entry
                newnm = tnm + "_m"
                k = 2
                while newnm in files:
                    newnm = "%s_m%d" % (tnm, k); k += 1
                files[newnm] = {**{kk: vv for kk, vv in tb.items() if kk != "nodes"},
                                "nodes": tb["nodes"][pos:]}
                for tk in _TERMS:
                    tb.pop(tk, None)
                tb["nodes"] = tb["nodes"][:pos]
                tb["next"] = newnm + ".json"
                tgt = newnm
            else:
                tgt = tnm
            body.pop("end", None)
            body["next"] = tgt + ".json"
            acted = True
            break
        if not acted:
            break

    # Forward-skip repair for segments that were linearly linked but carry a
    # trailing UNCONDITIONAL jump skipping over the segment they fell through to.
    # The gate/merge repair above only reconsiders segments left dangling (`end`);
    # a segment that already got a linear `next` is not revisited. But a title-only
    # arm of a gate can end with a `28` jump that skips a sibling arm's content --
    # e.g. As Time Goes By's sophomore chapter: the "high school loser" title
    # (var2000==2 arm) has `28 -> @14199`, jumping PAST the "I broke up with Sophie"
    # narration (which belongs only to the other, var2000!=2 arm) straight to the
    # shared "Sophomore year ended up miserable" beat. Without honoring the jump,
    # the win arm leaks the losing arm's narration.
    #
    # This is deliberately narrow to avoid rerouting legitimate linear links: it
    # fires only when the jump target is (a) in the SAME scene, (b) strictly
    # forward of where the current linear `next` segment starts (so it genuinely
    # skips that segment), and (c) resolves to a real node boundary. Cross-scene
    # jumps and option/goto targets (which land before or outside the linear next)
    # are left untouched.
    def _seg_span(b):
        os_ = _seg_offsets(b)
        return (min(os_), max(os_)) if os_ else (None, None)

    for nm in list(files.keys()):
        body = files[nm]
        _nxt = (body.get("next") or "").replace(".json", "")
        if not _nxt or _nxt not in files:
            continue
        scn = str(body.get("scene"))
        mj = mj_by_scene.get(scn) or {}
        offs = _seg_offsets(body)
        if not (offs and mj):
            continue
        last_off = max(offs)
        cand = [(int(j), j) for j in mj if last_off < int(j) <= last_off + 8]
        if not cand:
            continue
        cand.sort()
        target = int(mj[cand[0][1]])
        # the segment the current linear next begins at
        _nlo, _nhi = _seg_span(files[_nxt])
        if _nlo is None or target <= _nlo:
            continue                       # jump does not skip past the linear next
        # locate the same-scene segment whose offset span contains the jump
        # target, and the position within it of the first node at/after the target
        best = None
        for onm, ob in files.items():
            if str(ob.get("scene")) != scn or onm == nm:
                continue
            for pos, n in enumerate(ob["nodes"]):
                o = n.get("offset")
                if o is not None and int(o) >= target:
                    if best is None or int(o) < best[0]:
                        best = (int(o), onm, pos)
                    break
        if best is None or best[1] == nm or best[1] == _nxt:
            continue
        _, tnm, pos = best
        tb = files[tnm]
        if pos > 0:
            newnm = tnm + "_m"
            k = 2
            while newnm in files:
                newnm = "%s_m%d" % (tnm, k); k += 1
            files[newnm] = {**{kk: vv for kk, vv in tb.items() if kk != "nodes"},
                            "nodes": tb["nodes"][pos:]}
            for tk in _TERMS:
                tb.pop(tk, None)
            tb["nodes"] = tb["nodes"][:pos]
            tb["next"] = newnm + ".json"
            tgt = newnm
        else:
            tgt = tnm
        # replace the segment's trailing linear `next` node with the jump target
        _nl = body.get("nodes") or []
        if _nl and isinstance(_nl[-1], dict) and _nl[-1].get("type") == "next":
            _nl[-1] = {"type": "next", "next": tgt + ".json",
                       "source": "forward_skip_jump"}
        else:
            body["next"] = tgt + ".json"

    # variable map: which files write each var, which gates read it
    variables = {}

    def scan_writes(nd, fname):
        if not isinstance(nd, dict):
            return
        if nd.get("type") in ("var_set", "var_add"):
            v = str(nd["var"])
            variables.setdefault(v, {"written_in": [], "gated_in": []})
            op = ("=" + str(nd["value"])) if nd["type"] == "var_set" \
                else ("+" + str(nd["value"]))
            variables[v]["written_in"].append(fname + ".json " + op)
        for x in nd.get("branch_dialogue") or []:
            if isinstance(x, dict) and "lines" in x:
                for y in x["lines"]:
                    scan_writes(y, fname)
            else:
                scan_writes(x, fname)
        for oc in nd.get("outcomes") or []:
            for y in oc.get("lines", []):
                scan_writes(y, fname)
        for y in nd.get("after_choice") or []:
            scan_writes(y, fname)
        for y in nd.get("then") or []:
            scan_writes(y, fname)

    for fname, body in files.items():
        for nd in body["nodes"]:
            scan_writes(nd, fname)
        g = body.get("gate")
        if g:
            v = str(g["var"])
            variables.setdefault(v, {"written_in": [], "gated_in": []})
            variables[v]["gated_in"].append(fname + ".json ==" + str(g["equals"]))
    # The variable cross-reference is fully derivable from the segments, so it is
    # not emitted into the (lean) output. We still run its one useful check here:
    # a variable that is read by a gate but never written anywhere is either an
    # engine-initialised value or a decode miss worth surfacing. "score" is the
    # engine's running minigame score (op5f/op3f), not a variable-store slot, so
    # it is expected to be gated without an explicit writer and is excluded.
    rbw = sorted(v for v, x in variables.items()
                 if x["gated_in"] and not x["written_in"] and v.isdigit())
    if rbw:
        print("[!] %s: gated but never written: %s"
              % (doc.get("episode") or doc.get("title") or "episode",
                 ", ".join(rbw)), file=sys.stderr)
    # Prune genders to characters that ACTUALLY have dialogue lines in this
    # episode. Scene 1's resource table declares the full character library
    # available to the script -- Pixelberry Blast reuses a shared roster across
    # episodes ("Computer", "Offscreen Group", "Girl", etc.), and any given
    # episode uses only some of them. Keeping the unused declarations in
    # `genders` clutters the runtime lookup with characters that never speak.
    spoken_names = set()
    for _fname, body in files.items():
        for nd in body.get("nodes", []):
            if isinstance(nd, dict) and nd.get("type") == "dialogue":
                s = nd.get("speaker")
                if s:
                    spoken_names.add(s)
    genders_all = doc.get("genders") or {}
    genders_used = {n: g for n, g in genders_all.items() if n in spoken_names}
    index = {"title": doc.get("title"), "episode": doc.get("episode"),
             "cast": doc.get("cast"), "main_characters": doc.get("main_characters"),
             "genders": genders_used,
             "name_vars": doc.get("name_vars"),
             "entry": scene_entry.get("1"),
             **({"variable_defaults": doc["variable_defaults"]}
                if doc.get("variable_defaults") else {}),
             "scene_entries": scene_entry,
             "scene_minigame_banks": scene_banks,
             "section_dispatch": {str(sc["scene"]): sc["section_dispatch"]
                                  for sc in doc["scenes"]
                                  if "section_dispatch" in sc},
             "minigame_results": {str(sc["scene"]): sc["minigame_results"]
                                  for sc in doc["scenes"]
                                  if "minigame_results" in sc},
             "minigame_scoring": {str(sc["scene"]): sc["minigame_scoring"]
                                  for sc in doc["scenes"]
                                  if "minigame_scoring" in sc},
             "files": sorted(f + ".json" for f in files),
             "note": ("Each file is one linear run of story nodes; walk them in "
                      "order. The final node is a control node naming what comes "
                      "next: `choice` (options name the next file(s), `after` the "
                      "post-choice file), `next` (continue to that file), `gate` "
                      "(play `then` when the test holds else `else`; the test is "
                      "var==equals, or var<equals when the gate has op=='lt'), "
                      "`goto_scene` "
                      "(jump to another scene, with the section register value it "
                      "dispatches on), `minigame` (win/lose file by score), or "
                      "`end`.")}

    # Maps for resolving a section-dispatch jump to the section it actually
    # selects (not just the destination scene's first segment). A goto that writes
    # the section register lands at the scene's dispatcher, which routes by value;
    # the value->offset binding (observed, from the overlay) tells us where.
    scene_of_seg = {name: body.get("scene") for name, body in files.items()}
    seg_ranges = {}
    for name, body in files.items():
        offs = [int(n["offset"]) for n in body["nodes"]
                if isinstance(n, dict) and n.get("offset") is not None]
        # A segment whose only content is a control TERMINAL (a gate/minigame that
        # hasn't yet been folded into `nodes`) has no node offsets here, so it
        # would get no byte range and be invisible to seg_for_section -- meaning a
        # section-dispatch arm that should route to this gate lands on the next
        # segment instead, skipping the gate (e.g. Halloween scene 4's var2002
        # gate entry, whose else-arm is the twins-reunion scene). Fold the
        # terminal's own offset into the range so the gate segment is routable.
        for _tk in ("gate", "minigame"):
            _t = body.get(_tk)
            if isinstance(_t, dict) and _t.get("offset") is not None \
                    and str(_t["offset"]).lstrip("-").isdigit():
                offs.append(int(_t["offset"]))
        if offs:
            seg_ranges[name] = (min(offs), max(offs))
    bindings_by_scene = {}
    for sc in doc["scenes"]:
        sd = sc.get("section_dispatch") or {}
        for b in sd.get("bindings", []):
            if b.get("section_offset") is not None:
                bindings_by_scene.setdefault(str(sc["scene"]), {})[
                    str(b["value"])] = int(b["section_offset"])

    def seg_for_section(scene_num, section_offset):
        """The segment of `scene_num` that the section offset lands in. A
        dispatcher jump targets a section's start, but the section's first
        *decoded* node can sit a few bytes past that target, leaving the binding
        in a gap between segments. So when no segment contains the offset, pick
        the one whose start is nearest to it (which is the section being entered,
        not the preceding block)."""
        best, best_d = None, None
        for name, (lo, hi) in seg_ranges.items():
            if str(scene_of_seg.get(name)) != str(scene_num):
                continue
            if lo <= section_offset <= hi:
                return name
            d = abs(lo - section_offset)
            if best_d is None or d < best_d:
                best, best_d = name, d
        return best

    def seg_for_section_head(scene_num, section_offset):
        """Like seg_for_section, but for entering a section at its TRUE HEAD.

        When the dispatcher target falls in a gap just after a segment whose end
        is within a few bytes of the target, and that segment flows by a plain
        `next` into the segment that starts just after the target, the section
        actually BEGINS in the earlier segment (the target landed on a marker a
        few bytes past the section's opening content, which was already grouped
        into the preceding segment). Prefer that earlier segment. Byte-derived;
        only fires across an <=8-byte gap with a confirmed `next` link -- e.g.
        Making Some Dough's day-6 Andy encounter, whose val26 target @12230 sits
        3 bytes past s1_01's end and s1_01 -> s1_02, so the head is s1_01.

        Used ONLY for scene-entry / loop-back dispatch arms, never for choice
        options (which resolve with the plain nearest-start seg_for_section)."""
        nearest = seg_for_section(scene_num, section_offset)
        for name, (lo, hi) in seg_ranges.items():
            if str(scene_of_seg.get(name)) != str(scene_num):
                continue
            if lo <= section_offset <= hi:
                return name                    # a real container wins outright
        if nearest is not None:
            # Never pull back from a nearest-start segment that is itself a real
            # section head -- one that opens with a title_card or its own
            # scene-setup (background/music) followed by content. Such a segment
            # is a legitimate chapter start, so the dispatcher genuinely enters
            # there; the earlier segment is a different (preceding) section. This
            # stops the head-preference from hijacking e.g. ATGB value-7, whose
            # target sits just after s3_04_lose but whose real head is s3_05 (a
            # title-carded chapter "By junior year, Alison and I...").
            nbody = files.get(nearest) or {}
            nnds = nbody.get("nodes") or []
            if any(isinstance(x, dict) and x.get("type") == "title_card"
                   for x in nnds[:2]):
                return nearest
            for name, (lo, hi) in seg_ranges.items():
                if str(scene_of_seg.get(name)) != str(scene_num):
                    continue
                if 0 <= section_offset - hi <= 8:
                    body = files.get(name) or {}
                    term = body.get("next")
                    if term is None:
                        nds = body.get("nodes") or []
                        if nds and isinstance(nds[-1], dict) \
                                and nds[-1].get("type") == "next":
                            term = nds[-1].get("next")
                    if isinstance(term, str) \
                            and term.replace(".json", "") == nearest:
                        return name
        return nearest

    # Fold every terminal into the node stream so a segment is just {nodes:[...]}
    # whose last node is a typed control node. Drop per-segment scene/title (the
    # scene is in the file name; the title was identical across a scene's files).
    # Done after relink (which relies on top-level terminals) and after bank
    # resolution (which reads body["scene"]).
    for body in files.values():
        if "choice" in body:
            body["nodes"].append({"type": "choice", **body.pop("choice")})
        if "next" in body:
            body["nodes"].append({"type": "next", "next": body.pop("next")})
        if "gate" in body:
            body["nodes"].append({"type": "gate", **body.pop("gate")})
        if "goto_scene" in body:
            term = body.pop("goto_scene")
            nodes = body["nodes"]
            # the raw goto node is already the last node in the segment (it carries
            # the id that incoming `next` edges target); enrich it in place with the
            # terminal's scene/section_register/file rather than appending a second
            # goto node that would shadow it.
            if (nodes and isinstance(nodes[-1], dict)
                    and nodes[-1].get("type") == "goto_scene"):
                nodes[-1].update(term)
            else:
                nodes.append({"type": "goto_scene", **term})
        if "minigame" in body:
            body["nodes"].append({"type": "minigame", **body.pop("minigame")})
        if body.pop("end", None):
            body["nodes"].append({"type": "end"})
        body.pop("scene", None)
        body.pop("title", None)

    # Drop the vestigial node graph. Every node carried an `id` (the old
    # scene.offset naming) and most carried a node-level `next` that just points
    # at the physically next node -- redundant, because a segment is walked top to
    # bottom and its final node is the control terminal. Remove those redundant
    # `next` links, then remove every `id` that nothing references, leaving only
    # the rare genuine intra-segment jump (and the one id it targets) behind.
    import re as _re
    _INTERNAL_ID = _re.compile(r"^\d+\.\d+$")
    for body in files.values():
        nodes = [n for n in body["nodes"] if isinstance(n, dict)]
        pos = {n.get("id"): i for i, n in enumerate(nodes) if n.get("id")}
        referenced = set()
        for i, n in enumerate(nodes):
            if n.get("type") == "next":
                continue                       # terminal next -> a segment file
            nx = n.get("next")
            if nx is None:
                continue
            if nx in pos:
                if pos[nx] == i + 1:
                    n.pop("next", None)        # redundant: just the next node
                else:
                    referenced.add(nx)         # a real jump within the segment
            elif _INTERNAL_ID.match(str(nx)):
                # An internal node-id `next` (X.Y) that no longer resolves in
                # this segment: the target node was pulled into a *different*
                # segment by an earlier gate/choice split (e.g. a gate that
                # became its own segment, so the dialogue before it kept a
                # stale forward-pointer to the gate's old id). A segment is
                # walked top-to-bottom and the following node (here, the gate
                # that actually routes) runs next regardless, so this pointer
                # is redundant -- drop it rather than leave a dangling ref.
                n.pop("next", None)
        for n in nodes:
            if n.get("id") is not None and n["id"] not in referenced:
                n.pop("id", None)

    # Resolve and clean every goto_scene node. A goto names its destination by
    # `file` (the segment naming, matching `next`/`choice`/`gate`). The scene and
    # chunk indices it used to carry (scene, script, id, to_scene, to_node) are the
    # old bytecode-relative references and are dropped. `section_register` (the
    # dispatch value the destination scene reads) and, for the conditional day-hub
    # loop-backs, `when`/`loop_back`/`note` are kept; `offset` stays as provenance.
    for body in files.values():
        for idx, nd in enumerate(body["nodes"]):
            if not (isinstance(nd, dict) and nd.get("type") == "goto_scene"):
                continue
            scn = nd.get("scene")
            if scn is None and nd.get("to_scene") is not None:
                scn = nd["to_scene"]
            file_ref = nd.get("file") or (scene_entry.get(str(scn))
                                          if scn is not None else None)
            # if the jump writes the section register and the destination scene has
            # an observed value->offset binding, route to that *section's* segment
            # rather than the scene's first segment.
            sr = nd.get("section_register")
            if sr is not None and scn is not None:
                bmap = bindings_by_scene.get(str(scn))
                val = str(sr.get("value")) if isinstance(sr, dict) else None
                if bmap and val in bmap:
                    # Dispatch/loop-back arm: enter the section at its TRUE head
                    # (the head-preferring resolver rescues the case where the
                    # target landed a few bytes past the section's opening
                    # content, e.g. the day-6 Andy encounter -> s1_01 not s1_02).
                    seg = seg_for_section_head(scn, bmap[val])
                    if seg:
                        file_ref = seg + ".json"
            clean = {"type": "goto_scene", "file": file_ref}
            for k in ("section_register", "when", "loop_back", "note", "offset"):
                if nd.get(k) is not None:
                    clean[k] = nd[k]
            body["nodes"][idx] = clean

    # _far_offset resolution moved below my pre-pass splits, so seg_for_section
    # sees the post-split segments (each refuse-arm in its own segment).

    # A section whose terminating SEP jumps to another section body (the breakup
    # section -> the loser arc): resolve that body offset to its segment so the
    # section continues there instead of dead-ending. Byte-derived from the SEP
    # operand; no overlay.
    # If the loser section was split by a var-gate, its first decoded node sits
    # in the gate's THEN branch; enter at the GATE instead so the jump routes
    # through the var condition (the breakup, not setting var2000, falls to the
    # else branch -- the "let myself go after I broke up with Sophie" arc).
    gate_parent = {}
    for gname, gbody in files.items():
        for gnd in gbody["nodes"]:
            if isinstance(gnd, dict) and gnd.get("type") == "gate":
                for br in (gnd.get("then"), gnd.get("else")):
                    if isinstance(br, str):
                        gate_parent[br.replace(".json", "")] = gname
    # A section-jump SEP that lands on a var-gate entered it WITHOUT passing
    # through the dispatch that sets the gate variable (the dispatch entry sets
    # e.g. var2000 and falls to the `then` arm; the cross-jump did not). So a
    # cross-jump takes the gate's else / fall-through arm. The Sophie breakup
    # jumps into the loser section's var2000 gate and so plays the else arm
    # ("I kind of let myself go after I broke up with Sophie...") -- it never
    # shows the dispatch-loser `then` title card ("I was a high school loser").
    gate_live = {}
    for gname, gbody in files.items():
        for gnd in gbody["nodes"]:
            if isinstance(gnd, dict) and gnd.get("type") == "gate":
                live = gnd.get("else")
                if isinstance(live, str):
                    live = live.replace(".json", "")
                    then = gnd.get("then")
                    then = then.replace(".json", "") if isinstance(then, str) else then
                    gate_live[gname] = live          # entering the gate -> else arm
                    if then:
                        gate_live[then] = live       # its then-branch -> else arm
    for nm, body in files.items():
        tgt = body.pop("_section_goto_far", None)
        if tgt is None:
            continue
        scn = scene_of_seg.get(nm)
        seg = seg_for_section(scn, int(tgt)) if scn is not None else None
        if seg:
            seg = gate_live.get(seg) or gate_parent.get(seg, seg)
        nodes = body["nodes"]
        while nodes and isinstance(nodes[-1], dict) \
                and nodes[-1].get("type") in ("end", "next"):
            nodes.pop()
        body.pop("next", None)
        if seg and seg != nm:
            nodes.append({"type": "next", "next": seg + ".json"})
        else:
            nodes.append({"type": "end"})

    # Degenerate-sentinel terminators (see _resolve_degenerate_exits): a section
    # body that dead-ends at a `SEP op=<sentinel> ; 28 A ; 28 B` (with A == B, or
    # with a live but dead-code 28 pair) leaves the chapter at the convergence
    # rather than at its no-op jump target -- byte-derived, no overlay.
    def _max_off(body):
        m = None
        for nd in body.get("nodes", []):
            o = nd.get("offset") if isinstance(nd, dict) else None
            if o is not None:
                try:
                    o = int(o)
                except (TypeError, ValueError):
                    continue
                if m is None or o > m:
                    m = o
        return m

    # PRE-PASS A: split segments at any intermediate sentinel-exit terminator
    # (a `SEP op=<sentinel> ; 28 ; 28` that lies INSIDE a segment, strictly
    # between two of its nodes). The bytecode branches to convergence at every
    # such SEP, so a single decoded segment that physically straddles one is
    # really two distinct entry-arms sharing the same chapter exit. Carving at
    # each interior terminator gives each refuse-arm its own segment, and the
    # existing _far_offset / after_choice resolvers then pick the right one
    # for each option. ATGB junior has one such interior exit at @13628: the
    # Mr.Hotpants "Fine, have it your way" speech terminates there (sentinel ->
    # convergence), and the Travis "won't even look at my new clothes" beat-down
    # starts after it (a separate entry from the s3_09_else refuse-option).
    for sc in doc["scenes"]:
        cf = sc.get("control_flow") or {}
        dx = cf.get("degenerate_exits") or {}
        if not dx:
            continue
        scn = sc.get("scene")
        for term_off in sorted(int(t) for t in dx.keys()):
            target = None
            for name in list(files.keys()):
                if str(scene_of_seg.get(name)) != str(scn):
                    continue
                offs = []
                for n in files[name].get("nodes", []):
                    if isinstance(n, dict) and n.get("offset") is not None \
                            and str(n["offset"]).lstrip("-").isdigit():
                        offs.append(int(n["offset"]))
                if offs and min(offs) < term_off < max(offs):
                    target = name
                    break
            if target is None:
                continue
            nodes = files[target]["nodes"]
            # split at the first node whose offset is strictly past term_off
            split_at = None
            for i, n in enumerate(nodes):
                if not (isinstance(n, dict) and n.get("offset") is not None
                        and str(n["offset"]).lstrip("-").isdigit()):
                    continue
                if int(n["offset"]) > term_off:
                    split_at = i
                    break
            if split_at is None or split_at == 0 or split_at == len(nodes):
                continue
            base = target + "_alt"
            new_name = base
            suf = 2
            while new_name in files:
                new_name = "%s%d" % (base, suf); suf += 1
            post = nodes[split_at:]
            pre = nodes[:split_at]
            # Pre-half terminates at the sentinel exit; mark as `end` so the
            # consuming pass below rewrites it to `next: <convergence>`.
            pre.append({"type": "end"})
            files[target]["nodes"] = pre
            files[new_name] = {
                "file": new_name + ".json",
                "nodes": post,
            }
            scene_of_seg[new_name] = scn
            pre_offs = [int(n["offset"]) for n in pre
                        if isinstance(n, dict) and n.get("offset") is not None
                        and str(n["offset"]).lstrip("-").isdigit()]
            post_offs = [int(n["offset"]) for n in post
                         if isinstance(n, dict) and n.get("offset") is not None
                         and str(n["offset"]).lstrip("-").isdigit()]
            if pre_offs:
                seg_ranges[target] = (min(pre_offs), max(pre_offs))
            if post_offs:
                seg_ranges[new_name] = (min(post_offs), max(post_offs))

    # PRE-PASS B: split the convergence segment at its section-marker boundary.
    # The parser's "after_choice" emit groups a choice's shared continuation into
    # one segment, which can swallow both pre-convergence content (e.g. the
    # rejection arm of a Mr.Hotpants-style ending) AND the actual convergence
    # body (POV reset + var write + 1f0a goto). Splitting at the convergence
    # marker lets the sentinel-exit terminators route to the chapter exit without
    # replaying any pre-convergence content, while branches that legitimately fell
    # through to the rejection still play it (since their `next` points at the
    # ORIGINAL segment name, which now ends with `next: <new convergence>`).
    for sc in doc["scenes"]:
        cf = sc.get("control_flow") or {}
        dx = cf.get("degenerate_exits") or {}
        if not dx:
            continue
        scn = sc.get("scene")
        # All entries in dx share the same convergence offset (the chapter's
        # final section marker).
        conv_offs = {int(v) for v in dx.values()}
        if len(conv_offs) != 1:
            continue
        conv_off = next(iter(conv_offs))
        conv_name = seg_for_section(scn, conv_off)
        if not conv_name:
            continue
        body = files.get(conv_name)
        if not body:
            continue
        nodes = body.get("nodes", [])
        # Find the split point: first node whose offset >= conv_off. If all
        # nodes are >= or all are <, the segment already aligns with the
        # convergence -- nothing to split.
        split_at = None
        for i, nd in enumerate(nodes):
            if isinstance(nd, dict):
                off = nd.get("offset")
                if off is not None:
                    try:
                        if int(off) >= conv_off:
                            split_at = i
                            break
                    except (TypeError, ValueError):
                        continue
        if split_at is None or split_at == 0:
            continue
        # Generate a unique name for the post-split convergence segment. The
        # ORIGINAL segment keeps its name so that fall-through references (the
        # rejection-arm options that route via after_choice) still play the
        # pre-convergence content.
        base = conv_name + "_converge"
        new_name = base
        suf = 2
        while new_name in files:
            new_name = "%s%d" % (base, suf); suf += 1
        # Carve nodes[split_at:] into a new segment file. Strip any trailing
        # `end` first; we'll re-emit terminals in the existing fold pass below.
        post = nodes[split_at:]
        pre = nodes[:split_at]
        # Re-end the pre segment with a `next` -> new convergence segment.
        while pre and isinstance(pre[-1], dict) \
                and pre[-1].get("type") in ("end", "next"):
            pre.pop()
        pre.append({"type": "next", "next": new_name + ".json"})
        body["nodes"] = pre
        # The body's `scene` key was already popped by the terminal-fold pass;
        # look up the scene number from scene_of_seg, which is still intact.
        new_scene = scene_of_seg.get(conv_name)
        files[new_name] = {
            "file": new_name + ".json",
            "nodes": post,
        }
        scene_of_seg[new_name] = new_scene
        # Update seg_ranges so seg_for_section now routes the convergence offset
        # to the NEW segment.
        pre_offs = [int(n["offset"]) for n in pre
                    if isinstance(n, dict) and n.get("offset") is not None
                    and str(n["offset"]).lstrip("-").isdigit()]
        post_offs = [int(n["offset"]) for n in post
                     if isinstance(n, dict) and n.get("offset") is not None
                     and str(n["offset"]).lstrip("-").isdigit()]
        if pre_offs:
            seg_ranges[conv_name] = (min(pre_offs), max(pre_offs))
        if post_offs:
            seg_ranges[new_name] = (min(post_offs), max(post_offs))

    # Link choice options whose body is a non-contiguous jump into a later section
    # (carried as `_far_offset` on the option: the bytecode offset it jumps to).
    # Point the option at that section's segment (resolved like a dispatcher
    # landing) instead of the fall-through merge. This runs AFTER the segment
    # splits above so that an option whose far target lands in (say) the Travis
    # half of a split rejection segment resolves to that half rather than to
    # the combined pre-split segment that would also pull in the Mr.Hotpants
    # half. Byte-derived; no overlay.
    for nm, body in files.items():
        scn = scene_of_seg.get(nm)
        for nd in body["nodes"]:
            if not (isinstance(nd, dict) and nd.get("type") == "choice"):
                continue
            for opt in nd.get("options", []):
                tgt = opt.pop("_far_offset", None)
                if tgt is None or scn is None:
                    if tgt is not None:
                        opt.setdefault("next", None)
                    continue
                seg = seg_for_section(scn, int(tgt))
                opt["next"] = (seg + ".json") if seg else None

    for sc in doc["scenes"]:
        cf = sc.get("control_flow") or {}
        dx = cf.get("degenerate_exits") or {}
        if not dx:
            continue
        scn = sc.get("scene")
        # segments of this scene that dead-end OR end with an inferred `next`
        # (the parser's branch-falls-through-to-merge inference). Both can be the
        # source of a sentinel-exit terminator: a dead-end means we never resolved
        # any successor; a `next` means we inferred one from the merge calc, which
        # this rule supersedes when the terminator says "exit the chapter".
        candidates = []
        for name, body in files.items():
            if str(scene_of_seg.get(name)) != str(scn):
                continue
            nodes = body.get("nodes", [])
            if not (nodes and isinstance(nodes[-1], dict)
                    and nodes[-1].get("type") in ("end", "next")):
                continue
            mo = _max_off(body)
            if mo is not None:
                candidates.append((mo, name))
        for term_off, conv_off in dx.items():
            term_off = int(term_off)
            # source = the segment whose decoded content ends just before the
            # terminator (the branch that flows into the SEP;28;28).
            cands = [(mo, nm) for mo, nm in candidates if mo <= term_off]
            if not cands:
                continue
            src = max(cands)[1]
            conv = seg_for_section(scn, int(conv_off))
            if not conv or conv == src:
                continue
            conv = gate_live.get(conv) or gate_parent.get(conv, conv)
            nodes = files[src]["nodes"]
            # Rewrite the trailing terminal (end or inferred next) to point at the
            # chapter convergence. Branches that legitimately ended with a
            # bytecode-derived goto_scene have `goto_scene` (not `next`) and are
            # untouched.
            while nodes and isinstance(nodes[-1], dict) \
                    and nodes[-1].get("type") in ("end", "next"):
                nodes.pop()
            nodes.append({"type": "next", "next": conv + ".json"})

    # or quiz (the word/quiz games, from the bank), action_tap (engine-rendered
    # tap round, no script bank), prompt, feedback, or word_match.
    def _mg_variety(mg):
        su = mg.get("setup")
        if su and su.get("bank"):
            return su["bank"]
        k = mg.get("kind")
        # A round with inline correct/decoy option groups is a pick-the-word game
        # whether flat (kind=action) or carrying win/lose arms (content_fork). But
        # NOT when a real question/answer quiz bank is attached -- that is a quiz.
        opts = mg.get("options")
        if isinstance(opts, list) and len(opts) >= 2 \
                and k in ("action", "content_fork") \
                and not (mg.get("setup") or {}).get("bank") == "quiz":
            return "pick_word"
        if k == "action":
            return "action_tap"
        if k in ("prompt", "feedback"):
            return k
        if k == "word-match":
            return "word_match"
        return k or "unknown"
    for body in files.values():
        nodes = body["nodes"]
        for i, nd in enumerate(nodes):
            if isinstance(nd, dict) and nd.get("type") == "minigame":
                # An action round with inline correct/decoy option groups is its own
                # pick-the-word game; a quiz bank that got attached from elsewhere in
                # the scene is not this game's data (The Tutors' "Concentrate!" round
                # shares its scene with a vocabulary quiz). Drop that mis-attached
                # bank so the round reads as the pick_word it is.
                if (nd.get("kind") in ("action", "content_fork")
                        and isinstance(nd.get("options"), list)
                        and len(nd.get("options")) >= 2
                        and (nd.get("setup") or {}).get("bank") == "quiz"):
                    nd.pop("setup", None)
                variety = _mg_variety(nd)
                rebuilt = {"type": "minigame", "minigame_type": variety}
                rebuilt.update({k: v for k, v in nd.items()
                                if k not in ("type", "minigame_type")})
                nodes[i] = rebuilt

    # Scene-entry dispatch redirect: for a scene whose header runs a var[1001]
    # section dispatch with 2+ cases, the scene-entry segment (s<N>) should fall
    # through to the FIRST case body (rows[0], the natural landing) -- the body
    # the host VM reaches when the register holds the first valid value, i.e. the
    # common first-visit case. The flat linear-flow walk can instead carry the
    # entry's terminal `next` PAST that first body (through a nested inner
    # dispatch's fall-through) to a later mid-section node -- e.g. Making Some
    # Dough scene 2, whose entry landed on a mid-quiz question ("Who is the
    # narrator of The Great Gatsby?") instead of the quiz intro ("Kim heads down
    # to Ms. Lee's classroom"). Repoint the entry terminal at the rows[0] body
    # segment. Byte-derived: the rows[0] offset is the first SEP target / natural
    # landing already recorded in the scene's section_dispatch bindings; no
    # overlay, no guessing. This runs LAST (after every split pass) so seg_ranges
    # reflect the final, post-split segment boundaries -- the s<N>_alt body that
    # a mid-pass split carves out of the entry segment now exists as its own
    # segment and can be targeted.
    _sr = {}
    _scene_of = {}
    import re as _re2
    _scene_re = _re2.compile(r"^s(\d+)")
    for _nm, _bd in files.items():
        _os = [int(n["offset"]) for n in _bd.get("nodes", [])
               if isinstance(n, dict) and n.get("offset") is not None
               and str(n["offset"]).lstrip("-").isdigit()]
        if _os:
            _sr[_nm] = (min(_os), max(_os))
        # The scene field was already folded/popped out of the body, but every
        # segment name encodes its scene as the leading `s<N>` (s2, s2_alt,
        # s2_07_win, ...). Recover the scene number from the name.
        _m = _scene_re.match(_nm)
        _scene_of[_nm] = _m.group(1) if _m else None

    def _seg_at(scene_num, off):
        best, best_d = None, None
        for _nm, (lo, hi) in _sr.items():
            if str(_scene_of.get(_nm)) != str(scene_num):
                continue
            if lo <= off <= hi:
                return _nm
            d = abs(lo - off)
            if best_d is None or d < best_d:
                best, best_d = _nm, d
        return best

    for sc in doc.get("scenes", []):
        sd = sc.get("section_dispatch") or {}
        binds = sd.get("bindings", [])
        vv = sd.get("valid_values", [])
        if len(binds) < 2 or len(vv) < 2:
            continue
        scn = str(sc["scene"])
        entry_name = "s%s" % scn
        entry = files.get(entry_name)
        if entry is None:
            continue
        first_val = str(vv[0])
        row0_off = next((int(b["section_offset"]) for b in binds
                         if str(b.get("value")) == first_val
                         and b.get("section_offset") is not None), None)
        if row0_off is None:
            continue
        row0_seg = _seg_at(scn, row0_off)
        # When the row0 offset sits just past the entry's own content in a
        # control-flow-only gap (a dispatch/selector with no emitted content
        # nodes, e.g. scene 3's random-ad selector between the entry @12514 and
        # its fall-through s3_alt @12725), nearest-start resolves to the entry
        # itself. The true row0 body is the selector's fall-through: the nearest
        # FOLLOWING segment that isn't the entry. Recover it so the entry routes
        # to the section default instead of a hardwired later ad.
        if row0_seg == entry_name and row0_off > _sr.get(entry_name, (0, 0))[1]:
            following = None
            foll_lo = None
            for nm, (lo, hi) in _sr.items():
                if str(_scene_of.get(nm)) != scn or nm == entry_name:
                    continue
                if lo >= row0_off and (foll_lo is None or lo < foll_lo):
                    following, foll_lo = nm, lo
            if following is not None:
                row0_seg = following
        if not row0_seg or row0_seg == entry_name:
            continue
        nds = entry.get("nodes") or []
        if not nds or not isinstance(nds[-1], dict):
            continue
        last = nds[-1]
        if last.get("type") != "next":
            continue                       # only repoint a plain fall-through
        cur = str(last.get("next") or "").replace(".json", "")
        if cur == row0_seg:
            continue                       # already correct
        # Guard: the rows[0] body must sit at or after the entry's own offsets
        # (a forward fall-through), and strictly before the current mis-target,
        # so we only ever pull the entry BACK to the section it skipped -- never
        # push it forward past legitimate content.
        entry_max = _sr.get(entry_name, (0, 0))[1]
        row0_lo = _sr.get(row0_seg, (None, None))[0]
        cur_lo = _sr.get(cur, (None, None))[0]
        if row0_lo is None or row0_lo < entry_max:
            continue
        if cur_lo is not None and row0_lo >= cur_lo:
            continue                       # current target already precedes rows[0]
        last["next"] = row0_seg + ".json"

    # Byte-adjacency correction for dispatch-body fall-through. When the scene
    # entry was repointed to a first-case body (row0_seg above), that body may
    # itself carry a spurious terminal `next` that JUMPS past the byte-adjacent
    # continuation to a distant segment -- an artifact of dispatch-landing links
    # overriding the natural linear fall-through. Concretely: Making Some Dough's
    # quiz intro (s2_alt, bytes 4830-4880) falls straight through in the bytecode
    # to the first question (s2_01 @4889, only 9 bytes later, no intervening
    # jump), yet its `next` pointed at s2_07 (@5391), orphaning the whole first
    # question section. Detect this: if a segment's terminal `next` targets a
    # same-scene segment that is NOT the byte-adjacent successor, and a strictly
    # nearer same-scene segment begins within a small window just past this
    # segment's end, repoint to that adjacent segment. Byte-derived (adjacency),
    # conservative (only when the current target is strictly farther than the
    # adjacent one), and scoped to segments the entry redirect just touched.
    def _byte_adjacent_successor(seg_name):
        lo_hi = _sr.get(seg_name)
        if lo_hi is None:
            return None
        end = lo_hi[1]
        scn_of = _scene_of.get(seg_name)
        best, best_lo = None, None
        for nm, (lo, hi) in _sr.items():
            if nm == seg_name or str(_scene_of.get(nm)) != str(scn_of):
                continue
            if lo > end and (best_lo is None or lo < best_lo):
                best, best_lo = nm, lo
        # Require true adjacency: the next segment must start within a small gap
        # (fold padding / markers) of this one's end.
        if best is not None and best_lo - end <= 16:
            return best
        return best  # nearest following same-scene start (still the linear next)

    # Set of segments that carry NO offset-bearing node (choice-only blocks such
    # as the quiz's spelling question s2_16). These can't be positioned by byte
    # range, so a byte-adjacency repoint could silently skip them. Whenever the
    # current segment's terminal `next` targets one of these, leave it alone --
    # the offset-less segment IS the intended continuation and must not be jumped.
    _offsetless = set()
    for nm, bd in files.items():
        nds = bd.get("nodes") or []
        if nds and not any(isinstance(x, dict) and x.get("offset") is not None
                           and str(x["offset"]).lstrip("-").isdigit() for x in nds):
            _offsetless.add(nm)

    for sc in doc.get("scenes", []):
        sd = sc.get("section_dispatch") or {}
        if len(sd.get("bindings", [])) < 2:
            continue
        scn = str(sc["scene"])
        # Seed the fall-through walk from the entry chain AND from dispatch-body
        # section intros (entered on later visits via the outer var-dispatch, not
        # the entry chain). A section intro that skips its byte-adjacent first
        # block -- e.g. the quiz's Science intro s2_07 jumping past s2_08
        # (Kingdom/Phylum Q1) -- would otherwise never be corrected.
        entry = files.get("s%s" % scn)
        seeds = []
        if entry is not None and entry.get("nodes") \
                and isinstance(entry["nodes"][-1], dict) \
                and entry["nodes"][-1].get("type") == "next":
            seeds.append(str(entry["nodes"][-1]["next"]).replace(".json", ""))
        for b in sd.get("bindings", []):
            off = b.get("section_offset")
            if off is None:
                continue
            nm = _seg_at(scn, int(off))
            if nm:
                seeds.append(nm)
        visited = set()
        for seed in seeds:
            cur_seg = seed
            while cur_seg and cur_seg in files and cur_seg not in visited:
                visited.add(cur_seg)
                nds = files[cur_seg].get("nodes") or []
                if not nds or not isinstance(nds[-1], dict) \
                        or nds[-1].get("type") != "next":
                    break
                tgt = str(nds[-1]["next"]).replace(".json", "")
                # NEVER repoint away from an offset-less segment: it can't be
                # byte-positioned, so the current `next` is the only link to it
                # and dropping that link would orphan it (e.g. skipping the
                # quiz's spelling question s2_16 between s2_15 and s2_17_after).
                if tgt in _offsetless:
                    cur_seg = tgt
                    continue
                adj = _byte_adjacent_successor(cur_seg)
                if adj and adj != tgt and adj not in _offsetless:
                    tgt_lo = _sr.get(tgt, (None,))[0]
                    adj_lo = _sr.get(adj, (None,))[0]
                    cur_end = _sr.get(cur_seg, (0, 0))[1]
                    # Never override an explicit unconditional forward jump. If
                    # this segment ends with a `28` jump (recorded in merge_jumps)
                    # that deliberately skips the byte-adjacent successor to reach
                    # `tgt`, the adjacency is exactly what the bytecode jumps OVER
                    # -- repointing to it would re-introduce the skipped content
                    # (e.g. As Time Goes By's "high school loser" arm jumping past
                    # the "broke up with Sophie" narration). Detect the jump by the
                    # segment's own trailing `28` target in this scene's
                    # merge_jumps and, when it lands at/after the current target,
                    # leave the link alone.
                    _mjs = mj_by_scene.get(str(scn)) or {}
                    _cur_end = _sr.get(cur_seg, (None, None))[1]
                    _has_skip_jump = False
                    if _cur_end is not None:
                        for _jk, _jv in _mjs.items():
                            if _cur_end < int(_jk) <= _cur_end + 8 \
                                    and adj_lo is not None \
                                    and int(_jv) > adj_lo:
                                _has_skip_jump = True
                                break
                    # Repoint only if the adjacent successor is strictly nearer
                    # AND sits right after this segment (a real fall-through the
                    # jump skipped), never pushing past the current target, and
                    # only across a TIGHT gap (no room for an unpositioned block
                    # to hide in the skipped span).
                    if (not _has_skip_jump
                            and adj_lo is not None and tgt_lo is not None
                            and adj_lo < tgt_lo and adj_lo >= cur_end
                            and adj_lo - cur_end <= 40):
                        nds[-1]["next"] = adj + ".json"
                        tgt = adj
                cur_seg = tgt

    # Emit `random` nodes for runtime random section selectors (e.g. Help Ads).
    # A random draw has no single correct `next`, so instead of the scene entry
    # falling through to one target, replace its terminal with a `random` node
    # carrying the full option set + exhausted fall-through. The engine performs
    # the draw (pick a not-yet-done option; when all are done go to `exhausted`).
    # Each option offset is head-resolved to the ad's own intro segment (the
    # "Kim heads to the board" block that owns that story), so the engine enters
    # each ad at its narrative start. All byte-derived from resolve_random_selector.
    for sc in doc.get("scenes", []):
        sd = sc.get("section_dispatch") or {}
        selectors = sd.get("random_selectors") or []
        if not selectors:
            continue
        scn = str(sc["scene"])
        entry = files.get("s%s" % scn)
        if entry is None:
            continue

        def _owning_intro(off):
            # The segment whose byte range starts at or before `off` and is the
            # nearest such start in this scene -- i.e. the ad-intro block the
            # target belongs to (targets land mid-story on option outcomes; the
            # owning intro is the greatest segment-start <= off).
            best, best_lo = None, None
            for nm, (lo, hi) in _sr.items():
                if str(_scene_of.get(nm)) != scn:
                    continue
                if lo <= off and (best_lo is None or lo > best_lo):
                    best, best_lo = nm, lo
            return best

        for sel in selectors:
            opts = []
            seen = set()
            for o in sel.get("options", []):
                off = o.get("section_offset")
                if off is None:
                    continue
                seg = _owning_intro(int(off))
                if seg and seg not in seen:
                    seen.add(seg)
                    opts.append({"scene": seg + ".json"})
            # The Help Ads random resolves its roll arms to option OUTCOME
            # fragments (the branch reached after each ad's own choice), not to the
            # ad's entry -- so routing the draw straight there skips the whole ad
            # (its bulletin-board intro, the helpee conversation, and the choice),
            # dropping the player onto a money payout mid-scene. The real draw is
            # over the ad ENTRIES, each of which begins with the shared "Kim heads
            # to the Help Ads bulletin board" narration and plays intro -> choice/
            # minigame -> outcome in full. When those entry segments are present,
            # use them as the option set instead of the mid-ad fragments. Scoped to
            # this scene's selector; ordered by segment byte start = play order.
            _AD_ENTRY = "help ads bulletin board"
            _AD_LISTING = ("' needs", " needs ", " seeks ", "meet me",
                           " wanted", "seeks ")
            _AD_EXHAUSTED = ("no ads anymore", "no more ads",
                             "helped just about")
            _entries = []
            for nm, (lo, hi) in _sr.items():
                if str(_scene_of.get(nm)) != scn:
                    continue
                _fnodes = files.get(nm, {}).get("nodes") or []
                _fl = None
                for _n in _fnodes:
                    if isinstance(_n, dict) and _n.get("type") in ("dialogue",
                                                                    "narration"):
                        _fl = (_n.get("text") or "")
                        break
                # a real ad entry either opens on the bulletin-board narration or
                # jumps straight into the "Let's see here... '<listing>'" pitch (the
                # decoder sometimes splits the shared bulletin-board line onto the
                # previous ad's tail), AND posts a helpee listing ("... Needs ...",
                # "... Seeks ...") in its opening beats. The shared bulletin-board
                # narration alone also heads mid-ad continuation blocks and the "no
                # ads anymore" exhausted block, which must not be offered as draws.
                _blurb = " ".join((_n.get("text") or "")
                                  for _n in _fnodes[:6]
                                  if isinstance(_n, dict)).lower()
                _fl_low = (_fl or "").lower()
                _opens_ad = (_AD_ENTRY in _fl_low
                             or _fl_low.startswith("let's see here"))
                if _opens_ad \
                        and any(k in _blurb for k in _AD_LISTING) \
                        and not any(k in _blurb for k in _AD_EXHAUSTED):
                    _entries.append((lo, nm))
            if len(_entries) >= 2:
                opts = [{"scene": nm + ".json"}
                        for _lo, nm in sorted(_entries)]
            if len(opts) < 2:
                continue

            # Attach a per-option "done" guard so the engine can skip an ad it has
            # already run instead of tracking draws itself. Each ad, on entry, sets
            # a per-ad-pair flag (var2005/2006/2007) to a distinguishing value (1
            # or 2 -- two ads share each flag var). That flag write is byte-derived
            # from the ad's own body, so an ad is "already done" exactly when its
            # flag var equals its value. We read the flag write straight from each
            # option segment's nodes; when an option carries no explicit flag write
            # (the decoder didn't attach one for it), we fall back to the first
            # such flag found by walking the scene's flag writes in ad order. The
            # global counter (state_var, var2003) still drives the exhausted
            # transition; the per-option guard only tells the engine which ads to
            # exclude from the draw.
            _FLAG_VARS = ("2005", "2006", "2007")
            # The per-ad done-flag write sits at the top of each ad's body in the
            # bytecode, but the decoder only surfaced it inline for some ads. For
            # the rest, recover it from the raw flag-write scan: each write is a
            # `read var200X ; push V ; ADD ; write` immediately preceding an ad's
            # listing ("... Needs ...", "... Seeks ..."). Match each write to the
            # option whose listing follows it, and inject the flag as a var_set at
            # the head of that ad so every ad carries its own done-flag.
            _ad_flags = sel.get("_resolved_ad_flags")
            if _ad_flags is None:
                _ad_flags = {}
                _fw = sd.get("ad_flag_writes") or []
                _ordered_opts = [(_o["scene"][:-5]
                                  if _o["scene"].endswith(".json")
                                  else _o["scene"]) for _o in opts]
                # opts are already in byte (play) order; flag writes are too
                _fw_pairs = [(w["var"], w["value"]) for w in _fw
                             if str(w["var"]) in _FLAG_VARS]
                for _osid, _pair in zip(_ordered_opts, _fw_pairs):
                    _ad_flags[_osid] = _pair
                sel["_resolved_ad_flags"] = _ad_flags
            for _opt in opts:
                _osid = _opt["scene"][:-5] if _opt["scene"].endswith(".json") \
                    else _opt["scene"]
                _flag = None
                for _n in files.get(_osid, {}).get("nodes") or []:
                    if isinstance(_n, dict) \
                            and _n.get("type") in ("var_add", "var_set") \
                            and str(_n.get("var")) in _FLAG_VARS:
                        _flag = (str(_n["var"]), int(_n["value"]))
                        break
                if _flag is None and _osid in _ad_flags:
                    _fv, _fval = _ad_flags[_osid]
                    _flag = (str(_fv), int(_fval))
                    # inject the recovered flag at the head of the ad body. The
                    # flag is ACCUMULATED additively (var += value), not assigned:
                    # two ads share one flag var and contribute distinct bit values
                    # (1 and 2), so the variable is a bitmask of which ads are done.
                    _onodes = files.get(_osid, {}).get("nodes") or []
                    files[_osid]["nodes"] = [{"type": "var_add", "var": str(_fv),
                                              "value": int(_fval),
                                              "source": "ad_done_flag"}] + _onodes
                if _flag is not None:
                    # Filter this ad once its bit is set in the shared flag. Two ads
                    # share each flag var, contributing bit values 1 and 2, and the
                    # ad bodies ADD their value on entry -- so the var is a bitmask
                    # (0=none, 1=first done, 2=second done, 3=both done). An ad is
                    # done exactly when its bit is present: (flag & value) != 0.
                    # Equality would let an ad repeat (running the value-2 ad leaves
                    # the var != 1, so the value-1 ad looks undone); a >= test would
                    # skip an ad (running the value-2 ad first makes the value-1 ad
                    # look done). The bitmask test is order-independent and lets
                    # every ad play exactly once.
                    _opt["done_when"] = {"var": _flag[0], "op": "bitmask",
                                         "equals": str(_flag[1])}

            exhausted_seg = None
            if sel.get("exhausted_offset") is not None:
                exhausted_seg = _seg_at(scn, int(sel["exhausted_offset"]))
            node = {"type": "random",
                    "state_var": sel.get("state_var"),
                    "options": opts,
                    "note": ("Runtime random selector: pick one option that is not "
                             "yet done (per its `done_when` guard), enter its scene, "
                             "and mark it done. When no option remains -- or, if "
                             "present, after max_count draws -- go to `exhausted`. "
                             "Draw and done-tracking are the engine's job.")}
            # Optional hard cap on draws. Every option here is meant to play once,
            # so the pool empties on its own after len(options) draws via the
            # `done_when` guards; max_count is a redundant safety stop equal to the
            # number of options, emitted ONLY when every option is guarded (so the
            # pool provably empties). If any option is unguarded (repeatable), no
            # cap is emitted -- capping could stop such a selector too early.
            if opts and all(o.get("done_when") for o in opts):
                node["max_count"] = str(len(opts))
            if exhausted_seg:
                node["exhausted"] = exhausted_seg + ".json"
            # Replace the entry's terminal fall-through with the random node.
            nds = entry.get("nodes") or []
            if nds and isinstance(nds[-1], dict) \
                    and nds[-1].get("type") == "next":
                nds[-1] = node
            else:
                nds.append(node)
            entry["nodes"] = nds

        # Restore the win/lose split on any ad whose brainstorm/skill minigame the
        # decoder linearized straight into its win branch, orphaning the lose
        # outcome. The writer ad (Andrew) is the case in Making Some Dough: the
        # bytecode runs the brainstorm minigame, then `push T ; 0d ; 2b -> LOSE`
        # tests the score and, on a miss, jumps to the "this really doesn't make
        # sense... I'm not going to pay" outcome; on a hit it falls through to the
        # reward (`var2000 += 35`) and the "it's perfect!" resolution. The decoder
        # kept only the fall-through, so the reward and win dialogue play
        # unconditionally and the lose block is unreachable. Here we re-insert a
        # `minigame` fork: everything up to the brainstorm hand-off stays as the
        # ad's setup, the win branch is the remainder (reward + resolution), and
        # the lose branch is the orphaned outcome segment. Byte-derived from the
        # score test; guarded to fire only when the matching lose segment exists
        # and the ad still carries its reward inline (so it is idempotent).
        _LOSE_MARKERS = ("doesn't make a lot of sense", "wasn't helpful at all",
                         "going to save my")
        _WIN_SPLIT_AFTER = "help me brainstorm"
        for nm, bd in list(files.items()):
            if str(_scene_of.get(nm)) != scn:
                continue
            _nl = bd.get("nodes") or []
            # the ad must still have its reward inline AND a brainstorm hand-off
            _has_reward = any(isinstance(x, dict) and x.get("type") == "var_add"
                              and str(x.get("var")) == "2000" for x in _nl)
            _split_at = None
            for _i, x in enumerate(_nl):
                if isinstance(x, dict) and _WIN_SPLIT_AFTER in (x.get("text")
                                                                or "").lower():
                    _split_at = _i
                    break
            if _split_at is None or not _has_reward:
                continue
            # find the orphaned lose outcome segment in this scene
            _lose = None
            for onm, obd in files.items():
                if str(_scene_of.get(onm)) != scn or onm == nm:
                    continue
                _txt = " ".join((y.get("text") or "")
                                for y in (obd.get("nodes") or [])
                                if isinstance(y, dict)).lower()
                if any(mk in _txt for mk in _LOSE_MARKERS):
                    _lose = onm
                    break
            if _lose is None:
                continue
            # split: setup (through the brainstorm hand-off + the immediately
            # following "okay" beat), then the win remainder
            _win_start = _split_at + 1
            # keep the couple of narration/dialogue beats that set up the notebook
            while _win_start < len(_nl):
                x = _nl[_win_start]
                if isinstance(x, dict) and x.get("type") == "var_add" \
                        and str(x.get("var")) == "2000":
                    break
                _win_start += 1
            if _win_start >= len(_nl):
                continue
            _setup = _nl[:_split_at + 1]
            _win_nodes = _nl[_win_start:]
            _win_seg = "%s_win" % nm
            files[_win_seg] = {"nodes": _win_nodes}
            _mg = {"type": "minigame", "minigame_type": "pick_word",
                   "kind": "outcome_fork",
                   "win": _win_seg + ".json", "lose": _lose + ".json",
                   "timed": True, "timer_ms": "20000",
                   "note": ("Brainstorm skill check (TIME-BOUND: the game draws "
                            "prompts from its pool until the timer runs out; "
                            "win_threshold correct within the time wins). On "
                            "success Andrew loves the pitch and pays (win); on "
                            "failure the pitch makes no sense and he keeps his "
                            "money (lose). Byte-derived from the post-minigame "
                            "score test.")}
            # Attach the pick-word prompt the player plays. The brainstorm is a
            # "pick the real genres" board: the correct words are the sensible
            # genre combos, the decoys the nonsense ones. The prompt data is the
            # scene's own pick_word bank (byte-derived from the head-of-scene word
            # records); match the prompt to this ad by a subtitle keyword so the
            # right board is shown. `pool` is what the timed game draws from.
            _pw = (sc.get("minigames") or {}).get("pick_word") or []
            _round = None
            for _r in _pw:
                _sub = (_r.get("subtitle") or "").lower()
                if "genre" in _sub or "brainstorm" in _sub:
                    _round = _r
                    break
            if _round is None and _pw:
                _round = _pw[0]
            if _round is not None:
                # The win bar is byte-derived: the writer ad's brainstorm gate
                # tests `push 5 ; GTE` at its 1f51 score check (need score >= 5).
                # The synthetic Andrew fork node isn't produced by
                # resolve_minigame_gates, so carry that observed threshold here.
                _mg["win_threshold"] = "5"
                _mg["setup"] = {"bank": "pick_word", "rounds": [_round],
                                "timer_ms": "20000"}
            files[nm]["nodes"] = _setup + [_mg]

        # Each Help Ad minigame plays ONE specific pick-word board, but the scene's
        # whole pick_word bank (every ad's board) can get attached to a single
        # minigame node, leaving the engine to guess which round to show. Narrow
        # each ad minigame to the one round it actually plays, matched by a subtitle
        # keyword tied to the ad's premise (the martial-arts ad plays "Dodge the
        # attacks!", the writer ad plays "Pick the real genres!"). Byte-derived: the
        # rounds and their subtitles come from the scene's word bank; this only
        # selects which of them rides on which ad node.
        _AD_ROUND_KEYS = (("dodge", ("dodge", "duck", "attack")),
                          ("genre", ("genre", "brainstorm")))
        _scene_pw = (sc.get("minigames") or {}).get("pick_word") or []
        if len(_scene_pw) >= 2:
            for _body in files.values():
                if str(_body.get("scene")) != scn:
                    continue
                for _nd in _body.get("nodes") or []:
                    if not (isinstance(_nd, dict)
                            and _nd.get("type") == "minigame"
                            and _nd.get("minigame_type") == "pick_word"):
                        continue
                    _rounds = (_nd.get("setup") or {}).get("rounds") or []
                    if len(_rounds) <= 1:
                        continue
                    # pick the round whose subtitle best fits the ad's premise
                    _ad_text = " ".join((x.get("text") or "")
                                        for x in _body.get("nodes") or []
                                        if isinstance(x, dict)).lower()
                    _chosen = None
                    for _key, _subs in _AD_ROUND_KEYS:
                        _hit_ad = any(k in _ad_text for k in _subs)
                        if not _hit_ad:
                            continue
                        for _r in _rounds:
                            if any(k in (_r.get("subtitle") or "").lower()
                                   for k in _subs):
                                _chosen = _r
                                break
                        if _chosen:
                            break
                    if _chosen is not None:
                        _nd["setup"] = {"bank": "pick_word", "rounds": [_chosen]}

    # The academic quiz's per-section question rolls (History, Science, English)
    # each draw one question of a small pool. The decoder linearizes the pool into
    # a fixed chain (question 1 -> question 2 -> question 3 -> section end), so
    # only the first is ever the "real" one and the rest chain after it. Rebuild
    # each section as a `random` node: the section's owning intro routes to a draw
    # over its question segments, and each drawn question, once answered, goes to
    # the section end (the shared "That is... correct!" feedback that closes the
    # section). Byte-derived from resolve_quiz_random; scoped to scenes that carry
    # a quiz_random pool, so no other scene is affected.
    for sc in doc.get("scenes", []):
        sd = sc.get("section_dispatch") or {}
        rolls = sd.get("quiz_random") or []
        if not rolls:
            continue
        scn = str(sc["scene"])
        _section_owners = []       # (roll_offset, owner_segment) in byte order

        def _seg_starting_at(off):
            # the question segment whose byte range begins at/just after this roll
            # arm offset (the question body head)
            cont = after = al = None
            for nm, (lo, hi) in _sr.items():
                if str(_scene_of.get(nm)) != scn:
                    continue
                if lo <= off <= hi:
                    cont = nm
                elif lo >= off and (al is None or lo < al):
                    after, al = nm, lo
            return cont or after

        for roll in rolls:
            qsegs = []
            seen = set()
            for off in roll.get("options", []):
                seg = _seg_starting_at(int(off))
                # A roll arm can land on the section's intro segment (when the
                # first question's prompt prose is merged into that intro rather
                # than a standalone question segment); in that case the real
                # question is what the intro continues to via `next`.
                if seg is not None:
                    _snodes = files.get(seg, {}).get("nodes") or []
                    _has_choice = any(isinstance(x, dict)
                                      and x.get("type") == "choice"
                                      for x in _snodes)
                    if not _has_choice and _snodes:
                        # The arm landed on an intro segment whose prose ends with
                        # the question prompt but whose choice lives in a separate
                        # offset-less segment. Prefer the `next` link if present;
                        # otherwise match the intro's final prompt line to the
                        # offset-less choice segment that poses it.
                        _last = _snodes[-1]
                        _nxt = None
                        if isinstance(_last, dict) and _last.get("type") == "next":
                            _cand = str(_last["next"]).replace(".json", "")
                            if any(isinstance(x, dict) and x.get("type") == "choice"
                                   for x in files.get(_cand, {}).get("nodes") or []):
                                _nxt = _cand
                        if _nxt is None:
                            _tail = [(_n.get("text") or "").strip().strip("'\" ").lower()
                                     for _n in _snodes if isinstance(_n, dict)
                                     and _n.get("type") in ("dialogue", "narration")]
                            _ptext = _tail[-1] if _tail else None
                            if _ptext:
                                _cands = []
                                for _nm, _bd in files.items():
                                    if str(_scene_of.get(_nm)) != scn:
                                        continue
                                    for _n in _bd.get("nodes") or []:
                                        if isinstance(_n, dict) \
                                                and _n.get("type") == "choice" \
                                                and (_n.get("prompt") or "").strip().strip("'\" ").lower() == _ptext:
                                            _cands.append(_nm)
                                            break
                                if len(_cands) == 1:
                                    _nxt = _cands[0]
                        if _nxt is not None:
                            seg = _nxt
                if seg and seg not in seen:
                    seen.add(seg)
                    qsegs.append(seg)
            if len(qsegs) < 2:
                continue
            # The section these questions belong to: the segment that currently
            # feeds the first question via `next` (the "settle down at the desk"
            # intro). Its terminal becomes the draw.
            first_q = qsegs[0]
            owner = None
            for nm, bd in files.items():
                if str(_scene_of.get(nm)) != scn:
                    continue
                nl = bd.get("nodes") or []
                if nl and isinstance(nl[-1], dict) \
                        and nl[-1].get("type") == "next" \
                        and str(nl[-1].get("next")).replace(".json", "") == first_q:
                    owner = nm
                    break
            if owner is None:
                # The intro may not yet link to the first question by `next` (the
                # offset-less choice linker runs later). Fall back to matching the
                # first question's prompt against a segment's final narration line
                # (the "'Which of these words is spelled correctly?'" prompt that
                # the intro ends on), which uniquely identifies the owning intro.
                _prompt = None
                for nd in files.get(first_q, {}).get("nodes", []) or []:
                    if isinstance(nd, dict) and nd.get("type") == "choice":
                        _prompt = (nd.get("prompt") or "").strip().strip("'\" ").lower()
                        break
                if _prompt:
                    _matches = []
                    for nm, bd in files.items():
                        if str(_scene_of.get(nm)) != scn or nm == first_q:
                            continue
                        _nl = bd.get("nodes") or []
                        _texts = [(_n.get("text") or "").strip().strip("'\" ").lower()
                                  for _n in _nl if isinstance(_n, dict)
                                  and _n.get("type") in ("dialogue", "narration")]
                        if _texts and _texts[-1] == _prompt:
                            _matches.append(nm)
                    if len(_matches) == 1:
                        owner = _matches[0]
            # The section end: the segment the LAST question in the pool routes to
            # after being answered (the shared "That is... correct!" feedback that
            # closes the section and returns to the hub). Read it from the pool's
            # last question BEFORE rewriting any `after`, since the decoder chained
            # the questions so the earlier ones' `after` points at the next
            # question, not the section end.
            section_end = None
            for qs in reversed(qsegs):
                for nd in files.get(qs, {}).get("nodes", []) or []:
                    if isinstance(nd, dict) and nd.get("type") == "choice" \
                            and nd.get("after"):
                        cand = str(nd["after"]).replace(".json", "")
                        if cand not in qsegs:      # not another pooled question
                            section_end = cand
                            break
                if section_end:
                    break
            # Point every drawn question's choice.after AND each option's `next` at
            # the section end, so a single answered question closes the section
            # instead of chaining to the next question in the pool. The decoder
            # emits per-option `next` as the byte-adjacent segment (which for a
            # question is the *next* question), so without this the engine walks
            # the whole pool one question at a time instead of playing just the
            # drawn one.
            if section_end:
                for qs in qsegs:
                    for nd in files.get(qs, {}).get("nodes", []) or []:
                        if isinstance(nd, dict) and nd.get("type") == "choice":
                            nd["after"] = section_end + ".json"
                            for _opt in nd.get("options", []) or []:
                                if isinstance(_opt, dict) and "next" in _opt:
                                    _opt["next"] = section_end + ".json"
            node = {"type": "random",
                    "options": [{"scene": qs + ".json"} for qs in qsegs],
                    "note": ("Quiz question draw: the engine picks one question "
                             "from this section's pool at random; answering it "
                             "closes the section.")}
            if owner is not None:
                # The first question segment can absorb the section's shared intro
                # (the challenge briefing -- prize, rules -- that plays once before
                # ANY question). That dialogue lives at offsets before the roll, so
                # it would only be seen when the draw happens to pick question 1.
                # Move every node before the roll offset out of the first question
                # and onto the end of the owner (before the draw), so the briefing
                # always plays. Byte-derived: the roll's own `at` offset is the
                # boundary between the shared intro and question 1's body.
                _roll_at = None
                try:
                    _roll_at = int(roll.get("at"))
                except (TypeError, ValueError):
                    _roll_at = None
                if _roll_at is not None and qsegs:
                    _fq = qsegs[0]
                    _fqn = files.get(_fq, {}).get("nodes") or []
                    _intro, _rest = [], []
                    for _nd in _fqn:
                        _off = _nd.get("offset") if isinstance(_nd, dict) else None
                        try:
                            _oi = int(_off) if _off is not None else None
                        except (TypeError, ValueError):
                            _oi = None
                        # a node with an offset strictly before the roll is intro;
                        # offset-less nodes (e.g. the split-out choice) stay with
                        # the question
                        if _oi is not None and _oi < _roll_at:
                            _intro.append(_nd)
                        else:
                            _rest.append(_nd)
                    if _intro and _rest:
                        files[_fq]["nodes"] = _rest
                        _onl = files[owner].get("nodes") or []
                        # drop the owner's trailing `next` (it will be the draw)
                        if _onl and isinstance(_onl[-1], dict) \
                                and _onl[-1].get("type") == "next":
                            _onl = _onl[:-1]
                        files[owner]["nodes"] = _onl + _intro
                onl = files[owner].get("nodes") or []
                if onl and isinstance(onl[-1], dict) \
                        and onl[-1].get("type") == "next":
                    onl[-1] = node
                else:
                    onl.append(node)
                files[owner]["nodes"] = onl
                try:
                    _section_owners.append((int(roll.get("at")), owner))
                except (TypeError, ValueError):
                    pass

        # Route the quiz scene entry to the correct SECTION by progress. The three
        # sections (History, Science, English) play one per academic day, selected
        # by the completion counter var2002 (0 -> History, 1 -> Science, 2 ->
        # English). The decoder linearized the entry straight into the first
        # section, so every visit replayed History with the same intro. Replace the
        # entry's unconditional hand-off with a `dispatch` on var2002 whose arms are
        # the section owners in byte order (which is the play order). Byte-derived:
        # the section order follows the roll offsets, and var2002 is the counter the
        # hub already tests for the "academic still available?" gate.
        _entry = files.get("s%s" % scn)
        # Prefer the byte-derived section dispatch when the scene header actually
        # reads a section register with multiple case bindings (Making Some Dough
        # scene 2: `push 1001; VREAD; op5c(11,1); op5c(14,2); ...` -- a 9-case quiz
        # selector). The roll-based var2002 selector below only recovers the 2-3
        # sections that carry a question-draw roll, orphaning the other quiz-day
        # sections (the President / sequence / Gatsby questions). Build the entry
        # dispatch from the dispatch table's own value->section_offset bindings,
        # mapping each offset to the same-scene segment that opens at it, exactly as
        # resolve_section_dispatch derived them from the SEP case targets.
        _sd_bindings = [b for b in (sd.get("bindings") or [])
                        if b.get("section_offset") is not None]
        if (_entry is not None and sd.get("register") is not None
                and len(_sd_bindings) >= 3):
            def _seg_opening_at(_off):
                # the same-scene segment whose first node offset is at/just after
                # the binding offset (snap-forward, matching the runtime's seek).
                # Segments are identified by their s<scn>_ name prefix, since the
                # `scene` field is not populated on file bodies at this stage.
                _pref = "s%s" % scn
                _best, _bd = None, None
                for _fn, _b in files.items():
                    if not (_fn == _pref or _fn.startswith(_pref + "_")):
                        continue
                    _os = [int(x["offset"]) for x in _b.get("nodes", [])
                           if isinstance(x, dict) and x.get("offset") is not None
                           and str(x["offset"]).lstrip("-").isdigit()]
                    if not _os:
                        continue
                    _lo = min(_os)
                    if _lo >= _off - 40 and (_bd is None or _lo < _bd):
                        _bd, _best = _lo, _fn
                return _best
            _arms = []
            _seen_seg = set()
            for _b in sorted(_sd_bindings, key=lambda z: z["value"]):
                _seg = _seg_opening_at(int(_b["section_offset"]))
                if _seg is None or _seg == ("s%s" % scn):
                    continue
                _arms.append({"equals": str(_b["value"]),
                              "scene": _seg + ".json"})
                _seen_seg.add(_seg)
            if len(_arms) >= 3:
                _reg = sd.get("register")
                _disp = {"type": "dispatch", "state_var": str(_reg),
                         "arms": _arms[:-1],
                         "default": _arms[-1]["scene"],
                         "note": ("Quiz-section selector read from the scene header "
                                  "(byte-derived): var%s is the section register the "
                                  "hub sets per academic day; each value routes to "
                                  "that day's quiz section (the op5c case targets in "
                                  "the dispatch table). The final case is the "
                                  "bytecode's fall-through default." % _reg)}
                _enl = _entry.get("nodes") or []
                if _enl and isinstance(_enl[-1], dict) \
                        and _enl[-1].get("type") in ("next", "goto_scene",
                                                     "dispatch"):
                    _enl[-1] = _disp
                else:
                    _enl.append(_disp)
                _entry["nodes"] = _enl
                _entry = None  # suppress the roll-based var2002 selector below
        if _entry is not None and len(_section_owners) >= 2:
            _ordered = [nm for _off, nm in sorted(_section_owners)]
            # The bytecode tests var2002 against each value up to the last section
            # and lets the final section be the ELSE case (it tests == 0, == 1,
            # ... and falls through to the last), so the last section is the
            # dispatch default rather than an explicit arm. The hub only lets the
            # player enter while var2002 < (number of sections), so the counter is
            # always in range and the default is only ever hit as that last
            # section -- matching the bytecode's two-test-plus-else shape exactly.
            _arms = [{"equals": str(_i), "scene": nm + ".json"}
                     for _i, nm in enumerate(_ordered[:-1])]
            _disp = {"type": "dispatch", "state_var": "2002", "arms": _arms,
                     "default": _ordered[-1] + ".json",
                     "note": ("Academic-challenge section selector: one section "
                              "plays per day, chosen by the completion counter "
                              "var2002 (0=History, 1=Science, else=English -- the "
                              "last section is the bytecode's fall-through case).")}
            _enl = _entry.get("nodes") or []
            if _enl and isinstance(_enl[-1], dict) \
                    and _enl[-1].get("type") in ("next", "goto_scene"):
                _enl[-1] = _disp
            else:
                _enl.append(_disp)
            _entry["nodes"] = _enl
    # otherwise skips, leaving its arms orphaned. Making Some Dough's bakery
    # (scene 4) is the case: after the baking minigame the entry reads var2004
    # (the bakery-day counter) and a 5-arm cascade routes to that day's outcome
    # (easiest->hardest), falling through to a "bakery closed" body once every
    # day is done (var2004 > 4). The decoder linearized the entry straight into
    # that closed-arm content, skipping the cascade and orphaning all 5 days.
    # Here we split the closed-arm content out of the entry and replace it with a
    # `dispatch` node whose arms are the byte-derived cascade targets and whose
    # `default` is the (now separate) closed body. All byte-derived from
    # resolve_sep_cascade_gates; guarded to fire only when the entry actually
    # absorbed the cascade's fall-through (the arms are otherwise unreachable).
    for sc in doc.get("scenes", []):
        sd = sc.get("section_dispatch") or {}
        cascades = sd.get("cascades") or []
        if not cascades:
            continue
        scn = str(sc["scene"])
        entry = files.get("s%s" % scn)
        if entry is None:
            continue
        for casc in cascades:
            arms = casc.get("cases") or []
            if len(arms) < 3:
                continue
            cas_at = int(casc["at"])
            # Only handle a cascade the entry absorbed: the cascade sits just
            # past the entry's last pre-cascade node and the entry's own nodes
            # continue PAST the cascade (into the fall-through body). Detect by a
            # gap in the entry's offsets straddling cas_at.
            ends = [int(nd["offset"]) for nd in entry.get("nodes") or []
                    if isinstance(nd, dict) and nd.get("offset") is not None
                    and str(nd["offset"]).lstrip("-").isdigit()]
            if not ends:
                continue
            before = [e for e in ends if e < cas_at]
            after = [e for e in ends if e > cas_at]
            if not before or not after:
                continue                       # entry doesn't straddle the cascade
            # Split point: entry nodes with offset > cas_at are the fall-through
            # (closed) body. Move them into a new default segment.
            split_idx = None
            for i, nd in enumerate(entry.get("nodes") or []):
                o = nd.get("offset") if isinstance(nd, dict) else None
                if o is not None and str(o).lstrip("-").isdigit() \
                        and int(o) > cas_at:
                    split_idx = i
                    break
            if split_idx is None:
                continue
            tail = entry["nodes"][split_idx:]
            # The tail is the fall-through (bakery-closed) body. Its trailing
            # gate is NOT an artifact -- it is the closed body's real terminal
            # (var2002==3 -> the "what's left to do" menu, else -> the Help Ads
            # redirect), so keep the whole tail intact.
            default_name = "s%s_bakery_closed" % scn
            default_nodes = list(tail)
            # Resolve each arm's target segment (head-resolved to the arm body).
            arm_out = []
            for a in arms:
                off = a.get("section_offset")
                if off is None:
                    continue
                seg = _seg_at(scn, int(off))
                if seg:
                    arm_out.append({"equals": str(a.get("equals")),
                                    "scene": seg + ".json"})
            if len(arm_out) < 3:
                continue
            # Commit the split: entry keeps everything before the cascade, then a
            # dispatch node; the closed body becomes its own segment.
            files[default_name] = {"nodes": default_nodes}
            _sr[default_name] = (cas_at + 1, max(after))
            _scene_of[default_name] = scn
            node = {"type": "dispatch", "state_var": casc["var"],
                    "arms": arm_out, "default": default_name + ".json",
                    "note": ("Byte-derived var cascade: state_var==equals enters "
                             "the arm scene (the bakery-day outcome, easiest to "
                             "hardest); when no arm matches (all days done) enter "
                             "`default` (the bakery-closed body). The engine "
                             "advances the counter each visit.")}
            entry["nodes"] = entry["nodes"][:split_idx] + [node]
            break                              # one cascade per scene entry

    # Wire each bakery day's intro to its bake-the-cookies minigame (0x46).
    # The bytecode places the day intro (visitor greeting) immediately before a
    # 0x46 minigame whose result forks win/lose: win advances to a new day, lose
    # repeats the same day. The generic minigame-fork splitter can't segment
    # these safely (their arms return to the shared hub, and forcing the split
    # regresses other episodes' 0x46 minigames), so wire them here, scoped to the
    # gates the detector marked `bakery`. Each intro segment gains a trailing
    # `minigame` node routing win -> the pass_at arm, lose -> the fail_at arm,
    # all byte-derived from the fork offsets.
    def _seg_at_scene(scn, off):
        cont = after = al = None
        pref = "s%s" % scn
        for nm, bd in files.items():
            if not (nm == pref or nm.startswith(pref + "_")):
                continue
            lohi = _sr.get(nm)
            if not lohi:
                continue
            lo, hi = lohi
            if lo <= off <= hi:
                cont = nm
            elif lo >= off and (al is None or lo < al):
                after, al = nm, lo
        return cont or after

    def _seg_before_scene(scn, off, within=40):
        pref = "s%s" % scn
        best, gap = None, None
        for nm in files:
            if not (nm == pref or nm.startswith(pref + "_")):
                continue
            lohi = _sr.get(nm)
            if not lohi:
                continue
            lo, hi = lohi
            if hi < off and 0 <= off - hi <= within and (gap is None or off - hi < gap):
                best, gap = nm, off - hi
        return best

    for sc in doc.get("scenes", []):
        scn = str(sc["scene"])
        mgr = sc.get("minigame_results") or {}
        bgates = [g for g in (mgr.get("gates") or []) if g.get("bakery")]
        if not bgates:
            continue
        # The segment whose terminal is the var2004 lose-customer cascade dispatch
        # (the bakery entry). A bakery loss routes here so the counter advances
        # and the next distinct "no sale" interaction plays.
        _bakery_cascade_entry = None
        _pref = "s%s" % scn
        for nm, bd in files.items():
            if not (nm == _pref or nm.startswith(_pref + "_")):
                continue
            for _nd in bd.get("nodes") or []:
                if isinstance(_nd, dict) and _nd.get("type") == "dispatch" \
                        and str(_nd.get("state_var")) == "2004":
                    _bakery_cascade_entry = nm
                    break
            if _bakery_cascade_entry:
                break
        # Reconstruct the head of the var2009 loss-interaction chain. The chain
        # serves the next "no sale" customer by loss count: 1st -> Hannah, 2nd ->
        # Dexter, 3rd -> the twins (Kay/Kel), else -> the generic "no one bought"
        # close. The eq2 and eq3 gates segment correctly (the s4 "_else" gate
        # nodes), but the eq1 -> Hannah gate lands in an overlapping scene-1 byte
        # span and is lost, so build it here: a gate that tests var2009 == 1,
        # routing to the Hannah interaction on a match and to the existing eq2
        # gate otherwise. `_bakery_lose_head` is the entry the day minigames' lose
        # arm points at.
        _bakery_lose_head = None
        _has = lambda nm: nm in files

        def _find_seg(pred):
            for nm in files:
                if not (nm == _pref or nm.startswith(_pref + "_")):
                    continue
                for _n in files[nm].get("nodes") or []:
                    if isinstance(_n, dict) and pred(nm, _n):
                        return nm
            return None

        _hannah = _find_seg(lambda nm, n: n.get("speaker") == "Hannah"
                            and nm.endswith("_var2009_eq1"))
        if not _hannah:
            _hannah = _find_seg(lambda nm, n: n.get("speaker") == "Hannah")
        # the existing eq2 gate segment (var2009 == 2 -> Dexter)
        _eq2_gate = _find_seg(lambda nm, n: n.get("type") == "gate"
                              and str(n.get("var")) == "2009"
                              and str(n.get("equals")) == "2")
        if _hannah and _eq2_gate:
            _head_name = "s%s_lose" % scn
            if _head_name not in files:
                # The lose entry first advances the loss counter (var2009 += 1),
                # THEN routes by its new value: 1 -> Hannah, 2 -> Dexter, 3 -> the
                # twins, else -> the generic "no one bought a cookie" close. The
                # bytecode does `read var2009 ; +1 ; write` immediately before the
                # ==1 gate (byte offset ~12450); without that increment the counter
                # stays 0 and every loss falls straight through to the generic
                # close, so the Hannah/Dexter/twins escalation never plays. Emit the
                # increment as the head's first node so repeated losses escalate.
                files[_head_name] = {"nodes": [
                    {"type": "var_add", "var": "2009", "value": 1,
                     "source": "bakery_loss_counter"},
                    {"type": "gate", "var": "2009", "equals": "1",
                     "then": _hannah + ".json", "else": _eq2_gate + ".json"}]}
            _bakery_lose_head = _head_name

        # The bake days play the bakery word-sets IN ORDER: winning a day advances
        # to the next set, losing repeats the same set the next day. There are four
        # unique pick word-sets (Cinnamon, Coconut, Peanut butter, Snickerdoodles)
        # but five bake days -- day 4 repeats the Peanut-butter set before the final
        # Snickerdoodles day, matching the raw bytecode's five PICK sections (the
        # fourth reuses the third's word-set). So the day -> set index sequence is
        # [0, 1, 2, 2, 3]; days beyond that clamp to the last set.
        _pw_bank = (sc.get("minigames") or {}).get("pick_word") or []
        _bw_bank = (sc.get("minigames") or {}).get("build_word") or []
        _day_set_seq = [0, 1, 2, 2, 3]

        def _set_index(day_idx, bank_len):
            if bank_len <= 0:
                return 0
            raw = (_day_set_seq[day_idx] if day_idx < len(_day_set_seq)
                   else day_idx)
            return min(raw, bank_len - 1)

        for _day_idx, g in enumerate(bgates):
            trig = int(g["trigger"])
            intro = _seg_before_scene(scn, trig)
            if intro is None:
                continue
            # The bakery gate is `minigame; push 2; ==; 2b`: the win arm is the
            # customer sale that pays out (character interaction + "Kim earned N
            # dollars") and then ends the day at the hub; the loss repeats the
            # same day. The win content is the segment that immediately follows
            # this day's intro in bytecode order (verified against gameplay: day 1
            # -> Ashley $10, day 2 -> Spike $20, day 3 -> Mona $30, day 4 ->
            # Shapiro $40, day 5 -> Zero-G $50). The fork's raw fail_at/pass_at
            # offsets cross segment boundaries unreliably for the later days, so
            # anchor on the intro's own scene span instead: win = the next segment
            # in this scene after the intro that carries a payout line.
            intro_hi = (_sr.get(intro) or (0, 0))[1]
            win = None
            _cands = []
            for nm in files:
                if not (nm == _pref or nm.startswith(_pref + "_")):
                    continue
                lohi = _sr.get(nm)
                if not lohi or nm == intro:
                    continue
                lo, hi = lohi
                if lo > intro_hi:
                    _cands.append((lo, nm))
            for _lo, nm in sorted(_cands):
                _has_payout = any(isinstance(x, dict)
                                  and x.get("type") in ("dialogue", "narration",
                                                         "status")
                                  and "dollar" in (x.get("text") or "").lower()
                                  for x in (files[nm].get("nodes") or []))
                win = nm
                if _has_payout:
                    break
            if not win:
                continue
            # Loss: the bytecode increments the loss counter (var2009) and routes
            # into a gate chain that serves the next "no sale" customer by count --
            # 1st loss -> Hannah, 2nd -> Dexter, 3rd -> the twins (Kay/Kel), and
            # every loss after that -> the generic "no one bought a cookie" close;
            # each ends the day back at the hub, where the player picks the next
            # day's activity. Point `lose` at the head of that var2009 chain so the
            # loss path is reachable in order. The chain head is reconstructed
            # below (the eq1->Hannah gate lands in an overlapping scene-1 span in
            # the raw segmentation, so it is rebuilt here rather than trusted from
            # the offset attribution).
            lose = _bakery_lose_head or intro
            mg_node = {"type": "minigame", "minigame_type": "pick_word",
                       "kind": "bake",
                       "win": win + ".json", "lose": lose + ".json",
                       "win_threshold": str(g.get("win_threshold")),
                       "timed": True, "timer_ms": "20000",
                       "offset": str(g.get("trigger")),
                       "note": ("Bakery bake-the-cookies round. TIME-BOUND: the "
                                "game keeps drawing prompts from its round pool "
                                "until the timer (timer_ms) runs out ('Time's up!'); "
                                "there is no fixed round count. The player must get "
                                "win_threshold prompts correct before time expires "
                                "to win. Win: the customer buys and pays out, then "
                                "the day ends back at the hub. Loss: the loss "
                                "counter (var2009) advances and the next no-sale "
                                "interaction plays (Hannah, then Dexter, then the "
                                "twins, then the generic 'no one bought a cookie' "
                                "close), ending the day at the hub.")}
            # Attach the baking prompts the player plays. The bakery bake round was
            # authored in two device variants, both compiled into the script: a
            # tap-to-choose "pick word" game (opcode 0x47, drawing from the
            # pick_word bank -- "Bake cookies! Pick the right ingredients!") and a
            # spell-from-letters "build word" game (opcode 0x46, drawing from the
            # build_word bank -- "Add sugar and flour!" with a letter pool). The
            # original ran one or the other depending on the device (the pick / tap
            # variant on mobile, the build / spell variant on the larger-screen
            # tablet build); the selection happened in the native player, not via a
            # scripted variable, so both banks sit in the file. Carry BOTH so the
            # runtime can render whichever suits its target device: `setup` is the
            # default (mobile pick word) and `setup_alt` is the tablet build-word
            # variant. NOTE: `rounds` is the POOL the timed game draws from, not a
            # fixed sequence played to completion -- the game cycles prompts (the
            # "Next round" beat) until "Time's up!". The 0x47 pick game carries an
            # explicit 20000 ms timer operand; the 0x46 build game uses the same
            # engine timer, with its leading operand (9) being the pool size rather
            # than a round cap.
            # Each bake day is a SINGLE pick-word game: one word-set of two correct
            # and two decoy words. The list reshuffles as the player picks, and they
            # keep picking from that same set until the ~20s timer ends (tallying
            # correct picks toward win_threshold). Which set a day plays follows the
            # play order (day -> set index via _day_set_seq): winning advances to
            # the next set, losing repeats the same one.
            _pi = _set_index(_day_idx, len(_pw_bank))
            _bi = _set_index(_day_idx, len(_bw_bank))
            _pw = _pw_bank[_pi:_pi + 1] if _pw_bank else []
            _bw = _bw_bank[_bi:_bi + 1] if _bw_bank else []
            if _pw:
                mg_node["setup"] = {"bank": "pick_word", "rounds": _pw,
                                    "device": "mobile", "timer_ms": "20000"}
                mg_node["minigame_type"] = "pick_word"
                if _bw:
                    mg_node["setup_alt"] = {"bank": "build_word", "rounds": _bw,
                                            "device": "tablet",
                                            "timer_ms": "20000"}
            elif _bw:
                mg_node["setup"] = {"bank": "build_word", "rounds": _bw,
                                    "device": "tablet", "timer_ms": "20000"}
                mg_node["minigame_type"] = "build_word"
            nds = files[intro].get("nodes") or []
            # replace a trailing `next`/`goto_scene` with the minigame, else append
            if nds and isinstance(nds[-1], dict) \
                    and nds[-1].get("type") in ("next", "goto_scene"):
                nds[-1] = mg_node
            else:
                nds.append(mg_node)
            files[intro]["nodes"] = nds

        # Repair the bakery entry so it routes cleanly to the day intros. The raw
        # scene-4 entry linearizes the shared bake-the-cookies ingredient rounds
        # as five standalone `action` minigame nodes (win/lose null) ahead of the
        # var2004 day dispatch, and that dispatch's arms resolve to win/lose
        # fragments rather than the day intros (the scene-1/3/4 byte-offset overlap
        # corrupts the landing segments). Rebuild it: drop the dead entry
        # minigames and point the var2004 arms at the day intros in win-count order
        # (0 wins -> day 1 ... 4 wins -> day 5), with the all-days-done default
        # going to the bakery-closed close. Scoped to the scene that actually has
        # the bakery gates, so no other episode is touched.
        _entry = files.get(_pref)
        if _entry is not None:
            _enodes = _entry.get("nodes") or []
            # the ordered day intros in this scene (each carries a bake minigame)
            _day_intros = []
            for nm in sorted(files,
                             key=lambda n: (_sr.get(n) or (1 << 30, 0))[0]):
                if not (nm == _pref or nm.startswith(_pref + "_")):
                    continue
                if any(isinstance(x, dict) and x.get("type") == "minigame"
                       and x.get("kind") == "bake"
                       for x in files[nm].get("nodes") or []):
                    _day_intros.append(nm)
            _closed = next((nm for nm in files
                            if nm.startswith(_pref + "_")
                            and nm.endswith("_bakery_closed")), None)
            if _day_intros:
                # strip the leading dead action-minigames (win/lose both null)
                _kept = [x for x in _enodes
                         if not (isinstance(x, dict)
                                 and x.get("type") == "minigame"
                                 and x.get("kind") == "action"
                                 and not x.get("win") and not x.get("lose"))]
                # rebuild / correct the var2004 dispatch arms
                _fixed = []
                for x in _kept:
                    if isinstance(x, dict) and x.get("type") == "dispatch" \
                            and str(x.get("state_var")) == "2004":
                        x = dict(x)
                        x["arms"] = [{"equals": str(i), "scene": nm + ".json"}
                                     for i, nm in enumerate(_day_intros)]
                        if _closed:
                            x["default"] = _closed + ".json"
                    _fixed.append(x)
                # if there was no var2004 dispatch left, append one
                if not any(isinstance(x, dict) and x.get("type") == "dispatch"
                           and str(x.get("state_var")) == "2004" for x in _fixed):
                    _fixed.append({"type": "dispatch", "state_var": "2004",
                                   "arms": [{"equals": str(i), "scene": nm + ".json"}
                                            for i, nm in enumerate(_day_intros)],
                                   "default": (_closed + ".json") if _closed
                                   else (_day_intros[0] + ".json")})
                _entry["nodes"] = _fixed

            # First-time vs replay intro gate. A bakery day can carry an extra
            # "first visit" intro variant (a segment named ..._var2010_eq0, the
            # var2010==0 arm of a `read var2010 ; ==0 ; 2b` gate in the bytecode:
            # e.g. Day 4's full "This is the Bake-ronomicon..." reveal, shown only
            # the first time, versus the terse "...try the Bake-ronomicon again?"
            # on a replay). The day dispatch enters the terse intro directly, so the
            # first-time variant is left unreferenced. Reconnect it byte-derivably:
            # find the day intro that shares the variant's continuation (both flow
            # into the same day gameplay), and route the dispatch arm through a
            # gate -- var2010==0 -> the first-time variant, else -> the terse intro.
            _variant = next((nm for nm in files
                             if (nm == _pref or nm.startswith(_pref + "_"))
                             and nm.endswith("_var2010_eq0")), None)
            if _variant is not None:
                # the continuation the variant leads into (its terminal next)
                _vnodes = files[_variant].get("nodes") or []
                _vnext = None
                for x in reversed(_vnodes):
                    if isinstance(x, dict) and x.get("type") in ("next",
                                                                 "goto_scene"):
                        _vnext = (x.get("next") or x.get("file"))
                        break
                # the terse intro is the day intro whose gameplay is that same
                # continuation (the one the dispatch currently enters)
                _terse = None
                for nm in _day_intros:
                    _mynext = None
                    for x in files[nm].get("nodes") or []:
                        if isinstance(x, dict) and x.get("type") == "minigame" \
                                and x.get("kind") == "bake":
                            _mynext = x.get("win")
                    # match by shared continuation, or by adjacency in offset order
                    if _vnext and (_mynext == _vnext
                                   or files[nm] is files.get(
                                       (_vnext or "").replace(".json", ""))):
                        _terse = nm
                        break
                if _terse is None:
                    # fall back to the day intro immediately after the variant by
                    # offset (the variant sits just before its terse counterpart)
                    _voff = min((int(x["offset"]) for x in _vnodes
                                 if isinstance(x, dict)
                                 and x.get("offset") is not None), default=None)
                    if _voff is not None:
                        _after = sorted(
                            ((_sr.get(nm) or (1 << 30, 0))[0], nm)
                            for nm in _day_intros
                            if (_sr.get(nm) or (1 << 30, 0))[0] > _voff)
                        _terse = _after[0][1] if _after else None
                if _terse is not None:
                    # repoint the dispatch arm for the terse intro at a gate node
                    _gate_name = _terse + "_firstvisit_gate"
                    files[_gate_name] = {"nodes": [{
                        "type": "gate", "var": "2010", "equals": "0",
                        "then": _variant + ".json",
                        "else": _terse + ".json",
                        "source": "first_visit_intro"}]}
                    for x in _entry.get("nodes") or []:
                        if isinstance(x, dict) and x.get("type") == "dispatch" \
                                and str(x.get("state_var")) == "2004":
                            for a in x.get("arms") or []:
                                if (a.get("scene") or "") == _terse + ".json":
                                    a["scene"] = _gate_name + ".json"
                    # ensure the variant continues into the terse intro's gameplay
                    if _vnext is None:
                        files[_variant].setdefault("nodes", []).append(
                            {"type": "next", "next": _terse + ".json"})
    # real content at the gate's pass_at/fail_at offsets. The Help Ads (scene 3)
    # use score-forked minigames just like the bakery: on a good result the
    # helpee is satisfied and pays out, on a poor one they are unhappy. When the
    # generic fork splitter produced empty win[]/lose[] stubs (the arm content
    # sits in a segment the split could not capture), resolve each side to the
    # segment that actually contains the byte-derived arm offset so the outcome
    # (e.g. Andrew's dismissive "rock opera" reply) is reachable.
    def _has_content(nm):
        for x in files.get(nm, {}).get("nodes", []) or []:
            if isinstance(x, dict) and x.get("type") in ("dialogue", "narration",
                                                          "choice"):
                return True
        return False

    for sc in doc.get("scenes", []):
        scn = str(sc["scene"])
        by_gate = {int(g["gate"]): g for g in
                   ((sc.get("minigame_results") or {}).get("gates") or [])
                   if g.get("gate") is not None}
        if not by_gate:
            continue
        for nm, bd in files.items():
            for nd in bd.get("nodes") or []:
                if not isinstance(nd, dict) or nd.get("type") != "minigame":
                    continue
                off = nd.get("offset")
                g = by_gate.get(int(off)) if off is not None \
                    and str(off).lstrip("-").isdigit() else None
                if g is None:
                    continue
                w = (nd.get("win") or "").replace(".json", "")
                l = (nd.get("lose") or "").replace(".json", "")
                if w and not _has_content(w):
                    real = _seg_at_scene(scn, int(g["pass_at"]))
                    if real and _has_content(real):
                        nd["win"] = real + ".json"
                if l and not _has_content(l):
                    real = _seg_at_scene(scn, int(g["fail_at"]))
                    if real and _has_content(real):
                        nd["lose"] = real + ".json"

    # Link an offset-less CHOICE segment to the segment that poses its question.
    # A choice node carries no byte offset, so the byte-adjacency pass is blind
    # to it and cannot position it between the prompt and the following section.
    # When the decoder splits the question's answer choices into their own
    # offset-less segment, the prompt-bearing segment's `next` can skip past it
    # (e.g. Making Some Dough's English quiz: s2_15 ends with "Which of these
    # words is spelled correctly?" but jumps to the outcome s2_20, orphaning the
    # spelling-choice segment s2_16). Reconnect byte-derivably: when an
    # offset-less choice is unreferenced and EXACTLY ONE segment's final line
    # equals that choice's prompt, repoint that segment's terminal `next` to the
    # choice. Strict one-to-one matching on the full prompt text keeps this from
    # firing on generic prompts ("What should I do?") shared by many segments.
    def _norm(t):
        return (t or "").strip().strip("'\" ").lower()

    referenced = set()
    for _bd in files.values():
        for _nd in _bd.get("nodes") or []:
            if not isinstance(_nd, dict):
                continue
            for _k in ("next", "then", "else", "win", "lose", "file",
                       "to_scene", "exhausted", "default"):
                _v = _nd.get(_k)
                if isinstance(_v, str):
                    referenced.add(_v.replace(".json", ""))
            for _o in _nd.get("options", []) or []:
                if isinstance(_o, dict) and _o.get("next"):
                    referenced.add(_o["next"].replace(".json", ""))
            for _a in _nd.get("arms", []) or []:
                if isinstance(_a, dict) and _a.get("scene"):
                    referenced.add(_a["scene"].replace(".json", ""))

    for name, bd in files.items():
        nds = bd.get("nodes") or []
        if name in referenced:
            continue
        has_off = any(isinstance(x, dict) and x.get("offset") is not None
                      and str(x["offset"]).lstrip("-").isdigit() for x in nds)
        if has_off:
            continue
        prompt = next((x.get("prompt") for x in nds
                       if isinstance(x, dict) and x.get("type") == "choice"), None)
        if not prompt:
            continue
        # segments whose final displayed line equals this prompt
        askers = []
        for sid2, bd2 in files.items():
            if sid2 == name:
                continue
            texts = [x.get("text") for x in bd2.get("nodes") or []
                     if isinstance(x, dict) and x.get("text")]
            if texts and _norm(texts[-1]) == _norm(prompt):
                askers.append(sid2)
        if len(askers) != 1:
            continue                            # ambiguous or unmatched -> skip
        asker = files[askers[0]]
        anodes = asker.get("nodes") or []
        if anodes and isinstance(anodes[-1], dict) \
                and anodes[-1].get("type") == "next":
            anodes[-1]["next"] = name + ".json"

    # Wire the "Replay from checkpoint" retry option to its byte-derived target.
    # The retry choice on a fail screen offers "Replay from checkpoint." and
    # "Restart episode."; the decoder leaves the replay branch pointing at an empty
    # `end` stub because the jump is a raw 0x63-checkpoint SEP the resolver skips.
    # Using the scene's byte-derived checkpoint records (targets + var resets), find
    # each such stub and repoint it: emit a `checkpoint_replay` node carrying the
    # counter resets (var2000/var2001 -> 0, byte-derived from the checkpoint SET)
    # and a `next` to the segment containing the replay target offset. Restart
    # branches (var1001 := 0 -> scene entry) are already correct and untouched. This
    # only rewrites stubs that are otherwise dead, so episodes whose replay is
    # already wired (e.g. Halloween) are unaffected.
    for sc in doc["scenes"]:
        _scn = str(sc.get("scene"))
        _ck = (sc.get("section_dispatch") or {}).get("checkpoints") or {}
        _replays = _ck.get("replays") or []
        _sets = _ck.get("sets") or []
        if not _replays:
            continue
        # the reset payload: var writes zeroed by the checkpoint SET (var2001, and
        # var2000 when present); default to both if a SET is present without detail
        _resets = []
        for _s in _sets:
            for _v in _s.get("resets") or []:
                if _v not in _resets:
                    _resets.append(_v)
        if not _resets:
            _resets = [2000, 2001]
        # segments in this scene, by their min offset, to locate a target offset
        _scene_segs = []
        for _nm, _bd in files.items():
            if str(_scene_of.get(_nm)) != _scn:
                continue
            _offs = [int(x["offset"]) for x in (_bd.get("nodes") or [])
                     if isinstance(x, dict) and x.get("offset") is not None]
            if _offs:
                _scene_segs.append((min(_offs), max(_offs), _nm))
        _scene_segs.sort()

        def _seg_for_offset(off):
            # the segment the target offset heads. The checkpoint target is a
            # section head, which may sit a couple of instructions before the
            # segment's first *recorded* node offset, so prefer the nearest segment
            # whose start is at or just after the target (within a small window),
            # then fall back to a containing span, then to the last segment before.
            off = int(off)
            after = [(lo, nm) for lo, hi, nm in _scene_segs if 0 <= lo - off <= 24]
            if after:
                return min(after)[1]
            for _lo, _hi, _nm in _scene_segs:
                if _lo <= off <= _hi + 40:
                    return _nm
            best = None
            for _lo, _hi, _nm in _scene_segs:
                if _lo <= off:
                    best = _nm
            return best

        # find replay stub segments: a segment reached by a "Replay from
        # checkpoint" choice option whose only content is a terminal `end`
        for _nm, _bd in files.items():
            if str(_scene_of.get(_nm)) != _scn:
                continue
            for _nd in _bd.get("nodes") or []:
                if not (isinstance(_nd, dict) and _nd.get("type") == "choice"):
                    continue
                for _opt in _nd.get("options") or []:
                    if not isinstance(_opt, dict):
                        continue
                    _lbl = (_opt.get("label") or "").lower()
                    if "checkpoint" not in _lbl and "replay" not in _lbl:
                        continue
                    _tgt_seg = (_opt.get("next") or "").replace(".json", "")
                    _tn = files.get(_tgt_seg, {}).get("nodes") or []
                    _is_stub = (len(_tn) == 0) or all(
                        isinstance(x, dict) and x.get("type") in ("end", "next")
                        and not x.get("next") for x in _tn)
                    if not _is_stub:
                        continue                # already wired -> leave it
                    # resolve the replay target offset -> its segment
                    _dest = _seg_for_offset(_replays[0]["target"])
                    if _dest is None or _dest == _tgt_seg:
                        continue
                    _new = [{"type": "var_set", "var": str(_v), "value": 0,
                             "source": "checkpoint_reset"} for _v in _resets]
                    _new.append({"type": "checkpoint_replay",
                                 "next": _dest + ".json",
                                 "note": ("Replay from the last passed checkpoint: "
                                          "reset the run counters and resume at the "
                                          "checkpoint story beat. Byte-derived from "
                                          "the 0x63 checkpoint SEP target.")})
                    files[_tgt_seg]["nodes"] = _new

    # Attach timed (quick-time) metadata to choice nodes. A byte-derived timed
    # choice (offset -> timer_ms, from the scene's section_dispatch) marks the
    # action/reflex choices shown on a countdown; the engine runs the timer and,
    # on expiry, takes the same non-ideal branch a wrong pick does. Choice nodes do
    # not carry their own bytecode offset, so locate each choice's approximate
    # offset from its neighbours (the setup line just before the 1f01, or the first
    # outcome line just after it) and match to the nearest timed-choice offset.
    for sc in doc["scenes"]:
        _tc = (sc.get("section_dispatch") or {}).get("timed_choices") or {}
        if not _tc:
            continue
        _scn = str(sc.get("scene"))
        _tc_offs = sorted(int(k) for k in _tc)
        _used = set()
        # option-label signature per timed-choice offset, for offset-less choices
        _tc_labels = {}
        for _k, _v in _tc.items():
            _lbls = _v.get("labels")
            if _lbls:
                _tc_labels.setdefault(tuple(_lbls), []).append(int(_k))
        # scene-wide (min_offset, max_offset, name) index, to resolve a distinct
        # timeout branch's byte offset back to the segment that begins there
        _scene_spans = []
        for _nm2, _bd2 in files.items():
            if str(_scene_of.get(_nm2)) != _scn:
                continue
            _o2 = [int(x["offset"]) for x in (_bd2.get("nodes") or [])
                   if isinstance(x, dict) and x.get("offset") is not None]
            if _o2:
                _scene_spans.append((min(_o2), max(_o2), _nm2))
        _scene_spans.sort()

        def _seg_at(off):
            off = int(off)
            after = [(lo, nm) for lo, hi, nm in _scene_spans if 0 <= lo - off <= 24]
            if after:
                return min(after)[1]
            for lo, hi, nm in _scene_spans:
                if lo <= off <= hi + 24:
                    return nm
            return None

        for _nm, _bd in files.items():
            if str(_scene_of.get(_nm)) != _scn:
                continue
            _ns = _bd.get("nodes") or []
            for _i, _nd in enumerate(_ns):
                if not (isinstance(_nd, dict) and _nd.get("type") == "choice"):
                    continue
                # nearest preceding offset (the choice's 1f01 is a little past it)
                _prev = None
                for _j in range(_i - 1, -1, -1):
                    _po = _ns[_j].get("offset") if isinstance(_ns[_j], dict) else None
                    if _po is not None:
                        _prev = int(_po)
                        break
                # nearest following offset (the first outcome line after the 1f01)
                _next = None
                for _j in range(_i + 1, len(_ns)):
                    _no = _ns[_j].get("offset") if isinstance(_ns[_j], dict) else None
                    if _no is not None:
                        _next = int(_no)
                        break
                _match = None
                if _prev is not None:
                    _match = next((o for o in _tc_offs
                                   if o not in _used and 0 <= o - _prev <= 60), None)
                if _match is None and _next is not None:
                    _match = next((o for o in _tc_offs
                                   if o not in _used and 0 <= _next - o <= 60), None)
                if _match is None:
                    # offset-less choice (e.g. a crossing-loop round with no
                    # neighbouring lines): match by its option-label signature
                    _sig = tuple(o.get("label", "") for o in _nd.get("options") or [])
                    _cands = [o for o in _tc_labels.get(_sig, []) if o not in _used]
                    _match = _cands[0] if _cands else None
                if _match is None:
                    continue
                _used.add(_match)
                _info = _tc[str(_match)]
                _nd["timed"] = True
                if _info.get("timer_ms") is not None:
                    _nd["timer_ms"] = str(_info["timer_ms"])
                # Where does a timed-out choice go? When the bytecode gives the
                # timeout its own branch (a `==1000` test) that lands somewhere
                # other than the choice's default continuation, record it as an
                # explicit on_timeout target. In every observed case so far the
                # timeout resolves to the same place as `after` (an un-answered
                # timed choice simply follows the default continuation), so we do
                # NOT emit a redundant on_timeout in that case -- the engine uses
                # `after`. on_timeout appears only when the timeout genuinely
                # diverges from the default branch.
                _toff = _info.get("timeout_offset")
                _tseg = _seg_at(_toff) if _toff is not None else None
                if _tseg is not None and _tseg != _nm:
                    _tval = _tseg + ".json"
                    if _tval != _nd.get("after"):
                        _nd["on_timeout"] = _tval

    # Ending-terminal repair. Making Some Dough's several alternate endings are
    # laid out linearly and each ends with a `SEP op=63` "return to caller" jump
    # whose target (in a data region) does not resolve, so the decoder falls each
    # ending through into the *next* ending section in file order. That is wrong:
    # a `22`-marker sits between them, so they are independent alternate endings,
    # and (for example) the loan/Boss "Congratulations! You saved the bakery"
    # ending leaks straight into the "I wasn't able to get enough money" failure
    # text. Re-point every such ending to the shared end-of-episode survey instead,
    # so each ending concludes and then hands off to the survey rather than
    # replaying a different ending's narration.
    #
    # Byte-derived: an ending is a segment whose bytecode span ends with `0x42`
    # SEP carrying operand 63 immediately followed (within a few instructions) by a
    # `22 43 48 4a` section marker OR the file's end; the survey is the segment
    # that owns the "Take Survey.|Skip Survey." choice.
    _ending_scn_data = {}
    for sc in doc.get("scenes", []):
        _lbl = sc.get("script")
        _cf = (sc.get("control_flow") or {})
        # collect this scene's raw bytes-driven SEP op=63 terminals if present
        _seps63 = sc.get("_sep63_terminals")
        if _seps63:
            _ending_scn_data[str(sc.get("scene"))] = _seps63
    # locate the survey segment (owns a choice with a "Take Survey" option)
    _survey_seg = None
    for _nm, _bd in files.items():
        for _n in _bd.get("nodes") or []:
            if isinstance(_n, dict) and _n.get("type") == "choice" \
                    and any("take survey" in (o.get("label") or "").lower()
                            for o in _n.get("options") or []):
                _survey_seg = _nm
                break
        if _survey_seg:
            break
    if _survey_seg is not None and _ending_scn_data:
        # Split the survey choice out of the failure-preamble segment. This is only
        # needed when the scene has `SEP op=63` ending terminals (see
        # _ending_scn_data) that must bypass a failure-only preamble to reach the
        # shared survey -- the Making Some Dough case, where the survey choice is
        # packaged in the same segment as the "Yikes! Talk about a depressing
        # ending..." narration that belongs ONLY to the failure ending. Episodes
        # without such terminals (e.g. As Time Goes By, whose endings converge
        # cleanly on a single survey segment) are left untouched: their survey
        # narration is a legitimate shared convergence point, not failure-only
        # text, so splitting it would needlessly restructure a correct ending.
        _survey_nodes = files[_survey_seg].get("nodes") or []
        _split_at = None
        for _i, _n in enumerate(_survey_nodes):
            if isinstance(_n, dict) and _n.get("type") == "choice" \
                    and any("take survey" in (o.get("label") or "").lower()
                            for o in _n.get("options") or []):
                _split_at = _i
                break
        _survey_choice_seg = _survey_seg
        if _split_at is not None and _split_at > 0:
            _survey_choice_seg = _survey_seg + "_choice"
            files[_survey_choice_seg] = {"nodes": _survey_nodes[_split_at:]}
            _scene_of[_survey_choice_seg] = _scene_of.get(_survey_seg)
            files[_survey_seg]["nodes"] = _survey_nodes[:_split_at] + [
                {"type": "next", "next": _survey_choice_seg + ".json",
                 "source": "survey_split"}]

        for _scn, _term_offs in _ending_scn_data.items():
            _scene_segs = sorted(
                ((min(int(x["offset"]) for x in (files[nm].get("nodes") or [])
                      if isinstance(x, dict) and x.get("offset") is not None),
                  max(int(x["offset"]) for x in (files[nm].get("nodes") or [])
                      if isinstance(x, dict) and x.get("offset") is not None), nm)
                 for nm in files
                 if str(_scene_of.get(nm)) == _scn
                 and any(isinstance(x, dict) and x.get("offset") is not None
                         for x in files[nm].get("nodes") or [])),
                key=lambda t: t[0])
            for _toff in _term_offs:
                _toff = int(_toff)
                _owner = None
                for _lo, _hi, _nm in _scene_segs:
                    _lo, _hi = int(_lo), int(_hi)
                    if _lo <= _toff <= _hi + 40:
                        _owner = _nm
                # never redirect the survey segments themselves, nor the failure
                # preamble that legitimately leads INTO the survey (that chain
                # already resolves to the survey via the split above)
                if _owner is None or _owner in (_survey_seg, _survey_choice_seg):
                    continue
                _onodes = files[_owner].get("nodes") or []
                if _onodes and isinstance(_onodes[-1], dict) \
                        and _onodes[-1].get("type") == "next":
                    _cur = (_onodes[-1].get("next") or "").replace(".json", "")
                    if _cur != _survey_choice_seg:
                        _onodes[-1] = {"type": "next",
                                       "next": _survey_choice_seg + ".json",
                                       "source": "ending_terminal"}

    # Var2000 >= threshold gate that guards success vs failure endings. Making
    # Some Dough uses var2000 as money and gates the ending confrontation on
    # `var2000 >= 200`; the two-scene episodes (Tutors, Swim, Halloween Dance)
    # reuse var2000 as the minigame score and gate a rank/bonus branch on it. In
    # every case the gate lives in a small inter-section gap that segmentation
    # doesn't pin to a boundary, so the gate would be dropped and both arms would
    # merge (in dough: the failure ending orphaned and every ending running
    # regardless of the goal). Re-emit the gate at the head of its success entry
    # so the engine takes the intended branch.
    _threshold_gate = None
    _threshold_scene = None
    for sc in doc.get("scenes", []):
        _sc_gates = (var_gates or {}).get(sc.get("script")) or []
        for _g in _sc_gates:
            if str(_g.get("var")) == "2000" and _g.get("op") == "gte":
                _threshold_gate = _g
                _threshold_scene = str(sc.get("scene"))
                break
        if _threshold_gate:
            break
    if _threshold_gate is not None:
        _gate_end = int(_threshold_gate["then"][1])
        _entry_seg = None
        _fail_seg = None
        _cands = [(min(int(x["offset"]) for x in (files[nm].get("nodes") or [])
                       if isinstance(x, dict) and x.get("offset") is not None),
                   nm)
                  for nm in files
                  if str(_scene_of.get(nm)) == _threshold_scene
                  and any(isinstance(x, dict) and x.get("offset") is not None
                          for x in files[nm].get("nodes") or [])]
        _cands.sort()
        for _lo, _nm in _cands:
            if _lo >= _gate_end - 8:
                _entry_seg = _nm
                break
        # the failure/no-bonus counterpart is a peer segment whose name mirrors
        # the success entry but with an `_m` suffix (the segment layout is
        # <base>_else -> then arm, <base>_else_m -> else arm), so just look up
        # the same-scene sibling
        if _entry_seg is not None:
            for _lo, _nm in _cands:
                if _nm.endswith("_m") and _nm[:-2] == _entry_seg:
                    _fail_seg = _nm
                    break
                if _nm.endswith("_else_m"):
                    _fail_seg = _nm
        # Guard against a self-referential loop. When the threshold branch is
        # already resolved as a real gate in the bytecode, one segment (the
        # `_fail_seg` candidate) already gates on the same condition and routes
        # into `_entry_seg`. Materializing another gate and rewiring `_entry_seg`
        # references through it then points that segment's own gate at the new
        # gate, whose else points back -- an infinite bounce (Halloween Dance's
        # score screen looped this way at sub-bonus scores). If the chosen
        # `_fail_seg` already contains a gate that reaches `_entry_seg`, the real
        # gate is doing the job and no materialization is needed; skip it.
        if _entry_seg is not None and _fail_seg is not None:
            _already_gated = False
            for _n in files.get(_fail_seg, {}).get("nodes") or []:
                if isinstance(_n, dict) and _n.get("type") == "gate":
                    for _k in ("then", "else"):
                        if (_n.get(_k) or "").replace(".json", "") == _entry_seg:
                            _already_gated = True
            if _already_gated:
                _fail_seg = None
        if _entry_seg is not None and _fail_seg is not None:
            _gate_seg = _entry_seg + "_threshold_gate"
            files[_gate_seg] = {"nodes": [{
                "type": "gate", "var": "2000", "op": "gte",
                "equals": str(int(_threshold_gate["equals"])),
                "then": _entry_seg + ".json",
                "else": _fail_seg + ".json",
                "source": "var2000_threshold_gate"}]}
            _scene_of[_gate_seg] = _threshold_scene
            for _nm, _bd in files.items():
                if _nm == _gate_seg:
                    continue
                for _n in _bd.get("nodes") or []:
                    if not isinstance(_n, dict):
                        continue
                    for _k in ("next", "then", "else", "after"):
                        if (_n.get(_k) or "").replace(".json", "") == _entry_seg:
                            _n[_k] = _gate_seg + ".json"

    # Bonus-scene routing repair. A score gate can unlock a bonus scene: the
    # `then` (high score) branch shows a short "You have unlocked the bonus scene!"
    # card, and the `else` (low score) branch shows the ordinary end-of-episode
    # wrap-up. In the bytecode the bonus montage and the shared survey are welded
    # onto the tail of the wrap-up section, and the "bonus unlocked" card ends with
    # a jump INTO the montage -- but segmentation leaves the card dead-ending and
    # bundles promo + montage + survey into one segment, so the high-score path
    # never plays the montage and the low-score path wrongly plays it. Repair by
    # splitting that segment into promo / montage / survey and wiring:
    #   high score: "bonus unlocked" -> montage -> survey
    #   low score : promo            ->            survey
    # The montage boundary is the first narration after the promo/social lines that
    # begins the scene ("A while later, ... wanders out"); the survey is the
    # trailing Take/Skip-Survey choice. Byte-agnostic and self-contained: it only
    # fires when a gate's then-branch is a dead-end bonus card whose sibling
    # else-branch carries both a montage and a survey choice.
    def _seg_text(_nm, _idx):
        _n = (files.get(_nm, {}).get("nodes") or [])
        if 0 <= _idx < len(_n) and isinstance(_n[_idx], dict):
            return (_n[_idx].get("text") or "")
        return ""

    for _gnm in list(files.keys()):
        for _gn in files[_gnm].get("nodes") or []:
            if not (isinstance(_gn, dict) and _gn.get("type") == "gate"):
                continue
            _then_v, _else_v = _gn.get("then"), _gn.get("else")
            if not isinstance(_then_v, str) or not isinstance(_else_v, str):
                continue                       # inline (non-ref) arms: skip this pass
            _then = _then_v.replace(".json", "")
            _else = _else_v.replace(".json", "")
            if _then not in files or _else not in files:
                continue
            # then-branch must be a short "bonus unlocked" card that dead-ends
            _tn = files[_then].get("nodes") or []
            _t_txt = " ".join((n.get("text") or "") for n in _tn
                              if isinstance(n, dict)).lower()
            _t_deadend = not any(isinstance(n, dict) and n.get("type") in
                                 ("next", "goto_scene", "choice", "gate",
                                  "dispatch", "minigame")
                                 for n in _tn)
            if "bonus scene" not in _t_txt or not _t_deadend:
                continue
            # else-branch must carry a survey choice (Take/Skip Survey)
            _en = files[_else].get("nodes") or []
            _survey_idx = None
            for _i, _n in enumerate(_en):
                if isinstance(_n, dict) and _n.get("type") == "choice" \
                        and any("survey" in (o.get("label") or "").lower()
                                for o in _n.get("options") or []):
                    _survey_idx = _i
                    break
            if _survey_idx is None:
                continue
            # montage begins at the first narration mentioning the scene turn
            # ("a while later" / "wanders out") after the promo lines
            _montage_idx = None
            for _i, _n in enumerate(_en):
                if isinstance(_n, dict) and _n.get("type") == "narration":
                    _lt = (_n.get("text") or "").lower()
                    if "a while later" in _lt or "wanders out" in _lt:
                        _montage_idx = _i
                        break
            if _montage_idx is None or _montage_idx >= _survey_idx:
                continue
            # carve the survey tail into its own shared segment
            _survey_seg2 = _else + "_survey"
            files[_survey_seg2] = {"nodes": _en[_survey_idx:]}
            _scene_of[_survey_seg2] = _scene_of.get(_else)
            # carve the montage (between promo and survey) into its own segment,
            # ending by falling through to the shared survey
            _montage_seg = _else + "_bonus"
            files[_montage_seg] = {"nodes": _en[_montage_idx:_survey_idx] +
                                   [{"type": "next",
                                     "next": _survey_seg2 + ".json",
                                     "source": "bonus_montage"}]}
            _scene_of[_montage_seg] = _scene_of.get(_else)
            # the else-branch keeps only the promo prefix, then jumps to the survey
            files[_else]["nodes"] = _en[:_montage_idx] + [
                {"type": "next", "next": _survey_seg2 + ".json",
                 "source": "bonus_skip"}]
            # the bonus card (then-branch) now flows into the montage instead of
            # dead-ending
            files[_then]["nodes"] = _tn + [
                {"type": "next", "next": _montage_seg + ".json",
                 "source": "bonus_unlocked"}]

    # Wire a minigame content-fork whose fail (lose) arm would otherwise be
    # orphaned. A word-bank minigame can branch on the player's score: on a pass
    # the winning content plays inline and continues to the next scene; on a fail
    # the engine takes a 0x2b jump to a distinct lose block. The win path's own
    # jump-over-the-lose-block (`28 -> convergence`) means the lose block is
    # reached ONLY by the minigame's fail branch -- so if the fork isn't
    # materialized, that block is unreachable. A Float Is Born's scene-1 drawing
    # game is the case: win = "Stay focused... this looks really good" -> scene 2;
    # lose = "Too tired... passes out... the drawing is terrible" (s1_08) -> scene
    # 2. Both reconverge (content_fork), but the lose arm is real, distinct content.
    #
    # This is scoped tightly so it never disturbs the other episodes: it fires
    # only for a content_fork gate whose fail_at maps to a segment that NOTHING
    # else routes to (a genuine minigame-only lose path). The bakery's repeat-day
    # word games (Making Some Dough) fail BACKWARD to an earlier segment that the
    # day loop already reaches, and tut/swim/hallo's flagged content-forks fail to
    # blocks that a story choice already enters -- in every one of those the fail
    # target is reachable, so this pass leaves them untouched.
    def _norm(x):
        return x.replace(".json", "") if isinstance(x, str) else x

    def _edge_targets(_b):
        _out = []
        for _n in _b.get("nodes") or []:
            if not isinstance(_n, dict):
                continue
            _t = _n.get("type")
            if _t in ("next", "checkpoint_replay"):
                _out.append(_norm(_n.get("next")))
            elif _t == "goto_scene":
                _out.append(_norm(_n.get("file") or _n.get("to_scene")))
            elif _t == "gate":
                _out += [_norm(_n.get("then")), _norm(_n.get("else"))]
            elif _t == "choice":
                _out += [_norm(o.get("next")) for o in _n.get("options") or []]
            elif _t == "minigame":
                _out += [_norm(_n.get("win")), _norm(_n.get("lose"))]
            elif _t == "dispatch":
                _out += [_norm(a.get("scene")) for a in _n.get("arms") or []]
                _out.append(_norm(_n.get("default")))
            elif _t == "random":
                _out += [_norm(o.get("scene")) for o in _n.get("options") or []]
                _out.append(_norm(_n.get("exhausted")))
        return [x for x in _out if x]

    def _seg_first_off(_b):
        for _n in _b.get("nodes") or []:
            if isinstance(_n, dict) and _n.get("offset") is not None:
                try:
                    return int(_n["offset"])
                except (TypeError, ValueError):
                    pass
        return None

    _mg_gates = []
    for _sc in doc["scenes"]:
        _mr = _sc.get("minigame_results") or {}
        for _g in _mr.get("gates") or []:
            if _g.get("kind") == "content_fork" \
                    and _g.get("via") == "word_bank" \
                    and int(_g.get("fail_at", 0)) > int(_g.get("pass_at", 0)):
                _mg_gates.append((_sc.get("scene"), _g))

    def _seg_scene(_nm):
        _m = re.match(r"s(\d+)", _nm or "")
        return _m.group(1) if _m else None

    if _mg_gates:
        _incoming = {}
        for _nm, _b in files.items():
            for _tgt in _edge_targets(_b):
                _incoming.setdefault(_tgt, set()).add(_nm)
        for _scene_num, _g in _mg_gates:
            _pass_at = int(_g["pass_at"])
            _fail_at = int(_g["fail_at"])
            _trigger = int(_g.get("trigger") or _pass_at)
            # the lose segment: the one starting at fail_at that nothing reaches
            _lose_seg = None
            for _nm, _b in files.items():
                if _seg_scene(_nm) != str(_scene_num):
                    continue
                _fo = _seg_first_off(_b)
                if _fo is not None and abs(_fo - _fail_at) <= 20 \
                        and not _incoming.get(_nm):
                    _lose_seg = _nm
                    break
            if _lose_seg is None:
                continue                    # fail arm already reachable: skip
            # the trigger segment: holds the minigame lead-in and inlined win
            # content, and currently just flows onward (its win path). Find the
            # node index where the win content begins (offset >= pass_at) and the
            # win content's own terminal (a goto/next it already carries).
            _trig_seg = None
            for _nm, _b in files.items():
                if _seg_scene(_nm) != str(_scene_num):
                    continue
                _offs = [int(n["offset"]) for n in _b.get("nodes") or []
                         if isinstance(n, dict) and n.get("offset") is not None]
                if _offs and min(_offs) <= _trigger <= max(_offs) + 4:
                    _trig_seg = _nm
                    break
            if _trig_seg is None:
                continue
            _tn = files[_trig_seg]["nodes"]
            _win_start = next((i for i, n in enumerate(_tn)
                               if isinstance(n, dict) and n.get("offset") is not None
                               and int(n["offset"]) >= _pass_at), None)
            if _win_start is None:
                continue
            _lead = _tn[:_win_start]
            _win = _tn[_win_start:]
            # carve the win arm into its own segment so both arms are peers
            _win_seg = _trig_seg + "_win"
            files[_win_seg] = {"nodes": _win, "scene": _scene_num}
            # the trigger segment keeps the lead-in, then a minigame fork node
            _fork = {"type": "minigame", "kind": "content_fork",
                     "via": "word_bank",
                     "win_threshold": str(_g.get("win_threshold")),
                     "offset": str(_g.get("gate")),
                     "win": _win_seg + ".json",
                     "lose": _lose_seg + ".json",
                     "source": "minigame_fork"}
            files[_trig_seg]["nodes"] = _lead + [_fork]

    # Drop scene-header minigame DEFINITION duplicates. A by-id minigame (played
    # later via 1f51) is defined by a data-def block in the section header: the
    # pick_word / build_word bank the play point draws from. That bank is already
    # captured in `scene_minigame_banks`, and when the by-id play point is resolved
    # into a win/lose fork the header node is folded in and removed. But if a
    # scene's by-id play points are in OTHER segments (so no local fork consumes
    # the header node), the standalone definition is left stranded in the entry
    # segment as a spurious runtime minigame -- e.g. A Float Is Born's scene-1 and
    # scene-3 entries surfaced the "Design a great float / Stay awake" build-word
    # definition as a node before the story even starts.
    #
    # Such a node is identifiable and safe to drop: it is a minigame node with a
    # `setup` bank, NO win/lose arms (not a fork), that sits in its segment BEFORE
    # any spoken/narrated line (header position, not inline in the story). A real
    # in-flow minigame either carries win/lose arms or appears after story content,
    # so this never removes a playable round. (Verified against the five prior
    # episodes: their only standalone-with-setup minigame is inline, not header.)
    for _b in files.values():
        _nl = _b.get("nodes")
        if not isinstance(_nl, list):
            continue
        _first_story = None
        for _i, _n in enumerate(_nl):
            if isinstance(_n, dict) and _n.get("type") in ("dialogue", "narration"):
                _first_story = _i
                break
        _keep = []
        for _i, _n in enumerate(_nl):
            if isinstance(_n, dict) and _n.get("type") == "minigame" \
                    and _n.get("setup") \
                    and not (_n.get("win") or _n.get("lose")) \
                    and (_first_story is None or _i < _first_story):
                continue                       # header-def duplicate: drop it
            _keep.append(_n)
        if len(_keep) != len(_nl):
            _b["nodes"] = _keep

    # Register skill-check FAIL sections as runtime-reachable entries. A skill
    # minigame's failure screen (Swim's "She has failed.") is routed to by the
    # host VM's skill loop, not a static jump, so it surfaces as an orphaned
    # section. resolve_checkpoints flagged the byte pattern (a section body right
    # after a `1f63 CHECKPOINT ; goto_scene`); here, with the full segment graph,
    # we add an `enter_at_section` dispatch binding for each such section that is
    # ACTUALLY orphaned (nothing else routes to it). The orphan gate is what keeps
    # this from touching an already-reachable post-checkpoint section (e.g.
    # Halloween's "now playing as Zoe" POV hand-off, which a normal edge reaches).
    _fail_offs = []
    for _sc in doc["scenes"]:
        _ck = (_sc.get("section_dispatch") or {}).get("checkpoints") or {}
        for _fo_off in _ck.get("fail_sections") or []:
            _fail_offs.append((_sc.get("scene"), int(_fo_off)))
        # fail_sections is an internal signal for this pass only; drop it from the
        # emitted checkpoint metadata so the output is unchanged for scenes that
        # have no orphaned fail screen.
        if "fail_sections" in _ck:
            _ck.pop("fail_sections", None)
        # If the checkpoints dict now carries no real 0x63 data (only the empty
        # replays/sets stubs that the simulator-driven fail_sections attach adds),
        # remove it so scenes without checkpoints keep their original null value.
        _sd_here = _sc.get("section_dispatch")
        if isinstance(_sd_here, dict) and isinstance(_sd_here.get("checkpoints"), dict) \
                and not _sd_here["checkpoints"].get("replays") \
                and not _sd_here["checkpoints"].get("sets"):
            _sd_here.pop("checkpoints", None)
    if _fail_offs:
        def _norm2(x):
            return x.replace(".json", "") if isinstance(x, str) else x

        def _edges2(_b):
            _o = []
            for _n in _b.get("nodes") or []:
                if not isinstance(_n, dict):
                    continue
                _t = _n.get("type")
                if _t in ("next", "checkpoint_replay"):
                    _o.append(_norm2(_n.get("next")))
                elif _t == "goto_scene":
                    _o += [_norm2(_n.get("file")), _norm2(_n.get("to_scene"))]
                elif _t == "gate":
                    _o += [_norm2(_n.get("then")), _norm2(_n.get("else"))]
                elif _t == "choice":
                    _o += [_norm2(o.get("next")) for o in _n.get("options") or []]
                    _o.append(_norm2(_n.get("after")))
                elif _t == "minigame":
                    _o += [_norm2(_n.get("win")), _norm2(_n.get("lose"))]
                elif _t == "dispatch":
                    _o += [_norm2(a.get("scene")) for a in _n.get("arms") or []]
                    _o.append(_norm2(_n.get("default")))
                elif _t == "random":
                    _o += [_norm2(o.get("scene")) for o in _n.get("options") or []]
                    _o.append(_norm2(_n.get("exhausted")))
            return [x for x in _o if x]

        _incoming2 = set()
        for _b in files.values():
            _incoming2.update(_edges2(_b))
        # Also treat sections already covered by an existing dispatch binding as
        # reachable: those enter via the section register during normal play, so
        # a resume-after-goto that lands on one is not a genuine orphan (this keeps
        # the general simulator pass from adding redundant bindings to episodes
        # whose sections a normal dispatch already reaches, e.g. As Time Goes By).
        _already_bound = set()
        for _sdv in (index.get("section_dispatch") or {}).values():
            for _bb in _sdv.get("bindings") or []:
                if _bb.get("section_offset") is not None:
                    try:
                        _already_bound.add(int(_bb["section_offset"]))
                    except (TypeError, ValueError):
                        pass
        for _scene_num, _off in _fail_offs:
            # find the CLOSEST orphaned segment that opens at this resume offset.
            # Iterating by dict order can pick a farther segment (or one that is
            # already reached); rank by distance and require the segment be
            # orphaned (nothing in the graph routes to it).
            _cands = []
            for _nm, _b in files.items():
                _m = re.match(r"s(\d+)", _nm or "")
                if not _m or _m.group(1) != str(_scene_num):
                    continue
                if _nm in _incoming2:
                    continue                # already reachable -> not a resume
                _offs = [int(x["offset"]) for x in _b.get("nodes") or []
                         if isinstance(x, dict) and x.get("offset") is not None
                         and str(x["offset"]).lstrip("-").isdigit()]
                if not _offs:
                    continue
                _start = min(_offs)
                # skip a segment that already opens at a dispatch-binding target
                # (reachable via the section register during normal play)
                if any(abs(_start - _bo) <= 24 for _bo in _already_bound):
                    continue
                if abs(_start - _off) <= 24:
                    _cands.append((abs(_start - _off), _nm, _start))
            if not _cands:
                continue                    # already reachable, or not found
            _cands.sort()
            _seg = (_cands[0][1], _cands[0][2])
            # add an enter_at_section dispatch binding so the runtime (and the
            # reachability checker) treat this fail screen as a section entry
            _sd = index.setdefault("section_dispatch", {})
            _entry = _sd.setdefault(str(_scene_num), {
                "register": "1001", "valid_values": [], "bindings": []})
            _entry.setdefault("bindings", []).append({
                "value": None, "section_offset": _seg[1],
                "enter_at_section": True, "source": "checkpoint_fail",
                "_note": ("Skill-check failure screen. Reached by the host VM's "
                          "skill-loop exhaustion (not a static jump): the section "
                          "body immediately following a 1f63 checkpoint + "
                          "goto_scene. Registered as a runtime section entry "
                          "because it is otherwise unreachable.")})

    # Register conditional-branch ALTERNATE sections that are orphaned. A 0x2b
    # gate whose fall-through exits via goto_scene leaves its jump-target section
    # (the "continue" arm, e.g. Wrong Side of Town's journey/fight minigame)
    # unreached, because the segmenter follows the goto as the segment's only
    # exit. With the full graph, register each flagged alternate section as a
    # runtime section entry IFF it is actually orphaned (nothing else routes to
    # it) -- the orphan gate keeps this from touching branches that normal
    # segmentation already wires (e.g. Making Some Dough's, which stay 0-orphan).
    _alt_offs = []
    for _sc in doc["scenes"]:
        _sd_alt = _sc.get("section_dispatch") or {}
        for _ao in _sd_alt.get("branch_alt_sections") or []:
            _alt_offs.append((_sc.get("scene"), int(_ao)))
        if "branch_alt_sections" in _sd_alt:
            _sd_alt.pop("branch_alt_sections", None)
    if _alt_offs:
        def _norm3(x):
            return x.replace(".json", "") if isinstance(x, str) else x

        def _edges3(_b):
            _o = []
            for _n in _b.get("nodes") or []:
                if not isinstance(_n, dict):
                    continue
                _t = _n.get("type")
                if _t in ("next", "checkpoint_replay"):
                    _o.append(_norm3(_n.get("next")))
                elif _t == "goto_scene":
                    _o += [_norm3(_n.get("file")), _norm3(_n.get("to_scene"))]
                elif _t == "gate":
                    _o += [_norm3(_n.get("then")), _norm3(_n.get("else"))]
                elif _t == "choice":
                    _o += [_norm3(o.get("next")) for o in _n.get("options") or []]
                elif _t == "minigame":
                    _o += [_norm3(_n.get("win")), _norm3(_n.get("lose"))]
                elif _t == "dispatch":
                    _o += [_norm3(a.get("scene")) for a in _n.get("arms") or []]
                    _o.append(_norm3(_n.get("default")))
                elif _t == "random":
                    _o += [_norm3(o.get("scene")) for o in _n.get("options") or []]
                    _o.append(_norm3(_n.get("exhausted")))
            return [x for x in _o if x]

        _incoming3 = set()
        for _b in files.values():
            _incoming3.update(_edges3(_b))
        # Existing section-dispatch bindings and scene entries are also roots (a
        # segment reached only by a dispatch value has no node edge pointing at
        # it). Fold their target OFFSETS in so an already-bound section isn't
        # mistaken for an orphan (e.g. Making Some Dough's per-day quiz sections).
        _bound_offs = set()
        for _sdv in (index.get("section_dispatch") or {}).values():
            for _bb in _sdv.get("bindings") or []:
                if _bb.get("section_offset") is not None:
                    try:
                        _bound_offs.add(int(_bb["section_offset"]))
                    except (TypeError, ValueError):
                        pass
        for _ev in (index.get("scene_entries") or {}).values():
            _incoming3.add(_norm3(_ev))
        for _scene_num, _off in _alt_offs:
            _seg = None
            _best = None
            for _nm, _b in files.items():
                _m = re.match(r"s(\d+)", _nm or "")
                if not _m or _m.group(1) != str(_scene_num):
                    continue
                _offs = [int(x["offset"]) for x in _b.get("nodes") or []
                         if isinstance(x, dict) and x.get("offset") is not None
                         and str(x["offset"]).lstrip("-").isdigit()]
                if not _offs or _nm in _incoming3:
                    continue
                _start = min(_offs)
                # skip a segment that already opens at a dispatch-binding target
                if any(abs(_start - _bo) <= 40 for _bo in _bound_offs):
                    continue
                # the alternate section's first EMITTED node may sit a little past
                # the raw section-body offset (intervening condition/setup emits
                # nothing); accept the nearest orphaned segment starting at-or-after
                # the flagged offset, within a small window.
                if -8 <= _start - _off <= 200:
                    if _best is None or _start < _best[1]:
                        _best = (_nm, _start)
            if _best is not None:
                _seg = _best
            if _seg is None:
                continue                    # already reachable, or not found
            _sd = index.setdefault("section_dispatch", {})
            _entry = _sd.setdefault(str(_scene_num), {
                "register": "1001", "valid_values": [], "bindings": []})
            _entry.setdefault("bindings", []).append({
                "value": None, "section_offset": _seg[1],
                "enter_at_section": True, "source": "branch_alt",
                "_note": ("Conditional-branch alternate section. Reached by a "
                          "0x2b gate whose fall-through arm exits to another "
                          "scene; this is the continue arm (e.g. a journey/fight "
                          "minigame). Registered as a runtime section entry "
                          "because it is otherwise unreachable.")})

    # Register a dispatch DEFAULT arm that was left orphaned. When the scene entry
    # was redirected to a value-N section past the default arm, the default
    # (fall-through) section -- e.g. Wrong Side of Town's opening combat minigame,
    # which flows into the title section -- has no incoming edge. Wire it as a
    # runtime entry ONLY when actually orphaned; every scene whose default arm is
    # already reached by normal fall-through is untouched.
    _def_offs = []
    for _sc in doc["scenes"]:
        _sd_def = _sc.get("section_dispatch") or {}
        if _sd_def.get("default_arm") is not None:
            _def_offs.append((_sc.get("scene"), int(_sd_def["default_arm"])))
        if "default_arm" in _sd_def:
            _sd_def.pop("default_arm", None)
    if _def_offs:
        def _norm4(x):
            return x.replace(".json", "") if isinstance(x, str) else x

        def _edges4(_b):
            _o = []
            for _n in _b.get("nodes") or []:
                if not isinstance(_n, dict):
                    continue
                _t = _n.get("type")
                if _t in ("next", "checkpoint_replay"):
                    _o.append(_norm4(_n.get("next")))
                elif _t == "goto_scene":
                    _o += [_norm4(_n.get("file")), _norm4(_n.get("to_scene"))]
                elif _t == "gate":
                    _o += [_norm4(_n.get("then")), _norm4(_n.get("else"))]
                elif _t == "choice":
                    _o += [_norm4(o.get("next")) for o in _n.get("options") or []]
                elif _t == "minigame":
                    _o += [_norm4(_n.get("win")), _norm4(_n.get("lose"))]
                elif _t == "dispatch":
                    _o += [_norm4(a.get("scene")) for a in _n.get("arms") or []]
                    _o.append(_norm4(_n.get("default")))
                elif _t == "random":
                    _o += [_norm4(o.get("scene")) for o in _n.get("options") or []]
                    _o.append(_norm4(_n.get("exhausted")))
            return [x for x in _o if x]

        _incoming4 = set()
        for _b in files.values():
            _incoming4.update(_edges4(_b))
        for _ev in (index.get("scene_entries") or {}).values():
            _incoming4.add(_norm4(_ev))
        _bound_offs4 = set()
        for _sdv in (index.get("section_dispatch") or {}).values():
            for _bb in _sdv.get("bindings") or []:
                if _bb.get("section_offset") is not None:
                    try:
                        _bound_offs4.add(int(_bb["section_offset"]))
                    except (TypeError, ValueError):
                        pass
        for _scene_num, _off in _def_offs:
            _seg = None
            for _nm, _b in files.items():
                _m = re.match(r"s(\d+)", _nm or "")
                if not _m or _m.group(1) != str(_scene_num):
                    continue
                if _nm in _incoming4:
                    continue
                _offs = [int(x["offset"]) for x in _b.get("nodes") or []
                         if isinstance(x, dict) and x.get("offset") is not None
                         and str(x["offset"]).lstrip("-").isdigit()]
                if not _offs:
                    continue
                _start = min(_offs)
                # the default-arm body may start a little after the segment's first
                # emitted node (the segment can include the dispatch-table writes).
                if _start - 60 <= _off <= max(_offs) + 4 \
                        and not any(abs(_start - _bo) <= 40 for _bo in _bound_offs4):
                    _seg = (_nm, _start)
                    break
            if _seg is None:
                continue
            _sd = index.setdefault("section_dispatch", {})
            _entry = _sd.setdefault(str(_scene_num), {
                "register": "1001", "valid_values": [], "bindings": []})
            _entry.setdefault("bindings", []).append({
                "value": None, "section_offset": _seg[1],
                "enter_at_section": True, "source": "dispatch_default",
                "_note": ("Dispatch default (fall-through) section. Runs when the "
                          "register matches no case (e.g. an opening combat "
                          "minigame that flows into the title). Registered as a "
                          "runtime section entry because the scene-entry redirect "
                          "left it otherwise unreachable.")})

    # Wire computed-condition gate targets that land on a genuinely-orphaned block
    # in a DIFFERENT segment than the gate. A 0x2b forward skip whose test is
    # computed on the stack (not a bare 1f2d read) is invisible to
    # resolve_var_gates; when it is the only route to a distinct block (the
    # opening fight's loss-narration branch), that block orphans. These gates are
    # the VM's ordinary branch mechanism, so the orphan gate here uses EXACT
    # reachability (the same roots + edge kinds the runtime traverses): a target
    # that lands inside an already-reachable segment, or within the gate's own
    # segment (an in-line conditional), is left untouched -- which is what keeps
    # dough's minigame-internal gates and europe's in-segment romance branches
    # byte-identical.
    _cgt_pairs = []
    for _sc in doc["scenes"]:
        _sd_cg = _sc.get("section_dispatch") or {}
        for _pair in _sd_cg.get("computed_gate_targets") or []:
            _cgt_pairs.append((_sc.get("scene"), int(_pair[0]), int(_pair[1])))
        if "computed_gate_targets" in _sd_cg:
            _sd_cg.pop("computed_gate_targets", None)
    if _cgt_pairs:
        def _norm6(x):
            return x.replace(".json", "") if isinstance(x, str) else x

        def _edges6(_b):
            _o = []
            for _n in _b.get("nodes") or []:
                if not isinstance(_n, dict):
                    continue
                _t = _n.get("type")
                if _t == "next":
                    _o.append(_norm6(_n.get("next")))
                elif _t == "gate":
                    _o += [_norm6(_n.get("then")), _norm6(_n.get("else"))]
                elif _t == "goto_scene":
                    _o += [_norm6(_n.get("file")), _norm6(_n.get("to_scene")),
                           _norm6(_n.get("next"))]
                elif _t == "choice":
                    for _op in _n.get("options") or []:
                        _o.append(_norm6(_op.get("next")))
                    _o.append(_norm6(_n.get("after")))
                elif _t == "minigame":
                    _o += [_norm6(_n.get("win")), _norm6(_n.get("lose"))]
                elif _t == "random":
                    for _op in _n.get("options") or []:
                        _o.append(_norm6(_op.get("scene")))
                    _o.append(_norm6(_n.get("exhausted")))
                elif _t == "dispatch":
                    for _a in _n.get("arms") or []:
                        _o.append(_norm6(_a.get("scene")))
                    _o.append(_norm6(_n.get("default")))
                for _k in ("then", "else"):
                    _v = _n.get(_k)
                    if isinstance(_v, dict) and _v.get("type") == "gate":
                        _o += [_norm6(_v.get("then")), _norm6(_v.get("else"))]
            return [x for x in _o if isinstance(x, str) and x]

        # segment offset ranges
        _rng6 = {}
        for _nm, _b in files.items():
            _os = [int(x["offset"]) for x in _b.get("nodes") or []
                   if isinstance(x, dict) and x.get("offset") is not None
                   and str(x["offset"]).lstrip("-").isdigit()]
            if _os:
                _rng6[_nm] = (min(_os), max(_os))

        def _seg_for6(_off, _scn):
            _pref = re.compile(r"^s%s(_|$)" % _scn)
            _cont = _after = _al = None
            for _nm, (_lo, _hi) in _rng6.items():
                if not _pref.match(_nm):
                    continue
                if _lo <= _off <= _hi:
                    _cont = _nm
                elif _lo >= _off and (_al is None or _lo < _al):
                    _after, _al = _nm, _lo
            if _cont:
                return _cont
            return _after if (_after and _al - _off <= 64) else None

        def _seg_before6(_off, _scn):
            _pref = re.compile(r"^s%s(_|$)" % _scn)
            _best = _gap = None
            for _nm, (_lo, _hi) in _rng6.items():
                if not _pref.match(_nm):
                    continue
                if _hi < _off and 0 <= _off - _hi <= 12 \
                        and (_gap is None or _off - _hi < _gap):
                    _best, _gap = _nm, _off - _hi
            return _best

        def _compute_reach():
            _roots = []
            _e0 = _norm6(index.get("entry"))
            if _e0:
                _roots.append(_e0)
            for _ev in (index.get("scene_entries") or {}).values():
                _roots.append(_norm6(_ev))
            for _scn, _sdv in (index.get("section_dispatch") or {}).items():
                if not isinstance(_sdv, dict):
                    continue
                for _bb in _sdv.get("bindings") or []:
                    if _bb.get("section_offset") is not None:
                        _o2 = int(_bb["section_offset"])
                        _nm = _seg_for6(_o2, _scn)
                        if _nm:
                            _roots.append(_nm)
                        _nb = _seg_before6(_o2, _scn)
                        if _nb:
                            _roots.append(_nb)
            _roots = [r for r in _roots if r in files]
            _seen = set()
            _stk = list(_roots)
            while _stk:
                _x = _stk.pop()
                if _x in _seen:
                    continue
                _seen.add(_x)
                _stk += [e for e in _edges6(files.get(_x, {})) if e in files]
            return _seen

        _bound_offs6 = set()
        for _sdv in (index.get("section_dispatch") or {}).values():
            for _bb in _sdv.get("bindings") or []:
                if _bb.get("section_offset") is not None:
                    try:
                        _bound_offs6.add(int(_bb["section_offset"]))
                    except (TypeError, ValueError):
                        pass

        _added_cg = set()
        for _scene_num, _gate_off, _tgt in _cgt_pairs:
            # recompute reachability each time so a target wired earlier can help
            # reach later ones (chained loss branches).
            _reach = _compute_reach()
            _gate_seg = _seg_for6(_gate_off, _scene_num)
            _seg = None
            for _nm, _b in files.items():
                _m = re.match(r"s(\d+)", _nm or "")
                if not _m or _m.group(1) != str(_scene_num):
                    continue
                if _nm in _reach or _nm == _gate_seg or _nm in _added_cg:
                    continue
                _nds = [x for x in _b.get("nodes") or [] if isinstance(x, dict)]
                _offs = [int(x["offset"]) for x in _nds
                         if x.get("offset") is not None
                         and str(x["offset"]).lstrip("-").isdigit()]
                if not _offs:
                    continue
                _start = min(_offs)
                # Require a REAL content block: several spoken/narration lines
                # (a win/lose loss branch is substantial dialogue). Transient
                # mid-build fragments and pure-setup blocks are skipped, so the
                # match can't latch onto a segment that later merges away.
                _content = sum(1 for x in _nds
                               if x.get("type") in ("dialogue", "narration"))
                if _content < 3:
                    continue
                # the gate target must land essentially AT the block's first node
                # (a short setup gap is allowed but not a whole other block).
                if _start - 24 <= _tgt <= _start + 4 \
                        and not any(abs(_start - _bo) <= 40 for _bo in _bound_offs6):
                    _seg = (_nm, _start)
                    break
            if _seg is None:
                continue
            _added_cg.add(_seg[0])
            _sd = index.setdefault("section_dispatch", {})
            _entry = _sd.setdefault(str(_scene_num), {
                "register": "1001", "valid_values": [], "bindings": []})
            _entry.setdefault("bindings", []).append({
                "value": None, "section_offset": _seg[1],
                "enter_at_section": True, "source": "computed_gate",
                "_note": ("Computed-condition branch target. Reached by a 0x2b "
                          "forward gate whose test is computed on the stack (not "
                          "a bare variable read), e.g. an opening fight's win/lose "
                          "loss branch. Registered as a runtime section entry "
                          "because it lands on an otherwise-unreachable block.")})

    # Wire LINEAR-SEP section chains. Consecutive sections that fall through a
    # bare SEP into the next section (no goto/branch) have no explicit edge, so
    # the following section is orphaned even though the VM plays it next. For each
    # recorded chain, add a `next` edge from the segment that ENDS at the SEP to
    # the segment that BEGINS the next section -- but only when that target is
    # otherwise unreachable, so episodes already wired by dispatch/branch edges
    # (and their own linear chains, which normal flow already reaches) are byte-
    # identical. Iterates to a fixpoint so a chain reached via an earlier link can
    # carry flow into later ones.
    _lsc_pairs = []
    for _sc in doc["scenes"]:
        for _a, _b in (_sc.get("_linear_sep_chains") or []):
            _lsc_pairs.append((_sc.get("scene"), int(_a), int(_b)))
        if "_linear_sep_chains" in _sc:
            _sc.pop("_linear_sep_chains", None)
    if _lsc_pairs:
        def _norm7(x):
            return x.replace(".json", "") if isinstance(x, str) else x

        def _edges7(_b):
            _o = []
            for _n in _b.get("nodes") or []:
                if not isinstance(_n, dict):
                    continue
                _t = _n.get("type")
                if _t in ("next", "checkpoint_replay"):
                    _o.append(_norm7(_n.get("next")))
                elif _t == "goto_scene":
                    _o += [_norm7(_n.get("file")), _norm7(_n.get("to_scene"))]
                elif _t == "gate":
                    _o += [_norm7(_n.get("then")), _norm7(_n.get("else"))]
                elif _t == "choice":
                    _o += [_norm7(o.get("next")) for o in _n.get("options") or []]
                    _o.append(_norm7(_n.get("after")))
                elif _t == "minigame":
                    _o += [_norm7(_n.get("win")), _norm7(_n.get("lose"))]
                elif _t == "dispatch":
                    _o += [_norm7(a.get("scene")) for a in _n.get("arms") or []]
                    _o.append(_norm7(_n.get("default")))
                elif _t == "random":
                    _o += [_norm7(o.get("scene")) for o in _n.get("options") or []]
                    _o.append(_norm7(_n.get("exhausted")))
            return [x for x in _o if x]

        _rng7 = {}
        for _nm, _b in files.items():
            _os = [int(x["offset"]) for x in _b.get("nodes") or []
                   if isinstance(x, dict) and x.get("offset") is not None
                   and str(x["offset"]).lstrip("-").isdigit()]
            if _os:
                _rng7[_nm] = (min(_os), max(_os))

        def _seg_start_at7(_off, _scn):
            # segment whose FIRST node opens at (or just past) this body offset
            _pref = re.compile(r"^s%s(_|$)" % _scn)
            _best = _gap = None
            for _nm, (_lo, _hi) in _rng7.items():
                if not _pref.match(_nm):
                    continue
                if -8 <= _lo - _off <= 220 and (_gap is None or _lo < _best[1]):
                    _best, _gap = (_nm, _lo), _lo
            return _best[0] if _best else None

        def _seg_end_before7(_off, _scn):
            # segment whose LAST node sits just before this SEP/marker offset
            _pref = re.compile(r"^s%s(_|$)" % _scn)
            _best = _gap = None
            for _nm, (_lo, _hi) in _rng7.items():
                if not _pref.match(_nm):
                    continue
                if _hi < _off and 0 <= _off - _hi <= 60 \
                        and (_gap is None or _off - _hi < _gap):
                    _best, _gap = _nm, _off - _hi
            return _best

        def _reach7():
            _roots = []
            _e0 = _norm7(index.get("entry"))
            if _e0:
                _roots.append(_e0)
            for _ev in (index.get("scene_entries") or {}).values():
                _roots.append(_norm7(_ev))
            for _scn, _sdv in (index.get("section_dispatch") or {}).items():
                if not isinstance(_sdv, dict):
                    continue
                for _bb in _sdv.get("bindings") or []:
                    if _bb.get("section_offset") is not None:
                        _nm = _seg_start_at7(int(_bb["section_offset"]), _scn)
                        if _nm:
                            _roots.append(_nm)
            _roots = [r for r in _roots if r in files]
            _seen, _stk = set(), list(_roots)
            while _stk:
                _x = _stk.pop()
                if _x in _seen:
                    continue
                _seen.add(_x)
                _stk += [e for e in _edges7(files.get(_x, {})) if e in files]
            return _seen

        # apply to a fixpoint (a link can make a later chain's source reachable)
        for _pass in range(6):
            _reach = _reach7()
            _changed = False
            for _scn, _a, _b in _lsc_pairs:
                # the source is the segment ENDING just before the target's
                # section marker (b-4); the chain's own `a` offset may sit inside
                # a larger segment (choices split a section into several segments,
                # and it's the LAST of them that falls through).
                _src = _seg_end_before7(_b - 4, _scn)
                _dst = _seg_start_at7(_b, _scn)
                if not _src or not _dst or _src == _dst:
                    continue
                if _dst in _reach:
                    continue                # already reachable -> leave untouched
                _sb = files.get(_src)
                if not _sb:
                    continue
                _nodes = _sb.get("nodes") or []
                # A section that falls through a bare SEP into the next section
                # is often capped with a synthetic `end` (the segmenter read the
                # SEP as a terminal). That `end` is the mis-inference we're here
                # to correct: replace it with the fall-through `next`. But never
                # override a REAL terminal (choice/gate/goto/next/minigame/etc.),
                # so already-wired sections stay byte-identical.
                _real_term = any(isinstance(_n, dict) and _n.get("type") in
                                 ("next", "choice", "gate", "goto_scene",
                                  "minigame", "dispatch", "random")
                                 for _n in _nodes)
                if _real_term:
                    continue
                _end_idx = next((_ix for _ix, _n in enumerate(_nodes)
                                 if isinstance(_n, dict)
                                 and _n.get("type") == "end"), None)
                _edge = {"type": "next", "next": _dst, "source": "linear_sep"}
                if _end_idx is not None:
                    _nodes[_end_idx] = _edge
                else:
                    _nodes.append(_edge)
                _sb["nodes"] = _nodes
                _changed = True
            if not _changed:
                break

    return files, index


def export_story_dir(doc, out_dir, var_gates=None):
    """Write the episode as a folder of small per-segment JSON files plus index.json
    (the same segments as build_story_segments, one file each)."""
    import json as _json
    files, index = build_story_segments(doc, var_gates)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, body in files.items():
        (out / (fname + ".json")).write_text(
            _json.dumps(restore_int_values(stringify_numbers(body)),
                        indent=2, ensure_ascii=False))
    (out / "index.json").write_text(
        _json.dumps(restore_int_values(stringify_numbers(index)),
                    indent=2, ensure_ascii=False))
    return len(files) + 1


def annotate_pov(doc):
    """Mark the *current* POV through a multi-POV story graph.

    Episodes that hand the camera between several playable leads announce each switch
    with a card ("You are now playing as Cameron."). Without that, every playable
    character reads as "main" at all times. This walks the graph from `entry` and
    records who is in control at each point, so a runtime can place the current main
    on the left and everyone else on the right (the game's layout convention):

      * each switch card node gets   set_pov: "<character>"
      * each reachable segment gets  pov: "<character in control>"
      * each dialogue node gets      side: "left" (speaker is the current POV) | "right"

    Only `main_characters` can hold the POV, so a card naming a non-main avatar (e.g. a
    results-screen minigame) is ignored. Single-POV episodes (no switch cards) are left
    untouched -- their one main is the POV throughout."""
    segments = doc.get("segments")
    mains = doc.get("main_characters") or []
    if not segments or not mains:
        return doc
    mains_lower = {m.lower(): m for m in mains}

    def card_pov(text):
        m = _PLAYING_AS_RE.search(text or "")
        if not m:
            return None
        cand = m.group(1).strip(" .!?,").lower()
        return mains_lower.get(cand) or (mains_lower.get(cand.split()[0]) if cand.split() else None)

    switches = 0
    for seg in segments.values():
        for nd in seg.get("nodes", []):
            if isinstance(nd, dict) and nd.get("type") in ("narration", "title_card", "end_card"):
                who = card_pov(nd.get("text", ""))
                if who:
                    nd["set_pov"] = who
                    switches += 1
    if switches == 0:
        return doc                       # single-POV episode: the lone main is the POV

    def successors(seg):
        out = []
        def add(v):
            if isinstance(v, str) and v in segments:
                out.append(v)
        for nd in seg.get("nodes", []):
            if not isinstance(nd, dict):
                continue
            for key in ("next", "then", "else", "file", "win", "lose", "after"):
                add(nd.get(key))
            for o in nd.get("options", []) or []:
                if isinstance(o, dict):
                    add(o.get("next"))
        return out

    def exit_pov(seg, ep):
        cur = ep
        for nd in seg.get("nodes", []):
            if isinstance(nd, dict) and nd.get("set_pov"):
                cur = nd["set_pov"]
        return cur

    from collections import deque
    entry = doc.get("entry")
    start = mains[0]                      # the default lead you control before any switch
    entry_pov = {entry: start} if entry in segments else {}
    q = deque(entry_pov)
    while q:
        k = q.popleft()
        ep = exit_pov(segments[k], entry_pov[k])
        for nx in successors(segments[k]):
            if nx not in entry_pov:
                entry_pov[nx] = ep
                q.append(nx)

    for k, seg in segments.items():
        cur = entry_pov.get(k)
        seg_pov = cur
        seen_dialogue = False
        for nd in seg.get("nodes", []):
            if not isinstance(nd, dict):
                continue
            if nd.get("set_pov"):
                cur = nd["set_pov"]
                if not seen_dialogue:    # a switch before any line defines this segment's POV
                    seg_pov = cur
            if nd.get("type") == "dialogue":
                seen_dialogue = True
                if cur is not None:
                    nd["side"] = "left" if nd.get("speaker") == cur else "right"
        if seg_pov is not None:
            seg["pov"] = seg_pov
    doc["pov_note"] = (
        "Multi-POV episode. `pov` on a segment is the playable character in control "
        "on entry; a node's `set_pov` switches the current POV mid-stream (these are "
        "the game's 'You are now playing as X' cards). A dialogue node's `side` is "
        "'left' when the speaker is the current POV (the game shows the current main "
        "on the left) and 'right' otherwise. POV is propagated in execution order "
        "from `entry`; only main_characters can hold it.")
    return doc


def build_segmented_doc(doc, var_gates=None, score_tier_thresholds=None):
    """Build the single-document equivalent of the story export: the index metadata
    plus a `segments` map of {segment_name: body}, where each body is one linear run
    of nodes ending in a terminal that names the next segment(s) (choice/gate/next/
    goto_scene/end). Segments link to each other by name, so traversal never scans.
    Every segment reference -- next/then/else/win/lose/file/after, choice options,
    dispatch arms and default, and random options and exhausted -- is stored as the
    bare segment name (no `.json` suffix) so it matches the `segments` keys."""
    files, index = build_story_segments(doc, var_gates)

    def strip(v):
        return v[:-5] if isinstance(v, str) and v.endswith(".json") else v

    segments = {}
    for fname, body in files.items():
        seg = {k: v for k, v in body.items() if k != "file"}   # key IS the name now
        for nd in seg.get("nodes", []):
            if not isinstance(nd, dict):
                continue
            t = nd.get("type")
            if t == "choice":
                for o in nd.get("options", []):
                    if o.get("next"):
                        o["next"] = strip(o["next"])
                if nd.get("after"):
                    nd["after"] = strip(nd["after"])
            elif t == "next" and nd.get("next"):
                nd["next"] = strip(nd["next"])
            elif t == "gate":
                for k in ("then", "else"):
                    if nd.get(k):
                        nd[k] = strip(nd[k])
            elif t == "goto_scene" and nd.get("file"):
                nd["file"] = strip(nd["file"])
            elif t == "minigame":
                for k in ("win", "lose", "after"):
                    if nd.get(k):
                        nd[k] = strip(nd[k])
            elif t == "dispatch":
                for a in nd.get("arms", []):
                    if isinstance(a, dict) and a.get("scene"):
                        a["scene"] = strip(a["scene"])
                if nd.get("default"):
                    nd["default"] = strip(nd["default"])
            elif t == "random":
                for o in nd.get("options", []):
                    if isinstance(o, dict) and o.get("scene"):
                        o["scene"] = strip(o["scene"])
                if nd.get("exhausted"):
                    nd["exhausted"] = strip(nd["exhausted"])
            # Content nodes (status/dialogue/narration/var_*/music/etc.) may still
            # carry a vestigial `next` pointing to a normalize_graph node id in
            # "<scene>.<index>" form (e.g. "1.8"). Inside a segment, nodes already
            # play top-to-bottom, and node ids are stripped from the output, so
            # such a pointer is unresolvable and meaningless -- drop it. Real
            # inter-segment links live on the segment's terminal (a `next`-typed
            # node, choice, gate, goto_scene, minigame, dispatch or random), which
            # the branches above have already rewritten to segment names.
            elif t not in ("next", "goto_scene") and isinstance(nd.get("next"), str) \
                    and re.match(r"^\d+\.\d", nd["next"]):
                nd.pop("next", None)
        segments[fname] = seg

    out = {k: v for k, v in index.items() if k != "files"}
    if out.get("entry"):
        out["entry"] = strip(out["entry"])
    out["scene_entries"] = {k: strip(v) for k, v in (index.get("scene_entries") or {}).items()}
    # Prune dead emissions: segments that nothing references (not the entry, not a
    # scene entry, not named by any terminal) AND that carry no player-visible
    # content. These are stray byproducts of choice/gate splitting -- empty option
    # branches that only {end}, or duplicate guard nodes -- byte-grounded but
    # unreachable and contentless, so dropping them removes pure noise. Any segment
    # with real dialogue/narration/title/choice/minigame is always kept, even if
    # it is currently unreachable (e.g. dispatch-only content), so no story is lost.
    _CONTENT = {"dialogue", "narration", "title_card", "end_card", "choice", "minigame"}

    def _has_content(seg):
        return any(isinstance(nd, dict) and nd.get("type") in _CONTENT
                   for nd in seg.get("nodes", []))

    def _is_stray_fork(seg):
        # A segment that is ONLY a minigame fork node (no intro dialogue/narration
        # of its own) and that nothing references is a duplicate the generic
        # minigame-fork detector split out at a byte offset already covered by a
        # real intro segment's wired minigame (e.g. Making Some Dough's s4_11: a
        # bare bake fork at a Day-2 offset that the bakery pass already emits inside
        # s4_07). Its win/lose targets are reached through the real flow, so it is
        # dead weight. A genuine minigame always sits after its intro lines in the
        # same segment, so a lone, unreferenced minigame node is never real content.
        nds = [n for n in seg.get("nodes", []) if isinstance(n, dict)]
        return len(nds) == 1 and nds[0].get("type") == "minigame"

    # Prune iteratively to a fixpoint: removing a dead segment can leave a stub
    # that only IT referenced (e.g. Wrong Side of Town's s3_35_lose, pointed at
    # only by the empty gate segment s3_33 that is itself pruned). A single pass
    # would keep such a stub "referenced" by a segment about to vanish, so repeat
    # until no more removals. Content-bearing segments are always kept.
    #
    # A segment reached only as a section-dispatch LANDING (the runtime enters it
    # by byte offset via the var-dispatch, with no static incoming edge) has no
    # entry in the reference scan, so protect those offsets explicitly -- e.g.
    # Making Some Dough's s4_16_else, a lone background node the value-22 dispatch
    # lands just after. Match a binding offset to the segment that contains it, or
    # (for a binding that lands in the inter-section gap) the segment ending just
    # before it, exactly as the runtime / validator resolves the landing.
    def _seg_ranges():
        _r = {}
        for _nm, _sb in segments.items():
            _os = [int(x["offset"]) for x in _sb.get("nodes", [])
                   if isinstance(x, dict) and x.get("offset") is not None
                   and str(x["offset"]).lstrip("-").isdigit()]
            if _os:
                _r[_nm] = (min(_os), max(_os))
        return _r

    def _dispatch_landings():
        _r = _seg_ranges()
        _land = set()
        for _scn, _sd in (out.get("section_dispatch") or {}).items():
            if not isinstance(_sd, dict):
                continue
            _pref = re.compile(r"^s%s(_|$)" % _scn)
            for _b in _sd.get("bindings", []) or []:
                if _b.get("section_offset") is None:
                    continue
                try:
                    _off = int(_b["section_offset"])
                except (TypeError, ValueError):
                    continue
                # containing segment
                _cont = _after = _al = None
                _before = _bgap = None
                for _nm, (_lo, _hi) in _r.items():
                    if not _pref.match(_nm):
                        continue
                    if _lo <= _off <= _hi:
                        _cont = _nm
                    elif _lo >= _off and (_al is None or _lo < _al):
                        _after, _al = _nm, _lo
                    if _hi < _off and 0 <= _off - _hi <= 12 \
                            and (_bgap is None or _off - _hi < _bgap):
                        _before, _bgap = _nm, _off - _hi
                if _cont:
                    _land.add(_cont)
                elif _after and _al - _off <= 64:
                    _land.add(_after)
                # Also protect a just-before segment (a true inter-section-gap
                # landing the runtime enters by byte offset, e.g. Making Some
                # Dough's s4_16_else, a lone background node the value-22 dispatch
                # lands just past). But NOT a stray minigame-fork duplicate that
                # merely sits ahead of a real landing -- those are dead weight the
                # pruner should still remove.
                if _before and not _is_stray_fork(segments.get(_before, {})):
                    _land.add(_before)
        return _land

    while True:
        _referenced = set()
        if out.get("entry"):
            _referenced.add(out["entry"])
        _referenced.update(v for v in out["scene_entries"].values() if v)
        _referenced |= _dispatch_landings()
        for _seg in segments.values():
            for _nd in _seg.get("nodes", []):
                if not isinstance(_nd, dict):
                    continue
                for _kk in ("next", "then", "else", "file", "win", "lose",
                            "after", "default", "exhausted"):
                    if isinstance(_nd.get(_kk), str):
                        _referenced.add(_nd[_kk])
                for _o in _nd.get("options", []) or []:
                    if isinstance(_o, dict):
                        if isinstance(_o.get("next"), str):
                            _referenced.add(_o["next"])
                        if isinstance(_o.get("scene"), str):
                            _referenced.add(_o["scene"])
                for _a in _nd.get("arms", []) or []:
                    if isinstance(_a, dict) and isinstance(_a.get("scene"), str):
                        _referenced.add(_a["scene"])
                for _k2 in ("then", "else"):
                    _vv = _nd.get(_k2)
                    if isinstance(_vv, dict) and _vv.get("type") == "gate":
                        for _k3 in ("then", "else"):
                            if isinstance(_vv.get(_k3), str):
                                _referenced.add(_vv[_k3])
        _dead = [n for n, s in segments.items()
                 if n not in _referenced
                 and (not _has_content(s) or _is_stray_fork(s))]
        if not _dead:
            break
        for _name in _dead:
            del segments[_name]
    out["segments"] = segments

    # DANGLING CHOICE `after` REPAIR.
    # A choice's `after` is its shared post-merge continuation. Occasionally the
    # emitted `after` segment name does not correspond to a real segment (the
    # continuation was empty or got folded into the options' own convergence), so
    # the choice carries an `after` pointing nowhere. When every option instead
    # converges on a single existing segment, repoint `after` there; otherwise drop
    # the stale field. The options keep their own `next`, so playable flow is
    # unaffected either way -- this only removes a broken edge. (A Float Is Born
    # scene 3's second choice pointed `after` at a non-existent `s3_07_after` while
    # both options merged at `s3_01_after`.)
    def _seg_succ_next(_sid):
        for _n in segments.get(_sid, {}).get("nodes", []):
            if isinstance(_n, dict) and isinstance(_n.get("next"), str):
                return _n["next"].replace(".json", "")
        return None

    for _sid, _seg in segments.items():
        for _n in _seg.get("nodes", []):
            if not isinstance(_n, dict) or _n.get("type") != "choice":
                continue
            _aft = _n.get("after")
            if not isinstance(_aft, str):
                continue
            _aftk = _aft.replace(".json", "")
            if _aftk in segments:
                continue
            # the after target is missing -- find where the options converge
            _conv = set()
            for _o in _n.get("options", []) or []:
                if not isinstance(_o, dict):
                    continue
                _nx = _o.get("next")
                if isinstance(_nx, str):
                    _c = _seg_succ_next(_nx.replace(".json", "")) or _nx.replace(".json", "")
                    _conv.add(_c)
            _conv = {c for c in _conv if c in segments}
            if len(_conv) == 1:
                _n["after"] = next(iter(_conv)) + ".json"
            else:
                _n.pop("after", None)

    # The end-of-episode rank screen is a chain of score comparisons in bytecode,
    # each JMPF-guarding one rank narration so the VM shows exactly ONE. The linear
    # decoder emits all of them in a row. Detect a run of 3+ consecutive
    # rank/grade narrations and replace it with a single mutually-exclusive
    # `score_tier` node the runtime evaluates against the score, rather than a
    # flat list the runtime would show all of. (The Tutors / Swim Retreat /
    # Halloween Dance end screens are the cases this repairs.) The rank texts are
    # taken in order from the already-decoded nodes; they are mutually exclusive,
    # the last being the top/default tier shown when no lower tier's test passes.
    def _is_rank_line(nd):
        if not isinstance(nd, dict) or nd.get("type") not in ("narration", "status"):
            return False
        t = (nd.get("text") or "").lower()
        return "rank for this episode" in t or "grade for this episode" in t

    def _is_perfect_line(nd):
        # the "You got the `perfect` score!" banner accompanying the top grade
        if not isinstance(nd, dict) or nd.get("type") not in ("narration", "status"):
            return False
        return "perfect` score" in (nd.get("text") or "").lower() \
            or "perfect score" in (nd.get("text") or "").lower()

    _thr = score_tier_thresholds or {}
    for _sid, _seg in segments.items():
        _nodes = _seg.get("nodes") or []
        _i = 0
        while _i < len(_nodes):
            if _is_rank_line(_nodes[_i]):
                # extend over consecutive rank lines and the interleaved perfect-
                # score banner that belongs to the top grade
                _j = _i
                while _j < len(_nodes) and (_is_rank_line(_nodes[_j])
                                            or _is_perfect_line(_nodes[_j])):
                    _j += 1
                _run = _nodes[_i:_j]
                _ranks = [n for n in _run if _is_rank_line(n)]
                if len(_ranks) >= 3:
                    # A rank with a byte-derived threshold is a gated tier; the
                    # rank(s) without one are the fall-through top grade (default),
                    # shown when no lower tier's score test passed. The perfect-
                    # score banner rides along with the default.
                    _tiers, _default_ranks = [], []
                    for nd in _ranks:
                        _txt = nd.get("text") or ""
                        if _txt in _thr:
                            _tiers.append({
                                "rank": _txt,
                                "op": _thr[_txt]["op"],
                                "threshold": _thr[_txt]["threshold"]})
                        else:
                            _default_ranks.append(_txt)
                    _perfect = [n.get("text") for n in _run if _is_perfect_line(n)]
                    _default = {}
                    if _perfect:
                        _default["banner"] = _perfect[0]
                    if _default_ranks:
                        _default["rank"] = _default_ranks[-1]
                    _tier_node = {
                        "type": "score_tier",
                        "var": "2000",
                        "tiers": _tiers,
                        "default": _default,
                        "note": ("Mutually-exclusive end-of-episode rank. In "
                                 "bytecode each tier is a score comparison whose "
                                 "JMPF skips the other ranks, so the engine shows "
                                 "exactly one -- the first tier whose test passes, "
                                 "else the default (top grade). The linear decode "
                                 "emitted every rank; this restores the single "
                                 "gated choice."),
                    }
                    _off = _run[0].get("offset")
                    if _off is not None:
                        _tier_node["offset"] = _off
                    _nodes[_i:_j] = [_tier_node]
                    _i += 1
                    continue
            _i += 1


    # A word-match minigame whose data block sits at the very top of a scene (its
    # 0x47 play-point precedes the scene's first dialogue) is surfaced by the
    # decoder as a standalone segment at the scene boundary. When that segment
    # ends up unreferenced -- nothing in the flow routes to it -- it is a detached
    # scene-opening minigame, not dead content: the game plays first, then the
    # scene proper begins. Reconnect it as the scene entry, flowing into the
    # segment that was previously the entry. (A Float Is Born's scene 3 paint
    # minigame is the case this repairs; its "Paint the right colors" round plays
    # before the "Well, I guess I need to get to class" opening line.)
    _entries = out.get("scene_entries") or {}
    _referenced_seg = set()
    for _s in segments.values():
        for _n in _s.get("nodes", []):
            if not isinstance(_n, dict):
                continue
            for _k in ("next", "then", "else", "win", "lose", "after", "default"):
                if isinstance(_n.get(_k), str):
                    _referenced_seg.add(_n[_k].replace(".json", ""))
            for _o in _n.get("options", []) or []:
                if isinstance(_o, dict) and isinstance(_o.get("next"), str):
                    _referenced_seg.add(_o["next"].replace(".json", ""))
    for _nm, _s in list(segments.items()):
        _nodes = [_n for _n in _s.get("nodes", []) if isinstance(_n, dict)]
        # shape: a lone minigame node followed by a `next` (the detached opener)
        if not (len(_nodes) == 2 and _nodes[0].get("type") == "minigame"
                and _nodes[1].get("type") == "next"):
            continue
        if _nm in _referenced_seg:
            continue                    # already in the flow -> leave alone
        _scn = None
        _m = re.match(r"s(\d+)", _nm)
        if _m:
            _scn = _m.group(1)
        _entry = (_entries.get(_scn) or "").replace(".json", "") if _scn else ""
        if not _entry or _entry == _nm or _entry not in segments:
            continue
        # The opener plays before the scene's first line. Rather than rewire every
        # goto that targets the entry, splice the minigame node to the FRONT of the
        # entry segment so anything that reaches the scene plays it first. Then drop
        # the now-empty detached segment.
        _mg_node = dict(_nodes[0])
        _entry_nodes = segments[_entry].get("nodes", [])
        if _entry_nodes and isinstance(_entry_nodes[0], dict) \
                and _entry_nodes[0].get("type") == "minigame":
            continue                    # already has an opening minigame
        segments[_entry]["nodes"] = [_mg_node] + _entry_nodes
        del segments[_nm]

    # next episode") is a hard terminal: in the bytecode it is followed by an
    # unconditional jump that skips PAST the rest of the scene's gate cascade to
    # the episode terminator, so nothing plays after it. When such an arm was
    # instead linked by physical adjacency to a LATER gate branch (a sibling arm
    # it should never reach -- e.g. A Float Is Born's perfect-score bonus scene,
    # gated on var2000 == 14, must not fall through into the var2000 >= 6 "downer"
    # ending), replace that fall-through with a clean end. Scoped tightly: only
    # when the trailing `next` points at a gate-arm target (a then/else branch),
    # never a normal continuation such as a shared survey segment.
    _gate_arm_targets = set()
    for _s in segments.values():
        for _n in _s.get("nodes", []):
            if isinstance(_n, dict) and _n.get("type") == "gate":
                for _k in ("then", "else"):
                    _t = _n.get(_k)
                    if isinstance(_t, str):
                        _gate_arm_targets.add(_t.replace(".json", ""))
    for _nm, _s in segments.items():
        _nodes = [_n for _n in _s.get("nodes", []) if isinstance(_n, dict)]
        if len(_nodes) >= 2 and _nodes[-1].get("type") == "next" \
                and _nodes[-2].get("type") == "end_card":
            _tgt = (_nodes[-1].get("next") or "").replace(".json", "")
            if _tgt in _gate_arm_targets and _tgt != _nm:
                _s["nodes"][-1] = {"type": "end", "source": "credit_card_terminal"}

    # name the bank it draws from. Resolve it to a REAL minigame type from the
    # scene's bank ("build_word" / "pick_word" -- the only playable kinds) and
    # attach that bank's rounds as `setup`, matching the shape used by the
    # already-resolved play points. `kind` stays the fork classification
    # (content_fork / outcome_fork); it is not a minigame type.
    _banks = out.get("scene_minigame_banks") or {}

    def _scene_of_seg(_sid):
        _m = re.match(r"s(\d+)", _sid or "")
        return _m.group(1) if _m else None

    # Word minigames ship in TWO platform variants drawn from the scene's banks:
    # pick_word = mobile, build_word = tablet. A trigger that has both should carry
    # both, with pick_word as the default (mobile). Only one bank present -> single
    # variant. `kind` remains the fork classification, never a minigame type.
    _PLAYABLE = ("build_word", "pick_word")
    _PLATFORM = {"pick_word": "mobile", "build_word": "tablet"}

    def _variants_for(_scene):
        _b = _banks.get(str(_scene)) or {}
        return [{"platform": _PLATFORM[k], "bank": k, "rounds": _b[k]}
                for k in ("pick_word", "build_word") if _b.get(k)]

    for _sid, _body in segments.items():
        _nodes = _body.get("nodes") or []
        for _i, _nd in enumerate(_nodes):
            if not isinstance(_nd, dict) or _nd.get("type") != "minigame":
                continue
            _scene = _scene_of_seg(_sid)
            _vars = _variants_for(_scene)
            _is_fork = _nd.get("via") == "word_bank" and not _nd.get("minigame_type")
            _is_flat = bool(_nd.get("words")) and \
                _nd.get("minigame_type") not in _PLAYABLE
            _is_word = _nd.get("minigame_type") in _PLAYABLE
            if not (_is_fork or _is_flat or _is_word) or not _vars:
                continue
            # default variant = mobile (pick_word) when present, else the other
            _default = _vars[0]
            if not _nd.get("minigame_type") or _is_flat:
                _nd["minigame_type"] = _default["bank"]
                _nd["setup"] = {"bank": _default["bank"],
                                "rounds": _default["rounds"]}
                _nd.pop("words", None)   # flat dump superseded by the rounds
            if len(_vars) > 1:
                _nd["variants"] = _vars

    # Some scenes open with a word minigame whose 0x47 play-point precedes the
    # first line, but whose round strings are pushed as 0x1b pairs -- the inline
    # 0x47 handler only reads plain (0x1a) value pushes, so no minigame line is
    # emitted and the scene's bank never gets attached to a node (A Float Is Born
    # scene 2's "Rewire the car!" round). When a scene has a playable bank yet no
    # minigame node anywhere in its segments, synthesize one from the bank and
    # splice it to the front of the scene's entry segment, so it plays first --
    # exactly the shape the pair-decoding produces for the scenes that emit
    # normally. Guarded to scenes with zero existing minigame node, so scenes
    # already covered by the gate / word-bank path are untouched.
    _mg_scenes = set()
    for _sid, _body in segments.items():
        if any(isinstance(_n, dict) and _n.get("type") == "minigame"
               for _n in _body.get("nodes") or []):
            _sc = _scene_of_seg(_sid)
            if _sc:
                _mg_scenes.add(_sc)
    _entries = out.get("scene_entries") or {}
    for _scene, _b in _banks.items():
        if _scene in _mg_scenes:
            continue
        _vars = _variants_for(_scene)
        if not _vars:
            continue
        _entry = (_entries.get(_scene) or "").replace(".json", "")
        if not _entry or _entry not in segments:
            continue
        _default = _vars[0]
        _mg = {"type": "minigame", "minigame_type": _default["bank"],
               "kind": "word-match",
               "setup": {"bank": _default["bank"], "rounds": _default["rounds"]}}
        if len(_vars) > 1:
            _mg["variants"] = _vars
        _enodes = segments[_entry].get("nodes") or []
        if _enodes and isinstance(_enodes[0], dict) \
                and _enodes[0].get("type") == "minigame":
            continue
        segments[_entry]["nodes"] = [_mg] + _enodes

    out["note"] = ("Story as named segments (the single-file form of the story-export "
                   "folder). Each segment is one linear run of nodes walked in order; "
                   "the final node is a typed control node naming the next segment(s): "
                   "choice (option.next / after), next, gate (then/else by var==equals), "
                   "goto_scene (.file = entry segment of the destination scene), "
                   "minigame (win/lose), or end. Start at `entry`; scene_entries maps "
                   "each scene to its first segment.")
    out = annotate_pov(out)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _speak_place_counts(scripts, cast):
    """Per cast-row counts of spoken lines (1f 0d) and on-screen placements (1f 34)."""
    spoke = [0] * len(cast)
    placed = [0] * len(cast)
    for _label, s in scripts:
        if s[:4] != b"kiwi":
            continue
        strings = find_strings(s)
        bc = max((o + len(t) for o, t in strings if len(t) >= 12), default=0)
        p, n, pushes, active = bc, len(s), [], None
        while p < n - 1:
            op = s[p]
            if op in (0x1a, 0x41) and p + 2 < n:
                pushes.append((s[p + 1] << 8) | s[p + 2]); p += 3; continue
            if op == 0x1b and p + 2 < n:
                pushes.append(("pair", s[p + 1], s[p + 2])); p += 3; continue
            if op == 0x1f and p + 1 < n:
                sel = s[p + 1]
                if sel == SET_SPEAKER:
                    spk = next((pk[2] for pk in pushes if isinstance(pk, tuple)), None)
                    if spk is not None:
                        active = spk
                elif sel == DISPLAY_DIALOGUE:
                    spk = next((pk[2] for pk in pushes if isinstance(pk, tuple)), None)
                    if spk is None:
                        spk = next((pk for pk in pushes
                                    if isinstance(pk, int) and pk < len(cast)), None)
                    if spk is None:
                        spk = active
                    if spk is not None and 0 <= spk < len(cast):
                        spoke[spk] += 1
                elif sel == PLACE_SPRITE:
                    ch = next((pk for pk in pushes
                               if isinstance(pk, int) and pk < len(cast)), None)
                    if ch is not None:
                        placed[ch] += 1
                pushes = []; p += 2; continue
            if op in (0x2b, 0x28) and p + 2 < n:
                pushes = []; p += 3; continue
            if op == 0x42 and p + 3 < n:
                pushes = []; p += 4; continue
            p += 1
    return spoke, placed


_PLAYING_AS_RE = re.compile(r"play(?:ing)? as\s+([A-Z][A-Za-z.'\- ]{1,20})")


def _playing_as_povs(scripts, cast):
    """POV characters named by explicit "(now) playing as X" cards, in first-seen order.

    Multi-POV episodes announce each hand-off with a card ("You are now playing as
    Cameron."). The named character is a cast member, so the card text is an exact,
    byte-present POV declaration -- more reliable than the 1f 4a operand it sits with
    (which can index a different row).

    A card inside a minigame/scoring scene (one carrying a minigame bank) names the
    per-minigame avatar, not a story POV -- e.g. a results-screen "now playing as
    Ben" in an episode whose story POVs are Emily/Cameron/Hannah. So the story
    (bank-free) scenes are scanned first, and bank scenes contribute only as a
    fallback when no story scene announces anyone."""
    lower = {c.lower(): c for c in cast if c}

    def scan(want_scene):
        out = []
        for _label, s in scripts:
            if s[:4] != b"kiwi" or not want_scene(s):
                continue
            for _o, t in find_strings(s):
                for m in _PLAYING_AS_RE.finditer(t):
                    cand = m.group(1).strip(" .!?,").lower()
                    hit = (lower.get(cand)
                           or (lower.get(cand.split()[0]) if cand.split() else None))
                    if hit and hit not in out:
                        out.append(hit)
        return out

    story = scan(lambda s: not extract_minigame_banks(s))
    return story or scan(lambda s: True)


def derive_main_characters(scripts, cast, tag_mains, speak_thresh=5):
    """Resolve the playable/POV characters, layering byte signals over tag_pov.

    Priority:
      1. Explicit "(now) playing as X" cards -- authoritative for multi-POV episodes.
         Keep a significant-speaking lead (the default POV you start as, e.g. the
         tutor) and drop a non-speaking placeholder lead (e.g. a cast[0] that never
         talks), then add every announced POV.
      2. The "POV is never sprite-placed" signal -- a frequent speaker that is never
         positioned on screen is the camera/player. Guarded so it only fires when
         placement actually separates POV from the rest AND re-identifies the known
         lead (sound alignment).
      3. Otherwise the tag_pov result is kept unchanged.
    """
    spoke, placed = _speak_place_counts(scripts, cast)
    spoke_by_name = {}
    for i, nm in enumerate(cast):
        if nm:
            spoke_by_name[nm] = spoke_by_name.get(nm, 0) + spoke[i]

    explicit = _playing_as_povs(scripts, cast)
    if explicit:
        out = [m for m in tag_mains if spoke_by_name.get(m, 0) >= speak_thresh]
        for m in explicit:
            if m not in out:
                out.append(m)
        return out

    sig = [i for i in range(len(cast)) if cast[i] and spoke[i] >= speak_thresh]
    placed_sig = [i for i in sig if placed[i] > 0]
    unplaced_sig = [i for i in sig if placed[i] == 0]
    reliable = len(placed_sig) >= 2 and len(placed_sig) >= len(unplaced_sig)
    sig_mains, seen = [], set()
    for i in unplaced_sig:
        if cast[i] not in seen:
            sig_mains.append(cast[i]); seen.add(cast[i])
    consistent = bool(tag_mains) and all(m in sig_mains for m in tag_mains)
    if reliable and consistent:
        out = list(tag_mains)
        for m in sig_mains:
            if m not in out:
                out.append(m)
        return out
    return tag_mains


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode SHS .exp / .kiw files into transcripts.")
    ap.add_argument("input", help="Path to a .exp container or a .kiw script")
    ap.add_argument("-o", "--output", help="Write transcript here (default: stdout)")
    ap.add_argument("--format", choices=["md", "txt", "json"], default="md", help="Transcript format")
    ap.add_argument("--extract-dir", help="For .exp input: write all chunks (script + images) here")
    ap.add_argument("--no-backgrounds", action="store_true", help="Omit background-change markers")
    ap.add_argument("--branches", action="store_true",
                    help="Label choice option-branches (markdown only)")
    ap.add_argument("--cast-from", help="Borrow the cast table from another .kiw (for continuation scenes)")
    ap.add_argument("--cast", help="Comma-separated cast list, overriding any table (e.g. 'Dinah,Taylor,Paula,Linda')")
    ap.add_argument("--overlay", help="Per-episode overlay JSON of observed gameplay logic "
                    "(scene carving, per-option branches, effects, gates) merged by offset")
    ap.add_argument("--story-dir", help="Also write the episode as a folder of small "
                    "per-segment JSON files linked at every choice/gate/goto")
    ap.add_argument("--disasm", action="store_true",
                    help="Print an annotated bytecode listing (one line per VM "
                    "instruction: offset, opcode, mnemonic, decoded operands, and "
                    "notes like resolved jump targets / variable reads / arithmetic) "
                    "instead of a transcript. Respects --output.")
    ap.add_argument("--disasm-range",
                    help="With --disasm, limit output to a byte-offset window "
                    "LO:HI (e.g. 18440:18480). Applies per scene.")
    args = ap.parse_args(argv)

    raw = Path(args.input).read_bytes()

    # Resolve an explicit cast list if the user supplied one.
    cast_override = None
    if args.cast:
        cast_override = [c.strip() for c in args.cast.split(",")]
    elif args.cast_from:
        cast_override = read_cast(Path(args.cast_from).read_bytes())

    # Collect script chunk(s). A container may hold several scripts, which are
    # sequential SCENES of one episode (not alternate branches); the cast table
    # lives in the first scene and is shared by the rest.
    episode = None
    scripts = []  # (label, bytes)
    script_ids, image_ids, audio_ids = set(), set(), set()
    if raw[:5] == ExpArchive.MAGIC:
        arc = ExpArchive(raw)
        out_dir = Path(args.extract_dir) if args.extract_dir else Path(args.input).with_suffix("")
        arc.extract(out_dir)
        episode = arc.episode_title()
        _ep_meta = arc.episode_meta()
        for eid, payload, kind in arc.chunks():
            if kind == "script":
                scripts.append((f"0x{eid:04x}", payload))
                script_ids.add(eid)
            elif kind == "image":
                image_ids.add(eid)
            elif payload[:3] == b"ID3" or payload[:2] == b"\xff\xfb":
                audio_ids.add(eid)      # episode-packaged MP3 track
        print(f"[+] extracted {len(arc.entries)} chunks to {out_dir} "
              f"({len(scripts)} script scene(s))", file=sys.stderr)
        if not scripts:
            print("[!] no script chunk found in archive", file=sys.stderr)
            return 1
    elif raw[:4] == b"kiwi":
        scripts.append((Path(args.input).stem, raw))
    else:
        print("[!] unrecognized file (expected CSPUD .exp or kiwi .kiw)", file=sys.stderr)
        return 1

    # The cast used for SPEAKER lookups comes from the resource table when one
    # exists: hand-decoding confirmed that a line's speaker index is the
    # resource-table row index (each row's name_ref resolves the displayed
    # name). The slot-scan cast is the fallback for table-less scripts. The
    # two coincide when the table has one row per slot name; they diverge when
    # the table carries duplicate-name rows (e.g. the same character listed
    # both as an unnamed 'Girl' and under her real name, different sprites).
    base_cast = cast_override
    sprites_by_idx = None
    if base_cast is None:
        rows = _resource_rows(scripts[0][1])
        if rows and any(r["name"] for r in rows):
            by_idx = {r["idx"]: r for r in rows}
            hi = max(by_idx)
            base_cast = [(by_idx[i]["name"] or "") if i in by_idx else ""
                         for i in range(hi + 1)]
            sprites_by_idx = [by_idx[i]["asset"] if i in by_idx else None
                              for i in range(hi + 1)]
            # A character can occupy TWO rows: a global sprite base and an
            # EPISODE-PACKAGED one (>= EPISODE_ASSET_BASE, a PNG chunk shipped in
            # this .exp -- e.g. Halloween's costumed Kay/Kel at 26000/26005).
            # Speaker lookup resolves a name to its FIRST cast index, which would
            # pick the global row and lose the episode art.
            # Only upgrade when the pairing is UNAMBIGUOUS: exactly one global row
            # and exactly one packaged row for that name. A name with several
            # rows (e.g. As Time Goes By's Matt, who also has the packaged
            # personas "Mr. Hotpants"/"Sir Smoothness") is a costume set, where
            # the packaged art is worn contextually rather than always.
            _glob, _pack = {}, {}
            for _r in rows:
                _nm = _r.get("name")
                if not _nm:
                    continue
                (_pack if _r["asset"] >= EPISODE_ASSET_BASE else _glob)\
                    .setdefault(_nm, []).append(_r["asset"])
            _upgrade = {nm: _pack[nm][0] for nm in _pack
                        if len(_pack[nm]) == 1 and len(_glob.get(nm, [])) == 1}
            for _i, _nm in enumerate(base_cast):
                _cur = sprites_by_idx[_i] if _i < len(sprites_by_idx) else None
                if _nm in _upgrade and _cur is not None \
                        and _cur < EPISODE_ASSET_BASE:
                    sprites_by_idx[_i] = _upgrade[_nm]
    if base_cast is None:
        base_cast = read_cast(scripts[0][1])
    genders = read_genders(scripts[0][1], base_cast)
    # The scene-1 resource table (shared by later scenes) also defines costume
    # rows -- same character name, different sprite asset -- used for disguises
    # and restyles. Compute once and pass to every scene's decode.
    _costume_rows = _resource_rows(scripts[0][1])
    # Dynamic renames (1f2e) are episode-wide: collect from every scene (a rename
    # may be armed in scene 1 but the renamed character speaks in scene 2).
    _renames = {}
    for _lbl, _scdata in scripts:
        _renames.update(_name_renames(_scdata, _costume_rows))
    # Text-template substitutions (1f2e filling $Token placeholders in text). Some
    # tokens are STABLE (bound once -> substitute directly, e.g. $MAN1 -> "Ice
    # Cream Jim", $Player -> protagonist), others are REBOUND runtime variables
    # (e.g. $Antagonist, rebound to 14 different opponents across the fights). A
    # token that is rebound in ANY scene is treated as a runtime variable
    # EVERYWHERE (so a shared combat template is never baked to one name); its
    # per-scene binding schedule is emitted as metadata for the engine.
    _per_scene = {}      # label -> (stable, schedule)
    _label_to_scene = {}  # script label -> scene number (1-indexed, enum order)
    _rebound_tokens = set()
    for _i, (_lbl, _scdata) in enumerate(scripts, 1):
        _label_to_scene[_lbl] = _i
        _st, _sch = _text_substitutions(_scdata, _costume_rows)
        _per_scene[_lbl] = (_st, _sch)
        _rebound_tokens.update(_sch.keys())
    # A token stable in one scene but rebound in another is globally rebound.
    for _lbl, (_st, _sch) in _per_scene.items():
        for _tok in list(_st):
            if _tok in _rebound_tokens:
                _sch.setdefault(_tok, []).append((0, _st.pop(_tok)))
    # Global stable map (safe: these tokens are bound to one value everywhere).
    _text_subs = {}
    for _lbl, (_st, _sch) in _per_scene.items():
        _text_subs.update(_st)

    # Diagnostic disassembly: print the annotated bytecode listing and stop,
    # before the (much heavier) full segment decode. Handles one or many scenes.
    if args.disasm:
        _lo = _hi = None
        if args.disasm_range:
            try:
                _a, _b = args.disasm_range.split(":")
                _lo, _hi = int(_a), int(_b)
            except ValueError:
                print("[!] --disasm-range must look like LO:HI", file=sys.stderr)
                return 1
        _blocks = []
        for _i, (_label, _sc) in enumerate(scripts, 1):
            _lines = disassemble_script(_sc, cast=base_cast)
            if _lo is not None:
                _kept = []
                for _ln in _lines:
                    try:
                        _off = int(_ln.split()[0].lstrip("@"))
                    except (ValueError, IndexError):
                        continue
                    if _lo <= _off <= _hi:
                        _kept.append(_ln)
                _lines = _kept
            _hdr = "===== Scene %d (%s) =====" % (_i, _label)
            _blocks.append(_hdr + "\n" + "\n".join(_lines))
        _dis = "\n\n".join(_blocks)
        if args.output:
            Path(args.output).write_text(_dis + "\n")
            print("[+] wrote disassembly to %s" % args.output, file=sys.stderr)
        else:
            print(_dis)
        return 0

    # The overlay (observed gameplay logic) is loaded up front: some of it
    # steers decoding itself (minigame triggers, custom-background names).
    overlay = json.loads(Path(args.overlay).read_text()) if args.overlay else None
    if overlay and overlay.get("bg_names"):
        KNOWN_BG.update({int(k, 0): v for k, v in overlay["bg_names"].items()})

    all_lines = []
    scene_controls = {}
    minigame_banks = {}
    name_vars = {}
    choice_dispatch = {}
    chapter_heads = {}
    var_gates = {}
    section_dispatch = {}
    minigame_gates = {}
    buildword_scoring = {}
    merge_jumps = {}
    choice_forks = {}
    score_tier_thresholds = {}
    random_selectors = {}
    status_vars = {}
    for i, (label, sc) in enumerate(scripts, 1):
        lines, _ = decode_script(sc, cast=base_cast, episode_title=episode,
                                 script_ids=script_ids, image_ids=image_ids,
                                 audio_ids=audio_ids,
                                 sprites_by_idx=sprites_by_idx,
                                 costume_rows=_costume_rows,
                                 name_renames=_renames,
                                 text_subs=_text_subs)
        scene_controls[label] = scan_control_flow(sc)
        # Score-tier rank cascade: scan the scene for the end-of-episode rank
        # screen (a chain of score comparisons each JMPF-guarding one rank line)
        # and record each rank text's threshold/op, so the builder can collapse
        # the flattened narrations into one mutually-exclusive gated node.
        _casc = resolve_score_tier_cascade(sc)
        if _casc:
            for _ti in _casc["tiers"]:
                score_tier_thresholds[_ti["text"]] = {
                    "op": _ti["op"], "threshold": _ti["threshold"]}
            # also fold in bindings from any rescore cascade in the scene
            for _rt, _rb in (_casc.get("all_thresholds") or {}).items():
                score_tier_thresholds.setdefault(_rt, _rb)
        _s63 = _scan_sep63_terminals(sc)
        if _s63:
            scene_controls[label]["_sep63_terminals"] = _s63
        # Linear SEP section chains: consecutive sections that fall through a bare
        # SEP into the next section (no goto/jump wiring). Recorded per scene; the
        # final binding pass adds a `next` edge for each whose target is orphaned.
        _bc_here = max((o + len(t) for o, t in find_strings(sc)
                        if len(t) >= 12), default=0)
        _lsc = resolve_linear_sep_chains(sc, _bc_here)
        if _lsc:
            scene_controls[label]["_linear_sep_chains"] = _lsc
        merge_jumps[label] = resolve_unconditional_jumps(sc)
        choice_forks[label] = resolve_choice_forks(sc)
        banks = extract_minigame_banks(sc)
        if banks:
            minigame_banks[label] = banks
        choice_dispatch[label] = {c["choice"]: c
                                  for c in resolve_choice_branches(sc)}
        # Minigame gates already resolve their own win/fail scoring; collect their
        # gate offsets so the score-gate splitter skips them and only picks up the
        # win/fail gates nothing else handles (Fallon's intruder fight).
        _mg_here = resolve_minigame_gates(sc)
        _mg_offs = set()
        for _m in (_mg_here or []):
            if isinstance(_m, dict):
                for _k in ("gate", "trigger", "pass_at", "fail_at"):
                    if _m.get(_k) is not None:
                        _mg_offs.add(_m[_k])
        var_gates[label] = (resolve_var_gates(sc)
                            + resolve_score_gates(sc, _mg_offs))
        _svars = resolve_dynamic_status_vars(sc)
        if _svars:
            status_vars[str(i)] = _svars
        _sd = resolve_section_dispatch(sc)
        _casc = resolve_sep_cascade_gates(sc)
        _rand = resolve_random_selector(sc)
        # The academic quiz's per-section question rolls are plain 1-of-N draws
        # (no availability guard). Resolve them too, but exclude any roll offset
        # the ad selector already claims (the Help Ads share the 1f1b opcode but
        # add a "which items remain" guard, and are handled as `random_selectors`).
        _quiz = [r for r in resolve_quiz_random(sc)
                 if not any(r["at"] == s.get("at") for s in (_rand or []))]
        if _sd or _casc or _rand or _quiz:
            _sd = dict(_sd or {"register": None, "valid_values": [], "cases": []})
            if _quiz:
                # Byte-derived question pools for the quiz's random draws (History,
                # Science, English each roll one question of a few). Consumed during
                # segmentation to emit a `random` node per section so the engine
                # draws a question at runtime instead of always playing a fixed one.
                _sd["quiz_random"] = _quiz
            if _rand:
                # A runtime RANDOM section selector (e.g. Making Some Dough's Help
                # Ads: a randomly-chosen not-yet-done helpee each visit, falling
                # through to a "no more ads" body when all are done). Recorded as
                # byte-derived metadata; consumed during segmentation to emit a
                # `random` node whose options/exhausted resolve to segments. The
                # decoder cannot pick one static target for a random draw.
                _sd["random_selectors"] = _rand
                # Byte-derived per-ad done-flag writes (var2005/2006/2007), in
                # bytecode order. Each ad sets its flag on entry to a distinguishing
                # value; segmentation attaches these as per-option `done_when`
                # guards so the engine can skip an already-run ad. A flag write is a
                # `read var200X ; push V ; ADD/const ; write` where 2005<=var<=2007.
                _sd["ad_flag_writes"] = _scan_ad_flag_writes(sc)
            if _casc:
                # Reference-only: the byte-derived nested SEP-cascade mapping
                # (a secondary variable dispatch inside a top-level case body,
                # e.g. Making Some Dough's var2002 quiz-day / var2004 bakery-day
                # selectors). This documents which variable value routes to which
                # section body -- the "value -> day" mapping that lives in the
                # kiwi bytes. It is NOT consumed by segmentation (the arms are
                # already reachable through the quiz's in-flow minigame outcome
                # routing); it is emitted so a runtime / overlay author has the
                # exact byte-derived selector table without re-deriving it.
                _sd["cascades"] = [
                    {"at": c["at"], "var": c["var"],
                     "note": ("Nested section selector: var==equals routes to "
                              "the section body at section_offset. Arm ordinal "
                              "equals the compared value; targets are byte-"
                              "derived (fold-instruction SEP jumps)."),
                     "cases": c["cases"]}
                    for c in _casc]
            section_dispatch[label] = _sd
        # Byte-derived checkpoint records (opcode 0x63): the "replay from
        # checkpoint" retry option jumps back to the last passed checkpoint, and
        # the checkpoint SET zeroes the run counters (var2001, often var2000). The
        # decoder otherwise leaves the retry option pointing at an empty `end` stub
        # because the target is a raw SEP instruction-count the jump resolver skips.
        # Stash the resolved targets + resets so segmentation can wire the stub.
        _ckpt = _scan_checkpoint_replays(sc)
        if _ckpt["replays"] or _ckpt["sets"] or _ckpt.get("fail_sections"):
            _sd2 = section_dispatch.get(label)
            if _sd2 is None:
                _sd2 = {"register": None, "valid_values": [], "cases": []}
                section_dispatch[label] = _sd2
            _sd2["checkpoints"] = _ckpt
        # Byte-derived conditional-branch ALTERNATE sections: a 0x2b whose
        # fall-through exits via goto_scene while its jump target opens a distinct
        # section (Wrong Side of Town's scene-3 journey/fight gate). The alternate
        # arm would otherwise orphan because the segmenter follows the goto as the
        # only exit. Stashed for a repair pass that wires it only when orphaned.
        _balt = _branch_alt_sections(sc)
        if _balt:
            _sd4 = section_dispatch.get(label)
            if _sd4 is None:
                _sd4 = {"register": None, "valid_values": [], "cases": []}
                section_dispatch[label] = _sd4
            _sd4["branch_alt_sections"] = _balt
        # Byte-derived dispatch DEFAULT arm (the section after 1f12 that a plain
        # 0x28 in the table also targets). Normally already reachable; stashed so a
        # repair pass can wire it only if the scene-entry redirect left it orphaned
        # (Wrong Side of Town's opening combat, skipped when value 0 jumps forward).
        _ddef = _dispatch_default_arm(sc)
        if _ddef is not None:
            _sd5 = section_dispatch.get(label)
            if _sd5 is None:
                _sd5 = {"register": None, "valid_values": [], "cases": []}
                section_dispatch[label] = _sd5
            _sd5["default_arm"] = _ddef
        # Byte-derived computed-condition gate targets (0x2b forward skips whose
        # test is computed on the stack, not a plain 1f2d read). Stashed so a
        # repair pass can wire a target that lands on an orphaned block (e.g. the
        # opening fight's win/lose loss-narration branch), gated on exact
        # reachability so ordinary in-segment conditionals are never split.
        _cgt = _computed_gate_targets(sc)
        if _cgt:
            _sd6 = section_dispatch.get(label)
            if _sd6 is None:
                _sd6 = {"register": None, "valid_values": [], "cases": []}
                section_dispatch[label] = _sd6
            _sd6["computed_gate_targets"] = _cgt
        # Byte-derived timed (quick-time) choices: the action/reflex choices shown
        # on a countdown (Swim's ledge-crossing "Slow!|Rush!", Halloween's
        # "Scream!|Duck!"). Keyed by choice offset -> {timer_ms}. Segmentation
        # attaches `timed`/`timer_ms` to the matching choice node so the engine
        # runs a countdown and drops to the non-ideal branch on expiry.
        _tc = _scan_timed_choices(sc)
        if _tc:
            _sd3 = section_dispatch.get(label)
            if _sd3 is None:
                _sd3 = {"register": None, "valid_values": [], "cases": []}
                section_dispatch[label] = _sd3
            _sd3["timed_choices"] = {str(k): v for k, v in _tc.items()}
        _mg = resolve_minigame_gates(sc)
        if _mg:
            minigame_gates[label] = _mg
        _bws = resolve_buildword_scoring(sc)
        if _bws:
            buildword_scoring[label] = _bws
        # CHAPTER CARD candidate: a continuation script may open with a
        # title + subtitle pair at the head of its string table (e.g.
        # 'Sophomore year' / 'Dating Sophie...'). Confirmed structurally in
        # the builder: the candidate only fires when the SECOND string is
        # exactly the scene's first displayed line.
        _content = [(o, t) for o, t in normalize_strings(find_strings(sc))
                    if o >= 15 and t not in base_cast and t != "Event"]
        if len(_content) >= 2:
            _t0, _t1 = _content[0][1], _content[1][1]
            if (len(_t0) < 40 and "%" not in _t0 and "|" not in _t0
                    and not _is_system(_t0)):
                chapter_heads[label] = (_t0, _t1)
        for var, default in extract_name_vars(sc).items():
            name_vars.setdefault(var, default)
        all_lines.append(Line(f"Scene {i} ({label})", "__SCENE__", None))
        all_lines.extend(lines)

    all_lines = reposition_minigames(
        all_lines, (overlay or {}).get("minigame_triggers"))
    all_lines, mains = tag_pov(all_lines, base_cast)
    mains = derive_main_characters(scripts, base_cast, mains)

    text = render(all_lines, base_cast, title=Path(args.input).stem, fmt=args.format,
                  backgrounds=not args.no_backgrounds, episode=episode,
                  branches=args.branches, main_characters=mains,
                  scene_controls=scene_controls, overlay=overlay, genders=genders,
                  minigame_banks=minigame_banks, name_vars=name_vars,
                  choice_dispatch=choice_dispatch, chapter_heads=chapter_heads,
                  var_gates=var_gates, section_dispatch=section_dispatch,
                  minigame_gates=minigame_gates, buildword_scoring=buildword_scoring,
                  merge_jumps=merge_jumps, choice_forks=choice_forks)

    if args.story_dir:
        doc = json.loads(text) if args.format == "json" else None
        if doc is None:
            print("[!] --story-dir requires --format json", file=sys.stderr)
        else:
            nfiles = export_story_dir(doc, Path(args.story_dir),
                                      var_gates=var_gates)
            print(f"[+] wrote {nfiles} story files to {args.story_dir}",
                  file=sys.stderr)
    # The single JSON uses the same segmented shape as the story-export folder:
    # a map of named segments (each a linear node run), linked by name.
    if args.format == "json":
        _predoc = json.loads(text)
        # Attach the byte-derived source variable to each dynamic `%d` status
        # line ("Kim has %d dollars." -> var2000; "Kim has %d days left." ->
        # 10 - var2001), so the engine knows what to substitute for the
        # placeholder instead of guessing. Matched by the node's byte offset
        # against resolve_dynamic_status_vars, per scene.
        def _apply_status_vars(node, scn):
            if not isinstance(node, dict):
                return
            if node.get("type") == "status" and node.get("dynamic") \
                    and node.get("offset") is not None \
                    and "%d" in (node.get("text") or ""):
                svm = status_vars.get(str(scn)) or {}
                try:
                    info = svm.get(int(node["offset"]))
                except (TypeError, ValueError):
                    info = None
                if info:
                    src = {"var": str(info["var"])}
                    if info.get("consts"):
                        # a days-left line is 10 - var2001; carry the constant so
                        # the engine computes it rather than showing the counter
                        src["minus_from"] = str(info["consts"][0])
                    if info.get("multiply"):
                        # a scaled score line shows var * K (Halloween Dance's
                        # "%d out of 100" is var2000 * 5); carry the factor so the
                        # engine scales the raw value for display
                        src["multiply"] = str(info["multiply"])
                    node["value_from"] = src

        def _walk_nodes(nl, scn):
            for nd in nl or []:
                if not isinstance(nd, dict):
                    continue
                _apply_status_vars(nd, scn)
                # recurse into nested node holders (choice branches, outcomes)
                for key in ("branch_dialogue", "lines"):
                    if isinstance(nd.get(key), list):
                        _walk_nodes(nd[key], scn)
                for o in nd.get("options", []) or []:
                    if isinstance(o, dict):
                        _walk_nodes(o.get("lines", []), scn)
                for o in nd.get("outcomes", []) or []:
                    if isinstance(o, dict):
                        _walk_nodes(o.get("lines", []), scn)

        for _sc in _predoc.get("scenes", []):
            _walk_nodes(_sc.get("nodes", []), _sc.get("scene"))
        seg_doc = build_segmented_doc(_predoc, var_gates=var_gates,
                                      score_tier_thresholds=score_tier_thresholds)
        # Preserve the pack/season identity from the metadata chunk: pack_id is the
        # pack/season this episode belongs to and episode_id its number within that
        # pack. These are otherwise discarded (only the title string was kept).
        if _ep_meta:
            if _ep_meta.get("pack_id") is not None:
                seg_doc["pack_id"] = _ep_meta["pack_id"]
            if _ep_meta.get("episode_id") is not None:
                seg_doc["episode_id"] = _ep_meta["episode_id"]
            _lt = [t for t in (_ep_meta.get("titles") or []) if t]
            if len(set(_lt)) > 1:
                seg_doc["localized_titles"] = _ep_meta["titles"]
        # Runtime text variables: tokens like $Antagonist are rebound many times
        # (once per fight) and their combat-HUD template is shared, so they cannot
        # be baked into the text. Emit the per-scene binding SCHEDULE -- ordered
        # (offset, name) pairs -- so the engine substitutes the name bound most
        # recently (by bytecode offset) before the line it is rendering. Text
        # nodes keep the literal "$Antagonist" placeholder and their `offset`.
        _rtv = {}
        for _lbl, (_st, _sch) in _per_scene.items():
            if not _sch:
                continue
            _snum = _label_to_scene.get(_lbl)
            if _snum is None:
                continue
            _rtv[str(_snum)] = {
                _tok: [{"offset": _o, "name": _nm}
                       for _o, _nm in sorted(_seq)]
                for _tok, _seq in _sch.items()}
        if _rtv:
            seg_doc["runtime_text_vars"] = {
                "note": ("Placeholders like $Antagonist are true runtime variables: "
                         "the engine reassigns them as the story executes (a new "
                         "opponent is bound before each fight via a 1f2e rename) and "
                         "the combat-HUD template that reads the placeholder is "
                         "SHARED -- every fight jumps into the same template, so the "
                         "name is never baked in. Resolve it like a variable at "
                         "RUNTIME: keep the current value of each token and update it "
                         "whenever execution passes a binding; render the token with "
                         "whatever value is current when a line is shown. The schedule "
                         "below lists each binding as (bytecode offset, name) in "
                         "bytecode order. Bytecode order is NOT execution order, so do "
                         "not resolve by static offset; use the schedule to know the "
                         "possible values and to set the variable when execution "
                         "reaches each binding's offset. Text nodes keep the literal "
                         "$Token and their own `offset`."),
                "scenes": _rtv,
            }
        # Combat scenes: replace the flat combat dump with a playable turn-loop
        # graph (byte-derived moves / damage rule / feedback; antagonist HP is a
        # variable seeded by name). Runs AFTER runtime_text_vars so the fight's
        # antagonist name is available for the HP lookup. General -- only fires on
        # scenes detect_combat() recognises; non-combat episodes are untouched.
        try:
            import combat_decode
            combat_decode.integrate_combat(seg_doc, scripts)
        except Exception as _ce:
            print(f"[!] combat integration skipped: {_ce}", file=sys.stderr)
        text = json.dumps(strip_notes(restore_int_values(stringify_numbers(seg_doc))),
                          indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text)
        print(f"[+] wrote {len(all_lines)} entries to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
