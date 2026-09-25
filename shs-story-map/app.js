/* SHS Story Map — node/connection viewer for shs_decoder.py episode JSON.
 *
 * Accepts two shapes:
 *   1. Segment graph (current decoder): { entry, scene_entries, segments: { id: { nodes, next | gate | choice |
 *      minigame | goto_scene | end } } }. Terminals may also appear as the last node of a segment.
 *   2. Scene list (older decoder / overlay output): { scenes: [ { scene, script, nodes, control_flow?, gate? } ] }.
 * Nothing here invents routing: every edge comes from a field in the JSON. Edges derived from the older
 * shape's structure (choice regions, outcomes) are flagged "inferred" and drawn dotted.
 */
'use strict';

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
const str = v => (v == null ? '' : String(v));
const segKey = v => str(v).replace(/\.json$/i, '');
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
const isObj = v => v && typeof v === 'object' && !Array.isArray(v);
const numSort = (a, b) => {
  const na = parseFloat(a), nb = parseFloat(b);
  if (!isNaN(na) && !isNaN(nb) && na !== nb) return na - nb;
  return str(a).localeCompare(str(b), undefined, { numeric: true });
};

const KINDS = {
  next:   { label: 'Continues',            color: '--k-next' },
  option: { label: 'Choice option',        color: '--k-option' },
  after:  { label: 'Choice merge point',   color: '--k-option', dash: '6 5' },
  then:   { label: 'Condition true',       color: '--k-gate' },
  else:   { label: 'Condition false',      color: '--k-gate', dash: '6 5' },
  case:   { label: 'Condition case',       color: '--k-gate' },
  win:    { label: 'Minigame win',         color: '--k-win' },
  lose:   { label: 'Minigame lose',        color: '--k-lose' },
  mgafter:{ label: 'After minigame',       color: '--k-next', dash: '6 5' },
  fight:  { label: 'Fight',                color: '--k-lose', dash: '2 4' },
  goto:   { label: 'Go to scene',          color: '--k-goto', dash: '9 5' },
  other:  { label: 'Other reference',      color: '--k-other', dash: '2 4' },
};
const INFERRED_DASH = '1.5 4';

const CUE_TYPES = new Set(['background', 'music', 'sfx', 'vibrate', 'wobble', 'loading', 'set_ui_default',
  'set_scene_value', 'native_noop', 'dialogue_notice', 'scene_badge', 'stop_music', 'control', 'pov_change']);
const VAR_SET_TYPES = /^(var_set|set_number|set_var|set_string|set_number_bit)$/;
const VAR_ADD_TYPES = /^(var_add|add_number|add_var)$/;

// ------------------------------------------------------------------ normalization
function fmtCond(g) {
  if (!g) return '';
  if (typeof g.cond === 'string') return g.cond;
  if (typeof g.condition === 'string') return g.condition;
  const v = g.var ?? g.key ?? g.variable ?? g.name;
  const op = g.op ?? g.cmp ?? '==';
  const val = g.equals ?? g.value ?? g.threshold ?? g.gte ?? g.min;
  if (v != null && val != null) return `${v} ${op} ${val}`;
  // older overlay gates: { event, day, once }
  const parts = Object.entries(g).filter(([k, x]) => !['then', 'else', 'cases', 'next'].includes(k) && !isObj(x))
    .map(([k, x]) => `${k} ${x}`);
  return parts.join(', ');
}
function fmtEffect(e) {
  const v = e.var ?? e.key ?? e.name ?? '?';
  if (e.delta != null) { const d = parseFloat(e.delta); return `${v} ${d >= 0 ? '+' : '−'}${Math.abs(d)}`; }
  if (e.value != null) return `${v} = ${e.value}`;
  return String(v);
}

class Model {
  constructor(raw, name) {
    this.raw = raw; this.source = name;
    this.title = raw.episode || raw.title || name || 'Episode';
    this.cast = Array.isArray(raw.cast) ? raw.cast.filter(Boolean) : [];
    this.segs = new Map();      // id -> { id, scene, nodes, raw, badges[] }
    this.scenes = new Map();    // id -> { id, label, entry, segIds[], meta }
    this.edges = [];            // { from, to, kind, label, effects, inferred, note }
    this.vars = new Map();      // name -> [{ seg, kind: set|add|check|read, text }]
    this.entry = null;
    this.notes = [];
    if (isObj(raw.segments)) { this.schema = 'segments'; this.fromSegments(raw); }
    else if (Array.isArray(raw.scenes)) { this.schema = 'scenes'; this.fromScenes(raw); }
    else throw new Error('This JSON has neither a "segments" map nor a "scenes" list, so it does not look like decoder output.');
    this.collectVars();
    this.indexEdges();
  }

  addEdge(from, to, kind, extra = {}) {
    if (to == null || to === '') return;
    this.edges.push({ from, to: str(to), kind, ...extra });
  }

  // ---- current decoder: segment graph
  fromSegments(raw) {
    const S = raw.segments;
    for (const [id, s] of Object.entries(S)) {
      this.segs.set(id, { id, scene: null, nodes: Array.isArray(s.nodes) ? s.nodes : [], raw: s, badges: [] });
    }
    const has = id => this.segs.has(segKey(id));
    const sceneEntries = isObj(raw.scene_entries) ? raw.scene_entries : {};
    const entryOfScene = sc => (sceneEntries[str(sc)] != null ? segKey(sceneEntries[str(sc)]) : null);

    const gotoTarget = g => {
      if (typeof g === 'string') return segKey(g);
      if (!isObj(g)) return null;
      for (const k of ['file', 'segment', 'entry', 'target', 'next']) if (g[k] && has(g[k])) return segKey(g[k]);
      for (const k of ['to_scene', 'scene']) if (g[k] != null && entryOfScene(g[k])) return entryOfScene(g[k]);
      return g.file ? segKey(g.file) : null;
    };

    for (const [id, s] of Object.entries(S)) {
      const used = new Set();
      const take = (to, kind, extra) => { if (typeof to === 'string' || typeof to === 'number') { used.add(segKey(to)); this.addEdge(id, segKey(to), kind, extra); } };
      const choice = c => {
        (c.options || []).forEach((o, i) => {
          if (!isObj(o)) return;
          const label = o.label ?? o.text ?? `Option ${o.index ?? i}`;
          take(o.next ?? o.goto ?? o.target, 'option', { label: str(label), effects: o.effects, index: o.index ?? i });
          if (o.fight) take(o.fight, 'fight', { label: str(label) });
        });
        if (c.after) take(c.after, 'after', { label: 'merge' });
      };
      const gate = g => {
        const cond = fmtCond(g);
        if (g.then) take(g.then, 'then', { label: cond || 'true', cond });
        if (g.else) take(g.else, 'else', { label: 'otherwise', cond });
        if (Array.isArray(g.cases)) g.cases.forEach(c => isObj(c) && take(c.next ?? c.then ?? c.target, 'case',
          { label: c.label ?? `${g.var ?? ''} == ${c.equals ?? c.value ?? '?'}`.trim(), cond }));
      };
      const minigame = m => {
        if (typeof m.win === 'string') take(m.win, 'win', { label: m.win_label || 'win' });
        if (typeof m.lose === 'string') take(m.lose, 'lose', { label: m.lose_label || 'lose' });
        if (typeof m.after === 'string') take(m.after, 'mgafter', { label: 'after' });
      };
      const goto = g => {
        const t = gotoTarget(g);
        const sc = isObj(g) ? (g.to_scene ?? g.scene ?? g.script) : null;
        if (t) take(t, 'goto', { label: sc != null ? `scene ${sc}` : 'go to scene' });
      };

      if (typeof s.next === 'string') take(s.next, 'next');
      if (isObj(s.gate)) gate(s.gate);
      if (isObj(s.choice)) choice(s.choice);
      if (isObj(s.minigame)) minigame(s.minigame);
      if (s.goto_scene) goto(s.goto_scene);
      for (const n of this.segs.get(id).nodes) {
        if (!isObj(n)) continue;
        switch (n.type) {
          case 'choice': choice(n); break;
          case 'next': take(n.next, 'next'); break;
          case 'gate': gate(n); break;
          case 'goto_scene': goto(n); break;
          case 'minigame': minigame(n); break;
          case 'random_encounter': choice(n); break;
        }
        if (typeof n.next === 'string' && n.type !== 'next' && n.type !== 'choice') take(n.next, 'next');
      }
      // Anything else in the segment that names another segment (future fields, fight tables, ...)
      const scan = (obj, depth, keyPath) => {
        if (depth > 3 || obj == null) return;
        if (typeof obj === 'string') {
          const k = segKey(obj);
          if (k !== id && has(k) && !used.has(k)) { used.add(k); this.addEdge(id, k, 'other', { label: keyPath }); }
          return;
        }
        if (Array.isArray(obj)) { obj.forEach(x => scan(x, depth + 1, keyPath)); return; }
        if (isObj(obj)) for (const [k, v] of Object.entries(obj)) {
          if (['text', 'speaker', 'prompt', 'label', 'type', 'note', 'id'].includes(k)) continue;
          scan(v, depth + 1, k);
        }
      };
      scan(s, 0, '');
      if (s.end) this.segs.get(id).end = true;
    }

    // Scenes: BFS from each scene entry, never crossing a goto edge.
    const out = new Map();
    for (const e of this.edges) { if (!out.has(e.from)) out.set(e.from, []); out.get(e.from).push(e); }
    const keys = Object.keys(sceneEntries).sort(numSort);
    for (const k of keys) {
      const entry = segKey(sceneEntries[k]);
      const sc = { id: k, label: `Scene ${k}`, entry, segIds: [], meta: {} };
      this.scenes.set(k, sc);
      const q = [entry];
      while (q.length) {
        const id = q.shift(); const seg = this.segs.get(id);
        if (!seg || seg.scene != null) continue;
        seg.scene = k; sc.segIds.push(id);
        for (const e of out.get(id) || []) if (e.kind !== 'goto') q.push(e.to);
      }
      const es = this.segs.get(entry); if (es) es.badges.push({ cls: 'entry', text: `Scene ${k} start` });
    }
    const loose = [...this.segs.values()].filter(s => s.scene == null);
    if (loose.length) {
      const k = keys.length ? 'unplaced' : 'all';
      this.scenes.set(k, { id: k, label: keys.length ? 'Not reached from a scene start' : 'All segments',
        entry: loose[0].id, segIds: loose.map(s => s.id), meta: {} });
      loose.forEach(s => (s.scene = k));
    }
    this.entry = raw.entry ? segKey(raw.entry) : (keys.length ? segKey(sceneEntries[keys[0]]) : null);
    const en = this.segs.get(this.entry); if (en) en.badges.unshift({ cls: 'entry', text: 'Episode start' });
    for (const e of this.edges) if (e.kind === 'goto') {
      const sc = this.scenes.get(this.segs.get(e.to)?.scene);
      if (sc) e.label = sc.label;
    }
    if (raw.note) this.notes.push(str(raw.note));
  }

  // ---- older decoder: scenes -> nodes (with optional overlay splits)
  fromScenes(raw) {
    this.notes.push('Older scene-list format. Nodes are in bytecode order, so links between a choice and the lines after it are inferred from layout (dotted) unless the file split them per option.');
    const scenes = raw.scenes;
    const byNum = new Map(), byScript = new Map();
    scenes.forEach((sc, i) => {
      const k = str(sc.scene ?? i + 1);
      byNum.set(k, sc); if (sc.script && !byScript.has(str(sc.script))) byScript.set(str(sc.script), k);
    });
    const pendingGotos = [];
    scenes.forEach((sc, i) => {
      const k = str(sc.scene ?? i + 1);
      const label = `Scene ${k}` + (sc.name ? ` (${sc.name})` : '');
      const scene = { id: k, label, entry: null, segIds: [], meta: { script: sc.script, protagonist: sc.protagonist,
        gate: sc.gate, source: sc.source, control_flow: sc.control_flow } };
      this.scenes.set(k, scene);
      const observed = sc.source === 'observed';
      let n = 0, cur = null, pending = [];
      const newSeg = suffix => {
        const id = `sc${k}${suffix}`;
        const seg = { id, scene: k, nodes: [], raw: null, badges: [] };
        this.segs.set(id, seg); scene.segIds.push(id); if (!scene.entry) scene.entry = id;
        return seg;
      };
      const ensure = () => {
        if (!cur) {
          cur = newSeg(n === 0 ? '' : `_${n}`); n++;
          for (const p of pending) this.addEdge(p.from, cur.id, p.kind, p.extra);
          pending = [];
        }
        return cur;
      };
      for (const node of sc.nodes || []) {
        if (!isObj(node)) continue;
        if (node.type === 'choice') {
          const seg = ensure(); seg.nodes.push(node);
          const opts = node.options || [];
          const bd = node.branch_dialogue;
          const perOption = Array.isArray(bd) && bd.length && bd.every(b => isObj(b) && Array.isArray(b.lines));
          if (perOption) {
            opts.forEach((o, oi) => {
              const br = bd.find(b => str(b.index) === str(o.index ?? oi));
              const extra = { label: str(o.label ?? `Option ${oi}`), effects: o.effects, inferred: !observed, observed };
              if (br && br.lines.length) {
                const os = newSeg(`_${n}o${o.index ?? oi}`); n++;
                os.nodes = br.lines;
                this.addEdge(seg.id, os.id, 'option', extra);
                pending.push({ from: os.id, kind: 'next', extra: { inferred: true } });
              } else pending.push({ from: seg.id, kind: 'option', extra });
            });
          } else if (Array.isArray(node.outcomes) && node.outcomes.length) {
            node.outcomes.forEach((oc, oi) => {
              const os = newSeg(`_${n}r${oc.index ?? oi}`); n++;
              os.nodes = oc.lines || [];
              if (oc.ends_with_retry_prompt) os.badges.push({ cls: 'gate', text: 'Replay prompt' });
              this.addEdge(seg.id, os.id, 'option', { label: `outcome ${oc.index ?? oi}`, inferred: true });
              if (!oc.ends_with_retry_prompt) pending.push({ from: os.id, kind: 'next', extra: { inferred: true } });
            });
          } else if (Array.isArray(bd) && bd.length) {
            const os = newSeg(`_${n}all`); n++;
            os.nodes = bd;
            os.badges.push({ cls: 'gate', text: 'All options, unsplit' });
            this.addEdge(seg.id, os.id, 'option', { label: `any of ${opts.length} options`, inferred: true });
            pending.push({ from: os.id, kind: 'next', extra: { inferred: true } });
          } else {
            opts.forEach((o, oi) => pending.push({ from: seg.id, kind: 'option',
              extra: { label: str(o.label ?? `Option ${oi}`), effects: o.effects, inferred: true } }));
          }
          cur = null;
          continue;
        }
        if (node.type === 'goto_scene') {
          const seg = ensure(); seg.nodes.push(node);
          pendingGotos.push({ from: seg.id, to_scene: node.to_scene, script: node.script });
          cur = null; pending = [];
          continue;
        }
        ensure().nodes.push(node);
      }
      if (!scene.entry) ensure();
      const es = this.segs.get(scene.entry);
      es.badges.push({ cls: 'entry', text: `Scene ${k} start` });
      if (sc.gate) es.badges.push({ cls: 'gate', text: `Runs when ${fmtCond(sc.gate)}` });
      if (observed) es.badges.push({ cls: 'obs', text: 'Observed in play' });
    });
    for (const g of pendingGotos) {
      let k = g.to_scene != null ? str(g.to_scene) : null;
      if ((k == null || !this.scenes.has(k)) && g.script) k = byScript.get(str(g.script)) ?? k;
      const sc = k != null ? this.scenes.get(k) : null;
      if (sc) this.addEdge(g.from, sc.entry, 'goto', { label: `scene ${k}` });
    }
    const first = this.scenes.values().next().value;
    this.entry = first ? first.entry : null;
    const en = this.segs.get(this.entry); if (en) en.badges.unshift({ cls: 'entry', text: 'Episode start' });
  }

  collectVars() {
    const add = (name, rec) => {
      if (name == null || name === '') return;
      const k = str(name);
      if (!this.vars.has(k)) this.vars.set(k, []);
      this.vars.get(k).push(rec);
    };
    for (const seg of this.segs.values()) {
      const walk = nodes => nodes.forEach(n => {
        if (!isObj(n)) return;
        const name = n.var ?? n.key ?? n.name;
        if (VAR_ADD_TYPES.test(n.type || '') || (n.delta != null && /var|number/.test(n.type || ''))) add(name, { seg: seg.id, kind: 'add', text: fmtEffect(n) });
        else if (VAR_SET_TYPES.test(n.type || '')) add(name, { seg: seg.id, kind: 'set', text: fmtEffect(n) });
        if (n.value_from) add(n.value_from, { seg: seg.id, kind: 'read', text: `shown in "${trunc(str(n.text), 40)}"` });
        if (n.type === 'gate') add(n.var ?? n.key, { seg: seg.id, kind: 'check', text: fmtCond(n) });
        if (Array.isArray(n.branch_dialogue)) n.branch_dialogue.forEach(b => Array.isArray(b?.lines) && walk(b.lines));
      });
      walk(seg.nodes);
      const g = seg.raw && seg.raw.gate;
      if (isObj(g)) add(g.var ?? g.key ?? g.variable, { seg: seg.id, kind: 'check', text: fmtCond(g) });
    }
    for (const e of this.edges) {
      (e.effects || []).forEach(f => add(f.var ?? f.key ?? f.name,
        { seg: e.from, kind: f.delta != null ? 'add' : 'set', text: `${fmtEffect(f)} when "${e.label}" is picked` }));
    }
    for (const sc of this.scenes.values()) {
      const g = sc.meta.gate;
      if (isObj(g)) for (const [k, v] of Object.entries(g)) if (!isObj(v) && k !== 'once' && k !== 'event')
        add(k, { seg: sc.entry, kind: 'check', text: `scene ${sc.id} runs when ${fmtCond(g)}` });
      const cf = sc.meta.control_flow;
      if (cf && Array.isArray(cf.var_reads)) cf.var_reads.forEach(r =>
        add(`id ${r.value}`, { seg: sc.entry, kind: 'read', text: `probable read at offset ${r.offset}` }));
    }
  }

  indexEdges() {
    this.out = new Map(); this.inc = new Map();
    this.edges.forEach((e, i) => {
      e.i = i;
      if (!this.out.has(e.from)) this.out.set(e.from, []);
      if (!this.inc.has(e.to)) this.inc.set(e.to, []);
      this.out.get(e.from).push(e); this.inc.get(e.to).push(e);
    });
    this.missing = this.edges.filter(e => !this.segs.has(e.to));
  }

  lineCount(seg) { return seg.nodes.filter(n => isObj(n) && (n.type === 'dialogue' || n.type === 'narration')).length; }
  sceneOf(segId) { const s = this.segs.get(segId); return s ? s.scene : null; }
}

// ------------------------------------------------------------------ state
const state = {
  model: null, view: 'overview', scene: null, sel: null, showCues: false, cap: 8,
  k: 1, tx: 0, ty: 0, graph: null, highlightNode: null,
};
const stage = $('#stage'), world = $('#world'), svg = $('#edges');

// ------------------------------------------------------------------ card rendering
function speakerColor(name) {
  const m = state.model;
  let i = m ? m.cast.indexOf(name) : -1;
  if (i < 0) { i = 0; for (const c of str(name)) i = (i * 31 + c.charCodeAt(0)) >>> 0; }
  return `var(--spk${i % 6})`;
}

function renderNode(n, full) {
  if (!isObj(n)) return el('div', 'ln cue', str(n));
  const t = n.type;
  if (t === 'dialogue') {
    const d = el('div', 'ln');
    const sp = el('span', 'spk', str(n.speaker || '?')); sp.style.color = speakerColor(n.speaker);
    d.append(sp);
    if (n.emotion) d.append(el('span', 'emo', `${n.emotion} `));
    d.append(document.createTextNode(str(n.text)));
    return d;
  }
  if (t === 'narration') return el('div', 'ln narr', str(n.text));
  if (t === 'status') return el('div', 'ln status', str(n.text));
  if (t === 'title_card' || t === 'end_card') return el('div', 'ln cardtitle', str(n.text));
  if (t === 'notification') return el('div', 'ln status', str(n.text));
  if (VAR_SET_TYPES.test(t || '') || VAR_ADD_TYPES.test(t || '')) {
    const d = el('div', 'ln'); d.append(el('span', 'chip', fmtEffect(n))); return d;
  }
  if (t === 'gate') { const d = el('div', 'ln'); d.append(el('span', 'chip check', `if ${fmtCond(n)}`)); return d; }
  if (t === 'choice' || t === 'random_encounter') {
    const b = el('div', 'choice-box');
    b.append(el('div', 'box-label', n.prompt ? `Choice: ${n.prompt}` : 'Choice'));
    const ol = el('ol');
    (n.options || []).forEach(o => {
      const li = el('li', null, str(isObj(o) ? (o.label ?? o.text ?? o.name) : o));
      if (isObj(o) && o.effects) li.append(el('span', 'emo', ` (${o.effects.map(fmtEffect).join(', ')})`));
      ol.append(li);
    });
    b.append(ol);
    return b;
  }
  if (t === 'minigame') {
    const b = el('div', 'mg-box');
    const kind = n.minigame_type || n.kind || 'minigame';
    b.append(el('div', 'box-label', `Minigame: ${kind}` + (n.win_threshold != null ? `, win at ${n.win_threshold}` : '')));
    if (n.prompt) b.append(el('div', null, str(n.prompt)));
    const groups = n.options || (n.correct || n.decoys ? [n.correct || [], n.decoys || []] : null);
    if (Array.isArray(groups) && full) groups.forEach((g, i) =>
      b.append(el('div', 'emo', `${groups.length === 2 ? (i ? 'Decoys' : 'Correct') : 'Group ' + (i + 1)}: ${Array.isArray(g) ? g.join(', ') : str(g)}`)));
    if (Array.isArray(n.words) && full) b.append(el('div', 'emo', trunc(n.words.join(' | '), 400)));
    return b;
  }
  if (t === 'goto_scene') return el('div', 'ln cue', `Go to scene ${n.to_scene ?? n.script ?? n.file ?? ''}`);
  if (t === 'background') return el('div', 'ln cue', `Background ${n.name ? n.name + ' ' : ''}${n.id ?? ''}`);
  if (t === 'music') return el('div', 'ln cue', n.action === 'stop' ? 'Music stops' : `Music track ${n.id ?? ''}`);
  if (t === 'sfx') return el('div', 'ln cue', `Sound ${n.id ?? ''}`);
  const d = el('div', 'ln cue', `${t || 'node'}${n.text ? ': ' + n.text : ''}`);
  return d;
}

function isCue(n) { return isObj(n) && CUE_TYPES.has(n.type); }

function terminalText(m, seg) {
  const outs = m.out.get(seg.id) || [];
  if (!outs.length) return seg.end ? 'Episode ends' : 'No further links';
  const kinds = new Set(outs.map(e => e.kind));
  if (kinds.has('win') || kinds.has('lose')) return 'Minigame result decides';
  if (kinds.has('then') || kinds.has('case')) return `Checks ${outs.find(e => e.cond)?.cond || 'a condition'}`;
  if (kinds.has('option')) return `${outs.filter(e => e.kind === 'option').length} choice links`;
  if (kinds.has('goto')) { const l = outs.find(e => e.kind === 'goto').label; return /^scene/i.test(l) ? `Goes to ${l}` : 'Goes to another scene'; }
  return 'Continues';
}

function segCard(seg) {
  const m = state.model;
  const c = el('div', 'card segcard');
  c.dataset.id = seg.id; c.tabIndex = -1;
  const h = el('div', 'card-head');
  h.append(el('span', 'id', seg.id));
  { const lc = m.lineCount(seg); h.append(el('span', 'meta', `${lc} ${lc === 1 ? 'line' : 'lines'}`)); }
  c.append(h);
  if (seg.badges.length) {
    const b = el('div', 'badges');
    seg.badges.forEach(x => b.append(el('span', `badge ${x.cls}`, x.text)));
    c.append(b);
  }
  const body = el('div', 'card-body');
  const visible = seg.nodes.filter(n => state.showCues || !isCue(n));
  const cap = state.cap || Infinity;
  let shown = 0, hiddenLines = 0;
  for (const n of visible) {
    const heavy = isObj(n) && (n.type === 'choice' || n.type === 'minigame');
    if (shown < cap || heavy) { body.append(renderNode(n, false)); if (!heavy) shown++; }
    else hiddenLines++;
  }
  if (!visible.length) body.append(el('div', 'ln cue', seg.nodes.length ? `${seg.nodes.length} presentation cues` : 'Empty segment'));
  if (hiddenLines) body.append(el('div', 'more', `${hiddenLines} more, open to read all`));
  c.append(body);
  c.append(el('div', 'card-foot', terminalText(m, seg)));
  return c;
}

function portalCard(p) {
  const c = el('div', 'card portal');
  c.dataset.id = p.id; c.dataset.portal = p.scene; c.dataset.target = p.target || '';
  c.append(el('div', 'card-body', p.text));
  return c;
}

function sceneCard(sc) {
  const m = state.model;
  const c = el('div', 'card scene'); c.dataset.id = 'scene:' + sc.id; c.dataset.scene = sc.id;
  const h = el('div', 'card-head'); h.append(el('span', 'id', sc.label));
  if (sc.meta.protagonist) h.append(el('span', 'meta', `as ${sc.meta.protagonist}`));
  c.append(h);
  const b = el('div', 'card-body');
  let lines = 0, choices = 0, games = 0, gates = 0; const speakers = new Map();
  for (const id of sc.segIds) {
    const s = m.segs.get(id);
    lines += m.lineCount(s);
    s.nodes.forEach(n => {
      if (!isObj(n)) return;
      if (n.type === 'choice') choices++;
      if (n.type === 'minigame') games++;
      if (n.type === 'dialogue' && n.speaker) speakers.set(n.speaker, (speakers.get(n.speaker) || 0) + 1);
    });
    const outs = m.out.get(id) || [];
    if (outs.some(e => e.kind === 'then' || e.kind === 'case')) gates++;
    if (!s.nodes.some(n => isObj(n) && n.type === 'minigame') && outs.some(e => e.kind === 'win')) games++;
  }
  const stats = el('div', 'scene-stats');
  [[lines, 'lines'], [choices, 'choices'], [games, 'minigames'], [gates, 'checks']].forEach(([v, l]) => {
    const d = el('div'); d.append(el('b', null, String(v)), el('span', null, l)); stats.append(d);
  });
  b.append(stats);
  if (sc.meta.gate) { const g = el('div', 'ln'); g.append(el('span', 'chip check', `Runs when ${fmtCond(sc.meta.gate)}`)); b.append(g); }
  const top = [...speakers.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
  if (top.length) b.append(el('div', 'ln muted', `Speakers: ${top.join(', ')}`));
  const first = m.segs.get(sc.entry);
  const firstLine = first && first.nodes.find(n => isObj(n) && (n.type === 'dialogue' || n.type === 'narration'));
  if (firstLine) b.append(renderNode(firstLine));
  c.append(b);
  c.append(el('div', 'card-foot', `${sc.segIds.length} segments. Open scene flow`));
  return c;
}

// ------------------------------------------------------------------ graph building
function buildGraph() {
  const m = state.model;
  const nodes = [], edges = [];
  if (state.view === 'overview') {
    for (const sc of m.scenes.values()) nodes.push({ id: 'scene:' + sc.id, kind: 'scene', sc });
    const agg = new Map();
    for (const e of m.edges) {
      const a = m.sceneOf(e.from), b = m.sceneOf(e.to);
      if (a == null || b == null || a === b) continue;
      const key = `${a}>${b}>${e.kind}`;
      if (!agg.has(key)) agg.set(key, { from: 'scene:' + a, to: 'scene:' + b, kind: e.kind, n: 0, labels: new Set(), inferred: true });
      const g = agg.get(key); g.n++; if (e.label) g.labels.add(e.label); g.inferred = g.inferred && !!e.inferred;
    }
    for (const g of agg.values()) {
      const lbl = g.kind === 'goto' ? (g.n > 1 ? `${g.n} jumps` : 'jump') : `${KINDS[g.kind]?.label || g.kind}${g.n > 1 ? ' ×' + g.n : ''}`;
      edges.push({ from: g.from, to: g.to, kind: g.kind, label: lbl, inferred: g.inferred });
    }
    return { nodes, edges };
  }
  const inScene = id => state.view === 'all' || m.sceneOf(id) === state.scene;
  const segIds = state.view === 'all' ? [...m.segs.keys()] : (m.scenes.get(state.scene)?.segIds || []);
  segIds.forEach(id => nodes.push({ id, kind: 'seg', seg: m.segs.get(id) }));
  const portals = new Map();
  const portal = (id, sc, text, target) => { if (!portals.has(id)) portals.set(id, { id, kind: 'portal', scene: sc, text, target }); return id; };
  for (const id of segIds) for (const e of m.out.get(id) || []) {
    if (!m.segs.has(e.to)) {
      edges.push({ ...e, to: portal('missing:' + e.to, null, `Missing segment ${e.to}`, null) });
    } else if (inScene(e.to)) edges.push(e);
    else {
      const sc = m.sceneOf(e.to);
      edges.push({ ...e, to: portal(`out:${sc}:${e.to}`, sc, `Continue in ${m.scenes.get(sc)?.label || sc} at ${e.to}`, e.to) });
    }
  }
  if (state.view === 'scene') {
    for (const id of segIds) for (const e of m.inc.get(id) || []) {
      if (inScene(e.from)) continue;
      const sc = m.sceneOf(e.from);
      edges.push({ ...e, from: portal(`in:${sc}`, sc, `From ${m.scenes.get(sc)?.label || sc}`, e.from) });
    }
  }
  portals.forEach(p => nodes.push(p));
  return { nodes, edges };
}

// ------------------------------------------------------------------ layout + draw
function ensureMarkers() {
  const defs = $('#markers'); if (defs.childNodes.length) return;
  for (const k of Object.keys(KINDS)) {
    const mk = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    mk.setAttribute('id', 'arrow-' + k); mk.setAttribute('viewBox', '0 0 10 10');
    mk.setAttribute('refX', '9'); mk.setAttribute('refY', '5');
    mk.setAttribute('markerWidth', '7'); mk.setAttribute('markerHeight', '7'); mk.setAttribute('orient', 'auto-start-reverse');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', 'M0,0 L10,5 L0,10 z'); p.style.fill = `var(${KINDS[k].color})`;
    mk.append(p); defs.append(mk);
  }
}

function smoothPath(pts) {
  if (pts.length < 2) return '';
  let d = `M${pts[0].x},${pts[0].y}`;
  if (pts.length === 2) return d + ` L${pts[1].x},${pts[1].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i], n = pts[i + 1];
    const mx = (p.x + n.x) / 2, my = (p.y + n.y) / 2;
    d += i === pts.length - 2 ? ` Q${p.x},${p.y} ${n.x},${n.y}` : ` Q${p.x},${p.y} ${mx},${my}`;
  }
  return d;
}

function render({ keepView = false } = {}) {
  const m = state.model;
  world.querySelectorAll('.card, .elabel').forEach(x => x.remove());
  svg.querySelectorAll(':scope > path').forEach(x => x.remove());
  $('#empty').hidden = !!m;
  if (!m) return;
  ensureMarkers();
  const g = buildGraph();
  state.graph = g;

  const cards = new Map();
  const frag = document.createDocumentFragment();
  for (const n of g.nodes) {
    const c = n.kind === 'seg' ? segCard(n.seg) : n.kind === 'scene' ? sceneCard(n.sc) : portalCard(n);
    c.style.visibility = 'hidden'; cards.set(n.id, c); frag.append(c);
  }
  const labels = g.edges.map(e => {
    if (!e.label && !(e.effects && e.effects.length)) return null;
    const l = el('div', 'elabel');
    l.style.color = `var(${KINDS[e.kind]?.color || '--k-other'})`;
    l.append(document.createTextNode(trunc(str(e.label || ''), 60)));
    if (e.effects && e.effects.length) l.append(el('span', 'fx', e.effects.map(fmtEffect).join(', ')));
    if (e.inferred) l.title = 'Inferred from layout, not an explicit link in the JSON';
    l.style.visibility = 'hidden'; frag.append(l);
    return l;
  });
  world.append(frag);

  const G = new dagre.graphlib.Graph({ multigraph: true });
  G.setGraph({ rankdir: 'TB', nodesep: 36, ranksep: 64, edgesep: 14, marginx: 40, marginy: 40 });
  G.setDefaultEdgeLabel(() => ({}));
  for (const n of g.nodes) {
    const c = cards.get(n.id);
    G.setNode(n.id, { width: c.offsetWidth, height: c.offsetHeight });
  }
  g.edges.forEach((e, i) => {
    if (!G.hasNode(e.from) || !G.hasNode(e.to)) return;
    const l = labels[i];
    G.setEdge(e.from, e.to, { width: l ? l.offsetWidth : 0, height: l ? l.offsetHeight : 0, labelpos: 'c',
      minlen: e.kind === 'goto' ? 2 : 1 }, 'e' + i);
  });
  dagre.layout(G);

  for (const n of g.nodes) {
    const p = G.node(n.id), c = cards.get(n.id);
    c.style.left = (p.x - p.width / 2) + 'px'; c.style.top = (p.y - p.height / 2) + 'px'; c.style.visibility = '';
  }
  const gg = G.graph();
  svg.setAttribute('width', gg.width); svg.setAttribute('height', gg.height);
  world.style.width = gg.width + 'px'; world.style.height = gg.height + 'px';
  state.bounds = { w: gg.width, h: gg.height };
  g.edges.forEach((e, i) => {
    if (!G.hasNode(e.from) || !G.hasNode(e.to)) return;
    const ed = G.edge({ v: e.from, w: e.to, name: 'e' + i });
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const kd = KINDS[e.kind] || KINDS.other;
    path.setAttribute('d', smoothPath(ed.points));
    path.style.stroke = `var(${kd.color})`;
    const dash = e.inferred ? INFERRED_DASH : kd.dash;
    if (dash) path.setAttribute('stroke-dasharray', dash);
    path.setAttribute('marker-end', `url(#arrow-${KINDS[e.kind] ? e.kind : 'other'})`);
    path.dataset.from = e.from; path.dataset.to = e.to;
    svg.append(path);
    const l = labels[i];
    if (l) { l.style.left = ed.x + 'px'; l.style.top = ed.y + 'px'; l.style.visibility = ''; l.dataset.from = e.from; l.dataset.to = e.to; }
  });
  labels.forEach(l => { if (l && l.style.visibility === 'hidden') l.remove(); });

  applySelection();
  if (!keepView) {
    if (state.sel && cards.has(state.sel)) centerOn(state.sel); else fit();
  } else applyTransform();
}

// ------------------------------------------------------------------ view transform
function applyTransform() { world.style.transform = `translate(${state.tx}px,${state.ty}px) scale(${state.k})`; }
function fit() {
  if (!state.bounds) return;
  const r = stage.getBoundingClientRect();
  const k = Math.min(1.2, Math.max(0.05, Math.min(r.width / state.bounds.w, r.height / state.bounds.h) * 0.95));
  state.k = k;
  state.tx = (r.width - state.bounds.w * k) / 2;
  state.ty = Math.max(10, (r.height - state.bounds.h * k) / 2);
  applyTransform();
}
function zoomAt(f, cx, cy) {
  const k = Math.min(3, Math.max(0.04, state.k * f));
  const r = stage.getBoundingClientRect();
  cx = cx ?? r.width / 2; cy = cy ?? r.height / 2;
  state.tx = cx - (cx - state.tx) * (k / state.k);
  state.ty = cy - (cy - state.ty) * (k / state.k);
  state.k = k; applyTransform();
}
function centerOn(id, flash) {
  const c = world.querySelector(`.card[data-id="${CSS.escape(id)}"]`);
  if (!c) return;
  const r = stage.getBoundingClientRect();
  if (state.k < 0.6) state.k = 0.9;
  const x = c.offsetLeft + c.offsetWidth / 2, y = c.offsetTop + Math.min(c.offsetHeight / 2, r.height / 3);
  state.tx = r.width / 2 - x * state.k; state.ty = r.height / 2 - y * state.k;
  applyTransform();
  if (flash) { c.classList.remove('flash'); void c.offsetWidth; c.classList.add('flash'); }
}

// ------------------------------------------------------------------ selection + inspector
function applySelection() {
  const sel = state.sel;
  const near = new Set();
  if (sel) {
    near.add(sel);
    svg.querySelectorAll(':scope > path').forEach(p => { if (p.dataset.from === sel) near.add(p.dataset.to); if (p.dataset.to === sel) near.add(p.dataset.from); });
  }
  world.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('sel', c.dataset.id === sel);
    c.classList.toggle('dim', !!sel && !near.has(c.dataset.id));
  });
  svg.querySelectorAll(':scope > path').forEach(p => {
    const hot = sel && (p.dataset.from === sel || p.dataset.to === sel);
    p.classList.toggle('hot', !!hot); p.classList.toggle('dim', !!sel && !hot);
  });
  world.querySelectorAll('.elabel').forEach(l => l.classList.toggle('dim', !!sel && l.dataset.from !== sel && l.dataset.to !== sel));
}

function linkRow(e, dir) {
  const m = state.model;
  const other = dir === 'out' ? e.to : e.from;
  const b = el('button', 'linkrow');
  const kd = KINDS[e.kind] || KINDS.other;
  b.style.setProperty('--kc', `var(${kd.color})`);
  b.append(el('span', 'k', e.kind === 'option' ? 'Option' : kd.label));
  const txt = [e.label && e.kind !== 'next' ? `"${e.label}"` : null,
    e.effects && e.effects.length ? `sets ${e.effects.map(fmtEffect).join(', ')}` : null,
    `${dir === 'out' ? 'to' : 'from'} ${other}`,
    m.sceneOf(other) !== m.sceneOf(dir === 'out' ? e.from : e.to) ? `(${m.scenes.get(m.sceneOf(other))?.label || 'missing'})` : null,
    e.inferred ? '(inferred)' : null].filter(Boolean).join(' ');
  b.append(el('span', 'd', txt));
  b.onclick = () => goToSeg(other);
  return b;
}

function openInspector(id, hlText) {
  const m = state.model; const seg = m.segs.get(id);
  const body = $('#inspBody'); body.textContent = '';
  if (!seg) { $('#inspector').hidden = true; return; }
  $('#inspector').hidden = false;
  $('#inspTitle').textContent = seg.id;
  body.append(el('div', 'muted', `${m.scenes.get(seg.scene)?.label || ''}, ${m.lineCount(seg)} lines, ${seg.nodes.length} nodes`));
  seg.badges.forEach(b => body.append(el('span', `badge ${b.cls}`, b.text)));
  const outs = m.out.get(id) || [], ins = m.inc.get(id) || [];
  body.append(el('h3', null, 'Leads to'));
  if (outs.length) outs.forEach(e => body.append(linkRow(e, 'out')));
  else body.append(el('div', 'muted', seg.end ? 'The episode ends here.' : 'No outgoing links in the JSON.'));
  body.append(el('h3', null, 'Script'));
  let hlEl = null;
  seg.nodes.forEach(n => {
    const r = renderNode(n, true);
    if (isObj(n) && n.offset != null) r.append(el('span', 'off', `@${n.offset}`));
    if (isObj(n) && n.image != null) r.append(el('span', 'off', `img ${n.image}`));
    if (hlText && isObj(n) && str(n.text) === hlText) { r.classList.add('hl'); hlEl = hlEl || r; }
    body.append(r);
  });
  body.append(el('h3', null, 'Reached from'));
  if (ins.length) ins.forEach(e => body.append(linkRow(e, 'in')));
  else body.append(el('div', 'muted', id === m.entry ? 'Episode start.' : 'Nothing links here.'));
  const det = el('details'); det.append(el('summary', null, 'Raw JSON'));
  det.append(el('pre', null, JSON.stringify(seg.raw || { nodes: seg.nodes }, null, 1)));
  body.append(det);
  if (hlEl) hlEl.scrollIntoView({ block: 'center' });
}

function select(id, opts = {}) {
  state.sel = id;
  applySelection();
  if (id && state.model.segs.has(id)) openInspector(id, opts.hlText);
  else $('#inspector').hidden = true;
  writeHash();
}

function goToSeg(id, hlText) {
  const m = state.model;
  if (!m.segs.has(id)) return;
  const sc = m.sceneOf(id);
  state.sel = id;
  if (state.view === 'overview' || (state.view === 'scene' && state.scene !== sc)) {
    state.view = 'scene'; state.scene = sc; syncControls(); render();
  }
  select(id, { hlText });
  centerOn(id, true);
}

// ------------------------------------------------------------------ side panels
function renderSide() {
  const m = state.model;
  const sp = $('#scenesPanel'); sp.textContent = '';
  const vp = $('#varsPanel'); vp.textContent = '';
  $('#epInfo').textContent = '';
  if (!m) { sp.append(el('p', 'muted', 'No episode loaded.')); return; }
  for (const sc of m.scenes.values()) {
    const b = el('button', 'list-item');
    b.append(el('span', 't', sc.label));
    const lines = sc.segIds.reduce((a, id) => a + m.lineCount(m.segs.get(id)), 0);
    b.append(el('span', 's', `${sc.segIds.length} segments, ${lines} lines` + (sc.meta.gate ? `, runs when ${fmtCond(sc.meta.gate)}` : '')));
    b.setAttribute('aria-current', state.view === 'scene' && state.scene === sc.id ? 'true' : 'false');
    b.onclick = () => { state.view = 'scene'; state.scene = sc.id; state.sel = null; syncControls(); render(); renderSide(); writeHash(); closeSideNarrow(); };
    sp.append(b);
  }
  if (!m.vars.size) vp.append(el('p', 'muted', 'No variable writes or checks found in this file.'));
  const order = { check: 0, set: 1, add: 2, read: 3 };
  [...m.vars.keys()].sort(numSort).forEach(name => {
    const blk = el('div', 'var-block'); blk.append(el('h3', null, name));
    m.vars.get(name).slice().sort((a, b) => order[a.kind] - order[b.kind]).forEach(r => {
      const row = el('button', `list-item var-row ${r.kind === 'check' ? 'check' : 'set'}`);
      row.append(el('span', 'op', { check: 'checks ', set: 'sets ', add: 'changes ', read: 'reads ' }[r.kind]));
      row.append(document.createTextNode(`${r.text} in ${r.seg}`));
      row.onclick = () => { goToSeg(r.seg); closeSideNarrow(); };
      blk.append(row);
    });
    vp.append(blk);
  });
  const info = $('#epInfo');
  info.append(el('div', null, `${m.segs.size} segments, ${m.edges.length} links, ${m.scenes.size} scenes.`));
  if (m.missing.length) info.append(el('div', null, `${m.missing.length} links point at segments that are not in the file.`));
  m.notes.forEach(n => info.append(el('div', 'note', n)));
}

function runSearch() {
  const q = $('#searchInput').value.trim().toLowerCase();
  const out = $('#searchResults'); out.textContent = '';
  const m = state.model; if (!m || q.length < 2) return;
  let count = 0;
  for (const seg of m.segs.values()) {
    const segHit = seg.id.toLowerCase().includes(q);
    for (const n of seg.nodes) {
      if (!isObj(n)) continue;
      const text = str(n.text), who = str(n.speaker);
      if (!segHit && !text.toLowerCase().includes(q) && !who.toLowerCase().includes(q)) continue;
      const b = el('button', 'list-item result');
      b.append(el('span', 't', `${who || (n.type === 'narration' ? 'Narration' : n.type)} in ${seg.id}`));
      const s = el('span', 's'); const i = text.toLowerCase().indexOf(q);
      if (i >= 0) { s.append(text.slice(0, i)); s.append(el('mark', null, text.slice(i, i + q.length))); s.append(trunc(text.slice(i + q.length), 120)); }
      else s.append(trunc(text, 120));
      b.append(s);
      b.onclick = () => { goToSeg(seg.id, text); closeSideNarrow(); };
      out.append(b);
      if (++count >= 200) { out.append(el('p', 'muted', 'Showing the first 200 matches.')); return; }
      if (segHit) break;
    }
  }
  if (!count) out.append(el('p', 'muted', 'No matches.'));
}

// ------------------------------------------------------------------ controls + hash
function syncControls() {
  document.querySelectorAll('#viewSeg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
  const ss = $('#sceneSelect');
  ss.hidden = state.view !== 'scene';
  if (state.model && ss.options.length !== state.model.scenes.size) {
    ss.textContent = '';
    for (const sc of state.model.scenes.values()) { const o = el('option', null, sc.label); o.value = sc.id; ss.append(o); }
  }
  if (state.scene != null) ss.value = state.scene;
}

function writeHash() {
  const p = new URLSearchParams();
  if (state.epPath) p.set('ep', state.epPath);
  p.set('view', state.view);
  if (state.view === 'scene' && state.scene != null) p.set('scene', state.scene);
  if (state.sel) p.set('seg', state.sel);
  history.replaceState(null, '', '#' + p.toString());
}
function readHash() { return new URLSearchParams(location.hash.slice(1)); }

function loadModel(raw, name, epPath, fromHash) {
  let m;
  try { m = new Model(raw, name); }
  catch (err) { showError(`${name}: ${err.message}`); return; }
  showError('');
  state.model = m; state.epPath = epPath || null; state.sel = null;
  state.view = 'overview'; state.scene = m.sceneOf(m.entry) ?? m.scenes.keys().next().value;
  if (fromHash) {
    const h = readHash();
    if (['overview', 'scene', 'all'].includes(h.get('view'))) state.view = h.get('view');
    if (h.get('scene') && m.scenes.has(h.get('scene'))) state.scene = h.get('scene');
    if (h.get('seg') && m.segs.has(h.get('seg'))) { state.sel = h.get('seg'); state.scene = m.sceneOf(state.sel); if (state.view === 'overview') state.view = 'scene'; }
  }
  document.title = `${m.title} | SHS Story Map`;
  $('#inspector').hidden = true;
  syncControls(); renderSide(); render();
  if (state.sel) select(state.sel);
  writeHash();
}

function showError(msg) { $('#loadErr').textContent = msg; if (msg) { state.model = null; render(); } }

async function loadUrl(path, fromHash) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    loadModel(await r.json(), path.split('/').pop(), path, fromHash);
    $('#epSelect').value = path;
  } catch (err) { showError(`Could not load ${path}: ${err.message}`); }
}

function loadFile(file) {
  const rd = new FileReader();
  rd.onload = () => {
    try { loadModel(JSON.parse(rd.result), file.name, null, false); $('#epSelect').value = ''; }
    catch (err) { showError(`${file.name} is not valid JSON: ${err.message}`); }
  };
  rd.readAsText(file);
}

async function loadManifest() {
  try {
    const r = await fetch('episodes/index.json', { cache: 'no-cache' });
    if (!r.ok) return [];
    const list = await r.json();
    const sel = $('#epSelect');
    (Array.isArray(list) ? list : list.episodes || []).forEach(it => {
      const file = typeof it === 'string' ? it : it.file;
      const o = el('option', null, typeof it === 'string' ? it : (it.title || it.file));
      o.value = file.includes('/') ? file : 'episodes/' + file;
      sel.append(o);
    });
    return [...sel.options].slice(1).map(o => o.value);
  } catch { return []; }
}

function closeSideNarrow() { $('#side').classList.remove('open'); }

// ------------------------------------------------------------------ events
function initEvents() {
  $('#epSelect').onchange = e => { if (e.target.value) { history.replaceState(null, '', '#'); loadUrl(e.target.value, false); } };
  $('#fileInput').onchange = e => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ''; };
  document.querySelectorAll('#viewSeg button').forEach(b => b.onclick = () => {
    if (!state.model) return;
    state.view = b.dataset.view; if (state.view === 'overview') state.sel = null;
    $('#inspector').hidden = !state.sel;
    syncControls(); render(); renderSide(); writeHash();
  });
  $('#sceneSelect').onchange = e => { state.scene = e.target.value; state.sel = null; $('#inspector').hidden = true; render(); renderSide(); writeHash(); };
  $('#cuesToggle').onchange = e => { state.showCues = e.target.checked; render({ keepView: true }); };
  $('#capSelect').onchange = e => { state.cap = +e.target.value; render({ keepView: true }); };
  $('#themeBtn').onclick = () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('shs-map-theme', next); } catch {}
  };
  $('#inspClose').onclick = () => select(null);
  $('#toggleSide').onclick = () => $('#side').classList.toggle('open');
  $('#legendToggle').onclick = () => {
    const lg = $('#legend'); lg.classList.toggle('collapsed');
    $('#legendToggle').setAttribute('aria-expanded', String(!lg.classList.contains('collapsed')));
  };
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    document.querySelectorAll('.tabpanel').forEach(p => (p.hidden = p.dataset.panel !== b.dataset.tab));
    if (b.dataset.tab === 'search') $('#searchInput').focus();
  });
  let st; $('#searchInput').oninput = () => { clearTimeout(st); st = setTimeout(runSearch, 150); };
  $('#zoomIn').onclick = () => zoomAt(1.25);
  $('#zoomOut').onclick = () => zoomAt(0.8);
  $('#zoomFit').onclick = fit;

  // pan / zoom / click
  const pts = new Map(); let drag = null, pinch = null;
  stage.addEventListener('pointerdown', e => {
    if (e.target.closest('.legend, .zoom, button, select, input, a')) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) drag = { x: e.clientX, y: e.clientY, tx: state.tx, ty: state.ty, moved: false, target: e.target };
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), k: state.k }; drag = null;
    }
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch && pts.size === 2) {
      const [a, b] = [...pts.values()]; const r = stage.getBoundingClientRect();
      zoomAt((pinch.k * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d) / state.k, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      return;
    }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; stage.classList.add('panning'); }
    if (drag.moved) { state.tx = drag.tx + dx; state.ty = drag.ty + dy; applyTransform(); }
  });
  const end = e => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    if (drag && pts.size === 0) {
      stage.classList.remove('panning');
      if (!drag.moved) onClick(drag.target);
      drag = null;
    }
  };
  stage.addEventListener('pointerup', end);
  stage.addEventListener('pointercancel', end);
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - r.left, e.clientY - r.top);
    else { state.tx -= e.deltaX; state.ty -= e.deltaY; applyTransform(); }
  }, { passive: false });
  stage.addEventListener('dblclick', e => {
    const c = e.target.closest('.card');
    if (!c) { const r = stage.getBoundingClientRect(); zoomAt(1.4, e.clientX - r.left, e.clientY - r.top); }
  });

  document.addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === '+' || e.key === '=') zoomAt(1.2);
    else if (e.key === '-') zoomAt(1 / 1.2);
    else if (e.key === '0') fit();
    else if (e.key === 'Escape') select(null);
    else if (e.key === '/') { e.preventDefault(); document.querySelector('.tabs [data-tab="search"]').click(); }
    else if (e.key.startsWith('Arrow') && document.activeElement === stage) {
      const d = 60; state.tx += e.key === 'ArrowLeft' ? d : e.key === 'ArrowRight' ? -d : 0;
      state.ty += e.key === 'ArrowUp' ? d : e.key === 'ArrowDown' ? -d : 0; applyTransform(); e.preventDefault();
    }
  });

  // drag and drop files
  let depth = 0;
  window.addEventListener('dragenter', e => { if (e.dataTransfer?.types?.includes('Files')) { depth++; $('#dropveil').classList.add('on'); } });
  window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) $('#dropveil').classList.remove('on'); });
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => {
    e.preventDefault(); depth = 0; $('#dropveil').classList.remove('on');
    const f = e.dataTransfer.files[0]; if (f) loadFile(f);
  });
  window.addEventListener('resize', () => { if (!state.model) return; });
}

function onClick(target) {
  const lbl = target.closest('.elabel');
  if (lbl) { goToSeg(lbl.dataset.to); return; }
  const c = target.closest('.card');
  if (!c) { if (state.sel) select(null); return; }
  if (c.dataset.scene != null && c.classList.contains('scene')) {
    state.view = 'scene'; state.scene = c.dataset.scene; state.sel = null;
    syncControls(); render(); renderSide(); writeHash(); return;
  }
  if (c.classList.contains('portal')) {
    if (c.dataset.target) goToSeg(c.dataset.target);
    return;
  }
  select(c.dataset.id === state.sel ? null : c.dataset.id);
}

function buildLegend() {
  const ul = $('#legendList');
  const show = ['next', 'option', 'then', 'else', 'win', 'lose', 'goto'];
  for (const k of show) {
    const li = el('li');
    li.innerHTML = `<svg width="34" height="10"><line x1="1" y1="5" x2="33" y2="5" stroke-width="2.2" style="stroke:var(${KINDS[k].color})" ${KINDS[k].dash ? `stroke-dasharray="${KINDS[k].dash}"` : ''}/></svg>`;
    li.append(el('span', null, KINDS[k].label));
    ul.append(li);
  }
  const li = el('li');
  li.innerHTML = `<svg width="34" height="10"><line x1="1" y1="5" x2="33" y2="5" stroke-width="2.2" style="stroke:var(--k-next)" stroke-dasharray="${INFERRED_DASH}"/></svg>`;
  li.append(el('span', null, 'Inferred link (older format)'));
  ul.append(li);
}

// ------------------------------------------------------------------ boot
(async function boot() {
  try { const t = localStorage.getItem('shs-map-theme'); if (t) document.documentElement.dataset.theme = t; } catch {}
  buildLegend(); initEvents(); syncControls(); renderSide(); render();
  const eps = await loadManifest();
  const h = readHash();
  if (h.get('ep')) loadUrl(h.get('ep'), true);
  else if (eps.length === 1) loadUrl(eps[0], false);
})();
