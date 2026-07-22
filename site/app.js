/* Deuce - static dashboard.
 *
 * Everything here is a renderer. No model runs in the browser: the exporter
 * (src/serving/export_static.py) precomputes every probability, simulation and
 * SHAP attribution, and this file only fetches and draws the result.
 */
'use strict';

const ROUND_ORDER = ['first round', 'second round', 'third round',
                     'quarter-finals', 'semi-finals', 'final'];

const ROUND_LABEL = {
  'first round': 'Round 1', 'second round': 'Round 2', 'third round': 'Round 3',
  'quarter-finals': 'Quarter-finals', 'semi-finals': 'Semi-finals', 'final': 'Final',
};

// Bracket column heading: the named rounds by name, everything earlier by the
// size of the field it starts with. Derived from the match count rather than
// hardcoded, so a 64-draw reads R64 and an irregular draw stays honest.
function roundCode(round, nMatches) {
  if (round === 'final') return 'F';
  if (round === 'semi-finals') return 'SF';
  if (round === 'quarter-finals') return 'QF';
  return 'R' + (nMatches * 2);
}

// Column headings for the advancement table, where width is scarce.
const ROUND_SHORT = {
  'first round': 'R1', 'second round': 'R2', 'third round': 'R3',
  'quarter-finals': 'QF', 'semi-finals': 'SF', 'final': 'Final',
};

// Nation -> ISO-3166 alpha-2, for the regional-indicator flag. Covers every
// nationality present in the data; anything unmapped falls back to plain text.
const ISO = {
  'Australia': 'AU', 'Austria': 'AT', 'Azerbaijan': 'AZ', 'Bahrain': 'BH',
  'Belgium': 'BE', 'Brazil': 'BR', 'Bulgaria': 'BG', 'Canada': 'CA',
  'China': 'CN', 'Chinese Taipei': 'TW', 'Croatia': 'HR', 'Czech Republic': 'CZ',
  'Denmark': 'DK', 'Egypt': 'EG', 'England': 'GB', 'Estonia': 'EE',
  'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Guatemala': 'GT',
  'Hong Kong': 'HK', 'Hungary': 'HU', 'Iceland': 'IS', 'India': 'IN',
  'Indonesia': 'ID', 'Iran': 'IR', 'Ireland': 'IE', 'Israel': 'IL',
  'Italy': 'IT', 'Japan': 'JP', 'Kazakhstan': 'KZ', 'Latvia': 'LV',
  'Lithuania': 'LT', 'Macau': 'MO', 'Malaysia': 'MY', 'Maldives': 'MV',
  'Mexico': 'MX', 'Mongolia': 'MN', 'Nepal': 'NP', 'Netherlands': 'NL',
  'New Zealand': 'NZ', 'Nigeria': 'NG', 'Norway': 'NO', 'Peru': 'PE',
  'Poland': 'PL', 'Portugal': 'PT', 'Republic of Ireland': 'IE',
  'Russia': 'RU', 'Saudi Arabia': 'SA', 'Scotland': 'GB', 'Singapore': 'SG',
  'Slovakia': 'SK', 'Slovenia': 'SI', 'South Africa': 'ZA', 'South Korea': 'KR',
  'Spain': 'ES', 'Sri Lanka': 'LK', 'Sweden': 'SE', 'Switzerland': 'CH',
  'Syria': 'SY', 'Thailand': 'TH', 'Turkey': 'TR', 'Ukraine': 'UA',
  'United Arab Emirates': 'AE', 'United States': 'US', 'Vietnam': 'VN',
  'Wales': 'GB',
};

// promote.py's winner flips between libraries, so the payload carries the
// name and this only prettifies it. A point-in-time payload labels itself
// "xgb (point-in-time)"; the qualifier is shown separately, on its own line.
const MODEL_LABEL = {
  xgb: 'XGBoost', lgbm: 'LightGBM', catboost: 'CatBoost', tabnet: 'TabNet',
};
const modelName = raw => {
  const key = String(raw || '').split(' ')[0].toLowerCase();
  return MODEL_LABEL[key] || (raw ? String(raw) : 'Gradient boosting');
};

const TIER_LABEL = {
  100: 'Super 100', 300: 'Super 300', 500: 'Super 500',
  750: 'Super 750', 1000: 'Super 1000', 1500: 'Finals / Majors',
};

// Short form for the chip, where the full "Super 750" does not earn its width.
const TIER_CHIP = {
  100: 'S100', 300: 'S300', 500: 'S500',
  750: 'S750', 1000: 'S1000', 1500: 'Finals',
};

// ---------------------------------------------------------------- utilities

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function flag(nat) {
  const code = ISO[nat];
  if (!code) return '';
  return String.fromCodePoint(...[...code].map(c => 0x1f1a5 + c.charCodeAt(0)));
}

// Whole percent is the right resolution for a single match, where the second
// digit is well inside the model's error. It is the wrong resolution for a
// ranked list: the championship tail bunches under 10%, and rounding turned
// eleven distinct players into an undifferentiated wall of "1%". Callers
// showing a ranking pass fine=true to keep one decimal through the tail.
const pct = (p, fine) =>
  (p * 100).toFixed(p >= 0.995 || p < 0.005 || (fine && p < 0.095) ? 1 : 0) + '%';

// A published draw can name a slot before the qualifier is known. The model
// still scores it - against an unknown player it has no history for, so the
// number is an artifact of the defaults rather than a prediction. Show the
// slot, withhold the probability.
const isPlaceholder = name => /\bTBD\b|qualifier/i.test(name || '');

function fmtDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d))
    .toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

const _cache = new Map();
async function getJSON(path) {
  if (_cache.has(path)) return _cache.get(path);
  const p = fetch(path).then(r => (r.ok ? r.json() : null)).catch(() => null);
  _cache.set(path, p);
  return p;
}

// ---------------------------------------------------------------- state

const state = {
  index: [],
  players: [],
  view: 'tournaments',   // 'tournaments' | 'matchup'
  slug: null,
  tab: 'bracket',        // 'bracket' | 'monte'
  doc: null,
  selected: null,        // index into doc.matches
  lbMode: 'live',        // 'live' | 'pre' - only meaningful while a draw is live
  season: 'all',
  query: '',
  navCollapsed: false,   // the tournament list; collapsed only on request
  mu: { a: null, b: null },
};

// The one breakpoint the script has to know about: below it the explanation is
// a modal sheet rather than a rail, which changes where focus belongs. Keep it
// in step with the 820px query in styles.css.
const SHEET = matchMedia('(max-width: 820px)');

/**
 * Stacked, the sidebar sits above the content rather than beside it, so
 * picking a tournament or a player changes only what is off the bottom of the
 * screen - the list you tapped looks unchanged. Bring the answer into view.
 * On a wide screen both are already visible and this does nothing.
 */
function revealMain() {
  if (!SHEET.matches) return;
  $('#main').scrollIntoView({
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    block: 'start',
  });
}

// ---------------------------------------------------------------- sidebar

function seasons() {
  return [...new Set(state.index.map(t => t.date.slice(0, 4)))].sort().reverse();
}

function visibleTournaments() {
  const q = state.query.trim().toLowerCase();
  return state.index.filter(t =>
    (state.season === 'all' || t.date.startsWith(state.season)) &&
    (!q || t.name.toLowerCase().includes(q) || (t.host || '').toLowerCase().includes(q)));
}

function renderSidebar() {
  syncSeasonPicker();
  for (const b of document.querySelectorAll('.nav button'))
    b.setAttribute('aria-selected', String(b.dataset.view === state.view));
  $('#browse').hidden = state.view !== 'tournaments';
  $('#ratings').hidden = state.view !== 'matchup';
  if (state.view === 'matchup') { renderRatings(); return; }

  const list = $('#tlist');
  list.textContent = '';
  const rows = visibleTournaments();
  if (!rows.length) {
    list.append(el('p', 'empty', 'No tournaments match.'));
    return;
  }
  for (const t of rows) {
    const b = el('button', 'titem');
    b.type = 'button';
    b.setAttribute('aria-current', String(t.slug === state.slug));
    b.append(el('span', 't-name', t.name));
    const meta = el('div', 't-meta');
    // The chip rather than a coloured dot: same information, but it says which
    // tier instead of requiring the colour to be decoded.
    const chip = el('span', `tier-chip sm tier-${t.tier}`, TIER_CHIP[t.tier] || t.tier);
    chip.title = TIER_LABEL[t.tier] || '';
    meta.append(chip);
    meta.append(el('span', 'num', fmtDate(t.date)));
    if (t.status !== 'complete') meta.append(el('span', `pill ${t.status}`, t.status));
    b.append(meta);
    b.onclick = () => { location.hash = `#/t/${t.slug}`; revealMain(); };
    list.append(b);
  }
}

function renderRatings() {
  const list = $('#rlist');
  if (!list.dataset.filled) {
    list.dataset.filled = '1';
    state.players.slice(0, 40).forEach((p, i) => {
      const b = el('button', 'titem');
      b.type = 'button';
      b.dataset.slug = p.slug;
      const line = el('span', 't-name');
      line.append(el('span', 'seed num', String(i + 1) + ' '));
      line.append(document.createTextNode(`${flag(p.nat)} ${p.name}`.trim()));
      b.append(line);
      const meta = el('div', 't-meta');
      meta.append(el('span', 'num', `${Math.round(p.elo)} rating`));
      b.append(meta);
      b.onclick = () => pickPlayer(p);
      list.append(b);
    });
  }
  // Which two are in the comparison, marked on every render rather than at
  // build time - the rows are built once and reused.
  const picked = new Set([state.mu.a?.slug, state.mu.b?.slug].filter(Boolean));
  for (const b of list.children)
    b.setAttribute('aria-current', String(picked.has(b.dataset.slug)));
}

/**
 * The comparison is always between the last two players clicked. Slot B used
 * to be the only one a click could overwrite, which pinned the first pick
 * forever: clicking A, B, C gave A vs C, and there was no way to reach B vs C
 * without clearing the box by hand. Keeping the newest pick in B and pushing
 * the previous one into A means a run of clicks walks the comparison forward.
 */
function pickPlayer(p) {
  const prev = state.mu.b || state.mu.a;
  if (prev && prev.slug === p.slug) return;      // already the newest pick
  state.mu = prev ? { a: prev, b: p } : { a: p, b: null };
  goMatchup();
  revealMain();
}

// ---------------------------------------------------------------- bracket

function groupRounds(matches) {
  // Export order is the scraper's order, which is true bracket order within a
  // round - round-N winners feed round N+1 in pairs. Preserve it exactly.
  const by = new Map();
  matches.forEach((m, i) => {
    if (!by.has(m.round)) by.set(m.round, []);
    by.get(m.round).push({ ...m, i });
  });
  return ROUND_ORDER.filter(r => by.has(r)).map(r => [r, by.get(r)]);
}

function matchCard(m) {
  const called = m.pending ? null : (m.p > 0.5) === !!m.a_won;
  // No called/missed edge on a retirement: the match was decided by injury,
  // not by the two players' relative strength, so scoring the model on it
  // either way would be reading meaning into a coin the model never tossed.
  const node = el('button', 'match'
    + (m.pending ? ' pending' : m.wo ? '' : called ? ' hit' : ' miss'));
  node.type = 'button';
  node.dataset.mi = m.i;                 // so closing can hand focus back
  node.setAttribute('aria-pressed', String(state.selected === m.i));
  const unknown = isPlaceholder(m.a) || isPlaceholder(m.b);

  const side = (who, name, nat, seed, prob, won) => {
    const s = el('div', `side ${who}` + (won === true ? ' won' : won === false ? ' lost' : ''));
    s.style.setProperty('--fill', unknown ? '0%' : (prob * 100).toFixed(1) + '%');
    s.append(el('span', 'seed num', seed ? String(seed) : ''));
    const n = el('span', 'pname');
    const f = flag(nat);
    if (f) { const fl = el('span', 'flag', f); fl.title = nat; n.append(fl); }
    n.append(document.createTextNode(name));
    s.append(n);
    // Its own grid cell rather than part of the name, so a long name
    // ellipsises without ever truncating the one mark that says who advanced.
    s.append(el('span', 'winmark', won === true ? '✓' : ''));
    const p = el('span', 'prob num', unknown ? '\u2013' : pct(prob));
    if (unknown) p.title = 'Opponent not yet determined';
    s.append(p);
    return s;
  };

  const aWon = m.pending ? null : m.a_won;
  node.append(side('a', m.a, m.a_nat, m.a_seed, m.p, aWon));
  node.append(side('b', m.b, m.b_nat, m.b_seed, 1 - m.p, aWon === null ? null : !aWon));

  const foot = el('div', 'match-foot');
  if (unknown) {
    foot.append(el('span', null, 'Opponent not yet determined'));
  } else if (m.pending) {
    foot.append(el('span', null, 'Not yet played'));
  } else {
    foot.append(el('span', 'score', m.score || ''));
    if (m.wo) {
      // A partial score with no explanation reads as a data error. Which of
      // the two it was is decided by the score: a retirement leaves one, a
      // true walkover leaves nothing behind.
      const tag = el('span', 'wo', m.score ? 'retired' : 'walkover');
      tag.title = m.score
        ? 'One player retired. This result did not update either player’s rating or form.'
        : 'Walkover. This result did not update either player’s rating or form.';
      foot.append(tag);
    }
    // The coloured left edge carries called-vs-missed now, so the label is
    // gone from the footer - but colour alone is not readable for everyone,
    // and green/claret is the worst pair for it. Keep the fact in the
    // accessible name, where it costs nothing visually.
    if (!m.wo) {
      node.title = called
        ? 'The model called this one.'
        : 'The model missed this one.';
    }
  }
  node.append(foot);

  node.onclick = () => { selectMatch(state.selected === m.i ? null : m.i); };

  const slot = el('div', 'slot');
  slot.append(node);
  return slot;
}

function selectMatch(i) {
  const closed = i == null ? state.selected : null;
  state.selected = i;
  const done = renderMain();
  // Closing leaves focus on a button that the re-render has just destroyed,
  // which drops the caret back to the top of the document. Put it on the match
  // the explanation was about - where the reader already was.
  if (closed != null) {
    Promise.resolve(done).then(() => {
      document.querySelector(`.match[data-mi="${closed}"]`)?.focus();
    });
  }
}

function applyNav() {
  document.querySelector('.app').classList.toggle('nav-collapsed', state.navCollapsed);
  const b = $('#navtoggle');
  b.textContent = state.navCollapsed ? '\u203A' : '\u2039';
  const label = state.navCollapsed ? 'Show tournament list' : 'Collapse tournament list';
  b.title = label;
  b.setAttribute('aria-label', label);
  b.setAttribute('aria-expanded', String(!state.navCollapsed));
}

function renderBracket(doc) {
  const wrap = el('div');
  const scroller = el('div', 'bracket-wrap');
  const bracket = el('div', 'bracket');

  for (const [round, ms] of groupRounds(doc.matches)) {
    const col = el('div', 'round');
    const hd = el('div', 'round-name');
    const code = el('span', null, roundCode(round, ms.length));
    code.title = ROUND_LABEL[round] || round;
    hd.append(code);
    hd.append(el('span', 'n', String(ms.length)));
    col.append(hd);
    const body = el('div', 'round-body');
    // Wrap adjacent matches in pairs so the connectors can join the two
    // feeders of each next-round match. An odd tail (a bye, or a half-scraped
    // draw) gets a pair of one rather than being dropped.
    for (let i = 0; i < ms.length; i += 2) {
      const pair = el('div', 'pair');
      for (const m of ms.slice(i, i + 2)) pair.append(matchCard(m));
      body.append(pair);
    }
    col.append(body);
    bracket.append(col);
  }
  scroller.append(bracket);

  // The draw is a panel with its own header, and the header carries the hint,
  // so the layout does not change shape when a match is selected.
  const card = el('div', 'card-box');
  const chead = el('div', 'card-head');
  chead.append(el('span', 'lbl', 'Men’s singles bracket'));
  chead.append(el('span', 'hint',
    'Left edge: called or missed · ✓ advanced · click any match'));
  card.append(chead);
  card.append(scroller);

  const open = state.selected != null && doc.matches[state.selected];
  if (!open) {
    wrap.append(card);
    return wrap;
  }

  // Selected: the explanation becomes a rail beside the draw, so the match and
  // the reasoning about it are on screen together. The left sidebar collapses
  // to pay for the width.
  const grid = el('div', 'draw-grid');
  const left = el('div', 'draw-main');
  left.append(card);
  grid.append(left);
  const rail = el('aside', 'rail');
  rail.append(renderExplain(doc.matches[state.selected]));
  // Narrow, the rail is a modal bottom sheet and this element is its scrim
  // (see styles.css), so a click that lands on the aside rather than on the
  // sheet inside it is a click on the backdrop. Wide, nothing can hit the
  // aside itself, and the handler never fires.
  rail.onclick = e => { if (e.target === rail) selectMatch(null); };
  grid.append(rail);
  wrap.append(grid);

  requestAnimationFrame(() => {
    // Opening the rail narrows the draw, which can leave the match you just
    // clicked scrolled off to the right - worst for a final, the rightmost
    // column. Bring it back after layout settles.
    const sel = document.querySelector('.match[aria-pressed="true"]');
    if (sel) sel.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    // As a sheet it is a new surface laid over the page, so focus belongs
    // inside it. As a rail it sits beside the draw, and pulling focus out of
    // the bracket you are reading would be wrong.
    if (SHEET.matches) document.querySelector('.explain-close')?.focus();
  });
  return wrap;
}

// ---------------------------------------------------------------- SHAP

function renderExplain(m) {
  const box = el('div', 'explain');

  const head = el('div', 'rail-head');
  head.append(el('span', 'lbl', 'Why this prediction'));
  const close = el('button', 'explain-close', '\u2715');
  close.type = 'button';
  close.title = 'Close';
  close.setAttribute('aria-label', 'Close explanation');
  close.onclick = () => { selectMatch(null); };
  head.append(close);
  box.append(head);

  const body = el('div', 'rail-body');
  box.append(body);

  if (isPlaceholder(m.a) || isPlaceholder(m.b)) {
    body.append(el('p', 'note',
      'One slot in this match is still open. The model can only score a named '
      + 'player it has history for, so there is no meaningful prediction to '
      + 'explain until the qualifier is decided.'));
    return box;
  }

  // The two players, each with the colour its bars use below, so the direction
  // of a bar needs no legend to decode.
  const won = m.pending ? null : m.a_won;
  const vs = el('div', 'rail-vs');
  const sideRow = (who, name, prob, lost) => {
    const r = el('div', 'rail-side' + (lost ? ' dim' : ''));
    r.append(el('i', `swatch ${who}`));
    r.append(el('span', 'rs-name', name));
    r.append(el('span', 'rs-p num', pct(prob)));
    return r;
  };
  vs.append(sideRow('a', m.a, m.p, won === false));
  vs.append(sideRow('b', m.b, 1 - m.p, won === true));
  body.append(vs);

  // What actually happened, and whether the model got there.
  const called = m.pending ? null : (m.p > 0.5) === !!m.a_won;
  const verdict = el('div', 'rail-verdict ' + (m.pending ? 'pend' : m.wo ? 'wo' : called ? 'hit' : 'miss'));
  if (m.pending) {
    verdict.append(el('span', null, `${ROUND_LABEL[m.round] || m.round} \u00b7 not yet played`));
  } else {
    const winner = m.a_won ? m.a : m.b;
    const line = el('span');
    line.append(el('b', null, winner));
    line.append(document.createTextNode(
      m.wo ? (m.score ? ' won, opponent retired' : ' won by walkover')
           : called ? ' won, as the model expected' : ' won, and the model called it wrong'));
    verdict.append(line);
    if (m.score) verdict.append(el('span', 'sc', m.score));
  }
  body.append(verdict);

  // Bars only: the log-odds figure meant nothing to anyone reading it, and the
  // length already carries the magnitude.
  const max = Math.max(...m.shap.map(d => Math.abs(d.s)), 1e-6);
  const rows = el('div', 'shap');
  for (const d of m.shap) {
    const row = el('div', 'shap-row');
    row.append(el('span', 'shap-f', d.f));
    const track = el('div', 'shap-track');
    const bar = el('i', 'shap-bar ' + (d.s >= 0 ? 'pos' : 'neg'));
    bar.style.width = ((Math.abs(d.s) / max) * 50).toFixed(2) + '%';
    bar.title = `${d.f}: ${(d.s >= 0 ? '+' : '') + d.s.toFixed(3)} log-odds`;
    track.append(bar);
    row.append(track);
    rows.append(row);
  }
  body.append(rows);

  const key = el('p', 'shap-key');
  key.append(el('b', null, '\u2190 ' + m.b));
  key.append(el('b', null, m.a + ' \u2192'));
  body.append(key);
  return box;
}

// ---------------------------------------------------------------- leaderboard

function championOf(doc) {
  const fin = doc.matches.filter(m => m.round === 'final' && !m.pending).pop();
  return fin ? (fin.a_won ? fin.a : fin.b) : null;
}

function renderMonte(doc) {
  const wrap = el('div');
  const hasLive = Array.isArray(doc.leaderboard_live);
  const mode = hasLive ? state.lbMode : 'pre';
  const board = mode === 'live' ? doc.leaderboard_live : doc.leaderboard;

  const head = el('div', 'lb-head');
  if (hasLive) {
    const tg = el('div', 'toggle');
    for (const [k, label] of [['live', 'Conditioned on results so far'], ['pre', 'Pre-tournament']]) {
      const b = el('button', null, label);
      b.type = 'button';
      b.setAttribute('aria-selected', String(mode === k));
      b.onclick = () => { state.lbMode = k; renderMain(); };
      tg.append(b);
    }
    head.append(tg);
  }
  head.append(el('span', 'note', `${doc.sims.toLocaleString()} simulated brackets`));
  wrap.append(head);

  if (!board || !board.length) {
    wrap.append(el('div', 'empty',
      'No simulation for this draw: the bracket on Wikipedia is incomplete, so a forecast would be guesswork. The matches above are still real.'));
    return wrap;
  }

  const champ = championOf(doc);
  const top = board[0].p || 1;
  const lb = el('div', 'lb');

  // How far each player gets, not just whether they win. The simulation always
  // knew this - it plays every round - and the columns make a favourite with a
  // brutal quarter visibly different from one with an easy path to the semis.
  const advRounds = doc.adv_rounds || [];
  const cols = `26px minmax(0, 1fr) 92px ${'50px '.repeat(advRounds.length)}56px`;

  if (advRounds.length) {
    const head = el('div', 'lb-row lb-cols');
    head.style.gridTemplateColumns = cols;
    head.append(el('span'), el('span'), el('span'));
    for (const r of advRounds) head.append(el('span', 'c', ROUND_SHORT[r] || r));
    head.append(el('span', 'c', 'Title'));
    lb.append(head);
  }

  board.slice(0, 24).forEach((r, i) => {
    const row = el('div', 'lb-row' + (r.name === champ ? ' champ' : ''));
    row.style.gridTemplateColumns = cols;
    row.append(el('span', 'rank num', String(i + 1)));
    const nm = el('span');
    const f = flag(r.nat);
    if (f) nm.append(el('span', 'flag', f + ' '));
    nm.append(document.createTextNode(r.name));
    if (r.name === champ) nm.append(el('span', 'nat', ' (actual winner)'));
    row.append(nm);
    const bar = el('div', 'lbar');
    const fill = el('div');
    fill.style.width = ((r.p / top) * 100).toFixed(2) + '%';
    bar.append(fill);
    row.append(bar);
    // Advancement is a marginal, not a share of anything, so it is shown as
    // plain rounded percent - no `fine`, which exists for the title tail.
    for (const a of (r.adv || [])) row.append(el('span', 'adv num', pct(a)));
    row.append(el('span', 'lpct num', pct(r.p, true)));
    lb.append(row);
  });
  wrap.append(lb);

  // Biggest misses: completed matches the winner was priced to lose. Derived
  // here rather than exported - everything it needs is already in `matches`,
  // so it costs no payload and no version bump.
  const upsets = doc.matches
    .filter(m => !m.pending && !isPlaceholder(m.a) && !isPlaceholder(m.b))
    .map(m => ({
      winner: m.a_won ? m.a : m.b,
      loser: m.a_won ? m.b : m.a,
      wp: m.a_won ? m.p : 1 - m.p,
      round: m.round,
      score: m.score,
    }))
    .filter(u => u.wp < 0.5)
    .sort((x, y) => x.wp - y.wp)
    .slice(0, 5);

  if (upsets.length) {
    const box = el('div', 'upsets');
    box.append(el('h3', null, 'Biggest misses'));
    for (const u of upsets) {
      const row = el('div', 'upset');
      const top = el('div', 'upset-top');
      top.append(el('span', 'w', u.winner));
      // Always one decimal: this list is ordered by exactly this number, and
      // at whole percent the top three all read "25%" while being ranked.
      top.append(el('span', 'p num', (u.wp * 100).toFixed(1) + '%'));
      row.append(top);
      row.append(el('div', 'upset-sub',
        `beat ${u.loser} · ${ROUND_LABEL[u.round] || u.round}${u.score ? ' · ' + u.score : ''}`));
      box.append(row);
    }
    wrap.append(box);
  }

  return wrap;
}

// ---------------------------------------------------------------- tournament

async function renderTournament() {
  const main = $('#main');
  const meta = state.index.find(t => t.slug === state.slug);
  const doc = state.doc;

  main.textContent = '';
  const head = el('div', 'thead');
  head.append(el('h2', null, doc ? doc.tournament : (meta ? meta.name : 'Unknown')));
  main.append(head);

  if (!doc) {
    main.append(el('div', 'empty',
      'No exported draw for this tournament: the source page had no usable bracket.'));
    return;
  }

  const sub = el('div', 'sub');
  const chip = el('span', `tier-chip tier-${doc.tier}`, TIER_CHIP[doc.tier] || doc.tier);
  chip.title = TIER_LABEL[doc.tier] || '';
  sub.append(chip);
  sub.append(el('span', null, fmtDate(doc.date)));
  sub.append(el('span', 'dot', '·'));
  sub.append(el('span', null, doc.host));
  if (doc.status !== 'complete') sub.append(el('span', `pill ${doc.status}`, doc.status));
  head.append(sub);

  const stats = el('div', 'statrow');
  const stat = (k, v, small, sub, cls) => {
    const s = el('div', 'stat' + (cls ? ' ' + cls : ''));
    s.append(el('div', 'k', k));
    // "41%" reads better as 41 with a small muted %, which is what the
    // artifact does with every figure it sets large.
    const val = el('div', 'v num');
    const mUnit = /^(.*?)(%|\/\d+)$/.exec(String(v));
    if (mUnit && !String(cls || '').includes('name')) {
      val.append(document.createTextNode(mUnit[1]));
      val.append(el('span', 'unit', mUnit[2]));
    } else {
      val.textContent = v;
    }
    if (small) val.append(el('small', null, ' ' + small));
    s.append(val);
    if (sub) s.append(el('div', 's', sub));
    return s;
  };

  // The people are the story, so they are the value and the number is the
  // caption - the other way round buried Kunlavut Vitidsarn under "41%".
  // Pre-tournament odds, not the results-conditioned board: the point of this
  // row is what the model thought before any of it happened.
  const pre = doc.leaderboard;
  const champ = championOf(doc);
  if (pre && pre.length) {
    stats.append(stat('Pre-tournament favourite', pct(pre[0].p, true), null,
      pre[0].name + (pre[0].nat ? ` \u00b7 ${pre[0].nat}` : '')));
  }

  // The champion's own pre-tournament price, next to the favourite's. It is
  // often a long shot, and saying so is the honest framing - a UI that only
  // showed the favourite would look better than the model deserves.
  if (champ) {
    const row = (pre || []).find(r => r.name === champ);
    stats.append(stat('Actual champion', row ? pct(row.p, true) : '\u2013', null,
      champ + (row && row.nat ? ` \u00b7 ${row.nat}` : ''), 'flag'));
  } else if (doc.leaderboard_live && doc.leaderboard_live.length) {
    // A draw still running has no champion, which left the row a box short and
    // its most interesting number off the header entirely: who the model likes
    // *now*, with the rounds already played taken as given. Same slot, same
    // treatment - the pair reads as "thought then, thinks now".
    const now = doc.leaderboard_live[0];
    // The old price is only worth the words when the lead has changed hands:
    // when it has not, the box to the left is already showing exactly it.
    const moved = pre && pre.length && pre[0].name !== now.name;
    const was = moved ? pre.find(r => r.name === now.name) : null;
    stats.append(stat('Favourite now', pct(now.p, true), null,
      now.name + (now.nat ? ` \u00b7 ${now.nat}` : '')
      + (was ? ` \u00b7 was ${pct(was.p, true)}` : ''), 'flag'));
  }

  const acc = doc.accuracy;
  if (acc.n) {
    stats.append(stat('Matches called', `${acc.hit}/${acc.n}`,
      `(${Math.round(100 * acc.hit / acc.n)}%)`,
      `of ${doc.matches.length} in the draw`, 'accent'));
  }
  const pit = doc.model === 'point-in-time';
  stats.append(stat('Model', modelName(doc.model_name), null,
    doc.trained_through
      ? `${pit ? 'trained on matches up to' : 'preloaded · trained to'} ${fmtDate(doc.trained_through)}`
      : (pit ? 'point-in-time' : 'preloaded'),
    'name'));
  head.append(stats);

  const tabs = el('div', 'tabs');
  for (const [k, label] of [['bracket', 'Draw & predictions'], ['monte', 'Championship odds']]) {
    const b = el('button', null, label);
    b.type = 'button';
    b.setAttribute('aria-selected', String(state.tab === k));
    b.onclick = () => goTab(k);
    tabs.append(b);
  }
  main.append(tabs);

  main.append(state.tab === 'bracket' ? renderBracket(doc) : renderMonte(doc));

}

// ---------------------------------------------------------------- matchup

function autocomplete(input, listBox, onPick) {
  let active = -1, items = [];
  const close = () => { listBox.textContent = ''; listBox.hidden = true; active = -1; };

  const open = () => {
    const q = input.value.trim().toLowerCase();
    items = state.players
      .filter(p => !q || p.name.toLowerCase().includes(q))
      .slice(0, 40);
    listBox.textContent = '';
    if (!items.length) { close(); return; }
    items.forEach((p, i) => {
      const b = el('button', i === active ? 'active' : null);
      b.type = 'button';
      b.append(document.createTextNode(`${flag(p.nat)} ${p.name}`.trim()));
      b.append(el('span', 'nat num', String(Math.round(p.elo))));
      b.onmousedown = e => { e.preventDefault(); input.value = p.name; close(); onPick(p); };
      listBox.append(b);
    });
    listBox.hidden = false;
  };

  input.addEventListener('input', () => { active = -1; open(); });
  input.addEventListener('focus', open);
  input.addEventListener('blur', () => setTimeout(close, 120));
  input.addEventListener('keydown', e => {
    if (listBox.hidden) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      active = Math.max(0, Math.min(items.length - 1, active + (e.key === 'ArrowDown' ? 1 : -1)));
      [...listBox.children].forEach((c, i) => c.classList.toggle('active', i === active));
      listBox.children[active]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && active >= 0) {
      e.preventDefault();
      const p = items[active];
      input.value = p.name; close(); onPick(p);
    } else if (e.key === 'Escape') { close(); }
  });
}

const CMP_ROWS = [
  ['elo', 'Rating', v => Math.round(v), 1300, 2050],
  ['recent_win_rate', 'Win rate, 180d', v => pct(v), 0, 1],
  ['ema', 'Form trend', v => v.toFixed(2), 0, 1],
  ['avg_point_diff', 'Avg point diff', v => v.toFixed(1), -8, 8],
  ['avg_margin', 'Avg winning margin', v => v.toFixed(1), 0, 14],
  ['rubber_rate', 'Three-game rate', v => pct(v), 0, 1],
];

function compareCard(A, B) {
  const card = el('div', 'card');
  card.append(el('h3', null, 'Form going in'));
  const cmp = el('div', 'cmp');

  const head = el('div', 'cmp-head');
  head.append(el('span', 'l', A.name));
  head.append(el('span', 'm', ''));
  head.append(el('span', 'r', B.name));
  cmp.append(head);

  for (const [key, label, fmt, lo, hi] of CMP_ROWS) {
    const row = el('div', 'cmp-row');
    const norm = v => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));

    const left = el('span', 'sv l');
    left.append(el('span', null, fmt(A[key])));
    const lm = el('span', 'meter'); const li = el('i');
    li.style.width = (norm(A[key]) * 100).toFixed(1) + '%'; lm.append(li);
    left.prepend(lm);
    row.append(left);

    row.append(el('span', 'lab', label));

    const right = el('span', 'sv r');
    const rm = el('span', 'meter'); const ri = el('i');
    ri.style.width = (norm(B[key]) * 100).toFixed(1) + '%'; rm.append(ri);
    right.append(el('span', null, fmt(B[key])));
    right.append(rm);
    row.append(right);

    cmp.append(row);
  }
  card.append(cmp);
  return card;
}

function formCard(p) {
  const card = el('div', 'card');
  card.append(el('h3', null, `${p.name}, last ${p.form.length}`));
  const strip = el('div', 'form-strip');
  for (const f of [...p.form].reverse()) {
    const c = el('div', 'form-chip ' + (f.won ? 'w' : 'l'));
    c.append(el('span', 'res', f.won ? 'W' : 'L'));
    c.append(el('span', null, f.opp));
    strip.append(c);
  }
  if (!p.form.length) strip.append(el('span', 'note', 'No recent matches on record.'));
  card.append(strip);

  // Derived from the chips above, not from the exported `streak` feature: that
  // one is a pre-match value, so it describes form going into the most recent
  // match and would contradict the result shown beside it.
  const recent = [...p.form].reverse();
  if (recent.length) {
    const won = recent[0].won;
    let n = 0;
    while (n < recent.length && recent[n].won === won) n++;
    const capped = n === recent.length;
    const word = won ? (n > 1 ? 'wins' : 'win') : (n > 1 ? 'losses' : 'loss');
    card.append(el('p', 'note', `${capped ? n + '+' : n} ${word} in a row.`));
  }
  return card;
}

async function renderMatchup() {
  const main = $('#main');
  main.textContent = '';

  const head = el('div', 'thead');
  head.append(el('h2', null, 'Matchup analyzer'));
  main.append(head);

  // Everything below shares one column width, so the picker strip, the
  // prediction bar and the cards line up down both edges.
  const wrap = el('div', 'mu');
  main.append(wrap);

  const picks = el('div', 'mu-picks');
  const mk = which => {
    const ac = el('div', 'ac');
    const input = document.createElement('input');
    input.type = 'search';
    input.placeholder = which === 'a' ? 'First player…' : 'Second player…';
    input.value = state.mu[which]?.name || '';
    const list = el('div', 'ac-list');
    list.hidden = true;
    ac.append(input, list);
    autocomplete(input, list, p => { state.mu[which] = p; goMatchup(); });
    return ac;
  };
  const swap = el('button', 'mu-swap', '⇄');
  swap.type = 'button';
  swap.title = 'Swap sides';
  swap.setAttribute('aria-label', 'Swap sides');
  swap.onclick = () => { state.mu = { a: state.mu.b, b: state.mu.a }; goMatchup(); };
  picks.append(mk('a'), swap, mk('b'));
  wrap.append(picks);

  const { a, b } = state.mu;
  if (!a || !b) {
    wrap.append(el('div', 'empty', a
      ? 'Pick a second player, or click one in the rating leaders.'
      : 'Pick two players, or click them in the rating leaders.'));
    return;
  }
  if (a.slug === b.slug) {
    wrap.append(el('div', 'empty', 'Pick two different players.'));
    return;
  }

  const [mu, A, B] = await Promise.all([
    getJSON(`data/matchup/${a.slug}.json`),
    getJSON(`data/player/${a.slug}.json`),
    getJSON(`data/player/${b.slug}.json`),
  ]);
  if (state.mu.a?.slug !== a.slug || state.mu.b?.slug !== b.slug) return;  // stale

  if (!mu || !A || !B || !mu.vs.find(v => v.slug === b.slug)) {
    wrap.append(el('div', 'empty', 'No exported matchup for this pair.'));
    return;
  }
  const hit = mu.vs.find(v => v.slug === b.slug);

  // One bar split at the model's probability, as in the design mockup: the
  // share of the width *is* the number, so the two are never read separately.
  const card = el('div', 'mu-head');
  card.append(el('div', 'mu-ctx', 'Neutral Super 750 quarter-final'));
  const names = el('div', 'mu-names');
  const nm = (cls, player) => {
    const d = el('div', 'n' + cls);
    d.append(document.createTextNode(player.name));
    d.append(el('small', null, player.nat || ''));
    return d;
  };
  names.append(nm('', A), nm(' r', B));
  card.append(names);

  const bar = el('div', 'mu-bar');
  const segA = el('i', 'a', pct(hit.p));
  segA.style.width = (hit.p * 100).toFixed(2) + '%';
  const segB = el('i', 'b', pct(1 - hit.p));
  segB.style.width = ((1 - hit.p) * 100).toFixed(2) + '%';
  bar.append(segA, segB);
  card.append(bar);
  wrap.append(card);

  wrap.append(compareCard(A, B));
  const forms = el('div', 'mu-forms');
  forms.append(formCard(A), formCard(B));
  wrap.append(forms);
}

// ---------------------------------------------------------------- routing

async function renderMain() {
  // Derived from state on every render rather than only in the click handler,
  // so the DOM cannot disagree with `navCollapsed` about whether it is open.
  applyNav();
  renderSidebar();
  if (state.view === 'matchup') return renderMatchup();
  if (!state.slug) {
    $('#main').textContent = '';
    $('#main').append(el('div', 'empty', 'Pick a tournament from the list.'));
    return;
  }
  return renderTournament();
}

async function route() {
  const h = location.hash.replace(/^#\/?/, '');
  const [kind, arg, arg2] = h.split('/').map(decodeURIComponent);

  if (kind === 'matchup') {
    state.view = 'matchup';
    const find = s => state.players.find(p => p.slug === s) || null;
    if (arg) state.mu.a = find(arg);
    if (arg2) state.mu.b = find(arg2);
  } else if (kind === 't' && arg) {
    state.view = 'tournaments';
    state.tab = arg2 === 'odds' ? 'monte' : 'bracket';
    if (arg !== state.slug) {
      state.slug = arg;
      state.selected = null;
      state.lbMode = 'live';
      state.doc = null;
      renderMain();                                  // header + sidebar while loading
      state.doc = await getJSON(`data/tournament/${arg}.json`);
      if (state.slug !== arg) return;                // navigated away mid-fetch
      if (state.doc) state.season = state.doc.date.slice(0, 4);
    }
  } else {
    state.view = 'tournaments';
  }
  renderMain();
}

/* Tab and player choices live in the URL so any view can be linked to. */
function goTab(tab) { location.hash = `#/t/${state.slug}/${tab === 'monte' ? 'odds' : 'draw'}`; }
function goMatchup() {
  const { a, b } = state.mu;
  location.hash = '#/matchup' + (a ? `/${a.slug}` : '') + (a && b ? `/${b.slug}` : '');
}

// ---------------------------------------------------------------- boot

const isDark = () => (document.documentElement.dataset.theme
  || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';

function initTheme() {
  const saved = localStorage.getItem('sc-theme');
  if (saved) document.documentElement.dataset.theme = saved;

  // The glyph shows what a click will switch *to*, which is the convention
  // people already read: a moon means "go dark".
  const paint = () => { $('#theme').textContent = isDark() ? '☀' : '☾'; };
  paint();

  $('#theme').onclick = () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('sc-theme', next);
    paint();
  };
  // Follow the system until the user has expressed a preference.
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!localStorage.getItem('sc-theme')) paint();
  });
}

// Set by initSeasonPicker once the index has loaded; a no-op before then so
// renderSidebar can call it unconditionally.
let syncSeasonPicker = () => {};

// A native <select> renders as the platform widget, which cannot be themed and
// looks like it belongs to a different application. This is the same control
// built from a button and a listbox so it inherits the palette.
function initSeasonPicker() {
  const btn = $('#seasonBtn'), menu = $('#seasonMenu'), label = $('#seasonLabel');
  const opts = [['all', 'All seasons'], ...seasons().map(y => [y, y])];

  const close = () => {
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  };
  const open = () => {
    menu.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    const on = menu.querySelector('[aria-selected="true"]');
    (on || menu.firstElementChild)?.focus();
  };

  // Opening a tournament sets the season to its year (see route), so the
  // control has to be able to follow state rather than only drive it.
  syncSeasonPicker = () => {
    const opt = opts.find(o => o[0] === state.season);
    label.textContent = opt ? opt[1] : state.season;
    for (const item of menu.children)
      item.setAttribute('aria-selected', String(item.dataset.value === state.season));
  };

  const pick = value => {
    state.season = value;
    syncSeasonPicker();
    close();
    btn.focus();
    renderSidebar();
  };

  for (const [value, text] of opts) {
    const item = el('button', 'dd-opt', text);
    item.type = 'button';
    item.dataset.value = value;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', String(value === state.season));
    item.onclick = () => pick(value);
    menu.append(item);
  }

  btn.onclick = () => (menu.hidden ? open() : close());

  // Arrow keys move between options, Escape closes, Enter picks. Without this
  // the control would be a mouse-only regression on the <select> it replaces.
  menu.onkeydown = e => {
    const items = [...menu.children];
    const i = items.indexOf(document.activeElement);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const next = e.key === 'ArrowDown' ? i + 1 : i - 1;
      items[(next + items.length) % items.length].focus();
    } else if (e.key === 'Escape') {
      close(); btn.focus();
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      (e.key === 'Home' ? items[0] : items[items.length - 1]).focus();
    }
  };
  btn.onkeydown = e => {
    if (e.key === 'ArrowDown' && menu.hidden) { e.preventDefault(); open(); }
  };
  document.addEventListener('click', e => {
    if (!menu.hidden && !$('#seasonDd').contains(e.target)) close();
  });
}

async function boot() {
  initTheme();

  $('#tsearch').oninput = e => { state.query = e.target.value; renderSidebar(); };
  $('#navtoggle').onclick = () => {
    state.navCollapsed = !state.navCollapsed;
    applyNav();
  };
  // Collapsed, the whole strip is the target. It is 45px of dead space
  // otherwise, and hitting a 32px button to get the list back is fussy.
  //
  // The toggle lives inside the sidebar, so a click that collapses it bubbles
  // straight into this handler, which would see the collapsed state and undo
  // it: the button looked dead because every press was cancelled by the press
  // itself. Clicks originating on the toggle belong to the toggle alone.
  document.querySelector('.sidebar').onclick = e => {
    if (!state.navCollapsed || e.target.closest('#navtoggle')) return;
    state.navCollapsed = false;
    applyNav();
  };
  // Escape closes the explanation. It is the only surface on the site that
  // covers content, and on a phone the backdrop is the only other way out.
  // The season menu handles its own Escape and does not stop the event, so
  // skip while it is open or one key press would close both.
  addEventListener('keydown', e => {
    if (e.key === 'Escape' && state.selected != null && $('#seasonMenu').hidden)
      selectMatch(null);
  });

  for (const b of document.querySelectorAll('.nav button')) {
    b.onclick = () => {
      if (b.dataset.view === 'matchup') goMatchup();
      else location.hash = state.slug ? `#/t/${state.slug}/draw` : '#/';
    };
  }

  const [index, players] = await Promise.all([
    getJSON('data/tournaments.json'),
    getJSON('data/players.json'),
  ]);
  if (!index) {
    $('#main').append(el('div', 'empty',
      'Could not load data/tournaments.json. Run `make export` first, then serve the site with `make site`. Opening index.html from the filesystem will not work.'));
    return;
  }
  state.index = index;
  state.players = (players || []).sort((a, b) => b.elo - a.elo);
  initSeasonPicker();   // needs the index: its options are the seasons in it

  addEventListener('hashchange', route);
  if (!location.hash) {
    // Default to whatever is live, else the most recent completed draw.
    const t = index.find(x => x.status === 'live') || index[0];
    if (t) { location.hash = `#/t/${t.slug}`; return; }
  }
  route();
}

boot();
