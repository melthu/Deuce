/* ShuttleCast — static dashboard.
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

const TIER_LABEL = {
  100: 'Super 100', 300: 'Super 300', 500: 'Super 500',
  750: 'Super 750', 1000: 'Super 1000', 1500: 'Finals / Majors',
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

const pct = p => (p * 100).toFixed(p >= 0.995 || p < 0.005 ? 1 : 0) + '%';

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
  lbMode: 'live',        // 'live' | 'pre' — only meaningful while a draw is live
  season: 'all',
  query: '',
  mu: { a: null, b: null },
};

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
  const sel = $('#season');
  if (!sel.dataset.filled) {
    sel.append(new Option('All seasons', 'all'));
    for (const y of seasons()) sel.append(new Option(y, y));
    sel.dataset.filled = '1';
  }
  sel.value = state.season;

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
    meta.append(el('span', `tier-dot tier-${t.tier}`));
    meta.append(el('span', 'num', fmtDate(t.date)));
    if (t.status !== 'complete') meta.append(el('span', `pill ${t.status}`, t.status));
    b.append(meta);
    b.onclick = () => { location.hash = `#/t/${t.slug}`; };
    list.append(b);
  }
}

function renderRatings() {
  const list = $('#rlist');
  if (list.dataset.filled) return;
  list.dataset.filled = '1';
  state.players.slice(0, 40).forEach((p, i) => {
    const b = el('button', 'titem');
    b.type = 'button';
    const line = el('span', 't-name');
    line.append(el('span', 'seed num', String(i + 1) + ' '));
    line.append(document.createTextNode(`${flag(p.nat)} ${p.name}`.trim()));
    b.append(line);
    const meta = el('div', 't-meta');
    meta.append(el('span', 'num', `${Math.round(p.elo)} rating`));
    b.append(meta);
    // First click fills the empty slot; later clicks replace player two.
    b.onclick = () => {
      if (!state.mu.a) state.mu.a = p;
      else if (state.mu.a.slug !== p.slug) state.mu.b = p;
      goMatchup();
    };
    list.append(b);
  });
}

// ---------------------------------------------------------------- bracket

function groupRounds(matches) {
  // Export order is the scraper's order, which is true bracket order within a
  // round — round-N winners feed round N+1 in pairs. Preserve it exactly.
  const by = new Map();
  matches.forEach((m, i) => {
    if (!by.has(m.round)) by.set(m.round, []);
    by.get(m.round).push({ ...m, i });
  });
  return ROUND_ORDER.filter(r => by.has(r)).map(r => [r, by.get(r)]);
}

function matchCard(m) {
  const node = el('button', 'match' + (m.pending ? ' pending' : ''));
  node.type = 'button';
  node.setAttribute('aria-pressed', String(state.selected === m.i));

  const side = (who, name, nat, seed, prob, won) => {
    const s = el('div', `side ${who}` + (won === true ? ' won' : won === false ? ' lost' : ''));
    s.style.setProperty('--fill', (prob * 100).toFixed(1) + '%');
    s.append(el('span', 'seed num', seed ? String(seed) : ''));
    const n = el('span', 'pname');
    const f = flag(nat);
    if (f) { const fl = el('span', 'flag', f); fl.title = nat; n.append(fl); }
    n.append(document.createTextNode(name));
    s.append(n);
    s.append(el('span', 'prob num', pct(prob)));
    return s;
  };

  const aWon = m.pending ? null : m.a_won;
  node.append(side('a', m.a, m.a_nat, m.a_seed, m.p, aWon));
  node.append(side('b', m.b, m.b_nat, m.b_seed, 1 - m.p, aWon === null ? null : !aWon));

  const foot = el('div', 'match-foot');
  if (m.pending) {
    foot.append(el('span', null, 'Not yet played'));
  } else {
    foot.append(el('span', 'score', m.score || ''));
    const called = (m.p > 0.5) === m.a_won;
    foot.append(el('span', called ? '' : 'missed', called ? '' : 'model missed'));
  }
  node.append(foot);

  node.onclick = () => { state.selected = state.selected === m.i ? null : m.i; renderMain(); };
  return node;
}

function renderBracket(doc) {
  const wrap = el('div');
  const scroller = el('div', 'bracket-wrap');
  const bracket = el('div', 'bracket');

  for (const [round, ms] of groupRounds(doc.matches)) {
    const col = el('div', 'round');
    col.append(el('div', 'round-name', `${ROUND_LABEL[round] || round} · ${ms.length}`));
    const body = el('div', 'round-body');
    for (const m of ms) body.append(matchCard(m));
    col.append(body);
    bracket.append(col);
  }
  scroller.append(bracket);
  wrap.append(scroller);

  if (state.selected != null && doc.matches[state.selected]) {
    wrap.append(renderExplain(doc.matches[state.selected]));
  } else {
    wrap.append(el('p', 'note',
      'Select any match to see which factors drove the prediction. Shaded fill behind each name is that player’s win probability.'));
  }
  return wrap;
}

// ---------------------------------------------------------------- SHAP

function renderExplain(m) {
  const box = el('div', 'explain');
  box.append(el('h3', null, `${m.a} vs ${m.b}`));

  const fav = m.p >= 0.5 ? m.a : m.b;
  const favP = m.p >= 0.5 ? m.p : 1 - m.p;
  let lede = `${ROUND_LABEL[m.round] || m.round} · model favours ${fav} at ${pct(favP)}.`;
  if (!m.pending) {
    const winner = m.a_won ? m.a : m.b;
    lede += ` ${winner} won${m.score ? ` ${m.score}` : ''}.`;
    if ((m.p > 0.5) !== m.a_won) lede += ' The model called this one wrong.';
  }
  box.append(el('p', 'lede', lede));

  const max = Math.max(...m.shap.map(d => Math.abs(d.s)), 1e-6);
  const rows = el('div', 'drivers');
  for (const d of m.shap) {
    const row = el('div', 'driver');
    row.append(el('span', 'dname', d.f));
    const track = el('div', 'track');
    const bar = el('i', 'bar');
    const w = (Math.abs(d.s) / max) * 50;
    if (d.s >= 0) { bar.style.left = '50%'; bar.style.background = 'var(--court)'; }
    else { bar.style.right = '50%'; bar.style.background = 'var(--cork)'; }
    bar.style.width = w.toFixed(2) + '%';
    track.append(bar);
    row.append(track);
    row.append(el('span', 'dval', (d.s >= 0 ? '+' : '') + d.s.toFixed(2)));
    rows.append(row);
  }
  box.append(rows);

  const key = el('div', 'driver-key');
  key.append(el('span', null, `← favours ${m.b}`));
  key.append(el('span', null, `favours ${m.a} →`));
  box.append(key);

  box.append(el('p', 'note',
    'Contributions are TreeSHAP values in log-odds, grouped from the model’s 34 raw features into nine drivers. SHAP is additive, so the grouped figures sum exactly to the model’s output.'));
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
      'No simulation for this draw — the bracket on Wikipedia is incomplete, so a forecast would be guesswork. The matches above are still real.'));
    return wrap;
  }

  const champ = championOf(doc);
  const top = board[0].p || 1;
  const lb = el('div', 'lb');
  board.slice(0, 24).forEach((r, i) => {
    const row = el('div', 'lb-row' + (r.name === champ ? ' champ' : ''));
    row.append(el('span', 'rank num', String(i + 1)));
    const nm = el('span');
    const f = flag(r.nat);
    if (f) nm.append(el('span', 'flag', f + ' '));
    nm.append(document.createTextNode(r.name));
    if (r.name === champ) nm.append(el('span', 'nat', ' — actual winner'));
    row.append(nm);
    const bar = el('div', 'lbar');
    const fill = el('div');
    fill.style.width = ((r.p / top) * 100).toFixed(2) + '%';
    bar.append(fill);
    row.append(bar);
    row.append(el('span', 'lpct num', pct(r.p)));
    lb.append(row);
  });
  wrap.append(lb);

  let note = 'Each simulation plays the bracket out round by round, sampling every match from the model’s probability and updating both players’ ratings inside the bracket before the next round.';
  if (mode === 'live') note += ' Matches already played are pinned to their real result in every simulation.';
  if (champ) note += ' The actual winner is highlighted — including when the model rated them a long shot.';
  wrap.append(el('p', 'note', note));
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
      'No exported draw for this tournament — the source page had no usable bracket.'));
    return;
  }

  const sub = el('div', 'sub');
  sub.append(el('span', null, fmtDate(doc.date)));
  sub.append(el('span', null, '·'));
  sub.append(el('span', null, `${TIER_LABEL[doc.tier] || doc.tier} · ${doc.host}`));
  if (doc.status !== 'complete') sub.append(el('span', `pill ${doc.status}`, doc.status));
  head.append(sub);

  const stats = el('div', 'statrow');
  const stat = (k, v, small) => {
    const s = el('div', 'stat');
    s.append(el('div', 'k', k));
    const val = el('div', 'v num', v);
    if (small) val.append(el('small', null, ' ' + small));
    s.append(val);
    return s;
  };
  const acc = doc.accuracy;
  if (acc.n) {
    stats.append(stat('Model called', `${acc.hit}/${acc.n}`,
      `(${Math.round(100 * acc.hit / acc.n)}%)`));
  }
  stats.append(stat('Matches', String(doc.matches.length)));
  const board = doc.leaderboard_live || doc.leaderboard;
  if (board && board.length) stats.append(stat('Favourite', board[0].name, pct(board[0].p)));
  stats.append(stat('Model', doc.model === 'point-in-time' ? 'Point-in-time' : 'Preloaded',
    doc.trained_through ? `to ${doc.trained_through}` : ''));
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

  const foot = el('div', 'foot');
  foot.textContent = doc.model === 'point-in-time'
    ? `Predictions come from a model trained only on matches before ${doc.date} (${(doc.n_train_rows || 0).toLocaleString()} rows, through ${doc.trained_through}). It has never seen this tournament or anything after it, so these are genuine out-of-sample calls.`
    : 'Too little history preceded this tournament to fit a point-in-time model, so the current promoted model was used. Treat its calls on this draw as in-sample.';
  main.append(foot);
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
  card.append(el('h3', null, `${p.name} — last ${p.form.length}`));
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
  head.append(el('div', 'sub', 'Any two active players, at a neutral Super 750 quarter-final.'));
  main.append(head);

  const pickers = el('div', 'mu-grid');
  const mk = (which, label) => {
    const card = el('div', 'card');
    card.append(el('h3', null, label));
    const ac = el('div', 'ac');
    const input = document.createElement('input');
    input.type = 'search';
    input.placeholder = 'Search players…';
    input.value = state.mu[which]?.name || '';
    const list = el('div', 'ac-list');
    list.hidden = true;
    ac.append(input, list);
    autocomplete(input, list, p => { state.mu[which] = p; goMatchup(); });
    card.append(ac);
    return card;
  };
  pickers.append(mk('a', 'Player one'), mk('b', 'Player two'));
  main.append(pickers);

  const { a, b } = state.mu;
  if (!a || !b) {
    main.append(el('p', 'note', 'Pick two players to see the model’s call.'));
    return;
  }
  if (a.slug === b.slug) {
    main.append(el('div', 'empty', 'Pick two different players.'));
    return;
  }

  const [mu, A, B] = await Promise.all([
    getJSON(`data/matchup/${a.slug}.json`),
    getJSON(`data/player/${a.slug}.json`),
    getJSON(`data/player/${b.slug}.json`),
  ]);
  if (state.mu.a?.slug !== a.slug || state.mu.b?.slug !== b.slug) return;  // stale

  if (!mu || !A || !B) {
    main.append(el('div', 'empty', 'No exported matchup for this pair.'));
    return;
  }
  const hit = mu.vs.find(v => v.slug === b.slug);
  if (!hit) {
    main.append(el('div', 'empty', 'No exported matchup for this pair.'));
    return;
  }

  const verdict = el('div', 'verdict');
  const who = (cls, name, p) => {
    const d = el('div', `who ${cls}`);
    d.append(el('div', 'p num', pct(p)));
    d.append(el('div', 'n', name));
    return d;
  };
  verdict.append(who('a', A.name, hit.p));
  verdict.append(el('div', 'vs', 'VS'));
  verdict.append(who('b', B.name, 1 - hit.p));
  main.append(verdict);

  const grid = el('div', 'mu-grid');
  grid.append(compareCard(A, B));
  grid.append(formCard(A));
  grid.append(formCard(B));
  main.append(grid);

  main.append(el('p', 'foot',
    'Both players are evaluated at their current form, from the promoted model. Probabilities are order-invariant — the model is asked both ways round and the answers averaged, so swapping the two names gives exactly the complementary number.'));
}

// ---------------------------------------------------------------- routing

async function renderMain() {
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

function initTheme() {
  const saved = localStorage.getItem('sc-theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('#theme').onclick = () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('sc-theme', next);
  };
}

async function boot() {
  initTheme();

  $('#season').onchange = e => { state.season = e.target.value; renderSidebar(); };
  $('#tsearch').oninput = e => { state.query = e.target.value; renderSidebar(); };
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
      'Could not load data/tournaments.json. Run `make export` first, then serve the site with `make site` — opening index.html from the filesystem will not work.'));
    return;
  }
  state.index = index;
  state.players = (players || []).sort((a, b) => b.elo - a.elo);

  addEventListener('hashchange', route);
  if (!location.hash) {
    // Default to whatever is live, else the most recent completed draw.
    const t = index.find(x => x.status === 'live') || index[0];
    if (t) { location.hash = `#/t/${t.slug}`; return; }
  }
  route();
}

boot();
