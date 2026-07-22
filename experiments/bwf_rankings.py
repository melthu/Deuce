"""Real published BWF world rankings, from whatever is actually reachable.

BWF publishes weekly men's-singles rankings as one XLSX per week. Every BWF
host that serves them - `bwfbadminton.com`, `corporate.bwfbadminton.com`,
`system.bwfbadminton.com`, `extranet.bwfbadminton.com` - sits behind a
Cloudflare bot challenge and returns 403 to any script, headless browser
included. Working around a bot-protection control is not something this repo
does, so the reachable sources are:

  1. **the Wayback Machine** - it archived 22 weeks of 2017 from the old
     `system.bwfbadminton.com` paths, and nothing from the newer hosts;
  2. **files you download yourself** - open the BWF historical-rankings page in
     a real browser, which passes the challenge, and drop the XLSX files into
     `data/raw/rankings/`. They are picked up automatically.

That is 22 weeks against a corpus of 10,196 matches spanning 2010-2026, so
this is not enough to build a ranking feature from. It is enough to validate
the results-derived proxy in `candidate_features.py` against the real list,
which is what `validate_rank_proxy.py` uses it for - and the parser here means
a hand-downloaded archive needs no further work to become a real feature.

XLSX is parsed with zipfile and regex rather than openpyxl: the layout is four
title rows, a header row (Ranking / BWF ID / Last name / First name / Gender /
Country / Points / Tour), then one row per player, and pulling that out does
not justify a dependency the rest of the project does not have.

    python3 experiments/bwf_rankings.py
"""
import os
import re
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

CDX = ("http://web.archive.org/cdx/search/cdx?url=system.bwfbadminton.com"
       "&matchType=domain&output=text&fl=timestamp,original&collapse=urlkey")
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"
CACHE_DIR = "data/interim/bwf_xlsx"
LOCAL_DIR = "data/raw/rankings"
UA = {"User-Agent": "Deuce/1.0 (BWF match model; research)"}

MENS_SINGLES_SHEET = 0   # sheet order is MS, WS, MD, WD, XD


def _get(url: str, timeout: int = 60) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


# ---------------------------------------------------------------- xlsx

def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    raw = z.read("xl/sharedStrings.xml").decode("utf8", "ignore")
    return [re.sub(r"<[^>]+>", "", m)
            for m in re.findall(r"<si>(.*?)</si>", raw, re.S)]


def _sheet_rows(z: zipfile.ZipFile, index: int):
    """Yield each row as {column letter: value}."""
    names = sorted(n for n in z.namelist()
                   if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    order = sorted(names, key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)))
    strings = _shared_strings(z)
    sheet = z.read(order[index]).decode("utf8", "ignore")
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for ref, attrs, val in re.findall(
                r'<c r="([A-Z]+)\d+"([^>]*)>(?:<v>(.*?)</v>)?</c>', row, re.S):
            if val is None or val == "":
                continue
            cells[ref] = strings[int(val)] if 't="s"' in attrs else val
        if cells:
            yield cells


def parse_xlsx(data: bytes, sheet: int = MENS_SINGLES_SHEET) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """(published date, [rank, player, country, points, n_tournaments])."""
    import io
    z = zipfile.ZipFile(io.BytesIO(data))
    as_of, rows, seen_header = None, [], False
    for cells in _sheet_rows(z, sheet):
        a = cells.get("A", "")
        if as_of is None and "DATE:" in a:
            m = re.search(r"DATE:\s*\w+,\s*(.+)$", a.strip())
            if m:
                try:
                    as_of = pd.to_datetime(re.sub(r"\s+", " ", m.group(1)).strip())
                except Exception:
                    as_of = None
        if a.strip().lower() == "ranking":
            seen_header = True
            continue
        if not seen_header or not a.strip().isdigit():
            continue
        last, first = cells.get("C", "").strip(), cells.get("D", "").strip()
        rows.append({
            "rank":    int(a),
            "player":  f"{first} {last}".strip(),
            "country": cells.get("F", "").strip(),
            "points":  float(cells.get("G", "nan") or "nan"),
            "n_tournaments": int(float(cells.get("H", 0) or 0)),
        })
    return as_of, pd.DataFrame(rows)


# -------------------------------------------------------------- sources

def wayback_urls() -> list[tuple[str, str]]:
    """(timestamp, original url) for every archived historical-ranking XLSX."""
    raw = _get(CDX, timeout=120)
    if raw is None:
        return []
    out = []
    for line in raw.decode("utf8", "ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and "historical-ranking" in parts[1].lower() \
                and parts[1].lower().endswith(".xlsx"):
            out.append((parts[0], parts[1]))
    return sorted(out, key=lambda t: t[1])


def fetch_wayback(limit: int | None = None) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    snapshots = []
    for ts, url in (wayback_urls()[:limit] if limit else wayback_urls()):
        name = re.sub(r"[^\w.-]", "_", url.rsplit("/", 1)[-1])
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            data = open(path, "rb").read()
        else:
            data = _get(WAYBACK.format(ts=ts, url=url), timeout=90)
            if not data or not data.startswith(b"PK"):
                continue
            open(path, "wb").write(data)
        try:
            as_of, df = parse_xlsx(data)
        except Exception:
            continue
        if as_of is not None and len(df):
            snapshots.append((as_of, df))
    return snapshots


def load_local() -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Any XLSX you downloaded by hand into data/raw/rankings/."""
    if not os.path.isdir(LOCAL_DIR):
        return []
    out = []
    for name in sorted(os.listdir(LOCAL_DIR)):
        if not name.lower().endswith((".xlsx", ".xls")):
            continue
        data = open(os.path.join(LOCAL_DIR, name), "rb").read()
        try:
            as_of, df = parse_xlsx(data)
        except Exception:
            continue
        if as_of is not None and len(df):
            out.append((as_of, df))
    return out


def load_all() -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Every real weekly snapshot available, deduplicated by publication date."""
    by_date = {}
    for as_of, df in fetch_wayback() + load_local():
        by_date[as_of.normalize()] = df
    return sorted(by_date.items())


if __name__ == "__main__":
    snaps = load_all()
    if not snaps:
        print("no real ranking snapshots available")
        raise SystemExit
    print(f"{len(snaps)} real weekly snapshots  "
          f"{snaps[0][0].date()} .. {snaps[-1][0].date()}\n")
    as_of, df = snaps[0]
    print(f"{as_of.date()} - {len(df)} ranked players")
    print(df.head(8).to_string(index=False))
    if os.path.isdir(LOCAL_DIR):
        print(f"\n(also reading hand-downloaded files from {LOCAL_DIR}/)")
    else:
        print(f"\nDrop hand-downloaded weekly XLSX into {LOCAL_DIR}/ to extend this.")
