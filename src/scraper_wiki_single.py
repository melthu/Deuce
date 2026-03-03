import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Detects walkover / retirement in score strings (fallback for text-based scores)
WALKOVER_RE = re.compile(r"\bw\.o\.\b|walkover|retd\.?|retired", re.IGNORECASE)

# Matches a per-game score cell: "21", "15", "7r" (retirement mid-game)
_SCORE_CELL_RE = re.compile(r"^\d{1,2}\s*r?$", re.IGNORECASE)

# Matches a bare seed cell: "1" .. "32"
_SEED_CELL_RE = re.compile(r"^\d{1,2}$")


def scrape_wiki_single(url: str, tournament_name: str, tier: int) -> pd.DataFrame:
    """
    Scrapes Men's Singles match results from a BWF tournament Wikipedia page.

    Uses the 'Section & Bold' strategy:
    1. Isolates the Men's Singles section by navigating mw-heading2 divs.
    2. Maps each bracket table's columns to round names via <th> header cells.
    3. Tracks true column indices (accounting for rowspan/colspan) to assign rounds.
    4. Extracts player nationality from the <a title> inside each flagicon span.
    5. Determines the winner by checking if the flagicon's parent is a <b> tag.
    """
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        timeout=15,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Step 1: Find the mw-heading2 div wrapping the Men's Singles h2 ---
    ms_heading_div = None
    for div in soup.find_all("div", class_="mw-heading"):
        h = div.find(["h2", "h3"])
        if h and re.search(r"men.?s singles", h.get_text(), re.IGNORECASE):
            ms_heading_div = div
            break

    EMPTY_COLS = ["tournament", "tier", "round", "player_a", "player_a_nat",
                  "player_b", "player_b_nat", "player_a_won", "score",
                  "player_a_seed", "player_b_seed", "is_walkover"]

    if ms_heading_div is None:
        print("ERROR: Could not find a 'Men's Singles' section header on this page.")
        return pd.DataFrame(columns=EMPTY_COLS)

    stop_pattern = re.compile(r"(women|doubles|mixed)", re.IGNORECASE)
    ms_tables = []
    for sib in ms_heading_div.find_next_siblings():
        if sib.name == "div" and "mw-heading2" in sib.get("class", []):
            if stop_pattern.search(sib.get_text()):
                break
        if sib.name == "table":
            ms_tables.append(sib)

    if not ms_tables:
        print("ERROR: Found the Men's Singles header but no bracket tables beneath it.")
        return pd.DataFrame(columns=EMPTY_COLS)

    # --- Step 2: Build column→round map from the header row ---
    def build_round_ranges(table):
        """
        Parse the first row of a bracket table to produce a list of
        (start_col, end_col, round_name) for every non-empty header cell.
        """
        rows = table.find_all("tr")
        if not rows:
            return []
        col = 0
        ranges = []
        for cell in rows[0].find_all(["th", "td"]):
            cs = int(cell.get("colspan", 1))
            text = cell.get_text().strip().lower()
            if text:
                ranges.append((col, col + cs - 1, text))
            col += cs
        return ranges

    def classify_table(table):
        """Returns 'bracket', 'group_match', or 'skip'."""
        rows = table.find_all("tr")
        if not rows:
            return "skip"
        headers = {cell.get_text().strip() for cell in rows[0].find_all(["th", "td"])}
        if "Player 1" in headers and "Player 2" in headers:
            return "group_match"
        if headers & {"Seeds", "Rank", "NOCs", "W", "L", "Pld", "Pts", "Nation"}:
            return "skip"
        return "bracket"

    def col_to_round(col_idx, ranges):
        for start, end, name in ranges:
            if start <= col_idx <= end:
                return name
        return "Unknown"

    # --- Step 3: Walk each table row-by-row, tracking true column positions ---
    def extract_player_cells(table):
        """
        Returns an ordered list of 7-tuples:
        (col_idx, player_name, nationality, is_winner, game_scores, seed, has_retirement)

        Wikipedia bracket format (modern): each player occupies one row.
          Seed cell (bare int 1-32) immediately precedes the player cell.
          Per-game score cells (bare int, e.g. "21", "15", "7r") follow the player cell.
          'r' suffix on a score cell signals mid-match retirement.

        col_occupancy tracks true visual column indices for round-name mapping.
        """
        rows = table.find_all("tr")
        col_occupancy = {}
        result = []

        for ri, row in enumerate(rows):
            # Build (cell, true_col_idx, stripped_text) for every cell in this row
            row_cells = []
            col_idx = 0
            for cell in row.find_all(["td", "th"]):
                while col_idx in col_occupancy.get(ri, set()):
                    col_idx += 1
                cs = int(cell.get("colspan", 1))
                rs = int(cell.get("rowspan", 1))
                for r in range(ri, ri + rs):
                    for c in range(col_idx, col_idx + cs):
                        col_occupancy.setdefault(r, set()).add(c)
                row_cells.append((cell, col_idx, cell.get_text().strip()))
                col_idx += cs

            # For each cell that contains a flagicon, look at its row neighbours
            for pos, (cell, true_col, _) in enumerate(row_cells):
                flagicon = cell.find("span", class_="flagicon")
                if not flagicon:
                    continue

                # Nationality from the flagicon's <a title>
                flag_link = flagicon.find("a")
                nationality = None
                if flag_link:
                    raw_nat = flag_link.get("title") or (flag_link.find("img") or {}).get("alt", "")
                    nationality = re.sub(r"national badminton team", "", raw_nat, flags=re.IGNORECASE).strip() or None

                # Player link = first <a> NOT inside the flagicon, NOT a team link
                player_link = None
                for a in cell.find_all("a"):
                    if a.find_parent("span", class_="flagicon"):
                        continue
                    if "national badminton team" in (a.get("title") or "").lower():
                        continue
                    player_link = a
                    break

                if not player_link:
                    continue
                name = player_link.get("title") or player_link.get_text().strip()
                name = re.sub(r"\s*\(.*?\)", "", name).strip()
                if not name:
                    continue

                # Winner detection — two Wikipedia formats:
                #   Modern (2018+): <b><span class="flagicon">…</span><a>Name</a></b>
                #                   → flagicon.parent is the <b> tag
                #   Classic (2010-2017): <span class="flagicon">…</span><b><a>Name</a></b>
                #                        → player_link.find_parent("b") is not None
                is_winner = (
                    flagicon.parent.name == "b"
                    or player_link.find_parent("b") is not None
                )

                # Seed: cell immediately before player cell — bare integer 1-32
                seed = 0
                if pos > 0:
                    prev_text = row_cells[pos - 1][2]
                    if _SEED_CELL_RE.match(prev_text):
                        val = int(prev_text)
                        if 1 <= val <= 32:
                            seed = val

                # Game scores: up to 3 cells immediately after player cell
                game_scores = []
                has_retirement = False
                for j in range(pos + 1, min(pos + 4, len(row_cells))):
                    sc_text = row_cells[j][2]
                    if _SCORE_CELL_RE.match(sc_text):
                        digits = re.search(r"\d+", sc_text)
                        if digits:
                            game_scores.append(int(digits.group()))
                        if sc_text.lower().endswith("r"):
                            has_retirement = True
                    else:
                        break  # stop at first non-score cell

                result.append((true_col, name, nationality, is_winner, game_scores, seed, has_retirement))

        return result

    # --- Step 4: Assemble players with round labels, then pair sequentially ---
    all_players = []  # (round_name, name, nat, is_winner, game_scores, seed, has_retirement)
    for table in ms_tables:
        table_type = classify_table(table)
        if table_type == "skip":
            continue
        if table_type == "group_match":
            for _, name, nat, is_winner, g_scores, seed, has_ret in extract_player_cells(table):
                all_players.append(("group stage", name, nat, is_winner, g_scores, seed, has_ret))
        else:
            round_ranges = build_round_ranges(table)
            for col_idx, name, nat, is_winner, g_scores, seed, has_ret in extract_player_cells(table):
                round_name = col_to_round(col_idx, round_ranges)
                all_players.append((round_name, name, nat, is_winner, g_scores, seed, has_ret))

    matches = []
    for i in range(0, len(all_players) - 1, 2):
        round_a, player_a, nat_a, a_wins, scores_a, seed_a, ret_a = all_players[i]
        round_b, player_b, nat_b, _,      scores_b, seed_b, ret_b = all_players[i + 1]

        round_name = round_a if round_a != "Unknown" else round_b

        if player_a == player_b:
            continue

        # Reconstruct "W-L" score string from per-game integer arrays
        w_scores = scores_a if a_wins else scores_b
        l_scores = scores_b if a_wins else scores_a
        if w_scores and l_scores and len(w_scores) == len(l_scores):
            match_score = ", ".join(f"{w}-{l}" for w, l in zip(w_scores, l_scores))
        else:
            match_score = ""

        is_walkover = int(ret_a or ret_b or bool(WALKOVER_RE.search(match_score or "")))

        matches.append(
            {
                "tournament": tournament_name,
                "tier": tier,
                "round": round_name,
                "player_a": player_a,
                "player_a_nat": nat_a,
                "player_b": player_b,
                "player_b_nat": nat_b,
                "player_a_won": 1 if a_wins else 0,
                "score": match_score,
                "player_a_seed": seed_a,
                "player_b_seed": seed_b,
                "is_walkover": is_walkover,
            }
        )

    return pd.DataFrame(matches)


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/2026_Malaysia_Open_(badminton)"
    df = scrape_wiki_single(url=test_url, tournament_name="Malaysia Open 2026", tier=1000)

    if df.empty:
        print("Extraction failed or returned empty DataFrame.")
    else:
        print(f"Success! Extracted {len(df)} Men's Singles matches.\n")
        print(df.to_string(index=True))
