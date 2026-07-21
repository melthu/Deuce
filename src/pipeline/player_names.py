"""
Canonical player names.

Wikipedia spells the same player several ways — different word order
("Kidambi Srikanth" / "Srikanth Kidambi"), optional name parts ("Anthony
Ginting" / "Anthony Sinisuka Ginting"), inconsistent hyphenation and case,
and dropped diacritics. Left alone, each spelling becomes a separate player:
its own Elo, its own form, its own head-to-head record. Parupalli Kashyap's
history was split almost down the middle, and Prannoy's four ways.

This map is deliberately explicit rather than a normalisation rule applied at
runtime. Two real players can have names that normalise to the same string,
and silently merging them is a far worse error than leaving one player split.
Every entry below was checked against three pieces of evidence: same
nationality, plausible career span, and — decisively — whether the two names
ever appear in the same draw or play each other.

Deliberately NOT merged, despite looking alike:

* "Huang Yu" / "Huang Yu-kai" — they played each other in the third round of
  Kaohsiung Masters 2023. Different people.
* "Munawar Mohammed" (India, 2018) / "Mohammed Munawar" (UAE, 2025).
* "Georges Paul" / "Julien Paul" / "Georges Julien Paul" — all Mauritius, no
  shared draw, but plausibly two distinct players. Left split pending evidence.
* "Ravi" / "Ravi Ravi" — a single-token name is too ambiguous to merge on.

Note that sharing a draw is not on its own disqualifying: a player can advance
through a bracket under two spellings when Wikipedia's own page is
inconsistent (Parupalli Kashyap, China Open 2012; Arnaud Merklé, Syed Modi
2023). Only actually meeting is conclusive.

`data_checks.py` reports any new collision that is not covered here, so a
future spelling gets reviewed rather than silently splitting a player again.
"""

# alias -> canonical. Canonical is the most frequent spelling, breaking ties
# toward diacritics, then the shorter and less capitalised form.
ALIASES = {
    "Chou Tien-Chen": "Chou Tien-chen",
    "Chen Chou-tien": "Chou Tien-chen",
    "Chou Tien Chen": "Chou Tien-chen",
    "Ng Ka Long Angus": "Ng Ka Long",
    "Angus Ng Ka Long": "Ng Ka Long",
    "Kidambi Srikanth": "Srikanth Kidambi",
    "Anthony Ginting": "Anthony Sinisuka Ginting",
    "Wang Tzu-Wei": "Wang Tzu-wei",
    "H. S. Prannoy": "Prannoy H. S.",
    "H.S. Prannoy": "Prannoy H. S.",
    "Prannoy H.S.": "Prannoy H. S.",
    "Sai Praneeth B.": "B. Sai Praneeth",
    "Kashyap Parupalli": "Parupalli Kashyap",
    "Daren Liew": "Liew Daren",
    "Lee Dong Keun": "Lee Dong-keun",
    "Thammasin Sitthikom": "Sitthikom Thammasin",
    "Manjunath Mithun": "Mithun Manjunath",
    "Dionysius Rumbaka": "Dionysius Hayom Rumbaka",
    "Nguyễn Tiến Minh": "Nguyen Tien Minh",
    "Arnaud Merkle": "Arnaud Merklé",
    "Shon Wan Ho": "Shon Wan-ho",
    "S.Sankar Muthusamy Subramanian": "Sankar Subramanian",
    "Pablo Abian": "Pablo Abián",
    "Ygor Coelho de Oliveira": "Ygor Coelho",
    "Lucas Corvee": "Lucas Corvée",
    "Aidil Sholeh Ali Sadikin": "Aidil Sholeh",
    "Huan Gao": "Gao Huan",
    "Song Xue": "Xue Song",
    "Sathish Kumar Karunakaran": "Sathish Karunakaran",
    "Sheng Xiaodong": "Xiaodong Sheng",
    "Hsueh Hsuan-Yi": "Hsueh Hsuan-yi",
    "Kartikey Kumar": "Kartikey Gulshan Kumar",
    "Meiraba Luwang Maisnam": "Meiraba Maisnam",
    "Saputra Vicky Angga": "Vicky Angga Saputra",
    "Jason Ho-Shue": "Jason Ho-shue",
    "Christian Lind": "Christian Lind Thomsen",
    "Chun Seang Tan": "Tan Chun Seang",
    "Bismo Oktora": "Bismo Raya Oktora",
    "Hsieh Yu Hsing": "Hsieh Yu-hsing",
    "Rahul Bharadwaj": "B. M. Rahul Bharadwaj",
    "Rahul Bharadwaj B.M": "B. M. Rahul Bharadwaj",
    "Zi Liang Derek Wong": "Derek Wong",
    "Derek Wong Zi Liang": "Derek Wong",
    "Kestutis Navickas": "Kęstutis Navickas",
    "Ashton Chen Yong Zhao": "Yong Zhao Ashton Chen",
    "Lakshay SHARMA": "Lakshay Sharma",
    "Rohan Kumar": "A R Rohan Kumar",
    "Chun Kar Lung": "Kar Lung Chun",
    "Kavin Thangam Kavin": "Kavin Thangam",
    "Thangam Kavin": "Kavin Thangam",
    "Ansh Gupta": "Ansh Vishal Gupta",
    "Hemanth M. Gowda": "Hemanth Gowda",
    "Ville Lang": "Ville Lång",
    "Yuhan Tan": "Tan Yuhan",
    "M Atef Haikal Taufik": "M. Atef Haikal Taufik",
    "Chan Jie Ying": "Jie Ying Chan",
    "Nguyen Hai Dang": "Nguyễn Hai Dang",
    "Kho Henrikho Wibowo": "Henrikho Kho Wibowo",
    "Ryan Ng Zin Rei": "Ryan Ng",
    "K. Ajay Kumar": "Ajay Kumar K.",
    "Hsieh Yu-Hsin": "Hsieh Yu-hsin",
}


# Pairs that look like variants but have been checked and judged distinct (or
# too ambiguous to merge). Recorded so data_checks stops asking: a warning that
# fires on every run is a warning nobody reads.
REVIEWED_DISTINCT = {
    frozenset(("Huang Yu", "Huang Yu-kai")),
    frozenset(("Munawar Mohammed", "Mohammed Munawar")),
    frozenset(("Georges Paul", "Georges Julien Paul")),
    frozenset(("Julien Paul", "Georges Julien Paul")),
    frozenset(("Ravi", "Ravi Ravi")),
}


def canonical(name):
    """Map one player name to its canonical spelling. Passes through unknowns."""
    if not isinstance(name, str):
        return name
    return ALIASES.get(name.strip(), name.strip())


def canonicalise(df):
    """Rewrite every player column of a match frame in place and return it."""
    for col in ("player_a", "player_b"):
        if col in df.columns:
            df[col] = df[col].map(canonical)
    return df
