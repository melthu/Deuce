# experiments/

Exploratory work on making the prediction better. Nothing here runs in CI or is
imported by `src/`; it exists to decide what *should* move into `src/`.

## Why a separate harness

`promote.py` benchmarks candidates on a single temporal holdout — the latest
season, roughly a thousand matches — and the three candidates historically sit
within 0.002 AUC of each other. A holdout that thin cannot tell a real
improvement from resampling noise, so before this directory there was no honest
way to answer "does this feature help?".

`harness.py` scores everything identically:

* **rolling temporal evaluation** — train on all years strictly before Y,
  evaluate on Y, for Y in 2022–2026, and report the mean of the five;
* **train-side mirroring only** — each training match appears twice with the
  players swapped, exactly as `data_loader.py` does, so slot order carries no
  signal;
* **order-invariant scoring** — the validation slice is predicted as
  `(P(orig) + 1 − P(swapped)) / 2`, exactly as `predict_match` serves it.
  Scoring one orientation would flatter any model that learned a slot bias the
  real serving path removes.

The mirror spec is derived from column names (`player_a_*` ↔ `player_b_*`), so
a new per-player feature needs no registration. Pair-level features are the
only ones that transform under a swap and must be added to `PAIR_LEVEL`.

## The trap this directory was built to avoid

The first ablation added all 36 candidate columns to the shipped LightGBM and
reported that **every** feature group made the model worse. That result was an
artifact: the shipped config was Optuna-tuned for 30 features, including
`feature_fraction=0.42`, and handing it 66 features changes what that number
means. A feature set and its hyperparameters are one object and have to be
compared as one.

`run_search.py` is the fix — each feature set gets its own random search on an
identical budget, and selection is split from reporting:

| | years |
|---|---|
| select the config | 2022–2024 |
| report its score | 2025–2026 |

`run_capacity.py` is kept for the same reason: it shows the shipped *default*
params (1000 trees × 63 leaves) at 0.7157 AUC against 0.7326 for the tuned
ones, i.e. most of the "capacity" story was really a params story.

## Reference points

Measured on the rolling harness, mean over 2022–2026:

| | AUC | logloss |
|---|---|---|
| logistic regression on `elo_diff` alone | 0.7032 | 0.6261 |
| logistic regression, all 34 features | 0.7162 | 0.6184 |
| CatBoost, tuned | 0.7180 | 0.6302 |
| XGBoost, tuned | 0.7288 | 0.6108 |
| **LightGBM, tuned — the shipped baseline** | **0.7326** | **0.6060** |

Elo alone gets most of the way. That is the reason for `run_elo.py`.

## Findings

### Elo's own constants were never fit — tuning them is worth +0.027 AUC

`run_elo.py` scores the raw Elo expectancy with no model on top, so the number
is a statement about the rating rather than about the trees. It searches
parameters the shipped rating does not have at all:

* **margin of victory** — a 21-5 win moves the rating further than 22-20. The
  shipped rating parses scorelines for other features and then throws them away
  here.
* **provisional K** — a larger K for a player's first ~23 matches, so a
  newcomer converges off 1500 in a few events instead of over a season.
* **inactivity decay** — regression toward the mean during a layoff.
* **tier→K curve** — currently a hand-written lookup.

Selected on 2019–2023, reported on 2024–2026 which the fit never saw:

| | AUC | logloss |
|---|---|---|
| shipped Elo | 0.6868 | 0.6342 |
| tuned Elo | **0.7135** | **0.6197** |

The tuned parameters are worth reading for what they say about the sport:
`tier_alpha ≈ 0.06` means the tier of an event barely matters to how much a
result should move a rating — the shipped `K_BY_TIER` spread of 20→50 is doing
almost nothing. Nearly all of the gain comes from margin of victory and from
provisional K.

`tuned_elo.py` emits the tuned rating as columns so the tree model can be asked
whether a better input makes it a better model.

### The ranking feature, and what it actually is

Real BWF ranking history is not fetchable here: `bwfbadminton.com` and
`corporate.bwfbadminton.com` both return 403 behind Cloudflare, the community
mirror [raywan/bwf-data](https://github.com/raywan/bwf-data) covers only 2015
w1–2016 w7, and the Kaggle archive needs an authenticated download. **No
ranking column in this repo is downloaded data.**

What `candidate_features.py` builds instead is a proxy computed from the
results already in the corpus: each player is awarded the points their
finishing round earns at that tournament's tier, on BWF's own table, and the
best 10 events inside a rolling 52 weeks are summed — BWF's own accounting
rule. From that, a live ordinal rank.

`validate_rank_proxy.py` checks the proxy against the 58 real weekly snapshots
that *are* fetchable:

| | |
|---|---|
| Spearman ρ vs published BWF ranking | **0.854** (min 0.812, max 0.897) |
| real top 10 also in proxy top 10 | 6.9 / 10 |
| median absolute rank error | 6.6 places |

So it is the same quantity, imperfectly measured. Good enough to use as a
feature; not good enough to display to a user as "world ranking".

**To use the real thing instead**, download the year-by-year XLS from
[BWF Corporate → Historical Rankings](https://corporate.bwfbadminton.com/players/historical-rankings/)
(the page needs a browser; the download is one file per year) and drop them in
`data/raw/rankings/`. Only the loader in `candidate_features.py` would change.

## Files

| file | what it does |
|---|---|
| `harness.py` | rolling evaluation, mirroring, order-invariant scoring |
| `models.py` | model factories, shared param defaults |
| `candidate_features.py` | all candidate features in one chronological pass |
| `tuned_elo.py` | the tuned rating as feature columns |
| `run_baseline.py` | the shipped feature set, three model families |
| `run_capacity.py` | capacity/regularisation sweep, player-ID ablation |
| `run_features.py` | the naive (unfair) group ablation — kept as the warning |
| `run_search.py` | per-feature-set hyperparameter search, split select/report |
| `run_elo.py` | tunes Elo's own constants |
| `run_combined.py` | does the tuned rating help the trees, and does it compound with the ranking proxy |
| `validate_rank_proxy.py` | proxy vs the real published ranking |

Results land in `experiments/results/*.json`.
