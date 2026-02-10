import penaltyblog as pb
import pandas as pd
import numpy as np
import math
from collections import Counter

# -------- settings you change each game ----------
COMP = "ESP La Liga"
HOME = "Villarreal"
AWAY = "Espanol"
TOTAL_LINE = 2.5

SEASONS = ["2024-2025", "2025-2026"]
XI = 0.02

# Backtest windows are taken from CURRENT SEASON only (most recent season in SEASONS)
N_TEST = 70          # last N matches of current season used for reporting
MIN_CAL = 60         # try to keep at least this many calibration matches
MIN_BUCKET_N = 20    # minimum samples to trust a bucket (we will merge buckets if needed)

# Narrower buckets (your idea is fine; we’ll merge if buckets are too small)
BINS = [0.00, 0.35, 0.45, 0.52, 0.58, 0.64, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]

# Coverage rules (promoted / cold-start teams)
MIN_CUR = 10
MIN_CUR_PROMOTED = 14
REQUIRE_BOTH_SEASONS = False
# -----------------------------------------------

def pct(x):
    return f"{100*float(x):.1f}%"

def fair_odds(p):
    p = float(p)
    return float("inf") if p <= 0 else 1 / p

def load_season(season):
    fb = pb.scrapers.FootballData(COMP, season)
    df = fb.get_fixtures()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = season
    df = df[["season", "date", "team_home", "team_away", "goals_home", "goals_away"]].copy()

    # keep only played matches (sometimes scrapers include future fixtures)
    df = df.dropna(subset=["goals_home", "goals_away", "team_home", "team_away", "date"])
    df["team_home"] = df["team_home"].astype(str).str.strip()
    df["team_away"] = df["team_away"].astype(str).str.strip()
    df["goals_home"] = df["goals_home"].astype(int)
    df["goals_away"] = df["goals_away"].astype(int)
    return df

def teams_in(df_):
    return set(df_["team_home"]).union(set(df_["team_away"]))

def suggest_names(name, teams_list, k=10):
    key = name.lower()
    hits = [t for t in teams_list if key in t.lower()]
    if hits:
        return hits[:k]
    key2 = key[:4]
    hits2 = [t for t in teams_list if key2 in t.lower()]
    return hits2[:k]

def team_season_count(df_all, team, season):
    df_s = df_all[df_all["season"] == season]
    return int(((df_s["team_home"] == team) | (df_s["team_away"] == team)).sum())

def get_p_over25_from_goal_matrix(m):
    H, A = m.shape
    hg = np.arange(H)[:, None]
    ag = np.arange(A)[None, :]
    return float(m[(hg + ag) >= 3].sum())

def poisson_over25_from_total_lambda(L):
    # P(total goals >= 3) where total ~ Poisson(L)
    p_le_2 = math.exp(-L) * (1.0 + L + (L * L) / 2.0)
    return 1.0 - p_le_2

def find_lambda_scale_to_match_over25(target_p_over, lambda_home, lambda_away):
    target = float(target_p_over)
    base_L = float(lambda_home + lambda_away)
    if base_L <= 1e-9:
        return 1.0

    target = max(1e-6, min(1 - 1e-6, target))

    lo, hi = 0.10, 3.00
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p_mid = poisson_over25_from_total_lambda(base_L * mid)
        if p_mid < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def bucket_index(p, bins):
    # returns i such that p in [bins[i], bins[i+1]] (last bin inclusive)
    p = float(p)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            if lo <= p <= hi:
                return i
        else:
            if lo <= p < hi:
                return i
    return len(bins) - 2

def mask_for_range(preds, lo, hi, is_last):
    if is_last:
        return (preds >= lo) & (preds <= hi)
    return (preds >= lo) & (preds < hi)

def calibrate_over25_adaptive(raw_p, cal_preds, cal_actuals, bins, min_n, prior_strength=8.0):
    """
    Narrow bins, but adaptively MERGE outward until we have >= min_n samples.
    We return p_cal + debug info about which range we ended up using.

    prior_strength is a mild shrinkage so a tiny bucket doesn't whipsaw scaling:
    ratio = (act_sum + prior_strength*raw_p) / (pred_sum + prior_strength*raw_p)
    """
    raw_p = float(raw_p)
    i = bucket_index(raw_p, bins)
    left = i
    right = i + 1  # right is index into bins

    # expand outward until enough samples or we cover all bins
    while True:
        lo = bins[left]
        hi = bins[right]
        is_last = (right == len(bins) - 1)
        mask = mask_for_range(cal_preds, lo, hi, is_last)
        n = int(mask.sum())

        if n >= min_n or (left == 0 and right == len(bins) - 1):
            pred_sum = float(cal_preds[mask].sum())
            act_sum = float(cal_actuals[mask].sum())

            # shrinkage on ratio to avoid crazy jumps from small n
            denom = pred_sum + prior_strength * raw_p
            numer = act_sum + prior_strength * raw_p
            ratio = (numer / denom) if denom > 1e-9 else 1.0

            p_cal = raw_p * ratio
            p_cal = max(1e-6, min(1 - 1e-6, float(p_cal)))

            return p_cal, {
                "range": (lo, hi),
                "n": n,
                "pred_mean": float(cal_preds[mask].mean()) if n else None,
                "act_mean": float(cal_actuals[mask].mean()) if n else None,
                "ratio": float(ratio),
            }

        # expand strategy: alternate expanding left/right depending on which side exists
        can_left = left > 0
        can_right = right < len(bins) - 1
        if not (can_left or can_right):
            break

        # expand both if possible to keep it centered
        if can_left:
            left -= 1
        if can_right:
            right += 1

    # fallback
    return raw_p, {"range": None, "n": 0, "pred_mean": None, "act_mean": None, "ratio": 1.0}

# ----------------- LOAD DATA -----------------
dfs = [load_season(s) for s in SEASONS]
df_all = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)

teams = sorted(teams_in(df_all))
print("\n=== TEAM NAMES IN DATA (copy-paste these) ===")
for i, t in enumerate(teams, 1):
    print(f"{i:>2}. {t}")
print("=== END TEAM NAMES ===\n")

if HOME not in teams or AWAY not in teams:
    print("❌ Team name mismatch for current data.")
    print("You typed:", HOME, "vs", AWAY)
    print("Suggestions HOME:", suggest_names(HOME, teams))
    print("Suggestions AWAY:", suggest_names(AWAY, teams))
    raise SystemExit(1)

print(f"Training matches (ALL seasons used for match model): {len(df_all)} | date range: {df_all['date'].min().date()} to {df_all['date'].max().date()}")

# --- Coverage gate (promoted / cold-start protection) ---
S1, S2 = SEASONS[0], SEASONS[-1]
home_s1 = team_season_count(df_all, HOME, S1)
home_s2 = team_season_count(df_all, HOME, S2)
away_s1 = team_season_count(df_all, AWAY, S1)
away_s2 = team_season_count(df_all, AWAY, S2)

print("\nTeam season coverage check:")
print(f"{HOME}: {S1} matches={home_s1}, {S2} matches={home_s2}")
print(f"{AWAY}: {S1} matches={away_s1}, {S2} matches={away_s2}")

if REQUIRE_BOTH_SEASONS and (home_s1 == 0 or away_s1 == 0):
    print("\n❌ Strict mode: one team not present in BOTH seasons. Skipping prediction.")
    raise SystemExit(1)

if home_s2 < MIN_CUR or away_s2 < MIN_CUR:
    print(f"\n❌ Too few current-season matches (need >= {MIN_CUR}). Skipping prediction.")
    raise SystemExit(1)

if home_s1 == 0 and home_s2 < MIN_CUR_PROMOTED:
    print(f"\n❌ {HOME} looks promoted/new in {S2}. Need >= {MIN_CUR_PROMOTED} current-season matches. Skipping.")
    raise SystemExit(1)

if away_s1 == 0 and away_s2 < MIN_CUR_PROMOTED:
    print(f"\n❌ {AWAY} looks promoted/new in {S2}. Need >= {MIN_CUR_PROMOTED} current-season matches. Skipping.")
    raise SystemExit(1)

print("✅ Coverage OK. Proceeding...\n")

# ----------------- MAIN MATCH MODEL (ALL SEASONS) -----------------
weights_all = pb.models.dixon_coles_weights(df_all["date"], XI)

print("Weight check:")
print("min/max:", float(weights_all.min()), float(weights_all.max()))
print("avg weight current season:", float(weights_all[df_all["season"] == S2].mean()))
print("avg weight prior season  :", float(weights_all[df_all["season"] == S1].mean()))

model = pb.models.DixonColesGoalModel(
    df_all["goals_home"], df_all["goals_away"],
    df_all["team_home"], df_all["team_away"],
    weights_all
)
model.fit()

probs = model.predict(HOME, AWAY)
m = probs.goal_matrix

print("\nMatrix sanity:")
print("goal_matrix shape:", m.shape)
print("sum(goal_matrix):", float(m.sum()))
print("tail mass missing:", float(1 - m.sum()))

H, A = m.shape
hg = np.arange(H)[:, None]
ag = np.arange(A)[None, :]

exp_home_goals = float((m * hg).sum())
exp_away_goals = float((m * ag).sum())
L_raw = exp_home_goals + exp_away_goals

print("\nExpected goals (RAW):")
print("Home xG-like:", round(exp_home_goals, 2))
print("Away xG-like:", round(exp_away_goals, 2))
print("Total:", round(L_raw, 2))

p_home_win = float(m[hg > ag].sum())
p_draw     = float(m[hg == ag].sum())
p_away_win = float(m[hg < ag].sum())

p_over_raw = get_p_over25_from_goal_matrix(m)
p_under_raw = 1.0 - p_over_raw

# ----------------- CALIBRATION USING CURRENT SEASON ONLY -----------------
df_cur = df_all[df_all["season"] == S2].sort_values("date").reset_index(drop=True)
cur_teams = teams_in(df_cur)

# Warm-up end = first point where every team has appeared at least once
seen = set()
warmup_end = None
for i, r in df_cur.iterrows():
    seen.add(r["team_home"])
    seen.add(r["team_away"])
    if seen == cur_teams:
        warmup_end = i + 1
        break

if warmup_end is None:
    print("\n❌ Could not find a warm-up end where all teams appear (unexpected).")
    raise SystemExit(1)

# Choose TEST as last N_TEST matches, CAL as everything between warmup and test (at least MIN_CAL if possible)
n_cur = len(df_cur)
test_size = min(N_TEST, max(0, n_cur - warmup_end - MIN_CAL))
test_start = n_cur - test_size
cal_start = warmup_end
cal_end = test_start

cal_df = df_cur.iloc[cal_start:cal_end].copy()
test_df = df_cur.iloc[test_start:].copy() if test_size > 0 else None
train_cut_date = cal_df["date"].min()

# Train for calibration = all matches strictly before CAL starts (includes prior season + warmup current season)
train_bt = df_all[df_all["date"] < train_cut_date].copy()

train_teams = teams_in(train_bt)
missing_for_cur = sorted(list(cur_teams - train_teams))

print("\nBacktest split (CURRENT season only):")
print("Current season matches:", n_cur)
print("Warm-up (in TRAIN):   ", warmup_end)
print("CAL window:           ", len(cal_df))
print("TEST window:          ", len(test_df) if test_df is not None else 0)
print("Teams missing from TRAIN before CAL starts:", ("None ✅" if not missing_for_cur else missing_for_cur))

if missing_for_cur:
    print("\n❌ TRAIN does not contain all current-season teams before CAL starts.")
    print("This would cause 'unseen team' errors. Increase warm-up or check data.")
    raise SystemExit(1)

weights_train = pb.models.dixon_coles_weights(train_bt["date"], XI)
model_bt = pb.models.DixonColesGoalModel(
    train_bt["goals_home"], train_bt["goals_away"],
    train_bt["team_home"], train_bt["team_away"],
    weights_train
)
model_bt.fit()

def predict_over25_safe(model_, home, away):
    # if it errors, return None
    try:
        pr = model_.predict(home, away)
        return get_p_over25_from_goal_matrix(pr.goal_matrix)
    except ValueError:
        return None

# Build CAL arrays (should have zero skips now)
cal_preds = []
cal_actuals = []
cal_skips = 0
skip_pairs = Counter()

for _, r in cal_df.iterrows():
    p = predict_over25_safe(model_bt, r["team_home"], r["team_away"])
    if p is None:
        cal_skips += 1
        skip_pairs[(r["team_home"], r["team_away"])] += 1
        continue
    cal_preds.append(float(p))
    cal_actuals.append(1.0 if (r["goals_home"] + r["goals_away"]) >= 3 else 0.0)

cal_preds = np.array(cal_preds, dtype=float)
cal_actuals = np.array(cal_actuals, dtype=float)

print("\nCalibration buckets (Over2.5) [CAL window]:")
print(f"CAL matches: {len(cal_df)} | used: {len(cal_preds)} | skipped: {cal_skips}")
if cal_skips:
    print("Top skipped matchups (home, away) -> count:", skip_pairs.most_common(5))

for lo, hi in zip(BINS[:-1], BINS[1:]):
    is_last = (hi == BINS[-1])
    mask = mask_for_range(cal_preds, lo, hi, is_last)
    if mask.sum() >= 8:
        print(f"{lo:.2f}-{hi:.2f}  n={int(mask.sum()):3d}  pred={cal_preds[mask].mean():.3f}  actual={cal_actuals[mask].mean():.3f}")

print("\nCAL window summary:")
if len(cal_preds):
    print("avg predicted Over2.5:", round(float(cal_preds.mean()), 3))
    print("avg actual Over2.5   :", round(float(cal_actuals.mean()), 3))
    print("Brier (CAL window)   :", round(float(np.mean((cal_preds - cal_actuals) ** 2)), 3))
else:
    print("n/a (no CAL predictions)")

# Evaluate on TEST (optional)
if test_df is not None and len(test_df) > 0:
    test_preds = []
    test_actuals = []
    test_skips = 0

    for _, r in test_df.iterrows():
        p = predict_over25_safe(model_bt, r["team_home"], r["team_away"])
        if p is None:
            test_skips += 1
            continue
        test_preds.append(float(p))
        test_actuals.append(1.0 if (r["goals_home"] + r["goals_away"]) >= 3 else 0.0)

    test_preds = np.array(test_preds, dtype=float)
    test_actuals = np.array(test_actuals, dtype=float)

    print("\nTEST window Over2.5:")
    print(f"TEST matches: {len(test_df)} | used: {len(test_preds)} | skipped: {test_skips}")
    if len(test_preds):
        print("avg predicted Over2.5:", round(float(test_preds.mean()), 3))
        print("actual Over2.5 rate  :", round(float(test_actuals.mean()), 3))
        print("Brier score          :", round(float(np.mean((test_preds - test_actuals) ** 2)), 3))

print("\nTraining data averages (ALL data used for the match model):")
print("Avg home goals:", float(df_all["goals_home"].mean()))
print("Avg away goals:", float(df_all["goals_away"].mean()))
print("Avg total goals:", float((df_all["goals_home"] + df_all["goals_away"]).mean()))
print("Actual Over2.5 rate:", float(((df_all["goals_home"] + df_all["goals_away"]) >= 3).mean()))

# ----------------- APPLY CALIBRATION (Over2.5 only) -----------------
if len(cal_preds) < 30:
    print("\n⚠️ Too few CAL samples; falling back to RAW Over2.5.")
    p_over_cal = p_over_raw
    dbg = {"range": None, "n": 0, "pred_mean": None, "act_mean": None, "ratio": 1.0}
else:
    p_over_cal, dbg = calibrate_over25_adaptive(
        p_over_raw, cal_preds, cal_actuals, BINS, MIN_BUCKET_N, prior_strength=8.0
    )

p_under_cal = 1.0 - p_over_cal

print("\nTotals (RAW vs CAL):")
print("RAW Over 2.5:", pct(p_over_raw), "fair:", round(fair_odds(p_over_raw), 2))
print("CAL Over 2.5:", pct(p_over_cal), "fair:", round(fair_odds(p_over_cal), 2))
print("RAW Under 2.5:", pct(p_under_raw), "fair:", round(fair_odds(p_under_raw), 2))
print("CAL Under 2.5:", pct(p_under_cal), "fair:", round(fair_odds(p_under_cal), 2))

if dbg["range"] is not None:
    lo, hi = dbg["range"]
    print("\nCalibration used (after merging if needed):")
    print(f"range {lo:.2f}-{hi:.2f} | n={dbg['n']} | pred_mean={dbg['pred_mean']:.3f} | act_mean={dbg['act_mean']:.3f} | ratio={dbg['ratio']:.3f}")
else:
    print("\nCalibration used: fallback (no bucket info).")

# ----------------- TIMING CONSISTENT WITH CALIBRATED TOTAL INTENSITY -----------------
scale_s = find_lambda_scale_to_match_over25(p_over_cal, exp_home_goals, exp_away_goals)
lambda_home_adj = exp_home_goals * scale_s
lambda_away_adj = exp_away_goals * scale_s
L_adj = lambda_home_adj + lambda_away_adj

print("\nLambda scaling to match CAL Over2.5:")
print("scale s:", round(scale_s, 3))
print("Adj home lambda:", round(lambda_home_adj, 2))
print("Adj away lambda:", round(lambda_away_adj, 2))
print("Adj total lambda:", round(L_adj, 2))
print("Check Poisson Over2.5 from adj total:", pct(poisson_over25_from_total_lambda(L_adj)))

p_no_goals = math.exp(-L_adj)
p_home_first = (lambda_home_adj / L_adj) * (1 - p_no_goals) if L_adj > 0 else 0.0
p_away_first = (lambda_away_adj / L_adj) * (1 - p_no_goals) if L_adj > 0 else 0.0

print("\nFirst goal (Poisson timing, using CALIBRATED total intensity):")
print("Home scores first:", pct(p_home_first))
print("Away scores first:", pct(p_away_first))
print("No goals (0-0):    ", pct(p_no_goals))

def p_first_goal_by_minute(t_min):
    return 1 - math.exp(-L_adj * (t_min / 90.0))

for t in [10, 20, 30, 45, 60, 75]:
    print(f"First goal by {t:>2}':", pct(p_first_goal_by_minute(t)))

# ----------------- MATCH SUMMARY (RAW scoreline model) -----------------
print(f"\nMatch (RAW model): {HOME} vs {AWAY}")
print("Home win:", pct(p_home_win), "fair:", round(fair_odds(p_home_win), 2))
print("Draw    :", pct(p_draw),     "fair:", round(fair_odds(p_draw), 2))
print("Away win:", pct(p_away_win), "fair:", round(fair_odds(p_away_win), 2))

print("\nCommon AH (matrix-based, RAW model):")
print("Home -0.5 (must win):", pct(p_home_win), "fair:", round(fair_odds(p_home_win), 2))
print("Away +0.5 (win or draw):", pct(p_away_win + p_draw), "fair:", round(fair_odds(p_away_win + p_draw), 2))

print("\nLibrary helper (may use different convention):")
print("asian_handicap('home', -0.5) =", probs.asian_handicap('home', -0.5))
print("asian_handicap('away', +0.5) =", probs.asian_handicap('away', +0.5))