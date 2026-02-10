import penaltyblog as pb
import pandas as pd
import numpy as np
import math

# -------- settings you change each game ----------
COMP = "DEU Bundesliga 1"
HOME = "Union Berlin"
AWAY = "Ein Frankfurt"
TOTAL_LINE = 2.5

SEASONS = ["2024-2025", "2025-2026"]
XI = 0.02

# Backtest / calibration sizes (recent games)
N_TEST = 120   # final held-out window (reporting only)
N_CAL  = 200   # calibration window (used to build mapping)
MIN_WIN = 80   # minimum for either window; adjust if league has fewer matches

# Narrower calibration buckets (feel free to tweak)
# Key idea: narrower near the top so you don't get huge jumps.
BINS = [0.00, 0.35, 0.45, 0.52, 0.58, 0.64, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
MIN_BUCKET_N = 20  # require enough samples per bucket to trust it

# Coverage rules (for promoted / cold-start teams)
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
    return df[["season", "date", "team_home", "team_away", "goals_home", "goals_away"]].copy()

def poisson_over25_from_total_lambda(L):
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

def team_season_count(df_all, team, season):
    df_s = df_all[df_all["season"] == season]
    return int(((df_s["team_home"] == team) | (df_s["team_away"] == team)).sum())

def get_p_over25_from_goal_matrix(m):
    H, A = m.shape
    hg = np.arange(H)[:, None]
    ag = np.arange(A)[None, :]
    return float(m[(hg + ag) >= 3].sum())

def suggest_names(name, teams_list, k=10):
    key = name.lower()
    hits = [t for t in teams_list if key in t.lower()]
    if hits:
        return hits[:k]
    key2 = key[:4]
    hits2 = [t for t in teams_list if key2 in t.lower()]
    return hits2[:k]

def calibrate_over25_bucket(raw_p, preds, actuals, bins, min_n):
    """
    Piecewise calibration using (bins) on CAL window:
    p_cal = p_raw * (actual_mean / pred_mean) inside the bucket,
    but only if bucket has at least min_n samples.
    Otherwise fall back to raw_p.
    """
    raw_p = float(raw_p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # include hi only on last bucket
        in_bucket = (raw_p >= lo and (raw_p < hi or (hi == bins[-1] and raw_p <= hi)))
        if in_bucket:
            mask = (preds >= lo) & (preds < hi if hi != bins[-1] else preds <= hi)
            n = int(mask.sum())
            if n >= min_n:
                pred_mean = float(preds[mask].mean())
                act_mean = float(actuals[mask].mean())
                if pred_mean > 1e-9:
                    return raw_p * (act_mean / pred_mean)
            return raw_p
    return raw_p

# 1) Load and combine seasons
dfs = [load_season(s) for s in SEASONS]
df = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)

teams = sorted(set(df["team_home"]).union(set(df["team_away"])))
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

print("Training matches:", len(df), "| date range:", df["date"].min().date(), "to", df["date"].max().date())

# --- Coverage gate ---
S1, S2 = SEASONS[0], SEASONS[1]
home_s1 = team_season_count(df, HOME, S1)
home_s2 = team_season_count(df, HOME, S2)
away_s1 = team_season_count(df, AWAY, S1)
away_s2 = team_season_count(df, AWAY, S2)

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

# 2) Fit main model (for THIS match)
weights = pb.models.dixon_coles_weights(df["date"], XI)
w = weights

print("\nWeight check:")
print("min/max:", float(w.min()), float(w.max()))
print("avg weight 2025-26:", float(w[df["date"].dt.year >= 2025].mean()))
print("avg weight 2024-25:", float(w[df["date"].dt.year == 2024].mean()))

model = pb.models.DixonColesGoalModel(
    df["goals_home"], df["goals_away"],
    df["team_home"], df["team_away"],
    weights
)
model.fit()

# 3) Predict RAW match
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

p_over_raw  = get_p_over25_from_goal_matrix(m)
p_under_raw = 1.0 - p_over_raw

# 4) 3-way split for calibration mapping (TRAIN -> CAL)
df_bt = df.sort_values("date").reset_index(drop=True)
n = len(df_bt)

N_TEST_eff = min(N_TEST, max(MIN_WIN, n // 6))
N_CAL_eff  = min(N_CAL,  max(MIN_WIN, n // 4))

if N_TEST_eff + N_CAL_eff + 50 > n:
    N_TEST_eff = max(MIN_WIN, n // 8)
    N_CAL_eff  = max(MIN_WIN, n // 5)

cut_test = n - N_TEST_eff
cut_cal  = cut_test - N_CAL_eff

if cut_cal < 50:
    print("\n⚠️ Not enough matches for a clean train/cal/test split.")
    print("Try adding another season or lowering N_TEST / N_CAL.")
    cut_test = n
    cut_cal = max(50, n - N_CAL_eff)
    N_TEST_eff = 0

train_df = df_bt.iloc[:cut_cal]
cal_df   = df_bt.iloc[cut_cal:cut_test]
test_df  = df_bt.iloc[cut_test:] if N_TEST_eff > 0 else None

# Fit backtest model on TRAIN only
weights_train = pb.models.dixon_coles_weights(train_df["date"], XI)
model_bt = pb.models.DixonColesGoalModel(
    train_df["goals_home"], train_df["goals_away"],
    train_df["team_home"], train_df["team_away"],
    weights_train
)
model_bt.fit()

# Build calibration data on CAL window (skip matches with unseen teams)
cal_preds = []
cal_actuals = []
cal_skipped = 0

for _, r in cal_df.iterrows():
    try:
        pr = model_bt.predict(r["team_home"], r["team_away"])
    except ValueError:
        cal_skipped += 1
        continue
    p_over_25 = get_p_over25_from_goal_matrix(pr.goal_matrix)
    cal_preds.append(p_over_25)
    cal_actuals.append(1.0 if (r["goals_home"] + r["goals_away"]) >= 3 else 0.0)

cal_preds = np.array(cal_preds, dtype=float)
cal_actuals = np.array(cal_actuals, dtype=float)

print("\nCalibration buckets (Over2.5) [CAL window]:")
print(f"CAL matches: {len(cal_df)} | used: {len(cal_preds)} | skipped (unseen teams): {cal_skipped}")
for lo, hi in zip(BINS[:-1], BINS[1:]):
    mask = (cal_preds >= lo) & (cal_preds < hi if hi != BINS[-1] else cal_preds <= hi)
    if mask.sum() >= 8:
        print(f"{lo:.2f}-{hi:.2f}  n={int(mask.sum()):3d}  pred={cal_preds[mask].mean():.3f}  actual={cal_actuals[mask].mean():.3f}")

print("\nCAL window summary:")
print("avg predicted Over2.5:", round(float(cal_preds.mean()), 3) if len(cal_preds) else "n/a")
print("avg actual Over2.5   :", round(float(cal_actuals.mean()), 3) if len(cal_actuals) else "n/a")
print("Brier (CAL window)   :", round(float(np.mean((cal_preds - cal_actuals) ** 2)), 3) if len(cal_preds) else "n/a")

# Optional: evaluate on TEST window (also skip unseen teams)
if test_df is not None and len(test_df) > 0:
    test_preds = []
    test_actuals = []
    test_skipped = 0

    for _, r in test_df.iterrows():
        try:
            pr = model_bt.predict(r["team_home"], r["team_away"])
        except ValueError:
            test_skipped += 1
            continue
        p_over_25 = get_p_over25_from_goal_matrix(pr.goal_matrix)
        test_preds.append(p_over_25)
        test_actuals.append(1.0 if (r["goals_home"] + r["goals_away"]) >= 3 else 0.0)

    test_preds = np.array(test_preds, dtype=float)
    test_actuals = np.array(test_actuals, dtype=float)

    print("\nTEST window (unseen) Over2.5:")
    print(f"TEST matches: {len(test_df)} | used: {len(test_preds)} | skipped (unseen teams): {test_skipped}")
    if len(test_preds):
        print("avg predicted Over2.5:", round(float(test_preds.mean()), 3))
        print("actual Over2.5 rate  :", round(float(test_actuals.mean()), 3))
        print("Brier score          :", round(float(np.mean((test_preds - test_actuals) ** 2)), 3))

print("\nTraining data averages (ALL data used for the match model):")
print("Avg home goals:", float(df["goals_home"].mean()))
print("Avg away goals:", float(df["goals_away"].mean()))
print("Avg total goals:", float((df["goals_home"] + df["goals_away"]).mean()))
print("Actual Over2.5 rate:", float(((df["goals_home"] + df["goals_away"]) >= 3).mean()))

# 5) Calibrate Over2.5 using narrower buckets from CAL window
if len(cal_preds) < 30:
    print("\n⚠️ Calibration sample too small after skipping. Falling back to RAW for calibration.")
    p_over_cal = p_over_raw
else:
    p_over_cal = calibrate_over25_bucket(p_over_raw, cal_preds, cal_actuals, BINS, MIN_BUCKET_N)

p_over_cal = max(1e-6, min(1 - 1e-6, float(p_over_cal)))
p_under_cal = 1.0 - p_over_cal

print("\nTotals (RAW vs CAL):")
print("RAW Over 2.5:", pct(p_over_raw), "fair:", round(fair_odds(p_over_raw), 2))
print("CAL Over 2.5:", pct(p_over_cal), "fair:", round(fair_odds(p_over_cal), 2))
print("RAW Under 2.5:", pct(p_under_raw), "fair:", round(fair_odds(p_under_raw), 2))
print("CAL Under 2.5:", pct(p_under_cal), "fair:", round(fair_odds(p_under_cal), 2))

# 6) Timing / first goal consistent with calibrated total intensity
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

# 7) Match summary (RAW 1X2/AH from goal matrix)
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


