# ============================================================
# 15→15 BACKTEST (rigoureux) + exécution au 15 du mois suivant
# Règle: info macro du mois t (datée fin-de-mois EOM) -> trade le 15 de t+1
#        donc EOM 2024-01-31 -> trade 2024-02-15 (prix d’entrée)
# ============================================================

# import os
# import numpy as np
# import pandas as pd

# # ============================================================
# # 0) PARAMS À RÉGLER
# # ============================================================

# DATA_15_PATH = "data.xlsx"   # ton fichier avec niveaux au 15: S&P, Gold, 10Y (yield ou index)
# DATE_COL = "date"            # nom de la colonne date après rename
# SPX_COL  = "snp"             # nom col S&P niveaux au 15
# GOLD_COL = "gold"            # nom col Gold niveaux au 15
# Y10_COL  = "10Y"             # nom col 10Y au 15

# Y10_IS_YIELD = True          # True si 10Y = yield (%) ; False si c'est un indice obligataire (niveau)
# D, C = 10.0, 100.0           # duration/convexity approx si yield

# TCOST = 0.001                # 10 bps par switch (au 15)
# OUTDIR = "forecasting"

# # IMPORTANT: tu dois déjà avoir ces objets issus de ton modèle macro :
# # - pred : np.array shape (n_test,) valeurs 0..3
# # - proba: np.array shape (n_test,4) P(phase=0..3)
# # - y_test : pd.Series index EOM (fin de mois) sur le test
# # - phase_names : dict {0:"Recession",1:"Recovery",2:"Slowdown",3:"Expansion"}
# # Si tu n'as pas phase_names, je le définis ci-dessous.

# phase_names = {0:"Recession", 1:"Recovery", 2:"Slowdown", 3:"Expansion"}

# # ============================================================
# # 1) OUTILS DATE (ancrage 15 & mapping EOM->15 du mois suivant)
# # ============================================================

# def anchor_to_15(d) -> pd.Timestamp:
#     d = pd.Timestamp(d)
#     return pd.Timestamp(year=d.year, month=d.month, day=15)

# def eom_to_exec15(eom_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
#     """
#     EOM 2024-01-31 -> MonthBegin(1)=2024-02-01 -> +14j => 2024-02-15
#     """
#     return (pd.DatetimeIndex(eom_index) + pd.offsets.MonthBegin(1) + pd.offsets.Day(14)).to_period("D").to_timestamp()

# # ============================================================
# # 2) Construire rendements 15→15 depuis ton fichier niveaux au 15
# # ============================================================

# def build_15th_returns_from_levels(df_levels_15: pd.DataFrame,
#                                    date_col=DATE_COL, spx_col=SPX_COL, gold_col=GOLD_COL, y10_col=Y10_COL,
#                                    y10_is_yield=True, D=10.0, C=100.0) -> pd.DataFrame:
#     """
#     Retourne un DataFrame rets_15 indexé au 15:
#       rets_15.loc["2024-02-15"] = rendement 15/02 -> 15/03
#     Colonnes: ["bond","gold","snp"]
#     """
#     df = df_levels_15.copy()
#     df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
#     df = df.dropna(subset=[date_col]).sort_values(date_col)

#     # ancrer au 15 (sécurité)
#     df["date15"] = df[date_col].apply(anchor_to_15)
#     df = df.drop_duplicates("date15", keep="last").set_index("date15").sort_index()

#     # niveaux
#     spx  = df[spx_col].astype(float)
#     gold = df[gold_col].astype(float)

#     r_spx  = spx.pct_change()
#     r_gold = gold.pct_change()

#     if y10_is_yield:
#         # yield (%) -> approx return oblig (prix + carry)
#         y = (df[y10_col].astype(float) / 100.0)
#         dy = y.diff()
#         r_bond = (-D * dy) + (0.5 * C * (dy**2)) + (y.shift(1) / 12.0)
#     else:
#         # indice oblig niveau
#         bond = df[y10_col].astype(float)
#         r_bond = bond.pct_change()

#     rets_15 = pd.concat([r_bond.rename("bond"),
#                          r_gold.rename("gold"),
#                          r_spx.rename("snp")], axis=1).dropna()

#     # s'assurer index = 15 de chaque mois
#     rets_15.index = rets_15.index.to_period("M").to_timestamp("M") - pd.offsets.Day(15) + pd.offsets.Day(15)
#     rets_15 = rets_15[~rets_15.index.duplicated(keep="last")].sort_index()

#     return rets_15

# # ============================================================
# # 3) Helpers perf (CAGR, MaxDD, tables annual, rolling)
# # ============================================================

# def cagr_from_cum(cum: pd.Series, periods_per_year=12) -> float:
#     if len(cum) < 2:
#         return np.nan
#     n_years = len(cum) / periods_per_year
#     return float(cum.iloc[-1] ** (1 / n_years) - 1)

# def max_drawdown(cum: pd.Series) -> float:
#     roll = cum.cummax()
#     dd = cum / roll - 1
#     return float(dd.min())

# def annual_calendar_returns(strat: pd.Series, bench: pd.Series) -> pd.DataFrame:
#     ret_df = pd.DataFrame({"strat": strat, "spx": bench}).dropna()
#     annual = (1.0 + ret_df).resample("A-DEC").prod() - 1.0
#     annual.index = annual.index.year
#     annual["excess"] = annual["strat"] - annual["spx"]
#     return (annual * 100).round(2)

# def rolling_window_perf(strat: pd.Series, bench: pd.Series, window=12) -> pd.DataFrame:
#     df = pd.DataFrame({"strat": strat, "spx": bench}).dropna()
#     roll_strat = (1.0 + df["strat"]).rolling(window).apply(np.prod, raw=True) - 1.0
#     roll_spx   = (1.0 + df["spx"]).rolling(window).apply(np.prod, raw=True) - 1.0
#     out = pd.DataFrame({
#         f"Strat {window}m %": (roll_strat * 100).round(2),
#         f"S&P {window}m %":   (roll_spx   * 100).round(2),
#     }).dropna()
#     out[f"Excess {window}m %"] = (out[f"Strat {window}m %"] - out[f"S&P {window}m %"]).round(2)
#     return out

# # ============================================================
# # 4) Charger ton fichier 15 et créer rets_15
# # ============================================================

# lvl = pd.read_excel(DATA_15_PATH)

# # >>> adapte ces rename à ton fichier EXACT (tu l'avais déjà fait, je le remets)
# # Exemple: {'Unnamed: 1':'date','Unnamed: 2':'snp','Unnamed: 8':'gold','Unnamed: 10':'10Y'}
# lvl = lvl.rename(columns={
#     "Unnamed: 1": "date",
#     "Unnamed: 2": "snp",
#     "Unnamed: 8": "gold",
#     "Unnamed: 10": "10Y",
# })

# lvl = lvl[[DATE_COL, SPX_COL, GOLD_COL, Y10_COL]].copy()

# rets_15 = build_15th_returns_from_levels(
#     lvl, date_col=DATE_COL, spx_col=SPX_COL, gold_col=GOLD_COL, y10_col=Y10_COL,
#     y10_is_yield=Y10_IS_YIELD, D=D, C=C
# )

# # ============================================================
# # 5) BACKTEST FULL-STRIKE (100% sur l’actif du régime prédit)
# # ============================================================

# # mapping phase->actif
# phase_to_asset_name = {"Expansion":"snp", "Recovery":"snp", "Recession":"bond", "Slowdown":"gold"}

# test_eom = pd.DatetimeIndex(y_test.index)  # EOM
# sig_asset_eom = pd.Series(pred, index=test_eom).map(phase_names).map(phase_to_asset_name)

# # exécution au 15 du mois suivant
# sig_asset_15 = sig_asset_eom.copy()
# sig_asset_15.index = eom_to_exec15(sig_asset_15.index)

# # aligner sur la grille 15->15 (PAS de ffill avant 1er signal)
# sig_asset_15 = sig_asset_15.reindex(rets_15.index)
# first_sig = sig_asset_15.first_valid_index()
# sig_asset_15 = sig_asset_15.loc[first_sig:].ffill()

# rets_win = rets_15.loc[first_sig:].copy()

# W_fs = (pd.get_dummies(sig_asset_15)
#         .reindex(rets_win.index)
#         .reindex(columns=["snp","bond","gold"], fill_value=0.0))

# strat_fs = (rets_win[["snp","bond","gold"]] * W_fs).sum(axis=1)
# bench    = rets_win["snp"]

# # coûts de transaction au moment du switch (au 15)
# switch_fs = sig_asset_15.ne(sig_asset_15.shift(1)).fillna(False)
# switch_fs.iloc[0] = True
# strat_fs_tc = strat_fs - TCOST * switch_fs.astype(float)

# cum_fs    = (1 + strat_fs).cumprod()
# cum_fs_tc = (1 + strat_fs_tc).cumprod()
# cum_bench = (1 + bench).cumprod()

# print("\n=== FULL-STRIKE (exécution au 15) ===")
# print("Start:", str(cum_fs.index[0].date()), "| End:", str(cum_fs.index[-1].date()))
# print("Final x (no cost):", float(cum_fs.iloc[-1]))
# print("Final x (with cost):", float(cum_fs_tc.iloc[-1]))
# print("Final x (S&P):", float(cum_bench.iloc[-1]))
# print("CAGR (no cost):", f"{cagr_from_cum(cum_fs):.2%}", "| CAGR (S&P):", f"{cagr_from_cum(cum_bench):.2%}")
# print("MaxDD (no cost):", f"{max_drawdown(cum_fs):.2%}", "| MaxDD (S&P):", f"{max_drawdown(cum_bench):.2%}")
# print("Switches:", int(switch_fs.sum()))

# # ============================================================
# # 6) BACKTEST 50/50 TOP-2 PROBAS (toujours exécuté au 15)
# # ============================================================

# phase_probs_eom = pd.DataFrame(proba, index=test_eom, columns=[0,1,2,3])
# phase_to_asset_id = {0:"bond", 1:"snp", 2:"gold", 3:"snp"}

# def equal_top2_weights(row):
#     top2 = row.sort_values(ascending=False).index[:2].tolist()
#     w = {"snp":0.0, "bond":0.0, "gold":0.0}
#     for ph in top2:
#         w[phase_to_asset_id[ph]] += 0.5
#     return pd.Series(w)

# W50_eom = phase_probs_eom.apply(equal_top2_weights, axis=1)
# W50_15 = W50_eom.copy()
# W50_15.index = eom_to_exec15(W50_15.index)

# W50_15 = W50_15.reindex(rets_15.index)
# first_sig_50 = W50_15.dropna(how="all").index.min()
# W50_15 = W50_15.loc[first_sig_50:].ffill().fillna(0.0)

# rets_win_50 = rets_15.loc[W50_15.index].copy()
# strat_50 = (rets_win_50[["snp","bond","gold"]] * W50_15[["snp","bond","gold"]]).sum(axis=1)
# bench_50 = rets_win_50["snp"]

# # coûts (si allocation change, on compte comme switch)
# switch_50 = (W50_15.ne(W50_15.shift(1)).any(axis=1)).fillna(False)
# switch_50.iloc[0] = True
# strat_50_tc = strat_50 - TCOST * switch_50.astype(float)

# cum_50    = (1 + strat_50).cumprod()
# cum_50_tc = (1 + strat_50_tc).cumprod()
# cum_b50   = (1 + bench_50).cumprod()

# print("\n=== 50/50 TOP-2 (exécution au 15) ===")
# print("Start:", str(cum_50.index[0].date()), "| End:", str(cum_50.index[-1].date()))
# print("Final x (no cost):", float(cum_50.iloc[-1]))
# print("Final x (with cost):", float(cum_50_tc.iloc[-1]))
# print("Final x (S&P):", float(cum_b50.iloc[-1]))
# print("CAGR (no cost):", f"{cagr_from_cum(cum_50):.2%}", "| CAGR (S&P):", f"{cagr_from_cum(cum_b50):.2%}")
# print("MaxDD (no cost):", f"{max_drawdown(cum_50):.2%}", "| MaxDD (S&P):", f"{max_drawdown(cum_b50):.2%}")
# print("Switches:", int(switch_50.sum()))

# # ============================================================
# # 7) EXPORTS (rigoureux) + tables annual + rolling
# # ============================================================

# os.makedirs(OUTDIR, exist_ok=True)

# out_fs = pd.DataFrame({
#     "signal_asset": sig_asset_15,
#     "switch": switch_fs.astype(int),
#     "ret_strat": strat_fs,
#     "ret_strat_tc": strat_fs_tc,
#     "ret_spx": bench,
#     "cum_strat": cum_fs,
#     "cum_strat_tc": cum_fs_tc,
#     "cum_spx": cum_bench
# })
# out_fs.to_csv(f"{OUTDIR}/backtest_fullstrike_15th.csv", index_label="date15")

# out_50 = pd.DataFrame({
#     "w_snp": W50_15["snp"],
#     "w_bond": W50_15["bond"],
#     "w_gold": W50_15["gold"],
#     "switch": switch_50.astype(int),
#     "ret_strat": strat_50,
#     "ret_strat_tc": strat_50_tc,
#     "ret_spx": bench_50,
#     "cum_strat": cum_50,
#     "cum_strat_tc": cum_50_tc,
#     "cum_spx": cum_b50
# })
# out_50.to_csv(f"{OUTDIR}/backtest_top2_5050_15th.csv", index_label="date15")

# # annual calendar (sur index au 15: ça reste correct, c'est “année civile” basée sur dates)
# annual_fs = annual_calendar_returns(strat_fs, bench)
# annual_50 = annual_calendar_returns(strat_50, bench_50)
# annual_fs.to_csv(f"{OUTDIR}/annual_fullstrike_15th.csv", index_label="year")
# annual_50.to_csv(f"{OUTDIR}/annual_top2_5050_15th.csv", index_label="year")

# # rolling 12m
# roll12_fs = rolling_window_perf(strat_fs, bench, window=12)
# roll12_50 = rolling_window_perf(strat_50, bench_50, window=12)
# roll12_fs.to_csv(f"{OUTDIR}/rolling_12m_fullstrike_15th.csv", index_label="date15")
# roll12_50.to_csv(f"{OUTDIR}/rolling_12m_top2_5050_15th.csv", index_label="date15")

# print("\nSaved:")
# print(f"- {OUTDIR}/backtest_fullstrike_15th.csv")
# print(f"- {OUTDIR}/backtest_top2_5050_15th.csv")
# print(f"- {OUTDIR}/annual_fullstrike_15th.csv")
# print(f"- {OUTDIR}/annual_top2_5050_15th.csv")
# print(f"- {OUTDIR}/rolling_12m_fullstrike_15th.csv")
# print(f"- {OUTDIR}/rolling_12m_top2_5050_15th.csv")





#############

# Bon code ci dessous avec l'extraction des données nécessaires

#############


import os
import numpy as np
import pandas as pd

# =========================
# PARAMS
# =========================
PRED_PATH = "forecasting/test_true_vs_pred_plus_last_forecast.csv"  # ton fichier (avec mois t, mois t+1, pred, true)
PRICES_15_PATH = "data_15au15.xlsx"                                        # fichier niveaux au 15 : S&P, Gold, 10Y

TCOST = 0.001          # 10 bps par switch (au 15)
Y10_IS_YIELD = False    # True si 10Y = yield (%) ; False si index obligataire (niveau)
D, C = 10.0, 100.0     # duration/convexity approx si yield
OUTDIR = "forecasting"

# mapping phase -> nom -> actif
phase_names = {0:"Recession", 1:"Recovery", 2:"Slowdown", 3:"Expansion"}
phase_to_asset = {0:"bond", 1:"snp", 2:"gold", 3:"snp"}  # recession=bond, slowdown=gold, recovery/expansion=snp

# =========================
# HELPERS
# =========================
def to_15_of_month(x):
    """x peut être '2024-02-29' ou '2024-02-01' etc -> renvoie 2024-02-15"""
    t = pd.Timestamp(x)
    return pd.Timestamp(t.year, t.month, 15)
def build_15th_returns_from_levels(df, date_col, spx_col, gold_col, y10_col, y10_is_yield=True, D=10.0, C=100.0):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # ancrer au 15
    df["date15"] = df[date_col].apply(lambda d: pd.Timestamp(d.year, d.month, 15))
    df = df.drop_duplicates("date15", keep="last").set_index("date15").sort_index()

    spx  = df[spx_col].astype(float)
    gold = df[gold_col].astype(float)

    # ✅ forward returns (t -> t+1) indexés à t (date d'entrée)
    r_spx_fwd  = spx.shift(-1)  / spx  - 1
    r_gold_fwd = gold.shift(-1) / gold - 1

    if y10_is_yield:
        y = (df[y10_col].astype(float) / 100.0)
        dy = y.diff()
        bond_ret = (-D * dy) + (0.5 * C * (dy**2)) + (y.shift(1) / 12.0)
        bond_ret.index = df.index
        r_bond_fwd = bond_ret.shift(-1)
    else:
        bond = df[y10_col].astype(float)
        r_bond_fwd = bond.shift(-1) / bond - 1

    rets_15 = pd.concat([
        r_bond_fwd.rename("bond"),
        r_gold_fwd.rename("gold"),
        r_spx_fwd.rename("snp")
    ], axis=1).dropna()

    return rets_15


def cagr(cum, periods_per_year=12):
    if len(cum) < 2:
        return np.nan
    n_years = len(cum) / periods_per_year
    return float(cum.iloc[-1] ** (1/n_years) - 1)

def max_dd(cum):
    roll = cum.cummax()
    dd = cum/roll - 1
    return float(dd.min())

# =========================
# 1) LOAD PRED FILE
# =========================
pred_df = pd.read_csv(PRED_PATH)
# colonnes attendues (adaptables): asof_month, forecast_for_month, pred_phase_id, true_phase_id
# si tes noms diffèrent, remplace ici.
needed = ["forecast_for_month", "pred_phase_id"]
for c in needed:
    if c not in pred_df.columns:
        raise ValueError(f"Colonne manquante dans {PRED_PATH}: {c}. Colonnes dispo: {pred_df.columns.tolist()}")

pred_df["forecast_for_month"] = pd.to_datetime(pred_df["forecast_for_month"], errors="coerce")
pred_df = pred_df.dropna(subset=["forecast_for_month"])

# date d'exécution = 15 du mois forecast_for_month
pred_df["exec_date15"] = pred_df["forecast_for_month"].apply(to_15_of_month)

# signal = actif à détenir sur exec_date15 -> exec_date15 suivant
pred_df["signal_asset"] = pred_df["pred_phase_id"].map(phase_to_asset)

# garder seulement lignes avec signal valide
pred_df = pred_df.dropna(subset=["signal_asset"]).copy()
pred_df = pred_df.sort_values("exec_date15")

sig = pd.Series(pred_df["signal_asset"].values, index=pd.DatetimeIndex(pred_df["exec_date15"]), name="signal_asset")
sig = sig[~sig.index.duplicated(keep="last")].sort_index()

# =========================
# 2) LOAD 15th PRICES + BUILD 15→15 RETURNS
# =========================
lvl = pd.read_excel(PRICES_15_PATH, sheet_name='Feuil2')
print(lvl.head())
# >>> adapte ce rename à ton vrai fichier PRICES_15_PATH
lvl = lvl.rename(columns={
    "Unnamed: 0": "date",
    "S&P": "snp",
    "Gold": "gold",
    "10Y": "10Y",
})


lvl = lvl[["date", "snp", "gold", "10Y"]].copy()

rets_15 = build_15th_returns_from_levels(
    lvl, date_col="date", spx_col="snp", gold_col="gold", y10_col="10Y",
    y10_is_yield=Y10_IS_YIELD, D=D, C=C
)

# =========================
# 3) ALIGN SIGNALS ON RETURNS GRID (NO LOOKAHEAD)
# =========================
# On ne garde que les signaux qui tombent sur des dates où on a des retours 15->15
sig = sig.reindex(rets_15.index)

first_sig = sig.first_valid_index()
if first_sig is None:
    raise ValueError("Aucun signal ne match les dates 15->15 de tes prix. Vérifie les dates / fichiers.")

sig = sig.loc[first_sig:].ffill()     # OK après 1er signal seulement
rets_win = rets_15.loc[first_sig:].copy()

# =========================
# 4) FULL-STRIKE BACKTEST
# =========================
W = (pd.get_dummies(sig)
     .reindex(rets_win.index)
     .reindex(columns=["snp","bond","gold"], fill_value=0.0))

strat_rets = (rets_win[["snp","bond","gold"]] * W).sum(axis=1)
bench_rets = rets_win["snp"]

switch = sig.ne(sig.shift(1)).fillna(False)
switch.iloc[0] = True
strat_rets_tc = strat_rets - TCOST * switch.astype(float)

cum_strat = (1 + strat_rets).cumprod()
cum_tc    = (1 + strat_rets_tc).cumprod()
cum_spx   = (1 + bench_rets).cumprod()

print("\n=== FULL-STRIKE (exécution le 15, retours 15→15) ===")
print("Start:", cum_strat.index[0].date(), "| End:", cum_strat.index[-1].date())
print("Final x (no cost):", float(cum_strat.iloc[-1]))
print("Final x (with cost):", float(cum_tc.iloc[-1]))
print("Final x (S&P):", float(cum_spx.iloc[-1]))
print("CAGR strat:", f"{cagr(cum_strat):.2%}", "| CAGR S&P:", f"{cagr(cum_spx):.2%}")
print("MaxDD strat:", f"{max_dd(cum_strat):.2%}", "| MaxDD S&P:", f"{max_dd(cum_spx):.2%}")
print("Switches:", int(switch.sum()))

# =========================
# 5) BENCHMARK 1/3-1/3-1/3 (hold ou rééquilibré mensuellement)
# =========================

# Option 1 : Benchmark 1/3-1/3-1/3 SANS rééquilibrage (buy & hold)
# On suppose que tu investis 1/3 dans chaque actif au début et tu ne touches plus à l'allocation.
initial_weights = {"snp": 1/3, "bond": 1/3, "gold": 1/3}
benchmark_buy_hold_rets = (rets_win[["snp", "bond", "gold"]] * pd.Series(initial_weights)).sum(axis=1)
benchmark_buy_hold_cum = (1 + benchmark_buy_hold_rets).cumprod()

# Option 2 : Benchmark 1/3-1/3-1/3 AVEC rééquilibrage mensuel (plus réaliste)
# Chaque mois, on rééquilibre pour retrouver 1/3-1/3-1/3.
# Ici, pas de coût de transaction pour le benchmark (mais tu peux en ajouter si tu veux).
benchmark_rebalanced_rets = (rets_win[["snp", "bond", "gold"]] * pd.Series(initial_weights)).sum(axis=1)
benchmark_rebalanced_cum = (1 + benchmark_rebalanced_rets).cumprod()

# =========================
# 6) AFFICHAGE DES RÉSULTATS AVEC BENCHMARK
# =========================

print("\n=== COMPARAISON AVEC BENCHMARK 1/3-1/3-1/3 ===")
print("Start:", cum_strat.index[0].date(), "| End:", cum_strat.index[-1].date())
print("Final x (Full-Strike no cost):", float(cum_strat.iloc[-1]))
print("Final x (Full-Strike with cost):", float(cum_tc.iloc[-1]))
print("Final x (S&P):", float(cum_spx.iloc[-1]))
print("Final x (Benchmark Buy & Hold 1/3-1/3-1/3):", float(benchmark_buy_hold_cum.iloc[-1]))
print("Final x (Benchmark Rééquilibré 1/3-1/3-1/3):", float(benchmark_rebalanced_cum.iloc[-1]))

print("\nCAGR Full-Strike:", f"{cagr(cum_strat):.2%}")
print("CAGR S&P:", f"{cagr(cum_spx):.2%}")
print("CAGR Benchmark Buy & Hold:", f"{cagr(benchmark_buy_hold_cum):.2%}")
print("CAGR Benchmark Rééquilibré:", f"{cagr(benchmark_rebalanced_cum):.2%}")

print("\nMaxDD Full-Strike:", f"{max_dd(cum_strat):.2%}")
print("MaxDD S&P:", f"{max_dd(cum_spx):.2%}")
print("MaxDD Benchmark Buy & Hold:", f"{max_dd(benchmark_buy_hold_cum):.2%}")
print("MaxDD Benchmark Rééquilibré:", f"{max_dd(benchmark_rebalanced_cum):.2%}")


# =========================
# 5) EXPORT
# =========================
os.makedirs(OUTDIR, exist_ok=True)

out = pd.DataFrame({
    "signal_asset": sig,
    "switch": switch.astype(int),
    "ret_strat": strat_rets,
    "ret_strat_tc": strat_rets_tc,
    "ret_spx": bench_rets,
    "cum_strat": cum_strat,
    "cum_strat_tc": cum_tc,
    "cum_spx": cum_spx
})
out.to_csv(f"{OUTDIR}/backtest_from_pred_file_15th.csv", index_label="date15")
print(f"Saved -> {OUTDIR}/backtest_from_pred_file_15th.csv")

# (Option) précision classification si tu as true_phase_id dans le fichier
if "true_phase_id" in pred_df.columns:
    tmp = pred_df.dropna(subset=["true_phase_id"]).copy()
    tmp["true_phase_id"] = tmp["true_phase_id"].astype(int)
    tmp["pred_phase_id"] = tmp["pred_phase_id"].astype(int)
    acc = (tmp["true_phase_id"] == tmp["pred_phase_id"]).mean()
    print(f"Accuracy (sur lignes avec vérité dispo) : {acc:.3f}")

    # =========================
# 7) EXPORT AVEC BENCHMARK
# =========================

out["ret_benchmark_buy_hold"] = benchmark_buy_hold_rets
out["ret_benchmark_rebalanced"] = benchmark_rebalanced_rets
out["cum_benchmark_buy_hold"] = benchmark_buy_hold_cum
out["cum_benchmark_rebalanced"] = benchmark_rebalanced_cum

out.to_csv(f"{OUTDIR}/backtest_from_pred_file_15th_with_benchmark.csv", index_label="date15")
print(f"Saved -> {OUTDIR}/backtest_from_pred_file_15th_with_benchmark.csv")

# =========================
# 8) PERFS ANNUELLES 15→15
# =========================
def annual_15to15_returns(ret_series: pd.Series) -> pd.Series:
    """
    Regroupe les rendements mensuels indexés au 15 en 'années 15→15'.
    Convention: année Y = 15/02/Y .. 15/01/Y+1.
    """
    s = ret_series.dropna().copy()
    idx = pd.DatetimeIndex(s.index)

    # bucket d'année 15→15 : Feb..Dec -> year ; Jan -> year-1
    year15 = np.where(idx.month >= 2, idx.year, idx.year - 1)

    ann = (1.0 + s).groupby(year15).prod() - 1.0
    ann.index.name = "year15"
    return ann

# calculs (strat, strat_tc, spx, benchmark)
ann_strat   = annual_15to15_returns(strat_rets)
ann_stratTC = annual_15to15_returns(strat_rets_tc)
ann_spx     = annual_15to15_returns(bench_rets)
ann_bh      = annual_15to15_returns(benchmark_buy_hold_rets)
ann_rb      = annual_15to15_returns(benchmark_rebalanced_rets)

annual15 = pd.DataFrame({
    "strat_%":    (ann_strat   * 100).round(2),
    "strat_tc_%": (ann_stratTC * 100).round(2),
    "spx_%":      (ann_spx     * 100).round(2),
    "bh_1_3_%":   (ann_bh      * 100).round(2),
    "rb_1_3_%":   (ann_rb      * 100).round(2),
})
annual15["excess_vs_spx_%"]    = (annual15["strat_%"] - annual15["spx_%"]).round(2)
annual15["excess_tc_vs_spx_%"] = (annual15["strat_tc_%"] - annual15["spx_%"]).round(2)

print("\n=== Perfs annuelles 15→15 (année Y = 15/02/Y .. 15/01/Y+1) ===")
print(annual15.to_string())

annual15.to_csv(f"{OUTDIR}/annual_15to15_returns.csv", index_label="year15")
print(f"Saved -> {OUTDIR}/annual_15to15_returns.csv")