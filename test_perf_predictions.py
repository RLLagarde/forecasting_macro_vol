import pandas as pd
import numpy as np

# ============================
# Utils
# ============================
def rolling_ols_slope(s, window):
    def _slope(y):
        if np.any(~np.isfinite(y)):
            return np.nan
        x = np.arange(len(y))
        x = (x - x.mean()) / (x.std() if x.std() != 0 else 1.0)
        return np.cov(x, y, bias=True)[0,1] / np.var(x)
    return s.rolling(window, min_periods=window).apply(_slope, raw=True)

# ============================
# 1) Load predictions
# ============================
pred = pd.read_csv("forecasting/test_true_vs_pred.csv", parse_dates=["date"])
pred = pred.rename(columns={"date": "eom"})
pred["eom"] = pred["eom"].dt.to_period("M").dt.to_timestamp("M")
pred = pred.sort_values("eom")

# ============================
# 2) Load market data (levels at EOM)
# ============================
df = pd.read_excel("data.xlsx", sheet_name="Valeurs")
print(df.head())
df = df.rename(columns={
    "Unnamed: 0":"date",
    "S&P 500 INDEX":"snp",
    "Bloomberg Crude Oil Historical Price":"oil",
    "Unnamed: 8":"gold",
    "US Generic Govt 10 Yr":"bond"
})
df = df[["date","snp","gold","bond"]]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df["eom"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
df = df.drop(columns=["date"]).sort_values("eom").reset_index(drop=True)

# forward returns (t -> t+1), indexed at t
snp_px  = df.set_index("eom")["snp"].astype(float)
gold_px = df.set_index("eom")["gold"].astype(float)

spx_ret_fwd  = snp_px.shift(-1) / snp_px - 1
gold_ret_fwd = gold_px.shift(-1) / gold_px - 1

yld = (df["bond"] / 100.0).astype(float)
dy  = yld.diff()
D, C = 10.0, 100.0
bond_ret = (-D * dy) + (0.5 * C * dy**2) + (yld.shift(1) / 12.0)
bond_ret.index = df["eom"]
bond_ret_fwd = bond_ret.shift(-1)

rets = pd.DataFrame({
    "eom": spx_ret_fwd.index,
    "spx_ret": spx_ret_fwd.values,
    "gold_ret": gold_ret_fwd.values,
    "bond_ret": bond_ret_fwd.values
}).dropna()

# ============================
# 3) Build M2-up signal (monthly.csv)
# ============================
m = pd.read_csv("monthly.csv", skiprows=[1], sep=";")
print(m.head())
m["date"] = pd.to_datetime(m.iloc[:,0], errors="coerce")
m = m.drop(columns=m.columns[0])
m.index = m["date"].dt.to_period("M").dt.to_timestamp("M")
m = m.drop(columns=["date"])
print(m.columns)
m["CPI"] = np.log(m["CPIAUCSL"])
m["M2"]  = np.log(m["M2SL"])

m2_real = (m["M2"] - m["CPI"]).diff(12)
m2_real = m2_real.rolling(6, min_periods=6).mean()
m2_slope = rolling_ols_slope(m2_real, window=9)
m2_up = (m2_slope > 0).astype(int).rename("m2_up")

# ============================
# 4) Merge all
# ============================
df = (
    rets
    .merge(pred.rename(columns={"pred": "pred_phase_id"}), on="eom", how="inner")
    .merge(m2_up.rename("m2_up").reset_index().rename(columns={"date":"eom"}), on="eom", how="left")
)
# ============================
# 5) Overlay M2 on predictions
# ============================
df["phase_overlay"] = df["pred_phase_id"]

# Recession + M2-up → special regime
df.loc[(df["pred_phase_id"] == 0) & (df["m2_up"] == 1), "phase_overlay"] = 4

# ============================
# 6) Allocation
# ============================
def strat_return(row):
    p = int(row["phase_overlay"])
    if p in (1,3):      # Recovery / Expansion
        return row["spx_ret"]
    if p == 2:          # Slowdown
        return row["gold_ret"]
    if p == 0:          # Recession
        return row["bond_ret"]
    if p == 4:          # Recession + M2-up
        return row["spx_ret"]
    return 0.0

df["ret_strat"] = df.apply(strat_return, axis=1)
df["ret_spx"]   = df["spx_ret"]

# ============================
# 6bis) Transaction costs (10 bps per switch)
# ============================
TCOST = 0.001  # 10 bps

# actif choisi chaque mois (pour détecter les changements)
def chosen_asset(p):
    if p in (1,3):  # Recovery / Expansion
        return "spx"
    if p == 2:      # Slowdown
        return "gold"
    if p == 0:      # Recession
        return "bond"
    if p == 4:      # Recession + M2-up
        return "spx"
    return "cash"

df["asset"] = df["phase_overlay"].apply(chosen_asset)

# switch = 1 quand on change d'actif vs mois précédent
df["switch"] = (df["asset"] != df["asset"].shift(1)).astype(int)

# option conservatrice : on paie l'entrée initiale (premier mois investi)
if len(df) > 0:
    df.loc[df.index[0], "switch"] = 1

# rendement net après frais
df["ret_strat_tc"] = df["ret_strat"] - TCOST * df["switch"]

# # ============================
# # 7) Performance
# # ============================
# df["cap_strat"] = 100 * (1 + df["ret_strat"]).cumprod()
# df["cap_spx"]   = 100 * (1 + df["ret_spx"]).cumprod()

# df["year"] = df["eom"].dt.year
# annual = df.groupby("year")[["ret_strat","ret_spx"]].apply(lambda g: (1+g).prod()-1)
# annual.columns = ["strat","spx"]
# annual["excess"] = annual["strat"] - annual["spx"]

# ============================
# 7) Performance
# ============================
df["cap_strat"]    = 100 * (1 + df["ret_strat"]).cumprod()
df["cap_strat_tc"] = 100 * (1 + df["ret_strat_tc"]).cumprod()
df["cap_spx"]      = 100 * (1 + df["ret_spx"]).cumprod()

print("\n=== Résumé (multiplicateurs) ===")
print("Final SPX hold        :", (df["cap_spx"].iloc[-1] / 100).round(3), "x")
print("Final Strat (no cost) :", (df["cap_strat"].iloc[-1] / 100).round(3), "x")
print("Final Strat (10bp tc) :", (df["cap_strat_tc"].iloc[-1] / 100).round(3), "x")
print("Nb switches           :", int(df["switch"].sum()))

df["year"] = df["eom"].dt.year

annual = df.groupby("year")[["ret_strat","ret_strat_tc","ret_spx"]].apply(lambda g: (1+g).prod()-1)
annual.columns = ["strat_no_cost","strat_10bp_tc","spx"]
annual["excess_no_cost"] = annual["strat_no_cost"] - annual["spx"]
annual["excess_10bp_tc"] = annual["strat_10bp_tc"] - annual["spx"]


# ============================
# 8) Exports
# ============================
df.to_csv("path_strat_overlay_M2.csv", index=False)
(annual*100).round(2).to_csv("perf_annuelles_overlay_M2.csv")


import pandas as pd
import numpy as np

# ============================
# Utils
# ============================
def rolling_ols_slope(s, window):
    def _slope(y):
        if np.any(~np.isfinite(y)):
            return np.nan
        x = np.arange(len(y))
        x = (x - x.mean()) / (x.std() if x.std() != 0 else 1.0)
        return np.cov(x, y, bias=True)[0,1] / np.var(x)
    return s.rolling(window, min_periods=window).apply(_slope, raw=True)

# ============================
# 1) Load predictions
# ============================
pred = pd.read_csv("forecasting/test_true_vs_pred.csv", parse_dates=["date"])
pred = pred.rename(columns={"date": "eom"})
pred["eom"] = pred["eom"].dt.to_period("M").dt.to_timestamp("M")
pred = pred.sort_values("eom")

# ============================
# 2) Load market data (levels at EOM)
# ============================
df = pd.read_excel("data.xlsx", sheet_name="Valeurs")
df = df.rename(columns={
    "Unnamed: 0":"date",
    "S&P 500 INDEX":"snp",
    "Bloomberg Crude Oil Historical Price":"oil",
    "Unnamed: 8":"gold",
    "US Generic Govt 10 Yr":"bond"
})
df = df[["date","snp","gold","bond"]]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df["eom"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
df = df.drop(columns=["date"]).sort_values("eom").reset_index(drop=True)

# forward returns (t -> t+1), indexed at t
snp_px  = df.set_index("eom")["snp"].astype(float)
gold_px = df.set_index("eom")["gold"].astype(float)

spx_ret_fwd  = snp_px.shift(-1) / snp_px - 1
gold_ret_fwd = gold_px.shift(-1) / gold_px - 1

# bond: yield -> approx return (t -> t+1) then shift(-1) to make it forward like the others
yld = (df["bond"] / 100.0).astype(float)
dy  = yld.diff()
D, C = 10, 100.0
bond_ret = (-D * dy) + (0.5 * C * dy**2) + (yld.shift(1) / 12.0)
bond_ret.index = df["eom"]
bond_ret_fwd = bond_ret.shift(-1)

rets = pd.DataFrame({
    "eom": spx_ret_fwd.index,
    "spx_ret": spx_ret_fwd.values,
    "gold_ret": gold_ret_fwd.values,
    "bond_ret": bond_ret_fwd.values
}).dropna()

# ============================
# 3) Build M2-up signal (monthly.csv)
# ============================
m = pd.read_csv("monthly.csv", skiprows=[1], sep=";")
m["date"] = pd.to_datetime(m.iloc[:,0], errors="coerce")
m = m.drop(columns=m.columns[0])
m.index = m["date"].dt.to_period("M").dt.to_timestamp("M")
m = m.drop(columns=["date"])

m["CPI"] = np.log(m["CPIAUCSL"])
m["M2"]  = np.log(m["M2SL"])

m2_real = (m["M2"] - m["CPI"]).diff(12)
m2_real = m2_real.rolling(6, min_periods=6).mean()
m2_slope = rolling_ols_slope(m2_real, window=9)
m2_up = (m2_slope > 0).astype(int).rename("m2_up")

# ============================
# 4) Merge all
# ============================
df = (
    rets
    .merge(pred.rename(columns={"pred": "pred_phase_id"}), on="eom", how="inner")
    .merge(m2_up.rename("m2_up").reset_index().rename(columns={"date":"eom"}), on="eom", how="left")
)

# ============================
# 5) Overlay M2 on predictions
# ============================
df["phase_overlay"] = df["pred_phase_id"]
df.loc[(df["pred_phase_id"] == 0) & (df["m2_up"] == 1), "phase_overlay"] = 4  # Recession + M2-up

# ============================
# 6) Allocation (strategy)
# ============================
def strat_return(row):
    p = int(row["phase_overlay"])
    if p in (1,3):      # Recovery / Expansion
        return row["spx_ret"]
    if p == 2:          # Slowdown
        return row["gold_ret"]
    if p == 0:          # Recession
        return row["bond_ret"]
    if p == 4:          # Recession + M2-up
        return row["spx_ret"]
    return 0.0

df["ret_strat"] = df.apply(strat_return, axis=1)
df["ret_spx"]   = df["spx_ret"]

# ============================
# 6-ter) Benchmark 1/3 - 1/3 - 1/3 (rebal mensuel)
# ============================
df["ret_bench_333"] = (df["spx_ret"] + df["gold_ret"] + df["bond_ret"]) / 3.0

# ============================
# 6bis) Transaction costs (10 bps per switch) - strategy only
# ============================
TCOST = 0.001  # 10 bps

def chosen_asset(p):
    if p in (1,3):  return "spx"
    if p == 2:      return "gold"
    if p == 0:      return "bond"
    if p == 4:      return "spx"
    return "cash"

df["asset"] = df["phase_overlay"].apply(chosen_asset)
df["switch"] = (df["asset"] != df["asset"].shift(1)).astype(int)
if len(df) > 0:
    df.loc[df.index[0], "switch"] = 1

df["ret_strat_tc"] = df["ret_strat"] - TCOST * df["switch"]

# (Option) coûts benchmark 1/3-1/3-1/3 : si tu veux les compter, c'est ~0 (rebal mensuel) mais
# il faut des hypothèses de turnover. Ici je mets 0 par défaut.
df["ret_bench_333_tc"] = df["ret_bench_333"]

# ============================
# 7) Performance
# ============================
df["cap_strat"]       = 100 * (1 + df["ret_strat"]).cumprod()
df["cap_strat_tc"]    = 100 * (1 + df["ret_strat_tc"]).cumprod()
df["cap_spx"]         = 100 * (1 + df["ret_spx"]).cumprod()
df["cap_bench_333"]   = 100 * (1 + df["ret_bench_333"]).cumprod()
df["cap_bench_333_tc"]= 100 * (1 + df["ret_bench_333_tc"]).cumprod()

print("\n=== Résumé (multiplicateurs) ===")
print("Final SPX hold             :", (df["cap_spx"].iloc[-1] / 100).round(3), "x")
print("Final Bench 1/3-1/3-1/3     :", (df["cap_bench_333"].iloc[-1] / 100).round(3), "x")
print("Final Strat (no cost)      :", (df["cap_strat"].iloc[-1] / 100).round(3), "x")
print("Final Strat (10bp tc)      :", (df["cap_strat_tc"].iloc[-1] / 100).round(3), "x")
print("Nb switches (strategy)     :", int(df["switch"].sum()))

df["year"] = df["eom"].dt.year

annual = df.groupby("year")[["ret_strat","ret_strat_tc","ret_spx","ret_bench_333"]].apply(lambda g: (1+g).prod()-1)
annual.columns = ["strat_no_cost","strat_10bp_tc","spx","bench_333"]
annual["excess_vs_spx_no_cost"]   = annual["strat_no_cost"] - annual["spx"]
annual["excess_vs_spx_10bp_tc"]   = annual["strat_10bp_tc"] - annual["spx"]
annual["excess_vs_bench_no_cost"] = annual["strat_no_cost"] - annual["bench_333"]
annual["excess_vs_bench_10bp_tc"] = annual["strat_10bp_tc"] - annual["bench_333"]

print("\n=== Perf annuelles (%) ===")
print((annual*100).round(2))

# ============================
# 8) Exports
# ============================
df.to_csv("path_strat_overlay_M2_with_bench333.csv", index=False)
(annual*100).round(2).to_csv("perf_annuelles_overlay_M2_with_bench333.csv")