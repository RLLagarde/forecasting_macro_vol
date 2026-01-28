import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================
print("Chargement des données...")
df = pd.read_excel("vol_1.xlsx")
df = df.drop(df.index[0])

df = df.rename(columns={"SPX Index": "Returns30"})  # prix SPX
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

df = df[2300:]  # si tu veux tronquer
df_raw_spx = df["Returns30"].copy()
df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# ============================================================
# PARAMS
# ============================================================
HORIZON     = 4
LOOK_AHEAD  = 4
train_window = 2128

assert LOOK_AHEAD >= HORIZON, "⚠️ LOOK_AHEAD doit être >= HORIZON pour éviter le chevauchement / leakage."

total_obs = len(df)
print(f"Total observations: {total_obs}")
print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")
print(f"HORIZON: {HORIZON} | LOOK_AHEAD: {LOOK_AHEAD} | train_window: {train_window}")

# ============================================================
# TARGET (avec gestion propre des NaN)
# ============================================================
df["log_price"] = np.log(df["Returns30"])
df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]

# quantile sur la dernière année (252 j)
quantile = df["future_return"].iloc[-252:].quantile(0.15)

# ⚠️ IMPORTANT : préserver les NaN (sinon tu crées des faux 0 à la fin)
y = pd.Series(np.where(df["future_return"].isna(), np.nan, (df["future_return"] < quantile).astype(int)),
              index=df.index)

print(f"Taux 'crash' (y=1) approx (hors NaN): {np.nanmean(y.values):.2%}")

# ============================================================
# FEATURES
# ============================================================
features_to_drop = [
    "VIXTS Index",
    "1YCVBRNT Index",
    "BNPEPCN Index", "BNPEPEU Index", "BNPEPJP Index", "BNPEPUS Index",
    "BNPEPUS2 Index", "BNPEPUS4 Index", "BNPEPUS5 Index", "BNPEPUS6 Index", "BNPEPUS7 Index",
    "BNPTEUS6 Index",
    "BTC Index",
    "CGERGLEM Index",
    "COR1M Index",
    "EMUSTRUU Index",
    "EURUSDV3M BGN Curncy",
    "EUSS10 CMPN Curncy", "EUSS2 CMPN Curncy", "EUSS30 CMPN Curncy",
    "GFSI Index",
    "LBEATREU Index", "LEGATRUU Index", "LF98TRUU Index", "LG30TRUU Index", "LP01TREU Index",
    "MOVE Index",
    "MXWO Index",
    "SPJGBVRT Index",
    "SPX Index",
    "USDJPYV3M BGN Curncy",
    "USSFCT03 BGN Curncy", "USSFCT10 BGN Curncy", "USSFCT30 BGN Curncy",
    "V1X Index", "V2X Index",
    "VXEEM Index",
    "XBT Curncy",

    # colonnes à exclure (prix/target)
    "Returns30", "log_price", "future_return",
]

X = df.drop(columns=[c for c in features_to_drop if c in df.columns])

X = X.rename(columns={
    "VIX Index": "VIX",
    "VIX9D Index": "VIX9D",
    "VIX3M Index": "VIX3M",
    "VIX6M Index": "VIX6M",
    "VVIX Index": "VVIX",
    "SKEW Index": "SKEW",
    "VXN Index": "VXN",
    "GVZ Index": "GVZ",
    "OVX Index": "OVX",
    "PCUSEQTR Index": "PUTCALL_RATIO",
    "RVOL Index": "RVOL",
})

print(f"Features utilisées: {list(X.columns)}")
print(f"Nombre de features: {len(X.columns)}")

# ============================================================
# SCALING GLOBAL (attention: peut introduire du look-ahead indirect)
# ============================================================
print("\n=== APPLYING GLOBAL SCALING ===")
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X)
X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)

# ============================================================
# FIXE TES PARAMS (à la main, plus de grid search)
# ============================================================
best_params = {
    "n_estimators": 300,
    "max_features": 0.5,   # ou 0.5
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_depth": 12,          # ou None
}
best_threshold = 0.25

print("\n=== CONFIG FIXE ===")
print("Params:", best_params)
print("Threshold:", best_threshold)

# ============================================================
# WALK-FORWARD
# ============================================================
first_test_idx = train_window
last_test_idx  = total_obs - HORIZON - 1  # il faut que future_return soit connu

y_true, y_pred, y_proba, test_dates = [], [], [], []

for test_idx in range(first_test_idx, last_test_idx + 1):

    train_start = test_idx - train_window - LOOK_AHEAD
    train_end   = test_idx - LOOK_AHEAD   # exclusif dans iloc slicing ci-dessous

    if train_start < 0:
        continue

    X_train = X_scaled_global.iloc[train_start:train_end]
    y_train = y.iloc[train_start:train_end]

    # virer les NaN dans y_train (sinon crash)
    mask = y_train.notna()
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask].astype(int)

    # y_test doit exister (pas NaN)
    y_test = y.iloc[test_idx]
    if pd.isna(y_test):
        continue

    X_test = X_scaled_global.iloc[test_idx:test_idx+1]

    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_features=best_params["max_features"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_depth=best_params["max_depth"],
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[0, 1]
    pred  = 1 if proba > best_threshold else 0

    y_true.append(int(y_test))
    y_pred.append(int(pred))
    y_proba.append(float(proba))
    test_dates.append(df.index[test_idx])

    if len(y_true) % 200 == 0:
        print(f"  Prédictions: {len(y_true)}...")

# ============================================================
# METRICS
# ============================================================
print("\n=== RESULTS ===")
print("N preds:", len(y_true))

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan
    f1  = f1_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("Test period:", test_dates[0].date(), "->", test_dates[-1].date())

# ============================================================
# EXPORT CSV : TRUE vs PRED
# ============================================================
pred_vs_true = pd.DataFrame({
    "date": pd.to_datetime(test_dates),
    "y_true": y_true,
    "y_pred": y_pred,
    "proba_1": y_proba,
}).sort_values("date").reset_index(drop=True)

pred_vs_true["logret_fwd"] = df.loc[pred_vs_true["date"], "future_return"].astype(float).values
pred_vs_true["ret_fwd"]    = np.exp(pred_vs_true["logret_fwd"]) - 1

pred_vs_true["HORIZON"]     = HORIZON
pred_vs_true["LOOK_AHEAD"]  = LOOK_AHEAD
pred_vs_true["train_window"] = train_window
pred_vs_true["threshold"]   = best_threshold

pred_vs_true.to_csv("pred_vs_true_global_scaling.csv", index=False)
print("\n✔️ Exported -> pred_vs_true_global_scaling.csv")

# ============================================================
# ANALYSE DES "VRAIS" RETOURS PAR CAS (comme tu voulais)
# ============================================================
res = pred_vs_true.set_index("date").sort_index()

def mean_n(s):
    s = s.dropna()
    return float(s.mean()) if len(s) else np.nan, int(len(s))

print("\n=== Moyennes ret_fwd (simple return) sur", HORIZON, "jours ===")

m, n = mean_n(res.loc[res["y_pred"] == 1, "ret_fwd"])
print(f"Mean ret_fwd | pred=1 : {m:.6f} (n={n})")

m, n = mean_n(res.loc[res["y_pred"] == 0, "ret_fwd"])
print(f"Mean ret_fwd | pred=0 : {m:.6f} (n={n})")

m, n = mean_n(res.loc[res["y_true"] == 1, "ret_fwd"])
print(f"Mean ret_fwd | true=1 : {m:.6f} (n={n})")

m, n = mean_n(res.loc[res["y_true"] == 0, "ret_fwd"])
print(f"Mean ret_fwd | true=0 : {m:.6f} (n={n})")

tp = (res["y_pred"] == 1) & (res["y_true"] == 1)
fp = (res["y_pred"] == 1) & (res["y_true"] == 0)
tn = (res["y_pred"] == 0) & (res["y_true"] == 0)
fn = (res["y_pred"] == 0) & (res["y_true"] == 1)

m, n = mean_n(res.loc[tp, "ret_fwd"]); print(f"Mean ret_fwd | TP : {m:.6f} (n={n})")
m, n = mean_n(res.loc[fp, "ret_fwd"]); print(f"Mean ret_fwd | FP : {m:.6f} (n={n})")
m, n = mean_n(res.loc[tn, "ret_fwd"]); print(f"Mean ret_fwd | TN : {m:.6f} (n={n})")
m, n = mean_n(res.loc[fn, "ret_fwd"]); print(f"Mean ret_fwd | FN : {m:.6f} (n={n})")