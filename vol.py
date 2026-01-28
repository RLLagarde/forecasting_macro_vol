import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# ============================================================
# 1) LOAD DATA
# ============================================================
df = pd.read_excel("vol1.xlsx")
df = df.drop(df.index[0])

df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
df = df[2800:]
print(df.columns)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# Ensure numeric
df_raw_spx = df["Returns30"].copy() 
df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# ============================================================
# 2) Compute 30-day log returns (as in paper Eq.3)
# ============================================================
df["Returns30"] = np.log(df["Returns30"]).diff(20)


# y = df["Returns30"]
# y.index = df.index[:len(y)]  # si besoin pour aligner
# y.to_csv("true_returns_raw.csv")
# y = (df["Returns30"].shift(-15) > 0).astype(int)
# y.index = df.index[:len(y)]  # si besoin pour aligner
# y.to_csv("true_returns_sign.csv")

###little test 

# df = df.dropna()
# X = df.drop(columns=["Returns30"])
# X = X.iloc[:-1]
# T = len(X)
# train_window = 2128
# n_forecasts = T - train_window
# true_returns = df["Returns30"].shift(-1).iloc[train_window:train_window + n_forecasts].values
# pd.Series(true_returns).to_csv("true_returns.csv", index=df.index[train_window:train_window + n_forecasts])
# print("✔️ Exported true_returns.csv")


# ============================================================
# 3) FEATURE SELECTION FROM PAPER (LASSO RESULT)
# Remove VIX9D, VIX3M, VIX6M
# ============================================================
features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "VIX9D ", "VIX3M ", "VIX6M "]
for col in features_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# ============================================================
# 4) Target: sign of next 30d return (classification)
# ============================================================
y = (df["Returns30"].shift(-20) > 0).astype(int)
X = df.drop(columns=["Returns30"])

print(y)


X = X.iloc[:-20]
y = y.iloc[:-20]


# # ============================================================
# # 5) WALK-FORWARD VALIDATION (paper: window = 2128, test_size = 883)
# # ============================================================
# T = len(X)
# train_window = 2128
# n_forecasts = T - train_window

# y_true = []
# y_pred = []
# y_proba = []

# test_dates = []
# # train = [t , t+2127]

# X_train = X.iloc[0 : train_window]
# y_train = y.iloc[0 : train_window]

# # scaling only on training
# scaler = StandardScaler().fit(X_train)
# Xtr = scaler.transform(X_train)


#     # Random Forest with params from the paper
# model = RandomForestClassifier(
#     n_estimators=500,
#     random_state=42,
#     max_depth=None,
#     max_features=int(np.sqrt(X.shape[1])),  # mtry = sqrt(p)
#     class_weight="balanced",  # to adjust for class imbalance
#     n_jobs=-1  
#     )

# model.fit(Xtr, y_train)



# for t in range(n_forecasts):

#     # test = point t+2128
#     X_test = X.iloc[[t + train_window]]
#     Xte = scaler.transform(X_test)
#     y_test = y.iloc[t + train_window]

#     test_dates.append(X_test.index[0])

#     proba = model.predict_proba(Xte)[0, 1]
#     pred = int(proba > 0.55)

#     y_true.append(y_test)
#     y_pred.append(pred)
#     y_proba.append(proba)

# # ============================================================
# # 6) METRICS EXACTLY LIKE THE PAPER
# # ============================================================
# acc = accuracy_score(y_true, y_pred)
# auc = roc_auc_score(y_true, y_proba)
# f1  = f1_score(y_true, y_pred)

# print(f"Accuracy : {acc:.4f}")
# print(f"AUC      : {auc:.4f}")
# print(f"F1       : {f1:.4f}")
# print(f"Number of OOS forecasts = {len(y_true)}")










from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 5) WALK-FORWARD VALIDATION (paper: window = 2128, test_size = 883)
# ============================================================
T = len(X)
train_window = 2128
n_forecasts = T - train_window

y_true = []
y_pred = []
y_proba = []

test_dates = []

for t in range(n_forecasts):

    # train = [t , t+2127]
    X_train = X.iloc[t : t + train_window-5]
    y_train = y.iloc[t : t + train_window-5]

    # test = point t+2128
    X_test = X.iloc[[t + train_window]]
    y_test = y.iloc[t + train_window]


    test_dates.append(X_test.index[0])

    # scaling only on training
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    # # Random Forest with params from the paper
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        class_weight="balanced",  # to adjust for class imbalance
        max_features=5,  # mtry = sqrt(p)
        n_jobs=-1, 
    )
    from xgboost import XGBClassifier
    # model = XGBClassifier(
    #     n_estimators=500,
    #     learning_rate=0.03,
    #     subsample=0.7,
    #     colsample_bytree=0.8,
    #     reg_alpha=0.1,
    #     reg_lambda=1.0,
    #     eval_metric='logloss',
    #     njobs=-1,
    #     random_state=42)

    # model = BaggingClassifier(
    #     estimator=DecisionTreeClassifier(max_depth=None),
    #     n_estimators=500,
    #     bootstrap=True,
    #     n_jobs=-1,
    #     random_state=42
    # )
    from sklearn.linear_model import LogisticRegressionCV
    # model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=2000)
    model.fit(Xtr, y_train)

    proba = model.predict_proba(Xte)[0, 1]
    pred = int(proba > 0.55)
    
    y_true.append(y_test)
    y_pred.append(pred)
    y_proba.append(proba)

# ============================================================
# 6) METRICS EXACTLY LIKE THE PAPER
# ============================================================
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
f1  = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:", cm)
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Accuracy : {acc:.4f}")
print(f"AUC      : {auc:.4f}")
print(f"F1       : {f1:.4f}")
print(f"Number of OOS forecasts = {len(y_true)}")




from sklearn.metrics import roc_curve





import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -------------------------------
# Reconstituer les index temporels
# -------------------------------
forecast_dates = X.index[train_window:]   # dates des prédictions OOS
pred_series = pd.Series(y_pred, index=forecast_dates)
proba_series = pd.Series(y_proba, index=forecast_dates)

# -------------------------------
# Charger le SPX pour le plot
# (le tien s'appelle sans doute différemment)
# -------------------------------
spx = df_raw_spx.copy()   # <-- mets ici ta série de SPX
spx = spx.loc[forecast_dates.min():forecast_dates.max()]

# -------------------------------
# Fonction de couleur
# -------------------------------
def color_from_proba(p, base=0.55):
    """
    base = seuil de décision.
    En-dessous de base -> rouge
    Au-dessus ou égal à base -> vert
    Intensité = distance normalisée à base.
    """
    # distance relative à la base, normalisée dans [0,1]
    if p >= base:
        # distance vers 1
        strength = (p - base) / (1.0 - base) if base < 1.0 else 0.0
    else:
        # distance vers 0
        strength = (base - p) / base if base > 0.0 else 0.0

    strength = max(0.0, min(1.0, strength))  # clamp

    if p >= base:
        # vert avec alpha = strength
        return (0, 0.8, 0, strength)
    else:
        # rouge avec alpha = strength
        return (1, 0, 0, strength)

# -------------------------------
# Plot SPX + fond coloré
# -------------------------------
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(spx.index, spx.values, color="black", linewidth=1.5, label="SPX")

dates = pred_series.index
preds = pred_series.values
probas = proba_series.values

for i in range(len(dates)-1):
    d0, d1 = dates[i], dates[i+1]
    p = probas[i]
    col = color_from_proba(p, 0.55)
    ax.axvspan(d0, d1, color=col, lw=0)

# Légendes
green_patch = mpatches.Patch(color=(0,0.8,0,0.6), label='Prediction Up (confidence)')
red_patch   = mpatches.Patch(color=(1,0,0,0.6), label='Prediction Down (confidence)')
ax.legend(handles=[green_patch, red_patch])

ax.set_title("SPX avec fond coloré par prédiction mensuelle & intensité = certitude")
ax.set_ylabel("SPX Level")
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()



# ============================================================
# 7) EXPORT EXCEL WITH true values + predictions + probabilities
# ============================================================

# rebuild dataframe of predictions
pred_df = pd.DataFrame({
    "date_observation": test_dates,              # exact OOS date
    "predicted_future_date": pd.to_datetime(test_dates) + pd.Timedelta(days=20),
    "proba_up": y_proba,
    "predicted_class": y_pred,
    "true_class": y_true
})
#
# get true 30-day returns to also export raw performance
true_returns = df["Returns30"].shift(-20).iloc[train_window:train_window + n_forecasts].values
# true_returns.to_csv("true_returns.csv")
pred_df["true_30d_log_return"] = true_returns

# reorder cleanly
pred_df = pred_df[[
    "date_observation",
    "predicted_future_date",
    "proba_up",
    "predicted_class",
    "true_class",
    "true_30d_log_return"
]]

# export Excel
pred_df.to_csv("predictions_vs_truth_real_try.csv", index=False)
print("✔️ Exported predictions_vs_truth.csv")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # --- Load SPX ---
# vol = pd.read_excel("vol.xlsx").iloc[1:]
# vol = vol.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# vol["date"] = pd.to_datetime(vol["date"])
# vol = vol.set_index("date")
# spx = pd.to_numeric(vol["Returns30"], errors="coerce").dropna()

# # --- Load predictions ---
# pred = pd.read_excel("predictions_vs_truth.xlsx")
# pred["date_observation"] = pd.to_datetime(pred["date_observation"])
# pred = pred.set_index("date_observation").sort_index()

# # Synchronize with SPX dates
# pred = pred.loc[pred.index.intersection(spx.index)]

# # --- Color function ---
# def color_from_proba(p, cls):
#     conf = abs(p - 0.5) * 2
#     conf = min(max(conf, 0), 1)
#     return (0, 0.8, 0, conf) if cls == 1 else (1, 0, 0, conf)

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(15,7))
# ax.plot(spx.index, spx.values, color="black", lw=1.5)

# dates  = pred.index
# preds  = pred["predicted_class"].values
# probas = pred["proba_up"].values

# # Color each date of OBSERVATION, not future date
# for i in range(len(pred) - 1):
#     d0 = dates[i]          # observation date
#     d1 = dates[i+1]        # next observation date
    
#     ax.axvspan(d0, d1, color=color_from_proba(probas[i], preds[i]), lw=0)

# ax.legend(handles=[
#     mpatches.Patch(color=(0,0.8,0,0.6), label="Predicted UP in 30 days"),
#     mpatches.Patch(color=(1,0,0,0.6), label="Predicted DOWN in 30 days")
# ])

# ax.set_title("Color = prediction for +30 days (applied at observation date)")
# ax.grid(True, linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()








# ############### FIN ################
# #################################################

# import pandas as pd
# import numpy as np
# pred = pd.read_csv("predictions_vs_truth.csv")
# pred["predicted_class"] = pred["predicted_class"].shift(-30)
# pred.to_csv("predictions_vs_truth.csv")



# import pandas as pd
# import numpy as np

# # -----------------------------
# # Load the two files
# # -----------------------------
# pred = pd.read_excel("predictions_30days.xlsx")
# vol  = pd.read_excel("vol.xlsx")

# # -----------------------------
# # Clean vol (SPX levels)
# # -----------------------------
# vol = vol.drop(vol.index[0])
# vol = vol.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# vol["date"] = pd.to_datetime(vol["date"])
# vol = vol.set_index("date").sort_index()

# spx = pd.to_numeric(vol["Returns30"], errors="coerce").dropna()

# # -----------------------------
# # Compute TRUE future returns
# # log return over 30 days:
# # r(t) = log(spx_t / spx_{t-30})
# # -----------------------------
# true_ret = np.log(spx) - np.log(spx.shift(30))
# true_ret = true_ret.dropna()

# # Turn into UP/DOWN
# true_sign = (true_ret > 0).astype(int)

# # -----------------------------
# # Clean predictions
# # -----------------------------
# pred["date_observation"] = pd.to_datetime(pred["date_observation"])
# pred["predicted_future_date"] = pd.to_datetime(pred["predicted_future_date"])

# pred = pred.set_index("predicted_future_date").sort_index()

# # -----------------------------
# # Align prediction with true return
# # Key = predicted_future_date
# # -----------------------------
# df_final = pd.DataFrame(index=pred.index)

# df_final["prediction"] = pred["prediction_up(1)/down(0)"].shift(60)
# df_final["proba_up"]   = pred["proba_up"].shift(60)

# # Align the real SPX return to the same date
# df_final["true_return_30d"] = true_ret.reindex(df_final.index)
# df_final["true_sign"]       = true_sign.reindex(df_final.index)

# # Remove rows where truth is not available (e.g., last 30 days)
# df_final = df_final.dropna()

# # -----------------------------
# # Export clean corrected file
# # -----------------------------
# df_final.to_excel("predictions_vs_truth_clean.xlsx")

# print("✔ DONE — File saved as predictions_vs_truth_clean.xlsx")
# print(df_final.head())




# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# # # ============================================================
# # # 1) LOAD & CLEAN DATA
# # # ============================================================
# # df = pd.read_excel("vol.xlsx")
# # df = df.drop(df.index[0])  # remove header junk row
# # df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})

# # df["date"] = pd.to_datetime(df["date"])
# # df = df.set_index("date").sort_index()

# # df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # # 30-day log return
# # df["Returns30"] = np.log(df["Returns30"]).diff(30)
# # df = df.dropna()

# # # Remove unused volatility features (paper)
# # for col in ["VIX9D", "VIX3M", "VIX6M"]:
# #     if col in df.columns:
# #         df = df.drop(columns=[col])

# # # Target = direction of next 30 days
# # y = (df["Returns30"].shift(-1) > 0).astype(int)
# # X = df.drop(columns=["Returns30"])

# # # Remove last NaN row
# # X = X.iloc[:-1]
# # y = y.iloc[:-1]

# # # ============================================================
# # # 2) TRAIN ON LAST 2000 POINTS
# # # ============================================================
# # WINDOW = 2000
# # T = len(X)
# # N_LAST = 35
# # X_train = X.iloc[T - WINDOW  - N_LAST: T - N_LAST]
# # y_train = y.iloc[T - WINDOW - N_LAST : T - N_LAST]

# # # Scaling
# # scaler = StandardScaler().fit(X_train)
# # Xtr = scaler.transform(X_train)

# # # Model from paper
# # model = RandomForestClassifier(
# #     n_estimators=500,
# #     random_state=42,
# #     max_depth=None,
# #     max_features=int(np.sqrt(X.shape[1])),
# #     n_jobs=-1
# # )
# # model.fit(Xtr, y_train)

# # # ============================================================
# # # 3) PREDICT FOR THE LAST 20 POINTS
# # # ============================================================

# # X_last = X.iloc[T - N_LAST : T]

# # # scale using SAME scaler
# # X_last_scaled = scaler.transform(X_last)

# # probas = model.predict_proba(X_last_scaled)[:, 1]
# # preds = (probas > 0.5).astype(int)

# # # Format output
# # results = pd.DataFrame({
# #     "date": X_last.index,
# #     "prediction": ["UP" if p==1 else "DOWN" for p in preds],
# #     "probability": probas
# # })

# # print(results)




###### BAGGING TRY ######
##########################



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# # ============================================================
# # 1) LOAD DATA
# # ============================================================
# df = pd.read_excel("vol.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})

# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()

# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # 2) Compute 30-day log returns
# # ============================================================
# df["Returns30"] = np.log(df["Returns30"]).diff(30)

# # drop VIX9D, VIX3M, VIX6M
# # for col in ["VIX9D", "VIX3M", "VIX6M"]:
# #     if col in df.columns:
# #         df = df.drop(columns=[col])

# # ============================================================
# # 3) TARGET = direction of future 30-day return
# # ============================================================
# y = (df["Returns30"].shift(-30) > 0).astype(int)
# X = df.drop(columns=["Returns30"])

# X = X.iloc[:-30]
# y = y.iloc[:-30]

# # ============================================================
# # 4) WALK-FORWARD VALIDATION
# # ============================================================
# T = len(X)
# train_window = 2658   # same as your previous run
# n_forecasts = T - train_window

# y_true = []
# y_pred = []
# y_proba = []
# test_dates = []

# for t in range(n_forecasts):

#     X_train = X.iloc[t : t + train_window - 30]
#     y_train = y.iloc[t : t + train_window - 30]

#     X_test = X.iloc[[t + train_window]]
#     y_test = y.iloc[t + train_window]

#     test_dates.append(X_test.index[0])

#     scaler = StandardScaler().fit(X_train)
#     Xtr = scaler.transform(X_train)
#     Xte = scaler.transform(X_test)

#     # ============================================================
#     # BAGGING MODEL EXACTLY LIKE THE PAPER
#     # (500 trees, bootstrap)
#     # ============================================================
#     model = BaggingClassifier(
#         estimator=DecisionTreeClassifier(max_depth=None),
#         n_estimators=500,
#         bootstrap=True,
#         n_jobs=-1,
#         random_state=42
#     )

#     model.fit(Xtr, y_train)

#     proba = model.predict_proba(Xte)[0, 1]
#     pred = int(proba > 0.55)

#     y_true.append(y_test)
#     y_pred.append(pred)
#     y_proba.append(proba)

# # ============================================================
# # 5) METRICS
# # ============================================================
# acc = accuracy_score(y_true, y_pred)
# auc = roc_auc_score(y_true, y_proba)
# f1  = f1_score(y_true, y_pred)

# print(f"Accuracy : {acc:.4f}")
# print(f"AUC      : {auc:.4f}")
# print(f"F1       : {f1:.4f}")
# print(f"OOS forecasts = {len(y_true)}")

# # ============================================================
# # 6) EXPORT EXCEL (unchanged)
# # ============================================================
# pred_df = pd.DataFrame({
#     "date_observation": test_dates,
#     "predicted_future_date": pd.to_datetime(test_dates) + pd.Timedelta(days=30),
#     "proba_up": y_proba,
#     "predicted_class": y_pred,
#     "true_class": y_true
# })

# true_returns = df["Returns30"].shift(-30).iloc[train_window:train_window + n_forecasts].values
# pred_df["true_30d_log_return"] = true_returns

# pred_df.to_csv("predictions_vs_truth_BAGGING.csv", index=False)
# print("✔ Exported predictions_vs_truth_BAGGING.csv")