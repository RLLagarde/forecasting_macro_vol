# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LassoCV

# # Charger les données à partir du fichier Excel
# data = pd.read_excel("vol1.xlsx")
# data = data.drop(data.index[0])
# data = data.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# data["date"] = pd.to_datetime(data["date"])
# data = data.set_index("date").sort_index()
# data = data[2800:]
# data["Returns30"] = pd.to_numeric(data["Returns30"], errors="coerce")       
# # Calculer les rendements sur 30 jours (log returns)
# data['Returns30'] = np.log(data['Returns30'].shift(-30) / data['Returns30'])

# # Supprimer les lignes avec des valeurs manquantes
# data = data.dropna()

# # Séparer les variables explicatives (X) et la cible (y)
# X = data.drop(columns=['Returns30'])
# y_class = (data['Returns30'] > 0).astype(int)  # Classification : 1 si positif, 0 sinon
# y_reg = data['Returns30']  # Régression : valeur continue

# # Standardiser les variables
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# # Appliquer Lasso pour la sélection des variables
# lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y_reg)

# # Supprimer les variables avec un coefficient nul
# selected_features = X.columns[lasso.coef_ != 0]
# X_reduced = X_scaled[selected_features]

# # Définir la taille de la fenêtre d'entraînement et de test
# train_size = int(0.7 * len(X_reduced))
# test_size = len(X_reduced) - train_size

# # Initialiser les listes pour stocker les performances
# accuracies_class = []
# aucs_class = []
# f1s_class = []

# accuracies_reg = []
# aucs_reg = []
# f1s_reg = []

# # Walk-forward validation avec ajustement pour éviter le forward-looking bias
# for i in range(train_size + 30, len(X_reduced)):
#     # Définir les ensembles d'entraînement et de test en évitant le forward-looking bias
#     X_train = X_reduced.iloc[:i-30]  # Utiliser i-30 pour éviter le forward-looking bias
#     y_train_class = y_class.iloc[:i-30]
#     y_train_reg = y_reg.iloc[:i-30]

#     X_test = X_reduced.iloc[i:i+1]
#     y_test_class = y_class.iloc[i:i+1]
#     y_test_reg = y_reg.iloc[i:i+1]

#     # Entraîner le modèle Random Forest pour la classification
#     rf_class = RandomForestClassifier(
#         n_estimators=300,
#         max_features=4,
#         random_state=42
#     )
#     rf_class.fit(X_train, y_train_class)

#     # Prédictions pour la classification
#     y_pred_class = rf_class.predict(X_test)
#     y_pred_proba_class = rf_class.predict_proba(X_test)[:, 1]  # Probabilités pour l'AUC

#     # Calculer les métriques pour la classification
#     accuracy_class = accuracy_score(y_test_class, y_pred_class)
#     auc_class = roc_auc_score(y_test_class, y_pred_proba_class)
#     f1_class = f1_score(y_test_class, y_pred_class)

#     accuracies_class.append(accuracy_class)
#     aucs_class.append(auc_class)
#     f1s_class.append(f1_class)

#     # Entraîner le modèle Random Forest pour la régression
#     rf_reg = RandomForestRegressor(
#         n_estimators=200,
#         max_features='sqrt',
#         random_state=42
#     )
#     rf_reg.fit(X_train, y_train_reg)

#     # Prédictions pour la régression
#     y_pred_reg = rf_reg.predict(X_test)
#     y_pred_class_reg = (y_pred_reg > 0).astype(int)  # Transformer en binaire pour les métriques

#     # Calculer les métriques pour la régression transformée en classification
#     accuracy_reg = accuracy_score(y_test_class, y_pred_class_reg)
#     auc_reg = roc_auc_score(y_test_class, y_pred_reg)  # AUC pour les valeurs continues
#     f1_reg = f1_score(y_test_class, y_pred_class_reg)

#     accuracies_reg.append(accuracy_reg)
#     aucs_reg.append(auc_reg)
#     f1s_reg.append(f1_reg)

# # Calculer la moyenne des performances
# mean_accuracy_class = np.mean(accuracies_class)
# mean_auc_class = np.mean(aucs_class)
# mean_f1_class = np.mean(f1s_class)

# mean_accuracy_reg = np.mean(accuracies_reg)
# mean_auc_reg = np.mean(aucs_reg)
# mean_f1_reg = np.mean(f1s_reg)

# print("\nRésultats moyens pour Random Forest (Classification) :")
# print(f"Accuracy moyenne: {mean_accuracy_class:.4f}")
# print(f"AUC moyenne: {mean_auc_class:.4f}")
# print(f"F1-score moyen: {mean_f1_class:.4f}")

# print("\nRésultats moyens pour Random Forest (Régression transformée en Classification) :")
# print(f"Accuracy moyenne: {mean_accuracy_reg:.4f}")
# print(f"AUC moyenne: {mean_auc_reg:.4f}")
# print(f"F1-score moyen: {mean_f1_reg:.4f}")





############ TEST SIMPLE ############


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.linear_model import LassoCV

# # Charger les données à partir du fichier Excel
# data = pd.read_excel("vol1.xlsx")
# data = data.drop(data.index[0])
# data = data.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# data["date"] = pd.to_datetime(data["date"])
# data = data.set_index("date").sort_index()
# data = data[3000:]
# data["Returns30"] = pd.to_numeric(data["Returns30"], errors="coerce")       
# # Calculer les rendements sur 30 jours (log returns)
# data['Returns30'] = np.log(data['Returns30'].shift(-15) / data['Returns30'])

# # Supprimer les lignes avec des valeurs manquantes
# data = data.dropna()

# # Séparer les variables explicatives (X) et la cible (y)
# X = data.drop(columns=['Returns30', "VIX9D ", "VIX3M ", "VIX6M "])
# y_class = (data['Returns30'] > 0.04).astype(int)  # Classification : 1 si positif, 0 sinon
# y_reg = data['Returns30']*100  # Régression : valeur continue

# # Standardiser les variables
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
# # print("moyenne", X_scaled.mean())
# from sklearn.linear_model import LogisticRegressionCV
# # Appliquer Lasso pour la sélection des variables
# lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=30, random_state=42, max_iter=2000).fit(X_scaled, y_class)
# # Extraire les coefficients du Lasso avec les noms des variables
# lasso_coefs = pd.Series(lasso.coef_[0], index=X.columns)

# print(lasso_coefs)


###### Correlation #####

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

# # Ensure numeric
# df_raw_spx = df["Returns30"].copy() 
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # 2) Compute 30-day log returns (as in paper Eq.3)
# # ============================================================
# df["Returns30"] = np.log(df["Returns30"]).diff(20)

import numpy as np
import pandas as pd

# suppose df contient colonnes: 'SPX' (niveau ou prix) et 'VIX' (niveau)
# si tu n'as que Returns30, prends une série de prix SPX à part.
spx = pd.to_numeric(df["Returns30"], errors="coerce")
vix = pd.to_numeric(df["VIX "], errors="coerce")
panel = pd.concat([spx.rename("spx"), vix.rename("vix")], axis=1).dropna()

# returns daily
panel["r_spx"] = np.log(panel["spx"]).diff()
panel["dvix"]  = np.log(panel["vix"]).diff()

# spike definition: dvix in top 5%
thr = panel["dvix"].quantile(0.85)
panel["spike"] = (panel["dvix"] >= thr).astype(int)

# future cumulative return over next H days
def fwd_cumret(x, H):
    return x.shift(-1).rolling(H).sum().shift(-(H-1))

H = 2
panel[f"fwd_{H}d"] = fwd_cumret(panel["r_spx"], H)

# compare distributions
spike_ret = panel.loc[panel["spike"]==1, f"fwd_{H}d"].dropna()
normal_ret = panel.loc[panel["spike"]==0, f"fwd_{H}d"].dropna()

print("Mean fwd return after spike:", spike_ret.mean())
print("Mean fwd return no spike:", normal_ret.mean())
print("P(fwd return < 0) after spike:", (spike_ret < 0).mean())
print("P(fwd return < 0) no spike:", (normal_ret < 0).mean())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# panel doit contenir: r_spx, dvix, spike, fwd_5d (ou fwd_{H}d)
# Si tu as repris mon code: panel["spike"] et panel["fwd_5d"] existent déjà.

H = 2
fwd_col = f"fwd_{H}d"
panel2 = panel.dropna(subset=["spike", fwd_col]).copy()

spike_ret  = panel2.loc[panel2["spike"]==1, fwd_col].values
normal_ret = panel2.loc[panel2["spike"]==0, fwd_col].values

# =========================
# (1) Histogrammes comparés
# =========================
plt.figure(figsize=(10,4))
bins = 60
plt.hist(normal_ret, bins=bins, density=True, alpha=0.5, label="No spike")
plt.hist(spike_ret,  bins=bins, density=True, alpha=0.5, label="Spike")
plt.axvline(0, linewidth=1)
plt.title(f"Distribution des rendements cumulés sur {H} jours (log) : spike vs no spike")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# (2) ECDF comparées
# =========================
def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

x0, y0 = ecdf(normal_ret)
x1, y1 = ecdf(spike_ret)

plt.figure(figsize=(10,4))
plt.plot(x0, y0, label="No spike")
plt.plot(x1, y1, label="Spike")
plt.axvline(0, linewidth=1)
plt.title(f"ECDF des rendements cumulés sur {H} jours : spike vs no spike")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# (3) Event study autour des spikes
# =========================
# On trace le chemin moyen des rendements cumulés *autour* d'un spike
# (par ex. de t-10 à t+10 en cumulant les rendements quotidiens)
k = 10  # fenêtre avant/après

p = panel.dropna(subset=["r_spx", "spike"]).copy()
spike_dates = p.index[p["spike"]==1]

# construit une matrice (n_events x (2k+1)) des cumuls centrés sur l'événement
paths = []
for d in spike_dates:
    # fenêtre [d-k, d+k]
    window = p.loc[:d].tail(k+1).index.union(p.loc[d:].head(k+1).index)
    # méthode robuste: on récupère la tranche exacte par position
    try:
        i = p.index.get_loc(d)
        if isinstance(i, slice):  # cas rare index dupliqué
            continue
        if i-k < 0 or i+k >= len(p):
            continue
        seg = p["r_spx"].iloc[i-k:i+k+1].values  # rendements quotidiens
        cum = np.cumsum(seg)  # cumul log-return
        paths.append(cum)
    except KeyError:
        continue

paths = np.array(paths)
if paths.size == 0:
    print("Pas assez d'événements spikes pour faire un event study sur cette fenêtre.")
else:
    mean_path = paths.mean(axis=0)
    q25 = np.quantile(paths, 0.25, axis=0)
    q75 = np.quantile(paths, 0.75, axis=0)

    x = np.arange(-k, k+1)

    plt.figure(figsize=(10,4))
    plt.plot(x, mean_path, label="Moyenne")
    plt.fill_between(x, q25, q75, alpha=0.2, label="IQR (25%-75%)")
    plt.axvline(0, linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.title("Event study SPX (log cumul) autour d'un spike VIX (t=0)")
    plt.xlabel("Jours autour du spike")
    plt.ylabel("Cumul des log-returns SPX")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

# =========================
# (4) Bonus simple: barres de P(fwd<0)
# =========================
p_neg_spike  = (spike_ret < 0).mean()
p_neg_normal = (normal_ret < 0).mean()

plt.figure(figsize=(6,4))
plt.bar(["Spike", "No spike"], [p_neg_spike, p_neg_normal])
plt.ylim(0,1)
plt.title(f"P({H}j cumul < 0) : spike vs no spike")
plt.grid(True, axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()