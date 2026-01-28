
# # ============================
# # 1) Imports & utils
# # ============================
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # ----- utilitaires -----
# def pick_first_present(df: pd.DataFrame, candidates):
#     """Renvoie le 1er nom de colonne présent dans df parmi candidates, sinon lève une erreur claire."""
#     for c in candidates:
#         if c in df.columns:
#             return c
#     raise ValueError(f"Aucune des colonnes candidates {candidates} n'est présente. Colonnes dispo: {list(df.columns)[:12]} ...")

# def clean_fred_like_csv(path: str, skip_top: int = 2) -> pd.DataFrame:
#     """
#     Charge un CSV style FRED/QD_FRED :
#     - enlève 'skip_top' premières lignes (ex: 'Transform:' ...),
#     - prend la 1ère colonne comme dates, convertit en datetime,
#     - met ces dates en index fin-de-mois,
#     - force toutes les autres colonnes en float (coerce),
#     - renvoie un DataFrame mensuel propre.
#     """
#     raw = pd.read_csv(path)
#     df  = raw.iloc[skip_top:].copy()

#     date_col = df.columns[0]
#     dates = pd.to_datetime(df[date_col], errors="coerce")
#     df.index = dates
#     df = df.drop(columns=[date_col])

#     for c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     df = df[~df.index.isna()].sort_index()
#     df.index = df.index.to_period("M").to_timestamp("M")  # fin de mois
#     return df

# def _ols_slope_last_n(y: np.ndarray) -> float:
#     """Pente OLS sur la fenêtre (x standardisé). Renvoie NaN si la fenêtre contient des NaN."""
#     y = np.asarray(y, dtype=float)
#     if not np.isfinite(y).all():
#         return np.nan
#     n = len(y)
#     x = np.arange(n, dtype=float)
#     x = (x - x.mean()) / (x.std() if x.std() != 0 else 1.0)
#     return np.cov(x, y, bias=True)[0, 1] / (np.var(x) if np.var(x) != 0 else 1.0)

# def rolling_ols_slope(s: pd.Series, window: int = 9) -> pd.Series:
#     """Pente OLS glissante sur window mois (NaN tant que pas assez d'historique)."""
#     s = s.sort_index()
#     return s.rolling(window=window, min_periods=window).apply(_ols_slope_last_n, raw=True)

# # ============================
# # 2) Charger monthly & quarterly
# # ============================
# m = clean_fred_like_csv("monthly.csv",  skip_top=2)

# # quarterly (optionnel) -> mensualisé par ffill puis joint
# try:
#     q = clean_fred_like_csv("quaterly.csv", skip_top=2)
#     q_m = q.resample("M").ffill()
#     q_m.index = q_m.index.to_period("M").to_timestamp("M")
#     Xraw = m.join(q_m, how="left")
# except Exception:
#     Xraw = m.copy()

# # Transformer les niveaux en logs avant le calcul des pentes
# for c in ["CPIAUCSL", "INDPRO"]:
#     if c in Xraw.columns:
#         Xraw[c] = np.log(Xraw[c])

# # (Option) USREC pour visuel/diagnostic si tu veux plus tard
# try:
#     r = clean_fred_like_csv("USREC.csv", skip_top=2)
#     if r.shape[1] > 0:
#         first_col = r.columns[0]
#         r = r[[first_col]].rename(columns={first_col: "USREC"})
#         Xraw = Xraw.join(r, how="left")
# except Exception:
#     pass

# # garder seulement colonnes "denses"
# dense_cols = [c for c in Xraw.columns if Xraw[c].notna().sum() >= 24]
# Xraw = Xraw[dense_cols].sort_index()

# # ============================
# # 3) Choisir proxys CPI & Growth
# # ============================
# # ============================
# # 3) Choisir proxys CPI & Growth  — version corrigée "à la Pontes et al."
# # ============================
# cpi_col = pick_first_present(Xraw, ["CPIAUCSL", "PCEPI", "CPALTT01USM657N", "CPILFESL"])
# grw_col = pick_first_present(Xraw, ["INDPRO", "IPMAN"])

# print(f"[INFO] CPI proxy: {cpi_col} | Growth proxy: {grw_col}")
# assert len(Xraw) >= 60, "Trop peu d'observations après nettoyage."

# # --- Étape 1 : calculer les TAUX (variation annuelle, Δlog sur 12 mois) ---
# infl_rate = np.log(Xraw[cpi_col]).diff(12)    # inflation YoY
# grw_rate  = np.log(Xraw[grw_col]).diff(12)    # croissance YoY

# # --- Étape 2 : pente (tendance) sur 12 mois des TAUX ---
# infl_slope12 = rolling_ols_slope(infl_rate, window=12)
# grw_slope12  = rolling_ols_slope(grw_rate,  window=12)

# # --- Étape 3 : signe des pentes pour les 4 quadrants ---
# infl_up = (infl_slope12 > 0).astype("Int64")  # inflation qui MONTE
# grw_up  = (grw_slope12  > 0).astype("Int64")  # croissance qui MONTE

# phase = pd.Series(index=infl_slope12.index, dtype="Int64")
# phase[(infl_up == 0) & (grw_up == 0)] = 0  # Recession
# phase[(infl_up == 0) & (grw_up == 1)] = 1  # Recovery
# phase[(infl_up == 1) & (grw_up == 0)] = 2  # Slowdown
# phase[(infl_up == 1) & (grw_up == 1)] = 3  # Expansion

# # --- Étape 4 : imposer les récessions officielles (USREC) si dispo ---
# if "USREC" in Xraw.columns:
#     usrec = (Xraw["USREC"] > 0.5).astype("Int64").reindex(phase.index)
#     phase.loc[usrec == 1] = 0  # force Recession

# print("\n=== Aperçu des phases non décalées ===")
# phase_names = {0:"Recession",1:"Recovery",2:"Slowdown",3:"Expansion"}
# print(phase.map(phase_names).value_counts())

# # === Étape 5 : features classiques (pentes 9m des séries brutes) ===
# features = {}
# for c in Xraw.columns:
#     features[f"{c}_slope9"] = rolling_ols_slope(Xraw[c], window=9)
# X = pd.DataFrame(features, index=Xraw.index)

# # ============================
# # 4) Indices inflation & croissance (proxy simples lissés)
# # ============================
# # Taux annuels (Δlog sur 12 mois)
# infl_rate = np.log(Xraw[cpi_col]).diff(12)
# grw_rate  = np.log(Xraw[grw_col]).diff(12)

# # Lissage des taux avant pente (6 mois)
# infl_rate_smooth = infl_rate.rolling(6, min_periods=3).mean()
# grw_rate_smooth  = grw_rate.rolling(6, min_periods=3).mean()

# # Pentes plus stables (18 mois)
# infl_slope = rolling_ols_slope(infl_rate_smooth, window=18)
# grw_slope  = rolling_ols_slope(grw_rate_smooth,  window=18)

# # Assemble X minimal : on peut ajouter d’autres pentes si tu veux
# X = pd.DataFrame({
#     f"{cpi_col}_slope18": infl_slope,
#     f"{grw_col}_slope18": grw_slope,
# }, index=Xraw.index)

# # ============================
# # 5) Labels = 4 phases (quadrants)
# # ============================
# infl_up = (infl_slope > 0).astype("Int64")
# grw_up  = (grw_slope  > 0).astype("Int64")

# phase = pd.Series(index=X.index, dtype="Int64")
# phase[(infl_up == 0) & (grw_up == 0)] = 0   # Recession
# phase[(infl_up == 0) & (grw_up == 1)] = 1   # Recovery
# phase[(infl_up == 1) & (grw_up == 0)] = 2   # Slowdown
# phase[(infl_up == 1) & (grw_up == 1)] = 3   # Expansion

# # (Option) Forcer les récessions officielles si USREC est dispo
# if "USREC" in Xraw.columns:
#     usrec = (Xraw["USREC"] > 0.5).astype("Int64").reindex(phase.index)
#     phase.loc[usrec == 1] = 0

# # Cible = forecast à 1 mois
# y = phase.shift(-1)

# # Drop le STRICT MINIMUM (ne droppe pas toutes les features !)
# df = X.copy()
# df["phase_plus1"] = y
# df = df[df["phase_plus1"].notna()].copy()

# y = df["phase_plus1"].astype(int)
# X = df.drop(columns=["phase_plus1"])

# # Option de sécurité si des NaN restent dans X (ex: bords de fenêtres)
# X = X.fillna(method="ffill").fillna(method="bfill")

# assert len(X) > 100, "Pas assez d'échantillons après features/labels."




# # ============================
# # 6) Split temporel 80/20 + scale sur TRAIN
# # ============================
# # split = int(len(X) * 0.9)
# # X_train, X_test = X.iloc[:split], X.iloc[split:]
# # y_train, y_test = y.iloc[:split], y.iloc[split:]
# from sklearn.model_selection import train_test_split



# scaler = StandardScaler().fit(X_train)
# Xtr = scaler.transform(X_train)
# Xte = scaler.transform(X_test)

# # ============================
# # 7) Modèle : Logistic multinomiale (papier)
# # ============================
# clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=2000, n_jobs=None)
# clf.fit(Xtr, y_train)

# proba = clf.predict_proba(Xte)
# pred  = proba.argmax(axis=1)

# top1 = accuracy_score(y_test, pred)
# top2 = float(np.mean([y_test.iloc[i] in np.argsort(proba[i])[::-1][:2] for i in range(len(y_test))]))

# print(f"Top-1 accuracy (test 20%) : {top1:.3f}")
# print(f"Top-2 accuracy (test 20%) : {top2:.3f}")

# y.to_csv("forecasting/test_point_phase")


# # ============================
# # 8) Sauvegarde du bundle
# # ============================
# joblib.dump(
#     {"model": clf, "scaler": scaler, "features": list(X.columns), "cpi_col": cpi_col, "growth_col": grw_col},
#     "us_cycle_model.joblib"
# )
# print("Saved -> us_cycle_model.joblib")


# # Réindexer proprement pour les plots
# pred_series = pd.Series(pred, index=X_test.index, name="pred").sort_index()
# true_series = y_test.rename("true").sort_index()

# # ============================
# # 9) PLOTS & DIAGNOSTICS (version adaptée au split aléatoire)
# # ============================
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import pandas as pd
# import numpy as np

# # --- Définition des labels et couleurs ---
# phase_names = {0: "Recession", 1: "Recovery", 2: "Slowdown", 3: "Expansion"}
# phase_colors = {0: "#ff9999", 1: "#99e699", 2: "#ffd480", 3: "#a3c6ff"}

# # --- Séries vraies et prédites sur l'échantillon test ---
# pred_series = pd.Series(pred, index=X_test.index, name="pred").sort_index()
# true_series = y_test.rename("true").sort_index()

# # --- Vérification des phases globales (toutes les dates disponibles) ---
# print("\n=== Aperçu des phases globales (toutes les dates) ===")
# phase_global = pd.Series(y.values, index=X.index, name="phase_global")
# phase_global = phase_global.map(phase_names)
# print(phase_global.value_counts())
# phase_global.to_csv("forecasting/phases_globales.csv")
# print("Phases globales sauvegardées -> forecasting/phases_globales.csv")

# # --- Scatter plot pour les prédictions (aucun chevauchement, split aléatoire) ---
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.scatter(pred_series.index, pred_series.values, s=25, c=[phase_colors[int(p)] for p in pred_series])
# ax.set_yticks([0,1,2,3])
# ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
# ax.set_title("Phases PRÉDITES (split aléatoire, prévision à 1 mois)")
# ax.grid(True, linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # --- Scatter plot pour les phases vraies ---
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.scatter(true_series.index, true_series.values, s=25, c=[phase_colors[int(t)] for t in true_series])
# ax.set_yticks([0,1,2,3])
# ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
# ax.set_title("Phases VRAIES (split aléatoire)")
# ax.grid(True, linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # --- Comparaison directe (barres) ---
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(true_series.index, true_series.values, s=20, color="black", label="True", alpha=0.6)
# ax.scatter(pred_series.index, pred_series.values, s=20, color="red", label="Predicted", alpha=0.6)
# ax.set_yticks([0,1,2,3])
# ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
# ax.legend()
# ax.set_title("Comparaison directe des phases (points : vrai vs prédit)")
# ax.grid(True, linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # --- Erreurs par classe ---
# miss_counts = []
# for k in range(4):
#     nb_true_k = (true_series == k).sum()
#     nb_miss_k = ((true_series == k) & (pred_series != k)).sum()
#     miss_counts.append(nb_miss_k)

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.bar([phase_names[k] for k in range(4)], miss_counts, color=[phase_colors[k] for k in range(4)])
# ax.set_title("Erreurs de prédiction par phase vraie")
# ax.set_ylabel("Nombre de cas mal prédits")
# ax.grid(True, axis="y", linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # --- Focus 2008 : ce qui s'est passé vs ce qui est prédit ---
# start_2008, end_2009 = pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")
# true_2008 = true_series.loc[(true_series.index >= start_2008) & (true_series.index <= end_2009)]
# pred_2008 = pred_series.loc[(pred_series.index >= start_2008) & (pred_series.index <= end_2009)]

# if len(true_2008) == 0 or len(pred_2008) == 0:
#     print("[INFO] Pas de points test sur 2008–2009 (split aléatoire).")
# else:
#     comp_2008 = pd.DataFrame({"true": true_2008, "pred": pred_2008})
#     comp_2008["true_name"] = comp_2008["true"].map(phase_names)
#     comp_2008["pred_name"] = comp_2008["pred"].map(phase_names)
#     print("\n=== 2008–2009 : Phases vraies vs prédites ===")
#     print(comp_2008[["true_name", "pred_name"]].head(24))
#     print("\nFréquences 2008–2009 (vraies):", comp_2008["true_name"].value_counts().to_dict())
#     print("Fréquences 2008–2009 (prédites):", comp_2008["pred_name"].value_counts().to_dict())

# # --- Rappel clair ---
# print("\n[NOTE] Oui : c’est bien un forecast à 1 mois (phase(t+1) = cible).")

# # ============================
# # 10) Phase courante (non-shiftée) + plot S&P coloré
# # ============================
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # --- 10.1 Phase courante = "vraie phase du mois" (NON décalée) ---
# phase_names = {0: "Recession", 1: "Recovery", 2: "Slowdown", 3: "Expansion"}
# phase_colors = {0: "#ff9999", 1: "#99e699", 2: "#ffd480", 3: "#a3c6ff"}  # rouge/vert/orange/bleu clair

# # 'phase' existe déjà dans ton code (construite à partir des slopes 9m de CPI & Growth)
# phase_current = phase.dropna().astype(int)  # même mois, pas de shift

# # Sauvegarde propre (CSV)
# phase_current_df = (
#     pd.DataFrame({"phase_id": phase_current})
#     .assign(phase_name=lambda d: d["phase_id"].map(phase_names))
# )
# phase_current_df.to_csv("phases_current_month.csv", index=True)
# print("Saved -> phases_current_month.csv  (phase du mois courant, non décalée)")

# # --- 10.2 Récupérer un niveau S&P dans les colonnes dispo ---
# # On essaie plusieurs noms usuels; adapte la liste si nécessaire
# spx_candidates = [
#     "SP500", "S&P500", "SPX", "SP500_LEVEL", "SP500INX", "S.P.500", "SP500PR",
#     "SP500 Index", "S&P 500", "INDEXSP:.INX", "GSPC", "SP500EW", "Index"
# ]
# spx_col = None
# for c in spx_candidates:
#     if c in Xraw.columns:
#         spx_col = c
#         break
# if spx_col is None:
#     raise ValueError(
#         "Aucune colonne S&P 500 trouvée. Ajoute une des colonnes suivantes dans monthly.csv : "
#         + ", ".join(spx_candidates)
#     )
# spx = Xraw[spx_col].astype(float).dropna()

# # Aligner S&P et phase courante sur le même index
# plot_df = (
#     pd.concat([spx.rename("SPX"), phase_current_df], axis=1)
#     .dropna(subset=["SPX", "phase_id"])
#     .sort_index()
# )

# if plot_df.empty:
#     raise ValueError("Pas de recouvrement entre S&P et phases. Vérifie les dates/colonnes.")

# # --- 10.3 Plot S&P (ligne) + fond coloré par phase courante ---
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(plot_df.index, plot_df["SPX"].values, lw=1.6, label=f"S&P 500 ({spx_col})", color="black")

# # Fond coloré par segment mensuel (évite le chevauchement)
# dates = plot_df.index.to_list()
# phases_here = plot_df["phase_id"].tolist()
# for i in range(len(dates) - 1):
#     d0, d1 = dates[i], dates[i + 1]
#     p = int(phases_here[i])
#     ax.axvspan(d0, d1, color=phase_colors[p], alpha=0.20, lw=0)

# # Légende des phases
# phase_patches = [mpatches.Patch(color=phase_colors[k], alpha=0.20, label=f"{k} = {phase_names[k]}") for k in range(4)]
# line_leg = ax.legend(loc="upper left")
# ax.add_artist(line_leg)
# ax.legend(handles=phase_patches, loc="upper right", title="Phase courante (non shiftée)")

# ax.set_title("S&P 500 avec fond coloré par phase macro (phase du mois courant)")
# ax.set_xlabel("Date")
# ax.set_ylabel("Niveau")
# ax.grid(True, linestyle=":", alpha=0.5)
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

# print(
#     "Note : ce graphe utilise la phase du même mois (non décalée).\n"
#     "Pour le forecasting à 1 mois, tu utilises la cible y = phase.shift(-1) dans la partie entraînement."
# )




# # Comptage des occurrences par phase
# counts = phase_current_df["phase_name"].value_counts()
# print("\n=== Répartition des phases observées (vraies, non décalées) ===")
# print(counts)
# print("CPI proxy:", cpi_col)
# print("Growth proxy:", grw_col)



# ============================
# 1) Imports & utils
# ============================
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def pick_first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Aucune des colonnes candidates {candidates} n'est présente. Colonnes dispo: {list(df.columns)[:12]} ...")

# def clean_fred_like_csv(path: str, skip_top: int = 2) -> pd.DataFrame:
#     raw = pd.read_csv(path)
#     df  = raw.iloc[skip_top:].copy()
#     date_col = df.columns[0]
#     dates = pd.to_datetime(df[date_col], errors="coerce")
#     df.index = dates
#     df = df.drop(columns=[date_col])
#     # print("test 1", df.head())
#     for c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     # print("test 2", df.head())
#     df = df[~df.index.isna()].sort_index()
#     # print("test 3", df.head())
#     df.index = df.index.to_period("M").to_timestamp("M")
#     # print("test 4", df.head())
#     return df


def clean_fred_like_csv(path: str) -> pd.DataFrame:
    # 1) essaie séparateur virgule puis point-virgule
    raw = None
    for sep in [",", ";"]:
        try:
            tmp = pd.read_csv(path, sep=sep, engine="python")
            if tmp.shape[1] >= 2:
                raw = tmp
                break
        except Exception:
            pass

    if raw is None:
        raise ValueError(f"Impossible de parser {path} (séparateur ?, fichier cassé ?)")

    # 2) garder seulement les lignes dont la 1ère colonne est une date parsable
    d = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    raw = raw.loc[d.notna()].copy()
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    if raw.shape[0] == 0:
        raise ValueError(f"{path} : aucune ligne avec date parsable -> CSV probablement cassé")

    # 3) mettre index date et convertir le reste en numérique
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date").sort_index()

    raw.columns = raw.columns.astype(str).str.strip()
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # 4) ancrer fin de mois
    raw.index = raw.index.to_period("M").to_timestamp("M")
    return raw


def _ols_slope_last_n(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        return np.nan
    n = len(y)
    x = np.arange(n, dtype=float)
    x = (x - x.mean()) / (x.std() if x.std() != 0 else 1.0)
    return np.cov(x, y, bias=True)[0, 1] / (np.var(x) if np.var(x) != 0 else 1.0)

def rolling_ols_slope(s: pd.Series, window: int = 9) -> pd.Series:
    s = s.sort_index()
    return s.rolling(window=window, min_periods=window).apply(_ols_slope_last_n, raw=True)

# ============================
# 2) Chargement des données
# ============================
# m = clean_fred_like_csv("monthly.csv", skip_top=2)
m = clean_fred_like_csv("monthly.csv")
print("/n", "test 5", m.head(5))
try:
    q = clean_fred_like_csv("quaterly.csv", skip_top=2)
    q_m = q.resample("M").ffill()
    q_m.index = q_m.index.to_period("M").to_timestamp("M")
    Xraw = m.join(q_m, how="left")
except Exception:
    Xraw = m.copy()

# (conserver EXACTEMENT ce log-transform initial pour reproduire tes sorties)
for c in ["CPIAUCSL", "INDPRO"]:
    if c in Xraw.columns:
        Xraw[c] = np.log(Xraw[c])

try:
    r = clean_fred_like_csv("USREC.csv", skip_top=2)
    if r.shape[1] > 0:
        first_col = r.columns[0]
        r = r[[first_col]].rename(columns={first_col: "USREC"})
        Xraw = Xraw.join(r, how="left")
except Exception:
    pass

dense_cols = [c for c in Xraw.columns if Xraw[c].notna().sum() >= 24]
Xraw = Xraw[dense_cols].sort_index()


# ============================
# 3) Proxys CPI & Growth
# ============================
cpi_col = pick_first_present(Xraw, ["CPIAUCSL", "PCEPI", "CPALTT01USM657N", "CPILFESL"])
grw_col = pick_first_present(Xraw, ["INDPRO", "IPMAN"])
print(f"[INFO] CPI proxy: {cpi_col} | Growth proxy: {grw_col}")
assert len(Xraw) >= 60, "Trop peu d'observations après nettoyage."



# # ============================
# # GRID SEARCH SIMPLE SUR LES TRANSFORMATIONS
# # ============================
# import itertools
# from sklearn.metrics import f1_score

# def build_X_y_from_params(Xraw, cpi_col, grw_col,
#                           diff_infl=12, diff_grw=6,
#                           span_infl=9, span_grw=6,
#                           win_infl=12, win_grw=6,
#                           add_M1=True, add_UNRATE=True):
#     # 1) Taux (Δlog sur k mois)
#     infl_rate = np.log(Xraw[cpi_col]).diff(diff_infl)
#     grw_rate  = np.log(Xraw[grw_col]).diff(diff_grw)

#     # 2) Lissage exponentiel
#     infl_rate_smooth = infl_rate.ewm(span=span_infl, adjust=False, min_periods=3).mean()
#     grw_rate_smooth  = grw_rate.ewm(span=span_grw,  adjust=False, min_periods=3).mean()

#     # 3) Pentes locales
#     infl_slope = rolling_ols_slope(infl_rate_smooth, window=win_infl)
#     grw_slope  = rolling_ols_slope(grw_rate_smooth,  window=win_grw)

#     # 4) Features de base
#     X = pd.DataFrame({
#         f"{cpi_col}_slope_w{win_infl}_d{diff_infl}_s{span_infl}": infl_slope,
#         f"{grw_col}_slope_w{win_grw}_d{diff_grw}_s{span_grw}":   grw_slope,
#     }, index=Xraw.index)

#     # 5) (Option) quelques extras économiques stables
#     if add_M1 and ("M1SL" in Xraw.columns):
#         m1 = np.log(Xraw["M1SL"]).diff(12).ewm(span=12, adjust=False, min_periods=3).mean()
#         X["M1SL_slope_w9"] = rolling_ols_slope(m1, window=9)
#     if add_UNRATE and ("UNRATE" in Xraw.columns):
#         ur = np.log(Xraw["UNRATE"]).diff(3).ewm(span=4, adjust=False, min_periods=3).mean()
#         X["UNRATE_slope_w6"] = rolling_ols_slope(ur, window=6)

#     # 6) Labels quadrants (signes des pentes)
#     infl_up = (infl_slope > 0).astype("Int64")
#     grw_up  = (grw_slope  > 0).astype("Int64")
#     phase = pd.Series(index=X.index, dtype="Int64")
#     phase[(infl_up == 0) & (grw_up == 0)] = 0  # Recession
#     phase[(infl_up == 0) & (grw_up == 1)] = 1  # Recovery
#     phase[(infl_up == 1) & (grw_up == 0)] = 2  # Slowdown
#     phase[(infl_up == 1) & (grw_up == 1)] = 3  # Expansion

#     if "USREC" in Xraw.columns:
#         usrec = (Xraw["USREC"] > 0.5).astype("Int64").reindex(phase.index)
#         phase.loc[usrec == 1] = 0

#     # 7) Cible = phase(t+1)
#     y = phase.shift(-1)

#     # 8) Nettoyage minimal
#     df = X.copy()
#     df["y"] = y
#     df = df.dropna(subset=["y"]).copy()
#     y = df["y"].astype(int)
#     X = df.drop(columns=["y"]).fillna(method="ffill").fillna(method="bfill")

#     return X, y

# def eval_one_combo(X, y, C=3.0, test_size=0.2):
#     split = int((1 - test_size) * len(X))
#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     scaler = StandardScaler().fit(X_train)
#     Xtr = scaler.transform(X_train)
#     Xte = scaler.transform(X_test)

#     clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
#                              C=C, max_iter=2000)
#     clf.fit(Xtr, y_train)
#     proba = clf.predict_proba(Xte)
#     pred  = proba.argmax(axis=1)

#     acc  = accuracy_score(y_test, pred)
#     top2 = float(np.mean([y_test.iloc[i] in np.argsort(proba[i])[::-1][:2]
#                           for i in range(len(y_test))]))
#     f1   = f1_score(y_test, pred, average="macro")
#     return acc, top2, f1, clf, scaler, (X_train.index.min(), X_test.index.max())

# # ---- Grilles de paramètres (ajuste librement) ----
# diff_infl_grid = [12, 9]          # Δlog sur 12m (classique), 9m
# diff_grw_grid  = [6, 3, 12]       # croissance plus réactive
# span_infl_grid = [9, 12]          # lissage expo CPI
# span_grw_grid  = [4, 6, 9]        # lissage expo INDPRO
# win_infl_grid  = [9, 12, 18]      # horizon de pente CPI
# win_grw_grid   = [4, 6, 9, 12]    # horizon de pente growth
# C_grid         = [1.0, 2.0, 3.0]  # pénalisation logit

# # ---- Lancement des tests ----
# records = []
# best_bundle = None
# best_acc = -1

# for dI, dG, sI, sG, wI, wG, Cval in itertools.product(
#     diff_infl_grid, diff_grw_grid, span_infl_grid, span_grw_grid,
#     win_infl_grid, win_grw_grid, C_grid
# ):
#     try:
#         Xg, yg = build_X_y_from_params(
#             Xraw, cpi_col, grw_col,
#             diff_infl=dI, diff_grw=dG,
#             span_infl=sI, span_grw=sG,
#             win_infl=wI, win_grw=wG,
#             add_M1=True, add_UNRATE=True
#         )
#         if len(Xg) < 120:  # sécurité
#             continue
#         acc, top2, f1, clf_g, scaler_g, (t0, t1) = eval_one_combo(Xg, yg, C=Cval, test_size=0.2)

#         rec = {
#             "diff_infl": dI, "diff_grw": dG,
#             "span_infl": sI, "span_grw": sG,
#             "win_infl": wI, "win_grw": wG,
#             "C": Cval, "n_obs": len(Xg),
#             "acc": acc, "top2": top2, "macro_f1": f1,
#             "train_start": str(t0.date()), "test_end": str(t1.date())
#         }
#         records.append(rec)

#         if acc > best_acc:
#             best_acc = acc
#             best_bundle = (clf_g, scaler_g, list(Xg.columns), rec)

#     except Exception as e:
#         # On logge l’échec mais on continue
#         records.append({
#             "diff_infl": dI, "diff_grw": dG,
#             "span_infl": sI, "span_grw": sG,
#             "win_infl": wI, "win_grw": wG,
#             "C": Cval, "n_obs": None,
#             "acc": None, "top2": None, "macro_f1": None,
#             "error": str(e)
#         })
#         continue

# # ---- Résultats triés ----
# res_df = pd.DataFrame(records)
# res_ok = res_df.dropna(subset=["acc"]).sort_values(["acc","macro_f1","top2"], ascending=False)
# print("\n=== Top 15 combinaisons ===")
# print(res_ok.head(15))

# # Sauvegarde CSV
# import os
# os.makedirs("forecasting", exist_ok=True)
# res_ok.to_csv("forecasting/grid_results_transformations.csv", index=False)
# print("Saved -> forecasting/grid_results_transformations.csv")

# # ---- Sauvegarder le meilleur modèle (optionnel) ----
# if best_bundle is not None:
#     best_clf, best_scaler, best_feats, best_rec = best_bundle
#     joblib.dump(
#         {"model": best_clf, "scaler": best_scaler, "features": best_feats,
#          "cpi_col": cpi_col, "growth_col": grw_col, "params": best_rec},
#         "us_cycle_model_best.joblib"
#     )
#     print("\nBest params:", best_rec)
#     print("Saved -> us_cycle_model_best.joblib")



# === Étape A (comme ton code) : pentes 12m sur TAUX YoY -> aperçu non décalé ===
# infl_rate_A = np.log(Xraw[cpi_col]).diff(12)   # NOTE: conserve le 2e log pour même résultat
# grw_rate_A  = np.log(Xraw[grw_col]).diff(12)

infl_rate_A = Xraw[cpi_col].diff(12)   
grw_rate_A  = Xraw[grw_col].diff(12)

# infl_rate_smooth = infl_rate_A.rolling(9, min_periods=3).mean()
# grw_rate_smooth  = grw_rate_A.rolling(6, min_periods=3).mean()

# Nouveau (moyenne exponentielle, poids plus forts aux données récentes)
infl_rate_smooth = infl_rate_A.ewm(span=6, adjust=False, min_periods=3).mean()
grw_rate_smooth  = grw_rate_A.ewm(span=4, adjust=False, min_periods=3).mean()

infl_slope12 = rolling_ols_slope(infl_rate_smooth, window=9)
grw_slope12  = rolling_ols_slope(grw_rate_smooth,  window=9)

# Xraw[grw_col].plot()
# plt.show()
# grw_rate_A.plot()
# plt.show()
# grw_rate_smooth.plot()
# plt.show()
# grw_slope12.plot()
# plt.show()


infl_up_A = (infl_slope12 > 0).astype("Int64")
grw_up_A  = (grw_slope12  > 0).astype("Int64")


X = pd.DataFrame({
    f"{cpi_col}_slope18": infl_slope12,
    f"{grw_col}_slope18": grw_slope12,
}, index=Xraw.index)


infl_rate_A.plot()
X.plot()
# plt.show()


extra_indics = ["M2SL"]
for ind in extra_indics:
    Xraw["M2SL_neutral"] = Xraw["M2SL"].copy()
    if ind in Xraw.columns:
        print("YES, YES, YES", ind)
        Xraw[ind] = np.log(Xraw[ind]).diff(12)   # NOTE: conserve le 2e log pour même résultat
        Xraw[ind] = Xraw[ind].rolling(6, min_periods=6).mean()
        X[f"{ind}_slope12_adj"] = rolling_ols_slope(Xraw[ind], window=3)
        # Xraw[ind]
        # rolling_ols_slope(Xraw[ind], window=3)

# --- M2 réelle (déflatée par le CPI) ---
if "M2SL" in Xraw.columns and cpi_col in Xraw.columns:
    Xraw["M2_real"] = np.log(Xraw["M2SL_neutral"]) - Xraw[cpi_col]
    Xraw["M2_real"] = Xraw["M2_real"].diff(12)
    Xraw["M2_real"] = Xraw["M2_real"].rolling(6, min_periods=6).mean()
    Xraw["M2_real"] = Xraw["M2_real"].ffill()
    X["M2_real_slope12"] = rolling_ols_slope(Xraw["M2_real"], window=9)
else:
    print("⚠️ M2SL ou CPI manquant, M2 réelle non calculée.")   

# maxi_liq = X[f"M2_real_slope12"].max()
liq = X[f"M2_real_slope12"].copy()
# liq = liq/maxi_liq

X[f"M2_real_slope12"].plot(color='purple')


plt.show()
m2_up = (X[f"M2_real_slope12"] > 0).astype("Int64")
print("liquidité M2, derniere news", liq.tail(5))

phase_A = pd.Series(index=infl_slope12.index, dtype="Int64")
phase_A[(infl_up_A == 0) & (grw_up_A == 0) & (m2_up == 0)] = 0
phase_A[(infl_up_A == 0) & (grw_up_A == 0) & (m2_up == 1)] = 4
phase_A[(infl_up_A == 0) & (grw_up_A == 1)] = 1
phase_A[(infl_up_A == 1) & (grw_up_A == 0)] = 2
phase_A[(infl_up_A == 1) & (grw_up_A == 1)] = 3

print(phase_A)

if "USREC" in Xraw.columns:
    usrec_A = (Xraw["USREC"] > 0.5).astype("Int64").reindex(phase_A.index)
    phase_A.loc[usrec_A == 1] = 0



print("\n=== Aperçu des phases non décalées ===")
phase_names = {0: "Recession", 1: "Recovery", 2: "Slowdown", 3: "Expansion"}
print(phase_A.map(phase_names).value_counts())

# # ============================
# # 4) Indices lissés (version B de ton script) pour l’entraînement
# # ============================
# infl_rate = np.log(Xraw[cpi_col]).diff(12)     # conserve le 2e log
# grw_rate  = np.log(Xraw[grw_col]).diff(12)



# infl_slope = rolling_ols_slope(infl_rate_smooth, window=18)
# grw_slope  = rolling_ols_slope(grw_rate_smooth,  window=18)



extra_indics = ["M1SL"]
for ind in extra_indics:
    if ind in Xraw.columns:
        print("YES, YES, YES", ind)
        Xraw[ind] = np.log(Xraw[ind]).diff(12)   # NOTE: conserve le 2e log pour même résultat
        Xraw[ind] = Xraw[ind].rolling(12, min_periods=3).mean()
        X[f"{ind}_slope12"] = rolling_ols_slope(Xraw[ind], window=9)


extra_indics = ["UNRATE", "PERMIT", "PAYEMS"]
# DPCERA3M086SBEA
for ind in extra_indics:
    if ind in Xraw.columns:
        print("YES, YES, YES", ind)
        Xraw[ind] = np.log(Xraw[ind]).diff(1)   # NOTE: conserve le 2e log pour même résultat
        Xraw[ind] = Xraw[ind].rolling(4, min_periods=3).mean()
        X[f"{ind}_slope12"] = rolling_ols_slope(Xraw[ind], window=6)




infl_up = (infl_slope12 > 0).astype("Int64")
grw_up  = (grw_slope12  > 0).astype("Int64")


phase = pd.Series(index=X.index, dtype="Int64")
phase[(infl_up == 0) & (grw_up == 0)] = 0
phase[(infl_up == 0) & (grw_up == 1)] = 1
phase[(infl_up == 1) & (grw_up == 0)] = 2
phase[(infl_up == 1) & (grw_up == 1)] = 3


if "USREC" in Xraw.columns:
    usrec = (Xraw["USREC"] > 0.5).astype("Int64").reindex(phase.index)
    phase.loc[usrec == 1] = 0
# phase = phase.where(usrec_A != 1, 0)          # impose NBER
# phase = phase.mask((usrec_A == 0) & (phase_A == 0), 2)  # si ton algo avait mis 0 hors NBER, requalifie par ex. en Slowdown (ou laisse tel quel selon ta logique)


# Cible = forecast à 1 mois
y = phase.shift(-1)
X_all = X.copy() # this line is for the prediction on the latest available point later
df = X.copy()
df["phase_plus1"] = y
df = df[df["phase_plus1"].notna()].copy()

y = df["phase_plus1"].astype(int)
X = df.drop(columns=["phase_plus1"]).fillna(method="ffill").fillna(method="bfill")
assert len(X) > 100, "Pas assez d'échantillons après features/labels."


# ============================
# 5) Split, scale, MLR (identique)
# ============================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7, shuffle=False
)

print(y_test)
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)


print(Xtr[-5:])


clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=2000, n_jobs=None)
clf.fit(Xtr, y_train)

proba = clf.predict_proba(Xte)
pred  = proba.argmax(axis=1)

top1 = accuracy_score(y_test, pred)
top2 = float(np.mean([y_test.iloc[i] in np.argsort(proba[i])[::-1][:2] for i in range(len(y_test))]))

print(f"Top-1 accuracy (test 20%) : {top1:.3f}")
print(f"Top-2 accuracy (test 20%) : {top2:.3f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Matrice de confusion
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1,2,3])
disp.plot(cmap='Blues')
plt.title("Confusion matrix (phases macro)")
plt.show()

# Rapport détaillé
print(classification_report(y_test, pred, digits=3))

# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', LogisticRegression(
#         multi_class='multinomial',
#         solver='lbfgs',
#         max_iter=2000
#     ))
# ])

# params = {
#     'model__C': [2, 3, 4, 5, 6, 7, 8, 9],
#     'model__penalty': ['l2'],  # lbfgs ne supporte pas l1
# }

# grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
# grid.fit(X_train, y_train)

# print("Best parameters:", grid.best_params_)
# print("Best accuracy:", grid.best_score_)



y.to_csv("forecasting/test_point_phase")

joblib.dump(
    {"model": clf, "scaler": scaler, "features": list(X.columns), "cpi_col": cpi_col, "growth_col": grw_col},
    "us_cycle_model.joblib"
)
print("Saved -> us_cycle_model.joblib")

# ============================
# 6) Diagnostics & plots (identiques)
# ============================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pred_series = pd.Series(pred, index=X_test.index, name="pred").sort_index()
true_series = y_test.rename("true").sort_index()




# === Export clair des phases : vérité (cible) vs prédiction ===
import os
os.makedirs("forecasting", exist_ok=True)

# (0) Sauvegarder la cible complète (tous mois dispo)
y.rename("phase_target_next_month").to_csv(
    "forecasting/phase_target_full.csv", index_label="date"
)

# (1) Test-only : aligner vérité et prédiction
tvp = pd.concat([true_series, pred_series], axis=1).dropna()
tvp = tvp.rename(columns={"true": "true_phase_id", "pred": "pred_phase_id"}).astype(int)

# (2) Ajouter les noms lisibles
phase_names = {0: "Recession", 1: "Recovery", 2: "Slowdown", 3: "Expansion", 4: "Recession-M2Up"}
tvp["true_phase_name"] = tvp["true_phase_id"].map(phase_names)
tvp["pred_phase_name"] = tvp["pred_phase_id"].map(phase_names)

# (3) (Optionnel) ajouter les probabilités par phase (test uniquement)
proba_df = pd.DataFrame(
    proba, index=X_test.index, columns=["proba_recession","proba_recovery","proba_slowdown","proba_expansion"]
).loc[tvp.index]
tvp = tvp.join(proba_df)

# (4) Export CSV + Excel
tvp.to_csv("forecasting/test_true_vs_pred.csv", index_label="date")

with pd.ExcelWriter("forecasting/test_true_vs_pred.xlsx") as xw:
    tvp.to_excel(xw, sheet_name="true_vs_pred")

print("\n=== Aperçu des phases globales (toutes les dates) ===")
phase_global = pd.Series(y.values, index=X.index, name="phase_global").map(phase_names)
print(phase_global.value_counts())
phase_global.to_csv("forecasting/phases_globales.csv")
print("Phases globales sauvegardées -> forecasting/phases_globales.csv")

############## plot vraie phase sans le shift pour ne pas perdre une phase: on donne la prédiction même si on n'a pas de phase comparative ###

# =====================================================
# AJOUT "FORECAST-ONLY" SUR LA DERNIÈRE DATE DISPONIBLE
# =====================================================

import numpy as np
import pandas as pd

# 0) Tu dois avoir gardé X_all AVANT le drop sur y
#    (à faire plus haut, juste avant df["phase_plus1"] = y)
#    X_all = X.copy()

# 1) Remplir sur TOUT X_all (sinon sur une seule ligne ffill/bfill ne marche pas)
X_all_filled = X_all.ffill().bfill()

# 2) Colonnes toujours NaN après ffill/bfill (souvent NA partout) -> impute avec moyenne du train
train_means = X_train.mean()
X_all_filled = X_all_filled.fillna(train_means)

# 3) Aligner EXACTEMENT les colonnes du modèle (même set + même ordre)
X_all_filled = X_all_filled.reindex(columns=X_train.columns)

# 4) Dernière date + prédiction
last_date = X_all_filled.index.max()
X_last = X_all_filled.loc[[last_date]]

# Sécurité : plus aucun NaN
if X_last.isna().any().any():
    bad = X_last.columns[X_last.isna().iloc[0]].tolist()
    raise ValueError(f"Encore des NaN dans X_last sur les colonnes: {bad}")

X_last_scaled = scaler.transform(X_last)
proba_last = clf.predict_proba(X_last_scaled)[0]
pred_last  = int(proba_last.argmax())

# 5) Rendre explicite le mois prévu (t+1)
forecast_for_month = pd.Timestamp(last_date) + pd.offsets.MonthEnd(1)

# 6) Ajouter la ligne forecast-only à ton export tvp
last_row = pd.DataFrame(
    {
        "asof_month": [last_date],                    # mois d'info (t)
        "forecast_for_month": [forecast_for_month],   # mois prévu (t+1)
        "true_phase_id": [np.nan],                    # pas observable
        "pred_phase_id": [pred_last],
        "true_phase_name": [None],
        "pred_phase_name": [phase_names[pred_last]],
        "proba_recession": [proba_last[0]],
        "proba_recovery":  [proba_last[1]],
        "proba_slowdown":  [proba_last[2]],
        "proba_expansion": [proba_last[3]],
        "is_forecast_only": [1],
    },
    index=[last_date]
)

# (optionnel mais conseillé) rendre explicite aussi pour tvp (les lignes test)
tvp2 = tvp.copy()
tvp2["asof_month"] = tvp2.index
tvp2["forecast_for_month"] = tvp2.index + pd.offsets.MonthEnd(1)
tvp2["is_forecast_only"] = 0

tvp_plus = pd.concat([tvp2, last_row], axis=0).sort_index()

tvp_plus.to_csv("forecasting/test_true_vs_pred_plus_last_forecast.csv", index_label="date")
print("Saved -> forecasting/test_true_vs_pred_plus_last_forecast.csv")
print("Forecast-only asof:", last_date, "| forecast_for:", forecast_for_month.date(),
      "| pred:", pred_last, phase_names[pred_last])
#################### Plots divers #######

# (1) PRÉDITES
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(pred_series.index, pred_series.values, s=25,
           c=[{0:"#ff9999",1:"#99e699",2:"#ffd480",3:"#a3c6ff"}[int(p)] for p in pred_series])
ax.set_yticks([0,1,2,3]); ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
ax.set_title("Phases PRÉDITES (split aléatoire, prévision à 1 mois)")
ax.grid(True, linestyle=":", alpha=0.5); plt.tight_layout(); plt.show()

# (2) VRAIES
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(true_series.index, true_series.values, s=25,
           c=[{0:"#ff9999",1:"#99e699",2:"#ffd480",3:"#a3c6ff"}[int(t)] for t in true_series])
ax.set_yticks([0,1,2,3]); ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
ax.set_title("Phases VRAIES (split aléatoire)")
ax.grid(True, linestyle=":", alpha=0.5); plt.tight_layout(); plt.show()

# (3) Comparaison directe
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(true_series.index, true_series.values, s=20, color="black", label="True", alpha=0.6)
ax.scatter(pred_series.index, pred_series.values, s=20, color="red",   label="Predicted", alpha=0.6)
ax.set_yticks([0,1,2,3]); ax.set_yticklabels([phase_names[i] for i in [0,1,2,3]])
ax.legend(); ax.set_title("Comparaison directe des phases (points : vrai vs prédit)")
ax.grid(True, linestyle=":", alpha=0.5); plt.tight_layout(); plt.show()

# (4) Erreurs par classe
miss_counts = [((true_series == k) & (pred_series != k)).sum() for k in range(4)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([phase_names[k] for k in range(4)], miss_counts,
       color=[{0:"#ff9999",1:"#99e699",2:"#ffd480",3:"#a3c6ff"}[k] for k in range(4)])
ax.set_title("Erreurs de prédiction par phase vraie")
ax.set_ylabel("Nombre de cas mal prédits")
ax.grid(True, axis="y", linestyle=":", alpha=0.5); plt.tight_layout(); plt.show()

# (5) Focus 2008–2009
start_2008, end_2009 = pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")
true_2008 = true_series.loc[(true_series.index >= start_2008) & (true_series.index <= end_2009)]
pred_2008 = pred_series.loc[(pred_series.index >= start_2008) & (pred_series.index <= end_2009)]
if len(true_2008) == 0 or len(pred_2008) == 0:
    print("[INFO] Pas de points test sur 2008–2009 (split aléatoire).")
else:
    comp_2008 = pd.DataFrame({"true": true_2008, "pred": pred_2008})
    comp_2008["true_name"] = comp_2008["true"].map(phase_names)
    comp_2008["pred_name"] = comp_2008["pred"].map(phase_names)
    print("\n=== 2008–2009 : Phases vraies vs prédites ===")
    print(comp_2008[["true_name", "pred_name"]].head(24))
    print("\nFréquences 2008–2009 (vraies):", comp_2008["true_name"].value_counts().to_dict())
    print("Fréquences 2008–2009 (prédites):", comp_2008["pred_name"].value_counts().to_dict())
print("\n[NOTE] Oui : c’est bien un forecast à 1 mois (phase(t+1) = cible).")

# ============================
# 7) Phase courante + S&P coloré (identique)
# ============================
phase_current= phase_A.dropna().astype(int)
phase_current_df = (
    pd.DataFrame({"phase_id": phase_current})
    .assign(phase_name=lambda d: d["phase_id"].map(phase_names))
)
phase_current_df.to_csv("phases_current_month.csv", index=True)
print("Saved -> phases_current_month.csv  (phase du mois courant, non décalée)")

spx_candidates = ["SP500","S&P500","SPX","SP500_LEVEL","SP500INX","S.P.500","SP500PR",
                  "SP500 Index","S&P 500","INDEXSP:.INX","GSPC","SP500EW","Index"]
spx_col = next((c for c in spx_candidates if c in Xraw.columns), None)
if spx_col is None:
    raise ValueError("Aucune colonne S&P 500 trouvée. Ajoute : " + ", ".join(spx_candidates))
spx = Xraw[spx_col].astype(float).dropna()

plot_df = pd.concat([spx.rename("SPX"), phase_current_df], axis=1).dropna(subset=["SPX","phase_id"]).sort_index()
if plot_df.empty:
    raise ValueError("Pas de recouvrement entre S&P et phases. Vérifie les dates/colonnes.")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df.index, plot_df["SPX"].values, lw=1.6, label=f"S&P 500 ({spx_col})", color="black")
phase_colors = {0:"#ff9999",1:"#99e699",2:"#ffd480",3:"#a3c6ff", 4:"#950ef0"}
dates = plot_df.index.to_list(); phases_here = plot_df["phase_id"].tolist()
for i in range(len(dates) - 1):
    ax.axvspan(dates[i], dates[i+1], color=phase_colors[int(phases_here[i])], alpha=0.20, lw=0)
patches = [mpatches.Patch(color=phase_colors[k], alpha=0.20, label=f"{k} = {phase_names[k]}") for k in range(5)]
line_leg = ax.legend(loc="upper left"); ax.add_artist(line_leg)
ax.legend(handles=patches, loc="upper right", title="Phase courante (non shiftée)")
ax.set_title("S&P 500 avec fond coloré par phase macro (phase du mois courant)")
ax.set_xlabel("Date"); ax.set_ylabel("Niveau"); ax.grid(True, linestyle=":", alpha=0.5)
fig.autofmt_xdate(); plt.tight_layout(); plt.show()

counts = phase_current_df["phase_name"].value_counts()
print("\n=== Répartition des phases observées (vraies, non décalées) ===")
print(counts)
print("CPI proxy:", cpi_col)
print("Growth proxy:", grw_col)




#### TESTING ALLOCATION STRATEGY ########

from pandas.tseries.offsets import MonthBegin
df = pd.read_excel('data.xlsx')
# Renommer les colonnes utiles (adapte si besoin)
df = df.rename(columns={'Unnamed: 1':'date','Unnamed: 2':'snp','Unnamed: 7':'oil','Unnamed: 8':'gold','Unnamed: 10':'bond'})
df = df[['date','snp','gold','bond', 'oil']].copy()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')

# Met tout au même repère temporel: fin de mois (tes séries sont mensuelles)
df['date_m']  = df['date'].dt.to_period('M').dt.to_timestamp('M')

# Merge propre sur la date de fin de mois

df = df.drop(columns=['date'])        # on garde la clé harmonisée
df = df.rename(columns={'date_m':'date'})
df = df.sort_values('date').reset_index(drop=True)


# --- 1) Base table with a clean EOM date ---
ex = df[['date','snp','gold','bond']].copy()
ex['date'] = pd.to_datetime(ex['date'], errors='coerce')
ex = ex.dropna(subset=['date']).sort_values('date')
ex['date'] = ex['date'].dt.to_period('M').dt.to_timestamp('M')  # anchor = EOM

# --- 2) Build series with proper index ---
# Prices (levels) at EOM
snp_px  = ex.set_index('date')['snp'].astype(float)
gold_px = ex.set_index('date')['gold'].astype(float)

# Bond: monthly return already (duration/convexity+carry) -> keep as returns at EOM
y  = (ex['bond'] / 100.0).astype(float)
dy = y.diff()
D, C = 10.0, 100.0
bond_ret = (-D * dy) + (0.5 * C * (dy**2)) + (y.shift(1) / 12.0)
bond_ret.index = ex['date']  # EOM

# --- 3) Shift EOM -> next BOM (01/MM+1) for execution prices/returns ---
snp_mb  = snp_px.copy();  snp_mb.index  = snp_mb.index  + MonthBegin(1)
gold_mb = gold_px.copy(); gold_mb.index = gold_mb.index + MonthBegin(1)
bond_mb = bond_ret.copy(); bond_mb.index = bond_mb.index + MonthBegin(1)

# Helper: drop duplicates, sort, and reindex on pure MonthStart grid
def to_ms(s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s.asfreq('MS', method='pad')  # pad to exact 1st-of-month dates

snp_mb  = to_ms(snp_mb)
gold_mb = to_ms(gold_mb)
bond_mb = to_ms(bond_mb)

# --- 4) Monthly returns aligned at BOM ---
r_snp  = snp_mb.pct_change().rename('snp')
r_gold = gold_mb.pct_change().rename('gold')
r_bond = bond_mb.rename('bond')  # already a return

# Final returns panel (clean MS index, no NaNs)
rets_mb = pd.concat([r_bond, r_gold, r_snp], axis=1).dropna()

rets_mb.to_csv('main_actif.csv')
from pandas.tseries.offsets import MonthBegin

# ---------- (A) RETS au 01/mm déjà OK ----------
# r_snp, r_gold = % mensuels à MS ; r_bond = rendement mensuel déjà à MS
# rets_mb = concat([r_bond, r_gold, r_snp]).dropna()  # colonnes: bond, gold, snp (minuscules)

from pandas.tseries.offsets import MonthBegin


# # =====================================================
# # ALLOCATION PONDÉRÉE PAR LES 2 RÉGIMES LES PLUS PROBABLES
# # =====================================================

# from pandas.tseries.offsets import MonthBegin
# import numpy as np
# import pandas as pd

# # --- 1) Limiter aux DATES TEST UNIQUEMENT ---
# test_idx = y_test.index  # index temporel des cibles de test (EOM)
# phase_probs = pd.DataFrame(proba, index=test_idx, columns=[0, 1, 2, 3])  # 4 phases

# # Mapping des phases vers les actifs correspondants
# phase_to_asset = {0: "bond", 1: "snp", 2: "gold", 3: "snp"}  # recession=bond, slowdown=gold, recovery/expansion=snp

# # --- 2) Pour chaque mois, récupérer les 2 régimes les plus probables ---
# alloc_weights = []
# for i, row in phase_probs.iterrows():
#     top2 = row.sort_values(ascending=False).iloc[:2]
#     top2_assets = [phase_to_asset[idx] for idx in top2.index]
#     weights = top2.values / top2.values.sum()  # normaliser à 100%
#     alloc_weights.append(dict(zip(top2_assets, weights)))

# alloc_df = pd.DataFrame(alloc_weights, index=phase_probs.index).fillna(0.0)

# # --- 3) Décaler EOM -> 01/MM+1 (exécution au début du mois suivant) ---
# alloc_df.index = alloc_df.index + MonthBegin(1)

# # --- 4) Restreindre les rendements à la fenêtre TEST SEULEMENT ---
# start_mb, end_mb = alloc_df.index.min(), alloc_df.index.max()
# rets_win = rets_mb.loc[start_mb:end_mb].copy()

# # --- 5) Réindexer les poids sur la grille mensuelle (MS) ---
# alloc_df = alloc_df.reindex(rets_win.index).ffill()
# alloc_df = alloc_df[["snp", "bond", "gold"]].fillna(0.0)

# # --- 6) Calcul des rendements de la stratégie pondérée ---
# strat_rets = (rets_win[["snp", "bond", "gold"]] * alloc_df).sum(axis=1)

# # Benchmark buy&hold S&P sur la même fenêtre
# bench_rets = rets_win["snp"]

# # --- 7) Cumul et stats ---
# strat_cum = (1 + strat_rets).cumprod()
# bench_cum = (1 + bench_rets).cumprod()

# def _cagr(cum, periods_per_year=12):
#     if len(cum) < 2: return np.nan
#     n_years = len(cum) / periods_per_year
#     return cum.iloc[-1]**(1/n_years) - 1

# def _max_dd(cum):
#     roll = cum.cummax()
#     dd = cum/roll - 1
#     return dd.min()

# print("\n=== Backtest pondéré : top-2 régimes les plus probables ===")
# print(f"Début: {str(strat_rets.index[0].date())} | Fin: {str(strat_rets.index[-1].date())}")
# print(f"Valeur finale stratégie : {strat_cum.iloc[-1]:.3f}x | S&P : {bench_cum.iloc[-1]:.3f}x")
# print(f"CAGR stratégie : {_cagr(strat_cum):.2%} | S&P : {_cagr(bench_cum):.2%}")
# print(f"Max DD stratégie : {_max_dd(strat_cum):.2%} | S&P : {_max_dd(bench_cum):.2%}")

# # === Perf annuelle (identique à ta version) ===
# ret_df = pd.DataFrame({"strat": strat_rets, "spx": bench_rets}).dropna()
# annual = (1.0 + ret_df).resample("A-DEC").prod() - 1.0
# annual.index = annual.index.year
# annual["excess"] = annual["strat"] - annual["spx"]
# annual_pct = (annual * 100).round(2).rename(columns={"strat": "Strat %", "spx": "S&P %", "excess": "Excess %"})
# print("\n=== Perfs annuelles (top-2 weighted) ===")
# print(annual_pct)
# annual_pct.to_csv("annual_returns_top2_weighted.csv")
# print("Saved -> annual_returns_top2_weighted.csv")




# # === Export des prédictions mensuelles test ===
# # On enregistre pour chaque date : les 4 probabilités, les 2 actifs retenus et les poids utilisés
# pred_export = phase_probs.copy()
# pred_export.columns = ["proba_recession", "proba_recovery", "proba_slowdown", "proba_expansion"]

# # Ajouter les poids finaux utilisés dans la stratégie pondérée
# alloc_export = alloc_df.copy()
# alloc_export.columns = [f"weight_{c}" for c in alloc_export.columns]

# # Merge proba + poids + actif dominant
# pred_main_asset = alloc_df.idxmax(axis=1).rename("main_asset")
# pred_export = pd.concat([pred_export, alloc_export, pred_main_asset], axis=1)

# # Sauvegarde CSV
# pred_export.to_csv("predictions_test_weighted.csv", index_label="date")
# print("Saved -> predictions_test_weighted.csv  (probas + allocations pondérées)")


#### FULL STRIKE ON MOST LIKELY ####



# 1) Limiter aux DATES TEST UNIQUEMENT
test_idx = y_test.index  # index temporel des cibles de test (EOM)
# Série des prédictions avec index EOM du test
phase_pred_name = pd.Series(pred, index=test_idx).map(phase_names)
phase_to_asset = {"Expansion":"snp", "Recovery":"snp", "Recession":"bond", "Slowdown":"gold"}
asset_signal_eom = phase_pred_name.map(phase_to_asset)  # EOM (ex: 2024-01-31)

# 2) Décaler EOM -> 01/MM+1 (exécution au début du mois suivant)
asset_signal_mb = asset_signal_eom.copy()
asset_signal_mb.index = asset_signal_mb.index + MonthBegin(1)  # 2024-02-01

# 3) Restreindre les rendements à la fenêtre TEST SEULEMENT
start_mb = asset_signal_mb.index.min()
end_mb   = asset_signal_mb.index.max()
rets_win = rets_mb.loc[start_mb:end_mb].copy()

# 4) Réindexer les signaux sur la grille MS de la fenêtre, SANS créer d’histo avant le 1er signal
asset_signal_mb = asset_signal_mb.reindex(rets_win.index)
# IMPORTANT : ne pas ffill AVANT le premier signal
asset_signal_mb = asset_signal_mb.ffill()

# 5) Enlever les mois avant l’apparition du premier signal valide
first_sig = asset_signal_mb.first_valid_index()
rets_win  = rets_win.loc[first_sig:]
asset_signal_mb = asset_signal_mb.loc[first_sig:]

# 6) Backtest
switch = asset_signal_mb.ne(asset_signal_mb.shift(1)).fillna(True)
onehot = (pd.get_dummies(asset_signal_mb)
          .reindex(rets_win.index)
          .reindex(columns=["snp","bond","gold"], fill_value=0))

# Sélection de l’actif chaque mois
strat_rets = (rets_win[["snp","bond","gold"]] * onehot).sum(axis=1)

# Benchmark buy&hold S&P sur la même fenêtre
bench_rets = rets_win["snp"]

# ============================
# TRANSACTION COSTS (10 bps per switch)
# ============================
TCOST = 0.001  # 10 bps

# switch = True quand l'actif change
switch = asset_signal_mb.ne(asset_signal_mb.shift(1)).fillna(False)
switch.iloc[0] = True  # on paye l'entrée en position (conservateur)

# Rendements stratégie avec coût
strat_rets_tc = strat_rets - (TCOST * switch.astype(float))

# Cumuls
strat_cum    = (1 + strat_rets).cumprod()
strat_cum_tc = (1 + strat_rets_tc).cumprod()
bench_cum    = (1 + bench_rets).cumprod()

print("\n=== Avec coûts de transaction (10 bps par switch) ===")
print(f"Switches total : {int(switch.sum())}")
print(f"Valeur finale stratégie (sans coût) : {strat_cum.iloc[-1]:.3f}x")
print(f"Valeur finale stratégie (avec coût) : {strat_cum_tc.iloc[-1]:.3f}x")
print(f"Valeur finale S&P buy&hold          : {bench_cum.iloc[-1]:.3f}x")

# Export debug
out_tc = pd.DataFrame({
    "signal_asset": asset_signal_mb,
    "switch": switch.astype(int),
    "strat_rets": strat_rets,
    "strat_rets_tc": strat_rets_tc,
    "bench_rets": bench_rets,
    "strat_cum": strat_cum,
    "strat_cum_tc": strat_cum_tc,
    "bench_cum": bench_cum
})
out_tc.to_csv("forecasting/full_strike_with_tc.csv", index_label="date")
print("Saved -> forecasting/full_strike_with_tc.csv")
# === Perf annuelle (calendrier) : stratégie vs S&P ===
# On part des rendements mensuels au 01/MM (strat_rets, bench_rets)
ret_df = pd.DataFrame({
    "strat": strat_rets,
    "spx":   bench_rets
}).dropna()

# Produit des rendements par année civile (Jan→Dec ; la dernière année peut être partielle = YTD)
annual = (1.0 + ret_df).resample("A-DEC").prod() - 1.0
annual.index = annual.index.year
annual["excess"] = annual["strat"] - annual["spx"]

# Affichage propre en %
annual_pct = (annual * 100).round(2).rename(
    columns={"strat": "Strat %", "spx": "S&P %", "excess": "Excess %"}
)
print("\n=== Perfs annuelles (calendrier) ===")
print(annual_pct)

# (Option) Export CSV
annual_pct.to_csv("annual_returns_strat_vs_spx.csv")
print("Saved -> annual_returns_strat_vs_spx.csv")





##################
##################
#ALLOCATION PROPORTIONNELLE 


# from pandas.tseries.offsets import MonthBegin
# import numpy as np
# import pandas as pd

# # --- Hypothèses d'objets déjà définis ---
# # proba: np.ndarray shape (n_test, 4) = P(phase = 0..3) sur l'échantillon TEST
# # y_test: pandas Series, index = dates EOM correspondant aux lignes de `proba`
# # rets_mb: DataFrame mensuel à index MonthStart (MS) avec colonnes ["bond","gold","snp"]

# # 1) Probabilités par phase indexées sur EOM (index de y_test)
# test_idx = y_test.index
# phase_probs = pd.DataFrame(proba, index=test_idx, columns=[0, 1, 2, 3])

# # 2) Mapping phases -> actifs (poids = probas brutes)
# #    0: Recession -> bond
# #    1: Recovery  -> snp
# #    2: Slowdown  -> gold
# #    3: Expansion -> snp
# weights_eom = pd.DataFrame(index=phase_probs.index, columns=["snp","bond","gold"]).fillna(0.0)
# weights_eom["bond"] = phase_probs[0]
# weights_eom["gold"] = phase_probs[2]
# weights_eom["snp"]  = phase_probs[1] + phase_probs[3]

# # (Optionnel) normaliser ligne par ligne (devrait déjà sommer à 1, mais on sécurise)
# row_sum = weights_eom.sum(axis=1).replace(0, np.nan)
# weights_eom = weights_eom.div(row_sum, axis=0).fillna(0.0)

# # 3) Décaler EOM -> 01/MM+1 (exécution au début du mois suivant)
# weights_ms = weights_eom.copy()
# weights_ms.index = weights_ms.index + MonthBegin(1)

# # 4) Restreindre la fenêtre aux dates où on a des rendements
# start_mb, end_mb = weights_ms.index.min(), weights_ms.index.max()
# rets_win = rets_mb.loc[start_mb:end_mb].copy()

# # 5) Réindexer sur la grille MS et ne PAS créer de poids avant le 1er signal
# weights_ms = weights_ms.reindex(rets_win.index)
# first_sig = weights_ms.dropna(how="all").index.min()
# weights_ms = weights_ms.loc[first_sig:].ffill().fillna(0.0)
# rets_win   = rets_win.loc[weights_ms.index]

# # 6) Rendements de la stratégie pondérée par proba
# #    (vérifie l'ordre des colonnes)
# weights_ms = weights_ms[["snp","bond","gold"]]
# rets_win   = rets_win[["snp","bond","gold"]]
# strat_rets = (rets_win * weights_ms).sum(axis=1)

# # Benchmark buy&hold S&P (sur la même fenêtre)
# bench_rets = rets_win["snp"]

# # 7) Stats cumulatives
# strat_cum = (1 + strat_rets).cumprod()
# bench_cum = (1 + bench_rets).cumprod()

# def _cagr(cum, periods_per_year=12):
#     if len(cum) < 2:
#         return np.nan
#     n_years = len(cum) / periods_per_year
#     return cum.iloc[-1]**(1/n_years) - 1

# def _max_dd(cum):
#     roll = cum.cummax()
#     dd = cum/roll - 1
#     return dd.min()

# print("\n=== Backtest pondéré par probabilités (phase->actif) ===")
# print(f"Début: {str(strat_rets.index[0].date())} | Fin: {str(strat_rets.index[-1].date())}")
# print(f"Valeur finale stratégie : {strat_cum.iloc[-1]:.3f}x | S&P : {bench_cum.iloc[-1]:.3f}x")
# print(f"CAGR stratégie : {_cagr(strat_cum):.2%} | S&P : {_cagr(bench_cum):.2%}")
# print(f"Max DD stratégie : {_max_dd(strat_cum):.2%} | S&P : {_max_dd(bench_cum):.2%}")

# # 8) Perf annuelle calendrier
# ret_df = pd.DataFrame({"strat": strat_rets, "spx": bench_rets}).dropna()
# annual = (1.0 + ret_df).resample("A-DEC").prod() - 1.0
# annual.index = annual.index.year
# annual["excess"] = annual["strat"] - annual["spx"]
# annual_pct = (annual * 100).round(2).rename(columns={"strat": "Strat %", "spx": "S&P %", "excess": "Excess %"})
# print("\n=== Perfs annuelles (prob-weighted) ===")
# print(annual_pct)

# # 9) Exports utiles
# import os
# os.makedirs("forecasting", exist_ok=True)
# weights_ms.to_csv("forecasting/alloc_prob_weights.csv", index_label="date")
# strat_rets.rename("strat_rets").to_csv("forecasting/strat_rets_prob_weighted.csv", index_label="date")
# annual_pct.to_csv("forecasting/annual_returns_prob_weighted.csv")
# print("Saved -> forecasting/alloc_prob_weights.csv, strat_rets_prob_weighted.csv, annual_returns_prob_weighted.csv")








# =====================================================
# ALLOCATION 50/50 SUR LES 2 RÉGIMES LES PLUS PROBABLES
# =====================================================
from pandas.tseries.offsets import MonthBegin

# Hypothèses déjà définies plus haut :
# - proba: np.ndarray shape (n_test, 4) pour les phases {0,1,2,3}
# - y_test: index temporel (EOM) correspondant à proba
# - rets_mb: DataFrame mensuel indexé en MonthStart (MS) avec colonnes ["bond","gold","snp"]

# 1) Probabilités par phase, indexées EOM (comme y_test)
phase_probs = pd.DataFrame(proba, index=y_test.index, columns=[0, 1, 2, 3])

# 2) Mapping des phases vers actifs
#    0: Recession -> bond
#    1: Recovery  -> snp
#    2: Slowdown  -> gold
#    3: Expansion -> snp
phase_to_asset = {0: "bond", 1: "snp", 2: "gold", 3: "snp"}

# 3) Pour chaque mois, prendre les 2 phases les plus probables et allouer 50/50
def _equal_top2_weights(row):
    top2_phases = row.sort_values(ascending=False).index[:2].tolist()
    # Agrège 50% par phase, en mappant phase->actif (si les 2 phases mènent au même actif, on aura 100% cet actif)
    w = {"snp": 0.0, "bond": 0.0, "gold": 0.0}
    for ph in top2_phases:
        a = phase_to_asset[ph]
        w[a] += 0.5
    return pd.Series(w)

weights50_eom = phase_probs.apply(_equal_top2_weights, axis=1)

# 4) Décaler EOM -> 01/MM+1 (exécution au début du mois suivant)
weights50_ms = weights50_eom.copy()
weights50_ms.index = weights50_ms.index + MonthBegin(1)

# 5) Restreindre la fenêtre aux dates où on a des rendements et réindexer sur la grille MS
start_mb, end_mb = weights50_ms.index.min(), weights50_ms.index.max()
rets_win = rets_mb.loc[start_mb:end_mb].copy()

weights50_ms = weights50_ms.reindex(rets_win.index)
first_sig = weights50_ms.dropna(how="all").index.min()  # ne crée pas d’histo avant 1er signal
weights50_ms = weights50_ms.loc[first_sig:].ffill().fillna(0.0)
rets_win    = rets_win.loc[weights50_ms.index]

# 6) Rendements de la stratégie 50/50 top-2
weights50_ms = weights50_ms[["snp","bond","gold"]]
rets_win     = rets_win[["snp","bond","gold"]]
strat_rets_5050 = (rets_win * weights50_ms).sum(axis=1)

# 7) Stats & exports
bench_rets = rets_win["snp"]
strat_cum_5050 = (1 + strat_rets_5050).cumprod()
bench_cum      = (1 + bench_rets).cumprod()

def _cagr(cum, periods_per_year=12):
    if len(cum) < 2:
        return np.nan
    n_years = len(cum) / periods_per_year
    return cum.iloc[-1]**(1/n_years) - 1

def _max_dd(cum):
    roll = cum.cummax()
    dd = cum/roll - 1
    return dd.min()

print("\n=== Backtest 50/50 top-2 régimes (phase->actif) ===")
print(f"Début: {str(strat_rets_5050.index[0].date())} | Fin: {str(strat_rets_5050.index[-1].date())}")
print(f"Valeur finale stratégie : {strat_cum_5050.iloc[-1]:.3f}x | S&P : {bench_cum.iloc[-1]:.3f}x")
print(f"CAGR stratégie : {_cagr(strat_cum_5050):.2%} | S&P : {_cagr(bench_cum):.2%}")
print(f"Max DD stratégie : {_max_dd(strat_cum_5050):.2%} | S&P : {_max_dd(bench_cum):.2%}")

# Perf annuelle (calendrier)
ret_df_5050 = pd.DataFrame({"strat_5050": strat_rets_5050, "spx": bench_rets}).dropna()
annual_5050 = (1.0 + ret_df_5050).resample("A-DEC").prod() - 1.0
annual_5050.index = annual_5050.index.year
annual_5050["excess_5050"] = annual_5050["strat_5050"] - annual_5050["spx"]
annual_5050_pct = (annual_5050 * 100).round(2)

# Exports
import os
os.makedirs("forecasting", exist_ok=True)
weights50_ms.to_csv("forecasting/alloc_equal_top2.csv", index_label="date")
strat_rets_5050.rename("strat_rets_5050").to_csv("forecasting/strat_rets_equal_top2.csv", index_label="date")
annual_5050_pct.to_csv("forecasting/annual_returns_equal_top2.csv")
print("Saved -> alloc_equal_top2.csv, strat_rets_equal_top2.csv, annual_returns_equal_top2.csv")


# =====================================================
# TABLEAUX DE PERF : YEAR-BY-YEAR + LAST-12-MONTHS M/M
# =====================================================

# --- (A) Year by year (calendar) ---
annual_5050_pct = annual_5050_pct.rename(
    columns={"strat_5050": "Strat 50/50 %", "spx": "S&P %", "excess_5050": "Excess %"}
)[["Strat 50/50 %", "S&P %", "Excess %"]]

print("\n=== Year-by-year (calendar) : Strat 50/50 vs S&P ===")
print(annual_5050_pct)
annual_5050_pct.to_csv("forecasting/annual_returns_equal_top2.csv")  # overwrite with clean headers

# --- (B) Month-by-month over the last 12 months ---
# source mensuelle alignée sur MonthStart (MS)
ret_df_5050 = ret_df_5050.copy()
last12 = ret_df_5050.tail(12)

month_table = pd.DataFrame({
    "Strat 50/50 %": (last12["strat_5050"] * 100).round(2),
    "S&P %":         (last12["spx"] * 100).round(2),
    "Excess %":      ((last12["strat_5050"] - last12["spx"]) * 100).round(2),
    "Strat cum x":   (1.0 + last12["strat_5050"]).cumprod().round(3),
    "S&P cum x":     (1.0 + last12["spx"]).cumprod().round(3),
})

# index lisible "YYYY-MM"
month_table.index = month_table.index.strftime("%Y-%m")

print("\n=== Last 12 months (month-by-month) : Strat 50/50 vs S&P ===")
print(month_table)

# Exports
month_table.to_csv("forecasting/monthly_last12_equal_top2.csv", index_label="month")

# Petit résumé console
print(f"\n[L12M] De {list(month_table.index)[0]} à {list(month_table.index)[-1]}")
print(f"[L12M] Perf cum Strat 50/50 : {month_table['Strat cum x'].iloc[-1]:.3f}x | "
      f"S&P : {month_table['S&P cum x'].iloc[-1]:.3f}x | "
      f"Excess total : {(month_table['Strat 50/50 %'].sum() - month_table['S&P %'].sum()):.2f} pts")






# === Last 24 months (month-by-month) : FULL-STRIKE vs S&P ===
ret_df_fs = pd.DataFrame({
    "strat": strat_rets,   # full-strike monthly returns (MS)
    "spx":   bench_rets    # S&P monthly returns (MS)
}).dropna()

last24_fs = ret_df_fs.tail(120).copy()

month_table_fs24 = pd.DataFrame({
    "Strat %":   (last24_fs["strat"] * 100).round(2),
    "S&P %":     (last24_fs["spx"] * 100).round(2),
    "Excess %":  ((last24_fs["strat"] - last24_fs["spx"]) * 100).round(2),
    "Strat cum x": (1.0 + last24_fs["strat"]).cumprod().round(3),
    "S&P cum x":   (1.0 + last24_fs["spx"]).cumprod().round(3),
})
month_table_fs24.index = month_table_fs24.index.strftime("%Y-%m")

print("\n=== Last 24 months (month-by-month) : FULL-STRIKE vs S&P ===")
print(month_table_fs24)

# Export
import os; os.makedirs("forecasting", exist_ok=True)
month_table_fs24.to_csv("forecasting/monthly_last24_full_strike.csv", index_label="month")

# Résumé console
print(f"\n[L24M FULL] De {month_table_fs24.index[0]} à {month_table_fs24.index[-1]}")
print(f"[L24M FULL] Strat cum : {month_table_fs24['Strat cum x'].iloc[-1]:.3f}x | "
      f"S&P cum : {month_table_fs24['S&P cum x'].iloc[-1]:.3f}x | "
      f"Excess total (somme mensuelle en pts) : "
      f"{(month_table_fs24['Strat %'].sum() - month_table_fs24['S&P %'].sum()):.2f}")


# === Last 24 months (month-by-month) : 50/50 top-2 vs S&P ===
ret_df_5050_all = pd.DataFrame({
    "strat_5050": strat_rets_5050,  # 50/50 monthly returns (MS)
    "spx":        bench_rets        # S&P monthly returns (MS)
}).dropna()

last24_5050 = ret_df_5050_all.tail(120).copy()

month_table_5050_24 = pd.DataFrame({
    "Strat 50/50 %": (last24_5050["strat_5050"] * 100).round(2),
    "S&P %":         (last24_5050["spx"] * 100).round(2),
    "Excess %":      ((last24_5050["strat_5050"] - last24_5050["spx"]) * 100).round(2),
    "Strat cum x":   (1.0 + last24_5050["strat_5050"]).cumprod().round(3),
    "S&P cum x":     (1.0 + last24_5050["spx"]).cumprod().round(3),
})
month_table_5050_24.index = month_table_5050_24.index.strftime("%Y-%m")

print("\n=== Last 24 months (month-by-month) : 50/50 top-2 vs S&P ===")
print(month_table_5050_24)

# Export
import os; os.makedirs("forecasting", exist_ok=True)
month_table_5050_24.to_csv("forecasting/monthly_last24_equal_top2.csv", index_label="month")

# Résumé console
print(f"\n[L24M 50/50] De {month_table_5050_24.index[0]} à {month_table_5050_24.index[-1]}")
print(f"[L24M 50/50] Strat cum : {month_table_5050_24['Strat cum x'].iloc[-1]:.3f}x | "
      f"S&P cum : {month_table_5050_24['S&P cum x'].iloc[-1]:.3f}x | "
      f"Excess total (somme mensuelle en pts) : "
      f"{(month_table_5050_24['Strat 50/50 %'].sum() - month_table_5050_24['S&P %'].sum()):.2f}")
      

##### Test perf annuelle ####



import numpy as np
import pandas as pd
import os

def rolling_window_perf(strat_rets, bench_rets, window=12, label="FULL_STRIKE"):
    """
    Calcule la perf glissante sur 'window' mois pour la stratégie et le S&P,
    puis la surperf (strat - S&P), et renvoie aussi la moyenne annuelle
    de ces perfs glissantes.
    """
    # Série mensuelle propre
    df = pd.DataFrame({"strat": strat_rets, "spx": bench_rets}).dropna()

    # Perf cumulée sur window mois (produit (1+R) - 1)
    roll_strat = (1.0 + df["strat"]).rolling(window).apply(np.prod, raw=True) - 1.0
    roll_spx   = (1.0 + df["spx"]).rolling(window).apply(np.prod, raw=True) - 1.0

    # Table des 12m glissants
    roll = pd.DataFrame({
        "Strat 12m %": (roll_strat * 100).round(2),
        "S&P 12m %":   (roll_spx   * 100).round(2),
    }).dropna()

    roll["Excess 12m %"] = (roll["Strat 12m %"] - roll["S&P 12m %"]).round(2)

    # Moyenne par année des perfs 12m glissantes
    by_year = (
        roll
        .groupby(roll.index.year)[["Strat 12m %", "S&P 12m %", "Excess 12m %"]]
        .mean()
        .round(2)
    )
    by_year["N_windows"] = roll.groupby(roll.index.year).size()

    # Exports
    os.makedirs("forecasting", exist_ok=True)
    roll.to_csv(f"forecasting/rolling_{window}m_{label.lower()}.csv", index_label="date")
    by_year.to_csv(f"forecasting/rolling_{window}m_{label.lower()}_by_year.csv", index_label="year")

    print(f"\n=== Rolling {window}-month perf ({label}) : derniers points ===")
    print(roll.tail(12))
    print(f"\n=== Moyenne annuelle des perf {window}m ({label}) ===")
    print(by_year)

    return roll, by_year

# ---- Application à ta strat FULL-STRIKE (strat_rets vs bench_rets) ----
roll_full_12m, roll_full_12m_year = rolling_window_perf(
    strat_rets,      # rendements mensuels de la strat full-strike
    bench_rets,      # rendements mensuels du S&P
    window=12,
    label="FULL_STRIKE"
)

# ---- Application à la strat 50/50 top-2 (strat_rets_5050 vs bench_rets) ----
roll_5050_12m, roll_5050_12m_year = rolling_window_perf(
    strat_rets_5050,  # rendements mensuels de la strat 50/50
    bench_rets,       # S&P en benchmark
    window=12,
    label="TOP2_5050"
)


# import pandas as pd

# df = pd.read_csv('forecasting/rolling_12m_full_strike.csv', index_col='date', parse_dates=True)
# mean = df['Strat 12m %'].mean()
# print(f"Moyenne de Strat 12m % : {mean:.2f}%")
# mean2 = df['S&P 12m %'].mean()
# print(f"Moyenne de S&P 12m % : {mean2:.2f}%")
# std = df['Strat 12m %'].std()
# print(f"Écart-type de Strat 12m % : {std:.2f}%")
# std2 = df['S&P 12m %'].std()
# print(f"Écart-type de S&P 12m % : {std2:.2f}%")
# median = df['Strat 12m %'].median()
# print(f"Médiane de Strat 12m % : {median:.2f}%")
# median2 = df['S&P 12m %'].median()
# print(f"Médiane de S&P 12m % : {median2:.2f}%")
# min_val = df['Strat 12m %'].min()
# print(f"Valeur minimale de Strat 12m % : {min_val:.2f}%")
# min_val2 = df['S&P 12m %'].min()
# print(f"Valeur minimale de S&P 12m % : {min_val2:.2f}%")
# max_val = df['Strat 12m %'].max()
# print(f"Valeur maximale de Strat 12m % : {max_val:.2f}%")
# max_val2 = df['S&P 12m %'].max()
# print(f"Valeur maximale de S&P 12m % : {max_val2:.2f}%")
# quartile1 = df['Strat 12m %'].quantile(0.25)
# print(f"1er quartile de Strat 12m % : {quartile1:.2f}%")
# quartile12 = df['S&P 12m %'].quantile(0.25)
# print(f"1er quartile de S&P 12m % : {quartile12:.2f}%")
# quartile3 = df['Strat 12m %'].quantile(0.75)
# print(f"3e quartile de Strat 12m % : {quartile3:.2f}%")
# quartile32 = df['S&P 12m %'].quantile(0.75)
# print(f"3e quartile de S&P 12m % : {quartile32:.2f}%")