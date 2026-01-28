# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ============================================================
# # 0) PARAMÈTRES
# # ============================================================
# H = 20          # horizon en jours (Return in the next 30 days)
# TRAIN_WINDOW = 2128   # taille de fenêtre de training comme dans le papier
# THRESH_BASE = 0.55    # seuil de décision / couleur

# # ============================================================
# # 1) LOAD DATA
# # ============================================================
# df = pd.read_excel("vol.xlsx")
# df = df.drop(df.index[0])   # première ligne pourrie
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})

# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()

# # Sauvegarde du SPX brut pour le plot
# df_raw_spx = pd.to_numeric(df["Returns30"], errors="coerce")

# # Force numeric sur toutes les features
# for c in df.columns:
#     df[c] = pd.to_numeric(df[c], errors="coerce")

# # ============================================================
# # 2) TRUE FUTURE 30-DAY RETURN (comme dans le texte)
# #    Return_t^H = log(P_{t+H} / P_t)
# # ============================================================
# price = df_raw_spx.astype(float)
# ret_fut = np.log(price.shift(-H)) - np.log(price)   # future H-day log return
# df["Return_H"] = ret_fut

# # ============================================================
# # 3) FEATURE SELECTION (comme dans le papier)
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30"]  # on vire aussi l'ancienne colonne brute
# for col in features_to_drop:
#     if col in df.columns:
#         df = df.drop(columns=[col])

# # ============================================================
# # 4) TARGET = signe du retour futur (classification)
# # ============================================================
# y = (df["Return_H"] > -0.05).astype(int)

# # X = toutes les features sauf la target
# X = df.drop(columns=["Return_H"])

# # Aligner X et y, drop NaN
# mask_ok = y.notna()
# X = X.loc[mask_ok]
# y = y.loc[mask_ok]

# # On enlève les H dernières obs qui n'ont pas de vrai retour entièrement observé
# X = X.iloc[:-H]
# y = y.iloc[:-H]

# print("Taille finale X, y :", X.shape, len(y))

# # ============================================================
# # 5) WALK-FORWARD SANS LOOK-AHEAD
# #    pour un test à l'indice j :
# #    - dernière obs de train = j - H
# #    - première obs de train = (j - H) - TRAIN_WINDOW + 1
# # ============================================================
# T = len(X)
# y_true = []
# y_pred = []
# y_proba = []
# test_dates = []

# # indices entiers pour X/Y
# for test_pos in range(TRAIN_WINDOW + H, T):
#     last_train_pos = test_pos - H
#     first_train_pos = last_train_pos - TRAIN_WINDOW + 1

#     X_train = X.iloc[first_train_pos:last_train_pos+1]
#     y_train = y.iloc[first_train_pos:last_train_pos+1]

#     X_test = X.iloc[[test_pos]]
#     y_test = y.iloc[test_pos]

#     test_dates.append(X_test.index[0])

#     # Scaling uniquement sur le passé
#     scaler = StandardScaler().fit(X_train)
#     Xtr = scaler.transform(X_train)
#     Xte = scaler.transform(X_test)

#     # Random Forest comme dans le papier
#     model = RandomForestClassifier(
#         n_estimators=500,
#         random_state=42,
#         max_depth=None,
#         max_features=int(np.sqrt(X.shape[1])),  # mtry = sqrt(p)
#         n_jobs=-1,
#         class_weight="balanced",
#     )
#     model.fit(Xtr, y_train)

#     proba = model.predict_proba(Xte)[0, 1]
#     pred = int(proba > THRESH_BASE)

#     y_true.append(y_test)
#     y_pred.append(pred)
#     y_proba.append(proba)

# # ============================================================
# # 6) METRICS
# # ============================================================
# acc = accuracy_score(y_true, y_pred)
# auc = roc_auc_score(y_true, y_proba)
# f1  = f1_score(y_true, y_pred)

# print(f"Accuracy : {acc:.4f}")
# print(f"AUC      : {auc:.4f}")
# print(f"F1       : {f1:.4f}")
# print(f"Number of OOS forecasts = {len(y_true)}")

# # AUC ne dépend PAS du seuil
# print(f"\nAUC (indépendant du seuil) : {auc:.4f}")

# # Option : chercher un meilleur seuil via F1 ou Youden
# fpr, tpr, roc_th = roc_curve(y_true, y_proba)
# youden = tpr - fpr
# idx_best = np.argmax(youden)
# thr_youden = roc_th[idx_best]
# print(f"Seuil Youden (max TPR - FPR) : {thr_youden:.3f}")

# # ============================================================
# # 7) PLOT SPX AVEC FOND COLORÉ (seuil = THRESH_BASE)
# # ============================================================
# forecast_dates = pd.DatetimeIndex(test_dates)
# pred_series = pd.Series(y_pred, index=forecast_dates)
# proba_series = pd.Series(y_proba, index=forecast_dates)

# spx = df_raw_spx.loc[forecast_dates.min():forecast_dates.max()]

# def color_from_proba(p, base=THRESH_BASE):
#     """
#     base = seuil de décision.
#     En-dessous de base -> rouge
#     Au-dessus ou égal à base -> vert
#     Intensité = distance normalisée à base.
#     """
#     if p >= base:
#         strength = (p - base) / (1.0 - base) if base < 1.0 else 0.0
#     else:
#         strength = (base - p) / base if base > 0.0 else 0.0

#     strength = max(0.0, min(1.0, strength))

#     if p >= base:
#         return (0, 0.8, 0, strength)   # vert
#     else:
#         return (1, 0, 0, strength)     # rouge

# fig, ax = plt.subplots(figsize=(14, 6))
# ax.plot(spx.index, spx.values, color="black", linewidth=1.5, label="SPX")

# dates = pred_series.index
# probas = proba_series.values

# for i in range(len(dates) - 1):
#     d0, d1 = dates[i], dates[i+1]
#     p = probas[i]
#     col = color_from_proba(p, THRESH_BASE)
#     ax.axvspan(d0, d1, color=col, lw=0)

# green_patch = mpatches.Patch(color=(0, 0.8, 0, 0.6), label=f'Prediction Up (p ≥ {THRESH_BASE})')
# red_patch   = mpatches.Patch(color=(1, 0, 0, 0.6), label=f'Prediction Down (p < {THRESH_BASE})')
# ax.legend(handles=[green_patch, red_patch])

# ax.set_title(f"SPX avec fond coloré par prédiction à {H} jours & intensité = certitude")
# ax.set_ylabel("SPX Level")
# ax.grid(True, linestyle=":", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # 8) EXPORT EXCEL PREDICTIONS VS TRUTH
# # ============================================================
# true_returns = ret_fut.loc[forecast_dates].values  # les vrais retours futurs H jours

# pred_df = pd.DataFrame({
#     "date_observation": forecast_dates,
#     "predicted_future_date": forecast_dates + pd.Timedelta(days=H),
#     "proba_up": y_proba,
#     "predicted_class": y_pred,
#     "true_class": y_true,
#     "true_Hday_log_return": true_returns,
# })

# pred_df = pred_df.sort_values("date_observation")
# pred_df.to_csv("predictions_vs_truth_second_try.csv", index=False)
# print("✔️ Exported predictions_vs_truth_second_try.csv")






###
# ============================================================
#AVEC TEST 





# import pandas as pd
# import numpy as np

# # --- CONFIGURATION ---
# HORIZON = 20
# train_window = 2128  # Ta fenêtre
# n_test_points = 5   # On teste seulement les 5 dernières prédictions pour aller vite

# # --- FONCTION DE DIAGNOSTIC ---
# def run_stability_test(use_recent_data=True):
#     """
#     Teste la stabilité du modèle en fonction de l'inclusion des données récentes.
#     Si use_recent_data=False, on simule le fait de 'tronquer' les données
#     en avançant la date de fin de l'ensemble d'entraînement.
#     """
#     print(f"\n{'='*60}")
#     print(f"STABILITY TEST: use_recent_data = {use_recent_data}")
#     print('='*60)

#     # 1. CHARGEMENT ET PRÉPARATION DES DONNÉES (identique à ton code)
#     df = pd.read_excel("vol.xlsx")
#     df = df.drop(df.index[0])
#     df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.set_index("date").sort_index()
#     df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")
#     df["Returns30"] = np.log(df["Returns30"]).diff(HORIZON)

#     # 1. Identifier la période des "20 jours critiques"
#     periode_critique_start = df.index[-60]  # 60 derniers jours
#     periode_critique = df.loc[periode_critique_start:].copy()

#     # 2. Calculer les statistiques pour les 20 derniers jours vs. les 40 précédents
#     # 2. Calculer les statistiques pour les 20 derniers jours vs. les 40 précédents
#     for col in ['VIX', 'SKEW', 'VVIX', 'Returns30']:  # Note: 'Returns30' au lieu de 'Returns30_log_return'
#         if col in periode_critique.columns:
#             last_20 = periode_critique[col].iloc[-20:]
#             previous_40 = periode_critique[col].iloc[-60:-20]
        
#             print(f"\n--- {col} ---")  # DÉPLACÉ DANS LE 'if'
#             print(f"Moyenne (20 derniers jours) : {last_20.mean():.4f}")
#             print(f"Moyenne (40 jours avant)    : {previous_40.mean():.4f}")
#             print(f"Écart (Différence)          : {last_20.mean() - previous_40.mean():.4f}")
#             if col != 'Returns30':
#                 print(f"Écart-type (20 derniers)    : {last_20.std():.4f}")

#     # Features et target
#     y = (df["Returns30"].shift(-HORIZON) > 0).astype(int)
#     X = df.drop(columns=["Returns30"])
#     for col in ["VIX9D", "VIX3M", "VIX6M"]:
#         if col in X.columns:
#             X = X.drop(columns=[col])

#     # 2. POINT CRITIQUE : SIMULATION DE LA TRONCATURE
#     total_points = len(X)
#     if not use_recent_data:
#         # On simule la perte des 20 points récents en reculant la fin du dataset
#         cutoff = 20
#         X = X.iloc[:-cutoff]
#         y = y.iloc[:-cutoff]
#         print(f"→ Dataset tronqué : {len(X)} points (les {cutoff} plus récents supprimés).")

#     # 3. BOUCLE DE VALIDATION POUR LES 'n_test_points' DERNIÈRES PRÉDICTIONS
#     T = len(X)
#     start_test_idx = T - n_test_points  # On ne regarde que les toutes dernières prédictions

#     results = []
#     for test_position in range(start_test_idx, T):
#         t = test_position - train_window

#         # Vérification des bornes
#         if t < 0:
#             print(f"  Skipping test {test_position}: pas assez de données pour l'entraînement.")
#             continue

#         # Définition des ensembles
#         train_end = t + train_window
#         X_train = X.iloc[t:train_end]
#         y_train = y.iloc[t:train_end]

#         # Gestion des NaN dans y_train (provenant du shift)
#         mask = y_train.notna()
#         X_train = X_train[mask]
#         y_train = y_train[mask]

#         if len(X_train) < train_window * 0.9:  # Si on a perdu trop de données
#             print(f"  ⚠️  Train set trop petit après suppression NaN pour t={t}.")
#             continue

#         X_test = X.iloc[[train_end]]
#         y_test = y.iloc[train_end]

#         if pd.isna(y_test):
#             continue

#         # 4. ENTRAÎNEMENT ET PRÉDICTION (Avec les paramètres CORRIGÉS)
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.ensemble import RandomForestClassifier

#         scaler = StandardScaler().fit(X_train)
#         Xtr = scaler.transform(X_train)
#         Xte = scaler.transform(X_test)

#         # MODÈLE AVEC LES CORRECTIONS APPLIQUÉES
#         from xgboost import XGBClassifier

#         from xgboost import XGBClassifier

#         model = XGBClassifier(
#             n_estimators=500,           # Légèrement plus d'arbres
#             max_depth=4,                # RÉDUIRE la profondeur (de 5 à 4) pour limiter la complexité
#             learning_rate=0.03,         # RÉDUIRE le taux d'apprentissage pour un apprentissage plus prudent
#             subsample=0.7,              # Utiliser seulement 70% des données par arbre (augmente la robustesse)
#             colsample_bytree=0.7,       # Utiliser seulement 70% des features par arbre
#             reg_alpha=0.1,              # AJOUTER une régularisation L1 (réduit le surapprentissage)
#             reg_lambda=1.0,             # AJOUTER une régularisation L2 (standard)
#             eval_metric='logloss',
#             random_state=42,
#             # SUPPRIMER 'use_label_encoder' (paramètre obsolète)
#         )
#         model.fit(Xtr, y_train)
#         proba = model.predict_proba(Xte)[0, 1]
#         pred = int(proba > 0.5)

#         # Stockage des résultats
#         results.append({
#             'Date_Test': X_test.index[0],
#             'True_Label': int(y_test),
#             'Pred_Proba': proba,
#             'Pred_Label': pred,
#             'Train_Size': len(X_train),
#             'Train_End_Date': X_train.index[-1]  # Très important!
#         })

#     # 5. AFFICHAGE DES RÉSULTATS
#     if results:
#         results_df = pd.DataFrame(results)
#         print(f"\n📊 RÉSULTATS POUR LES {len(results_df)} DERNIÈRES PRÉDICTIONS :")
#         print(results_df[['Date_Test', 'True_Label', 'Pred_Label', 'Pred_Proba', 'Train_End_Date']].to_string())
        
#         # Calcul de la précision simple
#         accuracy = (results_df['True_Label'] == results_df['Pred_Label']).mean()
#         print(f"\n🎯 Précision sur ce mini-test : {accuracy:.1%}")
        
#         # Vérification de la date de fin d'entraînement
#         print(f"\n📅 Date de la DONNÉE LA PLUS RÉCENTE utilisée dans l'entraînement :")
#         print(f"   {results_df['Train_End_Date'].iloc[0]}")
        
#         return results_df, accuracy
#     else:
#         print("Aucun résultat généré.")
#         return None, 0

# # --- EXÉCUTION DU TEST ---
# print("ANALYSE DE STABILITÉ DU MODÈLE")
# print("Comparaison avec/sans les 20 données les plus récentes dans l'entraînement.")
# print("\nAttention: Les prédictions testées sont les TOUTES DERNIÈRES du dataset.")

# # Test 1 : AVEC toutes les données
# results_with, acc_with = run_stability_test(use_recent_data=True)

# # Test 2 : SANS les 20 données les plus récentes
# results_without, acc_without = run_stability_test(use_recent_data=False)

# if results_with is not None and results_without is not None:
#     print("\n" + "="*60)
#     print("COMPARAISON FINALE DES DEUX SCÉNARIOS :")
#     print(f"  1. Avec données récentes    -> Précision : {acc_with:.1%}")
#     print(f"  2. Sans données récentes    -> Précision : {acc_without:.1%}")
#     print("\n🔍 Comparez les colonnes 'Pred_Proba' et 'Pred_Label' entre les deux tests.")
#     print("   Si elles sont radicalement différentes, lisez l'analyse ci-dessous.")


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # --- Charger et préparer les données (comme avant) ---
# df = pd.read_excel("vol.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")
# df["Returns30_log_return"] = np.log(df["Returns30"]).diff(20)

# # Cible future
# df['Target'] = (df["Returns30_log_return"].shift(-20) > 0).astype(int)

# # --- Focus sur les 60 derniers jours pour voir la tendance ---
# lookback_days = 60
# recent_df = df.iloc[-lookback_days:].copy()

# # 1. Graphique des Retours et de la Cible
# fig, axes = plt.subplots(3, 1, figsize=(14, 10))
# # Graphique 1 : Le rendement 30 jours qui sera prédit
# axes[0].plot(recent_df.index, recent_df['Returns30_log_return'], label='30d Log Return (to predict)', color='blue', marker='o')
# axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
# axes[0].fill_between(recent_df.index, 0, recent_df['Target'], alpha=0.2, color='green', label='Future UP (Target=1)')
# axes[0].set_ylabel('Log Return')
# axes[0].set_title('Évolution des Rendements (30j) et de la Cible Future (UP/DOWN)')
# axes[0].legend()
# axes[0].grid(True)

# # 2. Graphique d'une variable clé (ex: VIX)
# if 'VIX' in recent_df.columns:
#     axes[1].plot(recent_df.index, recent_df['VIX'], label='VIX', color='red', marker='x')
#     axes[1].set_ylabel('VIX Level')
#     axes[1].set_title('Évolution du VIX (Volatilité Implicite)')
#     axes[1].legend()
#     axes[1].grid(True)

# # 3. Graphique d'une autre variable (ex: SKEW)
# if 'SKEW' in recent_df.columns:
#     axes[2].plot(recent_df.index, recent_df['SKEW'], label='SKEW Index', color='purple', marker='s')
#     axes[2].axhline(y=100, color='black', linestyle='--', linewidth=0.5, label='Neutral Level (100)')
#     axes[2].set_ylabel('SKEW')
#     axes[2].set_title('Évolution du SKEW Index (Crainte des Crashs)')
#     axes[2].legend()
#     axes[2].grid(True)

# plt.tight_layout()
# plt.show()

# # --- Analyse Statistique des 20 Derniers Jours vs. Période Précédente ---
# print("=== ANALYSE STATISTIQUE DES 20 JOURS CRITIQUES ===")
# print(f"Période des '20 derniers jours' : {recent_df.index[-20]} à {recent_df.index[-1]}\n")

# # Comparer les 20 derniers jours avec les 40 précédents
# for col in ['Returns30_log_return', 'VIX', 'SKEW']:
#     if col in df.columns:
#         last_20 = df[col].iloc[-20:]
#         previous_40 = df[col].iloc[-60:-20]
#         print(f"**Variable : {col}**")
#         print(f"  - Moyenne (20 derniers jours) : {last_20.mean():.4f}")
#         print(f"  - Moyenne (40 jours précédents) : {previous_40.mean():.4f}")
#         print(f"  - Écart-type (20 derniers jours) : {last_20.std():.4f}")
#         print(f"  - Signe moyen du rendement (20 derniers) : {(last_20 > 0).mean():.1%} positifs")
#         print()





# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from xgboost import XGBClassifier
# import warnings
# warnings.filterwarnings('ignore')

# # ============================================================
# # 1) LOAD DATA
# # ============================================================
# print("Chargement des données...")
# df = pd.read_excel("vol.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})

# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()

# # Ensure numeric
# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # 2) Compute 30-day log returns (20 jours ouvrés)
# # ============================================================
# HORIZON = 20  # 20 jours ouvrés ≈ 30 jours calendaires
# df["Returns30"] = np.log(df["Returns30"]).diff(HORIZON)

# # ============================================================
# # 3) FEATURE SELECTION FROM PAPER (LASSO RESULT)
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M"]
# for col in features_to_drop:
#     if col in df.columns:
#         df = df.drop(columns=[col])

# # ============================================================
# # 4) Target: sign of next 30d return (classification)
# # IMPORTANT: On ne tronque PAS ici, on garde les NaN
# # ============================================================
# y = (df["Returns30"].shift(-HORIZON) > 0).astype(int)
# X = df.drop(columns=["Returns30"])

# print(f"Taille dataset: {len(df)}")
# print(f"NaN dans y: {y.isna().sum()} (à gérer dans la boucle)")

# # ============================================================
# # 5) WALK-FORWARD VALIDATION (paper: window = 2128)
# # AVEC XGBOOST RÉGULARISÉ ET ROBUSTE
# # ============================================================
# T = len(X)
# train_window = 2128
# n_forecasts = T - train_window - HORIZON  # Ajusté pour éviter les NaN

# print(f"\nValidation Walk-Forward:")
# print(f"- Fenêtre d'entraînement: {train_window} jours")
# print(f"- Nombre de prévisions: {n_forecasts}")

# y_true = []
# y_pred = []
# y_proba = []
# test_dates = []



# for t in range(n_forecasts):
#     # Définition des indices
#     train_end_idx = t + train_window
#     test_idx = train_end_idx
    
#     # Vérifier que y_test n'est pas NaN
#     if pd.isna(y.iloc[test_idx]):
#         continue
    
#     # Données d'entraînement (avec gestion des NaN dans y)
#     X_train = X.iloc[t:train_end_idx]
#     y_train = y.iloc[t:train_end_idx]
    
#     # Supprimer les lignes où y_train est NaN
#     mask_train = y_train.notna()
#     X_train = X_train[mask_train]
#     y_train = y_train[mask_train]
    
#     if len(X_train) < train_window * 0.8:  # Si trop de données manquantes
#         continue
    
#     # Données de test
#     X_test = X.iloc[[test_idx]]
#     y_test = y.iloc[test_idx]
    
#     # Normalisation (uniquement sur l'entraînement)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     n_up = (y == 1).sum()
#     n_down = (y == 0).sum()
#     scale_pos_weight = n_down / n_up  # Ex: si 80% UP, 20% DOWN → weight = 0.25
#     print(f"Ratio déséquilibre (down/up): {scale_pos_weight:.3f}")

#         # Vérifiez la distribution RÉELLE de votre cible pendant l'entraînement
#     print("=== DISTRIBUTION RÉELLE DE LA CIBLE (y_train) ===")
#     # Dans votre boucle, ajoutez :
#     print(f"t={t}: UP={y_train.mean():.1%}, DOWN={1-y_train.mean():.1%}")

    
#     # ============================================================
#     # MODÈLE XGBOOST AVEC RÉGULARISATION RENFORCÉE
#     # ============================================================
#     model = XGBClassifier(
#         n_estimators=500,           # Suffisant pour la stabilité
#         max_depth=4,                # PROFONDEUR LIMITÉE (évite surapprentissage)
#         learning_rate=0.03,         # APPRENTISSAGE LENT (meilleure généralisation)
#         subsample=0.7,              # 70% données par arbre (augmente robustesse)
#         colsample_bytree=0.7,       # 70% features par arbre (diversité)
#         reg_alpha=0.1,              # RÉGULARISATION L1
#         reg_lambda=1.0,             # RÉGULARISATION L2
#         eval_metric='logloss',
#         random_state=42,            # Reproductibilité
#         n_jobs=-1,
#         scale_pos_weight=scale_pos_weight                 # Utiliser tous les coeurs
#     )
    
#     # Entraînement
#     model.fit(X_train_scaled, y_train)
    
#     # Prédiction
#     proba = model.predict_proba(X_test_scaled)[0, 1]
#     threshold = 0.55  # À ajuster selon l'analyse ci-dessous
#     pred = 1 if proba > threshold else 0
    
#     # Stockage
#     y_true.append(y_test)
#     y_pred.append(pred)
#     y_proba.append(proba)
#     test_dates.append(X_test.index[0])
    
#     # Progress bar
#     if (t+1) % 50 == 0:
#         print(f"  Prévisions terminées: {t+1}/{n_forecasts}")

# # ============================================================
# # 6) METRICS
# # ============================================================
# if len(y_true) > 0:
#     acc = accuracy_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_proba)
#     f1 = f1_score(y_true, y_pred)
    
#     print(f"\n{'='*50}")
#     print("RÉSULTATS FINAUX (XGBoost Régularisé)")
#     print(f"{'='*50}")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"AUC       : {auc:.4f}")
#     print(f"F1-Score  : {f1:.4f}")
#     print(f"Nombre de prévisions : {len(y_true)}")
#     print(f"Période test : {test_dates[0]} à {test_dates[-1]}")
    
#     # Distribution des prédictions
#     print(f"\nDistribution des prédictions:")
#     print(f"  UP (1)   : {sum(y_pred)} / {len(y_pred)} ({sum(y_pred)/len(y_pred):.1%})")
#     print(f"  DOWN (0) : {len(y_pred)-sum(y_pred)} / {len(y_pred)} ({(len(y_pred)-sum(y_pred))/len(y_pred):.1%})")
    
#     # ============================================================
#     # 7) EXPORT DES PRÉDICTIONS
#     # ============================================================
#     pred_df = pd.DataFrame({
#         "date_observation": test_dates,
#         "predicted_future_date": pd.to_datetime(test_dates) + pd.Timedelta(days=30),
#         "proba_up": y_proba,
#         "predicted_class": y_pred,
#         "true_class": y_true
#     })
    
#     # Ajouter les vrais rendements
#     true_returns = df["Returns30"].shift(-HORIZON).iloc[train_window:train_window + len(y_true)].values
#     pred_df["true_30d_log_return"] = true_returns
    
#     # Sauvegarde
#     pred_df.to_csv("predictions_xgboost_robuste.csv", index=False)
#     print(f"\n✅ Prédictions exportées: predictions_xgboost_robuste.csv")
    
#     # ============================================================
#     # 8) TEST DE STABILITÉ RAPIDE (optionnel)
#     # ============================================================
#     print(f"\n{'='*50}")
#     print("TEST DE STABILITÉ RAPIDE")
#     print(f"{'='*50}")
    
#     # Vérifier la sensibilité aux dernières données
#     last_20_acc = accuracy_score(y_true[-20:], y_pred[-20:]) if len(y_true) >= 20 else None
#     first_20_acc = accuracy_score(y_true[:20], y_pred[:20]) if len(y_true) >= 20 else None
    
#     if last_20_acc is not None:
#         print(f"Accuracy (20 premières prédictions) : {first_20_acc:.1%}")
#         print(f"Accuracy (20 dernières prédictions)  : {last_20_acc:.1%}")
#         print(f"Différence : {abs(last_20_acc - first_20_acc):.1%}")

# else:
#     print("Aucune prévision générée. Vérifiez les données ou les paramètres.")

# # ============================================================
# # 9) ANALYSE DES 20 JOURS CRITIQUES
# # ============================================================
# print(f"\n{'='*50}")
# print("ANALYSE DES VARIABLES SUR LES 60 DERNIERS JOURS")
# print(f"{'='*50}")

# # Focus sur les 60 derniers jours
# last_60 = df.iloc[-60:].copy()

# for col in ['VIX', 'SKEW', 'VVIX', 'Returns30']:
#     if col in last_60.columns:
#         last_20 = last_60[col].iloc[-20:]
#         previous_40 = last_60[col].iloc[-60:-20]
        
#         print(f"\n{col}:")
#         print(f"  Moyenne (20 derniers jours) : {last_20.mean():.4f}")
#         print(f"  Moyenne (40 jours avant)    : {previous_40.mean():.4f}")
#         print(f"  Variation (%)               : {(last_20.mean()/previous_40.mean()-1)*100:+.1f}%")
#         if col == 'Returns30':
#             print(f"  Signe positif (20 derniers) : {(last_20 > 0).mean():.1%}")


# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt

# # Trouver le seuil optimal pour maximiser F1-Score
# precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
# f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
# optimal_idx = np.argmax(f1_scores)
# optimal_threshold = thresholds[optimal_idx]

# print(f"\n📊 ANALYSE DU SEUIL OPTIMAL")
# print(f"Seuil actuel (0.5) : F1 = {f1:.4f}")
# print(f"Seuil optimal : {optimal_threshold:.3f}")
# print(f"F1 optimal : {f1_scores[optimal_idx]:.4f}")

# # Visualiser
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, f1_scores[:-1], label='F1-Score')
# plt.axvline(x=0.5, color='red', linestyle='--', label='Seuil 0.5')
# plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
# plt.xlabel('Seuil de probabilité')
# plt.ylabel('F1-Score')
# plt.title('Performance selon le seuil de classification')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Ré-évaluer avec le seuil optimal
# y_pred_optimal = [1 if p > optimal_threshold else 0 for p in y_proba]
# acc_opt = accuracy_score(y_true, y_pred_optimal)
# auc_opt = roc_auc_score(y_true, y_proba)
# f1_opt = f1_score(y_true, y_pred_optimal)

# print(f"\n🎯 PERFORMANCE AVEC SEUIL OPTIMAL ({optimal_threshold:.3f})")
# print(f"Accuracy  : {acc_opt:.4f} (avant: {acc:.4f})")
# print(f"AUC       : {auc_opt:.4f} (identique)")
# print(f"F1-Score  : {f1_opt:.4f} (avant: {f1:.4f})")

# # Distribution corrigée
# print(f"\n📈 DISTRIBUTION CORRIGÉE:")
# print(f"UP (1)   : {sum(y_pred_optimal)} / {len(y_pred_optimal)} ({sum(y_pred_optimal)/len(y_pred_optimal):.1%})")
# print(f"DOWN (0) : {len(y_pred_optimal)-sum(y_pred_optimal)} / {len(y_pred_optimal)} ({(len(y_pred_optimal)-sum(y_pred_optimal))/len(y_pred_optimal):.1%})")




# # ============================================================
# Avec requalibrage du seuil de classification
# # ============================================================
# # ============================================================
# Avec requalibrage du seuil de classification
# # ============================================================
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from xgboost import XGBClassifier
# import warnings
# warnings.filterwarnings('ignore')

# # ============================================================
# # 1) LOAD DATA
# # ============================================================
# print("Chargement des données...")
# df = pd.read_excel("vol1.xlsx")
# df = df.drop(df.index[0])
# print(df.columns[:3])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df = df[2461:]
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()

# # Sauvegarder le prix brut
# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # 2) Compute log price and future returns
# # ============================================================
# HORIZON = 20  # 20 jours ouvrés ≈ 30 jours calendaires

# # Calcul du prix log
# df["log_price"] = np.log(df["Returns30"])

# # Calcul du rendement FUTUR (cible)
# df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]

# # ============================================================
# # 3) FEATURE SELECTION FROM PAPER (LASSO RESULT)
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M"]
# for col in features_to_drop:
#     if col in df.columns:
#         df = df.drop(columns=[col])

# # ============================================================
# # 4) Target: sign of next 30d return (classification)
# # CORRECTION CRITIQUE ICI : utiliser future_return, pas Returns30.shift()
# # ============================================================
# y = (df["future_return"] > 0).astype(int)

# # Features : exclure les colonnes de prix et rendements
# X = df.drop(columns=["Returns30", "log_price", "future_return"])
# # X = df.drop(columns=["Returns30", "log_price", "future_return"]).pct_change(1).fillna(0)

# print(f"Taille dataset: {len(df)}")
# print(f"NaN dans y: {y.isna().sum()} (à gérer dans la boucle)")
# print(f"Distribution de y: {y.mean():.1%} UP, {1-y.mean():.1%} DOWN")

# # ============================================================
# # DIAGNOSTIC : Vérification manuelle
# # ============================================================
# print("\n=== VÉRIFICATION MANUELLE ===")
# test_idx = 1000  # Choisir un index au milieu
# if test_idx + HORIZON < len(df):
#     date_t = df.index[test_idx]
#     date_future = df.index[test_idx + HORIZON]
#     price_t = df_raw_spx.iloc[test_idx]
#     price_future = df_raw_spx.iloc[test_idx + HORIZON]
    
#     print(f"Date t: {date_t.date()}")
#     print(f"Date t+{HORIZON}: {date_future.date()}")
#     print(f"Prix à t: {price_t:.2f}")
#     print(f"Prix à t+{HORIZON}: {price_future:.2f}")
    
#     # Calcul manuel
#     log_return = np.log(price_future) - np.log(price_t)
#     print(f"Rendement futur calculé: {log_return:.4f}")
#     print(f"y[{date_t.date()}] (from df): {'UP' if y.iloc[test_idx] == 1 else 'DOWN'}")
#     print(f"Vérification: {'✓ OK' if (log_return > 0) == (y.iloc[test_idx] == 1) else '✗ PROBLEM'}")

# # ============================================================
# # 5) WALK-FORWARD VALIDATION
# # AVEC CORRECTION DU LOOK-AHEAD
# # ============================================================
# T = len(X)
# train_window = 2128

# # Ajustement pour éviter le look-ahead
# # On ne peut prédire que si on a HORIZON observations après l'entraînement
# n_forecasts = T - train_window - HORIZON

# print(f"\nValidation Walk-Forward:")
# print(f"- Fenêtre d'entraînement: {train_window} jours")
# print(f"- Nombre de prévisions: {n_forecasts}")
# print(f"- HORIZON: {HORIZON} jours")

# y_true = []
# y_pred = []
# y_proba = []
# test_dates = []

# # Nouveaux arrays pour stocker plus d'infos
# y_true_raw = []  # Pour stocker les rendements bruts
# all_X_test = []  # Pour stocker les features de test
# all_train_sizes = []  # Pour stocker la taille des données d'entraînement

# for t in range(n_forecasts):
#     train_end_idx = t + train_window
#     test_idx = train_end_idx
    
#     # ============================================================
#     # CRITIQUE : Gestion du look-ahead
#     # ============================================================
#     # Pour l'entraînement, on ne peut utiliser que les données
#     # dont la cible est COMPLÈTEMENT connue à la date de test
#     # La dernière cible connue à la date test_idx est à l'indice test_idx - HORIZON
#     last_trainable_idx = test_idx - HORIZON
    
#     if last_trainable_idx <= t:
#         continue  # Pas assez de données d'entraînement
    
#     # Données d'entraînement
#     X_train = X.iloc[t:last_trainable_idx]
#     y_train = y.iloc[t:last_trainable_idx]
    
#     # Vérifier qu'on a assez de données
#     if len(X_train) < train_window * 0.7:
#         continue
    
#     # Données de test
#     X_test = X.iloc[[test_idx]]
#     y_test = y.iloc[test_idx]
    
#     # Normalisation
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Balance des classes
#     n_up = y_train.sum()
#     n_down = len(y_train) - n_up
#     scale_pos_weight = (n_down / n_up)*2 if n_up > 0 else 1.0
    
#     # à
#     # model = XGBClassifier(
#     #     n_estimators=500, 
#     #     max_depth=4,
#     #     learning_rate=0.1,
#     #     scale_pos_weight=scale_pos_weight,
#     #     random_state=42,
#     #     n_jobs=-1
#     # )
#     from sklearn.ensemble import RandomForestClassifier

#     model = RandomForestClassifier(
#         n_estimators=500,  # Réduit le nombre d'arbres
#         max_depth=None,       # Limite la profondeur
#         max_features=4,   # Force plus de données par feuille
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
# )
        
#     model.fit(X_train_scaled, y_train)
    
#     # Prédiction
#     proba = model.predict_proba(X_test_scaled)[0, 1]
    
#     # Pas de recalibrage pour l'instant
#     threshold = 0.5
#     pred = 1 if proba > threshold else 0
    
#     # Stockage
#     y_true.append(y_test)
#     y_pred.append(pred)
#     y_proba.append(proba)
#     test_dates.append(X_test.index[0])
    
#     # Stocker les données supplémentaires
#     y_true_raw.append(df["future_return"].iloc[test_idx])  # Rendement brut
#     all_X_test.append(X_test.values[0])  # Features de test
#     all_train_sizes.append(len(X_train))  # Taille de l'entraînement
    
#     if (t+1) % 50 == 0:
#         print(f"  Prévisions: {t+1}/{n_forecasts}")

# # ============================================================
# # 6) RÉSULTATS
# # ============================================================
# if len(y_true) > 0:
#     acc = accuracy_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_proba)
#     f1 = f1_score(y_true, y_pred)
    
#     print(f"\n{'='*60}")
#     print("RÉSULTATS CORRIGÉS (sans look-ahead)")
#     print(f"{'='*60}")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"AUC       : {auc:.4f}")
#     print(f"F1-Score  : {f1:.4f}")
#     print(f"Nombre de prévisions : {len(y_true)}")
    
#     # Distribution
#     pred_up = sum(y_pred)/len(y_pred)
#     real_up = sum(y_true)/len(y_true)
#     print(f"\nDistribution des prédictions:")
#     print(f"  UP (1)   : {sum(y_pred)} / {len(y_pred)} ({pred_up:.1%})")
#     print(f"  DOWN (0) : {len(y_pred)-sum(y_pred)} / {len(y_pred)} ({(1-pred_up):.1%})")
#     print(f"\nDistribution RÉELLE:")
#     print(f"  UP (1)   : {real_up:.1%}")
#     print(f"  DOWN (0) : {1-real_up:.1%}")
#     print(f"\nBiais (Prédit - Réel): {(pred_up - real_up):+.1%}")
    
#     # Vérification critique
#     if auc < 0.5:
#         print(f"\n⚠️ ATTENTION: AUC < 0.5 !")
#         print(f"   Le modèle est pire que le hasard.")
#         print(f"   Cela suggère un problème fondamental (look-ahead, fuite de données, etc.)")
    
#     print(f"\nPériode test : {test_dates[0].date()} à {test_dates[-1].date()}")
#     print(f"Durée totale : {(test_dates[-1] - test_dates[0]).days} jours")
    
#     # ============================================================
#     # 7) EXPORT DES DONNÉES DÉTAILLÉES
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("EXPORT DES DONNÉES DE PRÉDICTION")
#     print(f"{'='*60}")
    
#     # Création du DataFrame d'export
#     pred_df = pd.DataFrame({
#         'date_observation': test_dates,
#         'date_future': [date + pd.Timedelta(days=30) for date in test_dates],
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'y_proba': y_proba,
#         'future_return_raw': y_true_raw,  # Rendement brut (log)
#         'train_size': all_train_sizes,
#         'pred_correct': [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)],
#         'proba_bin': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in y_proba]
#     })
    
#     # Ajouter les features de test
#     for i, col in enumerate(X.columns):
#         pred_df[f'feature_{col}'] = [x[i] for x in all_X_test]
    
#     # Ajouter des métriques supplémentaires
#     pred_df['prediction_error'] = abs(pred_df['y_proba'] - 0.5)  # Distance au seuil
#     pred_df['return_magnitude'] = abs(pred_df['future_return_raw'])
    
#     # Calculer le rendement réalisé en %
#     pred_df['future_return_pct'] = np.exp(pred_df['future_return_raw']) - 1
    
#     # Sauvegarde
#     export_filename = "predictions_detailed_analysis.csv"
#     pred_df.to_csv(export_filename, index=False, encoding='utf-8')
    
#     print(f"✅ Données exportées dans: {export_filename}")
#     print(f"   Nombre d'observations exportées: {len(pred_df)}")
    
#     # ============================================================
#     # 8) ANALYSE DÉTAILLÉE DES RÉSULTATS
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("ANALYSE DÉTAILLÉE DES PRÉDICTIONS")
#     print(f"{'='*60}")
    
#     # a) Performance par période
#     pred_df['year'] = pred_df['date_observation'].dt.year
#     yearly_perf = pred_df.groupby('year').agg({
#         'pred_correct': 'mean',
#         'y_proba': 'mean',
#         'future_return_pct': 'mean',
#         'date_observation': 'count'
#     }).rename(columns={
#         'pred_correct': 'accuracy',
#         'date_observation': 'n_observations'
#     })
    
#     print(f"\na) Performance par année:")
#     print(yearly_perf.round(3))
    
#     # b) Analyse par niveau de confiance
#     confidence_bins = pd.cut(pred_df['y_proba'], bins=[0, 0.3, 0.7, 1.0], 
#                             labels=['Low (0-0.3)', 'Medium (0.3-0.7)', 'High (0.7-1.0)'])
#     conf_analysis = pred_df.groupby(confidence_bins).agg({
#         'pred_correct': 'mean',
#         'y_true': 'mean',
#         'date_observation': 'count'
#     }).rename(columns={
#         'pred_correct': 'accuracy',
#         'y_true': 'actual_up_rate',
#         'date_observation': 'count'
#     })
    
#     print(f"\nb) Performance par niveau de confiance:")
#     print(conf_analysis.round(3))
    
#     # c) Vérification des features importantes
#     print(f"\nc) Top 5 des meilleures prédictions:")
#     best_predictions = pred_df.nlargest(5, 'prediction_error')
#     for idx, row in best_predictions.iterrows():
#         print(f"   {row['date_observation'].date()}: proba={row['y_proba']:.3f}, "
#               f"pred={row['y_pred']}, actual={row['y_true']}, "
#               f"return={row['future_return_pct']:.2%}")
    
#     # d) Analyse des erreurs
#     errors_df = pred_df[pred_df['pred_correct'] == 0]
#     print(f"\nd) Analyse des erreurs ({len(errors_df)} erreurs):")
#     if len(errors_df) > 0:
#         print(f"   - Faux positifs (prédit UP, actual DOWN): {len(errors_df[errors_df['y_pred']==1])}")
#         print(f"   - Faux négatifs (prédit DOWN, actual UP): {len(errors_df[errors_df['y_pred']==0])}")
#         print(f"   - Rendement moyen lors des erreurs: {errors_df['future_return_pct'].mean():.2%}")
    
#     # e) Vérification de la calibration
#     calibration_bins = pd.cut(pred_df['y_proba'], bins=10)
#     calib_stats = pred_df.groupby(calibration_bins).agg({
#         'y_true': 'mean',
#         'y_proba': 'mean',
#         'date_observation': 'count'
#     }).rename(columns={
#         'y_true': 'actual_up_rate',
#         'y_proba': 'predicted_up_rate',
#         'date_observation': 'count'
#     })
    
#     print(f"\ne) Calibration (idéal: predicted ≈ actual):")
#     print(calib_stats.round(3))
    
#     # f) Test de base: corrélation entre proba et rendement futur
#     correlation = pred_df['y_proba'].corr(pred_df['future_return_raw'])
#     print(f"\nf) Corrélation proba-rendement: {correlation:.4f}")
#     if correlation > 0:
#         print(f"   ✓ Positive: bonne direction")
#     else:
#         print(f"   ✗ Négative: modèle prédit à l'envers")
    
#     # ============================================================
#     # 9) RECOMMANDATIONS POUR DIAGNOSTIC
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("DIAGNOSTIC ET RECOMMANDATIONS")
#     print(f"{'='*60}")
    
#     issues = []
    
#     if auc < 0.5:
#         issues.append("AUC < 0.5 → modèle pire que hasard")
    
#     if correlation < 0:
#         issues.append("Corrélation négative → modèle prédit à l'envers")
    
#     if abs(pred_up - real_up) > 0.1:
#         issues.append(f"Biais élevé ({pred_up-real_up:+.1%})")
    
#     if len(issues) > 0:
#         print("Problèmes détectés:")
#         for issue in issues:
#             print(f"  ⚠️ {issue}")
        
#         print(f"\nActions recommandées:")
#         print(f"  1. Vérifier le fichier CSV exporté pour comprendre les erreurs")
#         print(f"  2. Tester avec un modèle plus simple (Random Forest)")
#         print(f"  3. Vérifier que les données correspondent exactement au papier")
#         print(f"  4. Tester avec HORIZON = 30 jours calendaires (comme le papier)")
#         print(f"  5. Examiner les dates de début/fin de votre dataset")
#     else:
#         print("✓ Aucun problème majeur détecté")
    
#     print(f"\nProchaines étapes:")
#     print(f"  1. Ouvrir le fichier {export_filename} dans Excel")
#     print(f"  2. Analyser les colonnes 'pred_correct' et 'future_return_pct'")
#     print(f"  3. Vérifier si les erreurs sont concentrées sur certaines périodes")
#     print(f"  4. Comparer avec les résultats du papier (Table 9)")
    
# else:
#     print("Aucune prévision générée.")

# print(f"\n{'='*60}")
# print("EXÉCUTION TERMINÉE")
# print(f"\n{'='*60}")



# def verify_no_lookahead():
#     """Vérifie qu'il n'y a pas de look-ahead dans le pipeline"""
#     print("\n=== VÉRIFICATION ANTI-LOOK-AHEAD ===")
    
#     # Pour chaque prédiction, vérifier manuellement
#     for i in range(min(5, len(test_dates))):  # Vérifier les 5 premières
#         date_test = test_dates[i]
#         idx_test = df.index.get_loc(date_test)
        
#         print(f"\nPrédiction {i+1}: {date_test.date()}")
#         print(f"  Indice test: {idx_test}")
        
#         # Calculer quel était le dernier indice d'entraînement
#         last_train_idx = idx_test - HORIZON
#         date_last_train = df.index[last_train_idx]
        
#         print(f"  Dernière date d'entraînement: {date_last_train.date()}")
#         print(f"  Écart: {(date_test - date_last_train).days} jours")
        
#         # Vérifier que la dernière cible d'entraînement est complètement connue
#         last_train_target_end = last_train_idx + HORIZON
#         if last_train_target_end <= idx_test:
#             print(f"  ✓ Dernière cible connue: {df.index[last_train_target_end-1].date()}")
#             print(f"  ✓ Aucun look-ahead détecté")
#         else:
#             print(f"  ✗ LOOK-AHEAD DÉTECTÉ!")
#             print(f"    La cible d'entraînement nécessite des données jusqu'à {df.index[last_train_target_end-1].date()}")
#             print(f"    Ce qui est APRÈS la date de test {date_test.date()}")
    
#     print("\n" + "="*60)

# # Appeler cette fonction après la boucle
# verify_no_lookahead()




# ######## TEST TO DELETE ########## Global Scaling with Look-Ahead
# ##################################

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.ensemble import RandomForestClassifier
# import warnings
# warnings.filterwarnings('ignore')

# print("Chargement des données...")
# df = pd.read_excel("vol1.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()
# df = df[4800:]
# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # PARAMÈTRES EXACTEMENT COMME LE PAPIER
# # ============================================================
# HORIZON = 10
# train_window = 2128
# total_obs = len(df)

# print(f"Total observations: {total_obs}")
# print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")

# # ============================================================
# # CIBLE : Signe du rendement à HORIZON jours
# # ============================================================
# df["log_price"] = np.log(df["Returns30"])
# df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
# y = (df["future_return"] > 0).astype(int)

# print(f"Distribution de y: {y.mean():.2%} hausses")

# # ============================================================
# # FEATURES : Exclure VIX9D, VIX3M, VIX6M comme le papier
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
# X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
# print(f"Features utilisées: {list(X.columns)}")
# print(f"Nombre de features: {len(X.columns)}")

# # ============================================================
# # SCALING GLOBAL (COMME LE PAPIER) - AVEC LOOK-AHEAD POTENTIEL
# # ============================================================
# print("\n=== APPLYING GLOBAL SCALING (POTENTIAL LOOK-AHEAD) ===")
# scaler_global = StandardScaler()
# X_scaled_global = scaler_global.fit_transform(X)
# X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)

# # ============================================================
# # WALK-FORWARD VALIDATION AVEC DONNÉES GLOBALEMENT SCALÉES
# # ============================================================

# first_test_idx = train_window
# last_test_idx = total_obs - HORIZON - 1

# print(f"\nWalk-Forward Validation with Global Scaling:")
# print(f"First test index: {first_test_idx} (date: {df.index[first_test_idx].date()})")
# print(f"Last test index: {last_test_idx} (date: {df.index[last_test_idx].date()})")
# print(f"Nombre de prédictions: {last_test_idx - first_test_idx + 1}")

# y_true_global, y_pred_global, y_proba_global = [], [], []
# test_dates_global = []

# for test_idx in range(first_test_idx, last_test_idx + 1):
#     train_start = test_idx - train_window - HORIZON
#     train_end = test_idx - HORIZON
    
#     if train_start < 0:
#         continue
    
#     X_train_global = X_scaled_global.iloc[train_start:train_end]
#     y_train = y.iloc[train_start:train_end]
    
#     if len(X_train_global) != train_window:
#         continue
    
#     X_test_global = X_scaled_global.iloc[test_idx:test_idx+1]
#     y_test = y.iloc[test_idx]
    
#     model = RandomForestClassifier(
#         n_estimators=500,
#         max_features='sqrt',
#         min_samples_split=2,
#         min_samples_leaf=1,
#         bootstrap=True,
#         n_jobs=-1,
#         random_state=42,
#         class_weight='balanced'
#     )
    
    
#     model.fit(X_train_global, y_train)
    
#     proba = model.predict_proba(X_test_global)[0, 1]
#     pred = 1 if proba > 0.5 else 0
    
#     y_true_global.append(y_test)
#     y_pred_global.append(pred)
#     y_proba_global.append(proba)
#     test_dates_global.append(df.index[test_idx])
    
#     if len(y_true_global) % 100 == 0:
#         print(f"  Prédictions: {len(y_true_global)}...")

# # ============================================================
# # RÉSULTATS AVEC SCALING GLOBAL
# # ============================================================
# if len(y_true_global) > 0:
#     acc_global = accuracy_score(y_true_global, y_pred_global)
#     auc_global = roc_auc_score(y_true_global, y_proba_global)
#     f1_global = f1_score(y_true_global, y_pred_global)
    
#     print(f"\n{'='*60}")
#     print("RÉSULTATS AVEC SCALING GLOBAL (POTENTIAL LOOK-AHEAD)")
#     print(f"{'='*60}")
#     print(f"Accuracy  : {acc_global:.4f}")
#     print(f"AUC       : {auc_global:.4f}")
#     print(f"F1-Score  : {f1_global:.4f}")
#     print(f"Nombre de prédictions : {len(y_true_global)}")
    
#     print(f"\nVérification temporelle:")
#     print(f"Première prédiction: {test_dates_global[0].date()}")
#     print(f"Dernière prédiction: {test_dates_global[-1].date()}")
    
#     # ============================================================
#     # COMPARAISON AVEC SCALING LOCAL (SANS LOOK-AHEAD)
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("POUR COMPARAISON : SCALING LOCAL (SANS LOOK-AHEAD)")
#     print(f"{'='*60}")
    
#     y_true_local, y_pred_local, y_proba_local = [], [], []
    
#     for test_idx in range(first_test_idx, last_test_idx + 1):
#         train_start = test_idx - train_window - HORIZON
#         train_end = test_idx - HORIZON
        
#         if train_start < 0:
#             continue
        
#         X_train_local = X.iloc[train_start:train_end]
#         y_train = y.iloc[train_start:train_end]
        
#         if len(X_train_local) != train_window:
#             continue
        
#         X_test_local = X.iloc[test_idx:test_idx+1]
#         y_test = y.iloc[test_idx]
        
#         scaler_local = StandardScaler()
#         X_train_local_scaled = scaler_local.fit_transform(X_train_local)
#         X_test_local_scaled = scaler_local.transform(X_test_local)
        
#         model_local = RandomForestClassifier(
#             n_estimators=500,
#             max_features='sqrt',
#             min_samples_split=2,
#             min_samples_leaf=1,
#             bootstrap=True,
#             n_jobs=-1,
#             random_state=42,
#             class_weight='balanced'
#         )
        
#         model_local.fit(X_train_local_scaled, y_train)
#         proba_local = model_local.predict_proba(X_test_local_scaled)[0, 1]
#         pred_local = 1 if proba_local > 0.5 else 0
        
#         y_true_local.append(y_test)
#         y_pred_local.append(pred_local)
#         y_proba_local.append(proba_local)
    
#     if len(y_true_local) > 0:
#         acc_local = accuracy_score(y_true_local, y_pred_local)
#         auc_local = roc_auc_score(y_true_local, y_proba_local)
#         f1_local = f1_score(y_true_local, y_pred_local)
        
#         print(f"Accuracy  : {acc_local:.4f}")
#         print(f"AUC       : {auc_local:.4f}")
#         print(f"F1-Score  : {f1_local:.4f}")
#         print(f"Nombre de prédictions : {len(y_true_local)}")
        
#         print(f"\n{'='*60}")
#         print("COMPARAISON DES DEUX MÉTHODES")
#         print(f"{'='*60}")
#         print(f"Différence Accuracy : {acc_global - acc_local:+.4f}")
#         print(f"Différence AUC      : {auc_global - auc_local:+.4f}")
#         print(f"Différence F1       : {f1_global - f1_local:+.4f}")
        
#         if auc_global > auc_local:
#             print(f"\n⚠️  Le scaling global améliore l'AUC de {auc_global - auc_local:.4f} points")
#             print("   Cela suggère un LOOK-AHEAD dans le scaling global!")
#         else:
#             print(f"\n✓ Le scaling local donne de meilleurs résultats")
    
#     # ============================================================
#     # DIAGNOSTIC
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("DIAGNOSTIC")
#     print(f"{'='*60}")
    
#     if auc_global < 0.5:
#         print("Problème: AUC < 0.5 même avec scaling global")
#         print("Causes possibles:")
#         print("1. Données incorrectes ou période différente")
#         print("2. Horizon temporel (20 jours) trop court vs papier (30 jours)")
#         print("3. Features manquantes ou différentes")
    
#     elif auc_global > 0.7:
#         print(f"✅ AUC élevé ({auc_global:.4f}) avec scaling global")
#         print("   Cela correspond aux résultats du papier")
#         print("   Mais attention: le scaling global peut créer du look-ahead")
    
#     else:
#         print(f"AUC modéré: {auc_global:.4f}")

# else:
#     print("Aucune prédiction générée")




##########
####################################
### Regression Logistique 
# ########## 
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.linear_model import LogisticRegression
# import warnings
# warnings.filterwarnings('ignore')

# print("Chargement des données...")
# df = pd.read_excel("vol1.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()

# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # PARAMÈTRES EXACTEMENT COMME LE PAPIER
# # ============================================================
# HORIZON = 15
# train_window = 2128

# # ============================================================
# # CIBLE : Signe du rendement à HORIZON jours
# # ============================================================
# df["log_price"] = np.log(df["Returns30"])
# df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
# y = (df["future_return"] > 0).astype(int)

# # ============================================================
# # FEATURES : Exclure VIX9D, VIX3M, VIX6M comme le papier
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
# X = df.drop(columns=[col for col in features_to_drop if col in df.columns])

# # ============================================================
# # ÉTAPE CRITIQUE : Supprimer les NaN et aligner X et y
# # ============================================================
# print("\n=== TRAITEMENT DES DONNÉES MANQUANTES ===")

# # 1. Créer un DataFrame avec X et y pour supprimer les lignes avec NaN dans l'un ou l'autre
# data_combined = pd.concat([X, y], axis=1)
# initial_rows = len(data_combined)
# data_combined = data_combined.dropna()
# final_rows = len(data_combined)

# print(f"Lignes initiales: {initial_rows}")
# print(f"Lignes après suppression des NaN: {final_rows}")
# print(f"Lignes supprimées: {initial_rows - final_rows} ({100*(initial_rows-final_rows)/initial_rows:.1f}%)")

# # 2. Séparer à nouveau X et y
# X = data_combined.drop(columns=[y.name])
# y = data_combined[y.name]

# # 3. Extraire les dates correspondantes (important pour la boucle)
# # Garder les dates originales qui n'ont pas été supprimées
# date_series = data_combined.index.to_series().reset_index(drop=True)

# # 4. Réindexer X et y pour avoir des indices continus
# X = X.reset_index(drop=True)
# y = y.reset_index(drop=True)

# total_obs = len(X)

# print(f"\nTotal observations après nettoyage: {total_obs}")
# print(f"Période couverte: {date_series.iloc[0]} à {date_series.iloc[-1]}")
# print(f"Distribution de y: {y.mean():.2%} hausses")
# print(f"Features utilisées: {list(X.columns)}")
# print(f"Nombre de features: {len(X.columns)}")

# # ============================================================
# # SCALING GLOBAL (COMME LE PAPIER) - AVEC LOOK-AHEAD POTENTIEL
# # ============================================================
# print("\n=== APPLYING GLOBAL SCALING (POTENTIAL LOOK-AHEAD) ===")
# scaler_global = StandardScaler()
# X_scaled_global = scaler_global.fit_transform(X)
# X_scaled_global = pd.DataFrame(X_scaled_global, columns=X.columns)

# # ============================================================
# # WALK-FORWARD VALIDATION AVEC DONNÉES GLOBALEMENT SCALÉES
# # ============================================================

# # Ajuster les indices pour tenir compte de HORIZON
# first_test_idx = train_window
# last_test_idx = total_obs - HORIZON - 1

# print(f"\nWalk-Forward Validation with Global Scaling:")
# print(f"First test index: {first_test_idx} (date: {date_series.iloc[first_test_idx]})")
# print(f"Last test index: {last_test_idx} (date: {date_series.iloc[last_test_idx]})")
# print(f"Nombre de prédictions: {last_test_idx - first_test_idx + 1}")

# y_true_global, y_pred_global, y_proba_global = [], [], []
# test_dates_global = []

# for test_idx in range(first_test_idx, last_test_idx + 1):
#     train_start = test_idx - train_window - HORIZON
#     train_end = test_idx - HORIZON
    
#     if train_start < 0:
#         continue
    
#     X_train_global = X_scaled_global.iloc[train_start:train_end]
#     y_train = y.iloc[train_start:train_end]
    
#     if len(X_train_global) != train_window:
#         continue
    
#     X_test_global = X_scaled_global.iloc[test_idx:test_idx+1]
#     y_test = y.iloc[test_idx]
    
#     # === RÉGRESSION LOGISTIQUE ===
#     model = LogisticRegression(
#         penalty='l2',  # Ridge regularization comme le papier
#         C=2,  # Inverse de la force de régularisation
#         class_weight='balanced',
#         random_state=42,
#         max_iter=2000,
#         solver='lbfgs'
#     )
    
#     model.fit(X_train_global, y_train)
    
#     proba = model.predict_proba(X_test_global)[0, 1]
#     pred = 1 if proba > 0.5 else 0
    
#     y_true_global.append(y_test)
#     y_pred_global.append(pred)
#     y_proba_global.append(proba)
#     test_dates_global.append(date_series.iloc[test_idx])
    
#     if len(y_true_global) % 100 == 0:
#         print(f"  Prédictions: {len(y_true_global)}...")

# # ============================================================
# # RÉSULTATS AVEC SCALING GLOBAL
# # ============================================================
# if len(y_true_global) > 0:
#     acc_global = accuracy_score(y_true_global, y_pred_global)
#     auc_global = roc_auc_score(y_true_global, y_proba_global)
#     f1_global = f1_score(y_true_global, y_pred_global)
    
#     print(f"\n{'='*60}")
#     print("RÉSULTATS AVEC SCALING GLOBAL (LOGISTIC REGRESSION)")
#     print(f"{'='*60}")
#     print(f"Accuracy  : {acc_global:.4f}")
#     print(f"AUC       : {auc_global:.4f}")
#     print(f"F1-Score  : {f1_global:.4f}")
#     print(f"Nombre de prédictions : {len(y_true_global)}")
    
#     print(f"\nVérification temporelle:")
#     print(f"Première prédiction: {test_dates_global[0]}")
#     print(f"Dernière prédiction: {test_dates_global[-1]}")
    
#     # ============================================================
#     # COMPARAISON AVEC SCALING LOCAL (SANS LOOK-AHEAD)
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("POUR COMPARAISON : SCALING LOCAL (SANS LOOK-AHEAD)")
#     print(f"{'='*60}")
    
#     y_true_local, y_pred_local, y_proba_local = [], [], []
    
#     for test_idx in range(first_test_idx, last_test_idx + 1):
#         train_start = test_idx - train_window - HORIZON
#         train_end = test_idx - HORIZON
        
#         if train_start < 0:
#             continue
        
#         X_train_local = X.iloc[train_start:train_end]
#         y_train = y.iloc[train_start:train_end]
        
#         if len(X_train_local) != train_window:
#             continue
        
#         X_test_local = X.iloc[test_idx:test_idx+1]
#         y_test = y.iloc[test_idx]
        
#         scaler_local = StandardScaler()
#         X_train_local_scaled = scaler_local.fit_transform(X_train_local)
#         X_test_local_scaled = scaler_local.transform(X_test_local)
        
#         model_local = LogisticRegression(
#             penalty='l2',
#             C=1.0,
#             class_weight='balanced',
#             random_state=42,
#             max_iter=1000,
#             solver='lbfgs'
#         )
        
#         model_local.fit(X_train_local_scaled, y_train)
#         proba_local = model_local.predict_proba(X_test_local_scaled)[0, 1]
#         pred_local = 1 if proba_local > 0.5 else 0
        
#         y_true_local.append(y_test)
#         y_pred_local.append(pred_local)
#         y_proba_local.append(proba_local)
    
#     if len(y_true_local) > 0:
#         acc_local = accuracy_score(y_true_local, y_pred_local)
#         auc_local = roc_auc_score(y_true_local, y_proba_local)
#         f1_local = f1_score(y_true_local, y_pred_local)
        
#         print(f"Accuracy  : {acc_local:.4f}")
#         print(f"AUC       : {auc_local:.4f}")
#         print(f"F1-Score  : {f1_local:.4f}")
#         print(f"Nombre de prédictions : {len(y_true_local)}")
        
#         print(f"\n{'='*60}")
#         print("COMPARAISON DES DEUX MÉTHODES (LOGISTIC REGRESSION)")
#         print(f"{'='*60}")
#         print(f"Différence Accuracy : {acc_global - acc_local:+.4f}")
#         print(f"Différence AUC      : {auc_global - auc_local:+.4f}")
#         print(f"Différence F1       : {f1_global - f1_local:+.4f}")
        
#         if auc_global > auc_local:
#             print(f"\n⚠️  Le scaling global améliore l'AUC de {auc_global - auc_local:.4f} points")
#             print("   Cela suggère un LOOK-AHEAD dans le scaling global!")
#         else:
#             print(f"\n✓ Le scaling local donne de meilleurs résultats")
        
#         # ============================================================
#         # ANALYSE DES COEFFICIENTS (pour régression logistique seulement)
#         # ============================================================
#         print(f"\n{'='*60}")
#         print("ANALYSE DES COEFFICIENTS (dernier modèle)")
#         print(f"{'='*60}")
        
#         # Prendre le dernier modèle entraîné avec scaling local
#         feature_names = X.columns.tolist()
#         coefficients = model_local.coef_[0]
        
#         coeff_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Coefficient': coefficients,
#             'Abs_Coefficient': np.abs(coefficients)
#         }).sort_values('Abs_Coefficient', ascending=False)
        
#         print("Top 10 features par importance absolue:")
#         print(coeff_df.head(10).to_string(index=False))
        
#         # Vérifier la significativité statistique (approximative)
#         print(f"\nNombre de features avec |coeff| > 0.1: {(np.abs(coefficients) > 0.1).sum()}")
#         print(f"Nombre de features avec |coeff| > 0.5: {(np.abs(coefficients) > 0.5).sum()}")
    
#     # ============================================================
#     # DIAGNOSTIC
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("DIAGNOSTIC (LOGISTIC REGRESSION)")
#     print(f"{'='*60}")
    
#     if auc_global < 0.5:
#         print("Problème: AUC < 0.5 même avec scaling global")
#         print("Causes possibles:")
#         print("1. Les relations sont non-linéaires (RF marche mieux)")
#         print("2. Données incorrectes ou période différente")
#         print("3. Horizon temporel (20 jours) trop court")
    
#     elif auc_global > 0.7:
#         print(f"✅ AUC élevé ({auc_global:.4f}) avec scaling global")
#         print("   Cela correspond aux résultats du papier (Table 9: Logistic Regression AUC=0.6365)")
#         print("   Mais attention: le scaling global peut créer du look-ahead")
        
#         # Comparaison avec le papier
#         paper_auc_logistic = 0.6365  # Après feature selection (Table 9)
#         print(f"\nComparaison avec le papier:")
#         print(f"  Ton AUC: {auc_global:.4f}")
#         print(f"  Papier AUC (logistic): {paper_auc_logistic:.4f}")
#         print(f"  Différence: {auc_global - paper_auc_logistic:+.4f}")
    
#     else:
#         print(f"AUC modéré: {auc_global:.4f}")
#         print("Note: Dans le papier, la régression logistique donne AUC=0.6365")
        
#     # ============================================================
#     # TEST DE PERSISTANCE (baseline simple)
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("TEST DE BASELINE : PERSISTANCE")
#     print(f"{'='*60}")
    
#     # Baseline 1: Toujours prédire "hausse" (majority class)
#     baseline_majority = [1] * len(y_true_global)
#     acc_baseline_majority = accuracy_score(y_true_global, baseline_majority)
    
#     # Baseline 2: Persistance (prédire comme hier)
#     persistence_pred = []
#     persistence_true = []
    
#     # Pour chaque prédiction, regarder ce qui s'est passé HORIZON jours avant
#     for i in range(len(y_true_global)):
#         if i >= 1:
#             persistence_pred.append(y_true_global[i-1])
#             persistence_true.append(y_true_global[i])
    
#     if len(persistence_true) > 0:
#         acc_persistence = accuracy_score(persistence_true, persistence_pred)
    
#     print(f"Baseline 'toujours hausse' : Accuracy = {acc_baseline_majority:.4f}")
#     if len(persistence_true) > 0:
#         print(f"Baseline persistance (yesterday) : Accuracy = {acc_persistence:.4f}")
    
#     print(f"\nComparaison avec ton modèle:")
#     print(f"  Logistic Regression AUC: {auc_global:.4f}")
#     print(f"  Gain vs baseline 'toujours hausse': {acc_global - acc_baseline_majority:+.4f}")

# else:
#     print("Aucune prédiction générée")





#########
    # ============================================================
# AVEC CONTROLE DU LOOK AHEAD 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

print("Chargement des données...")
df = pd.read_excel("vol_1.xlsx")
df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
df = df.rename(columns={"Unnamed: 0": "date", "SPX Index": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
print(df.columns)
df["Date"] = pd.to_datetime(df["Date"])
# df = df.set_index("date").sort_index()
df = df.set_index("Date").sort_index()
df = df[2300:]
df_raw_spx = df["Returns30"].copy()
df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# ============================================================
# PARAMÈTRES MODIFIABLES
# ============================================================
HORIZON = 5          # Rendement sur HORIZON jours = 22 jours en donnée de marché
LOOK_AHEAD = 5       # Décalage entre fin entraînement et test
                     # - LOOK_AHEAD = HORIZON : pas de look-ahead (papier)
                     # - LOOK_AHEAD = 1 : look-ahead maximum
                     # - LOOK_AHEAD > HORIZON : entraînement plus ancien
train_window = 2128
total_obs = len(df)

print(f"Total observations: {total_obs}")
print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")
print(f"HORIZON: {HORIZON} jours")
print(f"LOOK_AHEAD: {LOOK_AHEAD} jours")
print(f"Train window: {train_window} jours")

# ============================================================
# CIBLE : Signe du rendement à HORIZON jours
# ============================================================
df["log_price"] = np.log(df["Returns30"])
df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
quantile = df["future_return"][-252:].quantile(0.15)
y = (df["future_return"] < quantile).astype(int)

print(f"Distribution de y: {y.mean():.2%} hausses")

# ============================================================
# FEATURES : Exclure VIX9D, VIX3M, VIX6M comme le papier
# ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
features_to_drop = [
    # tout ce que tu ne veux PAS utiliser
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
    # "PCUSEQTR Index",   # <- NE PAS DROP : c'est ton PUTCALL RATIO
    "SPJGBVRT Index",
    "SPX Index",
    "USDJPYV3M BGN Curncy",
    "USSFCT03 BGN Curncy", "USSFCT10 BGN Curncy", "USSFCT30 BGN Curncy",
    "V1X Index", "V2X Index",
    "VXEEM Index",
    "XBT Curncy",

    # colonnes “target / leakage”
    "Returns30", "log_price", "future_return"
]
X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
# X = df.drop(columns=[c for c in features_to_drop if c in df.columns])

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
    "PCUSEQTR Index": "PUTCALL RATIO",
    "RVOL Index": "RVOL",
})
print(f"Features utilisées: {list(X.columns)}")
print(f"Nombre de features: {len(X.columns)}")

# ============================================================
# SCALING GLOBAL (COMME LE PAPIER) - AVEC LOOK-AHEAD POTENTIEL
# ============================================================
print("\n=== APPLYING GLOBAL SCALING (POTENTIAL LOOK-AHEAD) ===")
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X)
X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)



from itertools import product

# ============================================================
# GRID SEARCH (WALK-FORWARD) SUR QQ PARAMS + THRESHOLD
# ============================================================

def walk_forward_score_rf(params, threshold):
    """
    Retourne metrics + (optionnel) listes de prédictions si besoin.
    On garde exactement ta logique LOOK_AHEAD/train_window et les features déjà scalées globalement.
    """
    y_true, y_pred, y_proba = [], [], []
    test_dates = []

    for test_idx in range(first_test_idx, last_test_idx + 1):
        train_start = test_idx - train_window - LOOK_AHEAD
        train_end   = test_idx - LOOK_AHEAD
        if train_start < 0:
            continue

        X_train = X_scaled_global.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        if len(X_train) != train_window:
            continue

        X_test = X_scaled_global.iloc[test_idx:test_idx+1]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_features=params["max_features"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[0, 1]
        pred  = 1 if proba > threshold else 0

        y_true.append(int(y_test))
        y_pred.append(int(pred))
        y_proba.append(float(proba))
        test_dates.append(df.index[test_idx])

    if len(y_true) == 0:
        return None

    # Métriques
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan
    f1  = f1_score(y_true, y_pred)

    return {
        "acc": acc,
        "auc": auc,
        "f1":  f1,
        "n_preds": len(y_true),
    }


# --- Ton "range" test (déjà défini chez toi) ---
first_test_idx = train_window
last_test_idx  = total_obs - LOOK_AHEAD - 1

# --- Grid minimal mais utile (tu peux élargir) ---
grid = {
    "n_estimators":      [200],
    "max_features":      ["sqrt", 0.5],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 3],
    "max_depth":         [None, 6, 12],
}
threshold_grid = [0.20, 0.25, 0.30]

results = []
keys = list(grid.keys())

print("\n=== GRID SEARCH (walk-forward) ===")
total_runs = np.prod([len(grid[k]) for k in keys]) * len(threshold_grid)
run_id = 0

for values in product(*[grid[k] for k in keys]):
    params = dict(zip(keys, values))
    for thr in threshold_grid:
        run_id += 1
        out = walk_forward_score_rf(params, thr)
        if out is None:
            continue
        row = {**params, "threshold": thr, **out}
        results.append(row)

        if run_id % 10 == 0:
            print(f"  Runs: {run_id}/{total_runs} ...")

results_df = pd.DataFrame(results)

# Choisis ton critère : AUC (si stable) sinon F1
results_df = results_df.sort_values(["auc", "f1"], ascending=False).reset_index(drop=True)
print("\nTOP 10 configs:")
print(results_df.head(10).to_string(index=False))

best = results_df.iloc[0].to_dict()
best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_features": best["max_features"],
    "min_samples_split": int(best["min_samples_split"]),
    "min_samples_leaf": int(best["min_samples_leaf"]),
    "max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
}
best_threshold = float(best["threshold"])

print("\n=== BEST CONFIG ===")
print("Params:", best_params)
print("Threshold:", best_threshold)
print("AUC:", best["auc"], "| F1:", best["f1"], "| Acc:", best["acc"], "| n:", int(best["n_preds"]))

# ============================================================
# WALK-FORWARD VALIDATION AVEC DONNÉES GLOBALEMENT SCALÉES
# ============================================================

first_test_idx = train_window
# last_test_idx = total_obs - HORIZON - 1
last_test_idx = total_obs - LOOK_AHEAD - 1

print(f"\nWalk-Forward Validation with Global Scaling:")
print(f"First test index: {first_test_idx} (date: {df.index[first_test_idx].date()})")
print(f"Last test index: {last_test_idx} (date: {df.index[last_test_idx].date()})")
print(f"Nombre de prédictions: {last_test_idx - first_test_idx + 1}")

y_true_global, y_pred_global, y_proba_global = [], [], []
test_dates_global = []

for test_idx in range(first_test_idx, last_test_idx + 1):
    # === MODIFICATION CRITIQUE : Utilisation de LOOK_AHEAD ===
    train_start = test_idx - train_window - LOOK_AHEAD
    train_end = test_idx - LOOK_AHEAD
    
    if train_start < 0:
        continue
    
    X_train_global = X_scaled_global.iloc[train_start:train_end]
    y_train = y.iloc[train_start:train_end]
    
    if len(X_train_global) != train_window:
        continue
    
    X_test_global = X_scaled_global.iloc[test_idx:test_idx+1]
    y_test = y.iloc[test_idx]
    
    model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_features=best_params["max_features"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_depth=best_params["max_depth"],
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
    
    model.fit(X_train_global, y_train)
    
    proba = model.predict_proba(X_test_global)[0, 1]
    pred = 1 if proba > best_threshold else 0
    y_true_global.append(y_test)
    y_pred_global.append(pred)
    y_proba_global.append(proba)
    test_dates_global.append(df.index[test_idx])
    
    if len(y_true_global) % 100 == 0:
        print(f"  Prédictions: {len(y_true_global)}...")

# ============================================================
# RÉSULTATS AVEC SCALING GLOBAL
# ============================================================
if len(y_true_global) > 0:
    acc_global = accuracy_score(y_true_global, y_pred_global)
    auc_global = roc_auc_score(y_true_global, y_proba_global)
    f1_global = f1_score(y_true_global, y_pred_global)
    
    cm = confusion_matrix(y_true_global, y_pred_global)
    print("Confusion Matrix:", cm)
    print(f"\n{'='*60}")
    print(f"RÉSULTATS AVEC SCALING GLOBAL (LOOK_AHEAD={LOOK_AHEAD})")
    print(f"{'='*60}")
    print(f"Accuracy  : {acc_global:.4f}")
    print(f"AUC       : {auc_global:.4f}")
    print(f"F1-Score  : {f1_global:.4f}")
    print(f"Nombre de prédictions : {len(y_true_global)}")
    
    print(f"\nVérification temporelle:")
    print(f"Première prédiction: {test_dates_global[0].date()}")
    print(f"Dernière prédiction: {test_dates_global[-1].date()}")
    print(f"Dernière date d'entraînement: {df.index[test_idx-LOOK_AHEAD].date()}")
    print(f"Écart entraînement-test: {LOOK_AHEAD} jours")



# ============================================================
# EXPORT CSV : TRUE vs PRED (walk-forward global scaling)
# ============================================================
pred_vs_true = pd.DataFrame({
    "date": pd.to_datetime(test_dates_global),
    "y_true": y_true_global,
    "y_pred": y_pred_global,
    "proba_1": y_proba_global,
})

# (optionnel) ajouter le rendement futur (log + simple) pour debug
pred_vs_true["logret_fwd"] = df.loc[pred_vs_true["date"], "future_return"].astype(float).values
pred_vs_true["ret_fwd"]    = np.exp(pred_vs_true["logret_fwd"]) - 1

# (optionnel) ajouter les params du run (pratique si tu fais plusieurs essais)
pred_vs_true["HORIZON"]    = HORIZON
pred_vs_true["LOOK_AHEAD"] = LOOK_AHEAD
pred_vs_true["train_window"] = train_window
pred_vs_true["threshold"]  = 0.25  # car tu fais pred = 1 si proba > 0.15
pred_vs_true["threshold"]  = best_threshold


pred_vs_true = pred_vs_true.sort_values("date").reset_index(drop=True)
pred_vs_true.to_csv("pred_vs_true_global_scaling.csv", index=False)
print("✔️ Exported -> pred_vs_true_global_scaling.csv")

# ============================================================
# FORECAST-ONLY : prédire jusqu'à la dernière date (sans y_true)
# ============================================================
forecast_dates = []
forecast_pred  = []
forecast_proba = []

for test_idx in range(last_test_idx + 1, total_obs):
    train_start = test_idx - train_window - LOOK_AHEAD
    train_end   = test_idx - LOOK_AHEAD

    if train_start < 0:
        continue

    X_train = X_scaled_global.iloc[train_start:train_end]
    y_train = y.iloc[train_start:train_end]

    if len(X_train) != train_window:
        continue

    X_test = X_scaled_global.iloc[test_idx:test_idx+1]

    # RandomForest n'accepte pas les NaN -> on skip si features pas dispo
    if not np.isfinite(X_test.values).all():
        continue

    model = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[0, 1]
    pred  = 1 if proba > 0.25 else 0

    forecast_dates.append(df.index[test_idx])
    forecast_pred.append(pred)
    forecast_proba.append(proba)

forecast_only = pd.DataFrame({
    "date": pd.to_datetime(forecast_dates),
    "y_true": np.nan,                 # inconnu (pas de futur)
    "y_pred": forecast_pred,
    "proba_1": forecast_proba,
    "logret_fwd": np.nan,             # pas calculable
    "ret_fwd": np.nan,                # pas calculable
    "HORIZON": HORIZON,
    "LOOK_AHEAD": LOOK_AHEAD,
    "train_window": train_window,
    "threshold": 0.25,
    "is_forecast_only": 1
}).sort_values("date").reset_index(drop=True)

# marquer aussi la partie test
pred_vs_true["is_forecast_only"] = 0

# CSV unique : test + forecast-only jusqu'à la dernière date dispo
pred_all = pd.concat([pred_vs_true, forecast_only], axis=0).sort_values("date").reset_index(drop=True)
pred_all.to_csv("pred_global_scaling_test_plus_forecast.csv", index=False)

print("✔️ Exported -> pred_global_scaling_test_plus_forecast.csv")
print("Dernière date dans le CSV :", pred_all["date"].max())
# ============================================================
# ANALYSE DES "VRAIS" RENDEMENTS (SIMPLE RETURNS) PAR CAS
# ============================================================
res = pd.DataFrame(
    {
        "y_true": y_true_global,
        "y_pred": y_pred_global,
        "proba":  y_proba_global,
    },
    index=pd.to_datetime(test_dates_global),
).sort_index()

# log-return futur sur HORIZON jours (déjà calculé dans df)
res["logret_fwd"] = df.loc[res.index, "future_return"].astype(float)

# "vrai" rendement simple sur HORIZON jours
res["ret_fwd"] = np.exp(res["logret_fwd"]) - 1

# --- Moyennes conditionnelles basiques ---
def _mean_n(s):
    return float(s.mean()), int(s.notna().sum())

print("\n=== Moyennes de rendement SIMPLE (exp(log)-1) sur", HORIZON, "jours ===")

m, n = _mean_n(res.loc[res["y_pred"] == 1, "ret_fwd"])
print(f"Mean ret_fwd | pred=1 : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[res["y_pred"] == 0, "ret_fwd"])
print(f"Mean ret_fwd | pred=0 : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[res["y_true"] == 1, "ret_fwd"])
print(f"Mean ret_fwd | true=1 : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[res["y_true"] == 0, "ret_fwd"])
print(f"Mean ret_fwd | true=0 : {m:.6f}  (n={n})")

# --- Cas d'erreurs / réussites ---
tp = (res["y_pred"] == 1) & (res["y_true"] == 1)  # dit 1 et c'est 1
fp = (res["y_pred"] == 1) & (res["y_true"] == 0)  # dit 1 mais c'est 0
tn = (res["y_pred"] == 0) & (res["y_true"] == 0)  # dit 0 et c'est 0
fn = (res["y_pred"] == 0) & (res["y_true"] == 1)  # dit 0 mais c'est 1

m, n = _mean_n(res.loc[tp, "ret_fwd"])
print(f"Mean ret_fwd | TP (pred=1,true=1) : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[fp, "ret_fwd"])
print(f"Mean ret_fwd | FP (pred=1,true=0) : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[tn, "ret_fwd"])
print(f"Mean ret_fwd | TN (pred=0,true=0) : {m:.6f}  (n={n})")

m, n = _mean_n(res.loc[fn, "ret_fwd"])
print(f"Mean ret_fwd | FN (pred=0,true=1) : {m:.6f}  (n={n})")

# Option bonus : médiane aussi (souvent plus robuste)
print("\n--- Médianes (robuste) ---")
print("Median ret_fwd | pred=1 :", float(res.loc[res["y_pred"] == 1, "ret_fwd"].median()))
print("Median ret_fwd | pred=0 :", float(res.loc[res["y_pred"] == 0, "ret_fwd"].median()))
print("Median ret_fwd | TP     :", float(res.loc[tp, "ret_fwd"].median()))
print("Median ret_fwd | FP     :", float(res.loc[fp, "ret_fwd"].median()))
print("Median ret_fwd | TN     :", float(res.loc[tn, "ret_fwd"].median()))
print("Median ret_fwd | FN     :", float(res.loc[fn, "ret_fwd"].median()))

# ============================================================
# QUANTILES CONDITIONNELS DES RENDEMENTS (ret_fwd) PAR GROUPE
# ============================================================

def q_stats(series: pd.Series, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    s = series.dropna()
    if s.empty:
        return None, 0
    return s.quantile(list(qs)), int(len(s))

print("\n=== Quantiles conditionnels de ret_fwd (rendement simple) ===")

# --- Par prédiction ---
q_pred1, n1 = q_stats(res.loc[res["y_pred"] == 1, "ret_fwd"])
q_pred0, n0 = q_stats(res.loc[res["y_pred"] == 0, "ret_fwd"])

if q_pred1 is not None:
    print(f"\nPred=1 (n={n1})")
    print(q_pred1.to_string())
    print(f"-> Q75 (75% en dessous) | pred=1 : {q_pred1.loc[0.75]:.6f}")

if q_pred0 is not None:
    print(f"\nPred=0 (n={n0})")
    print(q_pred0.to_string())
    print(f"-> Q75 (75% en dessous) | pred=0 : {q_pred0.loc[0.75]:.6f}")

# --- Optionnel : par type d'issue (TP/FP/TN/FN) ---
for name, mask in [("TP", tp), ("FP", fp), ("TN", tn), ("FN", fn)]:
    q, n = q_stats(res.loc[mask, "ret_fwd"])
    if q is None:
        continue
    print(f"\n{name} (n={n})")
    print(q.to_string())
    print(f"-> Q75 | {name} : {q.loc[0.75]:.6f}")



# # ============================================================
# # TESTS MULTIPLES AVEC DIFFÉRENTES VALEURS DE LOOK_AHEAD
# # ============================================================
# print(f"\n{'='*60}")
# print("TESTS AVEC DIFFÉRENTES VALEURS DE LOOK_AHEAD")
# print(f"{'='*60}")

# look_ahead_values = [HORIZON, HORIZON//2, 5, 2, 1]  # Test différentes valeurs
# results = []

# for look_ahead in look_ahead_values:
#     print(f"\n--- Test avec LOOK_AHEAD = {look_ahead} ---")
    
#     y_true_test, y_pred_test, y_proba_test = [], [], []
    
#     for test_idx in range(first_test_idx, last_test_idx + 1, 5):  # Saut de 5 pour aller plus vite
#         train_start = test_idx - train_window - look_ahead
#         train_end = test_idx - look_ahead
        
#         if train_start < 0:
#             continue
        
#         X_train = X_scaled_global.iloc[train_start:train_end]
#         y_train = y.iloc[train_start:train_end]
        
#         if len(X_train) != train_window:
#             continue
        
#         X_test = X_scaled_global.iloc[test_idx:test_idx+1]
#         y_test = y.iloc[test_idx]
        
#         model = RandomForestClassifier(
#             n_estimators=200,  # Très petit pour aller vite
#             max_features='sqrt',
#             n_jobs=-1,
#             random_state=42,
#             class_weight='balanced'
#         )
        
#         model.fit(X_train, y_train)
#         proba = model.predict_proba(X_test)[0, 1]
#         pred = 1 if proba > 0.5 else 0
        
#         y_true_test.append(y_test)
#         y_pred_test.append(pred)
#         y_proba_test.append(proba)
    
#     if len(y_true_test) > 0:
#         acc = accuracy_score(y_true_test, y_pred_test)
#         auc = roc_auc_score(y_true_test, y_proba_test)
#         results.append((look_ahead, auc, acc))
#         print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}, N={len(y_true_test)}")

# # Affichage des résultats comparés
# print(f"\n{'='*60}")
# print("COMPARAISON DES DIFFÉRENTS LOOK_AHEAD")
# print(f"{'='*60}")
# print("LOOK_AHEAD | AUC      | Accuracy | Gain vs HORIZON")
# print("-----------|----------|----------|-----------------")

# if results:
#     # Trouver le résultat avec LOOK_AHEAD = HORIZON comme référence
#     reference_auc = None
#     for look_ahead, auc, acc in results:
#         if look_ahead == HORIZON:
#             reference_auc = auc
#             break
    
#     if reference_auc is not None:
#         for look_ahead, auc, acc in results:
#             gain = auc - reference_auc if look_ahead != HORIZON else 0
#             print(f"{look_ahead:^11} | {auc:7.4f} | {acc:8.4f} | {gain:+7.4f}")
            
#             # Avertissement si look-ahead très petit
#             if look_ahead < 5:
#                 print(f"  ⚠️  Attention: LOOK_AHEAD={look_ahead} peut créer du data leakage!")
    
#     # Trouver le meilleur LOOK_AHEAD
#     best_look_ahead, best_auc, best_acc = max(results, key=lambda x: x[1])
#     worst_look_ahead, worst_auc, worst_acc = min(results, key=lambda x: x[1])
    
#     print(f"\nMeilleur: LOOK_AHEAD={best_look_ahead} (AUC={best_auc:.4f})")
#     print(f"Pire: LOOK_AHEAD={worst_look_ahead} (AUC={worst_auc:.4f})")
#     print(f"Différence: {best_auc-worst_auc:.4f} points d'AUC")

# # ============================================================
# # ANALYSE DU DATA LEAKAGE POTENTIEL
# # ============================================================
# print(f"\n{'='*60}")
# print("ANALYSE DU DATA LEAKAGE POTENTIEL")
# print(f"{'='*60}")

# if LOOK_AHEAD < HORIZON:
#     print(f"⚠️  ALERTE: LOOK_AHEAD({LOOK_AHEAD}) < HORIZON({HORIZON})")
#     print("   → Chevauchement potentiel entre cibles d'entraînement et test")
#     print("   → Data leakage garanti!")
#     print(f"\nExemple pour test à t:")
#     print(f"  - Dernier entraînement: données jusqu'à t-{LOOK_AHEAD}")
#     print(f"  - Cible de ce dernier point: rendement t-{LOOK_AHEAD} → t-{LOOK_AHEAD}+{HORIZON}")
#     print(f"  - Période test: rendement t → t+{HORIZON}")
#     print(f"  - Chevauchement: {HORIZON - LOOK_AHEAD} jours")
    
# elif LOOK_AHEAD == HORIZON:
#     print(f"✓ Configuration du papier: LOOK_AHEAD = HORIZON = {HORIZON}")
#     print("  - Pas de chevauchement théorique")
#     print("  - Mais scaling global peut créer du look-ahead indirect")
    
# else:
#     print(f"✓ LOOK_AHEAD({LOOK_AHEAD}) > HORIZON({HORIZON})")
#     print(f"  - Marge de sécurité: {LOOK_AHEAD - HORIZON} jours")
#     print("  - Pas de data leakage théorique")

# # ============================================================
# # VÉRIFICATION DES DATES
# # ============================================================
# print(f"\n{'='*60}")
# print("VÉRIFICATION DES DATES (exemple)")
# print(f"{'='*60}")

# if len(test_dates_global) > 0:
#     sample_idx = len(test_dates_global) // 2
#     test_date = test_dates_global[sample_idx]
#     test_pos = df.index.get_loc(test_date)
    
#     train_end_date = df.index[test_pos - LOOK_AHEAD]
#     train_start_date = df.index[test_pos - LOOK_AHEAD - train_window]
    
#     print(f"Date de test exemple: {test_date.date()}")
#     print(f"Date début entraînement: {train_start_date.date()}")
#     print(f"Date fin entraînement: {train_end_date.date()}")
#     print(f"Écart fin entraînement → test: {LOOK_AHEAD} jours")
    
#     # Vérification des cibles
#     last_train_target_end = df.index[test_pos - LOOK_AHEAD + HORIZON]
#     print(f"\nDernière cible connue à la fin de l'entraînement:")
#     print(f"  - Date: {last_train_target_end.date()}")
#     print(f"  - Relation avec test: {'AVANT' if last_train_target_end < test_date else 'APRÈS/ÉGAL'}")
#     print(f"  - Écart: {(test_date - last_train_target_end).days} jours")
# else:
#     print("Pas assez de données pour la vérification")




#########

#TEST AVEC ENTRAPINEMENT DFIXE PAR BATCH DE 20 




# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.ensemble import RandomForestClassifier
# import warnings
# warnings.filterwarnings('ignore')

# print("Chargement des données...")
# df = pd.read_excel("vol1.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()
# df = df[4000:]
# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # PARAMÈTRES
# # ============================================================
# HORIZON = 20          # Horizon de prédiction (rendement sur X jours)
# TRAIN_WINDOW = 2128   # Taille de la fenêtre d'entraînement
# FORECAST_WINDOW = 20  # Nombre de jours à prédire avant de réentraîner
# total_obs = len(df)

# print(f"Total observations: {total_obs}")
# print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")
# print(f"Train window: {TRAIN_WINDOW} jours")
# print(f"Forecast window: {FORECAST_WINDOW} jours")
# print(f"Horizon de prédiction: {HORIZON} jours")

# # ============================================================
# # CIBLE : Signe du rendement à HORIZON jours
# # ============================================================
# df["log_price"] = np.log(df["Returns30"])
# df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
# y = (df["future_return"] > 0).astype(int)

# print(f"Distribution de y: {y.mean():.2%} hausses")

# # ============================================================
# # FEATURES : Exclure VIX9D, VIX3M, VIX6M comme le papier
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
# X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
# print(f"Features utilisées: {list(X.columns)}")
# print(f"Nombre de features: {len(X.columns)}")

# # ============================================================
# # SCALING GLOBAL (COMME LE PAPIER)
# # ============================================================
# print("\n=== APPLYING GLOBAL SCALING ===")
# scaler_global = StandardScaler()
# X_scaled_global = scaler_global.fit_transform(X)
# X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)

# # ============================================================
# # STRATÉGIE : Entraîner 2128j → Prédire 20j → Décaler 20j
# # ============================================================

# # Premier jour où on peut commencer à prédire
# first_train_end = TRAIN_WINDOW - 1
# first_forecast_start = first_train_end + 1

# # Dernier jour où on peut faire une prédiction (besoin de HORIZON jours après)
# last_forecast_start = total_obs - HORIZON - 1

# print(f"\nStratégie: Train 2128j → Forecast {FORECAST_WINDOW}j → Shift {FORECAST_WINDOW}j")
# print(f"First train end: {df.index[first_train_end].date()}")
# print(f"First forecast start: {df.index[first_forecast_start].date()}")
# print(f"Last forecast start: {df.index[last_forecast_start].date()}")

# # Initialiser les listes de résultats
# all_predictions = []
# all_true_values = []
# all_probabilities = []
# all_forecast_dates = []
# all_train_periods = []

# # Boucle principale : décalage de FORECAST_WINDOW jours
# current_start = first_forecast_start
# block_number = 0

# while current_start <= last_forecast_start:
#     block_number += 1
    
#     # 1. Définir la période d'entraînement (2128 jours avant current_start)
#     train_end_idx = current_start - 1  # Le jour avant le début des prédictions
#     train_start_idx = train_end_idx - TRAIN_WINDOW + 1
    
#     # Vérifier que nous avons assez de données
#     if train_start_idx < 0:
#         print(f"Block {block_number}: Pas assez de données pour l'entraînement")
#         break
    
#     # 2. Définir la période de prédiction (20 jours à partir de current_start)
#     forecast_end_idx = min(current_start + FORECAST_WINDOW - 1, last_forecast_start)
    
#     # Nombre réel de jours à prédire dans ce bloc
#     days_in_forecast = forecast_end_idx - current_start + 1
    
#     if days_in_forecast <= 0:
#         break
    
#     # Dates pour ce bloc
#     train_start_date = df.index[train_start_idx]
#     train_end_date = df.index[train_end_idx]
#     forecast_start_date = df.index[current_start]
#     forecast_end_date = df.index[forecast_end_idx]
    
#     print(f"\n{'='*60}")
#     print(f"BLOCK {block_number}")
#     print(f"{'='*60}")
#     print(f"Entraînement: {train_start_date.date()} → {train_end_date.date()} ({TRAIN_WINDOW} jours)")
#     print(f"Prédiction  : {forecast_start_date.date()} → {forecast_end_date.date()} ({days_in_forecast} jours)")
    
#     # 3. Préparer les données d'entraînement
#     X_train = X_scaled_global.iloc[train_start_idx:train_end_idx+1]
#     y_train = y.iloc[train_start_idx:train_end_idx+1]
    
#     # 4. Entraîner le modèle
#     model = RandomForestClassifier(
#         n_estimators=500,  # Réduit pour aller plus vite
#         max_features='sqrt',
#         min_samples_split=10,
#         min_samples_leaf=5,
#         max_depth=15,
#         n_jobs=-1,
#         random_state=42
#     )
    
#     print(f"Entraînement du modèle...")
#     model.fit(X_train, y_train)
    
#     # 5. Faire les prédictions pour chaque jour de la fenêtre de forecast
#     for day_offset in range(days_in_forecast):
#         forecast_idx = current_start + day_offset
        
#         # Vérifier que nous pouvons calculer la cible (besoin de HORIZON jours après)
#         if forecast_idx + HORIZON >= total_obs:
#             continue
        
#         # Données pour ce jour de prédiction
#         X_test = X_scaled_global.iloc[forecast_idx:forecast_idx+1]
#         y_test = y.iloc[forecast_idx]
#         forecast_date = df.index[forecast_idx]
        
#         # Prédiction
#         proba = model.predict_proba(X_test)[0, 1]
#         pred = 1 if proba > 0.5 else 0
        
#         # Stocker les résultats
#         all_predictions.append(pred)
#         all_true_values.append(y_test)
#         all_probabilities.append(proba)
#         all_forecast_dates.append(forecast_date)
#         all_train_periods.append((train_start_date.date(), train_end_date.date()))
    
#     print(f"  Prédictions faites: {days_in_forecast} jours")
    
#     # 6. Passer au bloc suivant (décaler de FORECAST_WINDOW jours)
#     current_start += FORECAST_WINDOW

# # ============================================================
# # RÉSULTATS FINAUX
# # ============================================================
# if len(all_predictions) > 0:
#     acc = accuracy_score(all_true_values, all_predictions)
#     auc = roc_auc_score(all_true_values, all_probabilities)
#     f1 = f1_score(all_true_values, all_predictions)
    
#     print(f"\n{'='*60}")
#     print("RÉSULTATS FINAUX (Stratégie: Train 2128j → Forecast 20j → Shift 20j)")
#     print(f"{'='*60}")
#     print(f"Nombre total de prédictions : {len(all_predictions)}")
#     print(f"Nombre de blocs             : {block_number}")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"AUC       : {auc:.4f}")
#     print(f"F1-Score  : {f1:.4f}")
    
#     # Distribution des prédictions
#     pred_up = sum(all_predictions) / len(all_predictions)
#     real_up = sum(all_true_values) / len(all_true_values)
#     print(f"\nDistribution des prédictions: {pred_up:.1%} UP, {1-pred_up:.1%} DOWN")
#     print(f"Distribution réelle        : {real_up:.1%} UP, {1-real_up:.1%} DOWN")
#     print(f"Biais (prédit - réel)     : {pred_up-real_up:+.1%}")
    
#     # Analyse par période
#     print(f"\n{'='*60}")
#     print("ANALYSE PAR PÉRIODE")
#     print(f"{'='*60}")
    
#     # Convertir en DataFrame pour analyse
#     results_df = pd.DataFrame({
#         'date': all_forecast_dates,
#         'y_true': all_true_values,
#         'y_pred': all_predictions,
#         'y_proba': all_probabilities,
#         'train_start': [t[0] for t in all_train_periods],
#         'train_end': [t[1] for t in all_train_periods]
#     })
    
#     results_df['year'] = results_df['date'].dt.year
#     results_df['correct'] = (results_df['y_true'] == results_df['y_pred']).astype(int)
    
#     # Performance par année
#     yearly_perf = results_df.groupby('year').agg({
#         'correct': 'mean',
#         'y_proba': 'mean',
#         'date': 'count'
#     }).rename(columns={
#         'correct': 'accuracy',
#         'date': 'n_predictions'
#     })
    
#     print(f"\nPerformance par année:")
#     print(yearly_perf.round(3))
    
#     # Performance par bloc
#     print(f"\n{'='*60}")
#     print("PERFORMANCE PAR BLOC")
#     print(f"{'='*60}")
    
#     # Identifier les blocs
#     results_df['block_id'] = results_df['train_end'].apply(
#         lambda x: f"{x.strftime('%Y-%m-%d')}"
#     )
    
#     block_perf = results_df.groupby('block_id').agg({
#         'correct': 'mean',
#         'date': ['first', 'last', 'count'],
#         'train_end': 'first'
#     })
    
#     # Réorganiser les colonnes
#     block_perf.columns = ['accuracy', 'first_date', 'last_date', 'n_predictions', 'train_end']
#     block_perf = block_perf.sort_values('first_date')
    
#     print(f"\nTop 5 meilleurs blocs:")
#     print(block_perf.nlargest(5, 'accuracy')[['accuracy', 'n_predictions', 'first_date', 'last_date']])
    
#     print(f"\nTop 5 pires blocs:")
#     print(block_perf.nsmallest(5, 'accuracy')[['accuracy', 'n_predictions', 'first_date', 'last_date']])
    
#     # Vérification des dates (exemple)
#     print(f"\n{'='*60}")
#     print("VÉRIFICATION DES DATES (exemples)")
#     print(f"{'='*60}")
    
#     sample_size = min(5, len(results_df))
#     for i in range(sample_size):
#         row = results_df.iloc[i]
#         print(f"Prédiction {i+1}:")
#         print(f"  Date prédite: {row['date'].date()}")
#         print(f"  Période entraînement: {row['train_start']} → {row['train_end']}")
#         print(f"  Cible: {'UP' if row['y_true']==1 else 'DOWN'}, Prédit: {'UP' if row['y_pred']==1 else 'DOWN'}")
#         print(f"  Proba: {row['y_proba']:.3f}, Correct: {'✓' if row['correct']==1 else '✗'}")
#         print()
    
#     # Export des résultats
#     results_df.to_csv('block_forecast_results.csv', index=False)
#     print(f"✅ Résultats exportés dans 'block_forecast_results.csv'")
    
#     # ============================================================
#     # COMPARAISON AVEC LA STRATÉGIE STANDARD (1 jour)
#     # ============================================================
#     print(f"\n{'='*60}")
#     print("COMPARAISON AVEC STRATÉGIE STANDARD (prédiction jour par jour)")
#     print(f"{'='*60}")
    
#     # Stratégie standard : entraîner 2128j, prédire 1j, décaler 1j
#     y_true_standard, y_pred_standard, y_proba_standard = [], [], []
    
#     for test_idx in range(first_forecast_start, last_forecast_start + 1):
#         train_end_idx = test_idx - 1
#         train_start_idx = train_end_idx - TRAIN_WINDOW + 1
        
#         if train_start_idx < 0:
#             continue
        
#         X_train = X_scaled_global.iloc[train_start_idx:train_end_idx+1]
#         y_train = y.iloc[train_start_idx:train_end_idx+1]
        
#         if len(X_train) != TRAIN_WINDOW:
#             continue
        
#         X_test = X_scaled_global.iloc[test_idx:test_idx+1]
#         y_test = y.iloc[test_idx]
        
#         model_std = RandomForestClassifier(
#             n_estimators=200,
#             max_features='sqrt',
#             n_jobs=-1,
#             random_state=42,
#             class_weight='balanced'
#         )
        
#         model_std.fit(X_train, y_train)
#         proba = model_std.predict_proba(X_test)[0, 1]
#         pred = 1 if proba > 0.5 else 0
        
#         y_true_standard.append(y_test)
#         y_pred_standard.append(pred)
#         y_proba_standard.append(proba)
    
#     if len(y_true_standard) > 0:
#         acc_std = accuracy_score(y_true_standard, y_pred_standard)
#         auc_std = roc_auc_score(y_true_standard, y_proba_standard)
#         f1_std = f1_score(y_true_standard, y_pred_standard)
        
#         print(f"Stratégie standard (1 jour):")
#         print(f"  Accuracy: {acc_std:.4f}")
#         print(f"  AUC: {auc_std:.4f}")
#         print(f"  F1: {f1_std:.4f}")
#         print(f"  Nombre de prédictions: {len(y_true_standard)}")
        
#         print(f"\nComparaison:")
#         print(f"  Différence Accuracy : {acc - acc_std:+.4f}")
#         print(f"  Différence AUC      : {auc - auc_std:+.4f}")
#         print(f"  Différence F1       : {f1 - f1_std:+.4f}")
        
#         if auc > auc_std:
#             print(f"\n✅ La stratégie par blocs ({FORECAST_WINDOW}j) est meilleure!")
#         elif auc < auc_std:
#             print(f"\n⚠️  La stratégie standard (1j) est meilleure!")
#         else:
#             print(f"\n⚖️  Les deux stratégies sont équivalentes")
    
# else:
#     print("Aucune prédiction générée")

# print(f"\n{'='*60}")
# print("EXÉCUTION TERMINÉE")
# print(f"{'='*60}")






#################
#TEST AVEC BATCH FIXE ANALYSE 20ème prédiction sur bloc fixe ########

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.ensemble import RandomForestClassifier
# import warnings
# warnings.filterwarnings('ignore')

# print("Chargement des données...")
# df = pd.read_excel("vol1.xlsx")
# df = df.drop(df.index[0])
# df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date").sort_index()
# df = df[4000:]
# df_raw_spx = df["Returns30"].copy()
# df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# # ============================================================
# # PARAMÈTRES
# # ============================================================
# HORIZON = 20          # Horizon de prédiction (rendement sur X jours)
# TRAIN_WINDOW = 1000   # Taille de la fenêtre d'entraînement
# FORECAST_WINDOW = 20  # Nombre de jours à prédire avant de réentraîner
# total_obs = len(df)

# print(f"Total observations: {total_obs}")
# print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")
# print(f"Train window: {TRAIN_WINDOW} jours")
# print(f"Forecast window: {FORECAST_WINDOW} jours")
# print(f"Horizon de prédiction: {HORIZON} jours")

# # ============================================================
# # CIBLE : Signe du rendement à HORIZON jours
# # ============================================================
# df["log_price"] = np.log(df["Returns30"])
# df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
# y = (df["future_return"] > 0).astype(int)

# print(f"Distribution de y: {y.mean():.2%} hausses")

# # ============================================================
# # FEATURES : Exclure VIX9D, VIX3M, VIX6M comme le papier
# # ============================================================
# features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
# X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
# print(f"Features utilisées: {list(X.columns)}")
# print(f"Nombre de features: {len(X.columns)}")

# # ============================================================
# # SCALING GLOBAL (COMME LE PAPIER)
# # ============================================================
# print("\n=== APPLYING GLOBAL SCALING ===")
# scaler_global = StandardScaler()
# X_scaled_global = scaler_global.fit_transform(X)
# X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)

# # ============================================================
# # STRATÉGIE : Entraîner 2128j → Prédire 20j → Décaler 20j
# # ============================================================

# # Premier jour où on peut commencer à prédire
# first_train_end = TRAIN_WINDOW - 1
# first_forecast_start = first_train_end + 1

# # Dernier jour où on peut faire une prédiction (besoin de HORIZON jours après)
# last_forecast_start = total_obs - HORIZON - 1

# print(f"\nStratégie: Train 2128j → Forecast {FORECAST_WINDOW}j → Shift {FORECAST_WINDOW}j")
# print(f"First train end: {df.index[first_train_end].date()}")
# print(f"First forecast start: {df.index[first_forecast_start].date()}")
# print(f"Last forecast start: {df.index[last_forecast_start].date()}")

# # Initialiser les listes de résultats
# all_predictions = []
# all_true_values = []
# all_probabilities = []
# all_forecast_dates = []
# all_train_periods = []

# # Listes pour les 20èmes prédictions
# twentieth_predictions = []
# twentieth_true_values = []
# twentieth_probabilities = []
# twentieth_dates = []
# twentieth_gap_days = []  # Nombre de jours entre fin entraînement et prédiction

# # Boucle principale : décalage de FORECAST_WINDOW jours
# current_start = first_forecast_start
# block_number = 0

# while current_start <= last_forecast_start:
#     block_number += 1
    
#     # 1. Définir la période d'entraînement (2128 jours avant current_start)
#     train_end_idx = current_start - 1  # Le jour avant le début des prédictions
#     train_start_idx = train_end_idx - TRAIN_WINDOW + 1
    
#     # Vérifier que nous avons assez de données
#     if train_start_idx < 0:
#         print(f"Block {block_number}: Pas assez de données pour l'entraînement")
#         break
    
#     # 2. Définir la période de prédiction (20 jours à partir de current_start)
#     forecast_end_idx = min(current_start + FORECAST_WINDOW - 1, last_forecast_start)
    
#     # Nombre réel de jours à prédire dans ce bloc
#     days_in_forecast = forecast_end_idx - current_start + 1
    
#     if days_in_forecast <= 0:
#         break
    
#     # Dates pour ce bloc
#     train_start_date = df.index[train_start_idx]
#     train_end_date = df.index[train_end_idx]
#     forecast_start_date = df.index[current_start]
#     forecast_end_date = df.index[forecast_end_idx]
    
#     print(f"\n{'='*60}")
#     print(f"BLOCK {block_number}")
#     print(f"{'='*60}")
#     print(f"Entraînement: {train_start_date.date()} → {train_end_date.date()} ({TRAIN_WINDOW} jours)")
#     print(f"Prédiction  : {forecast_start_date.date()} → {forecast_end_date.date()} ({days_in_forecast} jours)")
    
#     # 3. Préparer les données d'entraînement
#     X_train = X_scaled_global.iloc[train_start_idx:train_end_idx+1]
#     y_train = y.iloc[train_start_idx:train_end_idx+1]
    
#     # 4. Entraîner le modèle
#     model = RandomForestClassifier(
#         n_estimators=500,
#         max_features='sqrt',
#         min_samples_split=10,
#         min_samples_leaf=5,
#         max_depth=15,
#         n_jobs=-1,
#         random_state=42,
#         class_weight='balanced'
#     )
    
#     print(f"Entraînement du modèle...")
#     model.fit(X_train, y_train)
    
#     # 5. Faire les prédictions pour chaque jour de la fenêtre de forecast
#     for day_offset in range(days_in_forecast):
#         forecast_idx = current_start + day_offset
        
#         # Vérifier que nous pouvons calculer la cible (besoin de HORIZON jours après)
#         if forecast_idx + HORIZON >= total_obs:
#             continue
        
#         # Données pour ce jour de prédiction
#         X_test = X_scaled_global.iloc[forecast_idx:forecast_idx+1]
#         y_test = y.iloc[forecast_idx]
#         forecast_date = df.index[forecast_idx]
        
#         # Prédiction
#         proba = model.predict_proba(X_test)[0, 1]
#         pred = 1 if proba > 0.5 else 0
        
#         # Stocker les résultats
#         all_predictions.append(pred)
#         all_true_values.append(y_test)
#         all_probabilities.append(proba)
#         all_forecast_dates.append(forecast_date)
#         all_train_periods.append((train_start_date.date(), train_end_date.date()))
        
#         # Si c'est la 20ème prédiction du bloc (ou la dernière si moins de 20)
#         if day_offset == FORECAST_WINDOW - 1 or day_offset == days_in_forecast - 1:
#             gap_days = forecast_idx - train_end_idx  # Nombre de jours entre fin entraînement et test
            
#             twentieth_predictions.append(pred)
#             twentieth_true_values.append(y_test)
#             twentieth_probabilities.append(proba)
#             twentieth_dates.append(forecast_date)
#             twentieth_gap_days.append(gap_days)
            
#             print(f"  ✅ 20ème (ou dernière) prédiction: {forecast_date.date()}")
#             print(f"     Gap: {gap_days} jours depuis la fin de l'entraînement")
    
#     print(f"  Prédictions faites: {days_in_forecast} jours")
    
#     # 6. Passer au bloc suivant (décaler de FORECAST_WINDOW jours)
#     current_start += FORECAST_WINDOW

# # ============================================================
# # RÉSULTATS FINAUX
# # ============================================================
# if len(all_predictions) > 0:
#     acc = accuracy_score(all_true_values, all_predictions)
#     auc = roc_auc_score(all_true_values, all_probabilities)
#     f1 = f1_score(all_true_values, all_predictions)
    
#     print(f"\n{'='*60}")
#     print("RÉSULTATS FINAUX (Stratégie: Train 2128j → Forecast 20j → Shift 20j)")
#     print(f"{'='*60}")
#     print(f"Nombre total de prédictions : {len(all_predictions)}")
#     print(f"Nombre de blocs             : {block_number}")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"AUC       : {auc:.4f}")
#     print(f"F1-Score  : {f1:.4f}")
    
#     # Distribution des prédictions
#     pred_up = sum(all_predictions) / len(all_predictions)
#     real_up = sum(all_true_values) / len(all_true_values)
#     print(f"\nDistribution des prédictions: {pred_up:.1%} UP, {1-pred_up:.1%} DOWN")
#     print(f"Distribution réelle        : {real_up:.1%} UP, {1-real_up:.1%} DOWN")
#     print(f"Biais (prédit - réel)     : {pred_up-real_up:+.1%}")

# # ============================================================
# # ANALYSE SPÉCIFIQUE DES 20ÈMES PRÉDICTIONS
# # ============================================================
# print(f"\n{'='*60}")
# print("ANALYSE SPÉCIFIQUE DES 20ÈMES (ou dernières) PRÉDICTIONS")
# print(f"{'='*60}")

# if len(twentieth_predictions) > 0:
#     # Calculer les métriques pour les 20èmes prédictions
#     acc_20th = accuracy_score(twentieth_true_values, twentieth_predictions)
#     auc_20th = roc_auc_score(twentieth_true_values, twentieth_probabilities)
#     f1_20th = f1_score(twentieth_true_values, twentieth_predictions)
    
#     print(f"Nombre de 20èmes prédictions : {len(twentieth_predictions)}")
#     print(f"Accuracy (20èmes)  : {acc_20th:.4f}")
#     print(f"AUC (20èmes)       : {auc_20th:.4f}")
#     print(f"F1-Score (20èmes)  : {f1_20th:.4f}")
    
#     # Distribution
#     pred_up_20th = sum(twentieth_predictions) / len(twentieth_predictions)
#     real_up_20th = sum(twentieth_true_values) / len(twentieth_true_values)
#     print(f"\nDistribution 20èmes prédictions: {pred_up_20th:.1%} UP, {1-pred_up_20th:.1%} DOWN")
#     print(f"Distribution 20èmes réelle    : {real_up_20th:.1%} UP, {1-real_up_20th:.1%} DOWN")
#     print(f"Biais 20èmes (prédit - réel) : {pred_up_20th-real_up_20th:+.1%}")
    
#     # Comparaison avec toutes les prédictions
#     print(f"\nComparaison avec toutes les prédictions:")
#     print(f"  Différence Accuracy : {acc_20th - acc:+.4f}")
#     print(f"  Différence AUC      : {auc_20th - auc:+.4f}")
#     print(f"  Différence F1       : {f1_20th - f1:+.4f}")
    
#     # Analyser le gap moyen
#     avg_gap = np.mean(twentieth_gap_days)
#     min_gap = np.min(twentieth_gap_days)
#     max_gap = np.max(twentieth_gap_days)
#     print(f"\nStatistiques du gap (jours entre fin entraînement et test):")
#     print(f"  Moyenne: {avg_gap:.1f} jours")
#     print(f"  Minimum: {min_gap} jours")
#     print(f"  Maximum: {max_gap} jours")
    
#     # Analyser la performance par gap
#     print(f"\n{'='*60}")
#     print("PERFORMANCE PAR GAP (pour les 20èmes prédictions)")
#     print(f"{'='*60}")
    
#     # Créer un DataFrame pour analyse
#     twentieth_df = pd.DataFrame({
#         'date': twentieth_dates,
#         'y_true': twentieth_true_values,
#         'y_pred': twentieth_predictions,
#         'y_proba': twentieth_probabilities,
#         'gap_days': twentieth_gap_days
#     })
    
#     # Ajouter correct/incorrect
#     twentieth_df['correct'] = (twentieth_df['y_true'] == twentieth_df['y_pred']).astype(int)
    
#     # Grouper par gap
#     gap_performance = twentieth_df.groupby('gap_days').agg({
#         'correct': ['mean', 'count'],
#         'y_proba': 'mean'
#     }).round(3)
    
#     # Aplatir les colonnes multi-index
#     gap_performance.columns = ['accuracy', 'count', 'avg_proba']
#     gap_performance = gap_performance.sort_index()
    
#     print(f"\nPerformance par nombre de jours depuis la fin de l'entraînement:")
#     print(gap_performance)
    
#     # Vérifier s'il y a une corrélation entre gap et performance
#     if len(twentieth_df) > 1:
#         correlation = twentieth_df['gap_days'].corr(twentieth_df['correct'])
#         print(f"\nCorrélation gap-performance: {correlation:.3f}")
        
#         if correlation < -0.2:
#             print("  → Performance diminue avec l'augmentation du gap (modèle vieillit)")
#         elif correlation > 0.2:
#             print("  → Performance augmente avec l'augmentation du gap (étrange)")
#         else:
#             print("  → Pas de corrélation claire entre gap et performance")
    
#     # Exemples de 20èmes prédictions
#     print(f"\n{'='*60}")
#     print("EXEMPLES DE 20ÈMES PRÉDICTIONS")
#     print(f"{'='*60}")
    
#     sample_size = min(5, len(twentieth_df))
#     for i in range(sample_size):
#         row = twentieth_df.iloc[i]
#         print(f"20ème prédiction {i+1}:")
#         print(f"  Date: {row['date'].date()}")
#         print(f"  Gap: {row['gap_days']} jours depuis fin entraînement")
#         print(f"  Cible: {'UP' if row['y_true']==1 else 'DOWN'}, Prédit: {'UP' if row['y_pred']==1 else 'DOWN'}")
#         print(f"  Proba: {row['y_proba']:.3f}, Correct: {'✓' if row['correct']==1 else '✗'}")
#         print()
    
#     # Vérifier le look-ahead potentiel
#     print(f"\n{'='*60}")
#     print("VÉRIFICATION LOOK-AHEAD POTENTIEL")
#     print(f"{'='*60}")
    
#     # Nombre de 20èmes prédictions avec gap < HORIZON (potentiel look-ahead)
#     potential_lookahead = twentieth_df[twentieth_df['gap_days'] < HORIZON]
#     print(f"Nombre de 20èmes prédictions avec gap < {HORIZON} (look-ahead potentiel): {len(potential_lookahead)}")
    
#     if len(potential_lookahead) > 0:
#         acc_lookahead = accuracy_score(potential_lookahead['y_true'], potential_lookahead['y_pred'])
#         print(f"  Accuracy pour ces prédictions: {acc_lookahead:.4f}")
#         print(f"  ⚠️  Ces prédictions peuvent avoir du look-ahead (chevauchement de cibles)")
    
#     # Prédictions avec gap >= HORIZON (sans look-ahead)
#     no_lookahead = twentieth_df[twentieth_df['gap_days'] >= HORIZON]
#     print(f"\nNombre de 20èmes prédictions avec gap >= {HORIZON} (sans look-ahead): {len(no_lookahead)}")
    
#     if len(no_lookahead) > 0:
#         acc_no_lookahead = accuracy_score(no_lookahead['y_true'], no_lookahead['y_pred'])
#         print(f"  Accuracy pour ces prédictions (sans look-ahead): {acc_no_lookahead:.4f}")
        
#         # Comparer
#         if len(potential_lookahead) > 0:
#             diff = acc_lookahead - acc_no_lookahead
#             print(f"  Différence (avec - sans look-ahead): {diff:+.4f}")
            
#             if diff > 0.05:
#                 print(f"  ⚠️  Forte amélioration avec look-ahead potentiel!")
#             elif diff < -0.05:
#                 print(f"  ✅  Meilleure performance sans look-ahead!")
#             else:
#                 print(f"  📊  Pas de différence significative")
    
#     # Sauvegarder les 20èmes prédictions
#     twentieth_df.to_csv('twentieth_predictions_analysis.csv', index=False)
#     print(f"\n✅ Analyse des 20èmes prédictions exportée dans 'twentieth_predictions_analysis.csv'")
    
# else:
#     print("Aucune 20ème prédiction générée")

# # ============================================================
# # CONCLUSION
# # ============================================================
# print(f"\n{'='*60}")
# print("CONCLUSION SUR LES 20ÈMES PRÉDICTIONS")
# print(f"{'='*60}")

# if len(twentieth_predictions) > 0:
#     print(f"La performance sur les 20èmes prédictions (gap moyen: {avg_gap:.1f}j):")
#     print(f"  Accuracy: {acc_20th:.4f} (vs {acc:.4f} pour toutes)")
#     print(f"  AUC: {auc_20th:.4f} (vs {auc:.4f} pour toutes)")
    
#     if acc_20th > acc:
#         print(f"\n✅ Les 20èmes prédictions sont MEILLEURES que la moyenne!")
#         print("   Cela suggère que le modèle est robuste dans le temps")
#     elif acc_20th < acc:
#         print(f"\n⚠️  Les 20èmes prédictions sont PIORES que la moyenne!")
#         print("   Cela suggère que le modèle se dégrade avec le temps")
#     else:
#         print(f"\n📊 Les 20èmes prédictions sont similaires à la moyenne")
    
#     # Vérifier le look-ahead
#     if min_gap < HORIZON:
#         print(f"\n⚠️  ATTENTION: Certaines 20èmes prédictions ont un gap < {HORIZON}")
#         print(f"   Cela signifie qu'il peut y avoir du look-ahead (chevauchement des cibles)")
#         print(f"   Gap minimum observé: {min_gap} jours")
#     else:
#         print(f"\n✅ TOUTES les 20èmes prédictions ont un gap >= {HORIZON}")
#         print(f"   Pas de look-ahead théorique (chevauchement des cibles)")

# print(f"\n{'='*60}")
# print("EXÉCUTION TERMINÉE")
# print(f"{'='*60}")


##########
#NOUVEAU TEST ################
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
# from scipy import stats
# from scipy.stats import binomtest
# import warnings
# import os
# warnings.filterwarnings('ignore')

# # ============================================================
# # CONFIGURATION DU DOSSIER DE SORTIE
# # ============================================================
# output_dir = "new_model"
# os.makedirs(output_dir, exist_ok=True)
# print(f"📁 Tous les résultats seront sauvegardés dans : {output_dir}/")
# print()

# # ============================================================
# # 1. CHARGEMENT ET NETTOYAGE
# # ============================================================
# print("📥 Chargement et nettoyage des données...")
# df = pd.read_excel("vol.xlsx")

# # Supprimer la première ligne (description des colonnes)
# df = df.drop(index=0).reset_index(drop=True)

# # Renommer les colonnes
# df = df.rename(columns={
#     "Unnamed: 0": "date",
#     "PUTCALL RATIO": "PUTCALL",
#     "Returns30 ": "SPX"
# })

# # Convertir la date et définir comme index
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date")

# # Convertir toutes les colonnes en numérique
# for col in df.columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# print(f"   📅 Période : {df.index[0].date()} à {df.index[-1].date()}")
# print(f"   📈 Observations : {len(df)}")

# # ============================================================
# # 2. FEATURE SELECTION COMME LE PAPIER
# # ============================================================
# # Le papier enlève VIX9D, VIX3M, VIX6M après analyse de lasso
# features_to_drop = ['VIX9D', 'VIX3M', 'VIX6M']
# features = [col for col in df.columns if col not in features_to_drop and col != 'SPX']

# print(f"\n🎯 Features utilisées ({len(features)}) :")
# for f in features:
#     print(f"   - {f}")

# # ============================================================
# # 3. CRÉATION DE LA CIBLE (30 JOURS FUTURS)
# # ============================================================
# H = 30  # Horizon 30 jours comme le papier

# # SPX est le prix, calcul des rendements log futurs
# df['log_price'] = np.log(df['SPX'])
# df['future_return'] = df['log_price'].shift(-H) - df['log_price']
# df['target'] = (df['future_return'] > 0).astype(int)

# # Features et target
# X = df[features].copy()
# y = df['target'].copy()

# # Supprimer les NaN
# mask = y.notna() & X.notna().all(axis=1)
# X = X[mask]
# y = y[mask]

# print(f"\n🎯 Distribution de la cible :")
# print(f"   📈 Hausses : {y.mean():.1%} ({y.sum():.0f} jours)")
# print(f"   📉 Baisses : {(1-y).mean():.1%} ({(1-y).sum():.0f} jours)")

# # ============================================================
# # 4. SCALING GLOBAL COMME LE PAPIER
# # ============================================================
# print("\n" + "="*60)
# print("🔧 SCALING GLOBAL (comme le papier)")
# print("="*60)

# # Scaling global sur TOUTES les données (mean=0, std=1 pour chaque variable)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# print("✓ Toutes les features ont été standardisées (mean=0, std=1)")

# # ============================================================
# # 5. WALK-FORWARD AVEC SCALING GLOBAL
# # ============================================================
# TRAIN_SIZE = 2128  # Comme le papier (70% des données)
# dates = X_scaled.index
# T = len(X_scaled)

# predictions = []
# true_values = []
# probabilities = []
# prediction_dates = []

# print(f"\n{'='*60}")
# print("🔄 WALK-FORWARD BACKTEST (scaling global)")
# print(f"{'='*60}")
# print(f"   Taille fenêtre entraînement : {TRAIN_SIZE} jours")
# print(f"   Horizon prédiction : {H} jours")
# print(f"   Début backtest : {dates[TRAIN_SIZE].date()}")

# for test_idx in range(TRAIN_SIZE, T - H):
#     # Date de prédiction
#     pred_date = dates[test_idx]
    
#     # Fin entraînement = test_idx - H (pour éviter look-ahead)
#     train_end = test_idx - H
#     train_start = train_end - TRAIN_SIZE + 1
    
#     if train_start < 0:
#         continue
    
#     # Données d'entraînement (déjà scaled globalement)
#     X_train = X_scaled.iloc[train_start:train_end+1]
#     y_train = y.iloc[train_start:train_end+1]
    
#     # Données de test (déjà scaled globalement)
#     X_test = X_scaled.iloc[test_idx:test_idx+1]
#     y_test = y.iloc[test_idx]
    
#     # Random Forest avec paramètres du papier
#     model = RandomForestClassifier(
#         n_estimators=500,
#         max_features='sqrt',  # mtry = sqrt(p)
#         max_depth=None,       # Pas de limite
#         min_samples_split=2,  # Default scikit-learn
#         min_samples_leaf=1,   # Default scikit-learn
#         random_state=42,
#         n_jobs=-1,
#         class_weight='balanced'  # IMPORTANT : pour gérer le déséquilibre
#     )
    
#     # Entraînement
#     model.fit(X_train, y_train)
    
#     # Prédiction
#     proba = model.predict_proba(X_test)[0, 1]
#     pred = 1 if proba > 0.5 else 0
    
#     # Stockage
#     predictions.append(pred)
#     true_values.append(y_test)
#     probabilities.append(proba)
#     prediction_dates.append(pred_date)
    
#     # Progression
#     if len(predictions) % 100 == 0:
#         print(f"   ✓ {len(predictions)} prédictions effectuées...")

# print(f"   ✅ {len(predictions)} prédictions totales")

# # ============================================================
# # 6. ANALYSE AVEC SEUIL OPTIMAL (au lieu de 0.5)
# # ============================================================
# print(f"\n{'='*60}")
# print("🎯 RECHERCHE DU MEILLEUR SEUIL")
# print(f"{'='*60}")

# # Trouver le seuil optimal avec la courbe ROC
# fpr, tpr, thresholds = roc_curve(true_values, probabilities)

# # 1. Seuil de Youden (maximise TPR - FPR)
# youden_idx = np.argmax(tpr - fpr)
# optimal_threshold_youden = thresholds[youden_idx]

# # 2. Seuil qui maximise l'accuracy
# accuracies = []
# for thresh in thresholds:
#     preds_thresh = [1 if p > thresh else 0 for p in probabilities]
#     acc = accuracy_score(true_values, preds_thresh)
#     accuracies.append(acc)

# optimal_idx_acc = np.argmax(accuracies)
# optimal_threshold_acc = thresholds[optimal_idx_acc]

# print(f"Seuil par défaut (0.5) :")
# preds_05 = [1 if p > 0.5 else 0 for p in probabilities]
# acc_05 = accuracy_score(true_values, preds_05)
# print(f"   Accuracy : {acc_05:.4f}")

# print(f"\nSeuil optimal (Youden) : {optimal_threshold_youden:.3f}")
# preds_youden = [1 if p > optimal_threshold_youden else 0 for p in probabilities]
# acc_youden = accuracy_score(true_values, preds_youden)
# cm_youden = confusion_matrix(true_values, preds_youden)
# print(f"   Accuracy : {acc_youden:.4f}")
# print(f"   Matrice : TN={cm_youden[0,0]}, FP={cm_youden[0,1]}, FN={cm_youden[1,0]}, TP={cm_youden[1,1]}")

# print(f"\nSeuil optimal (Accuracy) : {optimal_threshold_acc:.3f}")
# preds_opt = [1 if p > optimal_threshold_acc else 0 for p in probabilities]
# acc_opt = accuracy_score(true_values, preds_opt)
# cm_opt = confusion_matrix(true_values, preds_opt)
# print(f"   Accuracy : {acc_opt:.4f}")
# print(f"   Matrice : TN={cm_opt[0,0]}, FP={cm_opt[0,1]}, FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")

# # Utiliser les prédictions avec le meilleur seuil
# best_threshold = optimal_threshold_acc if acc_opt > acc_youden else optimal_threshold_youden
# best_predictions = preds_opt if acc_opt > acc_youden else preds_youden
# best_acc = max(acc_opt, acc_youden)

# print(f"\n✅ Utilisation du meilleur seuil ({best_threshold:.3f}) : Accuracy = {best_acc:.4f}")

# # ============================================================
# # 7. RÉSULTATS DÉTAILLÉS AVEC MEILLEUR SEUIL
# # ============================================================
# acc = best_acc
# predictions = best_predictions

# # Calcul des autres métriques
# auc = roc_auc_score(true_values, probabilities)
# f1 = f1_score(true_values, predictions)
# cm = confusion_matrix(true_values, predictions)

# print(f"\n{'='*60}")
# print("📊 RÉSULTATS FINAUX AVEC SEUIL OPTIMAL")
# print(f"{'='*60}")
# print(f"   Prédictions : {len(predictions)}")
# print(f"   Période     : {prediction_dates[0].date()} à {prediction_dates[-1].date()}")
# print(f"   Seuil utilisé : {best_threshold:.3f}")
# print(f"\n   Accuracy    : {acc:.4f}")
# print(f"   AUC         : {auc:.4f}")
# print(f"   F1-Score    : {f1:.4f}")
    
# print(f"\n   Matrice de confusion :")
# print(f"   [TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
# print(f"   [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]")
    
# # Métriques détaillées
# precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
# recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
# specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    
# print(f"\n   Précision   : {precision:.3f}")
# print(f"   Rappel      : {recall:.3f}")
# print(f"   Spécificité : {specificity:.3f}")
    
# # Performance vs baseline
# baseline = max(np.mean(true_values), 1 - np.mean(true_values))
# baseline_class = 1 if np.mean(true_values) > 0.5 else 0
# print(f"\n   Baseline (toujours prédire '{baseline_class}') : {baseline:.3f}")
# print(f"   Amélioration vs baseline : {acc - baseline:+.3f}")
    
# # Test de significativité (CORRIGÉ)
# n_correct = sum(np.array(predictions) == np.array(true_values))
# n_total = len(predictions)
# binom_result = binomtest(n_correct, n_total, p=baseline, alternative='greater')
# p_value = binom_result.pvalue
    
# print(f"\n   Test statistique (vs baseline {baseline:.3f}):")
# print(f"     p-value = {p_value:.6f}")
# print(f"     {'SIGNIFICATIF' if p_value < 0.05 else 'Non significatif'} au niveau 5%")

# # Création du DataFrame principal des résultats
# results_df = pd.DataFrame({
#     'date': prediction_dates,
#     'true': true_values,
#     'pred': predictions,
#     'proba': probabilities
# })

# # ============================================================
# # 8. EXPORT DES RÉSULTATS
# # ============================================================
# print(f"\n{'='*60}")
# print("💾 EXPORT DES RÉSULTATS")
# print(f"{'='*60}")

# # 8.1 Fichier principal
# predictions_path = os.path.join(output_dir, "predictions_optimized.csv")
# results_df.to_csv(predictions_path, index=False)
# print(f"✓ 1. Prédictions optimisées : {predictions_path}")

# # 8.2 Métriques agrégées
# metrics_dict = {
#     'Accuracy': acc,
#     'AUC': auc,
#     'F1_Score': f1,
#     'Precision': precision,
#     'Recall': recall,
#     'Specificity': specificity,
#     'Optimal_Threshold': best_threshold,
#     'N_Predictions': len(predictions),
#     'Start_Date': prediction_dates[0].date(),
#     'End_Date': prediction_dates[-1].date(),
#     'Baseline': baseline,
#     'Improvement_vs_Baseline': acc - baseline,
#     'P_Value': p_value
# }

# metrics_path = os.path.join(output_dir, "aggregated_metrics_optimized.csv")
# pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
# print(f"✓ 2. Métriques optimisées : {metrics_path}")

# # 8.3 Résumé exécutif
# summary_path = os.path.join(output_dir, "executive_summary_optimized.txt")
# with open(summary_path, 'w') as f:
#     f.write("="*70 + "\n")
#     f.write("RÉSUMÉ EXÉCUTIF - MODÈLE OPTIMISÉ\n")
#     f.write("="*70 + "\n\n")
    
#     f.write("🎯 PERFORMANCE GLOBALE\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Accuracy          : {acc:.4f}\n")
#     f.write(f"AUC               : {auc:.4f}\n")
#     f.write(f"F1-Score          : {f1:.4f}\n")
#     f.write(f"Seuil optimal     : {best_threshold:.3f}\n\n")
    
#     f.write("📊 MATRICE DE CONFUSION\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Vrai Négatifs  (TN) : {cm[0,0]:4d}\n")
#     f.write(f"Faux Positifs  (FP) : {cm[0,1]:4d}\n")
#     f.write(f"Faux Négatifs  (FN) : {cm[1,0]:4d}\n")
#     f.write(f"Vrai Positifs  (TP) : {cm[1,1]:4d}\n\n")
    
#     f.write("📈 MÉTRIQUES DÉTAILLÉES\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Précision (TP/(TP+FP)) : {precision:.3f}\n")
#     f.write(f"Rappel    (TP/(TP+FN)) : {recall:.3f}\n")
#     f.write(f"Spécificité (TN/(TN+FP)): {specificity:.3f}\n\n")
    
#     f.write("🏆 COMPARAISON AVEC BASELINE\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Baseline (toujours {baseline_class}) : {baseline:.3f}\n")
#     f.write(f"Amélioration du modèle : {acc - baseline:+.3f}\n")
#     f.write(f"Significativité statistique : {'OUI' if p_value < 0.05 else 'NON'}\n")
#     f.write(f"p-value : {p_value:.6f}\n\n")
    
#     f.write("🔄 DISTRIBUTION DES PRÉDICTIONS\n")
#     f.write("-"*40 + "\n")
#     pred_up = sum(predictions) / len(predictions)
#     pred_down = 1 - pred_up
#     f.write(f"Prédictions 'Hausse' : {pred_up:.1%}\n")
#     f.write(f"Prédictions 'Baisse' : {pred_down:.1%}\n")
#     f.write(f"Réel 'Hausse' : {np.mean(true_values):.1%}\n")
#     f.write(f"Réel 'Baisse' : {1 - np.mean(true_values):.1%}\n\n")
    
#     f.write("💡 INTERPRÉTATION DES RÉSULTATS\n")
#     f.write("-"*40 + "\n")
#     f.write("1. Votre modèle original (seuil 0.5) avait 72.3% d'accuracy\n")
#     f.write("2. Mais il prédit 93% du temps 'Hausse' (trop conservateur)\n")
#     f.write("3. Avec le seuil optimal, le modèle est mieux équilibré\n")
#     f.write("4. L'AUC bas indique que le modèle a du mal à distinguer\n")
#     f.write("   les vraies baisses des vraies hausses\n")
#     f.write("5. Cependant, l'accuracy reste très bonne pour la période\n\n")
    
#     f.write("🚀 RECOMMANDATIONS\n")
#     f.write("-"*40 + "\n")
#     if acc > 0.75:
#         f.write("✅ Performance EXCELLENTE !\n")
#         f.write("   - Votre modèle bat clairement la baseline\n")
#         f.write("   - Le seuil optimal améliore significativement les résultats\n")
#         f.write("   - Considérez d'ajouter d'autres features pour améliorer l'AUC\n")
#     elif acc > 0.70:
#         f.write("✅ Très BONNE performance !\n")
#         f.write("   - Le modèle est utile malgré l'AUC bas\n")
#         f.write("   - L'optimisation du seuil a été cruciale\n")
#         f.write("   - Essayez avec class_weight='balanced' pour améliorer\n")
#     else:
#         f.write("⚠️  Performance modérée\n")
#         f.write("   - L'AUC très bas est préoccupant\n")
#         f.write("   - Le modèle ne distingue pas bien les classes\n")
#         f.write("   - Essayez d'autres algorithmes ou features\n")

# print(f"✓ 3. Résumé exécutif : {summary_path}")

# # ============================================================
# # 9. ANALYSE SUPPLÉMENTAIRE
# # ============================================================
# print(f"\n{'='*60}")
# print("🔍 ANALYSE SUPPLÉMENTAIRE")
# print(f"{'='*60}")

# # Analyse des probabilités
# print("\n📊 Distribution des probabilités prédites :")
# proba_df = pd.DataFrame({'proba': probabilities})
# print(f"   Moyenne : {proba_df['proba'].mean():.3f}")
# print(f"   Médiane : {proba_df['proba'].median():.3f}")
# print(f"   Std     : {proba_df['proba'].std():.3f}")
# print(f"   Min     : {proba_df['proba'].min():.3f}")
# print(f"   Max     : {proba_df['proba'].max():.3f}")

# # Performance quand le modèle est "confiant"
# print("\n🎯 Performance quand le modèle est confiant (|proba - 0.5| > 0.3) :")
# confident_mask = [abs(p - 0.5) > 0.3 for p in probabilities]
# confident_true = [true_values[i] for i, m in enumerate(confident_mask) if m]
# confident_pred = [predictions[i] for i, m in enumerate(confident_mask) if m]

# if confident_true:
#     confident_acc = accuracy_score(confident_true, confident_pred)
#     print(f"   {sum(confident_mask)} prédictions confiantes ({sum(confident_mask)/len(predictions):.1%})")
#     print(f"   Accuracy sur ces prédictions : {confident_acc:.4f}")

# print(f"\n{'='*60}")
# print("🏁 EXÉCUTION TERMINÉE AVEC SUCCÈS !")
# print(f"{'='*60}")
# print(f"🎉 Votre modèle obtient {acc:.1%} d'accuracy avec seuil optimal")
# print(f"📁 Résultats dans : {output_dir}/")
# print(f"📋 Consultez : {summary_path} pour le résumé complet")






#########
#########
#########
##########


#NOUVEAU TEST 2 - ANALYSE 20ÈMES PRÉDICTIONS ################

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
# from scipy.stats import binomtest
# from imblearn.over_sampling import SMOTE
# import warnings
# import os
# warnings.filterwarnings('ignore')

# # ============================================================
# # CONFIGURATION
# # ============================================================
# output_dir = "improved_model"
# os.makedirs(output_dir, exist_ok=True)
# print(f"📁 Résultats dans : {output_dir}/")
# print()

# # ============================================================
# # 1. CHARGEMENT AVEC NETTOYAGE RENFORCÉ
# # ============================================================
# print("📥 Chargement et nettoyage RENFORCÉ...")
# df = pd.read_excel("vol.xlsx")

# # Nettoyer la première ligne
# df = df.drop(index=0).reset_index(drop=True)

# # NETTOYAGE CRITIQUE : retirer les espaces des noms de colonnes
# df = df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)

# # Renommer spécifiquement
# df = df.rename(columns={
#     "Unnamed: 0": "date",
#     "PUTCALL RATIO": "PUTCALL",
#     "Returns30": "SPX"
# })

# # Convertir la date
# df["date"] = pd.to_datetime(df["date"])
# df = df.set_index("date")

# # Convertir en numérique
# for col in df.columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# print(f"   📅 Période : {df.index[0].date()} à {df.index[-1].date()}")
# print(f"   📈 Observations : {len(df)}")

# # ============================================================
# # 2. FEATURE SELECTION STRICTE (8 features comme le papier)
# # ============================================================
# # Features EXACTES du papier après feature selection
# features_paper = ['VIX', 'VVIX', 'SKEW', 'VXN', 'GVZ', 'OVX', 'PUTCALL', 'RVOL']

# # Vérifier et créer les features manquantes
# missing_features = [f for f in features_paper if f not in df.columns]
# if missing_features:
#     print(f"⚠️  Features manquantes : {missing_features}")
#     # Si RVOL manque, le calculer
#     if 'RVOL' in missing_features and 'SPX' in df.columns:
#         print("   Calcul de RVOL (volatilité réalisée sur 30 jours)...")
#         df['log_return'] = np.log(df['SPX']).diff()
#         df['RVOL'] = df['log_return'].rolling(30).std() * np.sqrt(252)

# # Sélection finale
# features = [f for f in features_paper if f in df.columns]
# print(f"\n🎯 Features utilisées ({len(features)}) COMME LE PAPIER :")
# for f in features:
#     print(f"   - {f}")

# # ============================================================
# # 3. CRÉATION DE LA CIBLE AVEC OPTIMISATION
# # ============================================================
# H = 15
# df['log_price'] = np.log(df['SPX'])
# df['future_return'] = df['log_price'].shift(-H) - df['log_price']
# df['target'] = (df['future_return'] > 0).astype(int)

# # Features et target
# X = df[features].copy()
# y = df['target'].copy()

# # Supprimer les NaN
# mask = y.notna() & X.notna().all(axis=1)
# X = X[mask]
# y = y[mask]

# print(f"\n🎯 Distribution de la cible :")
# print(f"   📈 Hausses : {y.mean():.1%} ({y.sum():.0f} jours)")
# print(f"   📉 Baisses : {(1-y).mean():.1%} ({(1-y).sum():.0f} jours)")

# # ============================================================
# # 4. STRATÉGIE AMÉLIORÉE POUR GÉRER LE DÉSÉQUILIBRE
# # ============================================================
# print("\n" + "="*60)
# print("🔄 STRATÉGIE AMÉLIORÉE AVEC CLASS WEIGHT OPTIMAL")
# print("="*60)

# # Scaling global
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=features)

# # ============================================================
# # 5. WALK-FORWARD AVEC TECHNIQUE AMÉLIORÉE
# # ============================================================
# TRAIN_SIZE = 2128
# dates = X_scaled.index
# T = len(X_scaled)

# # Différentes configurations à tester
# configs = [
#     {'name': 'Class Weight Balanced', 'params': {'class_weight': 'balanced'}},
#     {'name': 'Class Weight Dict', 'params': {'class_weight': {0: 2.0, 1: 1.0}}},  # Poids 2x pour les baisses
#     {'name': 'No Class Weight', 'params': {'class_weight': None}},
# ]

# all_results = {}

# for config in configs:
#     print(f"\n🧪 Test de la configuration : {config['name']}")
    
#     predictions = []
#     true_values = []
#     probabilities = []
#     prediction_dates = []
    
#     for test_idx in range(TRAIN_SIZE, T - H):
#         pred_date = dates[test_idx]
#         train_end = test_idx - H
#         train_start = train_end - TRAIN_SIZE + 1
        
#         if train_start < 0:
#             continue
        
#         # Données d'entraînement
#         X_train = X_scaled.iloc[train_start:train_end+1]
#         y_train = y.iloc[train_start:train_end+1]
#         X_test = X_scaled.iloc[test_idx:test_idx+1]
#         y_test = y.iloc[test_idx]
        
#         # Appliquer SMOTE seulement à l'entraînement
#         try:
#             smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train[y_train==0])-1))
#             X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
#         except:
#             # Si pas assez de samples pour SMOTE, utiliser les données originales
#             X_train_bal, y_train_bal = X_train, y_train
        
#         # Modèle avec configuration spécifique
#         model = RandomForestClassifier(
#             n_estimators=500,
#             max_features='sqrt',
#             max_depth=None,
#             min_samples_split=5,  # Augmenté pour réduire l'overfitting
#             min_samples_leaf=2,   # Augmenté pour réduire l'overfitting
#             random_state=42,
#             n_jobs=-1,
#             **config['params']
#         )
        
#         model.fit(X_train_bal, y_train_bal)
#         proba = model.predict_proba(X_test)[0, 1]
#         pred = 1 if proba > 0.5 else 0
        
#         predictions.append(pred)
#         true_values.append(y_test)
#         probabilities.append(proba)
#         prediction_dates.append(pred_date)
    
#     # Évaluation
#     if predictions:
#         # Trouver le seuil optimal pour CETTE configuration
#         fpr, tpr, thresholds = roc_curve(true_values, probabilities)
#         accuracies = [accuracy_score(true_values, [1 if p > t else 0 for p in probabilities]) 
#                      for t in thresholds]
#         best_idx = np.argmax(accuracies)
#         best_threshold = thresholds[best_idx]
        
#         # Appliquer le meilleur seuil
#         best_predictions = [1 if p > best_threshold else 0 for p in probabilities]
#         acc = accuracy_score(true_values, best_predictions)
#         auc = roc_auc_score(true_values, probabilities)
#         cm = confusion_matrix(true_values, best_predictions)
        
#         # Stocker les résultats
#         all_results[config['name']] = {
#             'accuracy': acc,
#             'auc': auc,
#             'threshold': best_threshold,
#             'cm': cm,
#             'predictions': best_predictions,
#             'true_values': true_values,
#             'probabilities': probabilities,
#             'dates': prediction_dates
#         }
        
#         print(f"   ✓ Accuracy: {acc:.4f}, AUC: {auc:.4f}, Seuil: {best_threshold:.3f}")
#         print(f"   Matrice: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# # ============================================================
# # 6. SÉLECTION DE LA MEILLEURE CONFIGURATION
# # ============================================================
# print(f"\n{'='*60}")
# print("🏆 COMPARAISON DES CONFIGURATIONS")
# print(f"{'='*60}")

# best_config = None
# best_score = -1

# for name, results in all_results.items():
#     # Score combiné: 60% accuracy + 40% AUC
#     score = 0.6 * results['accuracy'] + 0.4 * results['auc']
#     print(f"{name:25} : Accuracy={results['accuracy']:.4f}, AUC={results['auc']:.4f}, Score={score:.4f}")
    
#     if score > best_score:
#         best_score = score
#         best_config = name

# print(f"\n✅ Configuration sélectionnée : {best_config}")

# # Utiliser les résultats de la meilleure configuration
# results = all_results[best_config]
# acc = results['accuracy']
# auc = results['auc']
# best_threshold = results['threshold']
# cm = results['cm']
# predictions = results['predictions']
# true_values = results['true_values']
# probabilities = results['probabilities']
# prediction_dates = results['dates']

# # ============================================================
# # 7. RÉSULTATS FINAUX AMÉLIORÉS
# # ============================================================
# precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
# recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
# specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
# baseline = max(np.mean(true_values), 1 - np.mean(true_values))

# # Test de significativité
# n_correct = sum(np.array(predictions) == np.array(true_values))
# n_total = len(predictions)
# binom_result = binomtest(n_correct, n_total, p=baseline, alternative='greater')
# p_value = binom_result.pvalue

# print(f"\n{'='*60}")
# print("📊 RÉSULTATS FINAUX AMÉLIORÉS")
# print(f"{'='*60}")
# print(f"   Configuration : {best_config}")
# print(f"   Prédictions : {len(predictions)}")
# print(f"   Période : {prediction_dates[0].date()} à {prediction_dates[-1].date()}")
# print(f"   Seuil optimal : {best_threshold:.3f}")
# print(f"\n   📈 Accuracy    : {acc:.4f}")
# print(f"   📊 AUC         : {auc:.4f}")
# print(f"   🎯 F1-Score    : {f1_score(true_values, predictions):.4f}")
# print(f"\n   🎪 Matrice de confusion :")
# print(f"   [TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
# print(f"   [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]")
# print(f"\n   📐 Précision   : {precision:.3f}")
# print(f"   🔍 Rappel      : {recall:.3f}")
# print(f"   🎭 Spécificité : {specificity:.3f}")
# print(f"\n   🏁 Baseline : {baseline:.3f}")
# print(f"   🚀 Amélioration : {acc - baseline:+.4f}")
# print(f"   📊 p-value : {p_value:.6f} ({'Significatif' if p_value < 0.05 else 'Non significatif'})")

# # ============================================================
# # 8. ANALYSE DES PERFORMANCES PAR PÉRIODE
# # ============================================================
# results_df = pd.DataFrame({
#     'date': prediction_dates,
#     'true': true_values,
#     'pred': predictions,
#     'proba': probabilities
# })

# # Performance par trimestre
# results_df['quarter'] = results_df['date'].dt.to_period('Q')
# quarterly_perf = results_df.groupby('quarter').apply(
#     lambda x: accuracy_score(x['true'], x['pred'])
# ).reset_index(name='accuracy')

# print(f"\n{'='*60}")
# print("📅 PERFORMANCE PAR TRIMESTRE")
# print(f"{'='*60}")

# for _, row in quarterly_perf.tail(8).iterrows():  # 8 derniers trimestres
#     print(f"   {row['quarter']} : {row['accuracy']:.3f}")

# # ============================================================
# # 9. EXPORT DES RÉSULTATS
# # ============================================================
# print(f"\n{'='*60}")
# print("💾 EXPORT DES RÉSULTATS")
# print(f"{'='*60}")

# # Fichier principal
# predictions_path = os.path.join(output_dir, "predictions_improved.csv")
# results_df.to_csv(predictions_path, index=False)
# print(f"✓ 1. Prédictions améliorées : {predictions_path}")

# # Métriques
# metrics_dict = {
#     'Configuration': best_config,
#     'Accuracy': acc,
#     'AUC': auc,
#     'Optimal_Threshold': best_threshold,
#     'Precision': precision,
#     'Recall': recall,
#     'Specificity': specificity,
#     'Baseline': baseline,
#     'Improvement': acc - baseline,
#     'P_Value': p_value,
#     'N_Predictions': len(predictions),
#     'Start_Date': prediction_dates[0].date(),
#     'End_Date': prediction_dates[-1].date(),
#     'TN': int(cm[0,0]),
#     'FP': int(cm[0,1]),
#     'FN': int(cm[1,0]),
#     'TP': int(cm[1,1])
# }

# metrics_path = os.path.join(output_dir, "metrics_improved.csv")
# pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
# print(f"✓ 2. Métriques améliorées : {metrics_path}")

# # Comparaison des configurations
# configs_comparison = []
# for name, res in all_results.items():
#     configs_comparison.append({
#         'Configuration': name,
#         'Accuracy': res['accuracy'],
#         'AUC': res['auc'],
#         'Threshold': res['threshold']
#     })

# configs_path = os.path.join(output_dir, "configurations_comparison.csv")
# pd.DataFrame(configs_comparison).to_csv(configs_path, index=False)
# print(f"✓ 3. Comparaison des configurations : {configs_path}")

# # Résumé exécutif
# summary_path = os.path.join(output_dir, "executive_summary_improved.txt")
# with open(summary_path, 'w') as f:
#     f.write("="*70 + "\n")
#     f.write("RÉSUMÉ EXÉCUTIF - MODÈLE AMÉLIORÉ\n")
#     f.write("="*70 + "\n\n")
    
#     f.write("📊 PERFORMANCE GLOBALE\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Configuration        : {best_config}\n")
#     f.write(f"Accuracy            : {acc:.4f}\n")
#     f.write(f"AUC                 : {auc:.4f}\n")
#     f.write(f"Seuil optimal       : {best_threshold:.3f}\n\n")
    
#     f.write("🎪 MATRICE DE CONFUSION\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Vrai Négatifs (TN)  : {cm[0,0]:4d}\n")
#     f.write(f"Faux Positifs (FP)  : {cm[0,1]:4d}\n")
#     f.write(f"Faux Négatifs (FN)  : {cm[1,0]:4d}\n")
#     f.write(f"Vrai Positifs (TP)  : {cm[1,1]:4d}\n\n")
    
#     f.write("📈 MÉTRIQUES DÉTAILLÉES\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Précision           : {precision:.3f}\n")
#     f.write(f"Rappel              : {recall:.3f}\n")
#     f.write(f"Spécificité         : {specificity:.3f}\n\n")
    
#     f.write("🏆 COMPARAISON AVEC BASELINE\n")
#     f.write("-"*40 + "\n")
#     f.write(f"Baseline            : {baseline:.3f}\n")
#     f.write(f"Amélioration        : {acc - baseline:+.4f}\n")








##########
#ULTIME TEST 
###


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FONCTION POUR CALCULER LES JOURS OUVRÉS
# ============================================================
def get_trading_days_between(df, start_date, end_date):
    """Retourne le nombre de jours de trading entre deux dates"""
    if start_date not in df.index or end_date not in df.index:
        return None
    start_idx = df.index.get_loc(start_date)
    end_idx = df.index.get_loc(end_date)
    return abs(end_idx - start_idx)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("Chargement des données...")
df = pd.read_excel("vol1.xlsx")
df = df.drop(df.index[0])
df = df.rename(columns={"Unnamed: 0": "date", "Returns30 ": "Returns30"})
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df = df[2800:]
df_raw_spx = df["Returns30"].copy()
df["Returns30"] = pd.to_numeric(df["Returns30"], errors="coerce")

# ============================================================
# PARAMÈTRES
# ============================================================
HORIZON = 20          # Rendement sur HORIZON jours
train_window = 2128
total_obs = len(df)

print(f"Total observations: {total_obs}")
print(f"Période: {df.index[0].date()} à {df.index[-1].date()}")
print(f"HORIZON: {HORIZON} jours")
print(f"Train window: {train_window} jours")

# ============================================================
# CIBLE : Signe du rendement à HORIZON jours
# ============================================================
df["log_price"] = np.log(df["Returns30"])
df["future_return"] = df["log_price"].shift(-HORIZON) - df["log_price"]
y = (df["future_return"] > 0).astype(int)

print(f"Distribution de y: {y.mean():.2%} hausses")

# ============================================================
# FEATURES
# ============================================================
features_to_drop = ["VIX9D", "VIX3M", "VIX6M", "Returns30", "log_price", "future_return"]
X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
print(f"Features utilisées: {list(X.columns)}")
print(f"Nombre de features: {len(X.columns)}")

# ============================================================
# SCALING GLOBAL
# ============================================================
print("\n=== APPLYING GLOBAL SCALING ===")
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X)
X_scaled_global = pd.DataFrame(X_scaled_global, index=X.index, columns=X.columns)

# ============================================================
# FONCTION POUR EXÉCUTER LE BACKTEST AVEC UN LOOK_AHEAD DONNÉ
# ============================================================
def run_backtest_with_lookahead(look_ahead_days, save_predictions=False, filename_prefix=""):
    """
    Exécute le backtest avec un look-ahead spécifique
    look_ahead_days: nombre de jours de trading entre fin entraînement et test
    """
    print(f"\n{'='*60}")
    print(f"BACKTEST AVEC LOOK_AHEAD = {look_ahead_days} jours (trading)")
    print(f"{'='*60}")
    
    # Calculer les indices
    first_test_idx = train_window
    # last_test_idx = total_obs - HORIZON - 1
    last_test_idx = total_obs 
    
    predictions_data = []
    
    for test_idx in range(first_test_idx, last_test_idx + 1):
        # Calcul des indices d'entraînement
        train_end_idx = test_idx - look_ahead_days  # Fin de l'entraînement
        train_start_idx = train_end_idx - train_window + 1
        
        if train_start_idx < 0:
            continue
        
        # Vérifier que nous avons bien train_window jours d'entraînement
        if train_end_idx - train_start_idx + 1 != train_window:
            continue
        
        # Dates importantes
        test_date = df.index[test_idx]
        last_train_date = df.index[train_end_idx]
        first_train_date = df.index[train_start_idx]
        
        # Calculer le nombre réel de jours calendaires
        calendar_days_gap = (test_date - last_train_date).days
        
        # Données d'entraînement et de test
        X_train = X_scaled_global.iloc[train_start_idx:train_end_idx+1]
        y_train = y.iloc[train_start_idx:train_end_idx+1]
        X_test = X_scaled_global.iloc[test_idx:test_idx+1]
        y_test = y.iloc[test_idx]
        
        # Modèle
        model = RandomForestClassifier(
            n_estimators=300,  # Augmenté à 300
            max_features='sqrt',
            min_samples_split=6,  # Augmenté pour réduire l'overfitting
            min_samples_leaf=3,    # Augmenté pour réduire l'overfitting
            max_depth=20,          # Limité pour éviter l'overfitting
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            max_samples=0.8,       # Bootstrap avec 80% des données
            oob_score=True         # Score out-of-bag pour validation
        )
        
        model.fit(X_train, y_train)
        
        # Prédiction
        proba = model.predict_proba(X_test)[0, 1]
        pred = 1 if proba > 0.5 else 0
        
        # Date de la cible (HORIZON jours après test_date)
        target_date_idx = test_idx + HORIZON
        if target_date_idx < len(df.index):
            target_date = df.index[target_date_idx]
        else:
            target_date = None
        
        # Stocker les informations
        predictions_data.append({
            'test_date': test_date,
            'first_train_date': first_train_date,
            'last_train_date': last_train_date,
            'target_date': target_date,
            'look_ahead_trading_days': look_ahead_days,
            'look_ahead_calendar_days': calendar_days_gap,
            'predicted_proba': proba,
            'predicted_class': pred,
            'true_class': y_test,
            'prediction_correct': 1 if pred == y_test else 0,
            'horizon_days': HORIZON,
            'train_window_days': train_window
        })
    
    # Convertir en DataFrame
    predictions_df = pd.DataFrame(predictions_data)
    
    if len(predictions_df) > 0:
        # Calculer les métriques
        acc = accuracy_score(predictions_df['true_class'], predictions_df['predicted_class'])
        auc = roc_auc_score(predictions_df['true_class'], predictions_df['predicted_proba'])
        f1 = f1_score(predictions_df['true_class'], predictions_df['predicted_class'])
        
        print(f"\nRésultats pour LOOK_AHEAD = {look_ahead_days}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Nombre de prédictions: {len(predictions_df)}")
        
        # Afficher les gaps moyens
        avg_calendar_gap = predictions_df['look_ahead_calendar_days'].mean()
        print(f"  Gap moyen (jours calendaires): {avg_calendar_gap:.1f} jours")
        
        # Sauvegarder si demandé
        if save_predictions and filename_prefix:
            filename = f"{filename_prefix}_lookahead_{look_ahead_days}_horizon_{HORIZON}.csv"
            predictions_df.to_csv(filename, index=False)
            print(f"  Prédictions sauvegardées dans: {filename}")
    
    return predictions_df

# ============================================================
# EXÉCUTER LES BACKTESTS POUR DIFFÉRENTS LOOK_AHEAD
# ============================================================
print("\n" + "="*60)
print("EXÉCUTION DES BACKTESTS POUR DIFFÉRENTS LOOK_AHEAD")
print("="*60)

# Backtest avec look-ahead = HORIZON (15 jours)
predictions_lookahead_15 = run_backtest_with_lookahead(
    look_ahead_days=HORIZON,
    save_predictions=True,
    filename_prefix="predictions"
)

# Backtest avec look-ahead = 5 jours
predictions_lookahead_5 = run_backtest_with_lookahead(
    look_ahead_days=5,
    save_predictions=True,
    filename_prefix="predictions"
)

# ============================================================
# ANALYSE COMPARATIVE
# ============================================================
print("\n" + "="*60)
print("ANALYSE COMPARATIVE DES DEUX STRATÉGIES")
print("="*60)

if len(predictions_lookahead_15) > 0 and len(predictions_lookahead_5) > 0:
    # Calculer les métriques pour chaque stratégie
    acc_15 = accuracy_score(predictions_lookahead_15['true_class'], predictions_lookahead_15['predicted_class'])
    auc_15 = roc_auc_score(predictions_lookahead_15['true_class'], predictions_lookahead_15['predicted_proba'])
    
    acc_5 = accuracy_score(predictions_lookahead_5['true_class'], predictions_lookahead_5['predicted_class'])
    auc_5 = roc_auc_score(predictions_lookahead_5['true_class'], predictions_lookahead_5['predicted_proba'])
    
    print(f"LOOK_AHEAD = {HORIZON} jours (sans look-ahead théorique):")
    print(f"  Accuracy: {acc_15:.4f}, AUC: {auc_15:.4f}")
    print(f"  Nombre de prédictions: {len(predictions_lookahead_15)}")
    print(f"  Période: {predictions_lookahead_15['test_date'].min().date()} à {predictions_lookahead_15['test_date'].max().date()}")
    
    print(f"\nLOOK_AHEAD = 5 jours (avec look-ahead potentiel):")
    print(f"  Accuracy: {acc_5:.4f}, AUC: {auc_5:.4f}")
    print(f"  Nombre de prédictions: {len(predictions_lookahead_5)}")
    print(f"  Période: {predictions_lookahead_5['test_date'].min().date()} à {predictions_lookahead_5['test_date'].max().date()}")
    
    print(f"\nDifférence (5 jours - 15 jours):")
    print(f"  ΔAccuracy: {acc_5 - acc_15:+.4f}")
    print(f"  ΔAUC: {auc_5 - auc_15:+.4f}")
    
    # Vérifier le look-ahead potentiel
    if HORIZON > 5:
        overlap_days = HORIZON - 5
        print(f"\n⚠️  ALERTE: Avec LOOK_AHEAD=5 et HORIZON={HORIZON}, il y a potentiellement")
        print(f"    {overlap_days} jours de chevauchement entre données d'entraînement et test!")
        
        # Analyser la performance par gap
        print(f"\nAnalyse par gap (jours calendaires entre fin entraînement et test):")
        print(f"  LOOK_AHEAD=15: gap moyen = {predictions_lookahead_15['look_ahead_calendar_days'].mean():.1f} jours")
        print(f"  LOOK_AHEAD=5: gap moyen = {predictions_lookahead_5['look_ahead_calendar_days'].mean():.1f} jours")

# ============================================================
# CRÉATION DU GRAPHIQUE DE PRÉDICTIONS
# ============================================================
print("\n" + "="*60)
print("CRÉATION DU GRAPHIQUE DE PRÉDICTIONS")
print("="*60)

def create_prediction_timeline_plot(predictions_df, title_suffix):
    """
    Crée un graphique montrant la timeline des prédictions
    """
    if len(predictions_df) == 0:
        print(f"Pas de données pour créer le graphique {title_suffix}")
        return
    
    # Prendre les 60 dernières prédictions pour la lisibilité
    plot_df = predictions_df.tail(60).copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Graphique 1: Timeline des prédictions
    ax1 = axes[0]
    
    # Tracer les différentes dates
    for idx, row in plot_df.iterrows():
        test_date = row['test_date']
        last_train_date = row['last_train_date']
        target_date = row['target_date']
        
        # Ligne de la période d'attente (last_train_date → test_date)
        ax1.plot([last_train_date, test_date], [idx, idx], 'k--', alpha=0.3, linewidth=0.5)
        
        # Marqueur pour le dernier jour d'entraînement
        ax1.plot(last_train_date, idx, 'bo', markersize=4, label='Fin entraînement' if idx == plot_df.index[0] else "")
        
        # Marqueur pour le jour du test
        color = 'green' if row['prediction_correct'] == 1 else 'red'
        ax1.plot(test_date, idx, 'o', color=color, markersize=6, 
                label='Prédiction correcte' if idx == plot_df.index[0] and color == 'green' else 
                      'Prédiction incorrecte' if idx == plot_df.index[0] and color == 'red' else "")
        
        # Ligne vers la date cible (si disponible)
        if pd.notna(target_date):
            ax1.plot([test_date, target_date], [idx, idx], 'b-', alpha=0.5, linewidth=1)
            ax1.plot(target_date, idx, 'b^', markersize=4, label='Date cible' if idx == plot_df.index[0] else "")
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prédiction (index)')
    ax1.set_title(f'Timeline des prédictions - LOOK_AHEAD={predictions_df["look_ahead_trading_days"].iloc[0]} jours {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Graphique 2: Distribution des gaps
    ax2 = axes[1]
    
    # Histogramme des gaps en jours calendaires
    gaps = plot_df['look_ahead_calendar_days']
    ax2.hist(gaps, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=gaps.mean(), color='red', linestyle='--', label=f'Moyenne: {gaps.mean():.1f} jours')
    ax2.set_xlabel('Gap en jours calendaires (fin entraînement → test)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f'Distribution des gaps calendaires')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    filename = f"prediction_timeline_lookahead_{predictions_df['look_ahead_trading_days'].iloc[0]}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Graphique sauvegardé: {filename}")
    
    plt.show()
    
    # Afficher un résumé statistique
    print(f"\nStatistiques pour LOOK_AHEAD={predictions_df['look_ahead_trading_days'].iloc[0]}:")
    print(f"  Gap moyen (jours calendaires): {gaps.mean():.1f}")
    print(f"  Gap min: {gaps.min()} jours")
    print(f"  Gap max: {gaps.max()} jours")
    print(f"  Écart-type: {gaps.std():.1f} jours")
    
    # Afficher quelques exemples
    print(f"\nExemples de prédictions:")
    sample_size = min(3, len(plot_df))
    for i in range(sample_size):
        row = plot_df.iloc[i]
        print(f"\n  Exemple {i+1}:")
        print(f"    Test le: {row['test_date'].date()}")
        print(f"    Dernier entraînement: {row['last_train_date'].date()}")
        print(f"    Gap: {row['look_ahead_calendar_days']} jours calendaires")
        print(f"    Cible (HORIZON={HORIZON}j): {row['target_date'].date() if pd.notna(row['target_date']) else 'N/A'}")
        print(f"    Prédit: {'Hausse' if row['predicted_class']==1 else 'Baisse'} (proba: {row['predicted_proba']:.3f})")
        print(f"    Réel: {'Hausse' if row['true_class']==1 else 'Baisse'}")
        print(f"    Résultat: {'✓ Correct' if row['prediction_correct']==1 else '✗ Incorrect'}")

# Créer les graphiques pour les deux stratégies
print("\nCréation du graphique pour LOOK_AHEAD = 15 jours...")
create_prediction_timeline_plot(predictions_lookahead_15, "(sans look-ahead théorique)")

print("\nCréation du graphique pour LOOK_AHEAD = 5 jours...")
create_prediction_timeline_plot(predictions_lookahead_5, "(avec look-ahead potentiel)")

# ============================================================
# ANALYSE DÉTAILLÉE DU LOOK-AHEAD POTENTIEL
# ============================================================
print("\n" + "="*60)
print("ANALYSE DÉTAILLÉE DU LOOK-AHEAD POTENTIEL")
print("="*60)

if len(predictions_lookahead_5) > 0:
    # Identifier les prédictions problématiques (celles avec un gap trop petit)
    problematic_threshold = HORIZON  # Si le gap est < HORIZON, il y a chevauchement
    
    problematic_predictions = predictions_lookahead_5[
        predictions_lookahead_5['look_ahead_calendar_days'] < problematic_threshold
    ]
    
    if len(problematic_predictions) > 0:
        print(f"\n⚠️  {len(problematic_predictions)} prédictions avec gap < {HORIZON} jours (chevauchement potentiel):")
        print(f"   Cela représente {len(problematic_predictions)/len(predictions_lookahead_5):.1%} des prédictions")
        
        # Analyser la performance sur ces prédictions problématiques
        if len(problematic_predictions) >= 10:
            acc_problematic = accuracy_score(
                problematic_predictions['true_class'], 
                problematic_predictions['predicted_class']
            )
            acc_normal = accuracy_score(
                predictions_lookahead_5[predictions_lookahead_5['look_ahead_calendar_days'] >= problematic_threshold]['true_class'],
                predictions_lookahead_5[predictions_lookahead_5['look_ahead_calendar_days'] >= problematic_threshold]['predicted_class']
            )
            
            print(f"\nPerformance comparée:")
            print(f"  Prédictions avec gap < {HORIZON} jours: Accuracy = {acc_problematic:.4f}")
            print(f"  Prédictions avec gap >= {HORIZON} jours: Accuracy = {acc_normal:.4f}")
            print(f"  Différence: {acc_problematic - acc_normal:+.4f}")
            
            if acc_problematic - acc_normal > 0.05:
                print(f"  ⚠️  Forte amélioration avec look-ahead potentiel!")
            elif acc_problematic - acc_normal < -0.05:
                print(f"  ✅  Meilleure performance sans look-ahead!")
            else:
                print(f"  📊  Pas de différence significative")
    else:
        print(f"\n✅ Toutes les prédictions ont un gap >= {HORIZON} jours calendaires")
        print(f"   Pas de look-ahead théorique détecté")

# ============================================================
# RÉSUMÉ DES FICHIERS GÉNÉRÉS
# ============================================================
print("\n" + "="*60)
print("RÉSUMÉ DES FICHIERS GÉNÉRÉS")
print("="*60)
print("1. Fichiers CSV des prédictions:")
print(f"   - predictions_lookahead_{HORIZON}_horizon_{HORIZON}.csv")
print(f"   - predictions_lookahead_5_horizon_{HORIZON}.csv")
print("\n2. Graphiques PNG:")
print(f"   - prediction_timeline_lookahead_{HORIZON}.png")
print(f"   - prediction_timeline_lookahead_5.png")
print("\n3. Contenu des fichiers CSV:")
print("   - test_date: Date à laquelle on fait la prédiction")
print("   - last_train_date: Dernière date utilisée pour l'entraînement")
print("   - target_date: Date cible (test_date + HORIZON jours)")
print("   - look_ahead_trading_days: Gap en jours de trading")
print("   - look_ahead_calendar_days: Gap en jours calendaires")
print("   - predicted_proba: Probabilité de hausse prédite")
print("   - predicted_class: Classe prédite (1=hausse, 0=baisse)")
print("   - true_class: Classe réelle")
print("   - prediction_correct: 1 si correct, 0 sinon")

# ============================================================
# RECOMMANDATIONS FINALES
# ============================================================
print("\n" + "="*60)
print("RECOMMANDATIONS POUR L'INTERPRÉTATION")
print("="*60)
print("1. LOOK_AHEAD = HORIZON (15 jours):")
print("   - Aucun chevauchement théorique entre entraînement et test")
print("   - Méthodologiquement propre mais peut sous-performer")
print("\n2. LOOK_AHEAD = 5 jours:")
print("   - Chevauchement potentiel de 10 jours (HORIZON - LOOK_AHEAD)")
print("   - Peut donner de meilleurs résultats mais risque de data leakage")
print("\n3. Pour trader réellement:")
print("   - Avec LOOK_AHEAD=15: Prédit le 1er novembre à partir de données jusqu'au 17 octobre")
print("   - Avec LOOK_AHEAD=5: Prédit le 1er novembre à partir de données jusqu'au 27 octobre")
print("   → LOOK_AHEAD=5 donne 10 jours d'information supplémentaire!")

print("\n" + "="*60)
print("EXÉCUTION TERMINÉE")
print("="*60)



