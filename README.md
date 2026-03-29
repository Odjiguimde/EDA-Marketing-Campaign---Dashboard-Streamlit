# EDA Marketing Campaign — Notebook + Dashboard Streamlit

Analyse exploratoire complète des données de campagnes marketing client.
Projet réalisé dans le cadre d'un projet Data Science (Projet 1 — EDA).

---

## Demo en ligne

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://7mgvbwg6oftfzuyqqkytc2.streamlit.app/)

---

## Structure du projet
```
eda_marketing/
│
├── EDA_Marketing_Campaign.ipynb   # Notebook — analyse complète pas à pas
├── app.py                          # Application Streamlit interactive
├── marketing_campaign.csv          # Dataset (Marketing Campaign - Kaggle)
├── requirements.txt                # Dépendances Python
├── README.md
├── .gitignore
├── plots/                             # Graphiques exportés par le notebook
│   ├── 01_distributions_numeriques.png
│   ├── 02_boxplots_outliers.png
│   ├── 03_variables_categorielles.png
│   ├── 04_heatmap_correlations.png
│   ├── 05_correlations_target.png
│   ├── 06_scatter_plots.png
│   ├── 07_depenses_education.png
│   ├── 08_segments_conversion.png
│   ├── 09_performance_campagnes.png
│   └── 10_canaux_achat.png
└── .streamlit/
    └── config.toml                    # Thème sombre personnalisé
```

---

## Notebook — `EDA_Marketing_Campaign.ipynb`

Le notebook documente **toute la démarche analytique** étape par étape, idéal pour présenter votre travail à un recruteur ou un jury.

| Section | Contenu |
|---|---|
| **1. Chargement & aperçu** | Lecture du CSV, audit qualité, valeurs manquantes, doublons |
| **2. Nettoyage & Feature Engineering** | Imputation, nettoyage Marital_Status, 8 nouvelles variables |
| **3. Statistiques univariées** | Histogrammes, boxplots, skewness, variables catégorielles |
| **4. Statistiques multivariées** | Heatmap corrélations, scatter plots, dépenses x éducation |
| **5. Segments clients** | Taux de conversion par éducation, revenu, âge, statut marital |
| **6. Performance des campagnes** | Taux d'acceptation Cmp1→5, analyse des canaux d'achat |
| **7. Insights & Recommandations** | Synthèse automatique, export CSV nettoyé |

### Lancer le notebook
```bash
jupyter notebook EDA_Marketing_Campaign.ipynb
# ou
jupyter lab EDA_Marketing_Campaign.ipynb
```

---

## Application Streamlit — `app.py`

Dashboard interactif avec filtres dynamiques (éducation, âge, revenu, statut marital) et 5 onglets :

| Onglet | Contenu |
|---|---|
| **Aperçu** | Dataframe, types, nulls, stats descriptives |
| **Univarié** | Histogrammes + boxplots interactifs, camemberts |
| **Multivarié** | Heatmap, scatter plot configurable, dépenses x éducation |
| **Segments** | Conversion par segment, performance campagnes, canaux d'achat |
| **Insights** | Synthèse automatique, export CSV, prochaines étapes |

---

## Dataset

**Source :** [Kaggle — Marketing Campaign](https://www.kaggle.com/datasets/rodsaldanha/arketingcampaign/data)

| Variable | Description |
|---|---|
| `Year_Birth` | Année de naissance du client |
| `Education` | Niveau d'éducation |
| `Marital_Status` | Statut marital |
| `Income` | Revenu annuel (€) |
| `Kidhome / Teenhome` | Nombre d'enfants / adolescents |
| `Recency` | Nombre de jours depuis le dernier achat |
| `Mnt*` | Dépenses par catégorie (vins, viandes, etc.) |
| `Num*Purchases` | Nombre d'achats par canal |
| `AcceptedCmp1–5` | Réponse aux campagnes 1 à 5 |
| `Response` | Réponse à la dernière campagne (variable cible) |

---

## Installation locale
```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/eda-marketing.git
cd eda-marketing

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement à l'adresse : `http://localhost:8501`

---

## Déploiement sur Streamlit Cloud (gratuit)

### Etape 1 — Pousser sur GitHub
```bash
git init
git add .
git commit -m "feat: EDA Marketing Dashboard"
git branch -M main
git remote add origin https://github.com/votre-username/eda-marketing.git
git push -u origin main
```

### Etape 2 — Déployer sur Streamlit Cloud
1. Aller sur **[share.streamlit.io](https://share.streamlit.io)**
2. Se connecter avec votre compte GitHub
3. Cliquer sur **"New app"**
4. Sélectionner votre dépôt, branche `main`, fichier `app.py`
5. Cliquer **"Deploy"** → l'URL est générée en ~2 minutes

> Le fichier `marketing_campaign.csv` doit être inclus dans le dépôt GitHub pour que Streamlit Cloud puisse y accéder.

---

## Technologies utilisées

| Outil | Usage |
|---|---|
| `Python 3.11` | Langage principal |
| `Pandas` | Manipulation des données |
| `NumPy` | Calculs statistiques |
| `Matplotlib` | Visualisations de base |
| `Seaborn` | Heatmaps et visualisations statistiques |
| `Streamlit` | Interface web interactive |

---

## Variables créées (Feature Engineering)

| Variable | Formule |
|---|---|
| `Age` | `2024 - Year_Birth` |
| `TotalSpend` | Somme des 6 catégories de dépenses |
| `TotalPurchases` | Somme des 4 canaux d'achat |
| `TotalCampaigns` | Somme des 5 campagnes acceptées |
| `HasChildren` | `(Kidhome + Teenhome) > 0` |
| `Seniority` | Ancienneté en mois |
| `Age_Group` | Tranches d'âge (18–30, 31–45, 46–60, 60+) |
| `Income_Group` | Tranches de revenu (<30k, 30–60k, etc.) |

---

## Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, ouvrez d'abord une issue.

---

## Licence

MIT — Libre d'utilisation pour tout projet éducatif ou professionnel.
