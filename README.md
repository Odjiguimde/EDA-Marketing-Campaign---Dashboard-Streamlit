# 📊 EDA Marketing Campaign — Dashboard Streamlit

Analyse exploratoire interactive des données de campagnes marketing client.  
Projet réalisé dans le cadre d'un projet Data Science (Projet 1 — EDA).

---

## 🚀 Démo en ligne

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://votre-app.streamlit.app)

> Remplacez ce lien après le déploiement sur Streamlit Cloud.

---

## 📂 Structure du projet

```
eda_marketing/
│
├── app.py                    # Application Streamlit principale
├── marketing_campaign.csv    # Dataset (Marketing Campaign - Kaggle)
├── requirements.txt          # Dépendances Python
├── .streamlit/
│   └── config.toml           # Configuration thème Streamlit
├── .gitignore
└── README.md
```

---

## 📦 Dataset

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

## 🧠 Fonctionnalités

### 🗂 Aperçu des données
- Aperçu du dataframe, types de variables
- Valeurs manquantes, doublons
- Statistiques descriptives complètes

### 📈 Analyse univariée
- Histogrammes avec lignes moyenne/médiane
- Boxplots pour la détection d'outliers
- Graphiques camembert & barres pour les catégories
- Skewness affiché pour chaque variable

### 🔗 Analyse multivariée
- Heatmap de corrélations
- Corrélations avec la variable cible `Response`
- Scatter plot interactif (X, Y, couleur au choix)
- Dépenses par catégorie de produit × niveau d'éducation

### 🎯 Segments clients
- Taux de conversion par segment (éducation, âge, revenu...)
- Performance de chaque campagne (Cmp1→5 + dernière)
- Analyse des canaux d'achat (Web, Catalogue, Magasin, Promo)

### 💡 Insights & Recommandations
- Synthèse automatique des résultats clés
- Export CSV des données filtrées
- Prochaines étapes pour la modélisation

---

## 🛠 Installation locale

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

## ☁️ Déploiement sur Streamlit Cloud (gratuit)

### Étape 1 — Pousser sur GitHub
```bash
git init
git add .
git commit -m "feat: EDA Marketing Dashboard"
git branch -M main
git remote add origin https://github.com/votre-username/eda-marketing.git
git push -u origin main
```

### Étape 2 — Déployer sur Streamlit Cloud
1. Aller sur **[share.streamlit.io](https://share.streamlit.io)**
2. Se connecter avec votre compte GitHub
3. Cliquer sur **"New app"**
4. Sélectionner votre dépôt, branche `main`, fichier `app.py`
5. Cliquer **"Deploy"** → l'URL est générée en ~2 minutes ✅

> ⚠️ Le fichier `marketing_campaign.csv` doit être inclus dans le dépôt GitHub pour que Streamlit Cloud puisse y accéder.

---

## 🧰 Technologies utilisées

| Outil | Usage |
|---|---|
| `Python 3.11` | Langage principal |
| `Pandas` | Manipulation des données |
| `NumPy` | Calculs statistiques |
| `Matplotlib` | Visualisations de base |
| `Seaborn` | Heatmaps et visualisations statistiques |
| `Streamlit` | Interface web interactive |

---

## 📊 Variables créées (Feature Engineering)

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

## 🤝 Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, ouvrez d'abord une issue.

---

## 📄 Licence

MIT — Libre d'utilisation pour tout projet éducatif ou professionnel.
