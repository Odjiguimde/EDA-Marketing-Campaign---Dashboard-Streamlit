import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Marketing EDA Pro",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background:#0a0c14; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#d4d8f0 !important; }
[data-testid="stSidebar"] label { color:#6b7499 !important; font-size:.72rem !important;
    letter-spacing:.08em; text-transform:uppercase; }

[data-testid="metric-container"] { background:#131627; border:1px solid #1e2540;
    border-radius:12px; padding:.9rem 1.1rem; }
[data-testid="metric-container"] label { color:#6b7499 !important; font-size:.72rem !important;
    text-transform:uppercase; letter-spacing:.07em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e4e8ff !important;
    font-size:1.55rem !important; font-weight:700; }
[data-testid="stMetricDelta"] { font-size:.78rem !important; }

.stTabs [data-baseweb="tab-list"] { gap:2px; background:transparent;
    border-bottom:1px solid #1e2540; padding-bottom:0; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#6b7499;
    border-radius:8px 8px 0 0; padding:.45rem 1.1rem; font-size:.82rem; font-weight:500; }
.stTabs [aria-selected="true"] { background:#131627 !important; color:#7b9cff !important;
    border-bottom:2px solid #7b9cff; }

.sec { font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#7b9cff; border-left:3px solid #7b9cff; padding-left:.55rem;
    margin:1.4rem 0 .7rem; }

.kpi-badge { display:inline-block; background:#131627; border:1px solid #1e2540;
    border-radius:8px; padding:.3rem .8rem; font-size:.8rem; color:#a0a8cc;
    margin:.2rem .2rem .2rem 0; }
.kpi-badge b { color:#7b9cff; }

.insight { background:#0f1221; border:1px solid #1e2540; border-left:3px solid #7b9cff;
    border-radius:10px; padding:.9rem 1.1rem; margin:.4rem 0;
    font-size:.86rem; line-height:1.65; color:#b0b8d8; }
.insight b { color:#7b9cff; }
.insight.warn  { border-left-color:#f4a261; } .insight.warn  b { color:#f4a261; }
.insight.good  { border-left-color:#52b788; } .insight.good  b { color:#52b788; }
.insight.alert { border-left-color:#ef476f; } .insight.alert b { color:#ef476f; }

.seg-card { background:#0f1221; border:1px solid #1e2540; border-radius:12px;
    padding:1rem; text-align:center; }
.seg-card .seg-name { font-size:.85rem; font-weight:700; color:#e4e8ff; margin-bottom:.3rem; }
.seg-card .seg-val  { font-size:1.4rem; font-weight:700; color:#7b9cff; }
.seg-card .seg-sub  { font-size:.75rem; color:#6b7499; margin-top:.2rem; }

.main .block-container { background:#080a12; padding-top:1.2rem; max-width:1400px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# THEME GRAPHIQUES
# ----------------------------------------------------------------------
PAL  = ["#7b9cff","#64dfdf","#f4a261","#ef476f","#52b788",
        "#ffd166","#a8dadc","#c77dff","#e76f51","#06d6a0"]
BG   = "#131627"
GRID = "#1e2540"
TEXT = "#b0b8d8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "axes.titlecolor": "#e4e8ff", "axes.titlesize": 11,
    "axes.titleweight": "600", "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "legend.facecolor": BG, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "grid.color": GRID, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "axes.grid": True, "figure.dpi": 110,
})

def nfig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def th(ax, title="", xl="", yl=""):
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_linewidth(0.5)
    if title: ax.set_title(title)
    if xl:    ax.set_xlabel(xl)
    if yl:    ax.set_ylabel(yl)

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# CHARGEMENT DES DONNEES
# ----------------------------------------------------------------------
@st.cache_data
def load(file=None):
    df = pd.read_csv(file if file else "marketing_campaign.csv", sep=";")

    df["Income"]        = df["Income"].fillna(df["Income"].median())
    df["Marital_Clean"] = df["Marital_Status"].replace(
        {"Alone": "Single", "Absurd": "Other", "YOLO": "Other"})
    df["Dt_Customer"]   = pd.to_datetime(df["Dt_Customer"])

    df = df[df["Year_Birth"] > 1930]
    df = df[df["Income"] < 200000]

    now = pd.Timestamp.now()
    df["Age"]             = datetime.now().year - df["Year_Birth"]
    df["Seniority"]       = (now - df["Dt_Customer"]).dt.days // 30
    df["TotalSpend"]      = df[["MntWines","MntFruits","MntMeatProducts",
                                 "MntFishProducts","MntSweetProducts","MntGoldProds"]].sum(axis=1)
    df["TotalPurch"]      = df[["NumDealsPurchases","NumWebPurchases",
                                 "NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)
    df["TotalCamp"]       = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                                 "AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)
    df["HasChildren"]     = (df["Kidhome"] + df["Teenhome"]).clip(upper=1)
    df["SpendPerPurch"]   = (df["TotalSpend"] / df["TotalPurch"].replace(0, np.nan)).fillna(0)
    df["IncomePerPerson"] = df["Income"] / (1 + df["Kidhome"] + df["Teenhome"])

    df["Age_Group"]    = pd.cut(df["Age"], bins=[17,30,45,60,100],
                                labels=["18-30","31-45","46-60","60+"])
    df["Income_Group"] = pd.cut(df["Income"], bins=[0,30000,60000,90000,200000],
                                labels=["<30k","30-60k","60-90k",">90k"])

    df["R"] = pd.qcut(df["Recency"], 4, labels=[4,3,2,1]).astype(int)
    df["F"] = pd.qcut(df["TotalPurch"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    df["M"] = pd.qcut(df["TotalSpend"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    df["RFM"] = df["R"] + df["F"] + df["M"]
    df["RFM_Seg"] = df["RFM"].apply(lambda s:
        "Champions"  if s >= 10 else
        "Fideles"    if s >= 8  else
        "Potentiels" if s >= 6  else
        "A risque"   if s >= 4  else
        "Inactifs")

    return df

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Marketing EDA Pro")
    st.markdown("---")
    up = st.file_uploader("Charger un CSV", type=["csv"])
    df = load(up)

    sec("Filtres globaux")
    edu  = st.multiselect("Education",
                          sorted(df["Education"].unique()),
                          default=sorted(df["Education"].unique()))
    mar  = st.multiselect("Statut marital",
                          sorted(df["Marital_Clean"].unique()),
                          default=sorted(df["Marital_Clean"].unique()))
    ages = st.slider("Age",
                     int(df["Age"].min()), int(df["Age"].max()),
                     (int(df["Age"].min()), int(df["Age"].max())))
    inc  = st.slider("Revenu (EUR)", 0, 200000, (0, 200000), step=2000)
    resp = st.radio("Reponse campagne", ["Tous","Convertis","Non convertis"])

    sec("Affichage")
    show_raw = st.checkbox("Afficher donnees brutes", False)

    st.markdown("---")
    st.markdown(
        f"<small style='color:#4a5280'>v2.0 · {df.shape[0]:,} clients · "
        f"{df.shape[1]} variables</small>",
        unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FILTRES
# ----------------------------------------------------------------------
dff = df[df["Education"].isin(edu) & df["Marital_Clean"].isin(mar)].copy()
dff = dff[(dff["Age"] >= ages[0]) & (dff["Age"] <= ages[1])]
dff = dff[(dff["Income"] >= inc[0]) & (dff["Income"] <= inc[1])]
if resp == "Convertis":       dff = dff[dff["Response"] == 1]
elif resp == "Non convertis": dff = dff[dff["Response"] == 0]

# ----------------------------------------------------------------------
# EN-TETE
# ----------------------------------------------------------------------
st.markdown("# Marketing Campaign — Analyse Exploratoire Complete")
st.markdown(
    f"<span class='kpi-badge'><b>{dff.shape[0]:,}</b> clients selectionnes</span>"
    f"<span class='kpi-badge'>Taux conversion <b>{dff['Response'].mean()*100:.1f}%</b></span>"
    f"<span class='kpi-badge'>Revenu median <b>{dff['Income'].median():,.0f} EUR</b></span>"
    f"<span class='kpi-badge'>Depense moy. <b>{dff['TotalSpend'].mean():,.0f} EUR</b></span>",
    unsafe_allow_html=True)

cols_kpi = st.columns(6)
cols_kpi[0].metric("Clients",         f"{dff.shape[0]:,}")
cols_kpi[1].metric("Revenu median",   f"{dff['Income'].median():,.0f} EUR",
                   f"{((dff['Income'].median()/df['Income'].median())-1)*100:+.1f}% vs total")
cols_kpi[2].metric("Depense moyenne", f"{dff['TotalSpend'].mean():,.0f} EUR")
cols_kpi[3].metric("Taux conversion", f"{dff['Response'].mean()*100:.1f}%")
cols_kpi[4].metric("Achats moyens",   f"{dff['TotalPurch'].mean():.1f}")
cols_kpi[5].metric("Campagnes moy.",  f"{dff['TotalCamp'].mean():.2f}")

st.markdown("---")

# ----------------------------------------------------------------------
# ONGLETS
# ----------------------------------------------------------------------
tabs = st.tabs([
    "Apercu",
    "Univarie",
    "Multivarie",
    "Segments RFM",
    "Clustering",
    "Prediction",
    "Campagnes",
    "Insights",
])

# ======================================================================
# ONGLET 1 — APERCU
# ======================================================================
with tabs[0]:
    sec("Apercu du dataset")
    st.dataframe(dff.head(20), use_container_width=True, height=300)

    col1, col2, col3 = st.columns(3)

    with col1:
        sec("Qualite des donnees")
        nulls = dff.isnull().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            st.success("Aucune valeur manquante")
        else:
            st.dataframe(nulls.rename("Nulls"), use_container_width=True)
        st.info(f"Doublons : **{dff.duplicated().sum()}**")

    with col2:
        sec("Types de variables")
        tdf = pd.DataFrame({
            "Type":    dff.dtypes.astype(str),
            "Uniques": [dff[c].nunique() for c in dff.columns],
            "Nulls %": (dff.isnull().mean()*100).round(1).astype(str) + "%",
        })
        st.dataframe(tdf, use_container_width=True, height=280)

    with col3:
        sec("Distribution variable cible")
        vc = dff["Response"].value_counts().sort_index()
        fig, ax = nfig(4, 3)
        bars = ax.bar(["Non converti","Converti"], vc.values,
                      color=[PAL[3], PAL[0]], edgecolor=BG, alpha=.9)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                    f"{v}\n({v/len(dff)*100:.1f}%)",
                    ha="center", fontsize=9, fontweight="600")
        th(ax, "Repartition Response", yl="Effectif")
        ax.grid(axis="x", visible=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    sec("Statistiques descriptives")
    num_cols_desc = ["Age","Income","TotalSpend","TotalPurch","Recency",
                     "Seniority","MntWines","MntMeatProducts","MntFruits","SpendPerPurch"]
    desc = dff[num_cols_desc].describe().T
    st.dataframe(
        desc.style.format("{:.2f}").background_gradient(
            subset=["mean","std"], cmap="Blues"),
        use_container_width=True)

    if show_raw:
        sec("Donnees brutes completes")
        st.dataframe(dff, use_container_width=True)

# ======================================================================
# ONGLET 2 — UNIVARIE
# ======================================================================
with tabs[1]:
    sec("Analyse d'une variable numerique")
    va, vb = st.columns([2, 1])
    with va:
        num_var = st.selectbox("Variable", [
            "Age","Income","TotalSpend","TotalPurch","Recency",
            "MntWines","MntMeatProducts","MntFruits","MntFishProducts",
            "MntSweetProducts","MntGoldProds","NumWebVisitsMonth",
            "Seniority","SpendPerPurch","IncomePerPerson"])
    with vb:
        bins_n = st.slider("Nombre de bins", 10, 80, 35)

    data = dff[num_var].dropna()
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = nfig(5, 3.5)
        ax.hist(data, bins=bins_n, color=PAL[0], alpha=.85, edgecolor=BG, linewidth=.3)
        ax.axvline(data.mean(),   color=PAL[2], ls="--", lw=1.3,
                   label=f"Moy: {data.mean():.1f}")
        ax.axvline(data.median(), color=PAL[3], ls=":",  lw=1.3,
                   label=f"Med: {data.median():.1f}")
        ax.legend(fontsize=8)
        th(ax, f"Histogramme — {num_var}", xl=num_var, yl="Frequence")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = nfig(4, 3.5)
        ax.boxplot(data, patch_artist=True, vert=True,
                   medianprops=dict(color=PAL[2], lw=2.5),
                   boxprops=dict(facecolor=PAL[0], alpha=.45, lw=.8),
                   whiskerprops=dict(color=TEXT, lw=.8),
                   capprops=dict(color=TEXT, lw=.8),
                   flierprops=dict(marker="o", color=PAL[3], alpha=.35, ms=2.5))
        ax.set_xticklabels([num_var])
        th(ax, f"Boxplot — {num_var}")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c3:
        fig, ax = nfig(4, 3.5)
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PAL[0])
            pc.set_alpha(.5)
        parts["cmeans"].set_color(PAL[2])
        parts["cmedians"].set_color(PAL[3])
        ax.set_xticks([])
        th(ax, f"Violin — {num_var}")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    skew = data.skew()
    kurt = data.kurtosis()
    interp = ("Distribution tres asymetrique a droite" if skew > 1
              else "Asymetrie moderee" if skew > .5
              else "Distribution relativement symetrique")
    st.markdown(
        f"<div class='insight'><b>{num_var}</b> — Moy: <b>{data.mean():.2f}</b>"
        f" · Med: <b>{data.median():.2f}</b> · Ecart-type: <b>{data.std():.2f}</b>"
        f" · Min: <b>{data.min():.0f}</b> · Max: <b>{data.max():.0f}</b>"
        f" · Skewness: <b>{skew:.3f}</b> · Kurtosis: <b>{kurt:.3f}</b>"
        f"<br>-> {interp}</div>",
        unsafe_allow_html=True)

    sec("Distribution selon la reponse a la campagne")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    for rv, color, label in [(0,PAL[3],"Non converti"),(1,PAL[0],"Converti")]:
        sub = dff[dff["Response"] == rv][num_var].dropna()
        axes[0].hist(sub, bins=30, alpha=.6, color=color,
                     label=label, edgecolor=BG, lw=.3)
        axes[1].boxplot(sub, positions=[rv], patch_artist=True, widths=.4,
                        boxprops=dict(facecolor=color, alpha=.5),
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=TEXT),
                        capprops=dict(color=TEXT),
                        flierprops=dict(marker="o", color=color, alpha=.3, ms=2))
    axes[0].legend(fontsize=9)
    th(axes[0], f"{num_var} — Non convertis vs Convertis", xl=num_var)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Non conv.", "Converti"])
    th(axes[1], "Comparaison boxplots")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    sec("Variables categorielles")
    cat_var = st.selectbox("Variable categorielle", [
        "Education","Marital_Clean","Age_Group","Income_Group","HasChildren","RFM_Seg"])
    counts = dff[cat_var].value_counts().reset_index()
    counts.columns = ["Categorie","Effectif"]
    conv_map = dff.groupby(cat_var)["Response"].mean() * 100
    counts["Pct_Convertis"] = counts["Categorie"].map(conv_map).round(1)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig, ax = nfig(6, 4)
        bars = ax.barh(counts["Categorie"].astype(str), counts["Effectif"],
                       color=PAL[:len(counts)], edgecolor=BG, alpha=.88)
        for b, v in zip(bars, counts["Effectif"]):
            ax.text(v + 5, b.get_y() + b.get_height()/2,
                    f" {v}", va="center", fontsize=9)
        th(ax, f"Effectifs — {cat_var}", xl="Nb clients")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with cc2:
        fig, ax = nfig(6, 4)
        bars2 = ax.bar(counts["Categorie"].astype(str), counts["Pct_Convertis"],
                       color=PAL[:len(counts)], edgecolor=BG, alpha=.88)
        for b, v in zip(bars2, counts["Pct_Convertis"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + .2,
                    f"{v:.1f}%", ha="center", fontsize=9, fontweight="600")
        th(ax, f"Taux de conversion — {cat_var}", yl="%")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ======================================================================
# ONGLET 3 — MULTIVARIE
# ======================================================================
with tabs[2]:
    sec("Matrice de correlation")
    corr_cols = ["Age","Income","TotalSpend","TotalPurch","Recency","MntWines",
                 "MntMeatProducts","MntFruits","MntGoldProds","NumWebVisitsMonth",
                 "TotalCamp","Seniority","SpendPerPurch","Response"]
    corr = dff[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask,
                cmap=sns.diverging_palette(230, 20, as_cmap=True),
                center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size":7.5,"color":TEXT},
                linewidths=.4, linecolor=GRID, ax=ax,
                cbar_kws={"shrink":.65})
    ax.set_title("Matrice de correlations", color="#e4e8ff", fontsize=12, fontweight="600")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    sec("Scatter plot interactif")
    s1, s2, s3, s4 = st.columns(4)
    xv   = s1.selectbox("Axe X", ["Income","Age","TotalSpend","Recency","Seniority"], key="x")
    yv   = s2.selectbox("Axe Y", ["TotalSpend","TotalPurch","Income","MntWines","Recency"], key="y")
    hv   = s3.selectbox("Couleur", ["Response","Education","Age_Group","RFM_Seg"], key="h")
    size = s4.selectbox("Taille pts", ["Uniforme","TotalSpend","Income","Age"], key="sz")

    fig, ax = nfig(11, 5)
    for i, cat in enumerate(sorted(dff[hv].dropna().unique(), key=str)):
        sub = dff[dff[hv] == cat]
        sz  = 20 if size == "Uniforme" else (sub[size] / sub[size].max() * 80 + 5)
        ax.scatter(sub[xv], sub[yv], label=str(cat),
                   color=PAL[i % len(PAL)], alpha=.5, s=sz, edgecolors="none")
    th(ax, f"{xv} vs {yv}", xl=xv, yl=yv)
    ax.legend(fontsize=8, title=hv, title_fontsize=8,
              bbox_to_anchor=(1.01,.99), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    sec("Depenses par produit x segment")
    seg_x    = st.selectbox("Segmenter par",
                             ["Education","Age_Group","Income_Group","RFM_Seg"],
                             key="seg_multi")
    sp_cols  = ["MntWines","MntFruits","MntMeatProducts",
                "MntFishProducts","MntSweetProducts","MntGoldProds"]
    sp_names = ["Vins","Fruits","Viandes","Poissons","Sucreries","Or-Luxe"]
    sp_grp   = dff.groupby(seg_x)[sp_cols].mean()

    fig, ax = nfig(13, 5)
    x  = np.arange(len(sp_names))
    bw = .8 / len(sp_grp)
    for i, (seg, row) in enumerate(sp_grp.iterrows()):
        ax.bar(x + i*bw - .4, row.values, bw,
               label=str(seg), color=PAL[i % len(PAL)], alpha=.85, edgecolor=BG)
    ax.set_xticks(x)
    ax.set_xticklabels(sp_names, rotation=10, ha="right")
    th(ax, f"Depenses moyennes par produit — {seg_x}", yl="Montant moyen (EUR)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01,.99), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    sec("Pairplot des variables cles")
    pp_cols = st.multiselect("Variables a croiser",
                              ["Age","Income","TotalSpend","Recency","TotalPurch"],
                              default=["Income","TotalSpend","Recency"])
    if len(pp_cols) >= 2:
        n = len(pp_cols)
        fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
        fig.patch.set_facecolor(BG)
        for i, vi in enumerate(pp_cols):
            for j, vj in enumerate(pp_cols):
                ax = axes[i][j]
                ax.set_facecolor(BG)
                for sp in ax.spines.values():
                    sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
                ax.tick_params(colors=TEXT, labelsize=7)
                if i == j:
                    ax.hist(dff[vi].dropna(), bins=25,
                            color=PAL[0], alpha=.75, edgecolor=BG)
                else:
                    for rv, col in [(0,PAL[3]),(1,PAL[0])]:
                        sub = dff[dff["Response"] == rv]
                        ax.scatter(sub[vj], sub[vi], c=col,
                                   alpha=.3, s=6, edgecolors="none")
                if i == n-1: ax.set_xlabel(vj, fontsize=8, color=TEXT)
                if j == 0:   ax.set_ylabel(vi, fontsize=8, color=TEXT)
        fig.suptitle("Pairplot — colore par Response",
                     color="#e4e8ff", fontsize=11, fontweight="600")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ======================================================================
# ONGLET 4 — SEGMENTS RFM
# ======================================================================
with tabs[3]:
    sec("Analyse RFM — Recence, Frequence, Montant")
    st.markdown("""<div class='insight'>
    L'analyse <b>RFM</b> segmente les clients selon 3 axes :
    <b>Recence</b> (derniere visite), <b>Frequence</b> (nombre d'achats)
    et <b>Montant</b> (depenses totales).
    Chaque axe est note de 1 a 4 — le score total determine le segment.
    </div>""", unsafe_allow_html=True)

    rfm_stats = dff.groupby("RFM_Seg").agg(
        Clients=("Response","count"),
        Conv=("Response","mean"),
        Depense=("TotalSpend","mean"),
        Revenu=("Income","mean"),
    ).reset_index()
    rfm_stats["Conv"] *= 100

    seg_cols = st.columns(len(rfm_stats))
    for col_s, (_, row) in zip(seg_cols, rfm_stats.iterrows()):
        col_s.markdown(
            f"<div class='seg-card'>"
            f"<div class='seg-name'>{row['RFM_Seg']}</div>"
            f"<div class='seg-val'>{row['Clients']}</div>"
            f"<div class='seg-sub'>Conv: {row['Conv']:.1f}% · Dep: {row['Depense']:,.0f} EUR</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    with r1:
        sec("Depense moyenne par segment")
        rfm_s = rfm_stats.sort_values("Depense", ascending=True)
        fig, ax = nfig(6, 4)
        bars = ax.barh(rfm_s["RFM_Seg"], rfm_s["Depense"],
                       color=PAL[:len(rfm_s)], edgecolor=BG, alpha=.88)
        for b, v in zip(bars, rfm_s["Depense"]):
            ax.text(v+10, b.get_y()+b.get_height()/2,
                    f"{v:,.0f} EUR", va="center", fontsize=9, fontweight="600")
        th(ax, "Depense moyenne — Segments RFM", xl="EUR")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with r2:
        sec("Taux de conversion par segment")
        rfm_s2 = rfm_stats.sort_values("Conv", ascending=True)
        fig, ax = nfig(6, 4)
        bars2 = ax.barh(rfm_s2["RFM_Seg"], rfm_s2["Conv"],
                        color=PAL[:len(rfm_s2)], edgecolor=BG, alpha=.88)
        for b, v in zip(bars2, rfm_s2["Conv"]):
            ax.text(v+.2, b.get_y()+b.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=9, fontweight="600")
        th(ax, "Taux de conversion — Segments RFM", xl="%")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    sec("Distribution des scores RFM")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    for ax, score, label, color in zip(
            axes, ["R","F","M"],
            ["Recence (R)","Frequence (F)","Montant (M)"],
            [PAL[3],PAL[0],PAL[2]]):
        ax.set_facecolor(BG)
        vc = dff[score].value_counts().sort_index()
        bars = ax.bar(vc.index, vc.values, color=color, edgecolor=BG, alpha=.85)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    str(v), ha="center", fontsize=9, fontweight="600")
        ax.set_title(label)
        ax.set_xlabel("Score (1=faible, 4=fort)")
        ax.set_ylabel("Nb clients")
        th(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    sec("Tableau detaille des segments")
    detail = dff.groupby("RFM_Seg").agg(
        Clients=("Response","count"),
        Taux_conv=("Response","mean"),
        Depense_moy=("TotalSpend","mean"),
        Revenu_moy=("Income","mean"),
        Recence_moy=("Recency","mean"),
        Achats_moy=("TotalPurch","mean"),
        Campagnes_moy=("TotalCamp","mean"),
    ).round(2)
    detail["Taux_conv"]   = (detail["Taux_conv"]*100).round(1).astype(str) + "%"
    detail["Depense_moy"] = detail["Depense_moy"].apply(lambda x: f"{x:,.0f} EUR")
    detail["Revenu_moy"]  = detail["Revenu_moy"].apply(lambda x: f"{x:,.0f} EUR")
    st.dataframe(detail, use_container_width=True)

# ======================================================================
# ONGLET 5 — CLUSTERING
# ======================================================================
with tabs[4]:
    sec("Clustering K-Means — Segmentation non supervisee")
    st.markdown("""<div class='insight'>
    Le clustering K-Means regroupe automatiquement les clients en segments homogenes
    sans utiliser la variable cible.
    </div>""", unsafe_allow_html=True)

    cl1, cl2 = st.columns([1, 2])
    with cl1:
        n_clusters = st.slider("Nombre de clusters", 2, 8, 4)
        cluster_features = st.multiselect(
            "Variables pour le clustering",
            ["Age","Income","TotalSpend","TotalPurch","Recency","Seniority",
             "MntWines","MntMeatProducts","TotalCamp","NumWebVisitsMonth"],
            default=["Income","TotalSpend","TotalPurch","Recency","Age"])

    if len(cluster_features) >= 2:
        X_cl   = dff[cluster_features].dropna()
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_cl)

        with cl2:
            sec("Methode du coude")
            inertias = []
            for k in range(2, 10):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_sc)
                inertias.append(km.inertia_)
            fig, ax = nfig(6, 3)
            ax.plot(range(2,10), inertias, marker="o",
                    color=PAL[0], lw=2, ms=7)
            ax.axvline(n_clusters, color=PAL[2], ls="--", lw=1.3,
                       label=f"K={n_clusters} selectionne")
            ax.legend(fontsize=9)
            th(ax, "Inertie vs nombre de clusters", xl="K", yl="Inertie")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels   = km_final.fit_predict(X_sc)

        dff_cl = dff.copy()
        dff_cl.loc[X_cl.index, "Cluster"] = ["Cluster " + str(l) for l in labels]
        dff_cl["Cluster"] = dff_cl["Cluster"].fillna("Cluster ?")

        sec("Visualisation PCA des clusters")
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sc)
        var_exp = pca.explained_variance_ratio_

        fig, ax = nfig(10, 5)
        for i in range(n_clusters):
            mask_i = labels == i
            ax.scatter(X_pca[mask_i,0], X_pca[mask_i,1],
                       c=PAL[i%len(PAL)], label=f"Cluster {i}",
                       alpha=.55, s=20, edgecolors="none")
        c_pca = pca.transform(km_final.cluster_centers_)
        ax.scatter(c_pca[:,0], c_pca[:,1], marker="X", s=200,
                   c=PAL[:n_clusters], edgecolors="white",
                   linewidths=1.2, zorder=5)
        th(ax,
           f"PCA — {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% variance expliquee",
           xl=f"PC1 ({var_exp[0]*100:.1f}%)",
           yl=f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.legend(fontsize=9, bbox_to_anchor=(1.01,.99), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        sec("Profil des clusters")
        extra_cols = [c for c in ["TotalSpend","Income"]
                      if c not in cluster_features]
        cluster_profile = dff_cl.groupby("Cluster")[
            cluster_features + extra_cols].mean().round(2)
        # Ajouter taux de conversion comme colonne numerique separee
        cluster_profile["Taux_conv_pct"] = (
            dff_cl.groupby("Cluster")["Response"].mean() * 100).round(1)
        # Gradient uniquement sur colonnes numeriques
        num_cp = cluster_profile.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(
            cluster_profile.style.format("{:.2f}").background_gradient(
                cmap="Blues", axis=0, subset=num_cp),
            use_container_width=True)

        sec("Radar — profil normalise par cluster")
        radar_cols = [c for c in cluster_features if c in dff.columns][:6]
        if len(radar_cols) >= 3:
            angles = np.linspace(0, 2*np.pi, len(radar_cols), endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_cols, size=9, color=TEXT)
            ax.yaxis.set_tick_params(labelcolor=TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.grid(color=GRID, linewidth=.5)
            norm = dff_cl.groupby("Cluster")[radar_cols].mean()
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
            for i, (cname, row) in enumerate(norm.iterrows()):
                vals = row.tolist() + [row.iloc[0]]
                ax.plot(angles, vals, lw=2, color=PAL[i%len(PAL)], label=cname)
                ax.fill(angles, vals, alpha=.15, color=PAL[i%len(PAL)])
            ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=9,
                      labelcolor=TEXT, facecolor=BG, edgecolor=GRID)
            ax.set_title("Profil normalise par cluster",
                         color="#e4e8ff", fontsize=11, pad=20)
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ======================================================================
# ONGLET 6 — PREDICTION
# ======================================================================
with tabs[5]:
    sec("Prediction de conversion — Machine Learning")
    st.markdown("""<div class='insight'>
    Comparaison de 3 modeles ML pour predire si un client va repondre a la campagne.
    Entraine sur 80% des donnees filtrees, evalue sur 20%.
    </div>""", unsafe_allow_html=True)

    feat_cols = ["Age","Income","TotalSpend","TotalPurch","Recency","Seniority",
                 "MntWines","MntMeatProducts","MntFruits","MntGoldProds",
                 "NumWebVisitsMonth","TotalCamp","HasChildren","Kidhome","Teenhome"]
    feat_sel   = st.multiselect("Features du modele", feat_cols, default=feat_cols)
    model_name = st.selectbox("Modele", [
        "Random Forest","Gradient Boosting","Regression Logistique"])

    if len(feat_sel) >= 3:
        Xm = dff[feat_sel].fillna(0)
        ym = dff["Response"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xm, ym, test_size=.2, random_state=42, stratify=ym)

        models_dict = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42),
            "Regression Logistique": Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))])
        }
        with st.spinner(f"Entrainement — {model_name}..."):
            clf     = models_dict[model_name]
            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)[:,1]
            cv_auc  = cross_val_score(clf, Xm, ym, cv=5, scoring="roc_auc").mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{accuracy_score(y_te,y_pred)*100:.1f}%")
        m2.metric("Precision", f"{precision_score(y_te,y_pred)*100:.1f}%")
        m3.metric("Recall",    f"{recall_score(y_te,y_pred)*100:.1f}%")
        m4.metric("F1-Score",  f"{f1_score(y_te,y_pred)*100:.1f}%")
        m5.metric("AUC (CV)",  f"{cv_auc:.3f}")

        p1, p2, p3 = st.columns(3)

        with p1:
            sec("Matrice de confusion")
            fig, ax = nfig(4, 3.5)
            cm = confusion_matrix(y_te, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        annot_kws={"size":13,"fontweight":"600","color":"white"},
                        linewidths=.5, linecolor=BG,
                        xticklabels=["Non conv.","Converti"],
                        yticklabels=["Non conv.","Converti"])
            ax.set_xlabel("Predit", color=TEXT)
            ax.set_ylabel("Reel", color=TEXT)
            ax.tick_params(colors=TEXT, labelsize=9)
            th(ax, "Matrice de confusion")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with p2:
            sec("Courbe ROC")
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = nfig(4, 3.5)
            ax.plot(fpr, tpr, color=PAL[0], lw=2.5,
                    label=f"AUC = {roc_auc:.3f}")
            ax.plot([0,1],[0,1], color=GRID, lw=1, ls="--", label="Aleatoire")
            ax.fill_between(fpr, tpr, alpha=.12, color=PAL[0])
            ax.legend(fontsize=9)
            th(ax, "Courbe ROC",
               xl="Taux faux positifs", yl="Taux vrais positifs")
            ax.set_xlim([0,1])
            ax.set_ylim([0,1.02])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with p3:
            sec("Importance des features")
            if hasattr(clf, "feature_importances_"):
                imp = pd.Series(clf.feature_importances_,
                                index=feat_sel).sort_values(ascending=True)
            elif hasattr(clf, "named_steps"):
                coef = np.abs(clf.named_steps["lr"].coef_[0])
                imp  = pd.Series(coef, index=feat_sel).sort_values(ascending=True)
            else:
                imp = pd.Series(np.ones(len(feat_sel)), index=feat_sel)
            fig, ax = nfig(4, 3.5)
            ax.barh(imp.index, imp.values, color=PAL[0], edgecolor=BG, alpha=.85)
            th(ax, "Importance des variables", xl="Score")
            ax.grid(axis="y", visible=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        sec("Simulateur de prediction client")
        st.markdown("Entrez les caracteristiques d'un client pour predire sa probabilite de conversion :")
        sim_ui = st.columns(4)
        sim_vals = {}
        for i, feat in enumerate(feat_sel[:12]):
            col_ui = sim_ui[i % 4]
            mn  = float(dff[feat].min())
            mx  = float(dff[feat].max())
            med = float(dff[feat].median())
            sim_vals[feat] = col_ui.number_input(
                feat, min_value=mn, max_value=mx, value=med, key=f"sim_{feat}")

        if st.button("Predire la conversion", use_container_width=True):
            sim_df = pd.DataFrame([sim_vals])
            for fc in feat_sel:
                if fc not in sim_df.columns:
                    sim_df[fc] = dff[fc].median()
            prob = clf.predict_proba(sim_df[feat_sel])[0][1]
            pred = clf.predict(sim_df[feat_sel])[0]
            if pred == 1:
                st.success(
                    f"Ce client a **{prob*100:.1f}%** de probabilite de convertir — CIBLER")
            else:
                st.warning(
                    f"Ce client a **{prob*100:.1f}%** de probabilite de convertir — NE PAS CIBLER")

# ======================================================================
# ONGLET 7 — CAMPAGNES
# ======================================================================
with tabs[6]:
    sec("Performance des campagnes marketing")
    camp_cols  = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                  "AcceptedCmp4","AcceptedCmp5","Response"]
    camp_names = ["Cmp 1","Cmp 2","Cmp 3","Cmp 4","Cmp 5","Derniere"]
    rates  = dff[camp_cols].mean() * 100
    totals = dff[camp_cols].sum()

    kc = st.columns(6)
    for col_k, name, rate, tot in zip(kc, camp_names, rates, totals):
        col_k.metric(name, f"{rate:.1f}%", f"{int(tot)} clients")

    ca1, ca2 = st.columns(2)
    with ca1:
        sec("Taux d'acceptation par campagne")
        fig, ax = nfig(7, 4)
        bars = ax.bar(camp_names, rates.values,
                      color=PAL[:6], edgecolor=BG, alpha=.88)
        for b, v in zip(bars, rates.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.15,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="600")
        th(ax, "Taux d'acceptation (%)", yl="%")
        ax.set_ylim(0, rates.max()*1.25)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with ca2:
        sec("Cumul d'acceptations par client")
        fig, ax = nfig(7, 4)
        vc = dff["TotalCamp"].value_counts().sort_index()
        bars = ax.bar(vc.index.astype(int), vc.values,
                      color=PAL[0], edgecolor=BG, alpha=.88)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    str(v), ha="center", fontsize=9, fontweight="600")
        th(ax, "Nb de campagnes acceptees par client",
           xl="Nb campagnes", yl="Nb clients")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    sec("Profil des acceptants par campagne")
    camp_chosen = st.selectbox(
        "Analyser la campagne", camp_cols,
        format_func=lambda x: (
            x.replace("AcceptedCmp","Campagne ")
             .replace("Response","Derniere campagne")))

    acc  = dff[dff[camp_chosen] == 1]
    nacc = dff[dff[camp_chosen] == 0]
    cc1, cc2, cc3 = st.columns(3)
    for col_c, var, title in zip(
            [cc1,cc2,cc3],
            ["Income","TotalSpend","Age"],
            ["Revenu","Depense totale","Age"]):
        with col_c:
            fig, ax = nfig(4, 3)
            ax.hist(nacc[var], bins=25, alpha=.6, color=PAL[3],
                    label="Non-acceptants", edgecolor=BG, lw=.3)
            ax.hist(acc[var],  bins=25, alpha=.6, color=PAL[0],
                    label="Acceptants", edgecolor=BG, lw=.3)
            ax.legend(fontsize=8)
            th(ax, title)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    sec("Analyse des canaux d'achat")
    chan_cols  = ["NumWebPurchases","NumCatalogPurchases",
                  "NumStorePurchases","NumDealsPurchases"]
    chan_names = ["Web","Catalogue","Magasin","Promotions"]
    ch1, ch2 = st.columns(2)

    with ch1:
        chan_resp = dff.groupby("Response")[chan_cols].mean()
        chan_resp.columns = chan_names
        x  = np.arange(len(chan_names)); bw = .35
        fig, ax = nfig(6, 4)
        ax.bar(x-bw/2, chan_resp.loc[0], bw,
               label="Non convertis", color=PAL[3], alpha=.85, edgecolor=BG)
        ax.bar(x+bw/2, chan_resp.loc[1], bw,
               label="Convertis", color=PAL[0], alpha=.85, edgecolor=BG)
        ax.set_xticks(x); ax.set_xticklabels(chan_names)
        ax.legend(fontsize=9)
        th(ax, "Achats par canal x Conversion", yl="Nb moy.")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with ch2:
        chan_total = dff[chan_cols].sum()
        chan_total.index = chan_names
        fig, ax = nfig(5, 4)
        wedges, texts, autotexts = ax.pie(
            chan_total.values, labels=chan_names, colors=PAL[:4],
            autopct="%1.1f%%", startangle=120, pctdistance=.82,
            wedgeprops=dict(linewidth=1.5, edgecolor=BG))
        for t in texts:
            t.set_color(TEXT); t.set_fontsize(9)
        for at in autotexts:
            at.set_color(BG); at.set_fontsize(8); at.set_fontweight("600")
        ax.set_title("Repartition des canaux", color="#e4e8ff", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    sec("Evolution des inscriptions dans le temps")
    dff_t = dff.copy()
    dff_t["YearMonth"] = dff_t["Dt_Customer"].dt.to_period("M").astype(str)
    monthly = dff_t.groupby("YearMonth").agg(
        Inscriptions=("ID","count"),
        Conv_rate=("Response","mean")
    ).reset_index()
    monthly["Conv_rate"] *= 100

    fig, ax1 = nfig(13, 4)
    ax2 = ax1.twinx()
    ax1.bar(monthly["YearMonth"], monthly["Inscriptions"],
            color=PAL[0], alpha=.65, edgecolor=BG, label="Inscriptions")
    ax2.plot(monthly["YearMonth"], monthly["Conv_rate"],
             color=PAL[2], lw=2, marker="o", ms=4, label="Taux conv. (%)")
    ax1.set_ylabel("Inscriptions", color=TEXT)
    ax2.set_ylabel("Taux conv. (%)", color=PAL[2])
    ax2.tick_params(colors=PAL[2])
    th(ax1, "Inscriptions mensuelles et taux de conversion")
    step = max(1, len(monthly)//12)
    ax1.set_xticks(range(0, len(monthly), step))
    ax1.set_xticklabels(monthly["YearMonth"].iloc[::step],
                        rotation=35, ha="right", fontsize=8)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=9, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ======================================================================
# ONGLET 8 — INSIGHTS
# ======================================================================
with tabs[7]:
    sec("Synthese automatique des insights")

    conv          = dff["Response"].mean() * 100
    best_edu      = dff.groupby("Education")["Response"].mean().idxmax()
    best_edu_rate = dff.groupby("Education")["Response"].mean().max() * 100
    best_inc      = dff.groupby("Income_Group")["TotalSpend"].mean().idxmax()
    best_rfm      = dff.groupby("RFM_Seg")["Response"].mean().idxmax()
    best_rfm_rate = dff.groupby("RFM_Seg")["Response"].mean().max() * 100
    top_canal     = pd.Series({
        "Web":       dff["NumWebPurchases"].mean(),
        "Catalogue": dff["NumCatalogPurchases"].mean(),
        "Magasin":   dff["NumStorePurchases"].mean(),
    }).idxmax()
    no_child = dff[dff["HasChildren"]==0]["TotalSpend"].mean()
    w_child  = dff[dff["HasChildren"]==1]["TotalSpend"].mean()
    best_cmp = pd.Series({
        "Cmp1":dff["AcceptedCmp1"].mean(),
        "Cmp2":dff["AcceptedCmp2"].mean(),
        "Cmp3":dff["AcceptedCmp3"].mean(),
        "Cmp4":dff["AcceptedCmp4"].mean(),
        "Cmp5":dff["AcceptedCmp5"].mean(),
    }).idxmax()
    ratio = f"{(no_child/w_child-1)*100:.0f}%" if w_child > 0 else "N/A"

    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class='insight good'><b>Segment le plus receptif</b><br>
        Les clients <b>{best_edu}</b> ont le taux de conversion le plus eleve
        a <b>{best_edu_rate:.1f}%</b>.
        Prioriser ce segment pour les prochaines campagnes.</div>

        <div class='insight good'><b>Champion RFM</b><br>
        Le segment <b>{best_rfm}</b> convertit a <b>{best_rfm_rate:.1f}%</b>.
        Ce sont vos clients les plus precieux — leur proposer des offres exclusives.</div>

        <div class='insight'><b>Revenu et depenses</b><br>
        Les clients du groupe <b>{best_inc}</b> depensent le plus.
        Un ciblage premium sur ce segment maximisera le ROI.</div>

        <div class='insight'><b>Canal dominant</b><br>
        Le canal <b>{top_canal}</b> concentre le plus d'achats.
        Investir davantage dans ce canal pour augmenter les conversions.</div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
        <div class='insight warn'><b>Taux de conversion global</b><br>
        Seulement <b>{conv:.1f}%</b> des clients repondent aux campagnes.
        Un meilleur ciblage par segment RFM pourrait tripler ce taux sur les Champions.</div>

        <div class='insight warn'><b>Impact des enfants</b><br>
        Sans enfants : <b>{no_child:.0f} EUR</b> de depense moy.
        vs <b>{w_child:.0f} EUR</b> avec enfants ({ratio} de panier en plus).</div>

        <div class='insight alert'><b>Campagne la plus efficace</b><br>
        La <b>{best_cmp}</b> a le meilleur taux d'acceptation.
        Analyser son contenu pour reproduire la formule.</div>

        <div class='insight'><b>Predicteurs cles</b><br>
        <b>TotalSpend</b>, <b>Income</b> et <b>TotalCampaigns</b> sont les variables
        les plus correlees a la conversion.</div>
        """, unsafe_allow_html=True)

    sec("Recommandations strategiques")
    nb_champ    = dff[dff["RFM_Seg"]=="Champions"].shape[0]
    dep_premium = dff[dff["Income_Group"]==">90k"]["TotalSpend"].mean()
    nb_inactif  = dff[dff["RFM_Seg"]=="Inactifs"].shape[0]
    reco = [
        ("Cibler les Champions RFM",
         f"Lancer une campagne exclusive sur les {nb_champ} Champions — "
         "taux de conversion attendu > 25%"),
        ("Segmenter par revenu",
         f"Les clients >90k EUR depensent en moyenne {dep_premium:.0f} EUR — "
         "creer des offres premium"),
        (f"Optimiser le canal {top_canal}",
         "Renforcer l'investissement sur le canal le plus utilise"),
        ("Reengager les Inactifs",
         f"{nb_inactif} clients inactifs a reengager avec une offre de bienvenue"),
        ("Deployer le modele ML",
         "Utiliser le Random Forest pour scorer chaque prospect avant envoi de campagne"),
        ("Analyse RFM mensuelle",
         "Mettre en place un suivi mensuel des segments RFM"),
    ]
    for title, detail in reco:
        st.markdown(
            f"<div class='insight'><b>{title}</b><br>{detail}</div>",
            unsafe_allow_html=True)

    sec("Export des donnees")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button(
            "Telecharger dataset filtre (CSV)",
            dff.to_csv(index=False).encode("utf-8"),
            "marketing_filtered.csv", "text/csv",
            use_container_width=True)
    with ex2:
        rfm_exp = dff[["ID","RFM_Seg","R","F","M","RFM","TotalSpend","Income","Response"]]
        st.download_button(
            "Telecharger segments RFM (CSV)",
            rfm_exp.to_csv(index=False).encode("utf-8"),
            "rfm_segments.csv", "text/csv",
            use_container_width=True)
    with ex3:
        summary = dff.describe().T.reset_index()
        st.download_button(
            "Telecharger resume statistique (CSV)",
            summary.to_csv(index=False).encode("utf-8"),
            "summary_stats.csv", "text/csv",
            use_container_width=True)
