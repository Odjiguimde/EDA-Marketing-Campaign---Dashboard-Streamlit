import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Marketing Campaign",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fonts & base */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * { color: #e0e4ef !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label { color: #8892b0 !important; font-size: 0.78rem !important; letter-spacing: .06em; text-transform: uppercase; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #252840;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: .07em; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0e4ef !important; font-size: 1.6rem !important; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 1px solid #252840; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8892b0; border-radius: 8px 8px 0 0; padding: .5rem 1.2rem; font-size: 0.85rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #1a1d2e !important; color: #7b9cff !important; border-bottom: 2px solid #7b9cff; }

    /* Section headers */
    .section-header {
        font-size: 0.72rem; font-weight: 600; letter-spacing: .12em;
        text-transform: uppercase; color: #7b9cff;
        border-left: 3px solid #7b9cff; padding-left: .6rem;
        margin: 1.5rem 0 .8rem;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] { border: 1px solid #252840; border-radius: 10px; overflow: hidden; }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1a1d2e 0%, #141625 100%);
        border: 1px solid #252840; border-left: 3px solid #7b9cff;
        border-radius: 10px; padding: 1rem 1.2rem; margin: .5rem 0;
        font-size: .88rem; line-height: 1.6; color: #c0c8e0;
    }
    .insight-box b { color: #7b9cff; }

    /* Chart containers */
    .chart-card {
        background: #1a1d2e; border: 1px solid #252840;
        border-radius: 12px; padding: 1.2rem;
    }

    /* Main background */
    .main .block-container { background: #0d0f1a; padding-top: 1.5rem; }

    /* Hide default footer */
    footer { visibility: hidden; }

    /* Warning / info */
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────────
PALETTE = ["#7b9cff", "#64dfdf", "#f4a261", "#e76f51", "#a8dadc", "#52b788", "#ffd166", "#ef476f"]
BG      = "#1a1d2e"
GRID    = "#252840"
TEXT    = "#c0c8e0"

def apply_theme(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_color(TEXT); ax.yaxis.label.set_size(10)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, pad=10, fontweight="600")
    if xlabel: ax.set_xlabel(xlabel, color=TEXT)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=.6, linestyle="--", alpha=.5)
    ax.grid(axis="x", visible=False)

def new_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

# ── Data loading & cleaning ────────────────────────────────────────────────────
@st.cache_data
def load_data(filepath=None, uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=";")
    else:
        df = pd.read_csv(filepath, sep=";")

    # --- Feature engineering ---
    df["Age"]             = datetime.now().year - df["Year_Birth"]
    df["TotalSpend"]      = df[["MntWines","MntFruits","MntMeatProducts",
                                 "MntFishProducts","MntSweetProducts","MntGoldProds"]].sum(axis=1)
    df["TotalPurchases"]  = df[["NumDealsPurchases","NumWebPurchases",
                                 "NumCatalogPurchases","NumStorePurchases"]].sum(axis=1)
    df["TotalCampaigns"]  = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                                 "AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)
    df["HasChildren"]     = (df["Kidhome"] + df["Teenhome"]).clip(upper=1)
    df["Dt_Customer"]     = pd.to_datetime(df["Dt_Customer"])
    df["Seniority"]       = (pd.Timestamp.now() - df["Dt_Customer"]).dt.days // 30  # months

    # Clean Marital Status
    df["Marital_Clean"] = df["Marital_Status"].replace(
        {"Alone": "Single", "Absurd": "Other", "YOLO": "Other"})

    # Clean Income (fill nulls with median)
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Age bins
    df["Age_Group"] = pd.cut(df["Age"],
        bins=[17, 30, 45, 60, 100],
        labels=["18–30", "31–45", "46–60", "60+"])

    # Income bins
    df["Income_Group"] = pd.cut(df["Income"],
        bins=[0, 30000, 60000, 90000, 200000],
        labels=["<30k", "30–60k", "60–90k", ">90k"])

    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 EDA Marketing")
    st.markdown("---")

    uploaded = st.file_uploader("Charger un autre CSV", type=["csv"],
                                 help="Format attendu : séparateur `;`")

    try:
        df = load_data("marketing_campaign.csv") if uploaded is None else load_data(uploaded=uploaded)
        data_ok = True
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        data_ok = False

    if data_ok:
        st.markdown('<div class="section-header">Filtres</div>', unsafe_allow_html=True)

        edu_opts = ["Tous"] + sorted(df["Education"].dropna().unique().tolist())
        sel_edu  = st.selectbox("Niveau d'éducation", edu_opts)

        mar_opts = ["Tous"] + sorted(df["Marital_Clean"].dropna().unique().tolist())
        sel_mar  = st.selectbox("Statut marital", mar_opts)

        age_range = st.slider("Tranche d'âge",
            int(df["Age"].min()), int(df["Age"].max()),
            (int(df["Age"].min()), int(df["Age"].max())))

        inc_range = st.slider("Revenu (€)",
            int(df["Income"].min()), int(df["Income"].max()),
            (int(df["Income"].min()), int(df["Income"].max())),
            step=1000, format="%d€")

        response_filter = st.radio("Réponse dernière campagne",
            ["Tous", "A répondu (1)", "N'a pas répondu (0)"],
            horizontal=False)

        st.markdown("---")
        st.markdown(f"<small style='color:#8892b0'>Dataset : **{df.shape[0]}** clients · **{df.shape[1]}** variables</small>", unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
if data_ok:
    dff = df.copy()
    if sel_edu != "Tous":      dff = dff[dff["Education"]    == sel_edu]
    if sel_mar != "Tous":      dff = dff[dff["Marital_Clean"] == sel_mar]
    dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]
    dff = dff[(dff["Income"] >= inc_range[0]) & (dff["Income"] <= inc_range[1])]
    if response_filter == "A répondu (1)":       dff = dff[dff["Response"] == 1]
    elif response_filter == "N'a pas répondu (0)": dff = dff[dff["Response"] == 0]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if data_ok:
    st.markdown("# Analyse Exploratoire — Campagnes Marketing")
    st.markdown(f"<small style='color:#8892b0'>{dff.shape[0]} clients sélectionnés sur {df.shape[0]}</small>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clients", f"{dff.shape[0]:,}")
    k2.metric("Revenu médian", f"{dff['Income'].median():,.0f} €")
    k3.metric("Dépense moyenne", f"{dff['TotalSpend'].mean():,.0f} €")
    k4.metric("Taux de conversion", f"{dff['Response'].mean()*100:.1f}%")
    k5.metric("Campagnes acceptées", f"{dff['TotalCampaigns'].mean():.2f} / client")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂 Aperçu des données",
        "📈 Analyse univariée",
        "🔗 Analyse multivariée",
        "🎯 Segments clients",
        "💡 Insights",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Aperçu
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Aperçu du dataset</div>', unsafe_allow_html=True)
        st.dataframe(dff.head(10), use_container_width=True, height=280)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-header">Dimensions</div>', unsafe_allow_html=True)
            st.write(f"**Lignes :** {dff.shape[0]:,}")
            st.write(f"**Colonnes :** {dff.shape[1]}")
        with c2:
            st.markdown('<div class="section-header">Valeurs manquantes</div>', unsafe_allow_html=True)
            nulls = dff.isnull().sum()
            nulls = nulls[nulls > 0]
            if nulls.empty:
                st.success("Aucune valeur manquante ✅")
            else:
                st.dataframe(nulls.rename("Nulls"), use_container_width=True)
        with c3:
            st.markdown('<div class="section-header">Doublons</div>', unsafe_allow_html=True)
            dups = dff.duplicated().sum()
            if dups == 0:
                st.success("Aucun doublon ✅")
            else:
                st.warning(f"{dups} doublons détectés")

        st.markdown('<div class="section-header">Statistiques descriptives</div>', unsafe_allow_html=True)
        num_cols = dff.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(dff[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

        st.markdown('<div class="section-header">Types de variables</div>', unsafe_allow_html=True)
        type_df = pd.DataFrame({
            "Variable": dff.dtypes.index,
            "Type": dff.dtypes.values.astype(str),
            "Valeurs uniques": [dff[c].nunique() for c in dff.columns],
            "Exemple": [str(dff[c].dropna().iloc[0]) if not dff[c].dropna().empty else "—" for c in dff.columns]
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Univarié
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Variables numériques</div>', unsafe_allow_html=True)

        num_choice = st.selectbox("Choisir une variable", [
            "Age", "Income", "TotalSpend", "TotalPurchases",
            "Recency", "MntWines", "MntMeatProducts", "MntFruits",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds",
            "NumWebVisitsMonth", "Seniority"])

        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = new_fig(6, 4)
            ax.hist(dff[num_choice].dropna(), bins=35, color=PALETTE[0], alpha=.85, edgecolor=BG, linewidth=.4)
            apply_theme(ax, title=f"Distribution — {num_choice}", xlabel=num_choice, ylabel="Fréquence")
            # Add mean/median lines
            mean_v = dff[num_choice].mean()
            med_v  = dff[num_choice].median()
            ax.axvline(mean_v, color=PALETTE[2], linestyle="--", linewidth=1.2, label=f"Moyenne: {mean_v:.0f}")
            ax.axvline(med_v,  color=PALETTE[3], linestyle=":",  linewidth=1.2, label=f"Médiane: {med_v:.0f}")
            ax.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor=GRID)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            fig, ax = new_fig(6, 4)
            bp = ax.boxplot(dff[num_choice].dropna(), vert=True, patch_artist=True,
                            medianprops=dict(color=PALETTE[2], linewidth=2),
                            boxprops=dict(facecolor=PALETTE[0], alpha=.5, linewidth=.8),
                            whiskerprops=dict(color=TEXT, linewidth=.8),
                            capprops=dict(color=TEXT, linewidth=.8),
                            flierprops=dict(marker="o", color=PALETTE[3], alpha=.4, markersize=3))
            apply_theme(ax, title=f"Boxplot — {num_choice}", ylabel=num_choice)
            ax.set_xticklabels([num_choice])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Stats card
        s = dff[num_choice].describe()
        skew = dff[num_choice].skew()
        st.markdown(f"""
        <div class="insight-box">
        📌 <b>Statistiques — {num_choice}</b><br>
        Moyenne: <b>{s['mean']:.2f}</b> · Médiane: <b>{s['50%']:.2f}</b> · Écart-type: <b>{s['std']:.2f}</b>
        · Min: <b>{s['min']:.0f}</b> · Max: <b>{s['max']:.0f}</b>
        · Asymétrie (skewness): <b>{skew:.2f}</b>
        {"→ Distribution asymétrique à droite (queue longue vers les hauts revenus)" if skew > 1 else "→ Distribution relativement symétrique" if abs(skew) < 0.5 else "→ Distribution légèrement asymétrique"}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Variables catégorielles</div>', unsafe_allow_html=True)
        cat_choice = st.selectbox("Choisir une variable", ["Education","Marital_Clean","Age_Group","Income_Group","HasChildren"])

        col_c, col_d = st.columns(2)
        with col_c:
            counts = dff[cat_choice].value_counts()
            fig, ax = new_fig(6, 4)
            bars = ax.barh(counts.index.astype(str), counts.values,
                           color=PALETTE[:len(counts)], edgecolor=BG, linewidth=.4)
            for bar, val in zip(bars, counts.values):
                ax.text(val + counts.values.max()*0.01, bar.get_y() + bar.get_height()/2,
                        f" {val}", va="center", color=TEXT, fontsize=9)
            apply_theme(ax, title=f"Effectifs — {cat_choice}", xlabel="Nombre de clients")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_d:
            fig, ax = new_fig(5, 4)
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index.astype(str),
                colors=PALETTE[:len(counts)], autopct="%1.1f%%",
                startangle=140, pctdistance=0.82,
                wedgeprops=dict(linewidth=1.5, edgecolor=BG))
            for t in texts: t.set_color(TEXT); t.set_fontsize(9)
            for at in autotexts: at.set_color(BG); at.set_fontsize(8); at.set_fontweight("600")
            ax.set_title(f"Répartition — {cat_choice}", color=TEXT, fontsize=11, fontweight="600")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Multivarié
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Matrice de corrélation</div>', unsafe_allow_html=True)

        corr_cols = ["Age","Income","TotalSpend","TotalPurchases","Recency",
                     "MntWines","MntMeatProducts","MntFruits","MntGoldProds",
                     "NumWebVisitsMonth","TotalCampaigns","Response"]
        corr = dff[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 8, "color": TEXT},
                    linewidths=.5, linecolor=GRID, ax=ax,
                    cbar_kws={"shrink": .7})
        ax.set_title("Corrélations entre variables numériques", color=TEXT, fontsize=12, fontweight="600", pad=12)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.collections[0].colorbar.ax.tick_params(colors=TEXT, labelsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top correlations with Response
        st.markdown('<div class="section-header">Corrélations avec la variable cible (Response)</div>', unsafe_allow_html=True)
        resp_corr = corr["Response"].drop("Response").sort_values(key=abs, ascending=False)
        fig, ax = new_fig(10, 4)
        colors_bar = [PALETTE[0] if v > 0 else PALETTE[3] for v in resp_corr.values]
        ax.bar(resp_corr.index, resp_corr.values, color=colors_bar, edgecolor=BG, linewidth=.4)
        ax.axhline(0, color=TEXT, linewidth=.5)
        apply_theme(ax, title="Corrélation avec la réponse à la campagne", ylabel="Corrélation de Pearson")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-header">Scatter plot interactif</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([2,2,1])
        with sc1: x_var = st.selectbox("Axe X", ["Income","Age","TotalSpend","Recency","Seniority"], key="sx")
        with sc2: y_var = st.selectbox("Axe Y", ["TotalSpend","TotalPurchases","MntWines","Income","Recency"], key="sy")
        with sc3: hue_var = st.selectbox("Couleur", ["Response","Education","Age_Group"], key="sh")

        fig, ax = new_fig(10, 5)
        cats = dff[hue_var].unique()
        pal  = dict(zip(cats, PALETTE[:len(cats)]))
        for cat in cats:
            sub = dff[dff[hue_var]==cat]
            ax.scatter(sub[x_var], sub[y_var], label=str(cat),
                       color=pal[cat], alpha=.55, s=22, edgecolors="none")
        apply_theme(ax, title=f"{x_var} vs {y_var} (coloré par {hue_var})",
                    xlabel=x_var, ylabel=y_var)
        ax.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor=GRID, title=hue_var,
                  title_fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-header">Dépenses par catégorie et éducation</div>', unsafe_allow_html=True)
        spend_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
        spend_by_edu = dff.groupby("Education")[spend_cols].mean()
        fig, ax = new_fig(10, 5)
        x  = np.arange(len(spend_cols))
        bw = .15
        for i, (edu, row) in enumerate(spend_by_edu.iterrows()):
            ax.bar(x + i*bw, row.values, bw, label=edu, color=PALETTE[i], alpha=.85, edgecolor=BG)
        ax.set_xticks(x + bw*2)
        ax.set_xticklabels([c.replace("Mnt","") for c in spend_cols], rotation=20, ha="right", color=TEXT)
        apply_theme(ax, title="Dépenses moyennes par catégorie de produit et niveau d'éducation", ylabel="Montant moyen (€)")
        ax.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor=GRID)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Segments clients
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">Taux de conversion par segment</div>', unsafe_allow_html=True)

        seg_var = st.selectbox("Segmenter par", ["Education","Marital_Clean","Age_Group","Income_Group","HasChildren"])
        conv_df = dff.groupby(seg_var).agg(
            Clients=("Response","count"),
            Convertis=("Response","sum"),
            Taux_conversion=("Response","mean"),
            Revenu_moyen=("Income","mean"),
            Depense_moyenne=("TotalSpend","mean"),
        ).reset_index()
        conv_df["Taux_conversion"] = (conv_df["Taux_conversion"]*100).round(1)
        conv_df["Revenu_moyen"]    = conv_df["Revenu_moyen"].round(0)
        conv_df["Depense_moyenne"] = conv_df["Depense_moyenne"].round(0)
        conv_df = conv_df.sort_values("Taux_conversion", ascending=False)

        # Table
        st.dataframe(conv_df.set_index(seg_var).style.format({
            "Taux_conversion": "{:.1f}%",
            "Revenu_moyen": "{:,.0f} €",
            "Depense_moyenne": "{:,.0f} €",
        }).background_gradient(subset=["Taux_conversion"], cmap="Blues"),
        use_container_width=True)

        # Bar chart conversion
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(BG)

        ax1 = axes[0]
        ax1.set_facecolor(BG)
        bars1 = ax1.bar(conv_df[seg_var].astype(str), conv_df["Taux_conversion"],
                        color=PALETTE[:len(conv_df)], edgecolor=BG)
        for bar, val in zip(bars1, conv_df["Taux_conversion"]):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2,
                     f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="600")
        apply_theme(ax1, title="Taux de conversion (%)", ylabel="%")
        ax1.tick_params(axis="x", rotation=20)

        ax2 = axes[1]
        ax2.set_facecolor(BG)
        bars2 = ax2.bar(conv_df[seg_var].astype(str), conv_df["Depense_moyenne"],
                        color=PALETTE[:len(conv_df)], edgecolor=BG, alpha=.8)
        apply_theme(ax2, title="Dépense moyenne (€)", ylabel="€")
        ax2.tick_params(axis="x", rotation=20)
        fig.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-header">Performance des campagnes</div>', unsafe_allow_html=True)
        camp_cols  = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]
        camp_rates = dff[camp_cols].mean() * 100
        fig, ax = new_fig(10, 4)
        labels = [f"Cmp {i}" for i in range(1,6)] + ["Dernière\ncampagne"]
        bars = ax.bar(labels, camp_rates.values, color=PALETTE[:6], edgecolor=BG, alpha=.85)
        for bar, val in zip(bars, camp_rates.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.1,
                    f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="600")
        apply_theme(ax, title="Taux d'acceptation par campagne (%)", ylabel="%")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-header">Canal d\'achat préféré</div>', unsafe_allow_html=True)
        channel_cols  = ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases"]
        channel_names = ["Web", "Catalogue", "Magasin", "Promotions"]
        chan_total = dff[channel_cols].sum()

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            fig, ax = new_fig(5,4)
            ax.pie(chan_total.values, labels=channel_names, colors=PALETTE[:4],
                   autopct="%1.1f%%", startangle=120, pctdistance=.82,
                   wedgeprops=dict(linewidth=1.5, edgecolor=BG))
            ax.set_title("Part de chaque canal", color=TEXT, fontsize=11, fontweight="600")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_ch2:
            chan_by_resp = dff.groupby("Response")[channel_cols].mean()
            chan_by_resp.columns = channel_names
            x = np.arange(len(channel_names)); bw = .35
            fig, ax = new_fig(6,4)
            ax.bar(x - bw/2, chan_by_resp.loc[0], bw, label="Non-convertis", color=PALETTE[3], alpha=.8, edgecolor=BG)
            ax.bar(x + bw/2, chan_by_resp.loc[1], bw, label="Convertis",     color=PALETTE[0], alpha=.8, edgecolor=BG)
            ax.set_xticks(x); ax.set_xticklabels(channel_names)
            apply_theme(ax, title="Achats moyens par canal & conversion", ylabel="Nb achats moy.")
            ax.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor=GRID)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ════════════════════════════════════════════════════════════════
    # TAB 5 — Insights
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">Synthèse & recommandations</div>', unsafe_allow_html=True)

        best_seg  = dff.groupby("Education")["Response"].mean().idxmax()
        best_rate = dff.groupby("Education")["Response"].mean().max()*100
        top_spend = dff.groupby("Education")["TotalSpend"].mean().idxmax()
        conv_rate = dff["Response"].mean()*100
        top_canal = pd.Series({
            "Web": dff["NumWebPurchases"].mean(),
            "Catalogue": dff["NumCatalogPurchases"].mean(),
            "Magasin": dff["NumStorePurchases"].mean(),
        }).idxmax()
        rich_grp  = dff.groupby("Income_Group")["TotalSpend"].mean().idxmax()

        insights = [
            (f"Le segment <b>{best_seg}</b> présente le meilleur taux de conversion à <b>{best_rate:.1f}%</b>. "
             "Prioriser ce segment dans les prochaines campagnes."),
            (f"Le taux de conversion global est de <b>{conv_rate:.1f}%</b>, "
             "indiquant un fort potentiel d'optimisation — seulement 1 client sur "
             f"{int(100/conv_rate) if conv_rate>0 else '?'} répond aux campagnes."),
            (f"Les clients du segment revenu <b>{rich_grp}</b> dépensent en moyenne le plus. "
             "Cibler ce groupe avec des offres premium."),
            (f"Le canal <b>{top_canal}</b> génère le plus d'achats. "
             "Augmenter les investissements marketing sur ce canal."),
            (f"Les clients sans enfants (<b>HasChildren=0</b>) ont un panier moyen "
             f"de <b>{dff[dff['HasChildren']==0]['TotalSpend'].mean():.0f}€</b> vs "
             f"<b>{dff[dff['HasChildren']==1]['TotalSpend'].mean():.0f}€</b> pour ceux avec enfants."),
            ("La variable <b>Income</b> est fortement corrélée aux dépenses. "
             "Un modèle prédictif de conversion devrait inclure le revenu comme feature principale."),
            ("Les clients avec plus d'ancienneté (<b>Seniority</b>) tendent à accepter plus de campagnes. "
             "Mettre en place des programmes de fidélité pour les clients récents."),
        ]

        for ins in insights:
            st.markdown(f'<div class="insight-box">→ {ins}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Export des données filtrées</div>', unsafe_allow_html=True)
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_out = dff.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV filtré", csv_out, "marketing_filtered.csv", "text/csv",
                               use_container_width=True)
        with col_ex2:
            summary = dff.describe().T.reset_index().rename(columns={"index":"variable"})
            sum_csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger résumé statistique", sum_csv, "summary_stats.csv", "text/csv",
                               use_container_width=True)

        st.markdown('<div class="section-header">Prochaines étapes suggérées</div>', unsafe_allow_html=True)
        steps = [
            "**Modélisation prédictive** : Entraîner un modèle de classification (Random Forest, XGBoost) pour prédire la réponse aux campagnes.",
            "**Clustering** : Appliquer K-Means ou DBSCAN pour identifier des segments clients homogènes.",
            "**Test A/B** : Comparer les taux de conversion entre segments identifiés pour valider les hypothèses.",
            "**Feature engineering** : Créer des variables RFM (Récence, Fréquence, Montant) pour affiner l'analyse.",
            "**Dashboard temps réel** : Connecter l'application à une base de données pour des mises à jour automatiques.",
        ]
        for s in steps:
            st.markdown(f"- {s}")

else:
    st.error("Impossible de charger le dataset. Vérifiez le fichier.")
