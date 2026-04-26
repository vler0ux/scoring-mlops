import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scoring Monitor · Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'IBM Plex Sans', sans-serif;
  }
  .stApp { background: #0d1117; color: #e6edf3; }

  /* Header */
  .dash-header {
      background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
  }
  .dash-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin: 0; }
  .dash-sub { font-size: 0.85rem; color: #8b949e; font-family: 'IBM Plex Mono', monospace; margin: 0; }

  /* KPI cards */
  .kpi-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 1.2rem 1.5rem;
      text-align: center;
      transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #58a6ff; }
  .kpi-value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
  .kpi-delta { font-size: 0.8rem; margin-top: 0.4rem; }

  /* Alert banner */
  .alert-warn {
      background: rgba(210, 153, 34, 0.12);
      border: 1px solid #d29922;
      border-radius: 8px;
      padding: 0.7rem 1rem;
      color: #e3b341;
      font-size: 0.85rem;
      margin-bottom: 1rem;
  }
  .alert-ok {
      background: rgba(35, 134, 54, 0.12);
      border: 1px solid #238636;
      border-radius: 8px;
      padding: 0.7rem 1rem;
      color: #3fb950;
      font-size: 0.85rem;
      margin-bottom: 1rem;
  }

  /* Section titles */
  .section-title {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #8b949e;
      border-bottom: 1px solid #30363d;
      padding-bottom: 0.5rem;
      margin-bottom: 1rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #161b22 !important;
      border-right: 1px solid #30363d;
  }

  /* Plotly charts background */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(color="#8b949e", family="IBM Plex Sans"),
    margin=dict(t=30, b=30, l=10, r=10),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
)

COLOR_ACCORD  = "#3fb950"
COLOR_REFUSE  = "#f85149"
COLOR_SCORE   = "#58a6ff"
COLOR_LATENCY = "#d2a8ff"
COLOR_WARN    = "#e3b341"

SEUIL_DEFAULT = 0.519

@st.cache_data(ttl=30)
def load_logs(path: str) -> pd.DataFrame:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["accordé"]   = df["decision"].str.contains("Accord")
    df["age_ans"]   = (df["input.DAYS_BIRTH"] / 365).round(1)
    df["revenu_k"]  = (df["input.AMT_INCOME_TOTAL"] / 1000).round(1)
    df["credit_k"]  = (df["input.AMT_CREDIT"] / 1000).round(1)
    df["idx"]       = range(len(df))
    return df


def generate_demo_data(n: int = 200) -> pd.DataFrame:
    """Données synthétiques si le fichier de logs est absent (démo cloud)."""
    np.random.seed(42)
    t0 = datetime.utcnow() - timedelta(minutes=n)
    ts = [t0 + timedelta(seconds=i * 18) for i in range(n)]

    # Simulation d'un drift progressif → scores augmentent après n//2
    scores_base  = np.random.beta(2, 5, n // 2)
    scores_drift = np.random.beta(3, 3, n - n // 2)
    scores = np.concatenate([scores_base, scores_drift])

    income  = np.random.lognormal(11, 0.4, n)
    credit  = income * np.random.uniform(1, 4, n)
    annuity = credit / np.random.uniform(8, 15, n)
    days_birth = np.random.randint(7000, 20000, n).astype(float)

    seuil = SEUIL_DEFAULT
    df = pd.DataFrame({
        "timestamp":             ts,
        "score":                 np.clip(scores, 0, 1),
        "seuil":                 seuil,
        "inference_time_ms":     np.random.gamma(4, 25, n),
        "accordé":               scores < seuil,
        "decision":              ["✅ Accordé" if s < seuil else "❌ Refusé" for s in scores],
        "input.AMT_INCOME_TOTAL": income,
        "input.AMT_CREDIT":       credit,
        "input.AMT_ANNUITY":      annuity,
        "input.DAYS_BIRTH":       days_birth,
        "input.EXT_SOURCE_1":     np.random.beta(5, 2, n),
        "input.EXT_SOURCE_2":     np.random.beta(5, 2, n),
        "input.EXT_SOURCE_3":     np.random.beta(5, 2, n),
        "input.CODE_GENDER":      np.random.choice(["M", "F"], n),
        "input.NAME_EDUCATION_TYPE": np.random.choice(
            ["Higher education", "Secondary / secondary special", "Incomplete higher"], n
        ),
    })
    df["age_ans"]  = (df["input.DAYS_BIRTH"] / 365).round(1)
    df["revenu_k"] = (df["input.AMT_INCOME_TOTAL"] / 1000).round(1)
    df["credit_k"] = (df["input.AMT_CREDIT"] / 1000).round(1)
    df["idx"]      = range(n)
    return df


def alert_html(msg: str, level: str = "warn") -> None:
    cls = "alert-warn" if level == "warn" else "alert-ok"
    icon = "⚠️" if level == "warn" else "✅"
    st.markdown(f'<div class="{cls}">{icon} {msg}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    log_path = st.text_input("Chemin du fichier de logs", value="logs/predictions.jsonl")
    seuil    = st.slider("Seuil de décision", 0.0, 1.0, SEUIL_DEFAULT, 0.01)
    window   = st.selectbox("Fenêtre d'analyse", ["Toutes les requêtes", "50 dernières", "100 dernières"])
    st.divider()
    st.markdown("### 🚨 Alertes")
    alert_latence  = st.number_input("Latence max (ms)", value=200, step=10)
    alert_refus    = st.slider("Taux de refus max (%)", 0, 100, 60)
    alert_score    = st.slider("Score moyen max", 0.0, 1.0, 0.55, 0.01)
    st.divider()
    refresh = st.button("🔄 Rafraîchir les données")

# ── Load data ─────────────────────────────────────────────────────────────────
if refresh:
    st.cache_data.clear()

df_full = load_logs(log_path)
demo_mode = df_full.empty

if demo_mode:
    df_full = generate_demo_data(200)

# Fenêtre
n_map = {"Toutes les requêtes": len(df_full), "50 dernières": 50, "100 dernières": 100}
df = df_full.tail(n_map[window]).copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <div>
    <p class="dash-title">📊 Scoring Monitor · Prêt à Dépenser</p>
    <p class="dash-sub">
      {"🟡 MODE DÉMO — fichier introuvable · données synthétiques" if demo_mode else f"🟢 LIVE · {log_path}"}
      &nbsp;|&nbsp; {len(df)} requêtes · seuil = {seuil}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
total       = len(df)
n_accordes  = df["accordé"].sum()
n_refuses   = total - n_accordes
taux_refus  = n_refuses / total * 100
score_moy   = df["score"].mean()
lat_moy     = df["inference_time_ms"].mean()
lat_p95     = df["inference_time_ms"].quantile(0.95)

def kpi(value, label, color="#e6edf3", delta=None):
    delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""

cols = st.columns(6)
kpis = [
    (f"{total}", "Total requêtes", "#e6edf3", None),
    (f"{n_accordes}", "Accordés", COLOR_ACCORD, None),
    (f"{n_refuses}", "Refusés", COLOR_REFUSE, None),
    (f"{taux_refus:.1f}%", "Taux de refus", COLOR_WARN if taux_refus > alert_refus else COLOR_ACCORD, None),
    (f"{score_moy:.3f}", "Score moyen", COLOR_WARN if score_moy > alert_score else COLOR_SCORE, None),
    (f"{lat_moy:.0f}ms", "Latence moy.", COLOR_WARN if lat_moy > alert_latence else COLOR_LATENCY, f"p95: {lat_p95:.0f}ms"),
]
for col, (v, l, c, d) in zip(cols, kpis):
    with col:
        st.markdown(kpi(v, l, c, d), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Alertes automatiques ──────────────────────────────────────────────────────
if taux_refus > alert_refus:
    alert_html(f"Taux de refus élevé : {taux_refus:.1f}% (seuil : {alert_refus}%)", "warn")
if score_moy > alert_score:
    alert_html(f"Score moyen en hausse : {score_moy:.3f} (seuil : {alert_score})", "warn")
if lat_moy > alert_latence:
    alert_html(f"Latence moyenne trop haute : {lat_moy:.0f}ms (seuil : {alert_latence}ms)", "warn")
if taux_refus <= alert_refus and score_moy <= alert_score and lat_moy <= alert_latence:
    alert_html("Tous les indicateurs sont dans les limites normales.", "ok")

# ── Row 1 : Score dist + Timeline ─────────────────────────────────────────────
st.markdown('<div class="section-title">Distribution & Évolution des scores</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])

with c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["score"], nbinsx=30,
        marker_color=COLOR_SCORE, opacity=0.8,
        name="Score",
    ))
    fig.add_vline(x=seuil, line_dash="dash", line_color=COLOR_WARN,
                  annotation_text=f"Seuil {seuil}", annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, title="Distribution des scores", height=280)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["idx"], y=df["score"],
        mode="markers",
        marker=dict(
            color=df["score"],
            colorscale=[[0, COLOR_ACCORD], [0.5, COLOR_WARN], [1, COLOR_REFUSE]],
            size=6, opacity=0.8,
            colorbar=dict(title="Score", thickness=10),
        ),
        name="Score",
    ))
    fig2.add_hline(y=seuil, line_dash="dash", line_color=COLOR_WARN,
                   annotation_text=f"Seuil {seuil}")
    # Rolling average
    roll = df["score"].rolling(10, min_periods=1).mean()
    fig2.add_trace(go.Scatter(
        x=df["idx"], y=roll,
        mode="lines", line=dict(color="#e3b341", width=2, dash="dot"),
        name="Moy. mobile (10)"
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, title="Évolution des scores (avec moyenne mobile)", height=280)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 : Décisions + Latence ───────────────────────────────────────────────
st.markdown('<div class="section-title">Décisions & Latence</div>', unsafe_allow_html=True)

c3, c4, c5 = st.columns([1, 1, 2])

with c3:
    fig3 = go.Figure(go.Pie(
        labels=["Accordé", "Refusé"],
        values=[n_accordes, n_refuses],
        marker_colors=[COLOR_ACCORD, COLOR_REFUSE],
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(color="#e6edf3"),
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, title="Taux de décision", height=280,
                       showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = go.Figure()
    fig4.add_trace(go.Box(
        y=df["inference_time_ms"],
        marker_color=COLOR_LATENCY,
        name="Latence (ms)",
        boxmean=True,
    ))
    fig4.add_hline(y=alert_latence, line_dash="dash", line_color=COLOR_WARN,
                   annotation_text=f"Max {alert_latence}ms")
    fig4.update_layout(**PLOTLY_LAYOUT, title="Distribution latence", height=280)
    st.plotly_chart(fig4, use_container_width=True)

with c5:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df["idx"], y=df["inference_time_ms"],
        mode="lines+markers",
        line=dict(color=COLOR_LATENCY, width=1.5),
        marker=dict(size=4, color=COLOR_LATENCY),
        name="Latence",
    ))
    roll_lat = df["inference_time_ms"].rolling(10, min_periods=1).mean()
    fig5.add_trace(go.Scatter(
        x=df["idx"], y=roll_lat,
        mode="lines", line=dict(color=COLOR_WARN, width=2),
        name="Moy. mobile",
    ))
    fig5.add_hline(y=alert_latence, line_dash="dash", line_color=COLOR_REFUSE,
                   annotation_text=f"Seuil {alert_latence}ms")
    fig5.update_layout(**PLOTLY_LAYOUT, title="Latence par requête (ms)", height=280)
    st.plotly_chart(fig5, use_container_width=True)

# ── Row 3 : Features ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Distribution des features d\'entrée</div>', unsafe_allow_html=True)

c6, c7, c8 = st.columns(3)

with c6:
    fig6 = px.histogram(df, x="revenu_k", color="decision",
                        color_discrete_map={"✅ Accordé": COLOR_ACCORD, "❌ Refusé": COLOR_REFUSE},
                        nbins=25, barmode="overlay", opacity=0.75,
                        labels={"revenu_k": "Revenu (k€)"})
    fig6.update_layout(**PLOTLY_LAYOUT, title="Revenu par décision", height=250, showlegend=True)
    st.plotly_chart(fig6, use_container_width=True)

with c7:
    fig7 = px.histogram(df, x="credit_k", color="decision",
                        color_discrete_map={"✅ Accordé": COLOR_ACCORD, "❌ Refusé": COLOR_REFUSE},
                        nbins=25, barmode="overlay", opacity=0.75,
                        labels={"credit_k": "Crédit (k€)"})
    fig7.update_layout(**PLOTLY_LAYOUT, title="Montant crédit par décision", height=250, showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

with c8:
    fig8 = px.histogram(df, x="age_ans", color="decision",
                        color_discrete_map={"✅ Accordé": COLOR_ACCORD, "❌ Refusé": COLOR_REFUSE},
                        nbins=25, barmode="overlay", opacity=0.75,
                        labels={"age_ans": "Âge (ans)"})
    fig8.update_layout(**PLOTLY_LAYOUT, title="Âge par décision", height=250, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

# ── Row 4 : EXT_SOURCES + Genre + Education ──────────────────────────────────
st.markdown('<div class="section-title">Scores externes & profil emprunteur</div>', unsafe_allow_html=True)

c9, c10, c11 = st.columns(3)

with c9:
    fig9 = go.Figure()
    for col_name, color in [("input.EXT_SOURCE_1", "#58a6ff"),
                             ("input.EXT_SOURCE_2", "#d2a8ff"),
                             ("input.EXT_SOURCE_3", "#3fb950")]:
        fig9.add_trace(go.Violin(y=df[col_name], name=col_name.split(".")[-1],
                                 box_visible=True, meanline_visible=True,
                                 fillcolor=color, opacity=0.6, line_color=color))
    fig9.update_layout(**PLOTLY_LAYOUT, title="EXT_SOURCE distribution", height=260)
    st.plotly_chart(fig9, use_container_width=True)

with c10:
    gender_counts = df["input.CODE_GENDER"].value_counts()
    fig10 = go.Figure(go.Bar(
        x=gender_counts.index, y=gender_counts.values,
        marker_color=[COLOR_SCORE, COLOR_LATENCY],
        text=gender_counts.values, textposition="auto",
    ))
    fig10.update_layout(**PLOTLY_LAYOUT, title="Répartition genre", height=260)
    st.plotly_chart(fig10, use_container_width=True)

with c11:
    edu_counts = df["input.NAME_EDUCATION_TYPE"].value_counts()
    fig11 = go.Figure(go.Bar(
        x=edu_counts.values, y=edu_counts.index,
        orientation="h",
        marker_color=COLOR_SCORE,
        text=edu_counts.values, textposition="auto",
    ))
    fig11.update_layout(**PLOTLY_LAYOUT, title="Niveau d'éducation", height=260)
    st.plotly_chart(fig11, use_container_width=True)

# ── Drift indicator (1ère vs 2ème moitié) ────────────────────────────────────
st.markdown('<div class="section-title">Indicateur de dérive temporelle (1ère vs 2ème moitié)</div>', unsafe_allow_html=True)

mid = len(df) // 2
h1  = df.iloc[:mid]
h2  = df.iloc[mid:]

metrics = {
    "Score moyen":     (h1["score"].mean(), h2["score"].mean()),
    "Taux de refus":   (1 - h1["accordé"].mean(), 1 - h2["accordé"].mean()),
    "Latence moy.(ms)":(h1["inference_time_ms"].mean(), h2["inference_time_ms"].mean()),
    "Revenu moy.(k€)": (h1["revenu_k"].mean(), h2["revenu_k"].mean()),
    "Crédit moy.(k€)": (h1["credit_k"].mean(), h2["credit_k"].mean()),
}

drift_cols = st.columns(len(metrics))
for col, (name, (v1, v2)) in zip(drift_cols, metrics.items()):
    delta  = v2 - v1
    pct    = delta / v1 * 100 if v1 != 0 else 0
    color  = COLOR_WARN if abs(pct) > 15 else COLOR_ACCORD
    arrow  = "↑" if delta > 0 else "↓"
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{name}</div>
          <div class="kpi-value" style="font-size:1.2rem;color:#e6edf3">{v2:.3f}</div>
          <div class="kpi-delta" style="color:{color}">{arrow} {abs(pct):.1f}% vs 1ère moitié</div>
        </div>""", unsafe_allow_html=True)

# ── Raw logs table ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 Logs bruts (20 dernières requêtes)"):
    display_cols = ["timestamp", "score", "decision", "inference_time_ms",
                    "input.AMT_INCOME_TOTAL", "input.AMT_CREDIT", "input.DAYS_BIRTH",
                    "input.CODE_GENDER", "input.NAME_EDUCATION_TYPE"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].tail(20).rename(columns={
            "inference_time_ms": "latence_ms",
            "input.AMT_INCOME_TOTAL": "revenu",
            "input.AMT_CREDIT": "crédit",
            "input.DAYS_BIRTH": "jours_naissance",
            "input.CODE_GENDER": "genre",
            "input.NAME_EDUCATION_TYPE": "éducation",
        }).style.format(precision=3),
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#30363d;margin-top:2rem"/>
<p style="text-align:center;color:#484f58;font-size:0.75rem;font-family:'IBM Plex Mono',monospace">
  Prêt à Dépenser · Scoring Monitor · P8 MLOps ·
  Données mises à jour toutes les 30s (cache Streamlit)
</p>
""", unsafe_allow_html=True)
