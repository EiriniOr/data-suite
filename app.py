"""Miss Datrix — AI-powered end-to-end data analysis platform."""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as _components
from styles import inject_css, page_header, ai_panel_html, COMPARISON_HTML

from pipeline.loader import load_file, infer_task_type, DatasetMeta
from pipeline.cleaner import clean, auto_null_strategies, outlier_summary
from pipeline.eda import (
    compute_summary,
    get_distribution_figures,
    get_correlation_figure,
    get_target_distribution,
    get_ts_figures,
    flag_issues,
)
from pipeline.feature_eng import engineer_features
from pipeline.model_selector import benchmark_models
from pipeline.optimizer import optimize
from pipeline.reporter import (
    evaluate_model,
    get_feature_importance,
    pickle_model,
    generate_html_report,
)
from ai.analyst import DataAnalyst
from utils.plot_utils import (
    null_heatmap,
    model_leaderboard_chart,
    optimization_history_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Miss Datrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Login gate ────────────────────────────────────────────────────────────────
# Injected into the parent Streamlit page: hides chrome + styles the form.
_LOGIN_PAGE_CSS = """
<style>
section[data-testid="stSidebar"]{display:none!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.stDeployButton{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
.stTextInput>div>div>input{
  background:rgba(15,15,45,.85)!important;border:1px solid rgba(124,58,237,.3)!important;
  border-radius:10px!important;color:#e2e8f0!important;font-size:.9rem!important;box-shadow:none!important}
.stTextInput>div>div>input:focus{border-color:#7c3aed!important;box-shadow:0 0 0 2px rgba(124,58,237,.25)!important}
.stTextInput label{color:#64748b!important;font-size:.8rem!important;letter-spacing:.04em!important}
.stButton>button{
  background:linear-gradient(135deg,#7c3aed,#0891b2)!important;border:none!important;
  border-radius:10px!important;font-weight:700!important;letter-spacing:.06em!important;
  font-size:.9rem!important;box-shadow:0 4px 20px rgba(124,58,237,.3)!important}
.stButton>button:hover{opacity:.88!important}
.stAlert{border-radius:10px!important}
</style>
"""

# Rendered inside a components.v1.html iframe — bypasses Streamlit's markdown
# sanitiser so SVG, CSS animations, and HTML comments all work correctly.
_LOGIN_VISUAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#07071a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}
.page{display:grid;grid-template-columns:340px 1fr;height:100vh;max-width:1140px;margin:0 auto;padding:2rem 2.5rem;gap:3.5rem;align-items:center}
/* LEFT */
.left{display:flex;flex-direction:column;align-items:center;gap:.6rem}
.char{filter:drop-shadow(0 0 32px rgba(124,58,237,.55)) drop-shadow(0 0 64px rgba(6,182,212,.2));animation:float 6s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
.char-lbl{font-size:.65rem;color:#334155;letter-spacing:.14em;text-transform:uppercase;font-family:monospace}
/* RIGHT */
.right{display:flex;flex-direction:column;gap:1.5rem}
.title{font-size:3.6rem;font-weight:900;line-height:.92;letter-spacing:-.04em;
  background:linear-gradient(135deg,#22d3ee 0%,#a78bfa 45%,#f472b6 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{color:#475569;font-size:.92rem;margin:.4rem 0 0;line-height:1.55}
.sub b{color:#7c3aed;font-weight:600}
/* CAROUSEL */
.car{
  background:linear-gradient(145deg,rgba(30,27,75,.82),rgba(7,7,26,.96));
  border:1px solid rgba(124,58,237,.3);border-radius:22px;
  padding:1.8rem 2rem 1.5rem;position:relative;overflow:hidden;
  box-shadow:0 0 0 1px rgba(6,182,212,.05) inset,0 8px 60px rgba(124,58,237,.15),0 2px 10px rgba(0,0,0,.5)}
.car::before{content:'';position:absolute;top:-70px;right:-70px;width:260px;height:260px;border-radius:50%;
  background:radial-gradient(circle,rgba(124,58,237,.16) 0%,transparent 70%);pointer-events:none}
.car::after{content:'';position:absolute;bottom:-50px;left:-40px;width:180px;height:180px;border-radius:50%;
  background:radial-gradient(circle,rgba(6,182,212,.09) 0%,transparent 70%);pointer-events:none}
.car-lbl{font-size:.66rem;text-transform:uppercase;letter-spacing:.14em;color:#475569;font-family:monospace;margin-bottom:1.1rem}
.track{position:relative;min-height:155px}
.slide{position:absolute;top:0;left:0;right:0;opacity:0;display:flex;align-items:flex-start;gap:1.1rem;animation:cycle 36s infinite}
.slide:nth-child(1){animation-delay:0s}
.slide:nth-child(2){animation-delay:6s}
.slide:nth-child(3){animation-delay:12s}
.slide:nth-child(4){animation-delay:18s}
.slide:nth-child(5){animation-delay:24s}
.slide:nth-child(6){animation-delay:30s}
@keyframes cycle{0%{opacity:0;transform:translateX(14px)}3%{opacity:1;transform:translateX(0)}14%{opacity:1;transform:translateX(0)}17%,100%{opacity:0;transform:translateX(-10px)}}
.icon{width:52px;height:52px;flex-shrink:0;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:1.65rem;
  background:rgba(124,58,237,.16);border:1px solid rgba(124,58,237,.32);box-shadow:0 0 18px rgba(124,58,237,.18)}
.icon.c{background:rgba(6,182,212,.13);border-color:rgba(6,182,212,.28);box-shadow:0 0 18px rgba(6,182,212,.12)}
.icon.p{background:rgba(244,114,182,.11);border-color:rgba(244,114,182,.25);box-shadow:0 0 18px rgba(244,114,182,.1)}
.icon.a{background:rgba(251,191,36,.11);border-color:rgba(251,191,36,.25);box-shadow:0 0 18px rgba(251,191,36,.09)}
.stitle{color:#f1f5f9;font-size:1.12rem;font-weight:800;margin:0 0 .4rem;letter-spacing:-.015em}
.sdesc{color:#94a3b8;font-size:.84rem;line-height:1.65}
.sdesc b{color:#22d3ee;font-weight:600}
.sdesc i{color:#c084fc;font-style:normal}
.vs{display:inline-flex;gap:4px;background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.3);
  color:#fbbf24;font-size:.63rem;padding:2px 9px;border-radius:999px;margin-bottom:.38rem;
  font-weight:700;letter-spacing:.08em;text-transform:uppercase}
/* PROGRESS BARS */
.prog{display:flex;gap:5px;margin-top:1.3rem}
.bar{flex:1;height:3px;border-radius:2px;background:rgba(124,58,237,.18);overflow:hidden;position:relative}
.bar::after{content:'';position:absolute;top:0;left:0;height:100%;width:0%;border-radius:2px;
  background:linear-gradient(90deg,#7c3aed,#22d3ee);animation:fill 36s linear infinite}
.bar:nth-child(1)::after{animation-delay:0s}
.bar:nth-child(2)::after{animation-delay:6s}
.bar:nth-child(3)::after{animation-delay:12s}
.bar:nth-child(4)::after{animation-delay:18s}
.bar:nth-child(5)::after{animation-delay:24s}
.bar:nth-child(6)::after{animation-delay:30s}
@keyframes fill{0%{width:0%}14%{width:100%}16%{width:100%}17%,100%{width:0%}}
@media(max-width:780px){.page{grid-template-columns:1fr}.title{font-size:2.4rem}.char{max-width:200px}}
</style>
</head>
<body>
<div class="page">
  <div class="left">
    <div class="char">
      <svg width="300" height="378" viewBox="0 0 200 252" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="gA" cx="50%" cy="60%" r="50%"><stop offset="0%" stop-color="#7c3aed" stop-opacity=".5"/><stop offset="70%" stop-color="#0369a1" stop-opacity=".2"/><stop offset="100%" stop-color="#07071a" stop-opacity="0"/></radialGradient>
          <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#fcd5b5"/><stop offset="100%" stop-color="#f4a472"/></linearGradient>
          <linearGradient id="gH" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#3b0764"/><stop offset="55%" stop-color="#6d28d9"/><stop offset="100%" stop-color="#0369a1"/></linearGradient>
          <linearGradient id="gO" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#1e1b4b"/><stop offset="100%" stop-color="#07071a"/></linearGradient>
          <linearGradient id="gSc" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0f0b25"/><stop offset="100%" stop-color="#07071a"/></linearGradient>
          <filter id="gl"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <ellipse cx="100" cy="148" rx="92" ry="100" fill="url(#gA)"/>
        <g transform="translate(150,46)" opacity=".85" filter="url(#gl)">
          <rect width="44" height="34" rx="5" fill="rgba(8,8,32,.9)" stroke="#6d28d9" stroke-width="1.2"/>
          <rect x="1" y="1" width="42" height="10" rx="4" fill="rgba(124,58,237,.18)"/>
          <text x="4" y="9" font-size="5" fill="#a78bfa" font-family="monospace">accuracy</text>
          <polyline points="4,27 11,20 18,23 26,12 36,16 41,10" fill="none" stroke="#06b6d4" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="41" cy="10" r="2.2" fill="#06b6d4"/>
        </g>
        <g transform="translate(4,54)" opacity=".8">
          <circle cx="6" cy="10" r="4" fill="#7c3aed" filter="url(#gl)"/><circle cx="6" cy="23" r="4" fill="#7c3aed" filter="url(#gl)"/>
          <circle cx="21" cy="4" r="4" fill="#06b6d4" filter="url(#gl)"/><circle cx="21" cy="16" r="4" fill="#06b6d4" filter="url(#gl)"/><circle cx="21" cy="29" r="4" fill="#06b6d4" filter="url(#gl)"/>
          <circle cx="36" cy="10" r="4" fill="#a78bfa" filter="url(#gl)"/><circle cx="36" cy="23" r="4" fill="#a78bfa" filter="url(#gl)"/>
          <line x1="10" y1="10" x2="17" y2="4" stroke="#4c1d95" stroke-width="1"/><line x1="10" y1="10" x2="17" y2="16" stroke="#4c1d95" stroke-width="1"/>
          <line x1="10" y1="23" x2="17" y2="16" stroke="#4c1d95" stroke-width="1"/><line x1="10" y1="23" x2="17" y2="29" stroke="#4c1d95" stroke-width="1"/>
          <line x1="25" y1="4" x2="32" y2="10" stroke="#4c1d95" stroke-width="1"/><line x1="25" y1="16" x2="32" y2="10" stroke="#4c1d95" stroke-width="1"/>
          <line x1="25" y1="16" x2="32" y2="23" stroke="#4c1d95" stroke-width="1"/><line x1="25" y1="29" x2="32" y2="23" stroke="#4c1d95" stroke-width="1"/>
        </g>
        <text x="154" y="108" font-size="12" fill="#fbbf24" opacity=".95" filter="url(#gl)">&#x2736;</text>
        <text x="40" y="102" font-size="8" fill="#a78bfa" opacity=".75">&#x2736;</text>
        <text x="170" y="50" font-size="6" fill="#22d3ee" opacity=".6">&#x2736;</text>
        <path d="M54 252 Q50 214 42 196 Q66 181 100 179 Q134 181 158 196 Q150 214 146 252Z" fill="url(#gO)"/>
        <path d="M92 181 L100 198 L108 181" fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-linejoin="round"/>
        <path d="M70 220 Q100 215 130 220" fill="none" stroke="rgba(124,58,237,.25)" stroke-width="1"/>
        <rect x="91" y="166" width="18" height="17" rx="5" fill="url(#gS)"/>
        <ellipse cx="100" cy="134" rx="47" ry="51" fill="url(#gH)"/>
        <ellipse cx="100" cy="139" rx="39" ry="43" fill="url(#gS)"/>
        <path d="M61 119 Q63 87 82 80 Q100 74 118 80 Q137 87 139 119 Q126 108 100 106 Q74 108 61 119Z" fill="url(#gH)"/>
        <path d="M63 133 Q52 148 54 174 Q59 188 67 185 Q60 165 65 143Z" fill="url(#gH)"/>
        <path d="M137 133 Q148 148 146 174 Q141 188 133 185 Q140 165 135 143Z" fill="url(#gH)"/>
        <path d="M54 174 Q50 190 58 196 Q64 194 67 185" fill="url(#gH)"/>
        <path d="M146 174 Q150 190 142 196 Q136 194 133 185" fill="url(#gH)"/>
        <ellipse cx="61" cy="145" rx="7" ry="9" fill="url(#gS)"/><ellipse cx="139" cy="145" rx="7" ry="9" fill="url(#gS)"/>
        <ellipse cx="61" cy="145" rx="4" ry="6" fill="#e8a472"/><ellipse cx="139" cy="145" rx="4" ry="6" fill="#e8a472"/>
        <circle cx="61" cy="152" r="2.5" fill="#06b6d4" opacity=".9" filter="url(#gl)"/>
        <circle cx="139" cy="152" r="2.5" fill="#06b6d4" opacity=".9" filter="url(#gl)"/>
        <path d="M79 125 Q87 120 96 123" fill="none" stroke="#2d1050" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M104 123 Q113 120 121 125" fill="none" stroke="#2d1050" stroke-width="2.4" stroke-linecap="round"/>
        <rect x="74" y="129" width="23" height="17" rx="5" fill="rgba(6,182,212,.09)" stroke="#06b6d4" stroke-width="2" filter="url(#gl)"/>
        <rect x="103" y="129" width="23" height="17" rx="5" fill="rgba(6,182,212,.09)" stroke="#06b6d4" stroke-width="2" filter="url(#gl)"/>
        <line x1="97" y1="137" x2="103" y2="137" stroke="#06b6d4" stroke-width="2"/>
        <line x1="74" y1="137" x2="68" y2="138" stroke="#06b6d4" stroke-width="1.6"/>
        <line x1="126" y1="137" x2="132" y2="138" stroke="#06b6d4" stroke-width="1.6"/>
        <circle cx="85" cy="137" r="5" fill="#100030"/><circle cx="114" cy="137" r="5" fill="#100030"/>
        <circle cx="85" cy="137" r="3.2" fill="#4c1d95"/><circle cx="114" cy="137" r="3.2" fill="#4c1d95"/>
        <circle cx="86.8" cy="135.2" r="1.8" fill="white"/><circle cx="115.8" cy="135.2" r="1.8" fill="white"/>
        <path d="M97 150 Q100 157 103 150" fill="none" stroke="#c97c50" stroke-width="1.4" stroke-linecap="round"/>
        <path d="M87 163 Q100 173 113 163" fill="none" stroke="#c9735a" stroke-width="2" stroke-linecap="round"/>
        <path d="M94 160 Q100 157 106 160" fill="none" stroke="#e8956a" stroke-width="1.2" stroke-linecap="round"/>
        <ellipse cx="79" cy="158" rx="8" ry="5" fill="#f9a8a8" opacity=".28"/>
        <ellipse cx="121" cy="158" rx="8" ry="5" fill="#f9a8a8" opacity=".28"/>
        <path d="M83 80 Q100 75 115 82 Q108 89 100 88 Q92 88 83 80Z" fill="white" opacity=".16"/>
        <g transform="translate(60,208)">
          <rect width="80" height="50" rx="6" fill="#0c0a22" stroke="#4c1d95" stroke-width="1.8"/>
          <rect x="3" y="3" width="74" height="44" rx="4" fill="url(#gSc)"/>
          <rect x="3" y="3" width="74" height="44" rx="4" fill="rgba(124,58,237,.06)"/>
          <rect x="10" y="32" width="8" height="9" fill="#7c3aed" opacity=".95"/>
          <rect x="20" y="25" width="8" height="16" fill="#06b6d4" opacity=".95"/>
          <rect x="30" y="17" width="8" height="24" fill="#7c3aed" opacity=".95"/>
          <rect x="40" y="21" width="8" height="20" fill="#a78bfa" opacity=".95"/>
          <rect x="50" y="12" width="8" height="29" fill="#06b6d4" opacity=".95"/>
          <rect x="60" y="19" width="8" height="22" fill="#f472b6" opacity=".9"/>
          <line x1="7" y1="41" x2="72" y2="41" stroke="#1e293b" stroke-width="1"/>
          <line x1="47" y1="4" x2="47" y2="40" stroke="rgba(6,182,212,.2)" stroke-width="1" stroke-dasharray="2,2"/>
          <text x="11" y="10" font-size="5" fill="#475569" font-family="monospace">model scores</text>
        </g>
        <rect x="56" y="257" width="88" height="6" rx="3" fill="#0a0820" stroke="#2d1b5e" stroke-width="1"/>
        <rect x="85" y="259" width="30" height="3" rx="1.5" fill="rgba(124,58,237,.25)"/>
      </svg>
    </div>
    <p class="char-lbl">Miss Datrix &middot; AI Data Advisor</p>
  </div>

  <div class="right">
    <div>
      <div class="title">&#x2736; Miss Datrix</div>
      <p class="sub">Your AI-powered data science suite.<br><b>Upload data. Get real insights.</b></p>
    </div>
    <div class="car">
      <div class="car-lbl">What Miss Datrix does</div>
      <div class="track">
        <div class="slide"><div class="icon c">&#128194;</div><div><p class="stitle">Upload Any Dataset</p><p class="sdesc">CSV, Excel, or Parquet. Reads the schema, detects time series, and <b>proposes a workflow tailored to your goal</b> &mdash; not a fixed template.</p></div></div>
        <div class="slide"><div class="icon">&#129302;</div><div><p class="stitle">Real Model Training</p><p class="sdesc">Benchmarks RF, XGBoost, LightGBM, SVM, Logistic/Ridge with <b>5-fold cross-validation</b>. Ranked leaderboard by AUC, F1, or RMSE. You pick the winner.</p></div></div>
        <div class="slide"><div class="icon p">&#9881;&#65039;</div><div><p class="stitle">Optuna Hyperparameter Tuning</p><p class="sdesc"><b>Bayesian search</b> &mdash; smarter than grid search. Runs 100s of trials, shows an improvement curve, reveals what each parameter controls.</p></div></div>
        <div class="slide"><div class="icon c">&#128269;</div><div><p class="stitle">SHAP Explainability</p><p class="sdesc">Computes <b>real SHAP values</b> on your trained model. Feature importance, confusion matrix, ROC curve, residuals &mdash; plus a downloadable HTML report.</p></div></div>
        <div class="slide"><div class="icon p">&#129514;</div><div><p class="stitle">A/B Testing &amp; Causal Inference</p><p class="sdesc">Beyond correlation. Welch / chi-square tests, Cohen&rsquo;s d, <b>propensity score matching</b>, IPW, ATE with bootstrap SE, covariate balance (SMD).</p></div></div>
        <div class="slide"><div class="icon a">&#9889;</div><div><span class="vs">vs Claude.ai</span><p class="stitle">Not Just a Chatbot</p><p class="sdesc">Claude.ai <i>talks about</i> your data. Miss Datrix <b>runs real Python on it</b> &mdash; trains models you download, computes actual SHAP, runs Optuna, exports HTML reports.</p></div></div>
      </div>
      <div class="prog"><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
    </div>
  </div>
</div>
</body>
</html>"""


def _check_login() -> bool:
    """Gate the app behind a login form with visual branding."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown(_LOGIN_PAGE_CSS, unsafe_allow_html=True)
    _components.html(_LOGIN_VISUAL_HTML, height=520, scrolling=False)
    _, col, _ = st.columns([1, 0.9, 1])
    with col:
        username = st.text_input(
            "Username", key="_login_user", placeholder="your username"
        )
        password = st.text_input(
            "Password", type="password", key="_login_pass", placeholder="••••••••"
        )
        st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
        if st.button("Sign in →", use_container_width=True):
            users: dict = st.secrets.get("users", {})
            if username in users and users[username] == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()


_check_login()


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "stage": "upload",
        "mode": "ml_pipeline",  # "ml_pipeline" | "ab_test" | "causal"
        "ab_test_results": None,
        "causal_results": None,
        "df_raw": None,
        "df_clean": None,
        "df_engineered": None,
        "X": None,
        "y": None,
        "meta": None,
        "analyst": DataAnalyst(),
        "ai_response": {},  # {stage: response text}
        "leaderboard": None,
        "trained_models": None,
        "primary_metric": None,
        "best_model": None,
        "best_model_name": None,
        "best_params": None,
        "baseline_score": None,
        "best_score": None,
        "trials_df": None,
        "eval_results": None,
        "feature_importance": None,
        "workflow": [],
        "cleaning_strategies": None,
        "outlier_action": "clip",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="padding:1rem 0 0.5rem">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;
                  background:linear-gradient(135deg,#7c3aed,#db2777);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">✦ Miss Datrix</div>
      <div style="font-size:0.75rem;color:#64748b;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;">
        AI Data Analysis Platform
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Stage indicator
    stages = ["upload", "clean", "eda", "features", "models", "optimize", "report"]
    stage_labels = {
        "upload": "Load & Intent",
        "clean": "Clean",
        "eda": "EDA",
        "features": "Features",
        "models": "Models",
        "optimize": "Optimize",
        "report": "Report",
    }
    stage_icons = {
        "upload": "01",
        "clean": "02",
        "eda": "03",
        "features": "04",
        "models": "05",
        "optimize": "06",
        "report": "07",
    }
    current = st.session_state.stage
    current_idx = stages.index(current) if current in stages else -1
    for s in stages:
        if s not in st.session_state.workflow and s != "upload":
            continue
        label = stage_labels[s]
        num = stage_icons[s]
        if s == current:
            st.markdown(
                f'<div class="nav-item-active">→ {num} · {label}</div>',
                unsafe_allow_html=True,
            )
        elif stages.index(s) < current_idx:
            st.markdown(
                f'<div class="nav-item-done">✓ {num} · {label}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="nav-item-pending">{num} · {label}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Persistent chat
    if st.session_state.df_raw is not None:
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#7c3aed;margin-bottom:0.5rem;">✦ Ask the Analyst</div>',
            unsafe_allow_html=True,
        )
        user_q = st.text_input(
            "Ask anything about your data",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="e.g. Why was XGBoost chosen?",
        )
        if user_q:
            with st.spinner("Thinking..."):
                answer = st.session_state.analyst.chat(user_q)
            st.markdown(answer)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Convert common markdown patterns to styled HTML for the AI panel."""
    import re

    def inline(s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em style='color:#c4b5fd'>\1</em>", s)
        return s

    lines = text.split("\n")
    out = []
    in_ul = in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_lists()
            continue

        h2 = re.match(r"^#{1,2} (.+)", stripped)
        h3 = re.match(r"^#{3} (.+)", stripped)
        bullet = re.match(r"^[-•*] (.+)", stripped)
        numbered = re.match(r"^\d+\. (.+)", stripped)

        if h2:
            close_lists()
            out.append(
                f'<p style="font-family:Space Grotesk,sans-serif;font-weight:600;'
                f"color:#c4b5fd;margin:0.8rem 0 0.15rem;font-size:0.9rem;"
                f'text-transform:uppercase;letter-spacing:0.05em">{inline(h2.group(1))}</p>'
            )
        elif h3:
            close_lists()
            out.append(
                f'<p style="font-weight:600;color:#e2e8f0;margin:0.6rem 0 0.1rem">'
                f"{inline(h3.group(1))}</p>"
            )
        elif bullet:
            if not in_ul:
                close_lists()
                out.append(
                    '<ul style="margin:0.3rem 0 0.3rem 1rem;padding:0;'
                    'list-style:disc;color:var(--text-secondary)">'
                )
                in_ul = True
            out.append(f"<li style='margin:0.18rem 0'>{inline(bullet.group(1))}</li>")
        elif numbered:
            if not in_ol:
                close_lists()
                out.append(
                    '<ol style="margin:0.3rem 0 0.3rem 1.2rem;padding:0;'
                    'color:var(--text-secondary)">'
                )
                in_ol = True
            out.append(f"<li style='margin:0.18rem 0'>{inline(numbered.group(1))}</li>")
        else:
            close_lists()
            out.append(
                f'<p style="margin:0.3rem 0;color:var(--text-secondary);line-height:1.65">'
                f"{inline(stripped)}</p>"
            )

    close_lists()
    return "\n".join(out)


def ai_panel(stage: str, response_key: str | None = None):
    """Show AI response for a stage. Cache in session state."""
    key = response_key or stage
    if key in st.session_state.ai_response:
        content = st.session_state.ai_response[key]
        ai_panel_html(_md_to_html(content))


def go_to(stage: str):
    st.session_state.stage = stage
    st.rerun()


# ── Cached pipeline wrappers ──────────────────────────────────────────────────
# @st.cache_data keeps results across reruns so Streamlit doesn't re-execute
# heavy steps on every widget interaction. Wrappers live here to keep
# pipeline/ free of Streamlit imports.


@st.cache_data(max_entries=1, show_spinner=False)
def _cached_engineer_features(
    df,
    target_col,
    task_type,
    scaler_type,
    drop_high_corr,
    is_ts,
    dt_col,
    lag_windows,
    rolling_windows,
):
    return engineer_features(
        df,
        target_col=target_col,
        task_type=task_type,
        scaler_type=scaler_type,
        drop_high_corr=drop_high_corr,
        is_ts=is_ts,
        dt_col=dt_col,
        lag_windows=lag_windows,
        rolling_windows=rolling_windows,
    )


@st.cache_data(max_entries=1, show_spinner=False)
def _cached_benchmark_models(X, y, task_type, is_ts):
    return benchmark_models(X, y, task_type, is_ts)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 0: UPLOAD + INTENT
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    # ── Phase A: file upload + initial load (only when no data loaded yet) ────
    if st.session_state.df_raw is None:
        # Hero
        st.markdown(
            """
        <div style="padding: 3rem 0 2rem; text-align: center;">
          <div class="gradient-title">Miss Datrix</div>
          <div class="gradient-subtitle" style="margin-top:0.5rem;">
            AI-Powered Data Analysis · From raw file to trained model in minutes
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Mode selector
        st.markdown(
            """
        <div style="margin-bottom:1.5rem">
          <div class="gradient-subtitle" style="margin-bottom:0.75rem;">Choose analysis type</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        mode_col1, mode_col2, mode_col3, mode_col4 = st.columns(4, gap="medium")
        with mode_col1:
            st.markdown(
                """<div class="datrix-card" style="text-align:center;cursor:pointer">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">🤖</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">ML Pipeline</div>
                <div style="font-size:0.8rem;color:#64748b">Clean → EDA → Models → Tune → Report</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mode_col2:
            st.markdown(
                """<div class="datrix-card" style="text-align:center;cursor:pointer">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">🧪</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">A/B Test</div>
                <div style="font-size:0.8rem;color:#64748b">Statistical significance, effect size, power</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mode_col3:
            st.markdown(
                """<div class="datrix-card" style="text-align:center;cursor:pointer">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">🔍</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">Causal Analysis</div>
                <div style="font-size:0.8rem;color:#64748b">ATE via propensity score matching</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mode_col4:
            st.markdown(
                """<div class="datrix-card" style="text-align:center;cursor:pointer">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">🧭</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">Model Advisor</div>
                <div style="font-size:0.8rem;color:#64748b">Decision tree → model + loss + metrics</div>
                </div>""",
                unsafe_allow_html=True,
            )

        selected_mode = st.radio(
            "Mode",
            ["ml_pipeline", "ab_test", "causal", "model_advisor"],
            format_func=lambda m: {
                "ml_pipeline": "🤖 ML Pipeline",
                "ab_test": "🧪 A/B Test",
                "causal": "🔍 Causal Analysis",
                "model_advisor": "🧭 Model Advisor",
            }[m],
            horizontal=True,
            label_visibility="collapsed",
            key="mode_radio",
        )
        st.session_state.mode = selected_mode

        # Upload + intent
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown('<div class="datrix-card">', unsafe_allow_html=True)
            if selected_mode == "model_advisor":
                st.markdown(
                    "<div style='padding:1.5rem 0;text-align:center;color:#94a3b8'>"
                    "<div style='font-size:2.5rem;margin-bottom:0.75rem'>🧭</div>"
                    "<div style='font-weight:600;color:#e2e8f0;margin-bottom:0.5rem'>Model Advisor</div>"
                    "<div style='font-size:0.88rem'>Answer a few questions about your problem — "
                    "get a tailored recommendation for model, loss function, and metrics.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                uploaded = None
                if st.button(
                    "🧭 Launch Advisor →",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.mode = "model_advisor"
                    st.session_state.advisor_answers = {}
                    go_to("model_advisor")
            else:
                uploaded = st.file_uploader(
                    "Drop your dataset here",
                    type=["csv", "xlsx", "xls", "parquet"],
                    label_visibility="visible",
                )
                if selected_mode == "ml_pipeline":
                    user_goal = st.text_area(
                        "What do you want to do?",
                        placeholder='e.g. "predict customer churn" · "just explore" · "I don\'t know"',
                        height=90,
                        key="user_goal_input",
                    )
                if uploaded:
                    st.button(
                        "✦ Start Analysis",
                        type="primary",
                        use_container_width=True,
                        key="start_btn",
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                """
            <div class="datrix-card" style="height:100%">
              <div class="gradient-subtitle" style="margin-bottom:1rem;">What Miss Datrix does</div>
              <div style="display:flex;flex-direction:column;gap:0.6rem">
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;Loads your real file and runs against it</div>
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;AI analyst asks the right questions</div>
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;Auto-cleans, engineers features</div>
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;Benchmarks 5 models with real CV</div>
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;Tunes hyperparameters (Optuna)</div>
                <div style="color:#e2e8f0;font-size:0.88rem">✦ &nbsp;Exports model + HTML report</div>
              </div>
              <div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid rgba(124,58,237,0.2)">
                <div style="font-size:0.75rem;color:#64748b">Supports CSV · Excel · Parquet · up to 200 MB</div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Comparison table
        st.markdown(COMPARISON_HTML, unsafe_allow_html=True)

        if uploaded and st.session_state.get("start_btn"):
            with st.spinner("Loading file..."):
                try:
                    df, meta = load_file(uploaded)
                    st.session_state.df_raw = df
                    st.session_state.meta = meta
                    st.session_state.mode = selected_mode
                    st.session_state.user_goal = st.session_state.get(
                        "user_goal_input", ""
                    )
                except Exception as e:
                    st.error(f"Failed to load file: {e}")
                    st.stop()

            if selected_mode == "ml_pipeline":
                with st.spinner("AI is analyzing your dataset..."):
                    try:
                        response = st.session_state.analyst.analyze_intent(
                            df, st.session_state.user_goal
                        )
                        st.session_state.ai_response["intent"] = response
                    except Exception as e:
                        st.warning(
                            f"AI analysis unavailable (check ANTHROPIC_API_KEY): {e}"
                        )

            st.rerun()  # clean rerun so config form renders from session state

    # ── Phase B: configure task + approve workflow (data is loaded) ───────────
    else:
        df = st.session_state.df_raw
        meta = st.session_state.meta
        mode = st.session_state.mode

        all_cols = df.columns.tolist()

        # ── Preview (all modes) ───────────────────────────────────────────────
        mode_labels = {
            "ml_pipeline": "ML Pipeline",
            "ab_test": "A/B Test",
            "causal": "Causal Analysis",
        }
        page_header(
            "Step 01",
            "Dataset Loaded",
            f"Mode: {mode_labels.get(mode, mode)} · Configure columns and proceed",
        )
        st.subheader("Dataset Preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing values", f"{df.isnull().sum().sum():,}")
        st.dataframe(df.head(10), use_container_width=True)
        if df.isnull().sum().sum() > 0:
            st.plotly_chart(null_heatmap(df), use_container_width=True)

        st.divider()

        # ── Mode-specific config ──────────────────────────────────────────────
        if mode == "ml_pipeline":
            # AI panel + follow-up
            ai_panel("intent")
            if "intent" in st.session_state.ai_response:
                intent_reply = st.text_area(
                    "Your answers to the AI questions (optional)",
                    height=80,
                    key="intent_reply",
                )
                if intent_reply and st.button("Send reply to AI"):
                    with st.spinner("AI is updating..."):
                        try:
                            follow_up = st.session_state.analyst.chat(intent_reply)
                            st.session_state.ai_response["intent_followup"] = follow_up
                            st.rerun()
                        except Exception:
                            pass
                ai_panel("intent_followup")
            st.divider()

            st.subheader("Configure Task")
            target_col = st.selectbox(
                "Select target column",
                all_cols,
                index=len(all_cols) - 1,
                key="target_col_select",
            )
            inferred_task = infer_task_type(df, target_col)
            if meta.is_ts:
                inferred_task = "time_series"
            task_type = st.selectbox(
                "Task type",
                ["classification", "regression", "time_series"],
                index=["classification", "regression", "time_series"].index(
                    inferred_task
                ),
                key="task_type_select",
            )

            st.subheader("Workflow")
            full_workflow = ["clean", "eda", "features", "models", "optimize", "report"]
            selected_workflow = st.multiselect(
                "Select stages to run",
                full_workflow,
                default=full_workflow,
                format_func=lambda s: stage_labels[s],
                key="workflow_select",
            )

            if st.button("✅ Confirm & Proceed", type="primary"):
                meta.target_col = target_col
                meta.task_type = task_type
                meta.approved_workflow = selected_workflow
                st.session_state.meta = meta
                st.session_state.workflow = ["upload"] + selected_workflow
                next_stage = selected_workflow[0] if selected_workflow else "report"
                go_to(next_stage)

        elif mode == "ab_test":
            st.subheader("Configure A/B Test")
            treatment_col = st.selectbox(
                "Treatment / group column (2 unique values)",
                all_cols,
                key="ab_treatment_col",
            )
            outcome_col = st.selectbox(
                "Outcome column (what you're measuring)",
                [c for c in all_cols if c != treatment_col],
                key="ab_outcome_col",
            )
            alpha = st.select_slider(
                "Significance level (α)",
                options=[0.01, 0.05, 0.10],
                value=0.05,
                key="ab_alpha",
            )

            groups = df[treatment_col].dropna().unique()
            if len(groups) == 2:
                g = sorted(groups, key=str)
                st.info(
                    f"Control: **{g[0]}** · Treatment: **{g[1]}** · {len(df):,} rows"
                )
            else:
                st.warning(
                    f"Treatment column has {len(groups)} unique values — must be exactly 2."
                )

            if st.button("🧪 Run A/B Test", type="primary"):
                st.session_state.ab_treatment_col = treatment_col
                st.session_state.ab_outcome_col = outcome_col
                st.session_state.ab_alpha = alpha
                go_to("ab_test")

        elif mode == "causal":
            st.subheader("Configure Causal Analysis")
            treatment_col = st.selectbox(
                "Treatment column (binary: 0/1 or two labels)",
                all_cols,
                key="ca_treatment_col",
            )
            outcome_col = st.selectbox(
                "Outcome column (continuous or binary)",
                [c for c in all_cols if c != treatment_col],
                key="ca_outcome_col",
            )
            covariate_cols = st.multiselect(
                "Covariate columns (confounders to adjust for)",
                [c for c in all_cols if c not in [treatment_col, outcome_col]],
                key="ca_covariate_cols",
                help="Select variables that affect both treatment assignment and outcome.",
            )
            method = st.radio(
                "Estimation method",
                ["psm", "ipw"],
                format_func=lambda m: {
                    "psm": "Propensity Score Matching (1:1 nearest neighbor)",
                    "ipw": "Inverse Probability Weighting (IPW)",
                }[m],
                key="ca_method",
            )

            if not covariate_cols:
                st.warning("Select at least one covariate to adjust for confounding.")

            if covariate_cols and st.button("🔍 Run Causal Analysis", type="primary"):
                st.session_state.ca_treatment_col = treatment_col
                st.session_state.ca_outcome_col = outcome_col
                st.session_state.ca_covariate_cols = covariate_cols
                st.session_state.ca_method = method
                go_to("causal")

        if st.button("↺ Upload different file", type="secondary"):
            st.session_state.df_raw = None
            st.session_state.meta = None
            st.session_state.ai_response = {}
            st.session_state.analyst = DataAnalyst()
            st.session_state.ab_test_results = None
            st.session_state.causal_results = None
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1: CLEAN
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "clean":
    page_header(
        "Stage 02",
        "Data Cleaning",
        "Fix nulls, outliers, and duplicates before modeling",
    )
    df = st.session_state.df_raw
    meta: DatasetMeta = st.session_state.meta

    # AI analysis
    if "clean" not in st.session_state.ai_response:
        with st.spinner("AI is reviewing data quality..."):
            try:
                response = st.session_state.analyst.analyze_cleaning(df, meta.to_dict())
                st.session_state.ai_response["clean"] = response
            except Exception as e:
                st.session_state.ai_response["clean"] = f"*(AI unavailable: {e})*"
    ai_panel("clean")

    st.divider()

    # Auto strategies (user can override)
    auto_strategies = auto_null_strategies(df)
    st.subheader("Null Imputation Strategy")
    null_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().sum() > 0}

    if null_cols:
        strategies = {}
        for col, n in null_cols.items():
            pct = n / len(df) * 100
            default = auto_strategies.get(col, "keep")
            strat = st.selectbox(
                f"`{col}` — {n} nulls ({pct:.1f}%)",
                ["median", "mode", "mean", "knn", "zero", "drop_col", "keep"],
                index=[
                    "median",
                    "mode",
                    "mean",
                    "knn",
                    "zero",
                    "drop_col",
                    "keep",
                ].index(default),
                key=f"null_{col}",
            )
            strategies[col] = strat
    else:
        st.success("No null values found!")
        strategies = {}

    # Outlier handling
    st.subheader("Outlier Handling")
    out_summary = outlier_summary(df)
    if not out_summary.empty and out_summary["n_outliers"].sum() > 0:
        st.dataframe(
            out_summary[out_summary["n_outliers"] > 0], use_container_width=True
        )
    outlier_action = st.radio(
        "Outlier action", ["clip", "drop", "keep"], horizontal=True
    )

    # Duplicates
    n_dupes = df.duplicated().sum()
    drop_dupes = st.checkbox(f"Remove {n_dupes} duplicate rows", value=n_dupes > 0)

    if st.button("🧹 Apply Cleaning", type="primary"):
        with st.spinner("Cleaning..."):
            df_clean, report = clean(df, strategies, outlier_action, drop_dupes)
        st.session_state.df_clean = df_clean
        meta.cleaning_report = report
        st.rerun()

    # Show results + Next after cleaning is done (outside button block)
    if st.session_state.df_clean is not None:
        report = meta.cleaning_report
        st.success(
            f"Cleaned: {report['duplicates_removed']} dupes removed, "
            f"{len(report['dropped_cols'])} cols dropped, "
            f"{len(report['imputed'])} cols imputed, "
            f"{len(report['outliers'])} cols with outliers handled."
        )
        st.subheader("Before / After")
        c1, c2 = st.columns(2)
        c1.metric("Rows before", len(df))
        c2.metric("Rows after", len(st.session_state.df_clean))
        st.dataframe(st.session_state.df_clean.head(), use_container_width=True)

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("clean") if "clean" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_clean"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "eda":
    page_header(
        "Stage 03",
        "Exploratory Data Analysis",
        "Understand distributions, correlations, and patterns",
    )
    df = (
        st.session_state.df_clean
        if st.session_state.df_clean is not None
        else st.session_state.df_raw
    )
    meta: DatasetMeta = st.session_state.meta

    if "eda" not in st.session_state.ai_response:
        with st.spinner("AI is analyzing patterns..."):
            try:
                response = st.session_state.analyst.analyze_eda(df, meta.to_dict())
                st.session_state.ai_response["eda"] = response
            except Exception as e:
                st.session_state.ai_response["eda"] = f"*(AI unavailable: {e})*"
    ai_panel("eda")

    st.divider()
    summary = compute_summary(df)

    # Warnings
    issues = flag_issues(df, meta.target_col, meta.task_type)
    if issues:
        with st.expander("⚠️ Data Quality Flags", expanded=True):
            for w in issues:
                st.warning(w)

    # Numeric summary table
    st.subheader("Numeric Statistics")
    st.dataframe(
        summary["numeric_stats"].style.background_gradient(
            cmap="Blues", subset=["mean", "std"]
        ),
        use_container_width=True,
    )

    # Target distribution
    if meta.target_col and meta.target_col in df.columns:
        st.subheader("Target Distribution")
        st.plotly_chart(
            get_target_distribution(df, meta.target_col, meta.task_type),
            use_container_width=True,
        )

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.plotly_chart(get_correlation_figure(df), use_container_width=True)

    # Distributions
    st.subheader("Column Distributions")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if meta.target_col in numeric_cols:
        numeric_cols.remove(meta.target_col)
    selected_cols = st.multiselect(
        "Select columns", numeric_cols, default=numeric_cols[:4]
    )
    if selected_cols:
        figs = get_distribution_figures(df, selected_cols)
        cols = st.columns(min(2, len(selected_cols)))
        for i, (col, fig) in enumerate(figs.items()):
            cols[i % 2].plotly_chart(fig, use_container_width=True)

    # Time series plots
    if meta.is_ts and meta.datetime_col:
        st.subheader("Time Series")
        ts_num_cols = df.select_dtypes(include="number").columns.tolist()[:4]
        ts_figs = get_ts_figures(df, meta.datetime_col, ts_num_cols)
        for col, fig in ts_figs.items():
            st.plotly_chart(fig, use_container_width=True)

    workflow = st.session_state.meta.approved_workflow
    idx = workflow.index("eda") if "eda" in workflow else -1
    next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
    if st.button("Next →", key="next_eda"):
        go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "features":
    page_header(
        "Stage 04",
        "Feature Engineering",
        "Encode, scale, and create features for modeling",
    )
    df = (
        st.session_state.df_clean
        if st.session_state.df_clean is not None
        else st.session_state.df_raw
    )
    meta: DatasetMeta = st.session_state.meta

    if "features" not in st.session_state.ai_response:
        with st.spinner("AI is planning features..."):
            try:
                response = st.session_state.analyst.analyze_features(df, meta.to_dict())
                st.session_state.ai_response["features"] = response
            except Exception as e:
                st.session_state.ai_response["features"] = f"*(AI unavailable: {e})*"
    ai_panel("features")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        scaler_type = st.selectbox(
            "Numeric scaling",
            ["standard", "minmax", "robust", "none"],
            help="Standard: zero-mean unit-variance. Robust: IQR-based (outlier resistant). None: for tree models.",
        )
    with c2:
        drop_high_corr = st.checkbox(
            "Drop highly correlated features (r > 0.95)", value=True
        )

    lag_windows, rolling_windows = None, None
    if meta.is_ts:
        st.subheader("Time Series Features")
        lag_input = st.text_input("Lag windows (comma-separated)", "1,7,30")
        roll_input = st.text_input("Rolling windows (comma-separated)", "7,30")
        try:
            lag_windows = [int(x.strip()) for x in lag_input.split(",") if x.strip()]
            rolling_windows = [
                int(x.strip()) for x in roll_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Enter valid integers separated by commas.")

    if st.button("⚙️ Engineer Features", type="primary"):
        with st.spinner("Engineering features..."):
            try:
                X, y, feat_report = _cached_engineer_features(
                    df,
                    meta.target_col,
                    meta.task_type,
                    scaler_type,
                    drop_high_corr,
                    meta.is_ts,
                    meta.datetime_col,
                    tuple(lag_windows) if lag_windows else None,
                    tuple(rolling_windows) if rolling_windows else None,
                )
                st.session_state.X = X
                st.session_state.y = y
                meta.feature_report = feat_report
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")
                st.stop()
        st.rerun()

    # Show results + Next after engineering is done (outside button block)
    if st.session_state.X is not None:
        X = st.session_state.X
        feat_report = meta.feature_report
        st.success(f"Features ready: {X.shape[1]} features × {len(X):,} rows")
        c1, c2, c3 = st.columns(3)
        c1.metric("Features added", len(feat_report.get("added", [])))
        c2.metric("Columns encoded", len(feat_report.get("encoded", {})))
        c3.metric("Columns dropped", len(feat_report.get("dropped", [])))
        st.dataframe(X.head(), use_container_width=True)

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("features") if "features" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_features"):
            import gc

            st.session_state.df_clean = None  # X and y are ready; free DataFrame copies
            st.session_state.df_raw = None
            gc.collect()
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4: MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "models":
    page_header(
        "Stage 05", "Model Selection", "Benchmark multiple models and find the best fit"
    )
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta

    if X is None:
        st.error("Run Feature Engineering first.")
        st.stop()

    if st.session_state.leaderboard is None:
        if st.button("🚀 Benchmark Models", type="primary"):
            progress = st.progress(0, text="Training models...")
            with st.spinner("Benchmarking..."):
                leaderboard, trained_models, primary_metric = _cached_benchmark_models(
                    X,
                    y,
                    meta.task_type,
                    meta.is_ts,
                )
            progress.progress(100)
            st.session_state.leaderboard = leaderboard
            st.session_state.trained_models = trained_models
            st.session_state.primary_metric = primary_metric
            st.rerun()
    else:
        leaderboard = st.session_state.leaderboard
        primary_metric = st.session_state.primary_metric

        # AI analysis
        if "models" not in st.session_state.ai_response:
            with st.spinner("AI is interpreting results..."):
                try:
                    response = st.session_state.analyst.analyze_models(
                        leaderboard, meta.to_dict()
                    )
                    st.session_state.ai_response["models"] = response
                except Exception as e:
                    st.session_state.ai_response["models"] = f"*(AI unavailable: {e})*"
        ai_panel("models")

        st.divider()
        st.subheader("Leaderboard")
        st.dataframe(
            leaderboard.style.highlight_max(subset=[primary_metric], color="#d4edda"),
            use_container_width=True,
        )
        st.plotly_chart(
            model_leaderboard_chart(leaderboard, primary_metric),
            use_container_width=True,
        )

        best_name = leaderboard.iloc[0]["model"]
        selected_model_name = st.selectbox(
            "Select model for optimization",
            leaderboard["model"].tolist(),
            index=0,
            help=f"Auto-selected: {best_name} (top scorer)",
        )
        st.session_state.best_model_name = selected_model_name
        meta.best_model_name = selected_model_name
        meta.best_model_score = leaderboard.iloc[0][primary_metric]
        meta.best_model_metric = primary_metric

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("models") if "models" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_models"):
            import gc

            st.session_state.trained_models = None  # free 5 fitted model objects
            gc.collect()
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 5: OPTIMIZE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "optimize":
    page_header(
        "Stage 06",
        "Hyperparameter Optimization",
        "Bayesian search for the best model configuration",
    )
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta
    model_name = st.session_state.best_model_name or "XGBoost"

    if X is None:
        st.error("Run Feature Engineering first.")
        st.stop()

    n_trials = st.slider("Number of Optuna trials", 10, 200, 50)
    st.caption(
        "More trials = better params but longer runtime. 50 is a good starting point."
    )

    if st.session_state.best_model is None:
        if st.button(f"🎯 Optimize {model_name}", type="primary"):
            progress_bar = st.progress(0, text="Running Optuna trials...")

            def update_progress(trial_num, score):
                pct = int(trial_num / n_trials * 100)
                progress_bar.progress(
                    pct, text=f"Trial {trial_num}/{n_trials} — best score so far"
                )

            with st.spinner("Optimizing..."):
                best_model, best_params, baseline, best_score, trials_df = optimize(
                    model_name,
                    meta.task_type,
                    X,
                    y,
                    is_ts=meta.is_ts,
                    n_trials=n_trials,
                    progress_callback=update_progress,
                )
            progress_bar.progress(100)

            st.session_state.best_model = best_model
            st.session_state.best_params = best_params
            st.session_state.baseline_score = baseline
            st.session_state.best_score = best_score
            st.session_state.trials_df = trials_df
            st.rerun()
    else:
        best_params = st.session_state.best_params
        baseline = st.session_state.baseline_score
        best_score = st.session_state.best_score
        trials_df = st.session_state.trials_df
        primary_metric = st.session_state.primary_metric or "score"

        # AI analysis
        if "optimize" not in st.session_state.ai_response:
            with st.spinner("AI is interpreting results..."):
                try:
                    response = st.session_state.analyst.analyze_optimization(
                        best_params, baseline, best_score, model_name
                    )
                    st.session_state.ai_response["optimize"] = response
                except Exception as e:
                    st.session_state.ai_response["optimize"] = (
                        f"*(AI unavailable: {e})*"
                    )
        ai_panel("optimize")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline score", f"{baseline:.4f}")
        c2.metric("Best score", f"{best_score:.4f}")
        c3.metric("Improvement", f"{best_score - baseline:+.4f}")

        st.subheader("Best Hyperparameters")
        st.json(best_params)

        st.subheader("Optimization History")
        st.plotly_chart(
            optimization_history_chart(trials_df, trials_df.columns[-1]),
            use_container_width=True,
        )

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("optimize") if "optimize" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_opt"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 6: REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "report":
    page_header(
        "Stage 07",
        "Results & Report",
        "Final evaluation, feature importance, and export",
    )
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta
    model = (
        st.session_state.best_model
        if st.session_state.best_model is not None
        else (
            list(st.session_state.trained_models.values())[0]
            if st.session_state.trained_models
            else None
        )
    )

    if model is None or X is None:
        st.warning("No trained model found. Complete earlier stages first.")
        st.stop()

    # Evaluate if not done
    if st.session_state.eval_results is None:
        with st.spinner("Running final evaluation..."):
            eval_results = evaluate_model(model, X, y, meta.task_type, meta.is_ts)
            feature_imp = get_feature_importance(model, X.columns.tolist())
            st.session_state.eval_results = eval_results
            st.session_state.feature_importance = feature_imp

    eval_results = st.session_state.eval_results
    feature_imp = st.session_state.feature_importance

    # AI summary
    if "report" not in st.session_state.ai_response:
        with st.spinner("AI is writing summary..."):
            try:
                summary = st.session_state.analyst.generate_report_summary(
                    meta.to_dict()
                )
                st.session_state.ai_response["report"] = summary
            except Exception as e:
                st.session_state.ai_response["report"] = f"*(AI unavailable: {e})*"
    ai_panel("report")

    st.divider()

    # Metrics
    st.subheader("Performance Metrics")
    metrics = eval_results.get("metrics", {})
    if isinstance(metrics, dict):
        display_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        if display_metrics:
            cols = st.columns(len(display_metrics))
            for i, (k, v) in enumerate(display_metrics.items()):
                cols[i].metric(k, f"{v:.4f}")

    # Feature importance
    if feature_imp is not None:
        st.subheader("Feature Importance")
        from utils.plot_utils import feature_importance_chart

        st.plotly_chart(
            feature_importance_chart(feature_imp, top_n=20), use_container_width=True
        )

    # Eval figures
    for name, fig in eval_results.get("figures", {}).items():
        st.subheader(name.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    # Downloads
    st.divider()
    st.subheader("Downloads")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        model_bytes = pickle_model(model)
        st.download_button(
            "⬇️ Download Trained Model (.pkl)",
            data=model_bytes,
            file_name=f"{meta.best_model_name or 'model'}.pkl",
            mime="application/octet-stream",
        )

    with dcol2:
        with st.spinner("Generating HTML report..."):
            html = generate_html_report(
                meta=meta.to_dict(),
                eval_results=eval_results,
                leaderboard=st.session_state.leaderboard,
                ai_summary=st.session_state.ai_response.get("report", ""),
                feature_importance=feature_imp,
            )
        st.download_button(
            "⬇️ Download HTML Report",
            data=html,
            file_name="data_suite_report.html",
            mime="text/html",
        )

    if st.button("✦ Start New Analysis", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# A/B TEST STAGE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "ab_test":
    page_header(
        "A/B Test",
        "Statistical Analysis",
        "Significance, effect size, confidence intervals, and power",
    )
    df = st.session_state.df_raw
    treatment_col = st.session_state.get("ab_treatment_col")
    outcome_col = st.session_state.get("ab_outcome_col")
    alpha = st.session_state.get("ab_alpha", 0.05)

    if df is None or not treatment_col or not outcome_col:
        st.error("Missing configuration. Go back and set up the test.")
        st.stop()

    # Run test if not cached
    if st.session_state.ab_test_results is None:
        with st.spinner("Running statistical test..."):
            try:
                results = run_ab_test(df, treatment_col, outcome_col, alpha=alpha)
                st.session_state.ab_test_results = results
            except Exception as e:
                st.error(f"Test failed: {e}")
                st.stop()

    results = st.session_state.ab_test_results

    # AI interpretation
    if "ab_test" not in st.session_state.ai_response:
        with st.spinner("AI is interpreting results..."):
            try:
                response = st.session_state.analyst.analyze_ab_test(results)
                st.session_state.ai_response["ab_test"] = response
            except Exception as e:
                st.session_state.ai_response["ab_test"] = f"*(AI unavailable: {e})*"
    ai_panel("ab_test")

    st.divider()

    # Key metrics
    sig = results.get("significant")
    sig_label = "✅ Significant" if sig else "❌ Not significant"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "p-value",
        f"{results.get('p_value'):.6f}",
        help=(
            "Probability of seeing a result this extreme if there were truly no effect "
            "(the null hypothesis). Below your chosen α means statistically significant — "
            "but note: significance ≠ practical importance."
        ),
    )
    c2.metric(
        results.get("effect_size_label", "Effect size"),
        f"{results.get('effect_size'):.4f}",
        delta=results.get("effect_interpretation"),
        help=(
            "Cohen's d (continuous): standardised difference in means. "
            "0.2 = small, 0.5 = medium, 0.8 = large.\n\n"
            "Cramér's V (binary): association strength between groups. "
            "0.1 = small, 0.3 = medium, 0.5 = large.\n\n"
            "Effect size tells you practical importance — a tiny p-value with negligible "
            "effect size rarely matters in practice."
        ),
    )
    c3.metric(
        "Statistical Power",
        f"{results.get('power', 0):.2%}",
        help=(
            "Probability of detecting a true effect if one actually exists. "
            "Convention: ≥ 80% is acceptable. "
            "Below 80% means the test may have missed a real effect (Type II error). "
            "Power depends on sample size, effect size, and α."
        ),
    )
    c4.metric(
        "Result",
        sig_label,
        help=f"Compared against α = {results.get('alpha', 0.05)}. "
        "A significant result means the observed difference is unlikely under the null hypothesis. "
        "It does NOT prove causation.",
    )

    if results.get("outcome_type") == "continuous":
        test_explanations = {
            "Welch's t-test": "Compares means of two groups. Robust to unequal variances.",
            "Mann-Whitney U": "Non-parametric test — compares rank distributions, not means. Used when normality is violated.",
        }
        test_name = results["test_name"]
        test_tip = test_explanations.get(test_name, "")
        normality_ok = results.get("normal_assumption_met")
        st.markdown(
            f"""<div class="datrix-card" style="font-size:0.88rem;line-height:1.8">
            <span class="tt"><strong>{test_name}</strong><span class="tt-text">{test_tip}</span></span>
            &nbsp;·&nbsp;
            <span class="tt">Mean diff<span class="tt-text">
                Treatment mean minus control mean. Positive = treatment group scored higher.
            </span></span>: <strong>{results.get("mean_diff"):+.4f}</strong>
            &nbsp;·&nbsp;
            <span class="tt">95% CI<span class="tt-text">
                Confidence interval for the true mean difference. If this range excludes zero,
                the effect is statistically significant at α=0.05.
                Under repeated sampling, 95% of such intervals would contain the true value.
            </span></span>: <strong>{results.get("ci_95")}</strong>
            &nbsp;·&nbsp;
            <span class="tt">Normality<span class="tt-text">
                Shapiro-Wilk test checked whether each group's outcome is normally distributed.
                If not met, Mann-Whitney U is used instead of t-test.
            </span></span>: {"met" if normality_ok else "not met → Mann-Whitney used"}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        ctrl_rate = results.get("control_rate", 0)
        trt_rate = results.get("treatment_rate", 0)
        st.markdown(
            f"""<div class="datrix-card" style="font-size:0.88rem;line-height:1.8">
            <span class="tt"><strong>Chi-square test</strong><span class="tt-text">
                Tests whether group membership and outcome are independent.
                A significant result means the outcome distribution differs across groups.
            </span></span>
            &nbsp;·&nbsp;
            {results["control_label"]}: <strong>{ctrl_rate:.2%}</strong>
            &nbsp;·&nbsp;
            {results["treatment_label"]}: <strong>{trt_rate:.2%}</strong>
            &nbsp;·&nbsp;
            <span class="tt">Lift<span class="tt-text">
                Relative change in conversion rate: (treatment − control) / control × 100.
                A lift of +10% means the treatment group converted 10% more than control.
            </span></span>: <strong>{results.get("lift_pct"):+.1f}%</strong>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Group Statistics")
    st.dataframe(results["group_stats"], use_container_width=True)

    for name, fig in results.get("figures", {}).items():
        st.subheader(name.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    if st.button("↺ Start New Analysis", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# CAUSAL ANALYSIS STAGE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "causal":
    page_header(
        "Causal Analysis",
        "Treatment Effect Estimation",
        "Average Treatment Effect via propensity score adjustment",
    )
    df = st.session_state.df_raw
    treatment_col = st.session_state.get("ca_treatment_col")
    outcome_col = st.session_state.get("ca_outcome_col")
    covariate_cols = st.session_state.get("ca_covariate_cols", [])
    method = st.session_state.get("ca_method", "psm")

    if df is None or not treatment_col or not outcome_col or not covariate_cols:
        st.error("Missing configuration. Go back and set up the analysis.")
        st.stop()

    # Run analysis if not cached
    if st.session_state.causal_results is None:
        with st.spinner("Estimating treatment effect (bootstrap may take ~10s)..."):
            try:
                results = run_causal(
                    df, treatment_col, outcome_col, covariate_cols, method=method
                )
                st.session_state.causal_results = results
            except Exception as e:
                st.error(f"Causal analysis failed: {e}")
                st.stop()

    results = st.session_state.causal_results

    # AI interpretation
    if "causal" not in st.session_state.ai_response:
        with st.spinner("AI is interpreting results..."):
            try:
                response = st.session_state.analyst.analyze_causal(results)
                st.session_state.ai_response["causal"] = response
            except Exception as e:
                st.session_state.ai_response["causal"] = f"*(AI unavailable: {e})*"
    ai_panel("causal")

    st.divider()

    # Key metrics
    sig = results.get("significant")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Naive difference",
        f"{results.get('naive_diff'):+.4f}",
        help=(
            "Raw difference in average outcome between treated and untreated groups, "
            "with no adjustment for confounders. This is the biased estimate — "
            "the groups may differ on other variables that also affect the outcome."
        ),
    )
    c2.metric(
        "ATE (adjusted)",
        f"{results.get('ate'):+.4f}",
        help=(
            "Average Treatment Effect — the estimated causal effect of treatment on the outcome, "
            "after adjusting for observed confounders via propensity score methods. "
            "Interpreted as: 'on average, being treated changes the outcome by this amount.'"
        ),
    )
    c3.metric(
        "95% CI",
        f"[{results['ate_ci'][0]:+.3f}, {results['ate_ci'][1]:+.3f}]",
        help=(
            "Bootstrap confidence interval for the ATE. "
            "If this range excludes zero, the effect is statistically distinguishable from no effect. "
            "Wider CI = more uncertainty (small sample or high variance)."
        ),
    )
    c4.metric(
        "Significant",
        "✅ Yes" if sig else "❌ No (CI crosses zero)",
        help=(
            "Significance here means the 95% CI excludes zero — "
            "the adjusted effect is unlikely to be zero. "
            "Note: this does NOT account for unmeasured confounders."
        ),
    )

    st.markdown(
        f"""<div class="datrix-card" style="font-size:0.88rem;line-height:1.8">
        <span class="tt"><strong>Method</strong><span class="tt-text">
            PSM (Propensity Score Matching): pairs each treated unit with the most similar
            control unit based on propensity score. Creates a matched sample.<br><br>
            IPW (Inverse Probability Weighting): reweights all observations so the
            treated and control groups look similar, without discarding data.
        </span></span>: {results.get("method")}
        &nbsp;·&nbsp;
        <span class="tt">Confounder bias removed<span class="tt-text">
            How much the ATE estimate shifted after adjusting for confounders,
            relative to the naive difference. Large values mean confounders were
            significantly distorting the raw comparison.
        </span></span>: <strong>{results.get("bias_reduction_pct")}%</strong>
        &nbsp;·&nbsp;
        Matched n: <strong>{results.get("matched_n") or "N/A (IPW)"}</strong>
        </div>""",
        unsafe_allow_html=True,
    )

    # Balance table
    balance = results.get("balance_stats")
    if balance is not None and not balance.empty:
        st.subheader("Covariate Balance")
        st.markdown(
            """<p style="font-size:0.82rem;color:#94a3b8">
            <span class="tt" style="color:#94a3b8">SMD (Standardized Mean Difference)
            <span class="tt-text">Measures how different the treated and control groups are
            on each covariate, in standard deviation units. |SMD| &lt; 0.1 = well balanced.
            Values above 0.1 after matching indicate residual imbalance that may bias the ATE.
            </span></span> — lower is better after matching.
            </p>""",
            unsafe_allow_html=True,
        )
        st.dataframe(balance, use_container_width=True)

    for name, fig in results.get("figures", {}).items():
        st.subheader(name.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    if st.button("↺ Start New Analysis", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()




# ════════════════════════════════════════════════════════════════════════════════
# MODEL ADVISOR STAGE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "model_advisor":
    page_header(
        "Model Advisor",
        "Find the right approach for your problem",
        "One question at a time — get a tailored model, metric, and pitfall guide",
    )

    # ── Decision tree ─────────────────────────────────────────────────────────
    TREE = {
        "problem_type": {
            "q": "What type of machine learning problem are you solving?",
            "help": "If you have labeled training examples, choose Supervised. If you want to find structure in raw data, choose Unsupervised.",
            "options": [
                "Supervised — I have labeled training data",
                "Unsupervised — no labels, discover structure",
                "Anomaly / outlier detection",
                "Time series — data ordered in time",
                "NLP / text data",
                "Computer vision / images",
                "Causal inference — estimate the effect of an intervention",
                "A/B testing — compare two or more variants",
            ],
            "next": {
                "supervised": "sup_output",
                "unsupervised": "unsup_goal",
                "anomaly": "anomaly_labels",
                "time series": "ts_task",
                "nlp": "nlp_task",
                "computer vision": "vision_task",
                "causal inference": "causal_goal",
                "a/b testing": "ab_metric_type",
            },
        },
        # ── Supervised ────────────────────────────────────────────────────────
        "sup_output": {
            "q": "What does your model need to output?",
            "help": "This determines the loss function and evaluation metric family.",
            "options": [
                "A category or class label (classification)",
                "A numeric value (regression)",
                "A ranked list of items (ranking / recommendation)",
            ],
            "next": {
                "category": "clf_modality",
                "numeric": "reg_modality",
                "ranked": "ranking_type",
            },
        },
        # ── Classification ────────────────────────────────────────────────────
        "clf_modality": {
            "q": "What does your input data look like?",
            "help": "Data modality determines the model architecture family.",
            "options": [
                "Tabular — rows and columns (CSV, database tables)",
                "Text / language",
                "Images or video",
                "Time series / sequential measurements",
            ],
            "next": {
                "tabular": "clf_tab_classes",
                "text": "nlp_task",
                "images": "vision_task",
                "time series": "ts_task",
            },
        },
        "clf_tab_classes": {
            "q": "How many output classes are there?",
            "help": "Multi-label means a single sample can belong to multiple classes at once (e.g. a movie tagged as both Action and Comedy).",
            "options": [
                "Binary — exactly 2 classes (yes/no, churn/stay, fraud/legit)",
                "Multi-class — 3 or more mutually exclusive classes",
                "Multi-label — each sample can have multiple simultaneous labels",
            ],
            "next": {
                "binary": "clf_tab_imbalance",
                "multi-class": "clf_tab_imbalance",
                "multi-label": "clf_tab_interp",
            },
        },
        "clf_tab_imbalance": {
            "q": "How balanced are the classes in your dataset?",
            "help": "Severe imbalance (e.g. 1% fraud cases in 99% legitimate) changes which metrics and training strategies are appropriate.",
            "options": [
                "Balanced — classes are roughly equal",
                "Mild imbalance — minority class is 10–30% of data",
                "Severe imbalance — minority class is under 10%",
            ],
            "next": {
                "balanced": "clf_tab_interp",
                "mild": "clf_tab_interp",
                "severe": "clf_tab_interp",
            },
        },
        "clf_tab_interp": {
            "q": "How important is interpretability?",
            "help": "Regulated industries (finance, healthcare, hiring) often legally require explainable decisions.",
            "options": [
                "High — regulators or executives need to understand every prediction",
                "Medium — SHAP plots and feature importance are sufficient",
                "Low — only performance matters",
            ],
            "next": {
                "high": "clf_tab_probs",
                "medium": "clf_tab_probs",
                "low": "clf_tab_probs",
            },
        },
        "clf_tab_probs": {
            "q": "Do you need calibrated probability scores or just class labels?",
            "help": "Calibrated: P(churn)=0.7 means 70% of flagged players actually churn. Required when probabilities feed into scoring systems, EV formulas, or risk tiers.",
            "options": [
                "Yes — probabilities feed into a ranking, formula, or risk system",
                "No — I only need the predicted class",
            ],
            "next": {
                "yes": "rec_clf_tabular",
                "no": "rec_clf_tabular",
            },
        },
        # ── Regression ────────────────────────────────────────────────────────
        "reg_modality": {
            "q": "What does your input data look like?",
            "options": [
                "Tabular — structured rows and columns",
                "Time series — predict future values of a sequence",
                "Text — produce a numeric score from text",
                "Images — predict a measurement from an image",
            ],
            "next": {
                "tabular": "reg_tab_dist",
                "time series": "ts_task",
                "text": "nlp_task",
                "images": "vision_task",
            },
        },
        "reg_tab_dist": {
            "q": "What does your target variable distribution look like?",
            "help": "Skewed targets need log-transformation or RMSLE. Outlier-heavy distributions need robust losses. Bounded targets may need a sigmoid output.",
            "options": [
                "Normal / symmetric — roughly bell-curve shaped",
                "Right-skewed — long tail (revenue, counts, durations)",
                "Heavy outliers — extreme values dominate the variance",
                "Bounded — values in a fixed range (e.g. 0–1 ratings, percentages)",
            ],
            "next": {
                "normal": "reg_tab_rel",
                "right-skewed": "reg_tab_rel",
                "heavy outliers": "reg_tab_rel",
                "bounded": "reg_tab_rel",
            },
        },
        "reg_tab_rel": {
            "q": "How complex is the relationship between features and target?",
            "help": "Look at scatter plots and correlation coefficients. If they're near-linear, start simple. Non-linearity and interactions between features often benefit from tree models.",
            "options": [
                "Mostly linear — linear models perform close to tree models",
                "Non-linear with interactions — features combine in complex, non-obvious ways",
                "Unknown — I have not explored this yet",
            ],
            "next": {
                "mostly linear": "reg_tab_interp",
                "non-linear": "reg_tab_interp",
                "unknown": "reg_tab_interp",
            },
        },
        "reg_tab_interp": {
            "q": "How important is interpretability for this regression task?",
            "options": [
                "High — I need to explain individual predictions or regression coefficients",
                "Low — minimising prediction error is the only priority",
            ],
            "next": {
                "high": "rec_reg_tabular",
                "low": "rec_reg_tabular",
            },
        },
        # ── Ranking ───────────────────────────────────────────────────────────
        "ranking_type": {
            "q": "What kind of ranking or recommendation system are you building?",
            "options": [
                "Collaborative filtering — based on user–item interaction history",
                "Content-based — items have features, match users to item attributes",
                "Hybrid — both interaction history and item features available",
                "Learning-to-rank — explicit relevance labels (e.g. search, ads)",
            ],
            "next": {
                "collaborative": "rec_collab",
                "content-based": "rec_content",
                "hybrid": "rec_hybrid",
                "learning-to-rank": "rec_ltr",
            },
        },
        # ── Time series ───────────────────────────────────────────────────────
        "ts_task": {
            "q": "What is the time series task?",
            "options": [
                "Forecasting — predict future values",
                "Classification — assign a label to a sequence",
                "Anomaly detection — find unusual patterns over time",
            ],
            "next": {
                "forecasting": "ts_horizon",
                "classification": "rec_ts_clf",
                "anomaly": "rec_ts_anomaly",
            },
        },
        "ts_horizon": {
            "q": "What is your forecast horizon?",
            "help": "Short-horizon forecasting (next 1–7 steps) and long-horizon (90+ steps) require different approaches. Models that excel at one often underperform on the other.",
            "options": [
                "Short — next 1 to 7 time steps",
                "Medium — next 7 to 90 steps",
                "Long — more than 90 steps ahead",
            ],
            "next": {
                "short": "ts_series_count",
                "medium": "ts_series_count",
                "long": "ts_series_count",
            },
        },
        "ts_series_count": {
            "q": "How many time series are you working with?",
            "help": "Global models (trained across many series) often outperform local models (one per series) when series are related. But they need enough data per series.",
            "options": [
                "One — a single time series",
                "A few — 2 to ~50 related series",
                "Many — hundreds or thousands of related series",
            ],
            "next": {
                "one": "rec_ts_forecast",
                "few": "rec_ts_forecast",
                "many": "rec_ts_forecast",
            },
        },
        # ── NLP ───────────────────────────────────────────────────────────────
        "nlp_task": {
            "q": "What is the NLP task?",
            "options": [
                "Text classification — assign a label (sentiment, topic, intent, spam)",
                "Text generation — produce text (summarisation, QA, translation, chat)",
                "Information extraction — pull out entities, relations, or structured data",
                "Embeddings / semantic similarity — represent text as vectors for search or clustering",
            ],
            "next": {
                "classification": "nlp_size",
                "generation": "rec_nlp_gen",
                "extraction": "rec_nlp_extraction",
                "embeddings": "rec_nlp_emb",
            },
        },
        "nlp_size": {
            "q": "How large is your labelled text dataset?",
            "help": "With fewer than 1,000 examples, fine-tuning a pretrained transformer beats training from scratch by a huge margin.",
            "options": [
                "Small — fewer than 1,000 labelled examples",
                "Medium — 1,000 to 50,000 labelled examples",
                "Large — more than 50,000 labelled examples",
            ],
            "next": {
                "small": "rec_nlp_clf",
                "medium": "rec_nlp_clf",
                "large": "rec_nlp_clf",
            },
        },
        # ── Vision ────────────────────────────────────────────────────────────
        "vision_task": {
            "q": "What is the computer vision task?",
            "options": [
                "Image classification — assign a single label to the whole image",
                "Object detection — locate and classify multiple objects in one image",
                "Segmentation — label every pixel (semantic or instance)",
                "Embeddings / similarity — represent images as vectors for search or clustering",
            ],
            "next": {
                "classification": "rec_vision_clf",
                "detection": "rec_vision_det",
                "segmentation": "rec_vision_seg",
                "embeddings": "rec_vision_emb",
            },
        },
        # ── Unsupervised ──────────────────────────────────────────────────────
        "unsup_goal": {
            "q": "What is the goal of the unsupervised analysis?",
            "options": [
                "Clustering — discover natural groups in the data",
                "Dimensionality reduction — compress or visualise high-dimensional data",
                "Generative modelling — learn to generate new synthetic samples",
            ],
            "next": {
                "clustering": "clustering_k",
                "dimensionality": "dimred_purpose",
                "generative": "rec_generative",
            },
        },
        "clustering_k": {
            "q": "Do you know how many clusters to expect?",
            "help": "If you have domain knowledge (e.g. 4 customer segments), use it. If not, the algorithm can determine the number automatically.",
            "options": [
                "Yes — I have a good estimate of k",
                "No — I want the data to decide",
            ],
            "next": {
                "yes": "clustering_shape",
                "no": "clustering_shape",
            },
        },
        "clustering_shape": {
            "q": "What shape do you expect the clusters to have?",
            "help": "K-Means only finds round, equally-sized clusters. Real data is often more complex.",
            "options": [
                "Round and compact — spherical, roughly equal-size groups",
                "Arbitrary shapes — elongated, ring-shaped, or irregular",
                "Hierarchical — nested structure at different granularities",
                "Very unequal sizes — a few large clusters and many tiny niche groups",
            ],
            "next": {
                "round": "rec_clustering",
                "arbitrary": "rec_clustering",
                "hierarchical": "rec_clustering",
                "unequal": "rec_clustering",
            },
        },
        "dimred_purpose": {
            "q": "What is the dimensionality reduction for?",
            "options": [
                "Visualisation — plot high-dimensional data in 2D or 3D",
                "Feature compression — reduce input size before a downstream model",
                "Noise removal — retain signal, discard noisy variance",
            ],
            "next": {
                "visualisation": "dimred_linear",
                "feature compression": "dimred_linear",
                "noise": "dimred_linear",
            },
        },
        "dimred_linear": {
            "q": "Are the relationships in your data mostly linear?",
            "help": "PCA finds linear directions of maximum variance. Non-linear methods (UMAP, t-SNE) can capture manifold structure — useful for images, text embeddings, or molecular data.",
            "options": [
                "Linear — data clusters along straight axes",
                "Non-linear — data lives on a curved manifold",
                "Unknown",
            ],
            "next": {
                "linear": "rec_dimred",
                "non-linear": "rec_dimred",
                "unknown": "rec_dimred",
            },
        },
        # ── Anomaly detection ─────────────────────────────────────────────────
        "anomaly_labels": {
            "q": "Do you have labeled anomaly examples?",
            "help": "Even a small number of confirmed anomaly labels converts this from unsupervised (hard) to semi-supervised (much easier).",
            "options": [
                "No labels — fully unsupervised",
                "A small number of confirmed anomalies — semi-supervised",
                "Fully labeled dataset (both normal and anomalous) — treat as classification",
            ],
            "next": {
                "no labels": "anomaly_data",
                "small number": "anomaly_data",
                "fully labeled": "clf_modality",
            },
        },
        "anomaly_data": {
            "q": "What type of data are you detecting anomalies in?",
            "options": [
                "Tabular — transactions, sensor readings, user events",
                "Time series — detecting unusual patterns or changepoints over time",
                "Text — unusual documents, log messages, or queries",
                "Images — detecting defective or abnormal items",
            ],
            "next": {
                "tabular": "rec_anomaly_tabular",
                "time series": "rec_ts_anomaly",
                "text": "rec_nlp_anomaly",
                "images": "rec_vision_anomaly",
            },
        },
        # ── Causal inference ──────────────────────────────────────────────────
        "causal_goal": {
            "q": "What is the causal inference goal?",
            "help": "Causal inference estimates what would have happened under different conditions — something correlation alone cannot answer.",
            "options": [
                "Estimate the average effect of a treatment or intervention (ATE)",
                "Find which variables causally affect the outcome (causal discovery)",
                "Estimate individual-level treatment effects (uplift / heterogeneous effects)",
                "Build a causal graph / DAG from observational data",
            ],
            "next": {
                "average effect": "causal_data",
                "which variables": "rec_causal_discovery",
                "individual-level": "causal_data",
                "causal graph": "rec_causal_discovery",
            },
        },
        "causal_data": {
            "q": "What data do you have?",
            "options": [
                "Randomised experiment (RCT) — treatment was assigned randomly",
                "Observational data — treatment was not randomised, confounders exist",
                "Observational data with a natural experiment or instrument",
            ],
            "next": {
                "randomised": "rec_causal_rct",
                "observational data —": "rec_causal_obs",
                "natural experiment": "rec_causal_iv",
            },
        },
        # ── A/B testing ───────────────────────────────────────────────────────
        "ab_metric_type": {
            "q": "What type of outcome are you measuring?",
            "options": [
                "Continuous — revenue, session duration, score",
                "Binary / conversion — clicked, purchased, churned (yes/no)",
                "Count or rate — sessions per user, events per day",
                "Time-to-event — survival, time until purchase or churn",
            ],
            "next": {
                "continuous": "ab_sample",
                "binary": "ab_sample",
                "count": "ab_sample",
                "time-to-event": "rec_ab_survival",
            },
        },
        "ab_sample": {
            "q": "Do you have enough sample size or is this still being planned?",
            "options": [
                "Planning — I need to calculate required sample size before running",
                "Already running or completed — I need to analyse the results",
            ],
            "next": {
                "planning": "rec_ab_planning",
                "already running": "rec_ab_analysis",
            },
        },
    }

    # ── Recommendations ────────────────────────────────────────────────────────
    RECS = {
        "rec_clf_tabular": {
            "title": "Tabular Classification",
            "models": [
                ("LightGBM / XGBoost", "Best default for tabular data — fast, handles missing values, imbalance-aware via class_weight"),
                ("Random Forest", "Robust, less sensitive to hyperparameters, reliable feature importance"),
                ("Logistic Regression", "Use when interpretability is high — coefficients are directly readable"),
                ("SVM (RBF kernel)", "Strong on small datasets with clear decision boundaries"),
            ],
            "loss": "Binary cross-entropy (binary) · Categorical cross-entropy (multi-class) · Binary CE per label (multi-label)",
            "primary_metric": "AUC-ROC (balanced) · AUC-PR (imbalanced binary) · Macro F1 (multi-class)",
            "secondary_metrics": ["F1 / Precision / Recall", "Log loss (calibration quality)", "Confusion matrix"],
            "watch_out": [
                "**Imbalance:** Accuracy is misleading when classes are imbalanced. Use AUC-PR or F1, and set class_weight='balanced'.",
                "**Calibration:** LightGBM with class_weight='balanced' often produces miscalibrated probabilities. Always check the calibration curve and wrap with CalibratedClassifierCV(method='isotonic') if probabilities matter.",
                "**Threshold:** The default 0.5 threshold is almost never optimal. Use the Precision-Recall curve to find the right operating point for your cost asymmetry.",
                "**Data leakage:** Features derived from the target (or from post-event data) will inflate metrics dramatically. Validate feature timestamps carefully.",
                "**Overfitting on small datasets:** With fewer than 5K rows, tree ensembles overfit. Use cross-validation, not a single train/val split.",
            ],
        },
        "rec_reg_tabular": {
            "title": "Tabular Regression",
            "models": [
                ("LightGBM / XGBoost", "Best default — handles non-linearity, interactions, and missing values natively"),
                ("Ridge Regression", "Strong linear baseline with regularisation; use when interpretability is high"),
                ("Lasso Regression", "Like Ridge but also performs feature selection — good when many features are irrelevant"),
                ("Random Forest", "Robust non-linear baseline with reliable feature importance"),
            ],
            "loss": "MSE (symmetric target) · MAE / Huber (outlier-heavy) · RMSLE (skewed/log-scale)",
            "primary_metric": "RMSE (normal target) · MAE (outliers) · RMSLE (skewed)",
            "secondary_metrics": ["R² (proportion of variance explained)", "MAPE (percentage error, avoid when target near zero)", "Residual plots"],
            "watch_out": [
                "**Skewed targets:** Log-transform skewed targets before training. Predict log(y+1), then exponentiate predictions. Use RMSLE as the loss.",
                "**Outliers in target:** MSE squares errors, so outliers dominate. Use Huber loss or MAE, or clip the most extreme values.",
                "**Residual patterns:** Plot residuals vs predicted values. Any visible pattern (fan shape, curve) means model assumptions are violated.",
                "**R² can be misleading:** A high R² doesn't mean the model is well-calibrated. Always inspect residual distributions.",
                "**Feature scaling:** Tree models (LightGBM, RF) don't need scaling. Linear models (Ridge, Lasso) do — always scale for those.",
            ],
        },
        "rec_collab": {
            "title": "Collaborative Filtering",
            "models": [
                ("Matrix Factorisation (SVD / ALS)", "Classic baseline — decomposes user–item matrix into latent factors"),
                ("Neural Collaborative Filtering", "Replaces dot product with MLP — captures non-linear interactions"),
                ("Transformer-based (e.g. SASRec, BERT4Rec)", "Models the temporal order of interactions — best for sequential recommendation"),
                ("LightFM", "Hybrid: adds item/user features to MF — handles cold start better"),
            ],
            "loss": "BPR (Bayesian Personalised Ranking) · BCE on positive/negative pairs · Cross-entropy (next-item prediction)",
            "primary_metric": "NDCG@K · Hit@K · Recall@K",
            "secondary_metrics": ["MRR (mean reciprocal rank)", "Coverage (% of catalogue recommended)", "Novelty / diversity"],
            "watch_out": [
                "**Cold start:** Pure CF cannot recommend to new users or new items with no interaction history. You need a content-based fallback or hybrid model.",
                "**Popularity bias:** CF tends to recommend popular items repeatedly. Measure coverage and diversity alongside accuracy metrics.",
                "**Implicit vs explicit feedback:** Ratings are sparse explicit signal. Clicks/views are implicit but noisy. Model them differently — implicit feedback needs negative sampling.",
                "**Evaluation pitfall:** Random train/test split leaks future data into training. Always split by timestamp.",
                "**Scalability:** Full matrix factorisation is O(users × items). Use approximate nearest-neighbour search (FAISS, ScaNN) at inference.",
            ],
        },
        "rec_content": {
            "title": "Content-Based Recommendation",
            "models": [
                ("Cosine similarity on TF-IDF / embeddings", "Fast baseline — represent items as vectors, return nearest neighbours"),
                ("Two-tower neural model", "Separate encoders for user and item — scalable, good for large catalogues"),
                ("Logistic Regression on user–item feature pairs", "Interpretable, works well with rich engineered features"),
            ],
            "loss": "BCE on positive/negative pairs · Contrastive loss (two-tower)",
            "primary_metric": "NDCG@K · Recall@K",
            "secondary_metrics": ["Coverage", "Diversity", "Serendipity"],
            "watch_out": [
                "**Feature quality is everything:** Content-based systems are only as good as the item features. Invest in rich, clean metadata.",
                "**Filter bubble:** Users get items similar to what they already know — no serendipity. Mix in popularity or random exploration.",
                "**Stale embeddings:** If items change frequently, re-embed regularly or use online updates.",
            ],
        },
        "rec_hybrid": {
            "title": "Hybrid Recommendation",
            "models": [
                ("LightFM", "MF + side features in one model — best starting point for hybrid"),
                ("Two-tower model with interaction features", "Encode user history + item features separately, combine at scoring"),
                ("Ensemble of CF + content scores", "Blend CF and content rankings with a learned or heuristic weight"),
            ],
            "loss": "BPR · BCE · Contrastive",
            "primary_metric": "NDCG@K",
            "secondary_metrics": ["Hit@K", "Coverage", "Cold-start Hit@K"],
            "watch_out": [
                "**Complexity creep:** Hybrids are harder to debug. Start with separate CF and content models, then blend.",
                "**Cold start coverage:** Measure hit rate specifically for new users and new items — that's the main reason to go hybrid.",
            ],
        },
        "rec_ltr": {
            "title": "Learning to Rank",
            "models": [
                ("LightGBM with LambdaRank", "Best overall — gradient boosting tuned directly on NDCG"),
                ("XGBoost rank:pairwise", "Pairwise ranking loss — optimises relative order of document pairs"),
                ("Linear model (RankSVM)", "Fast interpretable baseline for low-latency serving"),
            ],
            "loss": "LambdaRank (listwise NDCG proxy) · Pairwise hinge · Pointwise cross-entropy",
            "primary_metric": "NDCG@K · MAP@K",
            "secondary_metrics": ["MRR", "Precision@K"],
            "watch_out": [
                "**Label quality:** LTR is only as good as the relevance labels. Noisy labels (click data) need careful treatment — clicks are biased by position.",
                "**Position bias:** Users click top results more regardless of relevance. Use inverse propensity scoring or randomised logging to correct.",
                "**Evaluation at the right K:** Choose K to match the actual UI (e.g. K=10 for a 10-result page). NDCG@100 is meaningless if users only see 5 results.",
            ],
        },
        "rec_ts_forecast": {
            "title": "Time Series Forecasting",
            "models": [
                ("LightGBM with lag + rolling features", "Best tabular approach — create lag features, rolling stats, then treat as regression"),
                ("Prophet", "Good for business time series with trend + seasonality + holidays. Fast to deploy."),
                ("N-BEATS / N-HiTS", "Pure neural, no feature engineering needed — strong on medium/long horizons"),
                ("Temporal Fusion Transformer (TFT)", "Best for many related series with covariates — state of the art on M5"),
                ("ARIMA / ETS", "Classical baselines — fast to fit, interpretable, use as benchmark"),
            ],
            "loss": "MAE · MSE · MASE (mean absolute scaled error, scale-independent)",
            "primary_metric": "RMSE or MAE (short horizon) · MASE (comparable across series) · MAPE (avoid when target near zero)",
            "secondary_metrics": ["Prediction interval coverage", "Bias (mean error)", "Quantile loss"],
            "watch_out": [
                "**Data leakage via the future:** Any feature computed using data after the forecast origin will leak. Be extremely careful with rolling aggregations — use expanding windows or properly lagged windows.",
                "**Backtest with walk-forward validation:** Random splits are wrong for time series. Use rolling-origin evaluation.",
                "**Seasonality and trend non-stationarity:** Many models assume stationarity. Test with ADF/KPSS, and difference or detrend if needed.",
                "**Forecast horizon vs model:** ARIMA and LSTM struggle at long horizons. N-HiTS and TFT are designed for long-range.",
                "**Prediction intervals matter:** Point forecasts alone are dangerous for planning. Report 80% and 95% prediction intervals.",
                "**Many series:** Training one model per series is rarely optimal. Global models (LightGBM trained on all series, TFT) usually win when series are related.",
            ],
        },
        "rec_ts_clf": {
            "title": "Time Series Classification",
            "models": [
                ("ROCKET / MiniROCKET", "State-of-the-art accuracy, extremely fast — random convolutional kernels + Ridge classifier"),
                ("LightGBM on statistical features", "Extract mean, std, autocorrelation, entropy per series — then standard tabular classification"),
                ("1D-CNN", "Good when local pattern shape matters — beats LSTMs on many benchmarks"),
                ("InceptionTime", "Ensemble of 1D-CNNs at multiple scales — strong default deep learning baseline"),
            ],
            "loss": "Cross-entropy",
            "primary_metric": "Accuracy (balanced) · Macro F1 (imbalanced classes)",
            "secondary_metrics": ["Per-class F1", "Confusion matrix"],
            "watch_out": [
                "**Variable-length series:** Most models require fixed-length input. Pad/truncate, or use models designed for variable length (e.g. shapelets).",
                "**ROCKET is underrated:** It beats deep learning on most UCR benchmark datasets while training in seconds. Try it before anything more complex.",
                "**Label granularity:** Is the label for the whole sequence or per time step? Sequence-level and segmentation tasks need different architectures.",
            ],
        },
        "rec_ts_anomaly": {
            "title": "Time Series Anomaly Detection",
            "models": [
                ("Isolation Forest on rolling window features", "Fast unsupervised baseline — extract rolling stats, then score each window"),
                ("LSTM Autoencoder", "Learns normal patterns, flags high reconstruction error as anomaly"),
                ("Prophet residual anomaly detection", "Fit Prophet, flag points where residuals exceed threshold"),
                ("Seasonal-Trend decomposition (STL) + residuals", "Simple, interpretable — anomalies are points where residual is extreme"),
            ],
            "loss": "Reconstruction MSE (autoencoder) · Anomaly score (isolation forest)",
            "primary_metric": "AUC-ROC (if labels exist) · F1@threshold · Precision@K",
            "secondary_metrics": ["Time-to-detect (lag)", "False alarm rate"],
            "watch_out": [
                "**Threshold selection is hard:** Without labels, there's no objective way to set the anomaly threshold. Use domain knowledge or percentile cutoffs.",
                "**Seasonality fools simple methods:** A spike that's normal on Monday is anomalous on Sunday. Model seasonality explicitly before flagging residuals.",
                "**Point anomalies vs contextual:** A value of 100 may be normal in summer but anomalous in winter. Ensure your model has access to temporal context.",
                "**Evaluation without labels:** Use held-out labeled incidents to validate, even if labels are partial.",
            ],
        },
        "rec_nlp_clf": {
            "title": "Text Classification",
            "models": [
                ("Fine-tune BERT / RoBERTa / DeBERTa", "Best default for most text classification tasks — add a linear head on [CLS] token"),
                ("DistilBERT", "50% smaller, 60% faster than BERT — good when latency matters and accuracy loss is acceptable"),
                ("SetFit (few-shot fine-tuning)", "Excellent with fewer than 100 examples — contrastive fine-tuning of sentence embeddings"),
                ("TF-IDF + Logistic Regression", "Strong baseline, interpretable, fast — often within 5% of transformers on clean data"),
            ],
            "loss": "Cross-entropy (single label) · BCE per class (multi-label)",
            "primary_metric": "AUC-ROC or Macro F1",
            "secondary_metrics": ["Per-class F1", "Confusion matrix", "Calibration (log loss)"],
            "watch_out": [
                "**Domain mismatch:** BERT pretrained on Wikipedia+Books may underperform on specialised domains (medical, legal, code). Use a domain-specific checkpoint.",
                "**Class imbalance in NLP:** Oversampling minority class text or using weighted loss helps more than SMOTE (text can't be interpolated like tabular features).",
                "**Sequence length:** BERT truncates at 512 tokens. For long documents, use longformer, sliding window, or hierarchical approaches.",
                "**Fine-tuning instability:** Transformers fine-tuning can be unstable on small datasets. Run 3–5 seeds and report mean ± std.",
                "**SetFit for small data:** If you have <500 labelled examples, SetFit almost always beats full fine-tuning.",
            ],
        },
        "rec_nlp_gen": {
            "title": "Text Generation",
            "models": [
                ("Fine-tune GPT-2 / GPT-Neo / LLaMA", "Autoregressive generation — fine-tune for summarisation, QA, or creative tasks"),
                ("Fine-tune T5 / BART / mT5", "Encoder-decoder — better for seq2seq tasks (summarisation, translation, QA with extractive answers)"),
                ("Prompt + few-shot with Claude / GPT-4", "Often outperforms fine-tuned smaller models, especially with few examples"),
                ("RAG (Retrieval-Augmented Generation)", "Combine retrieval + LLM — reduces hallucination and keeps knowledge current"),
            ],
            "loss": "Causal language modelling loss (next-token cross-entropy)",
            "primary_metric": "ROUGE-L (summarisation) · BLEU (translation) · BERTScore · Human eval",
            "secondary_metrics": ["Perplexity", "Faithfulness (factual grounding)", "Hallucination rate"],
            "watch_out": [
                "**Hallucination:** LLMs confidently generate false facts. Use RAG, grounding checks, or constitutional AI techniques.",
                "**Evaluation is hard:** ROUGE and BLEU are weak proxies for quality. Always supplement with human evaluation or LLM-as-judge.",
                "**Prompt engineering first:** Before fine-tuning, exhaustively test prompts on a capable base model. Fine-tuning is expensive and often unnecessary.",
                "**RLHF / RLAIF:** If you need to align outputs with preferences, RLHF or DPO is the current best practice.",
            ],
        },
        "rec_nlp_extraction": {
            "title": "Information Extraction (NER / Relation)",
            "models": [
                ("Fine-tune BERT with token classification head", "Standard approach for NER — BIO tagging over tokens"),
                ("Span-based extraction (SpanBERT)", "Better for overlapping or nested entities"),
                ("GLiNER", "Generalised NER — zero-shot entity extraction without task-specific fine-tuning"),
                ("spaCy + custom NER", "Fast deployment, good tooling — strong baseline for production pipelines"),
            ],
            "loss": "Token-level cross-entropy (NER) · Span classification loss",
            "primary_metric": "Entity-level F1 (strict match) · Span F1",
            "secondary_metrics": ["Boundary F1 (partial match)", "Per-entity-type F1"],
            "watch_out": [
                "**Entity boundary sensitivity:** Strict F1 penalises even one-token boundary errors harshly. Always report both strict and partial/boundary F1.",
                "**Nested entities:** Standard BIO tagging can't handle overlapping entities. Use span-based models if your schema has nesting.",
                "**Annotation consistency:** Extraction quality is bounded by label quality. Inter-annotator agreement (Cohen's κ) should be above 0.8 before training.",
            ],
        },
        "rec_nlp_emb": {
            "title": "Text Embeddings / Semantic Similarity",
            "models": [
                ("Sentence-BERT (SBERT) / all-MiniLM-L6-v2", "Fast, strong baseline for semantic similarity — fine-tune with contrastive loss"),
                ("OpenAI text-embedding-3-large / Cohere Embed", "API-based, no training needed — excellent for search and clustering"),
                ("E5 / BGE / GTE models", "Open-source, strong on MTEB benchmark — good for on-premise deployment"),
                ("ColBERT", "Late interaction — better retrieval quality, higher compute cost"),
            ],
            "loss": "Contrastive loss (InfoNCE) · Triplet loss · Multiple Negatives Ranking",
            "primary_metric": "NDCG@10 (retrieval) · Spearman ρ (similarity) · MTEB score",
            "secondary_metrics": ["Recall@K", "Mean Average Precision (MAP)"],
            "watch_out": [
                "**Domain gap:** General embeddings underperform on specialised domains (legal, biomedical, code). Fine-tune with in-domain pairs.",
                "**Approximate nearest-neighbour at scale:** Exact cosine search is O(n) per query. Use FAISS, Annoy, or ScaNN for large-scale retrieval.",
                "**Embedding drift:** If the underlying model is updated, all stored embeddings become stale. Plan for re-embedding pipelines.",
            ],
        },
        "rec_nlp_anomaly": {
            "title": "Text Anomaly Detection",
            "models": [
                ("Embed + Isolation Forest", "Embed documents with SBERT, then score with Isolation Forest in embedding space"),
                ("Perplexity-based scoring (language model)", "High-perplexity text is unusual — use a language model to score each document"),
                ("One-class SVM on TF-IDF", "Simple baseline for detecting out-of-distribution documents"),
            ],
            "loss": "Anomaly score (path length / distance)",
            "primary_metric": "AUC-ROC (if labels exist) · Precision@K",
            "secondary_metrics": ["False positive rate at operating threshold"],
            "watch_out": [
                "**Short texts are hard:** Embeddings of very short texts are noisy. Aggregate or use character n-gram features for logs and short messages.",
                "**Novelty vs anomaly:** Novel topics look anomalous but may be legitimate. Domain filtering before anomaly scoring helps.",
            ],
        },
        "rec_vision_clf": {
            "title": "Image Classification",
            "models": [
                ("Fine-tune ResNet / EfficientNet / ConvNeXt", "Strong CNN baselines — pretrained on ImageNet, add a linear head"),
                ("Fine-tune ViT / DeiT", "Vision Transformer — better than CNNs on large datasets, needs more data"),
                ("CLIP + linear probe", "Zero/few-shot — embed images and classify by similarity to text descriptions"),
                ("EfficientNet-B0", "Best accuracy/compute tradeoff for resource-constrained deployment"),
            ],
            "loss": "Cross-entropy (single label) · BCE per class (multi-label)",
            "primary_metric": "Top-1 Accuracy (balanced) · Macro F1 (imbalanced)",
            "secondary_metrics": ["AUC-ROC", "Confusion matrix"],
            "watch_out": [
                "**Data augmentation is critical:** Randomly flip, crop, colour-jitter, and rotate during training. Without it, models overfit to image-specific artefacts.",
                "**Pretrain → fine-tune:** Never train from scratch unless you have millions of images. Use ImageNet pretrained weights.",
                "**Class imbalance:** Weighted sampling or class_weight in loss is more stable than oversampling at the image level.",
                "**Image resolution vs batch size:** Higher resolution improves accuracy but reduces batch size and slows training. Find the best tradeoff for your GPU.",
            ],
        },
        "rec_vision_det": {
            "title": "Object Detection",
            "models": [
                ("YOLOv8 / YOLOv9", "Best speed–accuracy tradeoff for real-time detection — 1-stage, easy to deploy"),
                ("RT-DETR", "Transformer-based, no NMS post-processing needed, strong accuracy"),
                ("Faster R-CNN", "2-stage detector — higher accuracy but slower, good when precision matters more than latency"),
                ("DINO / GroundingDINO", "Zero-shot detection — detect arbitrary objects described in text"),
            ],
            "loss": "Classification loss + Regression loss (bounding box) + Objectness loss",
            "primary_metric": "mAP@0.5 · mAP@0.5:0.95",
            "secondary_metrics": ["AP per class", "Inference latency (ms)", "FPS"],
            "watch_out": [
                "**Annotation quality is the bottleneck:** Bounding box annotation is expensive and error-prone. Use label quality checks and inter-annotator agreement.",
                "**IoU threshold matters:** mAP@0.5 and mAP@0.5:0.95 tell different stories. Report both.",
                "**Small object detection:** Default anchor sizes miss small objects. Tune anchor scales or use models with FPN (Feature Pyramid Network).",
                "**NMS tuning:** Non-maximum suppression threshold affects recall vs precision. Tune for your use case.",
            ],
        },
        "rec_vision_seg": {
            "title": "Image Segmentation",
            "models": [
                ("SAM (Segment Anything Model)", "Zero-shot segmentation — works out of the box with prompts, no training needed"),
                ("Mask2Former", "State-of-the-art for both semantic and instance segmentation"),
                ("U-Net", "Classic encoder–decoder — excellent for medical imaging and when data is scarce"),
                ("DeepLab v3+", "Strong semantic segmentation baseline — simple to fine-tune"),
            ],
            "loss": "Pixel-wise cross-entropy + Dice loss",
            "primary_metric": "mIoU (mean intersection over union) · Dice coefficient",
            "secondary_metrics": ["Per-class IoU", "Pixel accuracy"],
            "watch_out": [
                "**Class imbalance at pixel level:** Background pixels dominate. Use Dice loss or weighted cross-entropy to address this.",
                "**SAM for zero-shot:** If you don't have a large annotated segmentation dataset, try SAM first — it may be good enough without any fine-tuning.",
                "**Boundary quality:** mIoU can look good even with fuzzy boundaries. Use boundary F1 if edge sharpness matters.",
            ],
        },
        "rec_vision_emb": {
            "title": "Image Embeddings / Visual Search",
            "models": [
                ("CLIP (ViT-L/14)", "Multimodal — enables text-to-image and image-to-image search"),
                ("Fine-tuned ResNet / EfficientNet + contrastive loss", "Train with triplet or InfoNCE loss on domain-specific pairs"),
                ("DINOv2", "Self-supervised ViT — strong visual features without label supervision"),
            ],
            "loss": "Contrastive (InfoNCE) · Triplet · Supervised contrastive",
            "primary_metric": "Recall@K · mAP (retrieval)",
            "secondary_metrics": ["R-precision", "NDCG@K"],
            "watch_out": [
                "**Hard negative mining:** Random negatives are too easy. Sample hard negatives (similar-looking but different class) to prevent embedding collapse.",
                "**Scalability:** Store embeddings in a vector database (FAISS, Weaviate, Pinecone) for scalable similarity search.",
            ],
        },
        "rec_clustering": {
            "title": "Clustering",
            "models": [
                ("K-Means", "Best for round, equal-size clusters when k is known — fast and scalable"),
                ("Gaussian Mixture Model (GMM)", "Like K-Means but soft assignments and handles elliptical clusters"),
                ("DBSCAN", "Finds arbitrary shapes, marks outliers as noise — no need to specify k"),
                ("HDBSCAN", "Hierarchical DBSCAN — more robust to varying densities, auto-selects k"),
                ("Agglomerative Clustering", "Produces a dendrogram — good when you want to explore multiple granularities"),
            ],
            "loss": "Inertia (K-Means) · BIC (GMM) · None (density-based)",
            "primary_metric": "Silhouette score · Davies-Bouldin index · Calinski-Harabasz index",
            "secondary_metrics": ["Elbow plot (inertia vs k)", "Visual inspection in 2D (UMAP/t-SNE projection)", "Cluster size distribution"],
            "watch_out": [
                "**K-Means assumes round, equal-size clusters:** If your data doesn't meet this assumption, K-Means will give misleading results.",
                "**Scale your features:** K-Means and GMM use Euclidean distance — a feature with range 0–10000 will dominate one with range 0–1. Always standardise first.",
                "**Silhouette score is not ground truth:** A high silhouette score means compact, well-separated clusters — not that the clusters are meaningful. Always validate with domain knowledge.",
                "**High dimensions:** Euclidean distance becomes unreliable above ~50 dimensions (curse of dimensionality). Run PCA or UMAP first, then cluster.",
                "**DBSCAN sensitivity:** Results are sensitive to eps and min_samples. Grid search or k-distance plot to choose eps.",
            ],
        },
        "rec_dimred": {
            "title": "Dimensionality Reduction",
            "models": [
                ("PCA", "Fast, linear, interpretable — explained variance ratio tells you how many components to keep"),
                ("UMAP", "Non-linear — preserves global structure, great for visualisation AND downstream ML"),
                ("t-SNE", "Non-linear — best 2D/3D visualisation, but don't use output for downstream tasks"),
                ("Autoencoder", "Non-linear, deep — good for very high-dimensional data (images, genomics)"),
                ("Kernel PCA", "Non-linear extension of PCA — useful when relationships are polynomial or RBF"),
            ],
            "loss": "Reconstruction error (PCA, Autoencoder) · KL divergence (t-SNE) · Cross-entropy (UMAP)",
            "primary_metric": "Explained variance ratio (PCA) · Trustworthiness score · Downstream task performance",
            "secondary_metrics": ["Reconstruction error", "Neighbour preservation"],
            "watch_out": [
                "**t-SNE is for visualisation only:** t-SNE embeddings are not stable across runs (random init) and distances between clusters are not meaningful. Never use t-SNE output as features.",
                "**UMAP is better than t-SNE for most purposes:** Faster, more stable, preserves global structure, and can be used as features.",
                "**PCA assumes linearity:** If your data has non-linear structure (images, manifolds), PCA loses signal. Use Kernel PCA or UMAP.",
                "**How many components?** For PCA: keep components that explain ≥ 90–95% of variance. For UMAP/t-SNE: typically 2D or 3D for visualisation, 10–50 for features.",
                "**Don't fit on test data:** PCA and UMAP must be fit on training data only, then transform both train and test. Fitting on the full dataset leaks test statistics.",
            ],
        },
        "rec_anomaly_tabular": {
            "title": "Tabular Anomaly Detection",
            "models": [
                ("Isolation Forest", "Best default — fast, scalable, works well on tabular data with many features"),
                ("Local Outlier Factor (LOF)", "Detects anomalies relative to local neighbourhood density — good for clusters of varying density"),
                ("One-Class SVM", "Strong on small, clean datasets where a tight boundary around normal data is possible"),
                ("Autoencoder", "Good for high-dimensional tabular data — anomalies have high reconstruction error"),
                ("ECOD (Empirical CDF-Based Outlier Detection)", "Simple, fast, interpretable — no hyperparameters"),
            ],
            "loss": "Anomaly score (path length, reconstruction error, distance)",
            "primary_metric": "AUC-ROC (if labels) · Precision@K · Average Precision",
            "secondary_metrics": ["False positive rate at operating threshold", "Contamination sensitivity"],
            "watch_out": [
                "**Contamination parameter:** Most unsupervised methods need an estimate of the anomaly rate (contamination). If you don't know it, try values between 0.01 and 0.1 and validate on any known anomalies.",
                "**Feature engineering matters:** Isolation Forest scores raw features — derived features (ratios, rolling stats, z-scores) often help it find anomalies.",
                "**No labels ≠ no validation:** Collect a small set of confirmed anomalies to validate your threshold, even if you can't train on them.",
                "**Ensemble of detectors:** Combine scores from multiple algorithms (average rank or mean normalised score) — almost always beats any single method.",
            ],
        },
        "rec_causal_rct": {
            "title": "Causal Inference from Randomised Experiments (RCT)",
            "models": [
                ("Two-sample t-test / Welch's t-test", "Estimate ATE as difference in means — valid under randomisation"),
                ("Regression adjustment (ANCOVA)", "Control for baseline covariates to reduce variance and improve precision"),
                ("CUPED (Controlled-experiment Using Pre-Experiment Data)", "Variance reduction using pre-experiment covariates — standard at Airbnb, Netflix"),
                ("Generalised linear model (GLM)", "Use for binary or count outcomes — logistic regression for conversion rates"),
            ],
            "loss": "N/A — this is estimation, not prediction",
            "primary_metric": "ATE (average treatment effect) with 95% confidence interval",
            "secondary_metrics": ["p-value", "Cohen's d (effect size)", "Statistical power", "CATE (heterogeneous effects by subgroup)"],
            "watch_out": [
                "**Run for at least one full business cycle (7 days):** Day-of-week effects, novelty effects, and ramp-up artefacts distort results from short experiments.",
                "**Multiple testing:** If you test many metrics, apply Bonferroni or Benjamini-Hochberg correction to control the false discovery rate.",
                "**SUTVA violation:** If treatment assignment of one unit affects others (social network effects, shared inventory), standard estimators are biased. Use cluster randomisation.",
                "**Peeking problem:** Looking at results before the experiment reaches the planned sample size inflates Type I error. Use sequential testing (e-values, mSPRT) if you need early stopping.",
                "**Practical vs statistical significance:** A statistically significant effect at 250M MAU can be tiny in absolute terms. Always report effect size (Cohen's d) alongside p-value.",
            ],
        },
        "rec_causal_obs": {
            "title": "Causal Inference from Observational Data",
            "models": [
                ("Propensity Score Matching (PSM)", "Match treated and control units with similar covariates — removes observed confounding"),
                ("Inverse Probability Weighting (IPW)", "Reweight observations so treated and control look similar without discarding data"),
                ("Doubly Robust Estimator (AIPW)", "Combines outcome model + propensity model — consistent if either is correct"),
                ("Difference-in-Differences (DiD)", "Compare before/after change across treated and control groups over time"),
                ("Regression Discontinuity (RDD)", "Exploit a threshold in an assignment rule as a natural experiment"),
            ],
            "loss": "N/A — causal estimation, not prediction",
            "primary_metric": "ATE with bootstrap confidence interval · ATT (average treatment effect on the treated)",
            "secondary_metrics": ["Standardised Mean Difference (SMD) for covariate balance", "Overlap (propensity score distribution)", "Sensitivity analysis (Rosenbaum bounds)"],
            "watch_out": [
                "**Unobserved confounders:** PSM and IPW only remove bias from observed confounders. If important confounders are unmeasured, estimates remain biased — and you can't tell from the data alone.",
                "**Overlap assumption:** If some subgroups only exist in treated or control (no overlap), causal identification fails for those groups. Check propensity score distributions.",
                "**Covariate balance check:** After matching or weighting, verify that SMD < 0.1 for all covariates. If not, the adjustment is insufficient.",
                "**DiD parallel trends:** Difference-in-differences assumes control and treatment would have trended the same without treatment. Test this with pre-treatment data.",
                "**Don't confuse correlation with causation:** A well-fitting predictive model does not imply causal identification. Causal validity requires structural assumptions about the data-generating process.",
            ],
        },
        "rec_causal_iv": {
            "title": "Instrumental Variables / Natural Experiments",
            "models": [
                ("2SLS (Two-Stage Least Squares)", "Classic IV estimator — instrument predicts treatment in stage 1, use fitted values in stage 2"),
                ("Fuzzy RDD", "When treatment probability jumps discontinuously at a threshold but isn't deterministic"),
                ("Wald estimator", "Simple IV for binary instrument and binary treatment"),
            ],
            "loss": "N/A",
            "primary_metric": "Local Average Treatment Effect (LATE) with 95% CI",
            "secondary_metrics": ["First-stage F-statistic (instrument strength)", "Exclusion restriction plausibility"],
            "watch_out": [
                "**Weak instruments:** If the instrument weakly predicts treatment (first-stage F < 10), IV estimates are biased and highly variable. Report F-statistic.",
                "**LATE not ATE:** IV estimates the effect only for compliers (units whose treatment status is actually changed by the instrument) — not the whole population.",
                "**Exclusion restriction is untestable:** You must argue from domain knowledge that the instrument affects the outcome only through the treatment, not directly.",
            ],
        },
        "rec_causal_discovery": {
            "title": "Causal Discovery",
            "models": [
                ("PC algorithm", "Constraint-based — uses conditional independence tests to infer the skeleton of the DAG"),
                ("FCI (Fast Causal Inference)", "Like PC but handles hidden confounders — outputs a PAG (partial ancestral graph)"),
                ("GES (Greedy Equivalence Search)", "Score-based — efficient search over DAG space"),
                ("LiNGAM", "Assumes linear non-Gaussian data — can identify a unique DAG, not just an equivalence class"),
                ("NOTEARS / DAG-GNN", "Continuous optimisation for DAG learning — scalable to many variables"),
            ],
            "loss": "BIC / likelihood-based scores · Conditional independence tests",
            "primary_metric": "SHD (structural Hamming distance vs ground truth if known) · F1 on edge recovery",
            "secondary_metrics": ["Precision and recall on edges", "Orientation accuracy"],
            "watch_out": [
                "**Causal discovery is hard:** These algorithms require very strong assumptions (Faithfulness, Causal Markov Condition) that are often only approximately true.",
                "**Output is a Markov equivalence class, not a unique DAG:** Most algorithms can't orient all edges — you get a CPDAG (completed partially directed acyclic graph).",
                "**Sample size requirements:** Reliable structure recovery typically needs thousands of samples per variable. With few samples, results are noisy.",
                "**Validate with domain knowledge:** Always cross-check the inferred graph with subject matter experts. Never trust a discovered DAG blindly.",
                "**Confounders break most algorithms:** PC and GES assume no hidden confounders. Use FCI or latent variable models if this assumption is doubtful.",
            ],
        },
        "rec_ab_planning": {
            "title": "A/B Test Sample Size Planning",
            "models": [
                ("Power analysis (frequentist)", "n = 2σ²(z_α + z_β)² / δ² for continuous outcomes"),
                ("Bayesian sequential design", "Set stopping rules based on posterior probability — allows early stopping with controlled error"),
                ("CUPED / variance reduction", "Reduce required n by controlling for pre-experiment covariates — often halves required sample size"),
            ],
            "loss": "N/A",
            "primary_metric": "Required n per arm · Expected experiment duration",
            "secondary_metrics": ["Power curve", "MDE (minimum detectable effect) at target n"],
            "watch_out": [
                "**Effect size assumption is critical:** The sample size formula is very sensitive to δ (minimum detectable effect). Be conservative — small effects need large samples.",
                "**Two-tailed vs one-tailed:** Use two-tailed tests unless you have a strong prior directional hypothesis. One-tailed halves the required n but doubles the risk of missing a negative effect.",
                "**Run for ≥ 1 week regardless of n:** Even if you hit your sample size in 2 days, novelty effects and day-of-week patterns require a full business cycle.",
                "**Plan guardrail metrics upfront:** Define which metrics must not decrease (e.g. session length, retention) before the experiment starts — not after seeing results.",
                "**Multiple metrics = multiple comparisons:** Correct for FWER or FDR if testing many metrics simultaneously.",
            ],
        },
        "rec_ab_analysis": {
            "title": "A/B Test Result Analysis",
            "models": [
                ("Welch's t-test", "Continuous outcome — does not assume equal variance between groups"),
                ("Mann-Whitney U", "Non-parametric alternative — use when normality is violated"),
                ("Chi-square / Fisher's exact", "Binary outcome (conversion, churn)"),
                ("Bootstrap confidence intervals", "Model-free — works for any metric, robust to distributional assumptions"),
                ("Bayesian A/B (Beta-Binomial / Normal-Normal)", "Provides posterior probability of treatment being better — more intuitive than p-values"),
            ],
            "loss": "N/A",
            "primary_metric": "p-value · 95% CI on the difference · Cohen's d",
            "secondary_metrics": ["Lift (%)", "Statistical power (post-hoc)", "Guardrail metric movements"],
            "watch_out": [
                "**p-value is not the probability that treatment is better:** It's the probability of seeing this result (or more extreme) if H₀ is true. A p-value of 0.04 does not mean 96% chance of a real effect.",
                "**Significant ≠ meaningful:** With 250M users, tiny effects become statistically significant. Always report effect size (Cohen's d) and decide if the effect is practically worth shipping.",
                "**Guardrail metrics:** Check that key health metrics (session length, D7 retention, crash rate) didn't worsen. A lift in revenue doesn't justify a collapse in engagement.",
                "**Simpson's Paradox:** A test that wins overall can lose in every subgroup if group sizes differ across variants. Segment your results.",
                "**Don't re-analyse after peeking:** If you checked results mid-experiment and now re-analyse at the end, your Type I error rate is inflated. Use sequential testing methods if you need interim looks.",
            ],
        },
        "rec_ab_survival": {
            "title": "Time-to-Event A/B Analysis",
            "models": [
                ("Kaplan-Meier estimator", "Non-parametric survival curves — visualise and compare survival distributions"),
                ("Log-rank test", "Test whether two survival curves are significantly different"),
                ("Cox proportional hazards model", "Estimate treatment effect while controlling for covariates (like regression for time-to-event)"),
            ],
            "loss": "Partial likelihood (Cox)",
            "primary_metric": "Hazard ratio (HR) with 95% CI · Log-rank p-value",
            "secondary_metrics": ["Median survival time", "Survival probability at fixed time points"],
            "watch_out": [
                "**Censoring:** Users who haven't yet churned/purchased by analysis date are right-censored. Standard t-tests ignore this — always use survival analysis for time-to-event outcomes.",
                "**Proportional hazards assumption:** Cox assumes the treatment effect is constant over time. Test with Schoenfeld residuals.",
                "**Competing risks:** If users can churn OR convert (mutually exclusive), use competing risks models instead of standard survival analysis.",
            ],
        },
        "rec_generative": {
            "title": "Generative Modelling",
            "models": [
                ("VAE (Variational Autoencoder)", "Learns a smooth latent space — good for interpolation and structured generation"),
                ("GAN (Generative Adversarial Network)", "Sharper samples than VAE — training is unstable, use StyleGAN2 or BigGAN as starting points"),
                ("Diffusion Models (DDPM / Stable Diffusion)", "State-of-the-art image generation — slow sampling but highest quality"),
                ("Normalising Flows", "Exact likelihood, invertible — good when you need probability density estimates"),
            ],
            "loss": "ELBO (VAE) · Adversarial (GAN) · Denoising score matching (Diffusion) · NLL (Flows)",
            "primary_metric": "FID (Fréchet Inception Distance) · Inception Score · Human evaluation",
            "secondary_metrics": ["Precision and recall (diversity vs fidelity tradeoff)", "LPIPS (perceptual similarity)"],
            "watch_out": [
                "**GAN training instability:** Mode collapse, training oscillation, and sensitivity to hyperparameters are common. Use WGAN-GP or StyleGAN architectures with gradient penalties.",
                "**FID can be gamed:** A model that memorises the training set has a perfect FID. Measure diversity and check for memorisation.",
                "**Diffusion is expensive:** Sampling from diffusion models requires hundreds of denoising steps. Use DDIM or consistency models for faster inference.",
                "**Tabular data generation:** GANs and VAEs underperform on tabular data. Use CTGAN or TVAE for synthetic tabular data generation.",
            ],
        },
    }

    # ── State init ─────────────────────────────────────────────────────────────
    if "advisor_path" not in st.session_state:
        st.session_state.advisor_path = []

    # ── Helper: get current node from path ────────────────────────────────────
    def _follow_path(path):
        node = "problem_type"
        for _, answer in path:
            next_map = TREE.get(node, {}).get("next", {})
            matched = None
            for key, target in next_map.items():
                if key in answer.lower():
                    matched = target
                    break
            if matched is None:
                break
            node = matched
        return node

    current_node = _follow_path(st.session_state.advisor_path)

    # ── Completed Q&A history ─────────────────────────────────────────────────
    for i, (node_id, answer) in enumerate(st.session_state.advisor_path):
        q_text = TREE.get(node_id, {}).get("q", "")
        st.markdown(
            f'<div class="datrix-card" style="opacity:0.65;padding:0.75rem 1.1rem;margin-bottom:0.5rem;">'
            f'<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem">Step {i + 1}</div>'
            f'<div style="color:#94a3b8;font-size:0.88rem;margin-bottom:0.3rem">{q_text}</div>'
            f'<div style="color:#22d3ee;font-weight:600;font-size:0.92rem">✓ &nbsp;{answer}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Current question or recommendation ────────────────────────────────────
    if current_node in TREE:
        q_data = TREE[current_node]
        step_num = len(st.session_state.advisor_path) + 1

        st.markdown(
            f'<div style="font-size:0.72rem;color:#7c3aed;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin:1.5rem 0 0.4rem;font-weight:600">Step {step_num}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem">'
            f'{q_data["q"]}</div>',
            unsafe_allow_html=True,
        )
        if q_data.get("help"):
            st.caption(q_data["help"])

        selected = st.radio(
            "Select an option",
            q_data["options"],
            key=f"advisor_radio_{current_node}",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        btn_col, back_col = st.columns([2, 1])
        with btn_col:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.advisor_path.append((current_node, selected))
                st.rerun()
        with back_col:
            if st.session_state.advisor_path and st.button(
                "← Back", use_container_width=True
            ):
                st.session_state.advisor_path.pop()
                st.rerun()

    elif current_node in RECS:
        # ── Recommendation panel ──────────────────────────────────────────────
        rec = RECS[current_node]

        st.markdown(
            f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.25rem;font-weight:800;'
            f'background:linear-gradient(135deg,#22d3ee,#a78bfa);-webkit-background-clip:text;'
            f'-webkit-text-fill-color:transparent;background-clip:text;margin:1.5rem 0 1rem">'
            f"✦ {rec['title']}</div>",
            unsafe_allow_html=True,
        )

        # Models
        st.markdown(
            '<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.6rem">Recommended approaches</div>',
            unsafe_allow_html=True,
        )
        for i, (name, reason) in enumerate(rec["models"]):
            badge = (
                '<span style="background:linear-gradient(135deg,#7c3aed,#0891b2);'
                'color:white;font-size:0.65rem;padding:1px 7px;border-radius:999px;'
                'font-weight:700;margin-right:8px">TOP PICK</span>'
                if i == 0
                else ""
            )
            st.markdown(
                f'<div class="datrix-card" style="margin-bottom:0.45rem;padding:0.75rem 1rem">'
                f"{badge}"
                f'<span style="color:#e2e8f0;font-weight:600">{name}</span>'
                f'<div style="color:#94a3b8;font-size:0.83rem;margin-top:0.2rem">{reason}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Loss + Metric
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="datrix-card"><div style="font-size:0.7rem;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">Loss function</div>'
                f'<div style="color:#22d3ee;font-weight:600;font-size:0.9rem">{rec["loss"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="datrix-card"><div style="font-size:0.7rem;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">Primary metric</div>'
                f'<div style="color:#22d3ee;font-weight:600;font-size:0.9rem">{rec["primary_metric"]}</div></div>',
                unsafe_allow_html=True,
            )

        # Secondary metrics
        st.markdown(
            '<div class="datrix-card" style="margin-top:0.5rem">'
            '<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.5rem">Secondary metrics</div>'
            + "".join(
                f'<span style="display:inline-block;background:rgba(124,58,237,0.15);'
                f'border:1px solid rgba(124,58,237,0.3);color:#c4b5fd;border-radius:6px;'
                f'padding:3px 12px;font-size:0.82rem;margin:2px 4px 2px 0">{m}</span>'
                for m in rec["secondary_metrics"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # Watch out for
        st.markdown(
            '<div style="font-size:0.72rem;color:#f59e0b;text-transform:uppercase;'
            'letter-spacing:0.08em;margin:1.2rem 0 0.5rem;font-weight:700">⚠ Watch out for</div>',
            unsafe_allow_html=True,
        )
        for note in rec["watch_out"]:
            st.markdown(
                f'<div class="datrix-card" style="margin-bottom:0.4rem;padding:0.7rem 1rem;'
                f'border-left:3px solid #f59e0b">'
                f'<span style="color:#e2e8f0;font-size:0.88rem;line-height:1.6">{note}</span></div>',
                unsafe_allow_html=True,
            )

        if st.session_state.advisor_path and st.button("← Back", use_container_width=False):
            st.session_state.advisor_path.pop()
            st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↺ Start over", use_container_width=True):
            st.session_state.advisor_path = []
            for k in list(st.session_state.keys()):
                if k.startswith("advisor_"):
                    del st.session_state[k]
            st.rerun()
    with c2:
        if st.button("← Back to home", use_container_width=True):
            go_to("upload")
