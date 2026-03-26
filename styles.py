"""Miss Datrix — custom CSS injection."""

import streamlit as st

THEME_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root variables ────────────────────────────────────────────────────────── */
:root {
    --bg-base:        #07071a;
    --bg-surface:     #0f0f2d;
    --bg-card:        rgba(255,255,255,0.04);
    --bg-card-hover:  rgba(255,255,255,0.07);
    --accent-purple:  #7c3aed;
    --accent-pink:    #db2777;
    --accent-blue:    #3b82f6;
    --gradient:       linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    --gradient-soft:  linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(219,39,119,0.15) 100%);
    --border:         rgba(124,58,237,0.25);
    --border-strong:  rgba(124,58,237,0.5);
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:     #64748b;
    --success:        #10b981;
    --warning:        #f59e0b;
    --danger:         #ef4444;
    --radius:         12px;
    --radius-sm:      8px;
    --shadow:         0 4px 24px rgba(0,0,0,0.4);
    --shadow-glow:    0 0 30px rgba(124,58,237,0.2);
}

/* ── Global reset ──────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Main content area ─────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }

/* ── Typography ────────────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
h1 { font-size: 2.2rem !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.4rem !important; margin-top: 2rem !important; }
p, li, span { color: var(--text-secondary) !important; }

/* ── Gradient headings ─────────────────────────────────────────────────────── */
.gradient-title {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.gradient-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Cards ─────────────────────────────────────────────────────────────────── */
.datrix-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.datrix-card:hover {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-glow);
}
.datrix-card-glow {
    background: var(--gradient-soft);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    padding: 1.5rem;
}

/* ── AI panel ──────────────────────────────────────────────────────────────── */
.ai-panel {
    background: linear-gradient(135deg, rgba(124,58,237,0.1) 0%, rgba(219,39,119,0.08) 100%);
    border: 1px solid rgba(124,58,237,0.4);
    border-left: 3px solid var(--accent-purple);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.ai-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent-purple) !important;
    margin-bottom: 0.5rem;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Inputs ────────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* ── File uploader ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-purple) !important;
}

/* ── DataFrames ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], iframe {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

/* ── Alerts ────────────────────────────────────────────────────────────────── */
.stSuccess { background: rgba(16,185,129,0.1) !important; border-color: var(--success) !important; border-radius: var(--radius-sm) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-color: var(--warning) !important; border-radius: var(--radius-sm) !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border-color: var(--danger)  !important; border-radius: var(--radius-sm) !important; }
.stInfo    { background: rgba(59,130,246,0.1)  !important; border-color: var(--accent-blue) !important; border-radius: var(--radius-sm) !important; }

/* ── Expander ──────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Progress bar ──────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: var(--gradient) !important;
}

/* ── Divider ───────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-soft) !important;
    color: var(--accent-purple) !important;
    border-bottom: 2px solid var(--accent-purple) !important;
}

/* ── Comparison table ──────────────────────────────────────────────────────── */
.compare-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
.compare-table th {
    background: var(--gradient);
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
}
.compare-table td {
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.88rem;
    vertical-align: top;
}
.compare-table tr:hover td { background: var(--bg-card); }
.pro  { color: #10b981 !important; font-weight: 600; }
.con  { color: #f87171 !important; font-weight: 600; }
.neutral { color: #94a3b8 !important; }
.check { color: #10b981 !important; font-size: 1.1rem; }
.cross { color: #f87171 !important; font-size: 1.1rem; }

/* ── Stage badge ───────────────────────────────────────────────────────────── */
.stage-badge {
    display: inline-block;
    background: var(--gradient-soft);
    border: 1px solid var(--border-strong);
    color: var(--accent-purple) !important;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 0.75rem;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

/* ── Sidebar nav items ─────────────────────────────────────────────────────── */
.nav-item-active {
    background: var(--gradient-soft);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 6px 12px;
    margin: 2px 0;
    font-weight: 600;
    color: var(--accent-purple) !important;
}
.nav-item-done {
    color: var(--text-muted) !important;
    padding: 4px 12px;
    font-size: 0.875rem;
    text-decoration: line-through;
}
.nav-item-pending {
    color: var(--text-muted) !important;
    padding: 4px 12px;
    font-size: 0.875rem;
}
</style>
"""

COMPARISON_HTML = """
<div class="datrix-card" style="margin: 1.5rem 0;">
<p class="gradient-subtitle" style="margin-bottom:0.4rem;">Miss Datrix vs Claude.ai</p>
<p style="color:var(--text-muted);font-size:0.82rem;margin-bottom:1rem;">
  Both are powered by Claude. The difference is what the application does with it.
</p>
<table class="compare-table">
<thead>
  <tr>
    <th>Capability</th>
    <th>Miss Datrix</th>
    <th>Claude.ai</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AI model</td>
    <td><span class="check">✓</span> Claude API (Sonnet)</td>
    <td><span class="check">✓</span> Claude (same model family)</td>
  </tr>
  <tr>
    <td>Automated ML pipeline</td>
    <td><span class="check">✓</span> End-to-end: clean → EDA → model → tune → report</td>
    <td><span class="cross">✗</span> General assistant — no pipeline</td>
  </tr>
  <tr>
    <td>Model training</td>
    <td><span class="check">✓</span> Trains 5 algorithms with real cross-validation scores</td>
    <td><span class="cross">✗</span> Cannot train models</td>
  </tr>
  <tr>
    <td>Hyperparameter tuning</td>
    <td><span class="check">✓</span> Optuna Bayesian search (50+ trials)</td>
    <td><span class="cross">✗</span> Cannot run Optuna or tune</td>
  </tr>
  <tr>
    <td>Feature importance</td>
    <td><span class="check">✓</span> SHAP values / tree importances on your trained model</td>
    <td><span class="cross">✗</span> Cannot compute on real model</td>
  </tr>
  <tr>
    <td>Downloadable artifacts</td>
    <td><span class="check">✓</span> Trained .pkl + standalone HTML report</td>
    <td><span class="cross">✗</span> No file exports</td>
  </tr>
  <tr>
    <td>Interactive charts</td>
    <td><span class="check">✓</span> Plotly — zoom, hover, linked to your data</td>
    <td><span class="cross">~</span> Static rendered images</td>
  </tr>
  <tr>
    <td>File handling</td>
    <td><span class="check">✓</span> CSV / Excel / Parquet up to 200 MB, processed in Python</td>
    <td><span class="check">~</span> Can read files, limited by context window</td>
  </tr>
  <tr>
    <td>Scope</td>
    <td><span class="con">~</span> ML / data science workflows only</td>
    <td><span class="pro">✓</span> Any topic or task</td>
  </tr>
  <tr>
    <td>Freeform questions</td>
    <td><span class="check">✓</span> Chat sidebar available at every stage</td>
    <td><span class="pro">✓</span> Native conversational interface</td>
  </tr>
</tbody>
</table>
<p style="color: var(--text-muted); font-size:0.8rem; margin-top:0.75rem;">
  ✓ available &nbsp;·&nbsp; ✗ not available &nbsp;·&nbsp; ~ partial or limited
</p>
</div>
"""


def inject_css():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(badge: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="stage-badge">{badge}</div>', unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<p style="color:var(--text-muted);margin-bottom:1.5rem">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def ai_panel_html(content: str):
    st.markdown(
        f"""
    <div class="ai-panel">
      <div class="ai-label">✦ Miss Datrix AI Analyst</div>
      <div style="color:var(--text-secondary);line-height:1.7">{content}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
