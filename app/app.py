"""CoolCanopy (Houston): dashboard + planner agent (Databricks App, Streamlit).

Sidebar: scope (citywide, a council district or a Super Neighborhood), units, map layer, and the
planting plan (a precomputed citywide strategy, or an optimized plan for just the focused area).
Header: the headline numbers for the current scope and plan.
Tabs: Plan map · Heat & canopy · Strategy comparison · Model · Data & method.
The planner agent lives in a floating panel (bottom right); its plans replace the Plan map's plan.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from heat import game
from heat.agent.constraints import PlanConstraints
from heat.agent.planner import Planner, PlannerData, h3_str
from heat.sim.optimize import baseline_order, in_area, optimize, simulate_sequence

ART = os.environ.get("HEAT_ARTIFACTS", "/Volumes/workspace/heat/artifacts/houston/app")
LLM = os.environ.get("LLM_ENDPOINT", "databricks-llama-4-maverick")
WAREHOUSE = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
SCHEMA = os.environ.get("HEAT_SCHEMA", "workspace.heat")
DEFAULT_COST = float(os.environ.get("DEFAULT_COST_PER_TREE", "1500"))
HOT_C = 3.0  # "hot cell": surface anomaly above +3 °C vs the city median
BENEFIT_C = 0.05  # a resident counts as "cooled" if their cell cools by >= this (optimize.BENEFIT_THRESHOLD_C)
R9_PER_R10 = 7.0  # an H3 res-9 cell covers ~7 res-10 cells; used to put both on one density scale in 3D
MAX_H = 3000.0    # metres: height of the densest cells in 3D (99th percentile of residents)


def col_height(res9: pd.Series, cap: float) -> pd.Series:
    """3D column height from residents per res-9-sized area. Square root, so mid-density areas still
    stand out next to the few densest apartment blocks; capped at the 99th percentile."""
    return np.sqrt(res9.clip(lower=0, upper=cap) / max(cap, 1.0)) * MAX_H

# ---- light palette (Vercel-like neutrals) -----------------------------------------------------
PAGE, SURF, SUBTLE, RAISED = "#ffffff", "#ffffff", "#fafafa", "#f5f5f5"
INK, INK2, MUTED = "#171717", "#525252", "#737373"
GRID, AXIS, BORDER = "#f0f0f0", "#d4d4d4", "#eaeaea"
STRAT = {  # fixed order = fixed colors; color follows the strategy everywhere
    "vulnerability": ("Vulnerable residents first", "#2f6fde"),
    "cooling": ("Most total cooling", "#d9531e"),   # max person-degrees, not a headcount
    "hottest_first": ("Hottest cells first", "#12875f"),
    "even": ("Even split by district", "#b7791f"),
    "random": ("Random placement", "#c2417a"),
}
OBJECTIVES = {"vulnerability": "Vulnerable residents first", "cooling": "Most total cooling"}
# map ramps for a light basemap: the neutral midpoint is near-white, extremes carry the color
HEAT_STOPS = [(-8, "#1d4ed8"), (-4, "#93c5fd"), (0, "#f5f5f4"), (4, "#f87171"), (8, "#991b1b")]
CANOPY_STOPS = [(0, "#f5f5f4"), (20, "#bbf7d0"), (40, "#22c55e"), (60, "#166534")]
CHANGE_STOPS = [(-15, "#ea580c"), (0, "#f5f5f4"), (15, "#16a34a")]
SVI_STOPS = [(0, "#f5f3ff"), (0.5, "#a78bfa"), (1, "#5b21b6")]
COOL_STOPS = [(0, "#bfdbfe"), (0.3, "#3b82f6"), (0.6, "#1e3a8a")]
HEAT_MUTED = [(-8, "#f4f4f5"), (0, "#e7e5e4"), (8, "#fca5a5")]   # plan maps: heat as a quiet base
BAND_COLORS = {"Low pavement (<30%)": "#fdba74", "Medium pavement (30–60%)": "#f97316",
               "High pavement (≥60%)": "#c2410c"}
HEAT_BAR_STOPS = [(-8, "#2f6fde"), (0, "#a3a3a3"), (4, "#ef4444"), (8, "#991b1b")]
GREEN, RED, VIOLET = "#16a34a", "#dc2626", "#7c3aed"

# ---- icons: inline Lucide SVGs (ISC license), stroke = currentColor ----------------------------
_ICON_PATHS = {
    "tree": '<path d="M8 19a4 4 0 0 1-2.24-7.32A3.5 3.5 0 0 1 9 6.03V6a3 3 0 1 1 6 0v.04a3.5 3.5 0 0 1 3.24 5.65A4 4 0 0 1 16 19Z"/><path d="M12 19v3"/>',
    "thermo": '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
    "spark": '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "bolt": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
}


def icon(name: str, size: int = 14, color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"{style}>'
            f'{_ICON_PATHS[name]}</svg>')


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _logo_uri(px: int = 96) -> str:
    """The CoolCanopy mark as a small inline PNG (data URI), so it needs no static file serving."""
    import base64
    from PIL import Image
    path = os.path.join(ASSETS, "logo_mark.png")
    if not os.path.exists(path):
        return ""
    buf = io.BytesIO()
    Image.open(path).resize((px, px), Image.LANCZOS).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


LOGO = _logo_uri()
_fav = os.path.join(ASSETS, "favicon.png")
st.set_page_config(page_title="CoolCanopy · Houston", page_icon=_fav if os.path.exists(_fav) else ":material/park:",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{
    font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  [data-testid="stIconMaterial"], [data-testid="stExpanderIcon"] {{
    font-family: "Material Symbols Rounded" !important; }}  /* keep icon glyphs (buttons, tabs, expanders) */
  .stApp {{ background: {PAGE}; color: {INK}; }}
  header[data-testid="stHeader"] {{ background: transparent; height: 2.4rem; }}
  [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{ display: none; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 6rem; max-width: 1480px; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.02em; color: {INK}; }}
  .ic {{ flex: none; vertical-align: -2px; }}
  code {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.82em; background: {RAISED};
          color: {INK}; border: 1px solid {BORDER}; border-radius: 5px; padding: 1px 5px; }}

  /* ---- sidebar ---- */
  section[data-testid="stSidebar"] {{ background: {SUBTLE}; border-right: 1px solid {BORDER}; }}
  [data-testid="stSidebarUserContent"] {{ padding-top: 16px; }}
  .brand {{ display: flex; align-items: center; gap: 10px; margin: 0 0 10px; }}
  .brand-mark {{ width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center;
                 background: {INK}; color: #fff; }}
  .brand-name {{ color: {INK}; font-weight: 700; font-size: 1rem; line-height: 1.15; letter-spacing: -0.01em; }}
  .bn-cool {{ color: #1f6fb2; }}
  .bn-can {{ color: #1f8a3b; }}
  .brand-logo {{ width: 40px; height: 40px; flex: none; object-fit: contain; }}
  .brand-logo.sm {{ width: 32px; height: 32px; }}
  .brand-sub {{ color: {MUTED}; font-size: 0.74rem; }}
  .brand-text {{ width: 124px; }}   /* fixed, so the collapse arrow can sit right after the name */
  /* Streamlit's sidebar header row (empty logo spacer + collapse arrow) takes 78px above the brand;
     fold it away and park the arrow on the brand row, just right of the name */
  [data-testid="stSidebarHeader"] {{ height: 0; padding: 0; margin: 0; min-height: 0; }}
  [data-testid="stSidebarCollapseButton"] {{ position: absolute; top: 18px; right: 14px; left: auto; z-index: 5;
    display: block !important; visibility: visible !important; }}
  [data-testid="stSidebarCollapseButton"] button {{ color: {MUTED}; }}
  [data-testid="stSidebarCollapseButton"] button:hover {{ color: {INK}; background: {RAISED}; }}
  .side-label {{ display: flex; align-items: center; gap: 7px; color: {MUTED}; font-size: 0.7rem;
                 font-weight: 600; text-transform: uppercase; letter-spacing: .07em; margin: 12px 0 6px;
                 padding-top: 12px; border-top: 1px solid {BORDER}; }}
  .budget {{ display: flex; justify-content: space-between; align-items: baseline; background: {SURF};
             border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 12px; margin-top: 4px;
             color: {MUTED}; font-size: 0.78rem; }}
  .budget b {{ color: {INK}; font-size: 0.95rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .side-note {{ color: {MUTED}; font-size: 0.74rem; line-height: 1.45; margin: 2px 0 6px; }}

  /* ---- page header ---- */
  .crumbs {{ color: {MUTED}; font-size: 0.8rem; display: flex; gap: 8px; align-items: center; }}
  .crumbs .sep {{ color: {AXIS}; }}
  .crumbs .here {{ color: {INK}; font-weight: 500; }}
  .page-title {{ color: {INK}; font-size: 1.75rem; font-weight: 650; letter-spacing: -0.03em;
                 line-height: 1.2; margin: 6px 0 16px; }}
  .page-sub {{ color: {INK2}; font-size: 0.92rem; margin: 0 0 12px; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 20px; }}
  .pill {{ display: inline-flex; align-items: center; gap: 6px; background: {SURF}; border: 1px solid {BORDER};
           border-radius: 999px; padding: 3px 10px; font-size: 0.75rem; color: {INK2}; }}
  .pill i {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}

  /* ---- metric strip ---- */
  .kpis {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); background: {SURF};
           border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; margin: 0 0 16px; }}
  .kpis.n4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
  .kpis.n6 {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
  .kpi {{ padding: 14px 16px 15px; border-left: 1px solid {BORDER}; min-width: 0; }}
  .kpi:first-child {{ border-left: none; }}
  .kpi-label {{ color: {MUTED}; font-size: 0.8rem; display: flex; align-items: flex-start; gap: 7px;
                line-height: 1.3; min-height: 2.1em; }}
  .kpi-label svg {{ flex: none; margin-top: 1px; }}
  .kpi-value {{ color: {INK}; font-size: 1.55rem; font-weight: 600; line-height: 1.2; margin-top: 6px;
                letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }}
  .kpi-sub {{ color: {MUTED}; font-size: 0.76rem; margin-top: 6px; line-height: 1.35; white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }}
  .delta {{ display: inline-block; border-radius: 5px; padding: 0 6px; margin-right: 6px; font-weight: 600;
            font-size: 0.72rem; line-height: 1.5rem; background: #eff6ff; color: #1d4ed8; }}
  .delta.warm {{ background: #fef2f2; color: #b91c1c; }}
  .delta.green {{ background: #f0fdf4; color: #15803d; }}
  .kgroups .kg-h {{ display: flex; align-items: center; gap: 6px; color: {MUTED}; font-size: 0.7rem;
                    font-weight: 600; text-transform: uppercase; letter-spacing: .07em; padding: 11px 16px 0;
                    border-left: 1px solid {BORDER}; background: {SURF}; line-height: 1.35; }}
  .kgroups .kg-h.s3 {{ grid-column: span 3; }}
  .kgroups .kg-h.kg-first {{ border-left: none; }}
  .kgroups .kpi {{ padding-top: 8px; }}
  .kgroups .kpi.kg-ft {{ border-left: none; }}
  .kwrap {{ container-type: inline-size; }}
  @container (max-width: 980px) {{
    .kpis.n6 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .kpis.n4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .kpis.n4 .kpi:nth-child(3) {{ border-left: none; }}
    .kpis.n4 .kpi:nth-child(n+3) {{ border-top: 1px solid {BORDER}; }}
    .kpis.n6:not(.kgroups) .kpi:nth-child(4) {{ border-left: none; }}
    .kpis.n6:not(.kgroups) .kpi:nth-child(n+4) {{ border-top: 1px solid {BORDER}; }}
    .kgroups .kg-h {{ grid-column: 1 / -1; }}
    .kgroups .kg-0 {{ order: 0; }} .kgroups .kpi.g0 {{ order: 1; }}
    .kgroups .kg-1 {{ order: 2; border-left: none; border-top: 1px solid {BORDER}; }}
    .kgroups .kpi.g1 {{ order: 3; }}
    .kgroups .kpi.kg-start {{ border-left: none; }}
  }}
  @container (max-width: 560px) {{
    .kpis, .kpis.n4, .kpis.n6 {{ grid-template-columns: minmax(0, 1fr); }}
    .kpi {{ border-left: none !important; border-top: 1px solid {BORDER}; }}
  }}

  /* ---- notice line ---- */
  .notice {{ display: flex; gap: 10px; align-items: flex-start; background: {SUBTLE}; border: 1px solid {BORDER};
             border-radius: 10px; padding: 11px 14px; color: {INK2}; font-size: 0.88rem; margin: 0 0 18px;
             line-height: 1.5; }}
  .notice .ic {{ margin-top: 3px; color: {MUTED}; }}
  .notice b {{ color: {INK}; font-weight: 600; }}
  .notice small {{ display: block; color: {MUTED}; font-size: 0.76rem; margin-top: 2px; }}
  .notice.blue {{ background: #f5f9ff; border-color: #cfe0fb; }}
  .notice.blue .ic {{ color: #2f6fde; }}

  /* ---- tabs: underline style ---- */
  div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
  div[data-testid="stTabs"] button[data-baseweb="tab"] {{ height: 40px; padding: 0 10px; background: transparent; }}
  div[data-testid="stTabs"] button[data-baseweb="tab"] p {{ font-size: 0.87rem; color: {MUTED}; font-weight: 500; }}
  div[data-testid="stTabs"] button[data-baseweb="tab"]:hover p {{ color: {INK}; }}
  div[data-testid="stTabs"] button[aria-selected="true"] p {{ color: {INK}; }}
  div[data-baseweb="tab-highlight"] {{ background-color: {INK} !important; height: 2px; }}
  div[data-baseweb="tab-border"] {{ display: none; }}
  div[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 18px; }}
  /* the view switcher (a segmented control) styled as underline tabs */
  .st-key-viewnav {{ border-bottom: 1px solid {BORDER}; margin-bottom: 10px; }}
  .st-key-viewnav [data-testid="stButtonGroup"] > div {{ gap: 4px; flex-wrap: wrap; }}
  .st-key-viewnav button {{ border: none !important; border-radius: 0 !important; background: transparent !important;
    height: 40px; padding: 0 10px; box-shadow: none !important; border-bottom: 2px solid transparent !important; }}
  .st-key-viewnav button p {{ font-size: 0.87rem; color: {MUTED}; font-weight: 500; }}
  .st-key-viewnav button:hover p {{ color: {INK}; }}
  .st-key-viewnav button[data-testid$="Active"] {{ border-bottom: 2px solid {INK} !important; }}
  .st-key-viewnav button[data-testid$="Active"] p {{ color: {INK}; }}

  /* ---- cards (bordered containers) ---- */
  div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div > div > div > .ph),
  div[data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"] > div > div > div > .ph) {{
       background: {SURF}; border: 1px solid {BORDER} !important; border-radius: 10px !important; }}
  .ph {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin: 2px 2px 4px; }}
  .ph-t {{ color: {INK}; font-size: 0.95rem; font-weight: 600; letter-spacing: -0.01em; }}
  .ph-s {{ color: {MUTED}; font-size: 0.78rem; margin-top: 2px; line-height: 1.4; }}
  .ph-r {{ color: {MUTED}; font-size: 0.74rem; white-space: nowrap; border: 1px solid {BORDER}; border-radius: 6px;
           padding: 1px 8px; }}

  /* ---- ranked list with inline bars (Vercel "top pages" style) ---- */
  .rank {{ display: flex; flex-direction: column; gap: 2px; margin-top: 4px; }}
  .rk-row {{ display: grid; grid-template-columns: 18px minmax(74px, 1.3fr) minmax(20px, 1fr) auto; align-items: center;
             gap: 10px; padding: 7px 8px; border-radius: 6px; }}
  .rk-row:hover {{ background: {SUBTLE}; }}
  .rk-row.on {{ background: {RAISED}; box-shadow: inset 0 0 0 1px {BORDER}; }}
  .rk-row.off {{ opacity: 0.45; }}
  .rk-n {{ color: {MUTED}; font-size: 0.74rem; font-variant-numeric: tabular-nums; }}
  .rk-name {{ color: {INK}; font-size: 0.85rem; font-weight: 500; line-height: 1.2; min-width: 0;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .rk-name small {{ display: block; color: {MUTED}; font-size: 0.72rem; font-weight: 400; margin-top: 2px;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .rk-bar {{ height: 6px; background: {RAISED}; border-radius: 3px; overflow: hidden; }}
  .rk-bar i {{ display: block; height: 100%; border-radius: 3px; }}
  .rk-val {{ color: {INK}; font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; font-weight: 500; }}

  .legend {{ display: flex; align-items: center; gap: 10px; color: {MUTED}; font-size: 0.76rem; margin: 10px 2px 2px; }}
  .legend .bar {{ flex: 1; max-width: 360px; height: 8px; border-radius: 4px; border: 1px solid {BORDER}; }}
  .note {{ color: {MUTED}; font-size: 0.76rem; line-height: 1.45; margin: 8px 2px 0; }}
  .chip {{ display: inline-flex; align-items: center; background: {SURF}; border: 1px solid {BORDER};
           border-radius: 6px; padding: 1px 8px; margin: 2px 4px 2px 0; font-size: 0.78rem; color: {INK2}; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; }}
  div[data-testid="stExpander"] details {{ border-color: {BORDER}; border-radius: 8px; background: {SURF}; }}
  .stButton button, .stDownloadButton button {{ border-radius: 7px; border-color: {BORDER}; box-shadow: 0 1px 1px rgba(0,0,0,.03); }}

  /* ---- floating planner chat (bottom-right button + panel) ---- */
  .st-key-chatfab {{ position: fixed; right: 24px; bottom: 22px; left: auto !important; z-index: 1001;
       width: max-content !important; }}
  .st-key-chatfab [data-testid="stElementContainer"], .st-key-chatfab [style*="width"] {{ width: auto !important; }}
  .st-key-chatfab button {{ border-radius: 999px; height: 44px; padding: 0 18px; background: {INK};
       border: 1px solid {INK}; color: #fff; font-weight: 500; box-shadow: 0 8px 24px rgba(0,0,0,.18); }}
  .st-key-chatfab button:hover, .st-key-chatfab button:focus, .st-key-chatfab button:active {{
       background: #383838; color: #fff; border-color: #383838; }}
  .st-key-chatfab button p {{ font-size: 0.88rem; }}
  .st-key-chatdock {{ position: fixed; right: 24px; bottom: 78px; left: auto !important; z-index: 1000;
       width: min(440px, calc(100vw - 32px)) !important; max-height: calc(100vh - 110px); overflow-y: auto;
       overflow-x: hidden; background: {SURF}; border: 1px solid {BORDER}; border-radius: 12px;
       padding: 14px 14px 12px; box-shadow: 0 18px 48px rgba(0,0,0,.14), 0 2px 6px rgba(0,0,0,.05); gap: 0.55rem; }}
  .st-key-chatdock > div {{ flex-shrink: 0; }}   /* scroll the panel instead of squashing its rows */
  .st-key-chatdock [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
  /* Streamlit pins element widths to the main column; size them to the panel instead */
  .st-key-chatdock [data-testid="stElementContainer"], .st-key-chatdock [style*="width"]:not(iframe):not(canvas) {{
       width: 100% !important; max-width: 100% !important; }}
  .st-key-chatdock .stTooltipHoverTarget, .st-key-chatdock .stButton button {{ width: 100% !important;
       max-width: 100% !important; }}
  .st-key-chatdock .stButton button {{ text-align: left; justify-content: flex-start; padding: 7px 10px;
       min-height: 0; white-space: normal; }}
  .st-key-chatdock .stButton button p {{ font-size: 0.8rem; color: {INK2}; }}
  .dock-h {{ display: flex; align-items: center; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid {BORDER}; }}
  .dock-h .brand-mark {{ width: 28px; height: 28px; }}
  .dock-t {{ color: {INK}; font-weight: 600; font-size: 0.92rem; line-height: 1.2; }}
  .dock-s {{ color: {MUTED}; font-size: 0.72rem; line-height: 1.35; }}
</style>""", unsafe_allow_html=True)


# ---- data + clients ---------------------------------------------------------------------------
@st.cache_resource
def ws():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


@st.cache_resource
def load(name: str) -> pd.DataFrame:
    path = f"{ART}/{name}.parquet"
    if os.path.exists(path):                       # local dev / mounted volume
        return pd.read_parquet(path)
    data = ws().files.download(path).contents.read()
    return pd.read_parquet(io.BytesIO(data))


@st.cache_resource
def planner_data() -> PlannerData:
    return PlannerData(load("sites"), load("effects"), load("hexes"), default_cost=DEFAULT_COST)


@st.cache_resource
def tracing_ready() -> bool:
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "/Shared/heat-trees-agent-traces"))
        mlflow.openai.autolog()
        return True
    except Exception:
        return False


def uc_call(name: str, args: dict) -> list[dict]:
    """Call a Unity Catalog table/scalar function through the SQL warehouse."""
    from databricks.sdk.service.sql import StatementParameterListItem as P
    params = [P(name=k, value=str(v)) for k, v in args.items()]
    placeholders = ", ".join(f":{k}" for k in args)
    stmt = (f"SELECT * FROM {SCHEMA}.{name}({placeholders})" if name != "nearest_cell"
            else f"SELECT {SCHEMA}.{name}({placeholders}) AS h3_cell")
    r = ws().statement_execution.execute_statement(warehouse_id=WAREHOUSE, statement=stmt,
                                                   parameters=params, wait_timeout="30s")
    if r.status.state.value != "SUCCEEDED":
        return [{"error": str(r.status.error)}]
    cols = [c.name for c in r.manifest.schema.columns]
    return [dict(zip(cols, row)) for row in (r.result.data_array or [])][:50]


def planner() -> Planner:
    if "planner" not in st.session_state:
        llm = ws().serving_endpoints.get_open_ai_client()
        st.session_state.planner = Planner(planner_data(), llm, LLM, uc_call=uc_call if WAREHOUSE else None)
    return st.session_state.planner


# ---- small helpers ----------------------------------------------------------------------------
def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp(values, stops, alpha: int = 200) -> list[list[int]]:
    xs = np.array([s[0] for s in stops], float)
    cols = np.array([hex2rgb(s[1]) for s in stops], float)
    v = np.clip(np.nan_to_num(np.asarray(values, float), nan=xs[0]), xs[0], xs[-1])
    rgb = np.stack([np.interp(v, xs, cols[:, k]) for k in range(3)], 1).astype(int)
    a = np.full((len(rgb), 1), alpha, int)
    return np.hstack([rgb, a]).tolist()


def legend(stops, fmt, title: str) -> None:
    grad = ", ".join(f"{c} {100 * (x - stops[0][0]) / (stops[-1][0] - stops[0][0]):.0f}%" for x, c in stops)
    st.markdown(f'<div class="legend"><span>{title}</span><span>{fmt(stops[0][0])}</span>'
                f'<div class="bar" style="background: linear-gradient(90deg, {grad});"></div>'
                f'<span>{fmt(stops[-1][0])}</span></div>', unsafe_allow_html=True)


def _tile(c: tuple, cls: str = "") -> str:
    lab, val, sub, ic = c[:4]
    pill = c[4] if len(c) > 4 and c[4] else ""
    return (f'<div class="kpi{cls}"><div class="kpi-label">{icon(ic) if ic else ""}{lab}</div>'
            f'<div class="kpi-value">{val}</div><div class="kpi-sub">{pill}{sub}</div></div>')


def kpi_row(cards: list[tuple], cls: str = "") -> None:
    """cards: (label, value, sub, icon_name[, delta_pill_html]). One panel, tiles split by hairlines."""
    st.markdown(f'<div class="kwrap"><div class="kpis {cls} n{len(cards)}">{"".join(_tile(c) for c in cards)}'
                f'</div></div>', unsafe_allow_html=True)


def kpi_groups(groups: list[tuple[str, str, list[tuple]]]) -> None:
    """Several labeled KPI groups (label, icon, cards) in one strip."""
    head, tiles = [], []
    for gi, (glab, gic, cards) in enumerate(groups):
        head.append(f'<div class="kg-h kg-{gi}{" kg-first" if gi == 0 else ""} s{len(cards)}">'
                    f'{icon(gic, 12)}{glab}</div>')
        for ci, c in enumerate(cards):
            extra = f" g{gi}" + (" kg-start" if ci == 0 and gi > 0 else (" kg-ft" if ci == 0 else ""))
            tiles.append(_tile(c, extra))
    n_ = sum(len(c) for *_, c in groups)
    st.markdown(f'<div class="kwrap"><div class="kpis kgroups n{n_}">{"".join(head)}{"".join(tiles)}</div></div>',
                unsafe_allow_html=True)


def notice(html: str, ic: str = "info", cls: str = "") -> None:
    st.markdown(f'<div class="notice {cls}">{icon(ic, 15)}<div>{html}</div></div>', unsafe_allow_html=True)


def panel(title: str, sub: str | None = None, right: str | None = None):
    """A titled card (Streamlit bordered container) with a one-line description."""
    box = st.container(border=True)
    box.markdown(f'<div class="ph"><div><div class="ph-t">{title}</div>'
                 + (f'<div class="ph-s">{sub}</div>' if sub else "") + "</div>"
                 + (f'<div class="ph-r">{right}</div>' if right else "") + "</div>", unsafe_allow_html=True)
    return box


def rank_list(labels, values, fmt, colors, subs=None, active=None, lo: float | None = None) -> None:
    """Ranked rows with an inline bar (largest first). active: label to highlight, others dimmed."""
    v = np.asarray(values, float)
    lo = min(0.0, float(np.nanmin(v))) if lo is None else lo
    hi = float(np.nanmax(v)) if len(v) else 1.0
    span = (hi - lo) or 1.0
    rows = []
    for i, k in enumerate(np.argsort(-v)):
        lab = labels[k]
        rc = "" if active is None else (" on" if lab == active else " off")
        sub = f"<small>{subs[k]}</small>" if subs is not None else ""
        w = max(2.0, 100 * (v[k] - lo) / span)
        rows.append(f'<div class="rk-row{rc}"><span class="rk-n">{i + 1}</span>'
                    f'<span class="rk-name">{lab}{sub}</span>'
                    f'<span class="rk-bar"><i style="width:{w:.0f}%;background:{colors[k]}"></i></span>'
                    f'<span class="rk-val">{fmt(v[k])}</span></div>')
    st.markdown(f'<div class="rank">{"".join(rows)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, h: int = 330, title: str | None = None, legend_top: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", paper_bgcolor=SURF, plot_bgcolor=SURF, height=h,
        font=dict(family='Inter, system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=12),
        title=dict(text=title or "", font=dict(color=INK, size=14), x=0.01, y=0.97),
        margin=dict(l=6, r=10, t=46 if title else 8, b=70 if legend_top else 6), barcornerradius=3,
        bargap=0.3,
        hoverlabel=dict(bgcolor=SURF, bordercolor=BORDER, font=dict(color=INK)),
        legend=dict(orientation="h", y=-0.22, yanchor="top", x=0.0, font=dict(color=INK2), title=None)
        if legend_top else None)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def money(v: float) -> str:
    return f"${v / 1e6:,.2f}M" if v >= 1e6 else f"${v / 1e3:,.0f}k"


def wkt_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if {"lat", "lon"} <= set(df.columns):
        return df
    return df.assign(lat=29.76, lon=-95.39)   # older export without cell centers: no zoom to focus


def optional(name: str) -> pd.DataFrame | None:
    try:
        return load(name)
    except Exception:
        return None


# ---- data prep (cached) -----------------------------------------------------------------------
@st.cache_data
def base_map() -> pd.DataFrame:
    m = wkt_latlon(load("map_r9").copy())
    m["district"] = m["district"].fillna("–")
    m["super_nbhd"] = m["super_nbhd"].fillna("–") if "super_nbhd" in m else "–"
    if "canopy_2011" in m:
        m["canopy_change"] = m["canopy_pct"] - m["canopy_2011"]
    return m


@st.cache_resource(show_spinner=False)
def tree_spots() -> tuple[np.ndarray, dict] | None:
    """Exact tree locations (off buildings and roads) as one lon/lat array sorted by cell and rank,
    plus h3 -> (start, count). cache_resource, not cache_data: no 1.2M-row copy on every rerun."""
    s = optional("spots")
    if s is None:
        return None
    s = s.sort_values(["h3", "spot_rank"])
    cells, start, count = np.unique(s["h3"].to_numpy(), return_index=True, return_counts=True)
    return s[["lon", "lat"]].to_numpy(float), dict(zip(cells, zip(start, count)))


def spots_for(picks_h3: pd.DataFrame) -> pd.DataFrame:
    """The first `trees` spots of every planted cell."""
    t = tree_spots()
    if t is None or not len(picks_h3):
        return pd.DataFrame(columns=["lon", "lat"])
    ll, idx = t
    parts = []
    for h, k in zip(picks_h3["h3"], picks_h3["trees"]):
        r = idx.get(h)
        if r is not None:
            parts.append(ll[r[0]:r[0] + min(int(k), int(r[1]))])
    a = np.concatenate(parts) if parts else np.empty((0, 2))
    return pd.DataFrame(a, columns=["lon", "lat"])


@st.cache_data
def hexes_s() -> pd.DataFrame:
    h = load("hexes").assign(h3=lambda d: d["hex"].map(h3_str))
    return h if "super_nbhd" in h else h.assign(super_nbhd=None)


@st.cache_data
def districts() -> pd.DataFrame:
    try:
        return load("districts").sort_values("district").reset_index(drop=True)
    except Exception:   # older export without the district rollup: build it from cells
        h = load("hexes").dropna(subset=["district"])
        g = h.groupby("district")
        return pd.DataFrame({
            "district": list(g.groups), "cells": g.size().values,
            "heat_pop_c": g.apply(lambda d: np.average(d.lst_anom_c, weights=d.residents + 1e-9)).values,
            "canopy_2021": g.canopy_pct_2021.mean().values, "residents": g.residents.sum().values,
            "residents_hot": g.apply(lambda d: d.residents[d.lst_anom_c > HOT_C].sum()).values})


plans, frames, all_picks, sites = load("plans"), load("frames"), load("picks"), load("sites")
bm, dist = base_map(), districts()
nb = optional("neighborhoods")
HAS_NB = nb is not None and len(nb) > 0 and bool((bm["super_nbhd"] != "–").any())
if HAS_NB:
    nb = nb.dropna(subset=["super_nbhd"]).reset_index(drop=True)
sizes = sorted(int(x) for x in plans["n_trees"].unique())          # precomputed plan sizes
SLIDER_SIZES = sorted(set(sizes) | {7500, 10000, 15000, 20000, 30000, 50000})   # larger ones solve live


def delta_stats(delta: pd.DataFrame, col: str = "district", val: str | None = None) -> dict:
    """Plan outcomes from per-cell deltas (h3, d_temp_c), optionally restricted to one area's cells.

    residents: people whose cell cools by >= BENEFIT_C; pdeg: person-degrees (no threshold);
    mean_c: resident-weighted mean cooling among those residents."""
    m = delta[["h3", "d_temp_c"]].merge(hexes_s()[["h3", "residents", "vulnerable", col]], on="h3", how="left")
    if val:
        m = m[m[col] == val]
    ben = m[m.d_temp_c <= -BENEFIT_C]
    r = float(ben.residents.sum())
    return {"residents": r, "vulnerable": float(ben.vulnerable.sum()),
            "pdeg": float((-m.d_temp_c * m.residents).sum()),
            "mean_c": float(np.average(-ben.d_temp_c, weights=ben.residents)) if r > 0 else 0.0}


@st.cache_data
def plan_stats(plan_id: str, col: str = "district", val: str | None = None) -> dict:
    return delta_stats(frames.loc[frames.plan_id == plan_id, ["h3", "d_temp_c"]], col, val)


@st.cache_data(show_spinner="Optimizing planting sites…", max_entries=48)
def live_plan(area: str | None, n_trees: int, strategy: str, cost: float) -> dict:
    """Solve a plan on the fly: citywide sizes beyond the precomputed ones, or only the planting sites
    inside one council district / Super Neighborhood (cooling can spill just past its boundary).
    Optimized strategies run the optimizer; the comparison strategies replay their planting order."""
    d = planner_data()
    s = d.sites if not area else d.sites[in_area(d.sites, area)]
    eff = d.effects[d.effects.site_id.isin(s.site_id)] if area else d.effects
    if strategy in OBJECTIVES:
        c = PlanConstraints(n_trees=int(n_trees), objective=strategy, cost_per_tree=float(cost))
        res = optimize(s, eff, d.hexes, c, d.default_cost, d.saturation_c)
    else:
        order = baseline_order(s, int(n_trees), strategy)
        res = simulate_sequence(s, eff, d.hexes, order, d.saturation_c, float(cost), [len(order)])
    cnt = res.picks.groupby("site_id").size()
    return {"delta": res.hex_delta.assign(h3=lambda x: x.hex.map(h3_str))[["h3", "d_temp_c"]],
            "picks": pd.DataFrame({"h3": [h3_str(i) for i in cnt.index], "trees": cnt.values}),
            "n_planned": int(len(res.picks)), "cost": float(res.summary.get("cost_usd", 0)),
            "capacity": int(s["capacity"].sum())}


# ---- sidebar ----------------------------------------------------------------------------------
def side_label(text: str, ic: str) -> None:
    st.markdown(f'<div class="side-label">{icon(ic, 13)}{text}</div>', unsafe_allow_html=True)


with st.sidebar:
    mark = (f'<img class="brand-logo" src="{LOGO}" alt="CoolCanopy logo"/>' if LOGO
            else f'<div class="brand-mark">{icon("tree", 17)}</div>')
    st.markdown(f'<div class="brand">{mark}<div class="brand-text">'
                '<div class="brand-name"><span class="bn-cool">Cool</span><span class="bn-can">Canopy</span></div><div class="brand-sub">Houston tree planner'
                '</div></div></div>', unsafe_allow_html=True)
    st.button("FAQ & definitions", icon=":material/help:", use_container_width=True, key="faq_btn",
              on_click=lambda: st.session_state.update(faq_open=True))
    side_label("Scope", "pin")
    fcol = "district"
    if HAS_NB:
        fcol = st.radio("Focus by", ["district", "super_nbhd"], horizontal=True, key="focus_by",
                        format_func={"district": "District", "super_nbhd": "Neighborhood"}.get)
    if fcol == "district":
        focus = st.selectbox("Council district", ["Citywide"] + [f"District {d}" for d in dist["district"]])
        fd = None if focus == "Citywide" else focus.split()[-1]
    else:
        focus = st.selectbox("Super Neighborhood", ["Citywide"] + sorted(nb["super_nbhd"]),
                             help="The city's 88 Super Neighborhoods: resident-led planning areas with "
                                  "their own councils and Neighborhood Plans.")
        fd = None if focus == "Citywide" else focus
    # district highlighted in the district charts: the focus district, or the neighborhood's main district
    dfd = fd if fcol == "district" else (nb.set_index("super_nbhd")["district"].get(fd) if fd else None)
    unit = st.radio("Units", ["°F", "°C"], horizontal=True)
    U = 1.8 if unit == "°F" else 1.0

    side_label("Map", "layers")
    layer = st.selectbox("Heat & canopy layer", ["Surface heat", "Tree canopy", "Canopy change",
                                                 "Social vulnerability", "Residents"])
    extrude = st.toggle("3D: height = residents", value=False,
                        help="Raises each cell by how many people live there (per area), on both maps.")

    side_label("Planting plan", "sliders")
    plan_scope = "city"
    if fd is not None:
        plan_scope = st.radio("Where to plant", ["city", "area"], horizontal=True, key="plan_scope",
                              format_func={"city": "Citywide", "area": "This area only"}.get,
                              help="This area only: every tree goes on a planting site inside the focused "
                                   "district or Super Neighborhood, optimized just for it.")
    if plan_scope == "area":
        area = fd if fcol == "district" else fd.upper()
        objective = st.selectbox("Optimize for", list(OBJECTIVES), format_func=OBJECTIVES.get,
                                 help="Vulnerable residents = residents weighted by the CDC Social Vulnerability Index "
                                 "(0–1, ranked within Texas). See FAQ & definitions at the top of the sidebar.")
        n = int(st.number_input("Trees in this area", min_value=10, max_value=200000, value=250, step=50,
                                key="area_n"))
        st.markdown(f'<div class="side-note">Planting only in <b>{focus}</b>.</div>', unsafe_allow_html=True)
        strat = objective
    else:
        strat = st.selectbox("Strategy", list(STRAT), format_func=lambda s: STRAT[s][0],
                             help="Vulnerable residents = residents weighted by the CDC Social Vulnerability Index "
                                 "(0–1, ranked within Texas). See FAQ & definitions at the top of the sidebar.")
        n = int(st.select_slider("Trees to plant", options=SLIDER_SIZES, value=1000 if 1000 in sizes else sizes[0],
                                 help=f"Up to {sizes[-1]:,} trees use precomputed plans; larger plans are "
                                      "solved live in a few seconds."))
    cost = st.number_input("Cost per tree ($, placeholder)", 100, 10000, int(DEFAULT_COST), step=100)
    st.markdown(f'<div class="budget"><span>Plan budget</span><b>{money(n * cost)}</b></div>',
                unsafe_allow_html=True)


def fmt_t(v: float, signed: bool = True) -> str:
    return f"{v * U:+.1f}{unit}" if signed else f"{v * U:.1f}{unit}"


# ---- FAQ & definitions (sidebar button) --------------------------------------------------------
@st.dialog("FAQ & definitions", width="large")
def faq() -> None:
    t_ben, t_hot, cost_ = f"{BENEFIT_C * U:.2f} {unit}", fmt_t(HOT_C), money(DEFAULT_COST)
    qa = [
        ("Who counts as a \"vulnerable resident\"?",
         "We use the CDC/ATSDR **Social Vulnerability Index (SVI) 2022**. For every census tract it ranks 16 "
         "Census measures against all Texas tracts, in four themes:\n"
         "- **Socioeconomic:** poverty, unemployment, housing cost burden, no health insurance, no high-school "
         "diploma\n"
         "- **Household:** age 65+, age 17 and under, disability, single-parent households, limited English\n"
         "- **Minority status:** racial and ethnic minority residents\n"
         "- **Housing and transportation:** apartments in large buildings, mobile homes, crowding, **no car**, "
         "group quarters\n\n"
         "The overall SVI is a percentile from 0 to 1: 0.9 means a tract is more vulnerable than 90% of Texas "
         "tracts. Each cell's **vulnerable residents = residents × SVI percentile**. So 1,000 people in a "
         "0.9 tract count as 900, and 1,000 people in a 0.2 tract count as 200. It is a weight, not a headcount "
         "of specific people. Tracts with no SVI score get 0.5."),
        ("Why does social vulnerability matter for heat?",
         "Heat hurts most where people are older or very young, have health conditions, can't afford air "
         "conditioning or power bills, or walk and wait for buses outdoors. SVI bundles those risk factors, "
         "so weighting by it sends shade to the people heat harms most."),
        ("What does \"residents cooled\" mean?",
         f"A resident counts as cooled when the plan lowers their cell's modeled surface temperature by at least "
         f"**{t_ben}**. That's a small amount on purpose: it marks who is reached at all. **Person-{unit}** adds "
         f"up every bit of cooling: 100 people cooled by 0.5{unit} = 50 person-{unit}."),
        ("What's the difference between the strategies?",
         "- **Vulnerable residents first:** maximizes cooling × vulnerable residents (the default).\n"
         f"- **Most total cooling:** maximizes total person-{unit} for everyone. It can stack trees where crowds "
         "already get cooled, so it doesn't always reach the most *people*.\n"
         "- **Hottest cells first:** plants the hottest open ground, even where few people live.\n"
         "- **Even split by district / Random placement:** the baselines to compare against."),
        ("Is this air temperature?",
         "No. It's **land surface temperature** from Landsat 8/9 at about 10:30 am, averaged over summers "
         f"2019–24, shown as the difference from the city median. A \"hot cell\" runs more than {t_hot} above "
         "the median. Hot pavement and roofs drive how hot it feels at street level, but air temperature changes "
         "less than surface temperature."),
        ("What is a cell?",
         "The city is split into hexagons from Uber's **H3** grid (resolution 10, about 15,000 m², or a "
         "city block). Heat, canopy, pavement and people are all added up per cell."),
        ("How many trees fit in a cell, and can a tree land on a building?",
         "No. Databricks spatial SQL subtracts every building footprint (plus a 2 m setback) and every road "
         "and rail line from each cell, using Overture Maps. Tree spots are placed about 16 m apart on what's "
         "left, and only half of them count as plantable, since some are private yards or utility strips. "
         "That caps each cell at 30 trees."),
        ("How is the cooling of one tree estimated?",
         "A LightGBM model learns surface temperature from canopy, pavement, water and surroundings. To score a "
         "tree we add a young crown to a cell and re-predict that cell and two rings of neighbors (about 400 m). "
         "Extra trees in an already-cooled spot count for less (diminishing returns)."),
        ("What does a tree cost here?",
         f"**{cost_} per tree is a placeholder** you can change in the sidebar. The game's budget round uses "
         "$1,000 on open ground and $2,500 where concrete has to be cut."),
        ("How does Beat the algorithm score me?",
         "Every tree you plant is scored with the same effect table and diminishing-returns rule as the "
         "optimizer. Your grade is your cooling points as a % of the optimizer's, minus 5 points for each paid "
         "map layer (residents, vulnerable residents) you opened."),
        ("What's the planner agent?",
         "A chat assistant (Llama 4 Maverick on Databricks) that turns a plain-English brief (budget, places to "
         "favor or avoid) into constraints, runs the optimizer through Unity Catalog tools, and puts the plan on "
         "the Plan map. It says when it lacks data, for example power lines."),
    ]
    for q, a in qa:
        with st.expander(q, expanded=q.startswith("Who counts")):
            st.markdown(a)


if st.session_state.pop("faq_open", False):
    faq()


# ---- the current plan: sidebar (precomputed citywide or area-only) or the planner agent's ------
H3_AREA = hexes_s().set_index("h3")
thr = f"≥{BENEFIT_C * U:.2f}{unit}"
where = "citywide" if fd is None else f"in {focus}"

if plan_scope == "area":
    ap = live_plan(area, n, objective, float(cost))
    side_plan = {"kind": "area", "label": f"{OBJECTIVES[objective]} · only {focus}", "color": STRAT[objective][1],
                 "delta": ap["delta"], "picks": ap["picks"], "n": ap["n_planned"], "requested": n,
                 "capacity": ap["capacity"]}
elif n not in sizes:   # citywide, bigger than the precomputed plans: solve live
    lp = live_plan(None, n, strat, float(cost))
    side_plan = {"kind": "live", "label": STRAT[strat][0], "color": STRAT[strat][1],
                 "delta": lp["delta"], "picks": lp["picks"], "n": lp["n_planned"], "requested": n}
else:
    top_p = all_picks[(all_picks.strategy == strat) & (all_picks["order"] <= n)]
    cnt = top_p.groupby("h3").size()
    side_plan = {"kind": "pre", "label": STRAT[strat][0], "color": STRAT[strat][1], "plan_id": f"{strat}_{n}",
                 "delta": frames.loc[frames.plan_id == f"{strat}_{n}", ["h3", "d_temp_c"]],
                 "picks": pd.DataFrame({"h3": cnt.index, "trees": cnt.values}), "n": n, "requested": n}


def stats_of(p: dict) -> dict:
    if p.get("plan_id"):
        return plan_stats(p["plan_id"], fcol, fd)
    return delta_stats(p["delta"], fcol, fd)


def trees_in_focus(picks: pd.DataFrame) -> int:
    return int(picks.loc[picks["h3"].map(H3_AREA[fcol]) == fd, "trees"].sum()) if fd else int(picks.trees.sum())


# ---- header + KPI strip -----------------------------------------------------------------------
st.markdown(
    f'<div class="crumbs"><span>Houston</span><span class="sep">/</span>'
    f'<span class="here">{"Citywide" if fd is None else focus}</span></div>'
    f'<div class="page-title">Summer surface heat &amp; trees</div>'
    , unsafe_allow_html=True)

hx = hexes_s()
scope = hx if fd is None else hx[hx[fcol] == fd]
res_hot = scope.loc[scope.lst_anom_c > HOT_C, "residents"].sum()
if fd is None:
    top = dist.sort_values("heat_pop_c", ascending=False).iloc[0]
    c1 = ("Hottest district", f"District {top.district}", "per resident vs median", "thermo",
          f'<span class="delta warm">{fmt_t(top.heat_pop_c)}</span>')
    can21 = bm["canopy_pct"].mean()
    can11 = bm["canopy_2011"].mean() if "canopy_2011" in bm else np.nan
else:
    units_ = dist if fcol == "district" else nb
    row = units_[units_[fcol] == fd].iloc[0]
    rank = int((units_.heat_pop_c > row.heat_pop_c).sum()) + 1
    kind = "districts" if fcol == "district" else "Super Neighborhoods"
    c1 = ("Surface heat per resident", fmt_t(row.heat_pop_c), f"rank of {len(units_)} {kind}", "thermo",
          f'<span class="delta warm">#{rank}</span>')
    can21, can11 = row.get("canopy_2021", np.nan), row.get("canopy_2011", np.nan)
can_sub = "since 2011" if pd.notna(can11) else "NLCD 2021"
can_pill = (f'<span class="delta {"green" if can21 >= can11 else "warm"}">{can21 - can11:+.1f} pts</span>'
            if pd.notna(can11) else "")

ps = stats_of(side_plan)
if side_plan["kind"] == "pre":
    # compare against random placement; when "random" itself is selected, compare to the vulnerability plan
    ref_id = f"random_{n}" if strat != "random" else f"vulnerability_{n}"
    ref = plan_stats(ref_id, fcol, fd)
    ratio = ps["residents"] / ref["residents"] if ref["residents"] else np.nan
elif side_plan["kind"] == "live":
    ref = delta_stats(live_plan(None, n, "random" if strat != "random" else "vulnerability", float(cost))["delta"],
                      fcol, fd)
    ratio = ps["residents"] / ref["residents"] if ref["residents"] else np.nan
else:
    ratio = np.nan
if side_plan["kind"] == "area":
    short = side_plan["n"] < side_plan["requested"]
    ratio_pill, ratio_txt = "", (f"only {side_plan['n']:,} fit in this area" if short else f"in {focus}")
elif not np.isfinite(ratio):
    ratio_pill, ratio_txt = "", "n/a"
elif strat != "random":
    ratio_pill, ratio_txt = f'<span class="delta">{ratio:.1f}×</span>', "vs random placement"
else:
    ratio_pill, ratio_txt = "", (f"targeted plan reaches {1 / ratio:.1f}× this" if ratio else "")
in_d = trees_in_focus(side_plan["picks"])
trees_sub = (f"{money(side_plan['n'] * cost)} citywide" if fd is None
             else f"{in_d:,} land in {focus}")

# the game is about one neighborhood: skip the citywide header there, so the board sits at the top
ON_GAME = (not st.session_state.get("force_live")
           and (st.session_state.get("view") or st.session_state.get("_last_view")) == "game")
if not ON_GAME:
    kpi_groups([
    (f"Heat context · {where}", "thermo", [
        c1,
        ("Tree canopy 2021", f"{can21:.1f}%", can_sub, "tree", can_pill),
        ("Residents in hot cells", f"{res_hot:,.0f}", f"live &gt;{fmt_t(HOT_C)}", "users",
         f'<span class="delta warm">{100 * res_hot / max(scope.residents.sum(), 1):.0f}%</span>'),
    ]),
    (f"This plan · {side_plan['label']} · results {where}", "sliders", [
        ("Trees in plan", f"{side_plan['n']:,}", trees_sub, "tree"),
        ("Residents cooled", f"{ps['residents']:,.0f}", ratio_txt, "users", ratio_pill),
        ("Avg cooling for them", f"{ps['mean_c'] * U:.2f}{unit}",
         f"{ps['pdeg'] * U:,.0f} person-{unit} total", "down"),
    ]),
])

if ON_GAME:
    pass
elif side_plan["kind"] == "area":
    notice(f'<b>{side_plan["n"]:,} trees</b> placed only on planting sites inside <b>{focus}</b>, chosen for '
           f'<b>{OBJECTIVES[objective].lower()}</b>. Results count residents {where}; a little cooling also '
           f'reaches cells just past the boundary.', "target")

# ---- game: "Beat the Algorithm" in one Super Neighborhood --------------------------------------
GAME_AREA = os.environ.get("GAME_AREA", "Gulfton")
INTEL = {"none": ("Streets only", 0), "heat": ("Surface heat", 0), "canopy": ("Tree canopy", 0),
         "residents": ("Residents", game.INTEL_COST["residents"]),
         "vulnerability": ("Vulnerable residents", game.INTEL_COST["vulnerability"])}
GRADE_COLOR = {"S": "#7c3aed", "A": "#16a34a", "B": "#2f6fde", "C": "#b7791f", "D": "#dc2626"}


@st.cache_resource(show_spinner=False)
def game_board() -> game.Board:
    d = planner_data()
    return game.Board(d.sites, d.effects, d.hexes, GAME_AREA.upper(), d.saturation_c)


@st.cache_resource(show_spinner=False)
def game_h3_index() -> dict:
    """h3 string -> site_id for the game board (map clicks come back as h3 strings)."""
    b = game_board()
    return dict(zip(b.sites.site_id.map(h3_str), b.sites.site_id))


@st.cache_resource(show_spinner=False)   # read-only results: no copy per call
def game_refs(key: str) -> dict:
    """The optimizer's plan and two naive baselines for one round, scored the same way as the player."""
    b, rnd = game_board(), next(r for r in game.ROUNDS if r.key == key)
    orders = {"algorithm": b.optimal_order(rnd), "random": b.baseline(rnd, "random"),
              "hottest": b.baseline(rnd, "hottest_first")}
    return {k: (o, b.evaluate(o, rnd)) for k, o in orders.items()}


def game_state() -> dict:
    if "game" not in st.session_state:
        st.session_state.game = {"round": 0, "orders": {r.key: [] for r in game.ROUNDS},
                                 "intel": {r.key: set() for r in game.ROUNDS}, "results": {},
                                 "sel": None, "saved": False}
    return st.session_state.game


def board_cells(b: game.Board, layer_key: str, counts: dict, sel) -> pd.DataFrame:
    c = b.sites.assign(h3=b.sites.site_id.map(h3_str))
    if layer_key == "heat":
        c["color"] = ramp(c.lst_anom_c, HEAT_STOPS, 190)
    elif layer_key == "canopy":
        c["color"] = ramp(c.canopy_pct_2021, CANOPY_STOPS, 190)
    elif layer_key == "residents":
        hi = float(c.residents.quantile(0.95)) or 1.0
        c["color"] = ramp(c.residents, [(0, "#f8fafc"), (hi / 2, "#93c5fd"), (hi, "#1e40af")], 200)
    elif layer_key == "vulnerability":
        hi = float(c.vulnerable.quantile(0.95)) or 1.0
        c["color"] = ramp(c.vulnerable, [(0, "#f5f3ff"), (hi / 2, "#a78bfa"), (hi, "#5b21b6")], 200)
    else:
        c["color"] = [[245, 245, 244, 150]] * len(c)
    c["trees"] = c.site_id.map(counts).fillna(0).astype(int)
    c["line"] = [[23, 23, 23, 255] if s == sel else ([22, 128, 61, 230] if t else [163, 163, 163, 160])
                 for s, t in zip(c.site_id, c.trees)]
    c["lw"] = [30 if s == sel else (12 if t else 1) for s, t in zip(c.site_id, c.trees)]   # meters
    show_people = layer_key in ("residents", "vulnerability")
    c["label"] = ("<b>" + c.trees.astype(str) + " / " + c.capacity.astype(str) + " trees</b><br/>"
                  + (c.lst_anom_c * U).map(lambda v: f"Surface {v:+.1f}{unit}") + " · canopy "
                  + c.canopy_pct_2021.map(lambda v: f"{v:.0f}%") + " · pavement "
                  + c.imperv_pct_2021.map(lambda v: f"{v:.0f}%")
                  + (("<br/>" + c.residents.map(lambda v: f"{v:,.0f} residents")) if show_people else ""))
    return c


def game_map(b: game.Board, cells: pd.DataFrame, counts: dict, delta: pd.DataFrame | None, key: str,
             selectable: bool, pop: dict | None = None):
    layers = []
    if delta is not None and len(delta):
        dd = delta.assign(h3=delta.hex.map(h3_str))[["h3", "d_temp_c"]]
        dd["color"] = ramp(-dd.d_temp_c, COOL_STOPS, 200)
        dd["label"] = (-dd.d_temp_c * U).map(lambda v: f"<b>{v:.2f}{unit} cooler</b>")
        layers.append(pdk.Layer("H3HexagonLayer", dd, get_hexagon="h3", get_fill_color="color", stroked=False,
                                extruded=False, pickable=False, opacity=0.8))
    cells = cells[["h3", "color", "line", "lw", "label", "trees"]]   # only what the map needs: no NaN, no int64 ids
    layers.append(pdk.Layer("H3HexagonLayer", cells, id="board", get_hexagon="h3", extruded=False,
                            get_fill_color="color" if delta is None else [0, 0, 0, 0], filled=True,
                            stroked=True, get_line_color="line", get_line_width="lw",
                            line_width_min_pixels=1, pickable=True))
    picks = pd.DataFrame({"h3": [h3_str(s) for s in counts], "trees": list(counts.values())})
    sp = spots_for(picks)[["lon", "lat"]]
    if len(sp):
        layers.append(pdk.Layer("ScatterplotLayer", sp, get_position="[lon, lat]", get_radius=5,
                                radius_min_pixels=2.5, get_fill_color=[22, 128, 61, 240],
                                stroked=True, get_line_color=[255, 255, 255, 220], line_width_min_pixels=0.5))
    if pop:   # the last planting's gain, as a label on that cell (more visible than a toast)
        lat_, lon_ = cell_latlon([h3_str(pop["site"])])[0]
        lab = pd.DataFrame({"lon": [lon_], "lat": [lat_], "text": [pop["text"]]})
        layers.append(pdk.Layer("ScatterplotLayer", lab, get_position="[lon, lat]", get_radius=60,
                                get_fill_color=[22, 163, 74, 70], stroked=True, get_line_color=[22, 128, 61, 255],
                                line_width_min_pixels=2))
        layers.append(pdk.Layer("TextLayer", lab, get_position="[lon, lat]", get_text="text", get_size=15,
                                get_color=[255, 255, 255, 255], get_pixel_offset=[0, -34], background=True,
                                get_background_color=[22, 128, 61, 240], background_padding=[8, 5, 8, 5],
                                font_weight=700, pickable=False,
                                # deck.gl's default glyph set is ASCII only; "auto" builds it from the text
                                # so ° and · render. pdk.types.String: a plain string would be read as an
                                # accessor expression (and so would a list).
                                character_set=pdk.types.String("auto")))
    ll = cell_latlon(cells.h3)
    view = pdk.ViewState(latitude=float(ll[:, 0].mean()), longitude=float(ll[:, 1].mean()), zoom=13.3)
    d = deck(layers, view)
    if selectable:
        return st.pydeck_chart(d, use_container_width=True, height=520, key=key, on_select="rerun",
                               selection_mode="single-object")
    st.pydeck_chart(d, use_container_width=True, height=360, key=key)
    return None


def game_points(v: float) -> str:
    return f"{v * U:,.0f}"


@st.fragment
def game_view() -> None:
    try:
        b = game_board()
    except Exception as e:   # no sites in the area (wrong name, old export)
        notice(f"The game needs planting sites in <b>{GAME_AREA}</b>: {e}", "info")
        return
    gs = game_state()
    rnd = game.ROUNDS[min(gs["round"], len(game.ROUNDS) - 1)]
    order, intel = gs["orders"][rnd.key], gs["intel"][rnd.key]
    finished = len(gs["results"]) == len(game.ROUNDS)

    pills = []
    for i, r in enumerate(game.ROUNDS):
        dot = GRADE_COLOR[gs["results"][r.key]["grade"]] if r.key in gs["results"] else "#d4d4d4"
        bold = ' style="font-weight:650;color:#171717"' if r.key == rnd.key and not finished else ""
        pills.append(f'<span class="pill"{bold}><i style="background:{dot}"></i>Round {i + 1} · {r.title}</span>')
    steps = "".join(pills)
    st.markdown(f'<div class="page-title">Beat the algorithm · {GAME_AREA}</div>'
                f'<div class="page-sub">You are the city forester. Place each tree where it cools the most '
                f'people, then see how you compare with the optimizer. Every tree is scored with the same heat '
                f'model and the same diminishing returns.</div><div class="pills">{steps}</div>',
                unsafe_allow_html=True)

    if finished:
        game_final(gs)
        return
    if rnd.key in gs["results"]:
        game_reveal(b, gs, rnd)
        return

    cur = b.evaluate(order, rnd)
    unit_pts = "vulnerable resident-°" if rnd.objective == "vulnerability" else "resident-°"
    left = (f"{rnd.n_trees - len(order)} of {rnd.n_trees}" if rnd.n_trees else
            money(rnd.budget_usd - cur.spent))
    notice(f"<b>Round {gs['round'] + 1}: {rnd.title}.</b> {rnd.brief} Click a cell on the map, then press "
           f"<b>Plant</b>. Green dots are real tree spots on open ground (off buildings and roads).", "target")
    kpi_row([
        ("Trees left" if rnd.n_trees else "Budget left", left, f"{len(order)} planted", "tree"),
        ("Cooling points", game_points(cur.score), f"{unit_pts}{unit[-1]} of cooling", "spark"),
        ("People cooled", f"{cur.residents:,}",
         f"{cur.mean_c * U:.2f}{unit} cooler on average" if cur.residents else f"cooled ≥{BENEFIT_C * U:.2f}{unit}",
         "users"),
        ("Biggest drop", f"{cur.peak_c * U:.2f}{unit}", f"surface cooling · {cur.cells_cooled} cell{'s' if cur.cells_cooled != 1 else ''} cooled", "down"),
    ])

    # Read the last map click before drawing anything, so the highlight and the plant panel update on the
    # same rerun (reading it after the map left the highlight one click behind).
    map_key = f"game_map_{rnd.key}"
    try:
        objs = (st.session_state[map_key]["selection"].get("objects") or {}).get("board") or []
    except Exception:
        objs = []
    if objs:   # match on the h3 string: 64-bit ids lose precision in JavaScript
        sid = game_h3_index().get(objs[0].get("h3"))
        if sid is not None:
            gs["sel"] = sid

    a, c2 = st.columns([2.2, 1], gap="medium")
    with c2:
        act = st.container(border=True)    # the selected cell and Plant buttons, at the top, next to the map
        with st.container(border=True):
            st.markdown('<div class="ph-t">Map intel</div>', unsafe_allow_html=True)
            opts = list(INTEL)
            lk = st.radio("Map layer", opts, key=f"game_layer_{rnd.key}", label_visibility="collapsed",
                          format_func=lambda k: INTEL[k][0] + (f"  (−{INTEL[k][1]} pts)" if INTEL[k][1] else ""))
            if INTEL[lk][1] and lk not in intel:
                intel.add(lk)
            if intel:
                note("Paid intel this round: " + ", ".join(INTEL[k][0] for k in sorted(intel))
                     + f" (−{sum(INTEL[k][1] for k in intel)} pts off your final %).")
    cells = board_cells(b, lk, cur.counts, gs["sel"])
    with a, panel(f"{GAME_AREA} planting cells", f"{len(cells)} cells with open ground · click one to select it"):
        game_map(b, cells, cur.counts, None, map_key, True, pop=gs.get("pop"))

    # Buttons change the plan in on_click callbacks, which run before the rerun: one rerun per click
    # instead of two (button → rerun → st.rerun), so the map redraws once.
    def plant(n_add: int) -> None:
        k0, before = len(order), b.evaluate(order, rnd).score
        for _ in range(n_add):
            if not b.can_plant(order, gs["sel"], rnd)[0]:
                break
            order.append(gs["sel"])
        after = b.evaluate(order, rnd)
        deg, pts, k = sum(after.per_tree_c[k0:]) * U, game_points(after.score - before), len(order) - k0
        gs["pop"] = {"site": gs["sel"], "text": f"+{deg:.2f}{unit} cooler · +{pts} pts"
                                                + (f" ({k} trees)" if k > 1 else "")}
        gs["toast"] = f"+{pts} cooling points · this block {deg:.2f}{unit} cooler"

    def undo() -> None:
        order.reverse(); order.remove(gs["sel"]); order.reverse()   # drop the last tree at this cell
        gs.pop("pop", None)

    def lock() -> None:
        me, best = b.evaluate(order, rnd), game_refs(rnd.key)["algorithm"][1]
        pct = game.final_pct(me, best, intel)
        gs["results"][rnd.key] = {"pct": pct, "raw_pct": game.pct_of(me, best), "grade": game.grade(pct),
                                  "trees": me.trees}

    if gs.get("toast"):
        st.toast(gs.pop("toast"), icon="🌳")
    with act:
        sel = gs["sel"] if gs["sel"] in b.pos else None
        if sel is None:
            st.markdown('<div class="ph-t">Pick a cell</div><div class="ph-s">Click any outlined cell on the map.'
                        '</div>', unsafe_allow_html=True)
        else:
            r = b.sites.iloc[b.pos[sel]]
            here = order.count(sel)
            people = (f"<br/>{r.residents:,.0f} residents · {r.vulnerable:,.0f} vulnerable"
                      if intel & {"residents", "vulnerability"} else "")
            st.markdown(f'<div class="ph-t">Cell {h3_str(sel)[-6:]}</div><div class="ph-s">'
                        f'Surface {r.lst_anom_c * U:+.1f}{unit} vs median · canopy {r.canopy_pct_2021:.0f}% · '
                        f'pavement {r.imperv_pct_2021:.0f}%{people}<br/><b>{here} / {int(r.capacity)}</b> trees · '
                        f'${b.cost_of(sel, rnd):,.0f} per tree</div>', unsafe_allow_html=True)
            ok, why_not = b.can_plant(order, sel, rnd)
            p1, p5 = st.columns(2)
            for btn, k, n_add, kind in ((p1, "game_plant", 1, "primary"), (p5, "game_plant5", 5, "secondary")):
                btn.button("Plant" if n_add == 1 else "Plant 5", icon=":material/park:", type=kind,
                           disabled=not ok, use_container_width=True, key=k, on_click=plant, args=(n_add,))
            st.button("Undo a tree here", icon=":material/undo:", disabled=here == 0, use_container_width=True,
                      key="game_remove", on_click=undo)
            if not ok:
                note(why_not)
        ready = b.done(order, rnd)
        if not ready and order:
            note("Use all your " + ("trees" if rnd.n_trees else "budget") + " to lock in.")
        st.button("Lock in my plan", icon=":material/flag:", disabled=not ready, use_container_width=True,
                  key="game_lock", on_click=lock)
        if order:
            st.button("Clear all trees", use_container_width=True, key="game_clear",
                      on_click=lambda: (order.clear(), gs.pop("pop", None)))


def game_reveal(b: game.Board, gs: dict, rnd: game.Round) -> None:
    order, intel, res = gs["orders"][rnd.key], gs["intel"][rnd.key], gs["results"][rnd.key]
    refs = game_refs(rnd.key)
    me = b.evaluate(order, rnd)
    best_order, best = refs["algorithm"]
    gcol = GRADE_COLOR[res["grade"]]
    beat = me.score >= best.score
    headline = "You beat the algorithm!" if beat else f"You reached {res['pct']:.0f}% of optimal"
    st.markdown(
        f'<div style="display:flex;gap:18px;align-items:center;margin:8px 0 14px">'
        f'<div style="font-size:56px;font-weight:700;color:{gcol};line-height:1">{res["grade"]}</div>'
        f'<div><div style="font-size:1.5rem;font-weight:650;color:{INK}">'
        f'{headline}'
        f'</div><div class="ph-s">Round {gs["round"] + 1} · {rnd.title} · raw {res["raw_pct"]:.0f}%'
        + (f', −{res["raw_pct"] - res["pct"]:.0f} pts for intel' if intel else "") + "</div></div></div>",
        unsafe_allow_html=True)

    rows = [("You", me), ("The algorithm", best), ("Hottest cells first", refs["hottest"][1]),
            ("Random placement", refs["random"][1])]
    kpi_row([(n, game_points(o.score),
              f"{o.residents:,} people · {o.mean_c * U:.2f}{unit} cooler on avg · peak {o.peak_c * U:.2f}{unit}", ic)
             for (n, o), ic in zip(rows, ["tree", "spark", "thermo", "target"])])

    m1, m2 = st.columns(2, gap="medium")
    cells = board_cells(b, "none", me.counts, None)
    with m1, panel("Your plan", "Blue = modeled surface cooling · green dots = your trees"):
        game_map(b, cells, me.counts, me.hex_delta, f"game_me_{rnd.key}", False)
    with m2, panel("The algorithm's plan", "Same trees or budget, placed by the optimizer"):
        game_map(b, board_cells(b, "none", best.counts, None), best.counts, best.hex_delta,
                 f"game_best_{rnd.key}", False)

    co = game.coaching(b, order, best_order, me)
    c1, c2 = st.columns(2, gap="medium")
    with c1, panel("Why the algorithm won" if not beat else "Where you and the algorithm differed",
                   "The optimizer's top cells that you skipped"):
        for f in co["missed"]:
            st.markdown(f"- **Cell {h3_str(f['site_id'])[-6:]}** ({f['trees']} trees): {game.why(f, U, unit)}")
        if not co["missed"]:
            note("You used every cell the algorithm used.")
    with c2, panel("Your weakest tree", "Smallest gain of any tree you planted"):
        w = co["worst"]
        if w:
            extra = (f" It was tree #{w['repeat'] + 1} in that cell, and each extra tree in one place cools less."
                     if w["repeat"] else "")
            st.markdown(f"**Cell {h3_str(w['site_id'])[-6:]}**, tree {w['tree_no']}: only "
                        f"+{game_points(w['gain'])} points ({w['gain_c'] * U:.2f}{unit} on its block). {game.why(w, U, unit)[:1].upper() + game.why(w, U, unit)[1:]}.{extra}")
        badges = game.achievements(rnd, me, best, res["pct"], intel)
        if badges:
            st.markdown("**Badges:** " + " · ".join(f"🏅 **{n}**: {d}" for n, d in badges))

    last = gs["round"] >= len(game.ROUNDS) - 1
    if st.button("See final results" if last else f"Next: Round {gs['round'] + 2} · {game.ROUNDS[gs['round'] + 1].title}",
                 type="primary", icon=":material/arrow_forward:", key=f"game_next_{rnd.key}"):
        gs["round"] += 1
        gs["sel"] = None
        gs.pop("pop", None)
        st.rerun(scope="fragment")


def leaderboard_rows() -> list[dict]:
    if not WAREHOUSE:
        return st.session_state.get("game_board_local", [])
    try:
        r = ws().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE, wait_timeout="30s",
            statement=f"SELECT player, total_pct, r1_pct, r2_pct, r3_pct FROM {SCHEMA}.game_scores "
                      f"WHERE area = '{GAME_AREA}' ORDER BY total_pct DESC LIMIT 10")
        cols = [c.name for c in r.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in (r.result.data_array or [])]
    except Exception:
        return st.session_state.get("game_board_local", [])


def save_score(player: str, pcts: list[float]) -> bool:
    total = float(np.mean(pcts))
    row = {"player": player, "total_pct": total, "r1_pct": pcts[0], "r2_pct": pcts[1], "r3_pct": pcts[2]}
    st.session_state.setdefault("game_board_local", []).append(row)
    if not WAREHOUSE:
        return False
    from databricks.sdk.service.sql import StatementParameterListItem as P
    try:
        r = ws().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE, wait_timeout="30s",
            statement=f"INSERT INTO {SCHEMA}.game_scores (player, area, total_pct, r1_pct, r2_pct, r3_pct, ts) "
                      "VALUES (:player, :area, :t, :a, :b, :c, current_timestamp())",
            parameters=[P(name="player", value=player[:24]), P(name="area", value=GAME_AREA),
                        P(name="t", value=f"{total:.2f}", type="DOUBLE"),
                        P(name="a", value=f"{pcts[0]:.2f}", type="DOUBLE"),
                        P(name="b", value=f"{pcts[1]:.2f}", type="DOUBLE"),
                        P(name="c", value=f"{pcts[2]:.2f}", type="DOUBLE")])
        return r.status.state.value == "SUCCEEDED"
    except Exception:
        return False


def game_final(gs: dict) -> None:
    pcts = [gs["results"][r.key]["pct"] for r in game.ROUNDS]
    total = float(np.mean(pcts))
    g = game.grade(total)
    kpi_row([(f"Round {i + 1} · {r.title}", f"{p:.0f}%", f"grade {gs['results'][r.key]['grade']}", ic)
             for i, (r, p, ic) in enumerate(zip(game.ROUNDS, pcts, ["users", "sliders", "heart"]))]
            + [("Overall", f"{total:.0f}%", f"grade {g}", "spark")])
    notice("What the rounds show: the hottest cell is rarely the best one. Trees pay off where many people "
           "live within a few hundred meters, stacking them in one place gives less each time, and a budget or "
           "an equity goal moves the best map. The optimizer does this for all of Houston's planting sites, "
           "and the planner agent lets a city manager drive it in plain English.", "spark")
    a, b2 = st.columns([1, 1.4], gap="medium")
    with a, st.container(border=True):
        st.markdown('<div class="ph-t">Save your score</div>', unsafe_allow_html=True)
        name = st.text_input("Your initials or team", max_chars=24, key="game_player")
        if st.button("Save to leaderboard", type="primary", disabled=gs["saved"] or not name.strip(),
                     key="game_save"):
            ok = save_score(name.strip(), pcts)
            gs["saved"] = True
            st.toast("Saved to the Delta leaderboard" if ok else "Saved for this session", icon="🏆")
            st.rerun(scope="fragment")
        if st.button("Play again", icon=":material/replay:", key="game_again"):
            del st.session_state["game"]
            st.rerun(scope="fragment")
    with b2, panel("Leaderboard", f"Average % of optimal across 3 rounds · {GAME_AREA}",
                   right=f"{SCHEMA}.game_scores"):
        rows = leaderboard_rows()
        if rows:
            lb = pd.DataFrame(rows)
            for c in ["total_pct", "r1_pct", "r2_pct", "r3_pct"]:
                lb[c] = pd.to_numeric(lb[c], errors="coerce")
            lb = lb.sort_values("total_pct", ascending=False).head(10)
            st.dataframe(lb.rename(columns={"player": "Player", "total_pct": "Overall %", "r1_pct": "R1 %",
                                            "r2_pct": "R2 %", "r3_pct": "R3 %"}),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.0f")
                                        for c in ["Overall %", "R1 %", "R2 %", "R3 %"]})
        else:
            note("No scores yet. Be the first.")


# The view switcher lives in session state, so a rerun (a sidebar change, a map click) keeps you on the
# page you were on; st.tabs resets to the first tab whenever the page above it changes. Only the active
# view is computed, which also makes every rerun faster.
VIEWS = {"plan": ":material/map: Plan map", "game": ":material/sports_esports: Beat the algorithm",
         "heat": ":material/thermostat: Heat & canopy", "strat": ":material/bar_chart: Strategy comparison",
         "model": ":material/model_training: Model", "data": ":material/description: Data & method"}
if st.session_state.get("force_live"):            # a new planner-agent plan: show it on the Plan map
    st.session_state.view = "plan"
if st.session_state.get("view") not in VIEWS:     # first run, or the active pill was clicked off
    st.session_state.view = st.session_state.get("_last_view", "plan")
with st.container(key="viewnav"):
    VIEW = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, key="view",
                                label_visibility="collapsed") or "plan"
st.session_state._last_view = VIEW


# ---- maps -------------------------------------------------------------------------------------
def view_state(pitch: float | None = None) -> pdk.ViewState:
    pitch = (50 if extrude else 0) if pitch is None else pitch
    bearing = -12 if pitch else 0
    if fd is None:
        return pdk.ViewState(latitude=29.76, longitude=-95.39, zoom=9.4 if not pitch else 9.8,
                             pitch=pitch, bearing=bearing)
    d = bm[bm[fcol] == fd]
    w = d["residents"] + 1
    return pdk.ViewState(latitude=float(np.average(d.lat, weights=w)), longitude=float(np.average(d.lon, weights=w)),
                         zoom=10.8 if fcol == "district" else 12.2, pitch=pitch, bearing=bearing)


def map_layer_df() -> tuple[pd.DataFrame, list, callable, str]:
    m = bm.copy()
    if layer == "Surface heat":
        stops, val, fmt, title = HEAT_STOPS, m.lst_anom_c, (lambda v: f"{v * U:+.0f}{unit}"), "Surface vs median"
    elif layer == "Tree canopy":
        stops, val, fmt, title = CANOPY_STOPS, m.canopy_pct, (lambda v: f"{v:.0f}%"), "Canopy"
    elif layer.startswith("Canopy change"):
        stops, val, fmt, title = CHANGE_STOPS, m.get("canopy_change", m.canopy_pct * 0), \
            (lambda v: f"{v:+.0f} pts"), "Canopy change"
    elif layer == "Social vulnerability":
        stops, val, fmt, title = SVI_STOPS, m.get("svi", m.canopy_pct * np.nan), (lambda v: f"{v:.1f}"), "CDC SVI"
    else:
        hi = float(m.residents.quantile(0.98)) or 1.0
        stops = [(0, "#f8fafc"), (hi / 2, "#93c5fd"), (hi, "#1e40af")]
        val, fmt, title = m.residents, (lambda v: f"{v:,.0f}"), "Residents"
    m["color"] = ramp(val, stops, 200)
    if fd is not None:
        dim = (m[fcol] != fd).to_numpy()
        m.loc[dim, "color"] = pd.Series([c[:3] + [45] for c in m.loc[dim, "color"]], index=m.index[dim])
    m["elev"] = col_height(m.residents, float(m.residents.quantile(0.99)))
    svi_txt = m.get("svi", pd.Series(np.nan, index=m.index)).map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    m["label"] = ("<b>District " + m.district.astype(str) + "</b> · " + m.super_nbhd.astype(str) +
                  "<br/>Surface " + (m.lst_anom_c * U).map(lambda v: f"{v:+.1f}") + unit + " vs median<br/>Canopy " +
                  m.canopy_pct.map(lambda v: f"{v:.0f}") + "% · pavement " +
                  m.imperv_pct.map(lambda v: f"{v:.0f}") + "%<br/>" +
                  m.residents.map(lambda v: f"{v:,.0f}") + " residents · SVI " + svi_txt)
    return m, stops, fmt, title


def deck(layers: list, view: pdk.ViewState | None = None) -> pdk.Deck:
    return pdk.Deck(layers=layers, initial_view_state=view or view_state(), map_provider="carto",
                    map_style=pdk.map_styles.CARTO_LIGHT,
                    tooltip={"html": "{label}", "style": {"backgroundColor": SURF, "color": INK, "fontSize": "12px",
                                                         "border": f"1px solid {BORDER}", "borderRadius": "6px",
                                                         "boxShadow": "0 4px 12px rgba(0,0,0,.08)"}})


def base_layer(m: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer("H3HexagonLayer", m, get_hexagon="h3", get_fill_color="color", pickable=True,
                     stroked=False, extruded=extrude, get_elevation="elev", elevation_scale=1,
                     opacity=1.0 if extrude else 0.9)


def cell_latlon(cells) -> np.ndarray:
    import h3
    f = getattr(h3, "cell_to_latlng", None) or getattr(h3, "h3_to_geo")
    return np.array([f(c) for c in cells], float).reshape(-1, 2)


def plan_view(picks_h3: pd.DataFrame) -> pdk.ViewState:
    """Fit the view to where the trees are (5th–95th percentile box, weighted by trees)."""
    v = view_state()
    if fd is not None or not len(picks_h3):
        return v
    try:
        ll = cell_latlon(picks_h3["h3"])
    except Exception:
        return v
    w = picks_h3["trees"].to_numpy(float)
    lat_lo, lat_hi = np.percentile(ll[:, 0], [5, 95])
    lon_lo, lon_hi = np.percentile(ll[:, 1], [5, 95])
    span = max((lon_hi - lon_lo), 2 * (lat_hi - lat_lo), 0.03)
    zoom = float(np.clip(np.log2(1400 / (1.3 * span)) - 1, 9.4, 12.5))
    return pdk.ViewState(latitude=float(np.average(ll[:, 0], weights=w)),
                         longitude=float(np.average(ll[:, 1], weights=w)), zoom=zoom, pitch=v.pitch, bearing=v.bearing)


def plan_layers(delta: pd.DataFrame, picks_h3: pd.DataFrame) -> list:
    """Quiet heat base + blue cooling + outlined planted cells. In 3D every cell is raised by its
    residents per area (res-9 base and res-10 plan cells on one density scale)."""
    m = bm.copy()
    # in 3D the walls need real color to read; flat, the heat stays a quiet background
    m["color"] = ramp(m.lst_anom_c, HEAT_STOPS, 235) if extrude else ramp(m.lst_anom_c, HEAT_MUTED, 170)
    if fd is not None:
        dim = (m[fcol] != fd).to_numpy()
        m.loc[dim, "color"] = pd.Series([c[:3] + [40] for c in m.loc[dim, "color"]], index=m.index[dim])
    cap9 = float(m.residents.quantile(0.99))
    m["elev"] = col_height(m.residents, cap9)
    m["label"] = ("District " + m.district.astype(str) + " · " + m.super_nbhd.astype(str) + " · surface " +
                  (m.lst_anom_c * U).map(lambda v: f"{v:+.1f}") + unit)
    # deck.gl's H3HexagonLayer defaults to 1 km tall columns: always pass `extruded` explicitly
    layers = [pdk.Layer("H3HexagonLayer", m, get_hexagon="h3", get_fill_color="color", pickable=not extrude,
                        stroked=False, extruded=extrude, get_elevation="elev", opacity=0.9 if extrude else 0.65)]
    res10 = hexes_s().set_index("h3")["residents"]

    def elev10(cells: pd.Series, lift: float) -> pd.Series:
        return col_height(cells.map(res10).fillna(0) * R9_PER_R10, cap9) * lift + 15

    if len(delta):
        dd = delta.copy()
        dd["color"] = ramp(-dd.d_temp_c, COOL_STOPS, 235)
        dd["elev"] = elev10(dd.h3, 1.02)
        dd["label"] = (-dd.d_temp_c * U).map(lambda v: f"<b>{v:.2f}{unit} surface cooling</b>")
        layers.append(pdk.Layer("H3HexagonLayer", dd, get_hexagon="h3", get_fill_color="color", pickable=True,
                                stroked=False, extruded=extrude, get_elevation="elev"))
    if len(picks_h3):
        if extrude:   # planted cells as dark columns standing just above their neighbors
            pk = picks_h3.assign(elev=elev10(picks_h3.h3, 1.2) + 300)
            layers.append(pdk.Layer("H3HexagonLayer", pk, get_hexagon="h3", get_fill_color=[23, 23, 23, 235],
                                    extruded=True, get_elevation="elev", pickable=True, stroked=False))
        else:
            layers.append(pdk.Layer("H3HexagonLayer", picks_h3, get_hexagon="h3", filled=False, stroked=True,
                                    extruded=False, get_line_color=[23, 23, 23, 220],
                                    line_width_min_pixels=1.5, pickable=True))
            try:   # a marker per planted cell, sized by trees, so a plan reads at city zoom
                ll = cell_latlon(picks_h3["h3"])
                pts = picks_h3.assign(lat=ll[:, 0], lon=ll[:, 1], r=90 + 80 * np.sqrt(picks_h3["trees"]))
                layers.append(pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_radius="r",
                                        radius_min_pixels=5, radius_max_pixels=40,
                                        get_fill_color=[47, 111, 222, 40], get_line_color=[23, 23, 23, 210],
                                        stroked=True, line_width_min_pixels=1.3, pickable=True))
            except Exception:
                pass
            sp = spots_for(picks_h3)   # one dot per tree, on open ground; visible once zoomed in
            if len(sp):
                layers.append(pdk.Layer("ScatterplotLayer", sp, get_position="[lon, lat]", get_radius=4,
                                        radius_min_pixels=0, radius_max_pixels=8,
                                        get_fill_color=[22, 128, 61, 230], pickable=False))
    return layers


def site_table(picks_h3: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    s = sites.assign(h3=sites["h3_cell"].astype(str).str.lower() if "h3_cell" in sites
                     else sites["hex"].map(h3_str))
    if "super_nbhd" not in s:
        s = s.assign(super_nbhd=None)
    t = picks_h3.merge(s[["h3", "district", "super_nbhd", "lst_anom_c", "canopy_pct_2021", "imperv_pct_2021",
                          "residents", "age65", "svi_rank"]], on="h3", how="left")
    t = t.merge(delta[["h3", "d_temp_c"]], on="h3", how="left")
    t["Surface heat"] = t.lst_anom_c * U
    t["Cooling here"] = -t.d_temp_c.fillna(0) * U
    t = t.sort_values(["trees", "lst_anom_c"], ascending=False)
    return t.rename(columns={"h3": "H3 cell", "district": "District", "super_nbhd": "Super Neighborhood",
                             "trees": "Trees", "canopy_pct_2021": "Canopy %", "imperv_pct_2021": "Pavement %",
                             "residents": "Residents", "age65": "Age 65+", "svi_rank": "SVI"})


# =============================== TAB: Plan map ==================================================
if VIEW == "plan":
    p_live = st.session_state.get("planner")
    live_id = p_live.state.current_plan if p_live else None
    src = "side"
    # touching any sidebar planting setting means "show me that plan": switch away from the agent's plan
    side_sig = (plan_scope, strat, n, float(cost), fcol, fd)
    side_changed = st.session_state.get("_side_sig", side_sig) != side_sig
    st.session_state._side_sig = side_sig
    if live_id:   # the toggle only appears once the agent has produced a plan
        if st.session_state.pop("force_live", False):   # a new agent plan: show it
            st.session_state.plan_src = "live"
        elif side_changed:
            st.session_state.plan_src = "side"
        src = st.radio("Plan source", ["side", "live"], horizontal=True, key="plan_src",
                       format_func={"side": "Sidebar plan", "live": "Planner agent plan"}.get)
    if src == "live" and live_id:
        res = p_live.state.results[live_id]
        cnt = res.picks.groupby("site_id").size()
        cur = {"label": "Planner agent plan", "color": "#2f6fde",
               "delta": res.hex_delta.assign(h3=lambda d: d.hex.map(h3_str))[["h3", "d_temp_c"]],
               "picks": pd.DataFrame({"h3": [h3_str(i) for i in cnt.index], "trees": cnt.values}),
               "n": int(res.summary.get("n_trees_planned", len(res.picks)))}
        chips = "".join(f'<span class="chip">{x}</span>' for x in p_live.state.constraints.chips()) \
            if p_live.state.constraints else ""
        notice(f'<b>Showing the planner agent\'s plan</b> <code>{live_id}</code>. Ask the planner (bottom right) '
               f'to change it. Changing any sidebar planting setting switches this map back to the sidebar plan.'
               f'<div style="margin-top:6px">{chips}</div>', "spark", "blue")
    else:
        cur = side_plan
        if not live_id:
            notice('Use the sidebar to pick a strategy or plant only in one area, or open <b>Ask the planner</b> '
                   '(bottom right) and describe a plan: budget, places to prioritize, things to avoid. '
                   'This map follows whichever plan is active.', "spark")
    delta, picks_h3 = cur["delta"], cur["picks"].copy()
    cs = delta_stats(delta, fcol, fd)
    picks_h3["label"] = picks_h3.trees.map(lambda t: f"<b>{t} tree{'s' if t > 1 else ''} planted here</b>")

    kpi_row([
        ("Trees planted", f"{cur['n']:,}", cur["label"], "tree"),
        (f"Residents cooled {thr}", f"{cs['residents']:,.0f}", where, "users"),
        ("Vulnerability-weighted", f"{cs['vulnerable']:,.0f}", "residents, weighted by CDC SVI", "heart"),
        ("Avg cooling for them", f"{cs['mean_c'] * U:.2f}{unit}", "modeled surface cooling", "down"),
    ])
    a, b = st.columns([2.2, 1], gap="medium")
    with a, panel("Modeled cooling from the plan",
                  "Blue = surface cooling (spreads about 400 m) · dark outline = cells that get trees · "
                  "warm base = surface heat" + (" · height = residents" if extrude else ""),
                  right="3D" if extrude else None):
        # Streamlit keeps a map's camera across reruns; a key per camera setting resets it when 3D or
        # the focus changes (otherwise toggling 3D never tilts the view and the columns stay invisible)
        st.pydeck_chart(deck(plan_layers(delta, picks_h3), plan_view(picks_h3)), use_container_width=True,
                        height=540, key=f"planmap_{extrude}_{fcol}_{fd}_{cur['label']}")
        legend(COOL_STOPS, lambda v: f"{v * U:.1f}{unit}", "Surface cooling")
    with b, panel("Where the trees go", "Trees per council district or Super Neighborhood"):
        wb = "district"
        if HAS_NB:
            wb = st.radio("Group by", ["district", "super_nbhd"], horizontal=True, key="where_by",
                          index=0 if fcol == "district" else 1, label_visibility="collapsed",
                          format_func={"district": "By district", "super_nbhd": "By neighborhood"}.get)
        area_of = picks_h3["h3"].map(H3_AREA[wb]).fillna("outside")
        bd = picks_h3.groupby(area_of)["trees"].sum().sort_values(ascending=False).head(12).astype(float)
        if wb == "district":
            labels = ["Outside city districts" if k == "outside" else f"District {k}" for k in bd.index]
            act = None if dfd is None else f"District {dfd}"
        else:
            labels = ["No Super Neighborhood" if k == "outside" else k for k in bd.index]
            act = fd if fcol == "super_nbhd" else None
        if len(bd):
            tot = max(float(picks_h3["trees"].sum()), 1)
            rank_list(labels, bd.values, lambda v: f"{v:,.0f}", ["#2f6fde"] * len(bd),
                      subs=[f"{100 * x / tot:.0f}% of plan" for x in bd.values], active=act)
        else:
            note("No trees in this plan.")
    tbl = site_table(picks_h3, delta)
    if fd is not None:
        tbl = tbl[tbl["District" if fcol == "district" else "Super Neighborhood"] == fd]
    hmax = float(max(tbl["Surface heat"].max(), 1)) if len(tbl) else 1.0
    hmin = float(min(tbl["Surface heat"].min(), 0)) if len(tbl) else 0.0
    with panel(f"Ranked planting cells{'' if fd is None else ' in ' + focus}",
               "Cells that receive trees, most trees first · top 200 shown", right=f"{len(tbl):,} cells"):
        st.dataframe(
            tbl[["H3 cell", "District", "Super Neighborhood", "Trees", "Surface heat", "Cooling here", "Canopy %",
                 "Pavement %", "Residents", "Age 65+", "SVI"]].head(200),
            hide_index=True, use_container_width=True, height=380,
            column_config={
                "Surface heat": st.column_config.ProgressColumn(f"Surface heat ({unit})", format="%+.1f",
                                                                min_value=hmin, max_value=hmax),
                "Cooling here": st.column_config.NumberColumn(f"Cooling here ({unit})", format="%.2f"),
                "Canopy %": st.column_config.ProgressColumn("Canopy %", format="%.0f", min_value=0, max_value=100),
                "Pavement %": st.column_config.NumberColumn(format="%.0f"),
                "Residents": st.column_config.NumberColumn(format="%.0f"),
                "Age 65+": st.column_config.NumberColumn(format="%.0f"),
                "SVI": st.column_config.NumberColumn(format="%.2f"),
            })
        st.download_button("Export plan as CSV", tbl.to_csv(index=False), "planting_plan.csv", mime="text/csv",
                           icon=":material/download:")


# =============================== TAB: Beat the algorithm =======================================
if VIEW == "game":
    # warm the caches outside the fragment: a cache spinner inside a fragment breaks Streamlit's delta paths
    with st.spinner("Setting up the game board…"):
        try:
            game_h3_index()
            for _r in game.ROUNDS:
                game_refs(_r.key)
        except Exception:
            pass
    game_view()


# =============================== TAB: Heat & canopy =============================================
if VIEW == "heat":
    m, stops, fmt, title = map_layer_df()
    a, b = st.columns([2.2, 1], gap="medium")
    with a, panel(f"{layer}", "Landsat 8/9 surface temperature at ~10:30 am, summers 2019–24 · not air "
                              "temperature" + (" · height = residents" if extrude else ""),
                  right=f"{'Citywide' if fd is None else focus}"):
        st.pydeck_chart(deck([base_layer(m)]), use_container_width=True, height=540,
                        key=f"heatmap_{extrude}_{fcol}_{fd}")
        legend(stops, fmt, title)
        note("Drawn on H3 res-9 cells (~0.1 km²) for speed; the model and plans use res-10 cells "
             "(~15,000 m²). Heat is an anomaly vs the city median.")
    with b, panel("Which districts run hottest?", f"Resident-weighted surface heat, {unit} vs city median"):
        d = dist.reset_index(drop=True)
        cols = [f"rgb({c[0]},{c[1]},{c[2]})" for c in ramp(d.heat_pop_c, HEAT_BAR_STOPS, 255)]
        can = d.get("canopy_2021", pd.Series(np.nan, index=d.index))
        rank_list([f"District {k}" for k in d.district], d.heat_pop_c * U, lambda v: f"{v:+.1f}°",
                  cols, subs=[f"{c:.0f}% canopy · {r / 1e3:,.0f}k people" for c, r in zip(can, d.residents)],
                  active=None if dfd is None else f"District {dfd}")

    if HAS_NB:
        big = nb[nb.residents >= 2000]
        n1, n2 = st.columns(2, gap="medium")
        with n1, panel("Hottest Super Neighborhoods", f"Resident-weighted surface heat, {unit} vs city median · "
                                                      "neighborhoods with 2,000+ residents",
                       right=f"top 12 of {len(nb)}"):
            d = big.sort_values("heat_pop_c", ascending=False).head(12)
            cols = [f"rgb({c[0]},{c[1]},{c[2]})" for c in ramp(d.heat_pop_c, HEAT_BAR_STOPS, 255)]
            rank_list(list(d.super_nbhd), d.heat_pop_c * U, lambda v: f"{v:+.1f}°", cols,
                      subs=[f"District {k} · {c:.0f}% canopy · {r / 1e3:,.0f}k people"
                            for k, c, r in zip(d.district, d.canopy_2021, d.residents)],
                      active=fd if fcol == "super_nbhd" else None)
        with n2, panel("Most people in hot cells", f"Residents living &gt;{fmt_t(HOT_C)} vs the city median",
                       right="top 12"):
            d = big.sort_values("residents_hot", ascending=False).head(12)
            rank_list(list(d.super_nbhd), d.residents_hot, lambda v: f"{v / 1e3:,.1f}k", [RED] * len(d),
                      subs=[f"District {k} · {100 * h / max(r, 1):.0f}% of {r / 1e3:,.0f}k residents"
                            for k, h, r in zip(d.district, d.residents_hot, d.residents)],
                      active=fd if fcol == "super_nbhd" else None)
        note("Super Neighborhoods are the city's 88 resident-led planning areas (Planning & Development Dept.); "
             "each has a council that drafts a Neighborhood Plan. A cell belongs to the neighborhood its "
             "center falls in.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, panel("More canopy, cooler surface", "At every pavement level, by % canopy in the cell"):
        cc = optional("canopy_curve")
        if cc is not None and len(cc):
            fig = go.Figure()
            for band, col in BAND_COLORS.items():
                s = cc[cc.imperv_band == band].sort_values("canopy_bin")
                fig.add_trace(go.Scatter(
                    x=s.canopy_bin + 2.5, y=s.anom_c * U, name=band, mode="lines+markers",
                    line=dict(color=col, width=2), marker=dict(size=7, line=dict(color=SURF, width=2)),
                    hovertext=[f"{band}<br>canopy {x:.0f}%: {y * U:+.1f}{unit} ({k:,} cells)"
                               for x, y, k in zip(s.canopy_bin + 2.5, s.anom_c, s.n)], hovertemplate="%{hovertext}<extra></extra>"))
            fig.update_xaxes(title="Tree canopy in cell (%)")
            fig.update_yaxes(title=f"Surface heat ({unit} vs median)")
            chart(style(fig, 330))
        else:
            st.info("Canopy curve not exported yet.")
    with c2, panel("Canopy 2011 → 2021", "Mean tree canopy by district; gray = 2011, green = 2021"):
        if "canopy_2011" in dist:
            d = dist.assign(chg=dist.canopy_2021 - dist.canopy_2011).sort_values("chg")
            fig = go.Figure()
            for _, r in d.iterrows():
                fig.add_trace(go.Scatter(x=[r.canopy_2011, r.canopy_2021], y=[f"District {r.district}"] * 2,
                                         mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(color=INK2 if dfd == r.district else AXIS,
                                                   width=3 if dfd == r.district else 2)))
            on = [dfd in (None, k) for k in d.district]
            fig.add_trace(go.Scatter(x=d.canopy_2011, y="District " + d.district, mode="markers", name="2011",
                                     marker=dict(color="#a3a3a3", size=9, line=dict(color=SURF, width=2),
                                                 opacity=[1 if o else 0.35 for o in on]),
                                     hovertext=[f"District {k}: {v:.1f}% in 2011" for k, v in zip(d.district, d.canopy_2011)],
                                     hovertemplate="%{hovertext}<extra></extra>"))
            fig.add_trace(go.Scatter(x=d.canopy_2021, y="District " + d.district, mode="markers", name="2021",
                                     marker=dict(color=GREEN, size=[13 if dfd == k else 10 for k in d.district],
                                                 line=dict(color=SURF, width=2), opacity=[1 if o else 0.35 for o in on]),
                                     hovertext=[f"District {k}: {v:.1f}% in 2021 ({c:+.1f} pts)"
                                                for k, v, c in zip(d.district, d.canopy_2021, d.chg)],
                                     hovertemplate="%{hovertext}<extra></extra>"))
            fig.update_xaxes(title="Mean tree canopy (%)")
            chart(style(fig, 330))
        else:
            st.info("Canopy trend needs the district export.")
    with c3, panel("Canopy vs heat", "One bubble per district, sized by residents"):
        d = dist
        size = 8 + 16 * np.sqrt(d.residents / d.residents.max())
        # letters sit beside the bubble (alternating sides) so clustered districts stay readable
        side = np.where(np.argsort(np.argsort(d.heat_pop_c.to_numpy())) % 2 == 0, "middle right", "middle left")
        fig = go.Figure(go.Scatter(
            x=d.get("canopy_2021", d.heat_pop_c * 0), y=d.heat_pop_c * U, mode="markers+text",
            text=d.district, textposition=side, textfont=dict(color=INK, size=12),
            marker=dict(size=size, color=["#2f6fde" if dfd in (None, k) else "#d4d4d4" for k in d.district],
                        opacity=0.85, line=dict(color=SURF, width=2)),
            hovertext=[f"District {k}<br>canopy {c:.1f}%<br>{h * U:+.1f}{unit} per resident<br>{r:,.0f} residents"
                       for k, c, h, r in zip(d.district, d.get("canopy_2021", d.heat_pop_c * 0), d.heat_pop_c,
                                             d.residents)],
            hovertemplate="%{hovertext}<extra></extra>"))
        fig.update_xaxes(title="Mean tree canopy 2021 (%)")
        fig.update_yaxes(title=f"Surface heat per resident ({unit})")
        chart(style(fig, 330, legend_top=False))


# =============================== TAB: Strategy comparison ======================================
if VIEW == "strat":
    nn = n if n in sizes else min(sizes, key=lambda z: abs(z - n))
    METRICS = {"vulnerable_benefiting": "Vulnerability-weighted residents cooled",
               "residents_benefiting": "Residents in cooled cells",
               "degree_vulnerable_c": "Vulnerable person-degrees of cooling",
               "hot_decile_hexes_reached": "Hottest-decile cells reached"}
    if nn != n:
        notice(f"Strategy comparisons use precomputed citywide plans; showing the nearest size, <b>{nn:,} trees</b>.")
    met = st.radio("Measure", list(METRICS), format_func=METRICS.get, horizontal=True)
    a, b = st.columns([1.6, 1], gap="medium")
    sel = strat if strat in STRAT else "vulnerability"
    with a, panel(f"{METRICS[met]} as the plan grows", "Citywide · dotted line = current plan size"):
        fig = go.Figure()
        for s, (label, col) in STRAT.items():
            d = plans[plans.strategy == s].sort_values("n_trees")
            y = d[met] * (U if met == "degree_vulnerable_c" else 1)
            fig.add_trace(go.Scatter(x=d.n_trees, y=y, name=label, mode="lines+markers",
                                     line=dict(color=col, width=3 if s == sel else 1.8),
                                     marker=dict(size=7 if s == sel else 5, line=dict(color=SURF, width=1.5)),
                                     hovertext=[f"{label}<br>{int(x):,} trees: {v:,.0f}" for x, v in zip(d.n_trees, y)],
                                     hovertemplate="%{hovertext}<extra></extra>"))
        fig.add_vline(x=nn, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(title="Trees planted")
        fig.update_yaxes(title=METRICS[met] + (f" ({unit})" if met == "degree_vulnerable_c" else ""))
        chart(style(fig, 400))
    with b, panel(f"Residents cooled at {nn:,} trees", f"{money(nn * cost)} budget · results {where} · × = vs random"):
        at = plans[plans.n_trees == nn].set_index("strategy").reindex(list(STRAT))
        sc = {s: plan_stats(f"{s}_{nn}", fcol, fd) for s in STRAT}
        rv = sc["random"]["residents"]
        vals = [sc[s]["residents"] for s in STRAT]
        fig = go.Figure(go.Bar(
            x=vals, y=[STRAT[s][0] for s in STRAT], orientation="h", marker_color=[STRAT[s][1] for s in STRAT],
            text=[f"{v / rv:.1f}×" if rv and s != "random" else "" for s, v in zip(STRAT, vals)],
            textposition="outside", textfont=dict(color=INK2), cliponaxis=False,
            hovertext=[f"{STRAT[s][0]}: {v:,.0f} residents" for s, v in zip(STRAT, vals)], hovertemplate="%{hovertext}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title=f"Residents cooled {thr}", range=[0, max(vals) * 1.25 if max(vals) else 1])
        chart(style(fig, 400, legend_top=False).update_layout(margin_r=28))
    show = pd.DataFrame({
        "Strategy": [STRAT[s][0] for s in STRAT],
        f"Residents cooled {thr}": [sc[s]["residents"] for s in STRAT],
        "Vulnerability-weighted": [sc[s]["vulnerable"] for s in STRAT],
        f"Person-{unit} of cooling": [sc[s]["pdeg"] * U for s in STRAT],
        f"Avg cooling per cooled resident ({unit})": [sc[s]["mean_c"] * U for s in STRAT],
        "Hot cells reached (citywide)": at["hot_decile_hexes_reached"].to_numpy(),
    })
    with panel(f"Results {where} at {nn:,} trees",
               "Same per-tree effect table and diminishing-returns rule for every strategy"):
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            f"Residents cooled {thr}": st.column_config.NumberColumn(format="%.0f"),
            "Vulnerability-weighted": st.column_config.NumberColumn(format="%.0f"),
            f"Person-{unit} of cooling": st.column_config.NumberColumn(format="%.0f"),
            f"Avg cooling per cooled resident ({unit})": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"Every strategy is scored with the same per-tree effect table and the same diminishing-returns "
               f"rule, so differences come only from where the trees go. A resident counts as cooled when their "
               f"cell's surface temperature drops by {thr}. Person-{unit} adds up all cooling, with no threshold. "
               f"Most total cooling maximizes person-{unit}, not the headcount: it can pile extra trees where "
               f"many people already get cooled, so another strategy that spreads trees thinner (often vulnerable "
               f"residents first) can push more people past the {thr} line.")


# =============================== TAB: Model ======================================================
if VIEW == "model":
    met_df = load("metrics").set_index("metric")["value"]
    g = met_df.get
    kpi_row([
        ("Spatial-CV R²", f"{g('spatial_cv_r2', np.nan):.3f}", "held-out H3 res-6 areas", "target"),
        ("Spatial-CV error", f"{g('spatial_cv_mae_c', np.nan) * U:.2f}{unit}", "mean absolute error", "target"),
        ("Random-CV R²", f"{g('random_cv_r2', np.nan):.3f}", "optimistic: spatial leakage", "info"),
        ("+10 pts canopy in cell", f"{g('ols_canopy_pct_2021_per10_c', np.nan) * U:+.2f}{unit}",
         f"± {g('ols_canopy_pct_2021_per10_se_c', np.nan) * U:.2f} (clustered SE, OLS)", "tree"),
        ("+10 pts canopy nearby", f"{g('ols_canopy_k2_per10_c', np.nan) * U:+.2f}{unit}",
         f"± {g('ols_canopy_k2_per10_se_c', np.nan) * U:.2f} within 2 rings", "tree"),
        ("+10 pts pavement in cell", f"{g('ols_imperv_pct_2021_per10_c', np.nan) * U:+.2f}{unit}",
         f"± {g('ols_imperv_pct_2021_per10_se_c', np.nan) * U:.2f}", "thermo"),
    ])
    c1, c2, c3 = st.columns(3, gap="medium")
    NAMES = {"imperv_k1": "Pavement, 1 ring", "imperv_pct_2021": "Pavement, cell", "canopy_pct_2021": "Canopy, cell",
             "canopy_k1": "Canopy, 1 ring", "canopy_k2": "Canopy, 2 rings", "imperv_k2": "Pavement, 2 rings",
             "lat": "Latitude", "lon": "Longitude", "water_frac": "Water fraction"}
    with c1, panel("What drives surface heat", "Mean absolute contribution per feature (SHAP-style)"):
        imp = optional("importance")
        if imp is not None:
            imp = imp.sort_values("mean_abs_contrib_c")
            fig = go.Figure(go.Bar(
                x=imp.mean_abs_contrib_c * U, y=imp.feature.map(lambda f: NAMES.get(f, f)), orientation="h",
                marker_color=[GREEN if "canopy" in f else RED if "imperv" in f else "#a3a3a3" for f in imp.feature],
                hovertext=[f"{NAMES.get(f, f)}: {v * U:.2f}{unit}" for f, v in zip(imp.feature, imp.mean_abs_contrib_c)],
                hovertemplate="%{hovertext}<extra></extra>"))
            fig.update_xaxes(title=f"Mean |contribution| ({unit})")
            chart(style(fig, 330, legend_top=False))
    with c2, panel("Canopy's effect on heat", "Model contribution of in-cell canopy, by canopy band"):
        dep = optional("dependence")
        if dep is not None:
            dep = dep.sort_values("canopy_band")
            fig = go.Figure(go.Scatter(
                x=dep.canopy_band + 5, y=dep.canopy_contrib_c * U, mode="lines+markers",
                line=dict(color=GREEN, width=2), marker=dict(size=7, line=dict(color=SURF, width=2)),
                hovertext=[f"canopy {b + 5:.0f}%: {v * U:+.1f}{unit} ({k:,} cells)"
                           for b, v, k in zip(dep.canopy_band, dep.canopy_contrib_c, dep.n)], hovertemplate="%{hovertext}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=AXIS, width=1))
            fig.update_xaxes(title="Canopy in cell (%)")
            fig.update_yaxes(title=f"Canopy's contribution ({unit})")
            chart(style(fig, 330, legend_top=False))
    with c3, panel("Held-out predictions vs observed", "5k cells from spatial-CV folds; dotted = perfect fit"):
        oof = optional("oof")
        if oof is not None:
            lim = [float(min(oof.lst_anom_c.min(), oof.pred_oof_c.min())) * U,
                   float(max(oof.lst_anom_c.max(), oof.pred_oof_c.max())) * U]
            fig = go.Figure(go.Scattergl(
                x=oof.pred_oof_c * U, y=oof.lst_anom_c * U, mode="markers",
                marker=dict(size=4, color="#2f6fde", opacity=0.3),
                hovertext=[f"predicted {a * U:+.1f}{unit}, observed {b * U:+.1f}{unit}"
                           for a, b in zip(oof.pred_oof_c, oof.lst_anom_c)], hovertemplate="%{hovertext}<extra></extra>"))
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                                     hoverinfo="skip"))
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title=f"Predicted, held-out area ({unit})")
            fig.update_yaxes(title=f"Observed ({unit})")
            chart(style(fig, 330, legend_top=False))
    st.caption("Monotone LightGBM (canopy can only cool, pavement only warm), cross-validated by H3 res-6 area "
               "so neighbors never leak between train and test. Per-tree effects re-score the model with an added "
               "50 m² crown; the 5 fold models give the uncertainty range shown in plan results.")


# =============================== TAB: Data & method =============================================
if VIEW == "data":
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.markdown("""
#### Sources
| Layer | Source |
|---|---|
| Surface temperature | Landsat 8/9 Collection 2 L2 (`lwir11`), summers 2019–24, via Microsoft Planetary Computer |
| Tree canopy | USFS/NLCD Tree Canopy Cover 2011, 2016, 2021 (MRLC) |
| Pavement | NLCD 2021 percent impervious |
| People | 2020 Census block groups (TIGERweb) and CDC/ATSDR SVI 2022 |
| Buildings & roads | Overture Maps building footprints and road/rail segments (2026-10-21), used to keep trees off roofs and pavement |
| Boundaries | City of Houston city limits, council districts as approved 10-12-2022 (in force since the 2023 election), and the 88 Super Neighborhoods (City of Houston ArcGIS REST) |

#### Pipeline on Databricks
Serverless job `heat-pipeline` (Asset Bundle). Bronze → silver → gold Delta tables in Unity Catalog
`workspace.heat`, joined on an **H3 res-10 grid** (~15,000 m² cells). The model is in MLflow and registered
as `workspace.heat.heat_model`. The planner agent calls Unity Catalog functions as tools, and Genie answers
ad-hoc questions over `genie_cells`.
""")
    with b, st.container(border=True):
        st.markdown("""
#### How the plan is chosen
1. For every candidate cell, the heat model predicts how much one young tree cools that cell and its
   neighbors (2 rings).
2. The optimizer adds one tree at a time wherever it cools the most **people** (optionally weighted by
   social vulnerability). Gains shrink as a place is already cooled (diminishing returns).
3. An area-only plan runs the same optimizer on just the planting sites inside the chosen district or
   Super Neighborhood.
4. Budgets, quotas, area preferences and exclusions from the planner agent become constraints.

#### Limitations
- *Surface*, not air, temperature, at ~10:30 am.
- Trees are only placed on open ground: Databricks spatial SQL subtracts every building (+2 m setback)
  and road from each cell, and each green dot is a ~16 m-spaced spot that is left. Parking lots and
  already-shaded blocks are skipped using NLCD. Half of the open spots count as plantable (private yards,
  utilities), and $/tree is still a placeholder.
- No power-line, bus-stop or sidewalk data yet. The agent says so rather than pretending.
- Population is about 10% overcounted at block-group edges; about 14% of cells (edge cells) have no
  council district.
""")


# =============================== Planner agent (floating panel) =================================
EXAMPLES = [
    "We've got $250k for about 600 trees this season. Stay away from power lines, prioritize District J, "
    "and favor bus stops.",
    "Plan 1,000 trees that help the most vulnerable residents, at least 30% of them in District F.",
    "Which hot places with lots of elderly residents have the least shade in District C?",
    "Plan 500 trees away from power lines, and prefer Gulfton and Sharpstown.",
]
EX_LABELS = ["$250k · ~600 trees · avoid power lines · prioritize District J",
             "1,000 trees for vulnerable residents · at least 30% in District F",
             "Hot, elderly, low-shade places in District C",
             "500 trees · avoid power lines · prefer Gulfton & Sharpstown"]
EX_ICONS = [":material/payments:", ":material/target:", ":material/search:", ":material/location_on:"]


def toggle_chat() -> None:
    st.session_state.chat_open = not st.session_state.get("chat_open", False)


def ask(p: Planner, prompt: str) -> None:
    hist = st.session_state.setdefault("history", [])
    tracing_ready()
    before = p.state.current_plan
    hist.append({"role": "user", "content": prompt})
    with st.spinner("Planning…"):
        out = p.chat([x for x in hist if x["role"] in ("user", "assistant")][-10:])
    reply = out["reply"]
    if out.get("ungrounded_numbers"):
        reply += f"\n\n_Not found in tool results: {', '.join(out['ungrounded_numbers'])}_"
    if p.state.current_plan and p.state.current_plan != before:
        st.session_state.force_live = True
        reply += f"\n\n_Plan `{p.state.current_plan}` is now on the **Plan map**._"
    hist.append({"role": "assistant", "content": reply})
    st.session_state.last_tool_log = out.get("tool_log", [])


def chat_dock_body(p: Planner) -> None:
    hist = st.session_state.setdefault("history", [])
    if not hist:
        st.markdown('<div class="dock-s">Try one of these, or type your own brief:</div>', unsafe_allow_html=True)
        for i, (lab, txt, ic) in enumerate(zip(EX_LABELS, EXAMPLES, EX_ICONS)):
            if st.button(lab, use_container_width=True, key=f"ex{i}", icon=ic):
                st.session_state.pending_prompt = txt
    c = p.state.constraints
    if c:
        st.markdown("".join(f'<span class="chip">{x}</span>' for x in c.chips()), unsafe_allow_html=True)
        with st.expander("Edit constraints", icon=":material/tune:"):
            n_e = st.number_input("Trees", 1, 20000, int(c.n_trees or c.resolved_n_trees(DEFAULT_COST) or 600))
            budget = st.number_input("Budget ($)", 0, 50_000_000, int(c.budget_usd or 0), step=10_000)
            cost_e = st.number_input("Cost per tree ($)", 50, 20000, int(c.cost_per_tree or DEFAULT_COST))
            obj = st.selectbox("Objective", ["vulnerability", "cooling", "balanced"],
                               index=["vulnerability", "cooling", "balanced"].index(c.objective))
            prefs = []
            for d in c.district_prefs:
                s_ = st.radio(d.label, ["soft", "hard"],
                              index=0 if d.strength != "hard" else 1, horizontal=True, key=f"d{d.district}")
                share = st.slider(f"{d.label} min share", 0.1, 0.9, d.min_share or 0.4,
                                  key=f"s{d.district}") if s_ == "hard" else None
                prefs.append({"district": d.district, "strength": s_, "min_share": share})
            if st.button("Re-run with these settings", type="primary"):
                c2 = PlanConstraints(n_trees=n_e, budget_usd=budget or None, cost_per_tree=cost_e,
                                     exclude=c.exclude, attractors=c.attractors,
                                     district_prefs=prefs, objective=obj)
                p.apply_chip_edit(c2)
                out = p._run_plan(c2)
                st.session_state.force_live = True
                hist.append({"role": "assistant",
                             "content": f"Re-ran with your edits → plan `{out.get('plan_id')}` is on the Plan map."})
                st.rerun()
    if hist:
        with st.container(height=360):
            for msg in hist[-14:]:
                with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user"
                                     else ":material/park:"):
                    st.markdown(msg["content"])
    prompt = st.chat_input("e.g. 600 trees, avoid power lines, prioritize Gulfton", key="dock_input")
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if prompt:
        ask(p, prompt)
        st.rerun()
    log = st.session_state.get("last_tool_log", [])
    if log:
        with st.expander(f"Show work: {len(log)} tool call(s) for the last message", icon=":material/terminal:"):
            for t in log:
                st.markdown(f"**`{t['tool']}`**  `{json.dumps(t['args'])[:160]}`")
                st.json(t["result"], expanded=False)


with st.container(key="chatfab"):
    is_open = st.session_state.get("chat_open")
    st.button("Close planner" if is_open else "Ask the planner", key="chatfab_btn", on_click=toggle_chat,
              icon=":material/close:" if is_open else ":material/forum:")
if st.session_state.get("chat_open"):
    with st.container(key="chatdock"):
        dmark = (f'<img class="brand-logo sm" src="{LOGO}" alt=""/>' if LOGO
                 else f'<div class="brand-mark">{icon("spark", 15)}</div>')
        st.markdown(f'<div class="dock-h">{dmark}<div>'
                    '<div class="dock-t">Planner agent</div><div class="dock-s">Llama 4 Maverick · Unity Catalog '
                    'tools · traced in MLflow. Plans appear on the Plan map.</div></div></div>',
                    unsafe_allow_html=True)
        try:
            planner_obj = planner()
        except Exception as e:   # no serving credentials: keep the rest of the dashboard alive
            planner_obj = None
            st.warning(f"The planner agent isn't available here ({type(e).__name__}). "
                       "The dashboard still works from the precomputed plans.")
        if planner_obj is not None:
            chat_dock_body(planner_obj)
