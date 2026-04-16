"""Global Streamlit styles — white-blue business theme."""
import math
import streamlit as st
import pandas as pd


# ── Plotly shared hover / layout ────────────────────────────

PLOTLY_HOVER = dict(
    bgcolor="rgba(15,23,42,0.94)",
    font_size=12.5,
    font_family="Inter, system-ui, sans-serif",
    font_color="#f8fafc",
    bordercolor="rgba(99,102,241,0.5)",
    align="left",
    namelength=-1,
)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    hoverlabel=PLOTLY_HOVER,
    font=dict(family="Inter, system-ui, sans-serif", color="#334155", size=12),
    # ВАЖНО: text="" обязателен — иначе Plotly иногда рендерит "undefined"
    # на месте заголовка, если страница не задала свой title.
    title=dict(
        text="",
        font=dict(size=14, color="#0f172a", family="Inter, sans-serif"),
        x=0.01, xanchor="left",
    ),
)

# Axis / legend / margin defaults — applied via plotly_defaults(fig)
_AXIS_STYLE = dict(
    gridcolor="rgba(226,232,240,0.55)",
    gridwidth=1,
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="rgba(203,213,225,0.7)",
    tickfont=dict(size=11, color="#64748b"),
    title_font=dict(size=12, color="#475569"),
)

_LEGEND_STYLE = dict(
    font=dict(size=11.5, color="#475569"),
    bgcolor="rgba(255,255,255,0.6)",
    bordercolor="rgba(226,232,240,0.6)",
    borderwidth=0,
    orientation="h",
    yanchor="bottom", y=1.02,
    xanchor="right", x=1,
)


def plotly_defaults(fig):
    """Apply polished axis/grid/legend defaults to any Plotly figure.

    Call AFTER update_layout so page-specific overrides win.
    """
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE, separatethousands=True)
    # Защитная мера от 'undefined' в заголовке: если страница не
    # задала title.text, ставим пустую строку (Plotly иногда подставляет
    # "undefined" от неинициализированных свойств).
    try:
        if not getattr(fig.layout.title, "text", None):
            fig.update_layout(title=dict(text=""))
    except Exception:
        pass
    # Merge our legend defaults on top of whatever the page set
    fig.update_layout(
        legend={**_LEGEND_STYLE, **fig.layout.legend.to_plotly_json()},
        margin=dict(l=10, r=10, t=40, b=10) if fig.layout.margin.l is None else {},
        # Премиальный modebar
        modebar=dict(
            bgcolor="rgba(255,255,255,0.7)",
            color="#94a3b8",
            activecolor="#2563eb",
        ),
    )
    return fig

# ── Plotly color palette ───────────────────────────────────
PLOTLY_COLORS = dict(
    blue="#3b82f6",
    blue_dark="#1e40af",
    blue_light="#93c5fd",
    green="#22c55e",
    green_dark="#16a34a",
    red="#ef4444",
    amber="#f59e0b",
    purple="#8b5cf6",
    indigo="#6366f1",
    slate="#94a3b8",
    teal="#14b8a6",
    rose="#f43f5e",
)


from datetime import date, timedelta

# ── Quick date presets ─────────────────────────────────────

_PRESETS = [
    ("7д", 7),
    ("14д", 14),
    ("30д", 30),
    ("90д", 90),
    ("Год", 365),
]

_PRESET_CSS = """
<style>
div[data-testid="stHorizontalBlock"] .quick-date-bar {display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.qd-btn {
    display:inline-block; padding:5px 14px; border-radius:999px;
    font-size:12px; font-weight:600; font-family:Inter,system-ui,sans-serif;
    cursor:pointer; border:1.5px solid #cbd5e1; color:#475569;
    background:#fff; transition:all .15s; text-decoration:none; line-height:1.4;
}
.qd-btn:hover {background:#f1f5f9;border-color:#94a3b8;color:#1e293b}
.qd-btn.active {
    background:linear-gradient(120deg,#2563eb,#3b82f6);
    color:#fff; border-color:transparent;
    box-shadow:0 4px 12px rgba(37,99,235,.25);
}
</style>
"""


def date_filter_bar(key_prefix: str = "df", default_days: int = 30):
    """Render quick-date preset pills + date inputs. Returns (d_from, d_to).

    Clicking a pill updates the calendar widgets to matching dates.
    ``key_prefix`` must be unique per page to avoid widget key collisions.

    The selected date range also persists across pages via the global
    keys ``_gbl_date_from`` / ``_gbl_date_to`` in session_state: when a page
    mounts for the first time, it picks up whatever the user selected
    elsewhere rather than the page's default.
    """
    st.markdown(_PRESET_CSS, unsafe_allow_html=True)
    today = date.today()

    sk = f"_qd_{key_prefix}"  # active-preset key
    k_from = f"{key_prefix}_from"
    k_to = f"{key_prefix}_to"
    g_from = "_gbl_date_from"
    g_to = "_gbl_date_to"

    # Initialise on first run: prefer global state if user touched dates elsewhere
    if sk not in st.session_state:
        if g_from in st.session_state and g_to in st.session_state:
            st.session_state[k_from] = st.session_state[g_from]
            st.session_state[k_to] = st.session_state[g_to]
            st.session_state[sk] = default_days
        else:
            st.session_state[sk] = default_days
            st.session_state[k_from] = today - timedelta(days=default_days)
            st.session_state[k_to] = today

    # ── Preset buttons ──────────────────────────────────────
    cols = st.columns([1] * len(_PRESETS) + [0.3, 1.5, 1.5])

    for i, (label, days) in enumerate(_PRESETS):
        with cols[i]:
            if st.button(label, key=f"{key_prefix}_qd_{days}", width="stretch"):
                st.session_state[sk] = days
                st.session_state[k_from] = today - timedelta(days=days)
                st.session_state[k_to] = today
                # Broadcast to global so other pages inherit it
                st.session_state[g_from] = st.session_state[k_from]
                st.session_state[g_to] = st.session_state[k_to]
                st.rerun()

    # Separator
    with cols[len(_PRESETS)]:
        st.markdown(
            "<div style='text-align:center;color:#94a3b8;padding-top:6px'>|</div>",
            unsafe_allow_html=True,
        )

    # ── Date inputs (values driven by session_state keys) ───
    with cols[len(_PRESETS) + 1]:
        d_from = st.date_input("от", key=k_from, label_visibility="collapsed")
    with cols[len(_PRESETS) + 2]:
        d_to = st.date_input("до", key=k_to, label_visibility="collapsed")

    # Keep global in sync with manual calendar changes
    st.session_state[g_from] = d_from
    st.session_state[g_to] = d_to

    return d_from, d_to


def inject_global_styles():
    st.markdown(
        """
        <style>
        /* ═══ Шрифт: Inter — современный, читабельный ═══ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stHeading {
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* ═══ Фон приложения: мягкий градиент с акцентами ═══ */
        .stApp {
            background:
                radial-gradient(ellipse 80% 60% at 10% 0%, rgba(37,99,235,0.10), transparent 50%),
                radial-gradient(ellipse 60% 40% at 90% 10%, rgba(99,102,241,0.08), transparent 50%),
                radial-gradient(ellipse 50% 50% at 50% 100%, rgba(14,165,233,0.05), transparent 60%),
                #f5f7fb !important;
            color: #0f172a;
        }

        /* ═══ Полноширинная компоновка ═══ */
        .block-container, [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
            padding-top: 2.2rem !important;
        }

        /* ═══ Заголовки: жирные, с градиентным акцентом ═══ */
        h1 {
            color: #0f172a !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
            font-size: 2rem !important;
            margin-bottom: 1.2rem !important;
        }
        h2, h3 {
            color: #0f172a !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        h2 { font-size: 1.45rem !important; }
        h3 { font-size: 1.15rem !important; margin-top: 1.6rem !important; }
        h3::after {
            content: "";
            display: block;
            width: 44px;
            height: 3px;
            border-radius: 999px;
            margin-top: 6px;
            background: linear-gradient(120deg, #2563eb, #6366f1, #06b6d4);
        }

        /* ═══ Tabs (вкладки) — стеклянные пилюли ═══ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: rgba(255,255,255,0.65);
            padding: 0.4rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.15);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 999px;
            padding: 0.4rem 1.3rem !important;
            color: #475569;
            font-weight: 600;
            transition: all 0.18s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #1e293b;
            background: rgba(241,245,249,0.7);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #2563eb, #4f46e5) !important;
            color: #ffffff !important;
            box-shadow: 0 6px 18px rgba(37,99,235,0.32);
        }

        /* ═══ Метрики (KPI карточки) — стеклянные, с подсветкой ═══ */
        [data-testid="stMetric"], [data-testid="metric-container"] {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            border: 1px solid rgba(226,232,240,0.9);
            box-shadow:
                0 1px 2px rgba(15,23,42,0.04),
                0 8px 24px rgba(15,23,42,0.06),
                inset 0 1px 0 rgba(255,255,255,0.9);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        [data-testid="stMetric"]::before, [data-testid="metric-container"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #2563eb, #6366f1, #06b6d4);
            opacity: 0.85;
        }
        [data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow:
                0 4px 8px rgba(15,23,42,0.05),
                0 16px 36px rgba(15,23,42,0.10),
                inset 0 1px 0 rgba(255,255,255,1);
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
            color: #64748b !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800 !important;
            font-size: 1.5rem !important;
            letter-spacing: -0.01em;
        }
        [data-testid="stMetricDelta"] svg { width: 14px; height: 14px; }

        /* ═══ Таблицы (нативные) ═══ */
        [data-testid="stTable"], .stDataFrame {
            background: rgba(255,255,255,0.95);
            border-radius: 16px;
            padding: 0.4rem;
            box-shadow: 0 8px 28px rgba(15,23,42,0.06);
            border: 1px solid rgba(226,232,240,0.7);
        }

        /* ═══ Кнопки — премиальные градиентные пилюли ═══ */
        .stButton>button, .stDownloadButton>button, .stForm button {
            background: linear-gradient(120deg, #1d4ed8, #2563eb 50%, #4f46e5);
            border: none;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 13px;
            box-shadow: 0 6px 16px rgba(37,99,235,0.28), inset 0 1px 0 rgba(255,255,255,0.2);
            transition: all 0.2s ease;
            letter-spacing: 0.01em;
        }
        .stButton>button:hover, .stDownloadButton>button:hover, .stForm button:hover {
            background: linear-gradient(120deg, #1e40af, #2563eb 50%, #4338ca);
            box-shadow: 0 10px 24px rgba(30,64,175,0.38), inset 0 1px 0 rgba(255,255,255,0.25);
            transform: translateY(-1px);
        }
        .stButton>button:active, .stDownloadButton>button:active, .stForm button:active {
            transform: translateY(0);
        }
        /* Secondary buttons (без формы) — мягче */
        button[kind="secondary"] {
            background: rgba(255,255,255,0.95) !important;
            color: #1e293b !important;
            border: 1.5px solid #cbd5e1 !important;
            box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
        }
        button[kind="secondary"]:hover {
            background: #f8fafc !important;
            border-color: #94a3b8 !important;
            color: #0f172a !important;
        }

        /* ═══ Формы и инпуты ═══ */
        .stForm {
            background: rgba(255,255,255,0.85);
            padding: 1.1rem 1.4rem;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            border: 1px solid rgba(226,232,240,0.8);
            backdrop-filter: blur(8px);
        }
        .stTextInput input, .stNumberInput input, .stDateInput input,
        .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {
            border-radius: 10px !important;
            border: 1.5px solid rgba(203,213,225,0.8) !important;
            transition: all 0.15s ease;
        }
        .stTextInput input:focus, .stNumberInput input:focus,
        .stTextArea textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        }

        /* ═══ Expanders ═══ */
        [data-testid="stExpander"] {
            background: rgba(255,255,255,0.7);
            border-radius: 14px;
            border: 1px solid rgba(226,232,240,0.8);
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #334155 !important;
        }
        [data-testid="stExpander"] summary:hover { color: #1e40af !important; }

        /* ═══ Алерты (info/warning/error/success) ═══ */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border-left-width: 4px !important;
            box-shadow: 0 2px 8px rgba(15,23,42,0.05);
        }

        /* ═══ Сайдбар: премиальный ═══ */
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.95), rgba(241,245,249,0.95)) !important;
            color: #0f172a !important;
            box-shadow: 4px 0 24px rgba(15,23,42,0.06);
            border-right: 1px solid rgba(226,232,240,0.6);
        }
        /* Скрыть streamlit-овский авто-нав */
        [data-testid="stSidebarNav"] { display: none !important; }

        /* ═══ Группированный сайдбар-нав ═══ */
        .sb-nav { padding: 0.5rem 0.2rem 0.3rem; }
        .sb-nav .sb-group-title {
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0.85rem 0.6rem 0.4rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid rgba(148,163,184,0.18);
        }
        .sb-nav .sb-group-title:first-child { margin-top: 0.1rem; }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            padding: 0.4rem 0.75rem !important;
            margin: 1px 0 !important;
            border-radius: 10px !important;
            font-size: 13px !important;
            transition: all 0.15s ease;
            position: relative;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(99,102,241,0.05)) !important;
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span {
            color: #1e293b !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] [data-testid="stIconEmoji"] {
            font-size: 16px !important;
        }
        /* Текущая страница (по aria-current) */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
            background: linear-gradient(90deg, rgba(37,99,235,0.14), rgba(99,102,241,0.07)) !important;
            box-shadow: inset 3px 0 0 #2563eb;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] *,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] p,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] span {
            color: #1e40af !important;
            font-weight: 700 !important;
        }

        /* ═══ User-бейдж (top-right) ═══ */
        .user-badge {
            position: fixed; top: 10px; right: 18px; z-index: 999;
            display: flex; align-items: center; gap: 9px;
            background: rgba(255,255,255,0.92); padding: 5px 16px 5px 6px;
            border-radius: 999px; box-shadow: 0 4px 14px rgba(15,23,42,0.10);
            font-family: Inter, system-ui, sans-serif; font-size: 13px;
            border: 1px solid rgba(226,232,240,0.8);
            backdrop-filter: blur(10px);
        }
        .user-badge .avatar {
            width: 30px; height: 30px; border-radius: 50%;
            background: linear-gradient(135deg, #2563eb, #6366f1);
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 700; font-size: 13px;
            box-shadow: 0 2px 8px rgba(37,99,235,0.30);
        }
        .user-badge .uname { color: #0f172a; font-weight: 600; }

        /* ═══ Фильтр-блоки ═══ */
        .filter-row {
            background: rgba(255,255,255,0.75);
            padding: 0.9rem 1.3rem;
            border-radius: 16px;
            border: 1px solid rgba(226,232,240,0.7);
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
            margin-bottom: 1rem;
            backdrop-filter: blur(6px);
        }

        /* ═══ Скроллбар ═══ */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(148,163,184,0.4);
            border-radius: 999px;
            border: 2px solid transparent;
            background-clip: padding-box;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(100,116,139,0.6); background-clip: padding-box; border: 2px solid transparent; }

        /* ═══ Подписи (caption) ═══ */
        [data-testid="stCaptionContainer"] p,
        .stCaption, [data-testid="caption"] {
            color: #64748b !important;
            font-size: 13px !important;
        }

        /* ═══ Multiselect и Selectbox — компактнее ═══ */
        .stMultiSelect [data-baseweb="tag"] {
            background: linear-gradient(120deg, #e0e7ff, #ddd6fe) !important;
            color: #4338ca !important;
            border-radius: 8px !important;
        }

        /* ═══ Спрятать кнопку Deploy (локальное приложение) ═══ */
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        /* Сместить главное меню «...» левее, чтобы не перекрывать user-badge */
        [data-testid="stToolbar"] {
            right: 0.5rem !important;
        }

        /* ═══ Скрыть служебный iframe-маркер активной страницы ═══ */
        iframe[srcdoc*="_markActive"] {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            visibility: hidden !important;
        }
        /* Скрыть и контейнер этого iframe-а, чтобы не было пустого блока */
        [data-testid="stIFrame"]:has(iframe[srcdoc*="_markActive"]),
        [data-testid="stElementContainer"]:has(iframe[srcdoc*="_markActive"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(val) -> str:
    """Format number as RUB currency string."""
    if val is None:
        return "0 \u20bd"
    return f"{val:,.0f} \u20bd".replace(",", " ")


def format_pct(val) -> str:
    """Format number as percentage."""
    if val is None:
        return "0%"
    return f"{val:.1f}%"


# ── Table formatters (for HTML tables) ──────────────────────

def fmt_number(v, decimals=0, suffix=""):
    """Format number with space separators for HTML tables. Returns '0' for zero, '' for NaN."""
    try:
        v = float(v)
    except (ValueError, TypeError):
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    if v == 0:
        return "0" + suffix
    return f"{v:,.{decimals}f}".replace(",", " ") + suffix


def fmt_pct_tbl(v, decimals=1, zero="0%"):
    """Format percentage for HTML tables."""
    try:
        v = float(v)
    except (ValueError, TypeError):
        return zero
    if math.isnan(v) or math.isinf(v) or v == 0:
        return zero
    return f"{v:.{decimals}f}%"


def table_css(prefix):
    """Generate standard HTML table CSS with given class prefix.

    Premium-grade: sticky thead, smooth zebra, gradient sort indicators,
    soft row hover, gradient totals footer.
    """
    return (
        f'<style>'
        # Wrapper with subtle shadow + clipped border-radius
        f'.{prefix}-wrap{{overflow:auto;border-radius:14px;'
        f'box-shadow:0 4px 18px rgba(15,23,42,.07),0 1px 3px rgba(15,23,42,.04);'
        f'margin:1rem 0;border:1px solid rgba(226,232,240,.85);'
        f'background:#fff;max-height:none}}'
        # Table base
        f'.{prefix}{{border-collapse:separate;border-spacing:0;width:100%;'
        f'font-size:12.5px;font-family:Inter,system-ui,-apple-system,sans-serif;'
        f'background:#fff;color:#1e293b}}'
        # Sticky header with gradient background
        f'.{prefix} thead th{{position:sticky;top:0;z-index:2;'
        f'background:linear-gradient(180deg,#f8fafc,#eef2f7);'
        f'padding:10px 12px;'
        f'border-bottom:2px solid #cbd5e1;'
        f'font-weight:700;font-size:11px;color:#334155;'
        f'text-align:center;white-space:nowrap;cursor:pointer;user-select:none;'
        f'letter-spacing:.02em;text-transform:uppercase;'
        f'transition:background .15s ease}}'
        f'.{prefix} thead th:not(:last-child){{border-right:1px solid rgba(203,213,225,.45)}}'
        f'.{prefix} thead th:hover{{background:linear-gradient(180deg,#eef2f7,#e2e8f0);color:#1e40af}}'
        # Sort arrows
        f'.{prefix} th .sort-arrow{{font-size:9px;margin-left:4px;color:#94a3b8;display:inline-block;transition:color .15s}}'
        f'.{prefix} th.sort-asc .sort-arrow::after{{content:"\\25B2";color:#2563eb;font-weight:700}}'
        f'.{prefix} th.sort-desc .sort-arrow::after{{content:"\\25BC";color:#2563eb;font-weight:700}}'
        f'.{prefix} th:not(.sort-asc):not(.sort-desc) .sort-arrow::after{{content:"\\25B4\\25BE";font-size:8px}}'
        # Cells
        f'.{prefix} tbody td{{padding:8px 12px;'
        f'border-bottom:1px solid rgba(241,245,249,.85);'
        f'white-space:nowrap;font-size:12.5px;line-height:1.4;'
        f'transition:background .12s ease}}'
        f'.{prefix} tbody td:not(:last-child){{border-right:1px solid rgba(248,250,252,.5)}}'
        # Zebra + hover
        f'.{prefix} tbody tr:nth-child(even) td{{background:#fafbfd}}'
        f'.{prefix} tbody tr:hover td{{background:linear-gradient(90deg,#eef2ff,#f5f3ff);'
        f'color:#0f172a}}'
        # Number / pos / neg
        f'.{prefix} .num{{text-align:right;font-variant-numeric:tabular-nums;'
        f'font-feature-settings:"tnum"}}'
        f'.{prefix} .ctr{{text-align:center}}'
        f'.{prefix} .pos{{color:#15803d;font-weight:700}}'
        f'.{prefix} .neg{{color:#b91c1c;font-weight:700}}'
        f'.{prefix} .pct{{color:#64748b;font-size:10.5px}}'
        # Footer (totals) with gradient
        f'.{prefix} tfoot td{{position:sticky;bottom:0;'
        f'background:linear-gradient(180deg,#f1f5f9,#e2e8f0);'
        f'font-weight:700;border-top:2px solid #94a3b8;'
        f'padding:10px 12px;font-size:12.5px;color:#0f172a;'
        f'box-shadow:0 -2px 8px rgba(15,23,42,.06)}}'
        # First column (label) — soft accent
        f'.{prefix} tbody td:first-child{{font-weight:600;color:#334155}}'
        f'</style>'
    )


# ── Sortable table JS ────────────────────────────────────────

SORT_JS = """
<script>
document.addEventListener('DOMContentLoaded', function() { _initSortTables(); });
const _mo = new MutationObserver(function() { _initSortTables(); });
_mo.observe(document.body, {childList: true, subtree: true});

function _initSortTables() {
    document.querySelectorAll('table[data-sortable]').forEach(function(tbl) {
        if (tbl.dataset.sortReady) return;
        tbl.dataset.sortReady = '1';
        tbl.querySelectorAll('thead th').forEach(function(th, idx) {
            if (!th.querySelector('.sort-arrow')) {
                th.innerHTML += '<span class="sort-arrow"></span>';
            }
            th.addEventListener('click', function() {
                _sortTable(tbl, idx, th);
            });
        });
    });
}

function _sortTable(tbl, colIdx, th) {
    var tbody = tbl.querySelector('tbody');
    if (!tbody) return;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var asc = !th.classList.contains('sort-asc');

    tbl.querySelectorAll('thead th').forEach(function(h) {
        h.classList.remove('sort-asc', 'sort-desc');
    });
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    rows.sort(function(a, b) {
        var av = _cellVal(a.cells[colIdx]);
        var bv = _cellVal(b.cells[colIdx]);
        if (typeof av === 'number' && typeof bv === 'number') {
            return asc ? av - bv : bv - av;
        }
        av = String(av).toLowerCase();
        bv = String(bv).toLowerCase();
        return asc ? av.localeCompare(bv, 'ru') : bv.localeCompare(av, 'ru');
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
}

function _cellVal(cell) {
    if (!cell) return '';
    var txt = cell.innerText.replace(/[\\s\\u00a0]/g, '').replace(/,/g, '.');
    txt = txt.replace(/[₽%шт]/g, '').replace(/\\+/g, '').trim();
    if (txt === '' || txt === '—') return Infinity;
    var n = parseFloat(txt);
    return isNaN(n) ? cell.innerText.trim() : n;
}
</script>
"""


def render_table(html: str, height: int = 600):
    """Render HTML table with SORT_JS inside an iframe so <script> executes."""
    # Wrap in full HTML document for proper rendering inside iframe
    doc = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<style>body{margin:0;font-family:Inter,system-ui,-apple-system,sans-serif;'
        'background:transparent;overflow:auto;}</style></head><body>'
        + html +
        '</body></html>'
    )
    st.iframe(doc, height=height)


def render_sortable_table(
    key: str,
    hdr: str,
    rows: str,
    *,
    foot: str = "",
    height: int = 600,
) -> None:
    """Compact helper: compose `table_css(key)` + thead/tbody/foot + SORT_JS and render.

    Parameters
    ----------
    key : str
        CSS class prefix — used as both table class and wrapper class ``{key}-wrap``.
    hdr : str
        Inner HTML of the ``<thead>`` section (usually a single ``<tr>...</tr>``).
    rows : str
        Inner HTML of the ``<tbody>`` section (multiple ``<tr>...</tr>``).
    foot : str, optional
        Inner HTML of the ``<tfoot>`` section (totals row). Empty by default.
    height : int, optional
        Iframe height passed to ``render_table``. Defaults to 600.
    """
    css = table_css(key)
    tfoot = f"<tfoot>{foot}</tfoot>" if foot else ""
    html = (
        f'{css}<div class="{key}-wrap"><table class="{key}" data-sortable>'
        f'<thead>{hdr}</thead><tbody>{rows}</tbody>{tfoot}</table></div>{SORT_JS}'
    )
    render_table(html, height=height)


def paginate(df, key_prefix: str, default_size: int = 50):
    """Compact paginator widget: returns (sliced_df, start_idx, end_idx, total).

    Renders two inputs side-by-side (page size + page number) and slices `df`.
    Use as: sliced, start, end, total = paginate(my_df, key_prefix="prf")
    """
    total = len(df)
    _c1, _c2, _c3 = st.columns([1, 1, 4])
    with _c1:
        sizes = [25, 50, 100, 250, 500]
        idx = sizes.index(default_size) if default_size in sizes else 1
        size = st.selectbox(
            "Строк", sizes, index=idx,
            key=f"{key_prefix}_page_size", label_visibility="collapsed",
        )
    total_pages = max((total - 1) // size + 1, 1)
    with _c2:
        page = st.number_input(
            "Страница", min_value=1, max_value=total_pages, value=1,
            key=f"{key_prefix}_page_num", label_visibility="collapsed",
        )
    start = (int(page) - 1) * int(size)
    end = min(start + int(size), total)
    with _c3:
        st.caption(f"Показано {start + 1}–{end} из {total} (стр. {int(page)}/{total_pages})")
    return df.iloc[start:end], start, end, total


def export_buttons(df, basename: str, key: str | None = None, *, sheet_name: str = "Report"):
    """Render CSV + Excel download buttons side-by-side for a dataframe.

    `basename` becomes the file stem (without extension).
    `key` disambiguates button widget keys on pages that render multiple exports.
    """
    import io

    _k = key or basename
    _c1, _c2 = st.columns(2)
    with _c1:
        st.download_button(
            "📥 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            f"{basename}.csv",
            "text/csv",
            key=f"dl_csv_{_k}",
            width="stretch",
        )
    with _c2:
        try:
            buf = io.BytesIO()
            # openpyxl не поддерживает tz-aware datetime и некоторые dtypes,
            # поэтому приводим их к совместимому виду перед записью.
            _df = df.copy()
            for _col in _df.columns:
                _s = _df[_col]
                # Strip timezone
                if pd.api.types.is_datetime64_any_dtype(_s):
                    try:
                        _df[_col] = _s.dt.tz_localize(None)
                    except TypeError:
                        # Already naive
                        pass
                # Convert pandas-nullable dtypes (Int64/Float64/boolean) to numpy
                elif str(_s.dtype) in ("Int64", "Int32", "Int16", "Float64", "Float32", "boolean"):
                    _df[_col] = _s.astype("object").where(_s.notna(), None)
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                _df.to_excel(w, index=False, sheet_name=sheet_name[:31])
            st.download_button(
                "📊 Excel",
                buf.getvalue(),
                f"{basename}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{_k}",
                width="stretch",
            )
        except Exception as e:
            st.caption(f"Excel недоступен: {type(e).__name__}: {e}")


# ── Grouped sidebar navigation ──────────────────────────────
# Reports grouped by type. Each tuple: (icon, label, page path relative to app/).
# The Home page is rendered first as a standalone link, groups follow.
_SIDEBAR_NAV_GROUPS = [
    ("Обзор и сводки", [
        ("📈", "KPI-дашборд", "pages/01_KPI_Дашборд.py"),
        ("📅", "Еженедельный отчёт", "pages/02_Еженедельный_отчёт.py"),
        ("📋", "Отчёт за период", "pages/07_Отчёт_за_период.py"),
        ("🔄", "Неделя к неделе", "pages/12_Неделя_к_неделе.py"),
        ("🫀", "Рука на Пульсе", "pages/15_РнП.py"),
    ]),
    ("Финансы и прибыль", [
        ("📊", "ОПИУ", "pages/11_ОПИУ.py"),
        ("💰", "Рентабельность", "pages/06_Рентабельность.py"),
        ("🧮", "Юнит-экономика", "pages/17_Юнит_экономика.py"),
    ]),
    ("Ассортимент и логистика", [
        ("📦", "Отчёт по артикулам", "pages/03_Отчёт_по_артикулам.py"),
        ("🏭", "Остатки на складах", "pages/04_Остатки_на_складах.py"),
        ("🔤", "ABC-анализ", "pages/05_ABC_анализ.py"),
        ("👥", "Когорты", "pages/18_Когорты.py"),
        ("📦", "Потребность", "pages/09_Потребность.py"),
    ]),
    ("Маркетинг и планирование", [
        ("📈", "Прогноз", "pages/08_Прогноз.py"),
        ("📢", "Конверсия рекламы", "pages/13_Конверсия_рекламы.py"),
        ("🏷️", "Калькулятор акций", "pages/10_Калькулятор_акций.py"),
    ]),
    ("Служебное", [
        ("🚨", "Алерты", "pages/16_Алерты.py"),
        ("🧪", "Качество данных", "pages/19_Качество_данных.py"),
        ("📚", "Справочники", "pages/14_Справочники.py"),
    ]),
]


def render_sidebar_nav():
    """Render the grouped sidebar navigation (replaces Streamlit's auto-nav).

    The auto-generated nav is hidden via CSS in ``inject_global_styles``.
    We output a custom nav with thematic group headers and compact page links.
    A small JS marker assigns ``aria-current="page"`` to the link matching
    the current URL, so the CSS in ``inject_global_styles`` can highlight it.
    """
    with st.sidebar:
        st.markdown('<div class="sb-nav">', unsafe_allow_html=True)
        # Home link on top
        try:
            st.page_link("Home.py", label="Главная", icon="🏠")
        except Exception:
            pass
        for title, items in _SIDEBAR_NAV_GROUPS:
            st.markdown(
                f'<div class="sb-group-title">{title}</div>',
                unsafe_allow_html=True,
            )
            for icon, label, path in items:
                try:
                    st.page_link(path, label=label, icon=icon)
                except Exception:
                    # Page file missing or inaccessible — skip silently
                    continue
        st.markdown('</div>', unsafe_allow_html=True)
    # ── Highlight current page in our custom nav ─────────────
    # ``st.page_link`` doesn't set aria-current. We inject a tiny iframe
    # whose script reaches parent.document and marks the matching link,
    # so the CSS in ``inject_global_styles`` can highlight it.
    _ACTIVE_NAV_JS = """
    <!DOCTYPE html><html><head><style>html,body{margin:0;padding:0;background:transparent}</style></head><body>
    <script>
    (function(){
      var doc = window.parent && window.parent.document;
      if(!doc) return;
      function _markActive(){
        var path = decodeURIComponent(window.parent.location.pathname || '');
        var trim = path.replace(/^\\/+|\\/+$/g, '');
        var links = doc.querySelectorAll('[data-testid="stPageLink-NavLink"]');
        links.forEach(function(a){
          var href = decodeURIComponent(a.getAttribute('href') || '');
          href = href.replace(/^\\/+|\\/+$/g, '');
          var base = href.split('/').pop();
          if (!trim) {
            if (base === '' || base === 'Home' || base === 'Home.py') a.setAttribute('aria-current','page');
            else a.removeAttribute('aria-current');
          } else if (base && trim.endsWith(base)) {
            a.setAttribute('aria-current','page');
          } else {
            a.removeAttribute('aria-current');
          }
        });
      }
      try { _markActive(); } catch(e){}
      try {
        var mo = new MutationObserver(function(){ _markActive(); });
        mo.observe(doc.body, {childList:true, subtree:true});
      } catch(e){}
    })();
    </script>
    </body></html>
    """
    # height=1 (Streamlit forbids 0); CSS in inject_global_styles hides
    # any 1-px iframe ``[data-testid="stIFrame"][height="1"]``.
    st.iframe(_ACTIVE_NAV_JS, height=1)


def render_sidebar_search():
    """Global article search shown in the sidebar on every page.

    Selecting an article stores its ``nm_id`` in ``st.session_state['_search_nm_id']``
    and navigates to the Article report page via ``st.switch_page``.
    Fails silently if the master query or the target page is unavailable.
    """
    try:
        from marts import fetch_dataframe, ARTICLES_MASTER_QUERY
    except Exception:
        return

    try:
        df = fetch_dataframe(ARTICLES_MASTER_QUERY, {})
    except Exception:
        return
    if df is None or df.empty:
        return

    with st.sidebar:
        st.divider()
        st.caption("🔍 Поиск артикула")
        opts = [("", "— выберите —")] + [
            (
                int(r["nm_id"]),
                f"{int(r['nm_id'])} · {r.get('supplier_article') or ''}",
            )
            for _, r in df.iterrows()
        ]
        labels = [o[1] for o in opts]
        choice = st.selectbox(
            "Артикул", labels, index=0, key="_sidebar_search",
            label_visibility="collapsed",
        )
        if choice and choice != opts[0][1]:
            # Map label → nm_id
            nm_id = next((o[0] for o in opts if o[1] == choice), None)
            if nm_id:
                st.session_state["_search_nm_id"] = nm_id
                try:
                    st.switch_page("pages/03_Отчёт_по_артикулам.py")
                except Exception:
                    st.caption(f"→ nm_id сохранён: {nm_id}")


def wb_link(nm_id, text=None):
    """Return HTML anchor for a WB product page."""
    nm = int(nm_id) if nm_id else 0
    if not nm:
        return str(text or "")
    label = text if text is not None else str(nm)
    return (
        f'<a href="https://www.wildberries.ru/catalog/{nm}/detail.aspx" '
        f'target="_blank" rel="noopener" '
        f'style="color:#2563eb;text-decoration:none;border-bottom:1px dashed #93c5fd"'
        f' title="Открыть на WB">{label}</a>'
    )
