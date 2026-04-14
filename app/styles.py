"""Global Streamlit styles — white-blue business theme."""
import math
import streamlit as st
import pandas as pd


# ── Plotly shared hover / layout ────────────────────────────

PLOTLY_HOVER = dict(
    bgcolor="rgba(15,23,42,0.88)",
    font_size=12,
    font_family="Inter, system-ui, sans-serif",
    font_color="#f1f5f9",
    bordercolor="rgba(99,102,241,0.35)",
)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    hoverlabel=PLOTLY_HOVER,
    font=dict(family="Inter, system-ui, sans-serif", color="#334155", size=12),
)

# Axis / legend / margin defaults — applied via plotly_defaults(fig)
_AXIS_STYLE = dict(
    gridcolor="rgba(226,232,240,0.6)",
    gridwidth=1,
    zeroline=False,
    tickfont=dict(size=11, color="#64748b"),
)

_LEGEND_STYLE = dict(
    font=dict(size=11, color="#475569"),
    bgcolor="rgba(255,255,255,0)",
    borderwidth=0,
)


def plotly_defaults(fig):
    """Apply polished axis/grid/legend defaults to any Plotly figure.

    Call AFTER update_layout so page-specific overrides win.
    """
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE, separatethousands=True)
    fig.update_layout(
        legend={**_LEGEND_STYLE, **fig.layout.legend.to_plotly_json()},
        margin=dict(l=10, r=10, t=32, b=10) if fig.layout.margin.l is None else {},
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
        .stApp {
            background: radial-gradient(circle at top, rgba(37,99,235,0.18), rgba(59,130,246,0.08)), #f0f4fa !important;
            color: #0f172a;
        }
        /* Full-width layout */
        .block-container, [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: rgba(255,255,255,0.7);
            padding: 0.4rem 0.6rem;
            border-radius: 999px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 999px;
            padding: 0.35rem 1.2rem;
            color: #475569;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #2563eb, #3b82f6);
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(37,99,235,0.30);
        }
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(59,130,246,0.12));
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(59,130,246,0.15);
            box-shadow: 0 10px 25px rgba(15,23,42,0.08);
        }
        [data-testid="stTable"], .stDataFrame {
            background: rgba(255,255,255,0.85);
            border-radius: 16px;
            padding: 0.4rem;
            box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        }
        .stButton>button, .stDownloadButton>button, .stForm button {
            background: linear-gradient(120deg, #1d4ed8, #2563eb);
            border: none;
            color: white;
            padding: 0.45rem 1.4rem;
            border-radius: 999px;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(37,99,235,0.25);
            transition: all 0.2s;
        }
        .stButton>button:hover, .stDownloadButton>button:hover, .stForm button:hover {
            background: linear-gradient(120deg, #1e40af, #1d4ed8);
            box-shadow: 0 12px 24px rgba(30,64,175,0.35);
        }
        .stForm {
            background: rgba(255,255,255,0.75);
            padding: 1rem 1.4rem;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(15,23,42,0.07);
            border: 1px solid rgba(148,163,184,0.15);
        }
        h1, h2, h3 {
            color: #0f172a !important;
        }
        h3:after {
            content: "";
            display: block;
            width: 50px;
            height: 3px;
            border-radius: 999px;
            margin-top: 5px;
            background: linear-gradient(120deg, #2563eb, #60a5fa);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff, #f0f4fa) !important;
            color: #0f172a !important;
            box-shadow: 3px 0 20px rgba(15,23,42,0.06);
        }
        /* ── User badge (top-right) ── */
        .user-badge {
            position: fixed; top: 8px; right: 16px; z-index: 999;
            display: flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.92); padding: 5px 14px 5px 6px;
            border-radius: 999px; box-shadow: 0 2px 10px rgba(15,23,42,0.10);
            font-family: Inter, system-ui, sans-serif; font-size: 13px;
            border: 1px solid rgba(148,163,184,0.18);
            backdrop-filter: blur(8px);
        }
        .user-badge .avatar {
            width: 28px; height: 28px; border-radius: 50%;
            background: linear-gradient(135deg, #2563eb, #60a5fa);
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 700; font-size: 12px;
        }
        .user-badge .uname { color: #1e293b; font-weight: 600; }
        /* ── Filter row styling ── */
        .filter-row {
            background: rgba(255,255,255,0.7);
            padding: 0.8rem 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.12);
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            margin-bottom: 1rem;
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
    """Generate standard HTML table CSS with given class prefix."""
    return (
        f'<style>'
        f'.{prefix}-wrap{{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);'
        f'margin:1rem 0;border:1px solid #e2e8f0}}'
        f'.{prefix}{{border-collapse:collapse;width:100%;font-size:12px;font-family:Inter,system-ui,sans-serif;'
        f'background:#fff;color:#1e293b}}'
        f'.{prefix} th{{background:#f1f5f9;padding:8px 10px;border-bottom:2px solid #cbd5e1;'
        f'border-right:1px solid #e2e8f0;font-weight:600;font-size:11px;color:#475569;'
        f'text-align:center;white-space:nowrap;cursor:pointer;user-select:none;position:relative}}'
        f'.{prefix} th:hover{{background:#e2e8f0}}'
        f'.{prefix} th .sort-arrow{{font-size:9px;margin-left:3px;color:#94a3b8;display:inline-block}}'
        f'.{prefix} th.sort-asc .sort-arrow::after{{content:"\\25B2";color:#2563eb}}'
        f'.{prefix} th.sort-desc .sort-arrow::after{{content:"\\25BC";color:#2563eb}}'
        f'.{prefix} th:not(.sort-asc):not(.sort-desc) .sort-arrow::after{{content:"\\25B4\\25BE";font-size:8px}}'
        f'.{prefix} td{{padding:6px 10px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;'
        f'white-space:nowrap;font-size:12px}}'
        f'.{prefix} tbody tr:nth-child(even){{background:#fafbfc}}'
        f'.{prefix} tbody tr:hover{{background:#eef2ff}}'
        f'.{prefix} .num{{text-align:right}}'
        f'.{prefix} .ctr{{text-align:center}}'
        f'.{prefix} .pos{{color:#16a34a;font-weight:700}}'
        f'.{prefix} .neg{{color:#dc2626;font-weight:700}}'
        f'.{prefix} .pct{{color:#64748b;font-size:10px}}'
        f'.{prefix} tfoot td{{background:#f1f5f9;font-weight:700;border-top:2px solid #cbd5e1}}'
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
            use_container_width=True,
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
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Excel недоступен: {type(e).__name__}: {e}")


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
