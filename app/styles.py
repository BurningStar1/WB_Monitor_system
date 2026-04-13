"""Global Streamlit styles — white-blue business theme."""
import streamlit as st


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
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stMultiSelect label {
            color: #1e3a5f !important;
            font-weight: 600;
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
