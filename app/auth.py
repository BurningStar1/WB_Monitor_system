"""Простая аутентификация для Streamlit-приложения."""
import hashlib
import os
import streamlit as st


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    """Build {login: sha256(password)} from the APP_USERS env variable.

    Format: ``login1:password1,login2:password2``. Falls back to the historic
    defaults if the variable is missing so existing deployments keep working.
    """
    raw = os.environ.get("APP_USERS", "").strip()
    if not raw:
        return {
            "admin": _hash("admin123"),
            "analyst": _hash("wb2024"),
        }
    users: dict = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        login, pw = pair.split(":", 1)
        login = login.strip()
        pw = pw.strip()
        if login and pw:
            users[login] = _hash(pw)
    return users or {"admin": _hash("admin123")}


USERS = _load_users()


def check_auth() -> bool:
    """Показать форму входа, если пользователь не авторизован.

    Возвращает True если авторизован, False если нет (и показывает форму).
    Вызывать в начале каждой страницы или в Home.py.
    """
    if st.session_state.get("authenticated"):
        return True

    # ── Стили формы ──────────────────────────────────────────
    st.markdown(
        """
        <style>
        .auth-container {
            max-width: 400px;
            margin: 5rem auto;
            padding: 2.5rem;
            background: white;
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.1);
        }
        .auth-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.3rem;
        }
        .auth-subtitle {
            text-align: center;
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .auth-logo {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Центрируем форму
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="auth-logo">📊</div>'
            '<div class="auth-title">WB Analytics</div>'
            '<div class="auth-subtitle">Войдите для доступа к аналитике</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Логин", placeholder="Введите логин")
            password = st.text_input("Пароль", type="password", placeholder="Введите пароль")
            submit = st.form_submit_button("Войти", width="stretch")

        if submit:
            if username in USERS and USERS[username] == _hash(password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Неверный логин или пароль")

    return False


def logout():
    """Бейдж пользователя (top-right) + кнопка выхода в сайдбаре."""
    if st.session_state.get("authenticated"):
        user = st.session_state.get("username", "")
        initial = user[0].upper() if user else "?"
        st.markdown(
            f'<div class="user-badge">'
            f'<div class="avatar">{initial}</div>'
            f'<span class="uname">{user}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Global article search (sidebar). Safe if DB/master query fails.
        try:
            from styles import render_sidebar_search
            render_sidebar_search()
        except Exception:
            pass
        with st.sidebar:
            st.divider()
            if st.button("\U0001f6aa Выйти", width="stretch"):
                st.session_state["authenticated"] = False
                st.session_state["username"] = ""
                st.rerun()
