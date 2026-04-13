"""Простая аутентификация для Streamlit-приложения."""
import hashlib
import streamlit as st


# ── Пользователи (login → password hash SHA-256) ─────────────
# Пароль хешируется: hashlib.sha256("пароль".encode()).hexdigest()
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("wb2024".encode()).hexdigest(),
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


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
            submit = st.form_submit_button("Войти", use_container_width=True)

        if submit:
            if username in USERS and USERS[username] == _hash(password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Неверный логин или пароль")

    return False


def logout():
    """Кнопка выхода в сайдбаре."""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.divider()
            user = st.session_state.get("username", "")
            st.caption(f"Пользователь: **{user}**")
            if st.button("Выйти", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["username"] = ""
                st.rerun()
