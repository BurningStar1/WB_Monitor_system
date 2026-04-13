import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
from styles import inject_global_styles

st.set_page_config(page_title="WB Analytics", layout="wide", page_icon="📊")
inject_global_styles()

st.title("Аналитический сервис Wildberries")
st.markdown(
    """
    Добро пожаловать в систему мониторинга продаж. Выберите отчёт
    в боковом меню для просмотра данных.

    **Доступные отчёты:**
    - 📈 **KPI-дашборд** — сводные показатели за период
    - 📅 **Еженедельный отчёт** — динамика по неделям
    - 📦 **Отчёт по артикулам** — детализация до товара
    - 🏭 **Остатки на складах** — распределение по складам
    - 🔤 **ABC-анализ** — классификация товаров по выручке
    - 💰 **Отчёт о прибыли** — финансовый результат
    - 📋 **Отчёт за период** — помесячная сводка
    """
)
