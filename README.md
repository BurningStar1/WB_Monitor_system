# WB Analytics Service (ВКР каркас)

Тема: аналитический сервис интерактивной и регламентной отчетности по данным Wildberries.

## 1. Архитектура

- Поток: `RAW -> STG -> MART`
- Подход: гибридный ETL/ELT
  - Python: загрузка API, первичная валидация, импорт внутренних справочников
  - PostgreSQL: нормализация, агрегации, витрина, представления
- Расширяемость: `api/endpoints.py` и клиент `api/wb_client.py` позволяют подключать другие маркетплейсы через отдельный коннектор.

## 2. Структура репозитория

- `config/` - конфигурация и env
- `db/` - engine/repository
- `api/` - клиент Wildberries и registry endpoint-ов
- `etl/` - RAW/STG/MART pipeline + загрузка внутренних справочников
- `marts/` - SQL запросы для интерфейса
- `app/` - Streamlit интерфейс и страницы отчетов
- `sql/` - DDL/индексы/view
- `utils/` - логирование/вспомогательные утилиты
- `checks/` - проверки качества
- `tests/` - базовые тесты
- `scripts/` - точка запуска и инициализация БД
- `data/internal/` - место для XLSX-файлов внутренних справочников

## 3. Принятые допущения по endpoint-ам WB

Базовый набор endpoint-ов (заменяемый):
- `orders`: `/api/v1/supplier/orders`
- `sales`: `/api/v1/supplier/sales`
- `stocks`: `/api/v1/supplier/stocks`
- `income` (опционально): `/api/v1/supplier/incomes`

Если в аккаунте/версии API часть endpoint недоступна, реестр можно заменить централизованно в `api/endpoints.py`.

## 4. Локальный запуск

1. Создать и активировать виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Подготовить env

```powershell
Copy-Item .env.example .env
```

Заполнить `.env` и/или `wb_api_key.txt` в корне проекта.

3. Инициализировать БД

```powershell
python scripts/init_db.py
```

4. (Опционально) загрузить внутренние справочники из XLSX

```powershell
python scripts/load_internal_dicts.py --cost data/internal/cost_reference.xlsx --expenses data/internal/extra_expenses.xlsx --tax data/internal/tax_reference.xlsx
```

5. Запустить ETL

```powershell
python scripts/run_etl.py --days-back 30
```

6. Запустить Streamlit

```powershell
streamlit run app/Home.py
```

7. Базовые тесты

```powershell
pytest -q
```

## 5. Безопасность

- Токен не хранится в коде.
- Поддержаны `.env` и `wb_api_key.txt`.
- Логирование без вывода токена.

## 6. Что дальше

- Добавить строгую валидацию схем ответов API (pydantic).
- Добавить идемпотентность импорта внутренних справочников через upsert.
- Расширить data quality checks и интеграционные тесты.
- Добавить регламентный экспорт (xlsx/pdf) и планировщик загрузок.
