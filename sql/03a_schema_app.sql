CREATE SCHEMA IF NOT EXISTS app;

-- --- Пользователи (сотрудники) --- --
CREATE TABLE IF NOT EXISTS app.users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --- Аккаунты WB (магазины / кабинеты продавца) --- --
CREATE TABLE IF NOT EXISTS app.wb_accounts (
    id BIGSERIAL PRIMARY KEY,
    account_name TEXT NOT NULL,
    api_token TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --- Связь пользователей и аккаунтов WB (M:N) --- --
CREATE TABLE IF NOT EXISTS app.user_accounts (
    user_id BIGINT NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
    account_id BIGINT NOT NULL REFERENCES app.wb_accounts(id) ON DELETE CASCADE,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, account_id)
);
