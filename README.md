# Photon - Multi-Agent Trading System

Система автоматической торговли с мульти-агентной архитектурой для финального проекта.

## Архитектура

Система состоит из трех агентов:

1. **Market Monitoring Agent** - мониторит данные рынка в реальном времени
2. **Decision-Making Agent** - анализирует данные и принимает решения (BUY/SELL/HOLD)
3. **Execution Agent** - выполняет сделки (пока только статус, реальные сделки не выполняются)

## Технологии

- **Backend**: Django 5 + DRF + Celery + PostgreSQL + Redis
- **Frontend**: React (будет добавлен позже)
- **Данные рынка**: 
  - **yfinance** (Yahoo Finance API) - для акций
  - **Bybit API** - для криптовалют (BTC, ETH и т.д.)
  - Автоматическое определение источника данных

## Структура проекта

```
photon/
├── backend/          # Django backend
│   ├── config/       # Настройки проекта
│   ├── core/         # Аутентификация и пользователи
│   └── trading/      # Торговая система
└── frontend/         # React frontend (будет добавлен)
```

## Быстрый старт

### 1. Настройка окружения

```bash
cd photon/backend
cp env.example .env
# Отредактируйте .env при необходимости
```

#### Настройка Bybit API (опционально)

Для получения публичных данных (цены, история) API ключи **НЕ обязательны**!

Если хотите использовать API ключи (для расширенных возможностей):
1. Следуйте инструкции в `BYBIT_API_GUIDE.md`
2. Добавьте ключи в `.env`:
   ```env
   BYBIT_API_KEY=your_api_key_here
   BYBIT_SECRET_KEY=your_secret_key_here
   ```

**Важно**: Для публичных данных ключи не нужны! Система автоматически использует публичные эндпойнты Bybit.

### 2. Запуск через Docker Compose

```bash
cd photon
docker compose up --build
```

Backend будет доступен на `http://localhost:666`

### 3. Запуск локально (без Docker)

```bash
# Установка зависимостей
cd photon/backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Настройка БД (убедитесь, что PostgreSQL и Redis запущены)
python manage.py migrate
python manage.py createsuperuser

# Запуск сервера
python manage.py runserver

# В отдельном терминале - Celery worker
celery -A config worker -l info

# В еще одном терминале - Celery beat (для периодических задач)
celery -A config beat -l info
```

## API Эндпойнты

### Аутентификация

- `POST /api/auth/register/` - Регистрация
- `POST /api/auth/login/` - Вход
- `POST /api/auth/refresh/` - Обновление токена
- `GET /api/auth/me/` - Профиль пользователя

### Торговая система

#### Символы
- `GET /api/trading/symbols/` - Список символов
- `POST /api/trading/symbols/` - Добавить символ (например, "AAPL", "TSLA")
- `GET /api/trading/symbols/{id}/` - Детали символа
- `DELETE /api/trading/symbols/{id}/` - Удалить символ

#### Данные рынка
- `GET /api/trading/market-data/` - История данных рынка
- `GET /api/trading/market-data/latest/` - Последние данные для всех символов
- `POST /api/trading/market-data/refresh/` - Обновить данные

#### Решения
- `GET /api/trading/decisions/` - История решений
- `POST /api/trading/decisions/` - Создать решение (через Decision-Making Agent)
- `GET /api/trading/decisions/statistics/` - Статистика по решениям

#### Агенты
- `GET /api/trading/agents/market-monitor/` - Статус Market Monitoring Agent
- `POST /api/trading/agents/market-monitor/` - Запустить/остановить мониторинг
  ```json
  {"action": "start"}  // или "stop"
  ```
- `GET /api/trading/agents/decision-maker/` - Статус Decision-Making Agent
- `POST /api/trading/agents/decision-maker/` - Запросить анализ
  ```json
  {"symbol_id": 1}
  ```
- `GET /api/trading/agents/execution/` - Статус Execution Agent

## Примеры использования

### 1. Добавить символ для отслеживания

**Акции (через yfinance):**
```bash
curl -X POST http://localhost:666/api/trading/symbols/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

**Криптовалюты (через Bybit):**
```bash
curl -X POST http://localhost:666/api/trading/symbols/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}'
```

Или просто:
```bash
curl -X POST http://localhost:666/api/trading/symbols/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC"}'  # Автоматически преобразуется в BTCUSDT
```

### 2. Запустить мониторинг рынка

```bash
curl -X POST http://localhost:666/api/trading/agents/market-monitor/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

### 3. Получить последние данные

```bash
curl http://localhost:666/api/trading/market-data/latest/ \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Запросить решение от Decision-Making Agent

```bash
curl -X POST http://localhost:666/api/trading/agents/decision-maker/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol_id": 1}'
```

## Модели данных

- **Symbol** - Отслеживаемые торговые символы (AAPL, TSLA и т.д.)
- **MarketData** - Данные рынка в реальном времени (цена, объем, изменения)
- **TradingDecision** - Решения агента (BUY/SELL/HOLD с обоснованием)
- **AgentStatus** - Статусы работы агентов

## Celery задачи

- `start_market_monitoring` - Запуск мониторинга для пользователя
- `periodic_market_update` - Периодическое обновление данных (запускается через Celery Beat)

## TODO

- [ ] Реализовать AI модель для Decision-Making Agent
- [ ] Добавить WebSocket для real-time обновлений на фронтенде
- [ ] Добавить исторические графики
- [ ] Улучшить обработку ошибок
- [ ] Добавить тесты

## Примечания

- Система автоматически определяет источник данных:
  - **yfinance** для акций (AAPL, TSLA, MSFT и т.д.)
  - **Bybit API** для криптовалют (BTC, ETH, BTCUSDT и т.д.)
- Реальные сделки **не выполняются** - система только принимает решения
- AI модель пока не реализована - решения возвращают заглушку "HOLD"
- API ключи Bybit **не обязательны** для публичных данных

## Источники данных

### Поддерживаемые форматы символов:

**Акции (yfinance):**
- `AAPL` - Apple
- `TSLA` - Tesla
- `MSFT` - Microsoft
- `GOOGL` - Google
- И другие символы с бирж NYSE, NASDAQ

**Криптовалюты (Bybit):**
- `BTC` или `BTCUSDT` - Bitcoin
- `ETH` или `ETHUSDT` - Ethereum
- `BNB` или `BNBUSDT` - Binance Coin
- И другие пары с Bybit

Система автоматически определяет тип символа и использует соответствующий API.

