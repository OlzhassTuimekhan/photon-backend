# Dashboard Page API

## Описание
Страница дашборда предоставляет общий обзор системы: ключевые показатели (KPI), статус агентов, график рынка, heatmap активных символов, ленту сделок и лог сообщений между агентами.

## Эндпойнты

### 1. GET /api/trading/dashboard/overview/
Получает ключевые показатели для KPI карточек.

**Ответ:**
```json
{
  "balance": 10245.50,
  "todayPnL": 245.50,
  "todayTradesCount": 12,
  "winRate": 68.5,
  "agentsStatus": "All Active",
  "activeAgentsCount": 3
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/dashboard/overview/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 2. GET /api/trading/dashboard/market-chart/
Получает данные для графика цены символа.

**Параметры запроса:**
- `symbol` (опционально) - символ для графика (по умолчанию "AAPL")
- `timeframe` (опционально) - таймфрейм: "15m", "1h", "4h", "1d" (по умолчанию "1h")

**Ответ:**
```json
{
  "symbol": "AAPL",
  "currentPrice": 150.32,
  "data": [
    {
      "timestamp": "2025-12-07T10:00:00Z",
      "price": 150.00,
      "volume": 1000000
    },
    {
      "timestamp": "2025-12-07T11:00:00Z",
      "price": 150.32,
      "volume": 1200000
    }
  ]
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/dashboard/market-chart/?symbol=AAPL&timeframe=1h" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Периоды данных по таймфреймам:**
- `15m` - последние 6 часов
- `1h` - последние 7 дней
- `4h` - последние 30 дней
- `1d` - последние 90 дней

---

### 3. GET /api/trading/dashboard/market-heatmap/
Получает данные для heatmap рынка (топ гейнеры и лузеры среди активных символов).

**Ответ:**
```json
[
  {
    "symbol": "AAPL",
    "price": 150.32,
    "previousPrice": 149.50,
    "volume": 1000000,
    "timestamp": "2025-12-07T14:51:00Z"
  },
  {
    "symbol": "MSFT",
    "price": 318.00,
    "previousPrice": 320.00,
    "volume": 500000,
    "timestamp": "2025-12-07T14:51:00Z"
  }
]
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/dashboard/market-heatmap/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Данные отсортированы по изменению цены (от большего к меньшему). Фронтенд может разделить на топ гейнеры (положительное изменение) и топ лузеры (отрицательное изменение).

---

## Использование существующих эндпойнтов

Dashboard также использует следующие эндпойнты:

**LiveAgentStatusCards:**
- `GET /api/trading/agents/detail/` - статус всех агентов

**TradeActivityFeed:**
- `GET /api/trading/trades/?limit=10` - последние 10 сделок

**MessageLog:**
- `GET /api/trading/messages/?limit=20` - последние 20 сообщений между агентами

---

## Использование на фронтенде

1. При загрузке страницы вызвать `/api/trading/dashboard/overview/` для KPI карточек
2. Вызвать `/api/trading/agents/detail/` для статуса агентов
3. Вызвать `/api/trading/dashboard/market-chart/?symbol=AAPL&timeframe=1h` для графика
4. Вызвать `/api/trading/dashboard/market-heatmap/` для heatmap
5. Вызвать `/api/trading/trades/?limit=10` для ленты сделок
6. Вызвать `/api/trading/messages/?limit=20` для лога сообщений

Все эндпойнты требуют аутентификации через JWT токен.

