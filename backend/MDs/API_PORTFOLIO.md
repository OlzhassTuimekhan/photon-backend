# Portfolio Page API

## Описание
Страница портфеля отображает текущее состояние торгового счета, открытые позиции, историю сделок и график изменения баланса.

## Эндпойнты

### 1. GET /api/trading/portfolio/
Получает общую информацию о портфеле пользователя.

**Ответ:**
```json
{
  "balance": 10245.50,
  "freeCash": 8245.50,
  "usedMargin": 2000.00,
  "totalTrades": 45,
  "todayPnL": 245.50,
  "totalPnL": 1245.50
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/portfolio/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 2. GET /api/trading/positions/
Получает список открытых позиций пользователя.

**Ответ:**
```json
[
  {
    "id": 1,
    "symbol": "AAPL",
    "quantity": 10,
    "entryPrice": 150.00,
    "currentPrice": 152.30,
    "pnl": 23.00,
    "pnlPercent": 1.53,
    "openedAt": "2025-12-07T10:00:00Z"
  }
]
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/positions/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 3. GET /api/trading/trades/
Получает последние 20 сделок пользователя.

**Параметры запроса:**
- `limit` (опционально) - количество сделок (по умолчанию 20)

**Ответ:**
```json
[
  {
    "id": 1,
    "symbol": "AAPL",
    "action": "BUY",
    "price": 150.00,
    "quantity": 10,
    "agentType": "DECISION_MAKER",
    "pnl": null,
    "timestamp": "2025-12-07T10:00:00Z"
  }
]
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/trades/?limit=20" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 4. GET /api/trading/portfolio/equity-curve/
Получает данные для графика изменения баланса (equity curve).

**Ответ:**
```json
{
  "initialBalance": 10000.00,
  "currentBalance": 10245.50,
  "maxDrawdown": -150.00,
  "sharpeRatio": 1.25,
  "data": [
    {
      "date": "2025-12-01",
      "balance": 10000.00
    },
    {
      "date": "2025-12-07",
      "balance": 10245.50
    }
  ]
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/portfolio/equity-curve/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Использование на фронтенде

1. При загрузке страницы вызвать `/api/trading/portfolio/` для получения общей информации
2. Вызвать `/api/trading/positions/` для отображения открытых позиций
3. Вызвать `/api/trading/trades/` для истории сделок
4. Вызвать `/api/trading/portfolio/equity-curve/` для построения графика баланса

Все эндпойнты требуют аутентификации через JWT токен.

