# Agents Page API

## Описание
Страница агентов отображает статус и активность всех трех агентов системы: Market Monitor, Decision Maker и Execution Agent.

## Эндпойнты

### 1. GET /api/trading/agents/detail/
Получает детальную информацию о всех агентах пользователя.

**Ответ:**
```json
[
  {
    "id": 1,
    "type": "market",
    "name": "Market Monitor",
    "status": "active",
    "lastAction": "Monitoring AAPL, BTCUSDT",
    "lastUpdated": "2025-12-07T14:51:00Z",
    "messagesProcessed": 125,
    "logs": [
      {
        "id": 1,
        "level": "INFO",
        "message": "Updated market data for AAPL",
        "timestamp": "2025-12-07T14:51:00Z"
      }
    ]
  },
  {
    "id": 2,
    "type": "decision",
    "name": "Decision Maker",
    "status": "active",
    "lastAction": "Analyzing market trends",
    "lastUpdated": "2025-12-07T14:50:30Z",
    "messagesProcessed": 98,
    "logs": []
  },
  {
    "id": 3,
    "type": "execution",
    "name": "Execution Agent",
    "status": "idle",
    "lastAction": "No actions yet",
    "lastUpdated": "2025-12-07T14:49:15Z",
    "messagesProcessed": 45,
    "logs": []
  }
]
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/agents/detail/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 2. GET /api/trading/messages/
Получает список сообщений, которыми обмениваются агенты.

**Параметры запроса:**
- `limit` (опционально) - количество сообщений (по умолчанию все)

**Ответ:**
```json
[
  {
    "id": "1",
    "timestamp": "2025-12-07T14:51:00Z",
    "from": "market",
    "to": "decision",
    "type": "MARKET_SNAPSHOT",
    "payload": {
      "symbol": "AAPL",
      "price": 150.32,
      "volume": 1000000
    }
  },
  {
    "id": "2",
    "timestamp": "2025-12-07T14:50:30Z",
    "from": "decision",
    "to": "execution",
    "type": "TRADE_DECISION",
    "payload": {
      "action": "BUY",
      "symbol": "AAPL",
      "confidence": 0.75
    }
  }
]
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/messages/?limit=50" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 3. POST /api/trading/agents/market-monitor/start/
Запускает агента Market Monitor.

**Пример запроса:**
```bash
curl -X POST "http://localhost:666/api/trading/agents/market-monitor/start/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json"
```

**Ответ:**
```json
{
  "status": "started",
  "message": "Market monitoring agent started"
}
```

---

### 4. POST /api/trading/agents/market-monitor/stop/
Останавливает агента Market Monitor.

**Пример запроса:**
```bash
curl -X POST "http://localhost:666/api/trading/agents/market-monitor/stop/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json"
```

---

### 5. POST /api/trading/agents/decision-maker/make-decision/
Запрашивает решение от агента Decision Maker для указанного символа.

**Тело запроса:**
```json
{
  "symbol": "AAPL"
}
```

**Пример запроса:**
```bash
curl -X POST "http://localhost:666/api/trading/agents/decision-maker/make-decision/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

**Ответ:**
```json
{
  "id": 1,
  "symbol": "AAPL",
  "decision": "BUY",
  "confidence": 75.5,
  "reasoning": "Strong upward trend detected",
  "timestamp": "2025-12-07T14:51:00Z"
}
```

---

### 6. POST /api/trading/agents/execution/execute/
Запускает агента Execution для выполнения сделок.

**Пример запроса:**
```bash
curl -X POST "http://localhost:666/api/trading/agents/execution/execute/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json"
```

**Ответ:**
```json
{
  "status": "executed",
  "message": "Execution agent executed trades"
}
```

---

## Использование на фронтенде

1. При загрузке страницы вызвать `/api/trading/agents/detail/` для получения статуса всех агентов
2. Вызвать `/api/trading/messages/` для отображения лога сообщений между агентами
3. Использовать POST эндпойнты для управления агентами (старт/стоп/выполнение действий)

Все эндпойнты требуют аутентификации через JWT токен.

