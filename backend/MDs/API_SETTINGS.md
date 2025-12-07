# Settings Page API

## Описание
Страница настроек позволяет пользователю настраивать параметры симуляции торговли, выбор источника данных, модель принятия решений и параметры управления рисками.

## Эндпойнты

### 1. GET /api/trading/settings/
Получает текущие настройки пользователя.

**Ответ:**
```json
{
  "status": "stopped",
  "speed": "1.0",
  "symbol": "AAPL",
  "timeframe": "1h",
  "dataProvider": "Yahoo Finance",
  "historyLength": "Last 1 year",
  "modelType": "Random Forest",
  "predictionHorizon": "1 hour",
  "confidenceThreshold": "0.55",
  "initialBalance": "10000.00",
  "maxPositionSize": "50.0",
  "riskLevel": "medium",
  "stopLoss": "-2.0",
  "takeProfit": "5.0",
  "maxLeverage": "1.0"
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/settings/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 2. PUT /api/trading/settings/
Полностью обновляет настройки пользователя.

**Тело запроса:**
```json
{
  "status": "running",
  "speed": "2.0",
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "dataProvider": "Bybit",
  "historyLength": "Last 6 months",
  "modelType": "LSTM",
  "predictionHorizon": "4 hours",
  "confidenceThreshold": "0.60",
  "initialBalance": "10000.00",
  "maxPositionSize": "30.0",
  "riskLevel": "low",
  "stopLoss": "-1.5",
  "takeProfit": "3.0",
  "maxLeverage": "1.0"
}
```

**Пример запроса:**
```bash
curl -X PUT "http://localhost:666/api/trading/settings/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "running",
    "speed": "2.0",
    "symbol": "BTCUSDT",
    "timeframe": "4h"
  }'
```

**Ответ:** Возвращает обновленные настройки в том же формате, что и GET.

---

### 3. PATCH /api/trading/settings/
Частично обновляет настройки пользователя (аналогично PUT).

**Тело запроса:**
```json
{
  "status": "running",
  "symbol": "MSFT"
}
```

**Пример запроса:**
```bash
curl -X PATCH "http://localhost:666/api/trading/settings/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "running",
    "symbol": "MSFT"
  }'
```

**Ответ:** Возвращает обновленные настройки.

---

## Поля настроек

**Симуляция:**
- `status` - статус симуляции: "stopped", "running", "paused"
- `speed` - скорость симуляции (множитель времени)
- `symbol` - торговый символ (например, "AAPL", "BTCUSDT")
- `timeframe` - таймфрейм: "15m", "1h", "4h", "1d"

**Источник данных:**
- `dataProvider` - провайдер данных: "Yahoo Finance", "Bybit"
- `historyLength` - длина истории: "Last 1 year", "Last 6 months", "Last 3 months"

**Модель:**
- `modelType` - тип модели: "Random Forest", "LSTM", "Transformer"
- `predictionHorizon` - горизонт прогноза: "1 hour", "4 hours", "1 day"
- `confidenceThreshold` - порог уверенности (0.0 - 1.0)

**Торговля:**
- `initialBalance` - начальный баланс
- `maxPositionSize` - максимальный размер позиции (%)
- `riskLevel` - уровень риска: "low", "medium", "high"
- `stopLoss` - стоп-лосс (%)
- `takeProfit` - тейк-профит (%)
- `maxLeverage` - максимальное плечо

---

## Использование на фронтенде

1. При загрузке страницы вызвать GET `/api/trading/settings/` для получения текущих настроек
2. При изменении настроек использовать PUT или PATCH для обновления
3. Все поля опциональны при PATCH, обязательны при PUT

Все эндпойнты требуют аутентификации через JWT токен.

