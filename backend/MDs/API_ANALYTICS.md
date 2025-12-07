# Analytics Page API

## Описание
Страница аналитики предоставляет детальную статистику торговой деятельности: метрики производительности, графики P&L и разбивку по периодам.

## Эндпойнты

### 1. GET /api/trading/analytics/performance-metrics/
Получает основные метрики производительности торговли.

**Ответ:**
```json
{
  "totalReturn": 12.45,
  "totalReturnPercent": 12.45,
  "sharpeRatio": 1.25,
  "winRate": 68.5,
  "maxDrawdown": -150.00,
  "maxDrawdownPercent": -1.5,
  "averageWin": 45.30,
  "averageLoss": -25.20,
  "profitFactor": 1.8,
  "totalTrades": 45,
  "winningTrades": 31,
  "losingTrades": 14
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/analytics/performance-metrics/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 2. GET /api/trading/analytics/pnl-curve/
Получает данные для графика P&L за последние 30 дней.

**Ответ:**
```json
{
  "data": [
    {
      "date": "2025-12-01",
      "pnl": 0.00
    },
    {
      "date": "2025-12-02",
      "pnl": 45.50
    },
    {
      "date": "2025-12-07",
      "pnl": 245.50
    }
  ],
  "totalPnL": 245.50
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/analytics/pnl-curve/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

### 3. GET /api/trading/analytics/monthly-breakdown/
Получает разбивку P&L по периодам: сегодня, вчера, эта неделя, и по месяцам.

**Ответ:**
```json
{
  "today": 45.50,
  "yesterday": -12.30,
  "thisWeek": 125.80,
  "thisMonth": 245.50,
  "lastMonth": 180.20,
  "monthly": [
    {
      "month": "2025-11",
      "pnl": 180.20
    },
    {
      "month": "2025-12",
      "pnl": 245.50
    }
  ]
}
```

**Пример запроса:**
```bash
curl -X GET "http://localhost:666/api/trading/analytics/monthly-breakdown/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Использование на фронтенде

1. При загрузке страницы вызвать `/api/trading/analytics/performance-metrics/` для отображения основных метрик
2. Вызвать `/api/trading/analytics/pnl-curve/` для построения графика P&L
3. Вызвать `/api/trading/analytics/monthly-breakdown/` для отображения разбивки по периодам

Все эндпойнты требуют аутентификации через JWT токен.

