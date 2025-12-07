# Тестирование AI Агентов

Руководство по тестированию работы AI агентов торговой системы.

## Быстрое тестирование (одна итерация)

Протестировать один полный цикл работы агентов:

```bash
docker compose exec backend python manage.py test_agents --email your@email.com --symbol AAPL
```

### Параметры:

- `--user-id ID` - ID пользователя
- `--email EMAIL` - Email пользователя
- `--symbol SYMBOL` - Символ для тестирования (по умолчанию: AAPL)
- `--iterations N` - Количество итераций (по умолчанию: 1)
- `--delay SECONDS` - Задержка между итерациями (по умолчанию: 5)
- `--execute` - Выполнять реальные сделки (по умолчанию только принимает решения)

### Примеры:

```bash
# Тест с одним пользователем и символом AAPL
docker compose exec backend python manage.py test_agents --email user@example.com

# Тест с несколькими итерациями
docker compose exec backend python manage.py test_agents --email user@example.com --iterations 5

# Тест с выполнением сделок
docker compose exec backend python manage.py test_agents --email user@example.com --execute

# Тест с криптовалютой
docker compose exec backend python manage.py test_agents --email user@example.com --symbol BTCUSDT --execute
```

## Долгосрочное тестирование

Запустить непрерывное тестирование на заданное время:

```bash
docker compose exec backend python manage.py long_term_test --email your@email.com --symbol AAPL --duration 60 --execute
```

### Параметры:

- `--user-id ID` - ID пользователя
- `--email EMAIL` - Email пользователя
- `--symbol SYMBOL` - Символ для тестирования (по умолчанию: AAPL)
- `--duration MINUTES` - Длительность тестирования в минутах
- `--iterations N` - Максимальное количество итераций
- `--interval SECONDS` - Интервал между итерациями (по умолчанию: 60)
- `--execute` - Выполнять реальные сделки

### Примеры:

```bash
# Тест на 1 час с интервалом 1 минута
docker compose exec backend python manage.py long_term_test \
  --email user@example.com \
  --symbol AAPL \
  --duration 60 \
  --interval 60 \
  --execute

# Тест на 100 итераций
docker compose exec backend python manage.py long_term_test \
  --email user@example.com \
  --symbol BTCUSDT \
  --iterations 100 \
  --interval 30 \
  --execute

# Тест на 24 часа (фоновый режим)
docker compose exec -d backend python manage.py long_term_test \
  --email user@example.com \
  --symbol AAPL \
  --duration 1440 \
  --interval 300 \
  --execute
```

## Что проверяется

### 1. MarketMonitoringAgent
- ✅ Получение данных рынка через yfinance/Bybit
- ✅ Расчет технических индикаторов (RSI, MACD, Bollinger Bands, SMA)
- ✅ Анализ рыночных условий
- ✅ Сохранение данных в БД (MarketData)

### 2. DecisionMakingAgent
- ✅ Принятие решений на основе данных рынка
- ✅ Использование AI моделей (Random Forest, Gradient Boosting)
- ✅ Применение риск-менеджмента
- ✅ Сохранение решений в БД (TradingDecision)
- ✅ Логирование действий

### 3. ExecutionAgent
- ✅ Выполнение сделок (BUY/SELL)
- ✅ Симуляция проскальзывания и комиссий
- ✅ Обновление позиций (Position)
- ✅ Обновление счета (Account)
- ✅ Расчет P&L
- ✅ Сохранение сделок в БД (Trade)

### 4. Интеграция
- ✅ Сообщения между агентами (Message)
- ✅ Логирование (AgentLog)
- ✅ Обновление статусов (AgentStatus)
- ✅ Использование настроек пользователя (UserSettings)

## Мониторинг результатов

### Через Django Admin

```bash
# Создать суперпользователя (если еще нет)
docker compose exec backend python manage.py createsuperuser
```

Затем откройте http://localhost:666/admin/ и проверьте:
- Trading Decisions - решения агентов
- Trades - выполненные сделки
- Positions - открытые позиции
- Messages - сообщения между агентами
- Agent Logs - логи агентов

### Через API

```bash
# Получить статус агентов
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:666/api/trading/agents/detail/

# Получить последние решения
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:666/api/trading/decisions/

# Получить портфель
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:666/api/trading/portfolio/

# Получить метрики производительности
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:666/api/trading/performance-metrics/
```

### Через логи Docker

```bash
# Логи backend
docker compose logs -f backend

# Логи celery
docker compose logs -f celery

# Все логи
docker compose logs -f
```

## Проверка работоспособности

### Быстрая проверка (30 секунд)

```bash
docker compose exec backend python manage.py test_agents \
  --email user@example.com \
  --symbol AAPL \
  --iterations 1
```

Ожидаемый результат:
- ✅ Данные получены
- ✅ Решение принято (BUY/SELL/HOLD)
- ✅ Логи созданы
- ✅ Сообщения отправлены

### Средняя проверка (5 минут)

```bash
docker compose exec backend python manage.py test_agents \
  --email user@example.com \
  --symbol AAPL \
  --iterations 5 \
  --delay 60 \
  --execute
```

Ожидаемый результат:
- ✅ Несколько решений принято
- ✅ Сделки выполнены (если не все HOLD)
- ✅ Позиции созданы/обновлены
- ✅ Баланс изменен
- ✅ P&L рассчитан

### Долгосрочная проверка (1 час+)

```bash
docker compose exec backend python manage.py long_term_test \
  --email user@example.com \
  --symbol AAPL \
  --duration 60 \
  --interval 60 \
  --execute
```

Ожидаемый результат:
- ✅ Система работает стабильно
- ✅ Нет утечек памяти
- ✅ База данных обновляется корректно
- ✅ Агенты работают согласованно
- ✅ Статистика накапливается правильно

## Типичные проблемы

### Ошибка "Symbol not found"
```bash
# Добавить символ пользователю
docker compose exec backend python manage.py add_default_symbols --email user@example.com
```

### Ошибка "No market data available"
- Проверьте интернет-соединение
- Убедитесь, что символ существует (AAPL, BTCUSDT и т.д.)
- Проверьте логи: `docker compose logs backend | grep -i error`

### Ошибка "Insufficient funds"
- Начальный баланс: $10,000
- Проверьте баланс: `curl -H "Authorization: Bearer TOKEN" http://localhost:666/api/trading/portfolio/`

### Агенты не принимают решения
- Проверьте настройки пользователя (UserSettings)
- Убедитесь, что confidence_threshold не слишком высокий
- Проверьте логи агентов в Django Admin

## Автоматизация тестирования

### Скрипт для ежедневного тестирования

Создайте файл `test_daily.sh`:

```bash
#!/bin/bash
EMAIL="user@example.com"
SYMBOL="AAPL"

echo "Запуск ежедневного теста агентов..."
docker compose exec -d backend python manage.py long_term_test \
  --email "$EMAIL" \
  --symbol "$SYMBOL" \
  --duration 1440 \
  --interval 300 \
  --execute

echo "Тест запущен в фоновом режиме"
echo "Проверьте логи: docker compose logs -f backend"
```

Запуск:
```bash
chmod +x test_daily.sh
./test_daily.sh
```

## Анализ результатов

После тестирования проверьте:

1. **Баланс** - должен изменяться при выполнении сделок
2. **Позиции** - открываются при BUY, закрываются при SELL
3. **P&L** - рассчитывается корректно
4. **Логи** - нет критических ошибок
5. **Сообщения** - агенты общаются между собой
6. **Статусы** - агенты переходят в правильные состояния

## Рекомендации

1. **Начните с быстрого теста** без `--execute` для проверки логики
2. **Затем протестируйте с `--execute`** для проверки выполнения сделок
3. **Используйте долгосрочное тестирование** для проверки стабильности
4. **Мониторьте логи** во время тестирования
5. **Проверяйте базу данных** после тестирования

## Вопросы?

Если что-то не работает:
1. Проверьте логи: `docker compose logs backend celery`
2. Проверьте базу данных: `docker compose exec postgres psql -U postgres -d trading_db`
3. Проверьте статус контейнеров: `docker compose ps`

