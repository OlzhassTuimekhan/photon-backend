import os

from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Настройка периодических задач
# ВАЖНО: periodic_market_update отключен, т.к. использует MarketDataService (yfinance)
# Это создавало слишком много запросов к Yahoo Finance и вызывало блокировку (429)
# Вместо этого используем MarketMonitoringAgent через API endpoints
# Данные получаются только когда пользователь явно запрашивает через агентов
app.conf.beat_schedule = {
    # Отключено для уменьшения нагрузки на Yahoo Finance
    # "periodic-market-update": {
    #     "task": "trading.tasks.periodic_market_update",
    #     "schedule": 300.0,  # Каждые 5 минут (если нужно включить)
    # },
}
app.conf.timezone = "Asia/Almaty"

