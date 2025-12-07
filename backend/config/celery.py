import os

from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Настройка периодических задач
app.conf.beat_schedule = {
    # Автоматический запуск ИИ агентов каждую минуту
    # Выполняет полный workflow: MarketMonitoring -> DecisionMaking -> Execution
    "ai-agents-workflow": {
        "task": "trading.tasks.run_ai_agents_workflow",
        "schedule": 60.0,  # Каждую минуту
    },
    # Отключено для уменьшения нагрузки на Yahoo Finance
    # "periodic-market-update": {
    #     "task": "trading.tasks.periodic_market_update",
    #     "schedule": 300.0,  # Каждые 5 минут (если нужно включить)
    # },
}
app.conf.timezone = "Asia/Almaty"

