import os

from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Настройка периодических задач
app.conf.beat_schedule = {
    "periodic-market-update": {
        "task": "trading.tasks.periodic_market_update",
        "schedule": 60.0,  # Каждую минуту
    },
}
app.conf.timezone = "Asia/Almaty"

