"""
Команда для проверки работы Celery Beat и задач.
"""
from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask
from celery import current_app


class Command(BaseCommand):
    help = "Проверяет работу Celery Beat и периодических задач"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(self.style.SUCCESS("ПРОВЕРКА CELERY BEAT"))
        self.stdout.write(self.style.SUCCESS("="*70))
        
        # Проверяем периодические задачи
        self.stdout.write("\n📅 ПЕРИОДИЧЕСКИЕ ЗАДАЧИ:")
        try:
            tasks = PeriodicTask.objects.all()
            if tasks.exists():
                for task in tasks:
                    status = "✓ Включена" if task.enabled else "✗ Отключена"
                    self.stdout.write(f"  {task.name:40} | {status}")
                    self.stdout.write(f"    Расписание: {task.schedule}")
                    self.stdout.write(f"    Задача: {task.task}")
            else:
                self.stdout.write(self.style.WARNING("  Нет периодических задач"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Ошибка при проверке задач: {e}"))
        
        # Проверяем зарегистрированные задачи
        self.stdout.write("\n📋 ЗАРЕГИСТРИРОВАННЫЕ ЗАДАЧИ:")
        try:
            registered_tasks = list(current_app.tasks.keys())
            ai_workflow_task = "trading.tasks.run_ai_agents_workflow"
            if ai_workflow_task in registered_tasks:
                self.stdout.write(self.style.SUCCESS(f"  ✓ {ai_workflow_task} зарегистрирована"))
            else:
                self.stdout.write(self.style.ERROR(f"  ✗ {ai_workflow_task} НЕ зарегистрирована"))
                self.stdout.write(f"  Доступные задачи: {len(registered_tasks)}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Ошибка при проверке зарегистрированных задач: {e}"))
        
        # Проверяем настройки Celery
        self.stdout.write("\n⚙️  НАСТРОЙКИ CELERY:")
        try:
            from django.conf import settings
            broker_url = getattr(settings, "CELERY_BROKER_URL", "Не настроен")
            result_backend = getattr(settings, "CELERY_RESULT_BACKEND", "Не настроен")
            self.stdout.write(f"  Broker: {broker_url}")
            self.stdout.write(f"  Result Backend: {result_backend}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Ошибка при проверке настроек: {e}"))
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write("\n💡 СОВЕТЫ:")
        self.stdout.write("  1. Проверьте, что Celery Beat запущен:")
        self.stdout.write("     docker compose ps | grep celery-beat")
        self.stdout.write("  2. Проверьте логи Celery Beat:")
        self.stdout.write("     docker compose logs celery-beat | tail -50")
        self.stdout.write("  3. Проверьте логи задачи:")
        self.stdout.write("     docker compose logs backend | grep 'ai agents workflow'")
        self.stdout.write("="*70)

