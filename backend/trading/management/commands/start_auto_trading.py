"""
Команда для включения автоматической торговли для пользователя.
Устанавливает статус "running" в UserSettings, чтобы Celery задача
run_ai_agents_workflow запускала агентов каждую минуту.
"""
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from trading.models import UserSettings, Symbol, Account
from decimal import Decimal

User = get_user_model()


class Command(BaseCommand):
    help = "Включает автоматическую торговлю для пользователя (агенты будут работать каждую минуту)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--email",
            type=str,
            help="Email пользователя",
        )
        parser.add_argument(
            "--user-id",
            type=int,
            help="ID пользователя",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Символ для торговли (по умолчанию: BTCUSDT)",
        )

    def handle(self, *args, **options):
        email = options.get("email")
        user_id = options.get("user-id")
        symbol_code = options.get("symbol", "BTCUSDT")

        # Получаем пользователя
        if user_id:
            try:
                user = User.objects.get(id=user_id)
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Пользователь с ID {user_id} не найден"))
                return
        elif email:
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Пользователь с email {email} не найден"))
                return
        else:
            user = User.objects.first()
            if not user:
                self.stdout.write(self.style.ERROR("Нет пользователей в системе"))
                return

        self.stdout.write(self.style.SUCCESS(f"Пользователь: {user.email}"))

        # Получаем или создаем настройки
        user_settings, created = UserSettings.objects.get_or_create(
            user=user,
            defaults={
                "status": "running",
                "symbol": symbol_code,
                "timeframe": "1h",
                "risk_level": "medium",
                "confidence_threshold": Decimal("0.55"),
            }
        )

        if not created:
            user_settings.status = "running"
            user_settings.symbol = symbol_code
            user_settings.save()

        # Получаем или создаем символ
        symbol, symbol_created = Symbol.objects.get_or_create(
            user=user,
            symbol=symbol_code,
            defaults={
                "name": f"{symbol_code}",
                "is_active": True,
            }
        )

        if symbol_created:
            self.stdout.write(self.style.SUCCESS(f"✓ Создан символ: {symbol_code}"))
        else:
            symbol.is_active = True
            symbol.save()
            self.stdout.write(self.style.SUCCESS(f"✓ Активирован символ: {symbol_code}"))

        # Получаем или создаем счет
        account, account_created = Account.objects.get_or_create(
            user=user,
            defaults={
                "balance": Decimal("10000.00"),
                "free_cash": Decimal("10000.00"),
                "initial_balance": Decimal("10000.00"),
            }
        )

        if account_created:
            self.stdout.write(self.style.SUCCESS(f"✓ Создан счет с балансом: ${account.balance}"))
        else:
            self.stdout.write(self.style.SUCCESS(f"✓ Счет существует: ${account.balance}"))

        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("АВТОМАТИЧЕСКАЯ ТОРГОВЛЯ ВКЛЮЧЕНА"))
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(f"\nСтатус: {user_settings.get_status_display()}")
        self.stdout.write(f"Символ: {user_settings.symbol}")
        self.stdout.write(f"Таймфрейм: {user_settings.timeframe}")
        self.stdout.write(f"Уровень риска: {user_settings.get_risk_level_display()}")
        self.stdout.write(f"Порог уверенности: {user_settings.confidence_threshold}%")
        
        self.stdout.write(self.style.SUCCESS("\n✓ Агенты будут работать автоматически каждую минуту!"))
        self.stdout.write("\nДля мониторинга используйте:")
        self.stdout.write("  docker compose logs -f backend | grep -i 'ai agents workflow\\|decision\\|trade executed'")
        self.stdout.write("\nИли используйте команду:")
        self.stdout.write(f"  python manage.py monitor_agents --email {user.email}")

