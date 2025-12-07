"""
Команда для мониторинга работы ИИ агентов в реальном времени.
Показывает последние решения, сделки и статистику.
"""
import time
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db.models import Count, Sum, Q
from trading.models import (
    UserSettings, Symbol, TradingDecision, Trade, AgentStatus, Account, Position
)
from decimal import Decimal

User = get_user_model()


class Command(BaseCommand):
    help = "Мониторинг работы ИИ агентов в реальном времени"

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
            "--watch",
            action="store_true",
            help="Режим наблюдения (обновление каждые 5 секунд)",
        )
        parser.add_argument(
            "--interval",
            type=int,
            default=5,
            help="Интервал обновления в секундах (по умолчанию: 5)",
        )

    def handle(self, *args, **options):
        email = options.get("email")
        user_id = options.get("user-id")
        watch_mode = options.get("watch", False)
        interval = options.get("interval", 5)

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

        try:
            user_settings = UserSettings.objects.get(user=user)
        except UserSettings.DoesNotExist:
            self.stdout.write(self.style.ERROR("Настройки пользователя не найдены. Запустите start_auto_trading сначала."))
            return

        if watch_mode:
            self.stdout.write(self.style.SUCCESS(f"\n{'='*70}"))
            self.stdout.write(self.style.SUCCESS("РЕЖИМ НАБЛЮДЕНИЯ ЗА АГЕНТАМИ"))
            self.stdout.write(self.style.SUCCESS(f"{'='*70}"))
            self.stdout.write(f"Пользователь: {user.email}")
            self.stdout.write(f"Обновление каждые {interval} секунд")
            self.stdout.write("Нажмите Ctrl+C для выхода\n")

            try:
                while True:
                    self._display_status(user, user_settings)
                    time.sleep(interval)
                    # Очищаем экран (работает в большинстве терминалов)
                    self.stdout.write("\033[2J\033[H")  # ANSI escape codes для очистки
            except KeyboardInterrupt:
                self.stdout.write(self.style.SUCCESS("\n\nМониторинг остановлен"))
        else:
            self._display_status(user, user_settings)

    def _display_status(self, user, user_settings):
        """Отображает текущий статус агентов"""
        now = timezone.now()

        # Статус настроек
        self.stdout.write(self.style.SUCCESS(f"\n{'='*70}"))
        self.stdout.write(self.style.SUCCESS("СТАТУС АВТОМАТИЧЕСКОЙ ТОРГОВЛИ"))
        self.stdout.write(self.style.SUCCESS(f"{'='*70}"))
        self.stdout.write(f"Пользователь: {user.email}")
        self.stdout.write(f"Статус: {user_settings.get_status_display()}")
        self.stdout.write(f"Символ: {user_settings.symbol}")
        self.stdout.write(f"Время: {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # Статус агентов
        self.stdout.write(self.style.SUCCESS(f"\n{'─'*70}"))
        self.stdout.write(self.style.SUCCESS("СТАТУС АГЕНТОВ"))
        self.stdout.write(self.style.SUCCESS(f"{'─'*70}"))

        agent_types = ["MARKET_MONITOR", "DECISION_MAKER", "EXECUTION"]
        for agent_type in agent_types:
            try:
                status_obj = AgentStatus.objects.get(user=user, agent_type=agent_type)
                last_activity = status_obj.last_activity.strftime('%H:%M:%S') if status_obj.last_activity else "N/A"
                self.stdout.write(
                    f"  {agent_type:20} | {status_obj.get_status_display():10} | "
                    f"Последняя активность: {last_activity}"
                )
            except AgentStatus.DoesNotExist:
                self.stdout.write(f"  {agent_type:20} | Не найден")

        # Статистика решений
        self.stdout.write(self.style.SUCCESS(f"\n{'─'*70}"))
        self.stdout.write(self.style.SUCCESS("СТАТИСТИКА РЕШЕНИЙ (последние 24 часа)"))
        self.stdout.write(self.style.SUCCESS(f"{'─'*70}"))

        from datetime import timedelta
        last_24h = now - timedelta(hours=24)

        decisions_24h = TradingDecision.objects.filter(
            user=user,
            created_at__gte=last_24h
        )

        total_decisions = decisions_24h.count()
        buy_decisions = decisions_24h.filter(decision="BUY").count()
        sell_decisions = decisions_24h.filter(decision="SELL").count()
        hold_decisions = decisions_24h.filter(decision="HOLD").count()

        self.stdout.write(f"  Всего решений: {total_decisions}")
        self.stdout.write(f"    - BUY:  {buy_decisions}")
        self.stdout.write(f"    - SELL: {sell_decisions}")
        self.stdout.write(f"    - HOLD: {hold_decisions}")

        # Последние решения
        last_decisions = decisions_24h.order_by("-created_at")[:5]
        if last_decisions.exists():
            self.stdout.write(self.style.SUCCESS(f"\n  Последние решения:"))
            for decision in last_decisions:
                time_str = decision.created_at.strftime('%H:%M:%S')
                self.stdout.write(
                    f"    {time_str} | {decision.decision:4} | "
                    f"Уверенность: {decision.confidence}% | {decision.symbol.symbol}"
                )

        # Статистика сделок
        self.stdout.write(self.style.SUCCESS(f"\n{'─'*70}"))
        self.stdout.write(self.style.SUCCESS("СТАТИСТИКА СДЕЛОК"))
        self.stdout.write(self.style.SUCCESS(f"{'─'*70}"))

        trades_24h = Trade.objects.filter(
            user=user,
            executed_at__gte=last_24h
        )

        total_trades = trades_24h.count()
        buy_trades = trades_24h.filter(action="BUY").count()
        sell_trades = trades_24h.filter(action="SELL").count()

        # PnL статистика
        completed_trades = trades_24h.filter(action="SELL", pnl__isnull=False)
        total_pnl = completed_trades.aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")
        winning_trades = completed_trades.filter(pnl__gt=0).count()
        losing_trades = completed_trades.filter(pnl__lt=0).count()

        self.stdout.write(f"  Всего сделок (24ч): {total_trades}")
        self.stdout.write(f"    - BUY:  {buy_trades}")
        self.stdout.write(f"    - SELL: {sell_trades}")
        self.stdout.write(f"\n  Завершенные сделки (SELL с PnL): {completed_trades.count()}")
        self.stdout.write(f"    - Прибыльных: {winning_trades}")
        self.stdout.write(f"    - Убыточных: {losing_trades}")
        self.stdout.write(f"    - Общий PnL: ${total_pnl:+.2f}")

        # Последние сделки
        last_trades = trades_24h.order_by("-executed_at")[:5]
        if last_trades.exists():
            self.stdout.write(self.style.SUCCESS(f"\n  Последние сделки:"))
            for trade in last_trades:
                time_str = trade.executed_at.strftime('%H:%M:%S')
                pnl_str = f"PnL: ${trade.pnl:+.2f}" if trade.pnl is not None else "PnL: N/A"
                self.stdout.write(
                    f"    {time_str} | {trade.action:4} | "
                    f"{trade.quantity} {trade.symbol.symbol} @ ${trade.price} | {pnl_str}"
                )

        # Портфель
        self.stdout.write(self.style.SUCCESS(f"\n{'─'*70}"))
        self.stdout.write(self.style.SUCCESS("ПОРТФЕЛЬ"))
        self.stdout.write(self.style.SUCCESS(f"{'─'*70}"))

        try:
            account = Account.objects.get(user=user)
            self.stdout.write(f"  Баланс: ${account.balance}")
            self.stdout.write(f"  Свободные средства: ${account.free_cash}")
            self.stdout.write(f"  Использованная маржа: ${account.used_margin}")
        except Account.DoesNotExist:
            self.stdout.write("  Счет не найден")

        # Открытые позиции
        open_positions = Position.objects.filter(user=user, is_open=True)
        if open_positions.exists():
            self.stdout.write(f"\n  Открытые позиции: {open_positions.count()}")
            for pos in open_positions[:3]:
                pnl = pos.pnl
                pnl_str = f"PnL: ${pnl:+.2f}" if pnl else "PnL: N/A"
                self.stdout.write(
                    f"    {pos.symbol.symbol:10} | {pos.quantity:8} @ ${pos.entry_price:8.2f} | "
                    f"Текущая: ${pos.current_price or 0:8.2f} | {pnl_str}"
                )
        else:
            self.stdout.write("\n  Открытых позиций нет")

        # Режим исследования
        completed_trades_count = Trade.objects.filter(
            user=user,
            action="SELL",
            pnl__isnull=False
        ).count()

        if completed_trades_count < 10:
            self.stdout.write(self.style.WARNING(f"\n{'─'*70}"))
            self.stdout.write(self.style.WARNING("РЕЖИМ ИССЛЕДОВАНИЯ АКТИВЕН"))
            self.stdout.write(self.style.WARNING(f"{'─'*70}"))
            self.stdout.write(
                f"  Завершенных сделок: {completed_trades_count}/10\n"
                f"  Порог уверенности снижен до 35% для сбора данных\n"
                f"  После накопления 10+ сделок переключится на нормальный режим"
            )

        self.stdout.write(self.style.SUCCESS(f"\n{'='*70}\n"))

