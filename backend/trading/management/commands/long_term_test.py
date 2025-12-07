"""
Команда для долгосрочного тестирования агентов

Запускает непрерывное тестирование на заданное время или количество итераций.
"""
import time
import signal
import sys
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone

from trading.models import Symbol, Account, Position, Trade, TradingDecision
from trading.management.commands.test_agents import Command as TestAgentsCommand

User = get_user_model()


class Command(BaseCommand):
    help = "Долгосрочное тестирование агентов (непрерывная работа)"

    def __init__(self):
        super().__init__()
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        self.stdout.write(self.style.WARNING("\n\nПолучен сигнал остановки. Завершение тестирования..."))
        self.running = False

    def add_arguments(self, parser):
        parser.add_argument(
            "--user-id",
            type=int,
            help="ID пользователя для тестирования",
        )
        parser.add_argument(
            "--email",
            type=str,
            help="Email пользователя для тестирования",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="AAPL",
            help="Символ для тестирования",
        )
        parser.add_argument(
            "--duration",
            type=int,
            help="Длительность тестирования в минутах",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            help="Максимальное количество итераций",
        )
        parser.add_argument(
            "--interval",
            type=int,
            default=60,
            help="Интервал между итерациями в секундах (по умолчанию: 60)",
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Выполнять реальные сделки",
        )

    def handle(self, *args, **options):
        user_id = options.get("user_id")
        email = options.get("email")
        symbol_code = options.get("symbol", "AAPL")
        duration_minutes = options.get("duration")
        max_iterations = options.get("iterations")
        interval = options.get("interval", 60)
        execute_trades = options.get("execute", False)

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

        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(self.style.SUCCESS("ДОЛГОСРОЧНОЕ ТЕСТИРОВАНИЕ АГЕНТОВ"))
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(f"Пользователь: {user.email}")
        self.stdout.write(f"Символ: {symbol_code}")
        self.stdout.write(f"Интервал: {interval} секунд")
        self.stdout.write(f"Выполнение сделок: {'Да' if execute_trades else 'Нет'}")
        
        if duration_minutes:
            end_time = timezone.now() + timedelta(minutes=duration_minutes)
            self.stdout.write(f"Длительность: {duration_minutes} минут (до {end_time.strftime('%H:%M:%S')})")
        if max_iterations:
            self.stdout.write(f"Максимальное количество итераций: {max_iterations}")
        
        self.stdout.write("\nНажмите Ctrl+C для остановки\n")

        # Получаем начальную статистику
        account, _ = Account.objects.get_or_create(
            user=user,
            defaults={"balance": 10000.00, "free_cash": 10000.00}
        )
        initial_balance = account.balance
        initial_trades = Trade.objects.filter(user=user).count()
        initial_decisions = TradingDecision.objects.filter(user=user).count()
        initial_positions = Position.objects.filter(user=user, is_open=True).count()

        self.stdout.write(f"Начальный баланс: ${initial_balance}")
        self.stdout.write(f"Начальное количество сделок: {initial_trades}")
        self.stdout.write(f"Начальное количество решений: {initial_decisions}")
        self.stdout.write(f"Начальное количество позиций: {initial_positions}")
        self.stdout.write("")

        # Создаем команду для тестирования
        test_command = TestAgentsCommand()
        test_command.stdout = self.stdout
        test_command.style = self.style

        iteration = 0
        start_time = timezone.now()

        try:
            while self.running:
                iteration += 1
                current_time = timezone.now()

                # Проверяем условия остановки
                if duration_minutes and current_time >= end_time:
                    self.stdout.write(self.style.WARNING(f"\nДостигнуто время окончания тестирования"))
                    break

                if max_iterations and iteration > max_iterations:
                    self.stdout.write(self.style.WARNING(f"\nДостигнуто максимальное количество итераций"))
                    break

                elapsed = (current_time - start_time).total_seconds() / 60
                self.stdout.write(self.style.SUCCESS(f"\n{'='*70}"))
                self.stdout.write(f"Итерация #{iteration} | Время работы: {elapsed:.1f} минут")
                self.stdout.write(f"{'='*70}")

                # Запускаем одну итерацию тестирования
                try:
                    # Имитируем вызов test_agents с одной итерацией
                    from trading.agents import MarketMonitoringAgent, DecisionMakingAgent, ExecutionAgent
                    from trading.agents.integration import (
                        MarketAgentIntegration,
                        DecisionAgentIntegration,
                        ExecutionAgentIntegration
                    )
                    from trading.models import Symbol as SymbolModel

                    symbol = SymbolModel.objects.get(user=user, symbol=symbol_code)

                    # Market Agent
                    market_integration = MarketAgentIntegration(user)
                    market_agent = MarketMonitoringAgent(
                        ticker=symbol_code,
                        interval="1h",
                        period="1mo",
                        enable_cache=True,
                        request_delay=5.0,
                        max_retries=5,
                        backoff_factor=3.0
                    )
                    market_message = market_integration.process_and_save(
                        symbol=symbol,
                        market_agent=market_agent,
                        save_to_db=True
                    )

                    # Decision Agent
                    decision_integration = DecisionAgentIntegration(user)
                    decision_agent = DecisionMakingAgent(
                        model_type="random_forest",
                        risk_tolerance="medium",
                        min_confidence=0.55,
                        enable_ai=True
                    )
                    from trading.models import MarketData
                    latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()

                    decision = decision_integration.make_decision(
                        symbol=symbol,
                        market_data_obj=latest_data,
                        market_message=market_message,
                        decision_agent=decision_agent
                    )

                    self.stdout.write(f"Решение: {decision.decision} (уверенность: {decision.confidence}%)")

                    # Execution Agent
                    if decision.decision != "HOLD" and execute_trades:
                        execution_integration = ExecutionAgentIntegration(user)
                        execution_agent = ExecutionAgent(
                            execution_mode="simulated",
                            enable_slippage=True,
                        )

                        decision_dict = {
                            "action": decision.decision,
                            "ticker": symbol_code,
                            "quantity": decision.metadata.get("quantity", 1),
                            "price": decision.metadata.get("price", float(latest_data.price) if latest_data else 0.0),
                            "confidence": float(decision.confidence / 100) if decision.confidence else 0.5,
                            "timestamp": decision.created_at.isoformat(),
                            "reasoning": decision.reasoning,
                        }

                        execution_result = execution_agent.receive_decision(decision_dict)
                        if execution_result.get("status") == "executed":
                            trade = execution_integration.execute_trade(
                                symbol=symbol,
                                decision_obj=decision,
                                execution_agent=execution_agent,
                                execution_result=execution_result
                            )
                            if trade:
                                self.stdout.write(f"✓ Сделка выполнена: {trade.action} {trade.quantity} @ ${trade.price}")

                    # Показываем текущую статистику
                    account.refresh_from_db()
                    current_balance = account.balance
                    current_trades = Trade.objects.filter(user=user).count()
                    current_positions = Position.objects.filter(user=user, is_open=True).count()

                    balance_change = current_balance - initial_balance
                    self.stdout.write(f"\nСтатистика:")
                    self.stdout.write(f"  Баланс: ${current_balance} (изменение: ${balance_change:+.2f})")
                    self.stdout.write(f"  Сделок: {current_trades} (+{current_trades - initial_trades})")
                    self.stdout.write(f"  Позиций: {current_positions}")

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Ошибка в итерации {iteration}: {str(e)}"))
                    import traceback
                    self.stdout.write(traceback.format_exc())

                # Ожидание перед следующей итерацией
                if self.running and (not max_iterations or iteration < max_iterations):
                    if duration_minutes and timezone.now() < end_time:
                        self.stdout.write(f"\nОжидание {interval} секунд...")
                        for _ in range(interval):
                            if not self.running:
                                break
                            time.sleep(1)

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n\nПолучен сигнал прерывания"))

        # Финальная статистика
        self.stdout.write(self.style.SUCCESS(f"\n{'='*70}"))
        self.stdout.write(self.style.SUCCESS("ФИНАЛЬНАЯ СТАТИСТИКА"))
        self.stdout.write(self.style.SUCCESS(f"{'='*70}"))

        account.refresh_from_db()
        final_balance = account.balance
        final_trades = Trade.objects.filter(user=user).count()
        final_decisions = TradingDecision.objects.filter(user=user).count()
        final_positions = Position.objects.filter(user=user, is_open=True).count()

        total_time = (timezone.now() - start_time).total_seconds() / 60

        self.stdout.write(f"Время работы: {total_time:.1f} минут")
        self.stdout.write(f"Итераций выполнено: {iteration}")
        self.stdout.write(f"\nБаланс:")
        self.stdout.write(f"  Начальный: ${initial_balance}")
        self.stdout.write(f"  Финальный: ${final_balance}")
        self.stdout.write(f"  Изменение: ${final_balance - initial_balance:+.2f}")

        self.stdout.write(f"\nСделки:")
        self.stdout.write(f"  Начальное: {initial_trades}")
        self.stdout.write(f"  Финальное: {final_trades}")
        self.stdout.write(f"  Выполнено: {final_trades - initial_trades}")

        self.stdout.write(f"\nРешения:")
        self.stdout.write(f"  Всего: {final_decisions}")
        self.stdout.write(f"  Новых: {final_decisions - initial_decisions}")

        self.stdout.write(f"\nПозиции:")
        self.stdout.write(f"  Открыто: {final_positions}")

        self.stdout.write(self.style.SUCCESS("\n✓ Долгосрочное тестирование завершено!"))

