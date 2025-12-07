"""
Команда для тестирования работы AI агентов

Тестирует полный цикл:
1. MarketMonitoringAgent - получение данных
2. DecisionMakingAgent - принятие решения
3. ExecutionAgent - выполнение сделки
"""
import time
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone

from trading.models import Symbol, TradingDecision, Trade, Position, Account, AgentStatus, Message, AgentLog
from trading.agents import MarketMonitoringAgent, DecisionMakingAgent, ExecutionAgent
from trading.agents.integration import (
    MarketAgentIntegration,
    DecisionAgentIntegration,
    ExecutionAgentIntegration
)

User = get_user_model()


class Command(BaseCommand):
    help = "Тестирует работу AI агентов (Market → Decision → Execution)"

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
            help="Символ для тестирования (по умолчанию: AAPL)",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=1,
            help="Количество итераций тестирования (по умолчанию: 1)",
        )
        parser.add_argument(
            "--delay",
            type=int,
            default=5,
            help="Задержка между итерациями в секундах (по умолчанию: 5)",
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Выполнять реальные сделки (по умолчанию только принимает решения)",
        )

    def handle(self, *args, **options):
        user_id = options.get("user_id")
        email = options.get("email")
        symbol_code = options.get("symbol", "AAPL")
        iterations = options.get("iterations", 1)
        delay = options.get("delay", 5)
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
            # Используем первого пользователя
            user = User.objects.first()
            if not user:
                self.stdout.write(self.style.ERROR("Нет пользователей в системе. Создайте пользователя сначала."))
                return

        self.stdout.write(self.style.SUCCESS(f"Тестирование для пользователя: {user.email}"))
        self.stdout.write(f"Символ: {symbol_code}")
        self.stdout.write(f"Итераций: {iterations}")
        self.stdout.write(f"Выполнение сделок: {'Да' if execute_trades else 'Нет (только решения)'}")
        self.stdout.write("")

        # Получаем или создаем символ
        symbol, created = Symbol.objects.get_or_create(
            user=user,
            symbol=symbol_code,
            defaults={
                "name": f"Test {symbol_code}",
                "is_active": True,
            }
        )

        if created:
            self.stdout.write(self.style.SUCCESS(f"Создан символ: {symbol_code}"))

        # Получаем начальный баланс
        account, _ = Account.objects.get_or_create(
            user=user,
            defaults={"balance": Decimal("10000.00"), "free_cash": Decimal("10000.00")}
        )
        initial_balance = account.balance
        self.stdout.write(f"Начальный баланс: ${initial_balance}")

        # Статистика
        stats = {
            "decisions": 0,
            "trades": 0,
            "errors": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
        }

        for iteration in range(1, iterations + 1):
            self.stdout.write(self.style.WARNING(f"\n{'='*60}"))
            self.stdout.write(self.style.WARNING(f"Итерация {iteration}/{iterations}"))
            self.stdout.write(self.style.WARNING(f"{'='*60}"))

            try:
                # Шаг 1: MarketMonitoringAgent
                self.stdout.write("\n[1/3] MarketMonitoringAgent: Получение данных рынка...")
                market_integration = MarketAgentIntegration(user)
                market_agent = MarketMonitoringAgent(
                    ticker=symbol_code,
                    interval="1h",
                    period="1mo",
                    enable_cache=True,
                    request_delay=5.0,  # Увеличенная задержка для обхода блокировок
                    max_retries=5,
                    backoff_factor=3.0
                )

                market_message = market_integration.process_and_save(
                    symbol=symbol,
                    market_agent=market_agent,
                    save_to_db=True
                )

                from trading.models import MarketData
                latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()

                self.stdout.write(self.style.SUCCESS(f"✓ Данные получены: {market_message.get('ohlcv', {}).get('close', 'N/A')}"))

                # Шаг 2: DecisionMakingAgent
                self.stdout.write("\n[2/3] DecisionMakingAgent: Принятие решения...")
                decision_integration = DecisionAgentIntegration(user)

                decision_agent = DecisionMakingAgent(
                    model_type="random_forest",
                    risk_tolerance="medium",
                    min_confidence=0.55,
                    enable_ai=True,
                    use_historical_training=True,  # Используем реальные данные для обучения
                    training_ticker=symbol_code,  # Используем тот же тикер (для крипты сработает Bybit fallback)
                    training_period="1mo"  # Месяц данных достаточно для обучения
                )

                decision = decision_integration.make_decision(
                    symbol=symbol,
                    market_data_obj=latest_data,
                    market_message=market_message,
                    decision_agent=decision_agent
                )

                stats["decisions"] += 1
                decision_action = decision.decision
                if decision_action == "BUY":
                    stats["buy_decisions"] += 1
                elif decision_action == "SELL":
                    stats["sell_decisions"] += 1
                else:
                    stats["hold_decisions"] += 1

                self.stdout.write(self.style.SUCCESS(
                    f"✓ Решение: {decision_action} "
                    f"(уверенность: {decision.confidence}%, "
                    f"рассуждение: {decision.reasoning[:50]}...)"
                ))

                # Шаг 3: ExecutionAgent (если не HOLD и execute_trades=True)
                if decision_action != "HOLD" and execute_trades:
                    self.stdout.write(f"\n[3/3] ExecutionAgent: Выполнение сделки {decision_action}...")
                    execution_integration = ExecutionAgentIntegration(user)

                    execution_agent = ExecutionAgent(
                        execution_mode="simulated",
                        enable_slippage=True,
                        slippage_factor=0.001,
                        commission_rate=0.001,
                    )

                    decision_dict = {
                        "action": decision_action,
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
                            stats["trades"] += 1
                            self.stdout.write(self.style.SUCCESS(
                                f"✓ Сделка выполнена: {trade.action} {trade.quantity} @ ${trade.price}"
                            ))
                        else:
                            self.stdout.write(self.style.WARNING("⚠ Сделка не выполнена (недостаточно средств/позиции)"))
                    else:
                        self.stdout.write(self.style.WARNING(
                            f"⚠ Сделка отклонена: {execution_result.get('message', 'Unknown reason')}"
                        ))
                elif decision_action == "HOLD":
                    self.stdout.write(self.style.SUCCESS("\n[3/3] ExecutionAgent: Пропущено (решение HOLD)"))
                else:
                    self.stdout.write(self.style.SUCCESS("\n[3/3] ExecutionAgent: Пропущено (--execute не указан)"))

                # Обновляем баланс
                account.refresh_from_db()
                current_balance = account.balance
                balance_change = current_balance - initial_balance

                self.stdout.write(f"\nТекущий баланс: ${current_balance} (изменение: ${balance_change:+.2f})")

                # Показываем открытые позиции
                open_positions = Position.objects.filter(user=user, is_open=True)
                if open_positions.exists():
                    self.stdout.write(f"\nОткрытые позиции: {open_positions.count()}")
                    for pos in open_positions[:3]:  # Показываем первые 3
                        pnl = pos.pnl
                        pnl_str = f"${pnl:+.2f}" if pnl else "N/A"
                        self.stdout.write(
                            f"  - {pos.symbol.symbol}: {pos.quantity} @ ${pos.entry_price} "
                            f"(текущая: ${pos.current_price or 0}, P&L: {pnl_str})"
                        )

            except Exception as e:
                stats["errors"] += 1
                self.stdout.write(self.style.ERROR(f"\n✗ Ошибка в итерации {iteration}: {str(e)}"))
                import traceback
                self.stdout.write(traceback.format_exc())

            # Задержка между итерациями
            if iteration < iterations:
                self.stdout.write(f"\nОжидание {delay} секунд перед следующей итерацией...")
                time.sleep(delay)

        # Финальная статистика
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS("ФИНАЛЬНАЯ СТАТИСТИКА"))
        self.stdout.write(self.style.SUCCESS(f"{'='*60}"))
        self.stdout.write(f"Всего решений: {stats['decisions']}")
        self.stdout.write(f"  - BUY: {stats['buy_decisions']}")
        self.stdout.write(f"  - SELL: {stats['sell_decisions']}")
        self.stdout.write(f"  - HOLD: {stats['hold_decisions']}")
        self.stdout.write(f"Выполнено сделок: {stats['trades']}")
        self.stdout.write(f"Ошибок: {stats['errors']}")

        account.refresh_from_db()
        final_balance = account.balance
        total_change = final_balance - initial_balance
        return_percent = (total_change / initial_balance * 100) if initial_balance > 0 else 0

        self.stdout.write(f"\nБаланс:")
        self.stdout.write(f"  Начальный: ${initial_balance}")
        self.stdout.write(f"  Финальный: ${final_balance}")
        self.stdout.write(f"  Изменение: ${total_change:+.2f} ({return_percent:+.2f}%)")

        # Показываем последние сообщения между агентами
        messages = Message.objects.filter(user=user).order_by("-timestamp")[:5]
        if messages.exists():
            self.stdout.write(f"\nПоследние сообщения между агентами: {messages.count()}")
            for msg in messages:
                self.stdout.write(
                    f"  - {msg.from_agent} → {msg.to_agent}: {msg.message_type} "
                    f"({msg.timestamp.strftime('%H:%M:%S')})"
                )

        # Показываем последние логи
        logs = AgentLog.objects.filter(agent_status__user=user).order_by("-timestamp")[:5]
        if logs.exists():
            self.stdout.write(f"\nПоследние логи агентов: {logs.count()}")
            for log in logs:
                self.stdout.write(
                    f"  - [{log.level}] {log.agent_status.agent_type}: {log.message[:50]}"
                )

        self.stdout.write(self.style.SUCCESS("\n✓ Тестирование завершено!"))

