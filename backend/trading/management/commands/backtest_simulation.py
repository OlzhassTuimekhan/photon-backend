"""
Команда для симуляции торговли на исторических данных с ускорением времени.

Симулирует работу агентов на исторических данных, где 1 секунда = 1 час.
Это позволяет быстро протестировать модель на реальных данных за месяцы.
"""
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
import pandas as pd
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone as django_timezone
from django.db import transaction

from trading.models import Symbol, Account, UserSettings, Position, Trade, TradingDecision, MarketData
from trading.agents import MarketMonitoringAgent, DecisionMakingAgent, ExecutionAgent
from trading.agents.integration import (
    MarketAgentIntegration,
    DecisionAgentIntegration,
    ExecutionAgentIntegration
)

User = get_user_model()


class Command(BaseCommand):
    help = "Симуляция торговли на исторических данных с ускорением времени (1 сек = 1 час)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--email",
            type=str,
            required=True,
            help="Email пользователя для симуляции",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Символ для симуляции (по умолчанию: BTCUSDT)",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            help="Начальная дата в формате YYYY-MM-DD (по умолчанию: 1 месяц назад)",
        )
        parser.add_argument(
            "--end-date",
            type=str,
            help="Конечная дата в формате YYYY-MM-DD (по умолчанию: сегодня)",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            help="Интервал данных (1h, 4h, 1d) - по умолчанию: 1h",
        )
        parser.add_argument(
            "--speed",
            type=float,
            default=1.0,
            help="Скорость симуляции (секунды на интервал) - по умолчанию: 1.0",
        )
        parser.add_argument(
            "--initial-balance",
            type=Decimal,
            default=Decimal("10000.00"),
            help="Начальный баланс (по умолчанию: 10000.00)",
        )

    def handle(self, *args, **options):
        email = options["email"]
        symbol_code = options["symbol"].upper()
        interval = options["interval"]
        speed = options["speed"]
        initial_balance = options["initial_balance"]

        # Определяем даты
        if options.get("end_date"):
            end_date = datetime.strptime(options["end_date"], "%Y-%m-%d")
        else:
            end_date = django_timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

        if options.get("start_date"):
            start_date = datetime.strptime(options["start_date"], "%Y-%m-%d")
        else:
            start_date = end_date - timedelta(days=30)  # 1 месяц назад

        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(self.style.SUCCESS("СИМУЛЯЦИЯ НА ИСТОРИЧЕСКИХ ДАННЫХ"))
        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(f"Пользователь: {email}")
        self.stdout.write(f"Символ: {symbol_code}")
        self.stdout.write(f"Период: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        self.stdout.write(f"Интервал: {interval}")
        self.stdout.write(f"Скорость: {speed} сек/интервал (1 сек = 1 час)")
        self.stdout.write(f"Начальный баланс: ${initial_balance}\n")

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Пользователь с email '{email}' не найден"))
            return

        # Инициализация
        symbol, _ = Symbol.objects.get_or_create(
            user=user,
            symbol=symbol_code,
            defaults={"name": f"{symbol_code} Trading", "is_active": True}
        )

        account, _ = Account.objects.get_or_create(
            user=user,
            defaults={
                "balance": initial_balance,
                "free_cash": initial_balance,
                "initial_balance": initial_balance,
            }
        )
        account.balance = initial_balance
        account.free_cash = initial_balance
        account.save()

        # Загружаем исторические данные
        self.stdout.write("\n[1/4] Загрузка исторических данных...")
        try:
            market_agent = MarketMonitoringAgent(
                ticker=symbol_code,
                interval=interval,
                period="1y",  # Загружаем год данных
                enable_cache=True,
                request_delay=2.0,
                max_retries=3,
                backoff_factor=2.0
            )
            
            # Получаем обработанные данные
            # get_processed_data(analyze=True) возвращает кортеж (DataFrame, Dict)
            result = market_agent.get_processed_data(analyze=True)
            
            if isinstance(result, tuple):
                historical_data, analysis = result
            else:
                historical_data = result
            
            if historical_data is None or historical_data.empty:
                self.stdout.write(self.style.ERROR("Не удалось загрузить исторические данные"))
                return
            
            # Убеждаемся, что индекс - это timestamp
            # После preprocess может быть колонка timestamp, а не индекс
            if 'timestamp' in historical_data.columns:
                historical_data = historical_data.set_index('timestamp')
            
            # Убеждаемся, что индекс - это DatetimeIndex
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.index = pd.to_datetime(historical_data.index)
            
            # Отладочная информация: показываем реальные даты в данных
            self.stdout.write(f"\nОтладка: Данные содержат {len(historical_data)} записей")
            if len(historical_data) > 0:
                first_date = historical_data.index[0]
                last_date = historical_data.index[-1]
                self.stdout.write(f"Первая дата в данных: {first_date}")
                self.stdout.write(f"Последняя дата в данных: {last_date}")
                self.stdout.write(f"Запрошенный период: {start_date} - {end_date}")
            
            # Конвертируем start_date и end_date в тот же тип, что и индекс
            # Используем numpy datetime64 для совместимости
            import numpy as np
            start_ts = np.datetime64(start_date)
            end_ts = np.datetime64(end_date)
            
            # Конвертируем индекс в numpy datetime64 для сравнения
            index_as_numpy = historical_data.index.values.astype('datetime64[ns]')
            
            # Фильтруем по датам используя numpy сравнение
            mask = (index_as_numpy >= start_ts) & (index_as_numpy <= end_ts)
            filtered_data = historical_data.loc[mask]
            
            if filtered_data.empty:
                self.stdout.write(self.style.WARNING(
                    f"\n⚠ Нет данных в указанном периоде ({start_date} - {end_date}). "
                    f"Используем все доступные данные ({first_date} - {last_date})."
                ))
                # Используем все данные, если фильтрация не дала результатов
                # Это нормально для Bybit, который возвращает только последние N свечей
                historical_data = historical_data
            else:
                historical_data = filtered_data
                self.stdout.write(self.style.SUCCESS(
                    f"✓ Отфильтровано: {len(historical_data)} записей в запрошенном периоде"
                ))
            
            if historical_data.empty:
                self.stdout.write(self.style.ERROR("Нет данных для симуляции"))
                return
            
            self.stdout.write(self.style.SUCCESS(
                f"✓ Загружено {len(historical_data)} записей "
                f"({historical_data.index[0]} - {historical_data.index[-1]})"
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Ошибка загрузки данных: {e}"))
            return

        # Инициализация агентов
        self.stdout.write("\n[2/4] Инициализация агентов...")
        decision_agent = DecisionMakingAgent(
            model_type="random_forest",
            risk_tolerance="medium",
            min_confidence=0.05,  # Очень низкий порог для симуляции (чтобы модель делала больше сделок)
            enable_ai=True,
            use_historical_training=True,
            training_ticker=symbol_code,
            training_period="1mo",
            user_id=user.id,
            enable_continuous_learning=False  # Отключаем переобучение в симуляции для уменьшения логов
        )
        
        execution_agent = ExecutionAgent(
            execution_mode="simulated",
            enable_slippage=True,
            slippage_factor=0.001,
            commission_rate=0.001,
        )
        
        market_integration = MarketAgentIntegration(user)
        decision_integration = DecisionAgentIntegration(user)
        execution_integration = ExecutionAgentIntegration(user)
        
        self.stdout.write(self.style.SUCCESS("✓ Агенты инициализированы"))

        # Статистика
        stats = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "total_pnl": Decimal("0.00"),
            "profitable_trades": 0,
            "losing_trades": 0,
        }

        # Симуляция
        self.stdout.write("\n[3/4] Запуск симуляции...")
        self.stdout.write(self.style.WARNING("Нажмите Ctrl+C для остановки\n"))
        
        start_time = time.time()
        last_display_time = start_time
        
        try:
            for idx, (timestamp, row) in enumerate(historical_data.iterrows()):
                current_time = time.time()
                
                # Показываем прогресс каждые 5 секунд
                if current_time - last_display_time >= 5.0:
                    progress = (idx + 1) / len(historical_data) * 100
                    elapsed = current_time - start_time
                    estimated_total = elapsed / (idx + 1) * len(historical_data)
                    remaining = estimated_total - elapsed
                    
                    self.stdout.write(
                        f"\r[{idx+1}/{len(historical_data)}] "
                        f"Прогресс: {progress:.1f}% | "
                        f"Время: {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                        f"Осталось: {remaining:.0f}с | "
                        f"Баланс: ${account.balance:.2f} | "
                        f"Сделок: {stats['total_trades']} | "
                        f"PnL: ${stats['total_pnl']:+.2f}",
                        ending=""
                    )
                    last_display_time = current_time
                
                # Конвертируем timestamp в aware datetime (с timezone) для Django
                if pd.api.types.is_datetime64_any_dtype(type(timestamp)):
                    # Если timestamp naive, добавляем UTC timezone
                    if timestamp.tzinfo is None:
                        timestamp_aware = timestamp.replace(tzinfo=dt_timezone.utc)
                    else:
                        timestamp_aware = timestamp
                else:
                    # Если не datetime, конвертируем
                    timestamp_aware = pd.to_datetime(timestamp)
                    if timestamp_aware.tzinfo is None:
                        timestamp_aware = timestamp_aware.replace(tzinfo=dt_timezone.utc)
                
                # Подготовка данных рынка
                # После preprocess колонки в нижнем регистре (open, close, а не Open, Close)
                # row - это pandas Series, нужно правильно извлекать значения
                def get_value(series, key, default=0):
                    """Безопасное извлечение значения из Series"""
                    try:
                        if key in series.index:
                            val = series[key]
                            # Если это Series (не должно быть, но на всякий случай)
                            if isinstance(val, pd.Series):
                                return float(val.iloc[0]) if len(val) > 0 else default
                            # Если это скалярное значение
                            if pd.isna(val):
                                return default
                            return float(val)
                        # Пробуем альтернативные имена колонок
                        alt_key = key.capitalize() if key.islower() else key.lower()
                        if alt_key in series.index:
                            val = series[alt_key]
                            if isinstance(val, pd.Series):
                                return float(val.iloc[0]) if len(val) > 0 else default
                            if pd.isna(val):
                                return default
                            return float(val)
                    except (ValueError, TypeError, KeyError):
                        pass
                    return default
                
                def get_str_value(series, key, default=""):
                    """Безопасное извлечение строкового значения из Series"""
                    try:
                        if key in series.index:
                            val = series[key]
                            if isinstance(val, pd.Series):
                                return str(val.iloc[0]) if len(val) > 0 else default
                            if pd.isna(val):
                                return default
                            return str(val)
                    except (ValueError, TypeError, KeyError):
                        pass
                    return default
                
                market_message = {
                    "timestamp": timestamp_aware.isoformat(),
                    "ticker": symbol_code,
                    "ohlcv": {
                        "open": get_value(row, "open", get_value(row, "Open", 0)),
                        "high": get_value(row, "high", get_value(row, "High", 0)),
                        "low": get_value(row, "low", get_value(row, "Low", 0)),
                        "close": get_value(row, "close", get_value(row, "Close", 0)),
                        "volume": get_value(row, "volume", get_value(row, "Volume", 0)),
                    },
                    "indicators": {
                        "sma10": get_value(row, "sma10", 0),
                        "sma20": get_value(row, "sma20", 0),
                        "rsi14": get_value(row, "rsi14", 50.0),
                        "macd": get_value(row, "macd", 0),
                        "macd_hist": get_value(row, "macd_hist", 0),
                        "volatility": get_value(row, "volatility", 0),
                        "price_change": get_value(row, "price_change", 0),
                    },
                    "analysis": {
                        "trend": get_str_value(row, "trend", "sideways"),
                        "strength": get_value(row, "strength", 0.5),
                        "signals": {
                            "rsi_state": get_str_value(row, "rsi_state", "neutral"),
                            "sma_cross": int(get_value(row, "sma_cross", 0)),
                        }
                    },
                    "meta": {
                        "source": "historical_backtest",
                        "timestamp": timestamp_aware.isoformat(),
                    }
                }
                
                # Создаем MarketData для БД (для совместимости)
                # Используем aware datetime для избежания предупреждений
                market_data_obj, _ = MarketData.objects.get_or_create(
                    symbol=symbol,
                    timestamp=timestamp_aware,
                    defaults={
                        "price": Decimal(str(market_message["ohlcv"]["close"])),
                        "volume": int(market_message["ohlcv"]["volume"]),
                        "high": Decimal(str(market_message["ohlcv"]["high"])),
                        "low": Decimal(str(market_message["ohlcv"]["low"])),
                        "open_price": Decimal(str(market_message["ohlcv"]["open"])),
                        "change": Decimal("0.0"),
                        "change_percent": Decimal("0.0"),
                    }
                )
                
                # Принимаем решение
                try:
                    decision = decision_integration.make_decision(
                        symbol=symbol,
                        market_data_obj=market_data_obj,
                        market_message=market_message,
                        decision_agent=decision_agent
                    )
                    
                    stats["total_decisions"] += 1
                    if decision.decision == "BUY":
                        stats["buy_decisions"] += 1
                    elif decision.decision == "SELL":
                        stats["sell_decisions"] += 1
                    else:
                        stats["hold_decisions"] += 1
                    
                    # Выполняем сделку если не HOLD
                    if decision.decision != "HOLD":
                        decision_dict = {
                            "action": decision.decision,
                            "ticker": symbol_code,
                            "quantity": decision.metadata.get("quantity", 1),
                            "price": decision.metadata.get("price", float(market_data_obj.price)),
                            "confidence": float(decision.confidence / 100) if decision.confidence else 0.5,
                            "timestamp": timestamp_aware.isoformat(),
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
                                stats["total_trades"] += 1
                                if trade.action == "BUY":
                                    stats["buy_trades"] += 1
                                else:
                                    stats["sell_trades"] += 1
                                
                                if trade.pnl is not None:
                                    stats["total_pnl"] += trade.pnl
                                    if trade.pnl > 0:
                                        stats["profitable_trades"] += 1
                                    elif trade.pnl < 0:
                                        stats["losing_trades"] += 1
                
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"\nОшибка на шаге {idx+1}: {e}"))
                    continue
                
                # Обновляем баланс из БД
                account.refresh_from_db()
                
                # Задержка для визуализации
                if speed > 0:
                    time.sleep(speed)
        
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n\nСимуляция остановлена пользователем"))
        
        # Финальная статистика
        self.stdout.write(self.style.SUCCESS("\n\n[4/4] Финальная статистика:"))
        self.stdout.write(self.style.SUCCESS("=" * 70))
        
        account.refresh_from_db()
        final_balance = account.balance
        total_return = final_balance - initial_balance
        total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0
        
        self.stdout.write(f"Начальный баланс: ${initial_balance:.2f}")
        self.stdout.write(f"Финальный баланс: ${final_balance:.2f}")
        self.stdout.write(f"Общая прибыль/убыток: ${total_return:+.2f} ({total_return_pct:+.2f}%)")
        self.stdout.write(f"\nРешения:")
        self.stdout.write(f"  Всего: {stats['total_decisions']}")
        self.stdout.write(f"  BUY: {stats['buy_decisions']}")
        self.stdout.write(f"  SELL: {stats['sell_decisions']}")
        self.stdout.write(f"  HOLD: {stats['hold_decisions']}")
        self.stdout.write(f"\nСделки:")
        self.stdout.write(f"  Всего: {stats['total_trades']}")
        self.stdout.write(f"  BUY: {stats['buy_trades']}")
        self.stdout.write(f"  SELL: {stats['sell_trades']}")
        self.stdout.write(f"  Прибыльных: {stats['profitable_trades']}")
        self.stdout.write(f"  Убыточных: {stats['losing_trades']}")
        self.stdout.write(f"  Общий PnL: ${stats['total_pnl']:+.2f}")
        
        # Открытые позиции
        open_positions = Position.objects.filter(user=user, is_open=True)
        if open_positions.exists():
            self.stdout.write(f"\nОткрытые позиции: {open_positions.count()}")
            for pos in open_positions:
                pnl = pos.pnl
                pnl_str = f"${pnl:+.2f}" if pnl else "N/A"
                self.stdout.write(
                    f"  {pos.symbol.symbol}: {pos.quantity} @ ${pos.entry_price:.2f} "
                    f"(PnL: {pnl_str})"
                )
        
        elapsed_time = time.time() - start_time
        self.stdout.write(f"\nВремя симуляции: {elapsed_time:.1f} секунд")
        self.stdout.write(f"Симулированный период: {(end_date - start_date).days} дней")
        self.stdout.write(self.style.SUCCESS("=" * 70))

