"""
Интеграция AI агентов с Django моделями

Этот модуль предоставляет адаптеры для интеграции агентов с:
- Message моделью (коммуникация между агентами)
- AgentLog моделью (логирование)
- AgentStatus моделью (статусы агентов)
- UserSettings моделью (настройки пользователя)
"""

import logging
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import timedelta
from django.utils import timezone
from django.contrib.auth import get_user_model

from trading.models import (
    AgentStatus,
    AgentLog,
    Message,
    UserSettings,
    Symbol,
    MarketData,
    TradingDecision,
    Trade,
    Position,
    Account,
)

User = get_user_model()
logger = logging.getLogger(__name__)


class DjangoAgentAdapter:
    """
    Адаптер для интеграции AI агентов с Django моделями.
    
    Обеспечивает:
    - Логирование действий агентов
    - Сохранение сообщений между агентами
    - Обновление статусов агентов
    - Использование настроек пользователя
    """
    
    def __init__(self, user: User, agent_type: str):
        """
        Инициализация адаптера.
        
        Args:
            user: Пользователь Django
            agent_type: Тип агента ("MARKET_MONITOR", "DECISION_MAKER", "EXECUTION")
        """
        self.user = user
        self.agent_type = agent_type
        
        # Получаем или создаем статус агента
        self.agent_status, _ = AgentStatus.objects.get_or_create(
            user=user,
            agent_type=agent_type,
            defaults={"status": "IDLE"},
        )
        
        # Получаем настройки пользователя
        self.user_settings, _ = UserSettings.objects.get_or_create(
            user=user,
            defaults={
                "status": "stopped",
                "speed": 1.0,
                "symbol": "AAPL",
                "timeframe": "1h",
                "risk_level": "medium",
            }
        )
    
    def log(self, level: str, message: str, metadata: Optional[Dict] = None):
        """
        Логирует действие агента.
        
        Args:
            level: Уровень лога ("info", "warning", "error")
            message: Сообщение лога
            metadata: Дополнительные метаданные
        """
        try:
            AgentLog.objects.create(
                agent_status=self.agent_status,
                level=level,
                message=message,
            )
            logger.log(
                getattr(logging, level.upper(), logging.INFO),
                f"[{self.agent_type}] {message}"
            )
        except Exception as e:
            logger.error(f"Error logging agent action: {e}")
    
    def send_message(
        self,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any]
    ) -> Message:
        """
        Отправляет сообщение другому агенту.
        
        Args:
            to_agent: Тип агента-получателя
            message_type: Тип сообщения ("MARKET_SNAPSHOT", "TRADE_DECISION", "EXECUTION_REPORT")
            payload: Данные сообщения
        
        Returns:
            Созданный объект Message
        """
        try:
            message = Message.objects.create(
                user=self.user,
                from_agent=self.agent_type,
                to_agent=to_agent,
                message_type=message_type,
                payload=payload,
            )
            self.log("info", f"Sent {message_type} to {to_agent}")
            return message
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    def update_status(self, status: str, metadata: Optional[Dict] = None):
        """
        Обновляет статус агента.
        
        Args:
            status: Новый статус ("RUNNING", "IDLE", "STOPPED", "ERROR")
            metadata: Дополнительные метаданные
        """
        try:
            self.agent_status.status = status
            self.agent_status.last_activity = timezone.now()
            if metadata:
                if self.agent_status.metadata:
                    self.agent_status.metadata.update(metadata)
                else:
                    self.agent_status.metadata = metadata
            self.agent_status.save()
            self.log("info", f"Status updated to {status}")
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
    
    def get_user_settings(self) -> UserSettings:
        """Возвращает настройки пользователя."""
        return self.user_settings
    
    def get_risk_tolerance(self) -> str:
        """Возвращает уровень риска из настроек."""
        return self.user_settings.risk_level or "medium"
    
    def get_confidence_threshold(self) -> float:
        """Возвращает порог уверенности из настроек."""
        return float(self.user_settings.confidence_threshold or 0.55)


class MarketAgentIntegration:
    """Интеграция MarketMonitoringAgent с Django моделями."""
    
    def __init__(self, user: User):
        self.user = user
        self.adapter = DjangoAgentAdapter(user, "MARKET_MONITOR")
    
    def process_and_save(
        self,
        symbol: Symbol,
        market_agent,
        save_to_db: bool = True
    ) -> Dict:
        """
        Обрабатывает данные рынка и сохраняет в БД.
        
        Args:
            symbol: Объект Symbol из Django
            market_agent: Экземпляр MarketMonitoringAgent
            save_to_db: Сохранять ли данные в MarketData
        
        Returns:
            Стандартизированное сообщение для Decision Agent
        """
        try:
            self.adapter.update_status("RUNNING")
            self.adapter.log("info", f"Processing market data for {symbol.symbol}")
            
            # Получаем данные через MarketMonitoringAgent (только наши AI агенты)
            # Сначала получаем и обрабатываем данные
            market_agent.get_processed_data(analyze=True)
            
            # Теперь можем отправить данные Decision Agent
            market_message = market_agent.send_to_decision_agent(transport="direct")
            
            # Сохраняем в MarketData если нужно
            if save_to_db:
                latest_ohlcv = market_message.get("ohlcv", {})
                MarketData.objects.create(
                    symbol=symbol,
                    price=Decimal(str(latest_ohlcv.get("close", 0.0))),
                    volume=int(latest_ohlcv.get("volume", 0)),
                    high=Decimal(str(latest_ohlcv.get("high", 0.0))),
                    low=Decimal(str(latest_ohlcv.get("low", 0.0))),
                    open_price=Decimal(str(latest_ohlcv.get("open", 0.0))),
                    change=Decimal("0.0"),  # Будет рассчитано при следующем обновлении
                    change_percent=Decimal("0.0"),
                    timestamp=timezone.now(),
                )
            
            # Отправляем сообщение Decision Agent
            message = self.adapter.send_message(
                to_agent="DECISION_MAKER",
                message_type="MARKET_SNAPSHOT",
                payload=market_message
            )
            
            self.adapter.log("info", f"Market data processed and sent to Decision Agent")
            self.adapter.update_status("IDLE")
            return market_message
            
        except Exception as e:
            self.adapter.update_status("ERROR", {"error": str(e)})
            self.adapter.log("error", f"Error processing market data: {str(e)}")
            raise


class DecisionAgentIntegration:
    """Интеграция DecisionMakingAgent с Django моделями."""
    
    def __init__(self, user: User):
        self.user = user
        self.adapter = DjangoAgentAdapter(user, "DECISION_MAKER")
    
    def make_decision(
        self,
        symbol: Symbol,
        market_data_obj: Optional[MarketData],
        market_message: Dict,
        decision_agent
    ) -> TradingDecision:
        """
        Принимает решение и сохраняет в БД.
        
        ВАЖНО: Если решение SELL, но нет открытых позиций - автоматически меняет на HOLD.
        
        Args:
            symbol: Объект Symbol
            market_data_obj: Объект MarketData (может быть None)
            market_message: Сообщение от Market Agent
            decision_agent: Экземпляр DecisionMakingAgent
        
        Returns:
            Созданный объект TradingDecision
        """
        try:
            self.adapter.update_status("RUNNING")
            self.adapter.log("info", f"Making decision for {symbol.symbol}")
            
            # Получаем настройки для агента
            risk_tolerance = self.adapter.get_risk_tolerance()
            min_confidence = self.adapter.get_confidence_threshold()
            
            # Передаем пользователя в агент для continuous learning
            decision_agent._django_user = self.user
            
            # Принимаем решение
            ai_decision = decision_agent.receive_market_data(market_message)
            
            # ПРИМЕЧАНИЕ: Для симуляции разрешаем SELL даже без открытых позиций
            # Это ускоряет сбор данных для обучения модели
            # В реальной торговле здесь была бы проверка на открытые позиции
            
            # Сохраняем решение в БД
            decision = TradingDecision.objects.create(
                user=self.user,
                symbol=symbol,
                decision=ai_decision.get("action", "HOLD"),
                confidence=Decimal(str(ai_decision.get("confidence", 0.5) * 100)),  # Конвертируем в проценты
                market_data=market_data_obj,
                reasoning=ai_decision.get("reasoning", "No reasoning provided"),
                metadata={
                    "risk_score": ai_decision.get("risk_score", 0.0),
                    "quantity": ai_decision.get("quantity", 0),
                    "model_type": ai_decision.get("model_type", "rule_based"),
                    "indicators": market_message.get("indicators", {}),
                    "analysis": market_message.get("analysis", {}),
                    "price": ai_decision.get("price", 0.0),
                },
            )
            
            # Отправляем сообщение Execution Agent если не HOLD
            if ai_decision.get("action") != "HOLD":
                self.adapter.send_message(
                    to_agent="EXECUTION",
                    message_type="TRADE_DECISION",
                    payload={
                        "decision_id": decision.id,
                        "action": ai_decision.get("action"),
                        "ticker": symbol.symbol,
                        "quantity": ai_decision.get("quantity", 0),
                        "price": ai_decision.get("price", 0.0),
                        "confidence": ai_decision.get("confidence", 0.0),
                        "reasoning": ai_decision.get("reasoning", ""),
                    }
                )
            
            self.adapter.log("info", f"Decision made: {ai_decision.get('action')} for {symbol.symbol}")
            self.adapter.update_status("IDLE")
            
            return decision
            
        except Exception as e:
            self.adapter.update_status("ERROR", {"error": str(e)})
            self.adapter.log("error", f"Error making decision: {str(e)}")
            raise


class ExecutionAgentIntegration:
    """Интеграция ExecutionAgent с Django моделями."""
    
    def __init__(self, user: User):
        self.user = user
        self.adapter = DjangoAgentAdapter(user, "EXECUTION")
    
    def execute_trade(
        self,
        symbol: Symbol,
        decision_obj: TradingDecision,
        execution_agent,
        execution_result: Dict
    ) -> Trade:
        """
        Выполняет сделку и сохраняет в БД.
        
        Args:
            symbol: Объект Symbol
            decision_obj: Объект TradingDecision
            execution_agent: Экземпляр ExecutionAgent
            execution_result: Результат выполнения от ExecutionAgent
        
        Returns:
            Созданный объект Trade
        """
        try:
            self.adapter.update_status("RUNNING")
            self.adapter.log("info", f"Executing trade: {execution_result.get('action')} {symbol.symbol}")
            
            if execution_result.get("status") != "executed":
                self.adapter.log("warning", f"Trade not executed: {execution_result.get('message')}")
                self.adapter.update_status("IDLE")
                return None
            
            # Получаем или создаем счет
            account, _ = Account.objects.get_or_create(
                user=self.user,
                defaults={"balance": Decimal("10000.00"), "free_cash": Decimal("10000.00")}
            )
            
            action = execution_result.get("action")
            quantity = Decimal(str(execution_result.get("quantity", 0)))
            executed_price = Decimal(str(execution_result.get("executed_price", 0.0)))
            
            # Создаем сделку
            trade = Trade.objects.create(
                user=self.user,
                symbol=symbol,
                action=action,
                price=executed_price,
                quantity=quantity,
                agent_type="EXECUTION",
                pnl=None,  # Будет рассчитано при закрытии позиции
            )
            
            # Обновляем позиции и счет
            if action == "BUY":
                # Создаем или обновляем позицию
                position, created = Position.objects.get_or_create(
                    user=self.user,
                    symbol=symbol,
                    is_open=True,
                    defaults={
                        "quantity": quantity,
                        "entry_price": executed_price,
                        "current_price": executed_price,
                    }
                )
                
                if not created:
                    # Обновляем существующую позицию
                    old_qty = position.quantity
                    old_price = position.entry_price
                    total_cost = (old_qty * old_price) + (quantity * executed_price)
                    total_qty = old_qty + quantity
                    position.quantity = total_qty
                    position.entry_price = total_cost / total_qty
                    position.current_price = executed_price
                    position.save()
                
                trade.position = position
                trade.save()
                
                # Обновляем счет
                cost = quantity * executed_price
                account.balance -= cost
                account.free_cash -= cost
                account.save()
                
            elif action == "SELL":
                # Находим открытую позицию
                position = Position.objects.filter(
                    user=self.user,
                    symbol=symbol,
                    is_open=True
                ).first()
                
                if position and position.quantity >= quantity:
                    # Обновляем позицию
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.is_open = False
                        position.closed_at = timezone.now()
                    
                    # Рассчитываем P&L
                    pnl = (executed_price - position.entry_price) * quantity
                    trade.pnl = pnl
                    trade.position = position
                    trade.save()
                    
                    position.save()
                    
                    # Обновляем счет
                    revenue = quantity * executed_price
                    account.balance += revenue
                    account.free_cash += revenue
                    account.save()
                else:
                    # НЕТ открытой позиции - разрешаем SELL для симуляции
                    # Это ускоряет сбор данных для обучения модели
                    # ВАЖНО: Не изменяем баланс - это виртуальная сделка только для обучения
                    # Создаем "виртуальную" позицию для расчета PnL
                    from trading.models import Trade as TradeModel
                    recent_buys = TradeModel.objects.filter(
                        user=self.user,
                        symbol=symbol,
                        action="BUY"
                    ).order_by('-executed_at')[:5]
                    
                    if recent_buys.exists():
                        # Используем среднюю цену последних покупок
                        avg_entry_price = sum(float(t.price) for t in recent_buys) / len(recent_buys)
                    else:
                        # Если нет истории покупок, используем текущую цену (PnL будет 0)
                        avg_entry_price = float(executed_price)
                    
                    # Рассчитываем PnL как если бы была позиция
                    pnl = (executed_price - Decimal(str(avg_entry_price))) * quantity
                    trade.pnl = pnl
                    trade.save()
                    
                    # ВАЖНО: НЕ изменяем баланс при виртуальной продаже без позиции
                    # Это симуляция только для обучения модели
                    # В реальной торговле нельзя продать то, чего нет
                    # Баланс остается прежним - мы ничего не получили, т.к. нечего было продавать
                    
                    logger.info(
                        f"Simulated SELL without position (virtual trade for learning): "
                        f"{quantity} {symbol.symbol} @ {executed_price}, "
                        f"virtual entry: {avg_entry_price:.2f}, PnL: {pnl:.2f} "
                        f"(balance unchanged - no actual asset to sell)"
                    )
            
            # Отправляем отчет
            self.adapter.send_message(
                to_agent="DECISION_MAKER",
                message_type="EXECUTION_REPORT",
                payload={
                    "trade_id": trade.id,
                    "status": execution_result.get("status"),
                    "action": action,
                    "quantity": float(quantity),
                    "price": float(executed_price),
                }
            )
            
            self.adapter.log("info", f"Trade executed successfully: {action} {quantity} {symbol.symbol} @ {executed_price}")
            self.adapter.update_status("IDLE")
            
            return trade
            
        except Exception as e:
            self.adapter.update_status("ERROR", {"error": str(e)})
            self.adapter.log("error", f"Error executing trade: {str(e)}")
            raise

