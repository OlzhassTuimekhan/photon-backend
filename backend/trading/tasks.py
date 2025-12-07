"""
Celery задачи для торговой системы
"""
import logging
from datetime import datetime

from celery import shared_task
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.conf import settings

from trading.models import Symbol, MarketData, AgentStatus
from trading.services import get_market_data_service

User = get_user_model()
logger = logging.getLogger(__name__)

# Хранилище для активных задач мониторинга
_active_monitoring_tasks = {}


@shared_task
def start_market_monitoring(user_id: int):
    """Запускает периодический мониторинг рынка для пользователя"""
    try:
        user = User.objects.get(id=user_id)
        symbols = Symbol.objects.filter(user=user, is_active=True)

        if not symbols.exists():
            logger.warning(f"No active symbols for user {user_id}")
            return

        # Обновляем данные для всех символов
        market_service = get_market_data_service()
        updated_count = 0
        for symbol in symbols:
            data = market_service.get_latest_data(symbol.symbol)
            if data:
                MarketData.objects.create(symbol=symbol, **data)
                updated_count += 1

        # Обновляем статус агента
        status_obj, _ = AgentStatus.objects.get_or_create(
            user=user,
            agent_type="MARKET_MONITOR",
            defaults={"status": "RUNNING"},
        )
        status_obj.status = "RUNNING"
        status_obj.last_activity = timezone.now()
        status_obj.save()

        logger.info(f"Market monitoring updated {updated_count} symbols for user {user_id}")
        return {"updated": updated_count, "symbols": list(symbols.values_list("symbol", flat=True))}

    except User.DoesNotExist:
        logger.error(f"User {user_id} not found")
    except Exception as e:
        logger.error(f"Error in market monitoring: {str(e)}", exc_info=True)
        # Обновляем статус на ошибку
        try:
            user = User.objects.get(id=user_id)
            status_obj, _ = AgentStatus.objects.get_or_create(
                user=user,
                agent_type="MARKET_MONITOR",
                defaults={"status": "ERROR"},
            )
            status_obj.status = "ERROR"
            status_obj.error_message = str(e)
            status_obj.save()
        except Exception:
            pass


@shared_task
def periodic_market_update():
    """
    Периодическая задача для обновления данных рынка (запускается по расписанию)
    Обновляет данные для ВСЕХ активных символов, независимо от статуса агента
    """
    from trading.models import Symbol

    # Получаем уникальных пользователей, у которых есть активные символы
    user_ids = Symbol.objects.filter(is_active=True).values_list("user_id", flat=True).distinct()

    total_updated = 0
    for user_id in user_ids:
        try:
            user = User.objects.get(id=user_id)
            # Обновляем данные для всех активных символов пользователя
            # НЕ проверяем статус агента - данные должны обновляться всегда!
            market_service = get_market_data_service()
            user_symbols = Symbol.objects.filter(user=user, is_active=True)
            user_updated = 0
            
            for sym in user_symbols:
                try:
                    data = market_service.get_latest_data(sym.symbol)
                    if data:
                        MarketData.objects.create(symbol=sym, **data)
                        user_updated += 1
                        total_updated += 1
                except Exception as e:
                    logger.error(f"Error updating data for symbol {sym.symbol}: {str(e)}", exc_info=True)

            # Обновляем статус агента (если существует) для отслеживания активности
            try:
                status_obj = AgentStatus.objects.get(user=user, agent_type="MARKET_MONITOR")
                # Если статус был RUNNING, обновляем last_activity
                if status_obj.status == "RUNNING":
                    status_obj.last_activity = timezone.now()
                    status_obj.save()
            except AgentStatus.DoesNotExist:
                # Если агент не существует, создаем его со статусом IDLE
                # Это означает, что данные обновляются автоматически, но агент не был явно запущен
                AgentStatus.objects.create(
                    user=user,
                    agent_type="MARKET_MONITOR",
                    status="IDLE",
                    last_activity=timezone.now(),
                )
            
            if user_updated > 0:
                logger.info(f"Updated {user_updated} symbols for user {user_id}")
                
        except User.DoesNotExist:
            logger.warning(f"User {user_id} not found, skipping")
            continue
    
    logger.info(f"Periodic market update completed: {total_updated} total updates")
    return {"updated": total_updated}


def stop_market_monitoring(user_id: int):
    """Останавливает мониторинг для пользователя"""
    try:
        user = User.objects.get(id=user_id)
        status_obj = AgentStatus.objects.get(user=user, agent_type="MARKET_MONITOR")
        status_obj.status = "STOPPED"
        status_obj.save()
        logger.info(f"Market monitoring stopped for user {user_id}")
    except Exception as e:
        logger.error(f"Error stopping market monitoring: {str(e)}")


@shared_task
def run_ai_agents_workflow():
    """
    Автоматический запуск workflow ИИ агентов каждую минуту.
    
    Выполняет полный цикл:
    1. MarketMonitoringAgent - получение данных рынка
    2. DecisionMakingAgent - принятие решения
    3. ExecutionAgent - выполнение сделки (если не HOLD)
    
    Запускается для всех пользователей со статусом "running" в UserSettings.
    """
    from trading.models import UserSettings, Symbol, Trade
    from trading.agents import MarketMonitoringAgent, DecisionMakingAgent, ExecutionAgent
    from trading.agents.integration import (
        MarketAgentIntegration,
        DecisionAgentIntegration,
        ExecutionAgentIntegration
    )
    from django.utils import timezone as tz
    
    # Получаем всех пользователей с активной торговлей
    active_users = UserSettings.objects.filter(status="running").select_related("user")
    
    if not active_users.exists():
        logger.debug("No active users for AI agents workflow")
        return {"processed": 0}
    
    total_processed = 0
    total_decisions = 0
    total_trades = 0
    
    for user_settings in active_users:
        user = user_settings.user
        
        try:
            # Получаем активный символ пользователя
            symbol_obj = Symbol.objects.filter(
                user=user,
                symbol=user_settings.symbol,
                is_active=True
            ).first()
            
            if not symbol_obj:
                logger.warning(f"No active symbol {user_settings.symbol} for user {user.id}")
                continue
            
            # Шаг 1: MarketMonitoringAgent
            try:
                market_integration = MarketAgentIntegration(user)
                market_agent = MarketMonitoringAgent(
                    ticker=user_settings.symbol,
                    interval=user_settings.timeframe,
                    period="1mo",
                    enable_cache=True,
                    request_delay=5.0,
                    max_retries=5,
                    backoff_factor=3.0
                )
                
                market_message = market_integration.process_and_save(
                    symbol=symbol_obj,
                    market_agent=market_agent,
                    save_to_db=True
                )
                
                # Получаем последние данные из БД
                from trading.models import MarketData
                latest_data = MarketData.objects.filter(
                    symbol=symbol_obj
                ).order_by("-timestamp").first()
                
                if not latest_data:
                    logger.warning(f"No market data for {user_settings.symbol}")
                    continue
                
            except Exception as e:
                logger.error(f"Error in MarketMonitoringAgent for user {user.id}: {e}", exc_info=True)
                continue
            
            # Шаг 2: DecisionMakingAgent
            try:
                decision_integration = DecisionAgentIntegration(user)
                
                # Проверяем, сколько завершенных сделок у пользователя (для exploration режима)
                completed_trades_count = Trade.objects.filter(
                    user=user,
                    action="SELL",
                    pnl__isnull=False
                ).count()
                
                # Exploration режим: если данных для обучения мало, снижаем порог уверенности
                enable_exploration = completed_trades_count < 10  # Меньше 10 завершенных сделок
                exploration_confidence = 0.35 if enable_exploration else float(user_settings.confidence_threshold)
                
                decision_agent = DecisionMakingAgent(
                    model_type="random_forest" if user_settings.model_type == "Random Forest" else "gradient_boosting",
                    risk_tolerance=user_settings.risk_level,
                    min_confidence=exploration_confidence,  # Сниженный порог в exploration режиме
                    enable_ai=True,
                    use_historical_training=True,
                    training_ticker=user_settings.symbol,
                    training_period="1mo",
                    user_id=user.id  # Для доступа к БД для обучения
                )
                
                decision = decision_integration.make_decision(
                    symbol=symbol_obj,
                    market_data_obj=latest_data,
                    market_message=market_message,
                    decision_agent=decision_agent
                )
                
                total_decisions += 1
                decision_action = decision.decision
                
                if enable_exploration:
                    logger.info(f"Exploration mode: User {user.id}, Decision: {decision_action}, Confidence: {decision.confidence}%")
                
            except Exception as e:
                logger.error(f"Error in DecisionMakingAgent for user {user.id}: {e}", exc_info=True)
                continue
            
            # Шаг 3: ExecutionAgent (если не HOLD)
            if decision_action != "HOLD":
                try:
                    execution_integration = ExecutionAgentIntegration(user)
                    
                    execution_agent = ExecutionAgent(
                        execution_mode="simulated",
                        enable_slippage=True,
                        slippage_factor=0.001,
                        commission_rate=0.001,
                    )
                    
                    decision_dict = {
                        "action": decision_action,
                        "ticker": user_settings.symbol,
                        "quantity": decision.metadata.get("quantity", 1),
                        "price": decision.metadata.get("price", float(latest_data.price)),
                        "confidence": float(decision.confidence / 100) if decision.confidence else 0.5,
                        "timestamp": decision.created_at.isoformat(),
                        "reasoning": decision.reasoning,
                    }
                    
                    execution_result = execution_agent.receive_decision(decision_dict)
                    
                    if execution_result.get("status") == "executed":
                        trade = execution_integration.execute_trade(
                            symbol=symbol_obj,
                            decision_obj=decision,
                            execution_agent=execution_agent,
                            execution_result=execution_result
                        )
                        
                        if trade:
                            total_trades += 1
                            logger.info(f"Trade executed for user {user.id}: {trade.action} {trade.quantity} @ ${trade.price}")
                    
                except Exception as e:
                    logger.error(f"Error in ExecutionAgent for user {user.id}: {e}", exc_info=True)
                    continue
            
            total_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing user {user.id} in AI agents workflow: {e}", exc_info=True)
            continue
    
    logger.info(f"AI agents workflow completed: {total_processed} users, {total_decisions} decisions, {total_trades} trades")
    return {
        "processed": total_processed,
        "decisions": total_decisions,
        "trades": total_trades
    }
