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
    """Периодическая задача для обновления данных рынка (запускается по расписанию)"""
    from trading.models import Symbol

    # Получаем уникальных пользователей, у которых есть активные символы
    user_ids = Symbol.objects.filter(is_active=True).values_list("user_id", flat=True).distinct()

    for user_id in user_ids:
        try:
            user = User.objects.get(id=user_id)
            # Проверяем, запущен ли мониторинг для этого пользователя
            try:
                status_obj = AgentStatus.objects.get(user=user, agent_type="MARKET_MONITOR")
                if status_obj.status == "RUNNING":
                    # Обновляем данные для всех символов пользователя
                    market_service = get_market_data_service()
                    user_symbols = Symbol.objects.filter(user=user, is_active=True)
                    for sym in user_symbols:
                        data = market_service.get_latest_data(sym.symbol)
                        if data:
                            MarketData.objects.create(symbol=sym, **data)

                    status_obj.last_activity = timezone.now()
                    status_obj.save()
            except AgentStatus.DoesNotExist:
                pass
        except User.DoesNotExist:
            logger.warning(f"User {user_id} not found, skipping")
            continue


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

