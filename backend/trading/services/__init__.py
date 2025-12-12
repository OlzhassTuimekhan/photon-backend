"""
Сервисы для работы с данными рынка
"""
from .binance_api import BinanceAPIService
from .binance_websocket import BinanceWebSocketService

# Реэкспорт BybitDataService из родительского модуля для обратной совместимости
# BybitDataService находится в trading.services (файл), а не в trading.services (папка)
try:
    # Импортируем из родительского модуля trading.services (файл)
    import importlib.util
    import sys
    
    # Получаем модуль trading.services (файл)
    if 'trading.services' in sys.modules:
        parent_services = sys.modules['trading.services']
        BybitDataService = getattr(parent_services, 'BybitDataService', None)
    else:
        # Импортируем напрямую
        import trading.services as parent_services
        BybitDataService = getattr(parent_services, 'BybitDataService', None)
except Exception:
    BybitDataService = None

__all__ = [
    'BinanceAPIService',
    'BinanceWebSocketService',
]

if BybitDataService:
    __all__.append('BybitDataService')

