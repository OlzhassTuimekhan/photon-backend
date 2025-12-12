"""
Сервисы для работы с данными рынка
"""
import os
import sys
import importlib.util

from .binance_api import BinanceAPIService
from .binance_websocket import BinanceWebSocketService

# Импорт из файла trading/services.py (не из папки)
# Используем importlib для загрузки модуля из файла
try:
    # Получаем путь к файлу trading/services.py
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    services_file_path = os.path.join(parent_dir, 'services.py')
    
    # Используем уникальное имя модуля, чтобы избежать конфликтов
    module_name = 'trading._services_file_module'
    
    # Проверяем, не загружен ли уже модуль
    if module_name not in sys.modules:
        # Добавляем родительскую директорию в sys.path для правильной загрузки зависимостей
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        spec = importlib.util.spec_from_file_location(module_name, services_file_path)
        if spec and spec.loader:
            trading_services_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = trading_services_module
            spec.loader.exec_module(trading_services_module)
        else:
            trading_services_module = None
    else:
        trading_services_module = sys.modules[module_name]
    
    # Экспортируем нужные классы и функции
    if trading_services_module:
        MarketDataService = getattr(trading_services_module, 'MarketDataService', None)
        get_market_data_service = getattr(trading_services_module, 'get_market_data_service', None)
        BybitDataService = getattr(trading_services_module, 'BybitDataService', None)
    else:
        MarketDataService = None
        get_market_data_service = None
        BybitDataService = None
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import from trading/services.py: {e}", exc_info=True)
    MarketDataService = None
    get_market_data_service = None
    BybitDataService = None

__all__ = [
    'BinanceAPIService',
    'BinanceWebSocketService',
]

if MarketDataService:
    __all__.append('MarketDataService')
if get_market_data_service:
    __all__.append('get_market_data_service')
if BybitDataService:
    __all__.append('BybitDataService')

