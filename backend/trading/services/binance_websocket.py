"""
Сервис для получения данных через Binance WebSocket
"""
import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Optional, Dict, List, Callable
from decimal import Decimal
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceWebSocketService:
    """Сервис для получения данных через Binance WebSocket"""
    
    def __init__(self):
        self.base_url = "wss://stream.binance.com:9443"
        self.is_connected = False
        self.websocket = None
        self.callbacks: Dict[str, List[Callable]] = {}
        
    async def connect_ticker(self, symbol: str, callback: Callable):
        """
        Подключается к стриму тикера для символа
        
        Args:
            symbol: Символ (например, "btcusdt")
            callback: Функция для обработки данных (принимает dict с данными)
        """
        symbol = symbol.lower()
        url = f"{self.base_url}/ws/{symbol}@ticker"
        
        try:
            async with websockets.connect(url) as websocket:
                self.is_connected = True
                logger.info(f"Connected to Binance WebSocket ticker stream for {symbol.upper()}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await callback(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
            raise
    
    async def connect_kline(self, symbol: str, interval: str, callback: Callable):
        """
        Подключается к стриму свечей (kline) для символа
        
        Args:
            symbol: Символ (например, "btcusdt")
            interval: Интервал (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            callback: Функция для обработки данных (принимает dict с данными свечи)
        """
        symbol = symbol.lower()
        url = f"{self.base_url}/ws/{symbol}@kline_{interval}"
        
        try:
            async with websockets.connect(url) as websocket:
                self.is_connected = True
                logger.info(f"Connected to Binance WebSocket kline stream for {symbol.upper()} @ {interval}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'k' in data:
                            await callback(data['k'])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
            raise
    
    @staticmethod
    def parse_ticker_data(data: Dict) -> Dict:
        """
        Парсит данные тикера в стандартный формат
        
        Returns:
            dict с полями: price, volume, high, low, open_price, change, change_percent, timestamp
        """
        try:
            price = Decimal(str(data.get('c', '0')))
            volume_24h = Decimal(str(data.get('v', '0')))
            high_24h = Decimal(str(data.get('h', '0')))
            low_24h = Decimal(str(data.get('l', '0')))
            open_24h = Decimal(str(data.get('o', '0')))
            change = Decimal(str(data.get('p', '0')))
            change_percent = Decimal(str(data.get('P', '0')))
            
            # Timestamp в миллисекундах
            event_time = data.get('E', 0)
            timestamp = datetime.fromtimestamp(event_time / 1000) if event_time else datetime.now()
            
            return {
                "price": price,
                "volume": int(volume_24h),
                "high": high_24h,
                "low": low_24h,
                "open_price": open_24h,
                "change": change,
                "change_percent": change_percent,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error(f"Error parsing ticker data: {e}")
            return None
    
    @staticmethod
    def parse_kline_data(kline_data: Dict) -> Dict:
        """
        Парсит данные свечи в стандартный формат
        
        Returns:
            dict с полями: timestamp, open, high, low, close, volume
        """
        try:
            # Время открытия свечи (миллисекунды)
            open_time = int(kline_data.get('t', 0))
            timestamp = datetime.fromtimestamp(open_time / 1000) if open_time else datetime.now()
            
            return {
                "timestamp": timestamp,
                "open": Decimal(str(kline_data.get('o', '0'))),
                "high": Decimal(str(kline_data.get('h', '0'))),
                "low": Decimal(str(kline_data.get('l', '0'))),
                "close": Decimal(str(kline_data.get('c', '0'))),
                "volume": int(Decimal(str(kline_data.get('v', '0')))),
                "is_closed": kline_data.get('x', False),  # Закрыта ли свеча
            }
        except Exception as e:
            logger.error(f"Error parsing kline data: {e}")
            return None


async def collect_historical_data_from_websocket(
    symbol: str,
    interval: str = "1h",
    max_candles: int = 200
) -> List[Dict]:
    """
    Собирает исторические данные через WebSocket (свечи в реальном времени)
    
    Args:
        symbol: Символ (например, "btcusdt")
        interval: Интервал свечей
        max_candles: Максимальное количество свечей для сбора
    
    Returns:
        Список словарей с данными свечей
    """
    service = BinanceWebSocketService()
    collected_data = []
    
    async def kline_callback(kline_data: Dict):
        parsed = BinanceWebSocketService.parse_kline_data(kline_data)
        if parsed and parsed.get('is_closed'):  # Собираем только закрытые свечи
            collected_data.append(parsed)
            logger.info(f"Collected candle {len(collected_data)}/{max_candles}: {parsed['timestamp']} @ ${parsed['close']}")
    
    try:
        # Запускаем сбор данных
        task = asyncio.create_task(service.connect_kline(symbol, interval, kline_callback))
        
        # Ждем пока соберем нужное количество свечей
        while len(collected_data) < max_candles:
            await asyncio.sleep(1)
            if len(collected_data) >= max_candles:
                task.cancel()
                break
        
        return collected_data[:max_candles]
        
    except asyncio.CancelledError:
        logger.info(f"Collected {len(collected_data)} candles")
        return collected_data
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return collected_data


def get_historical_data_sync(symbol: str, interval: str = "1h", max_candles: int = 200) -> List[Dict]:
    """
    Синхронная обертка для получения исторических данных
    
    Args:
        symbol: Символ
        interval: Интервал
        max_candles: Максимум свечей
    
    Returns:
        Список данных свечей
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        collect_historical_data_from_websocket(symbol, interval, max_candles)
    )

