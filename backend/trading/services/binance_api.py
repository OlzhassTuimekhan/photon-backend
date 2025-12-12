"""
Сервис для получения исторических данных через Binance REST API
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from decimal import Decimal
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceAPIService:
    """Сервис для получения данных через Binance REST API"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Получает исторические свечи (kline) через Binance REST API
        
        Args:
            symbol: Символ (например, "BTCUSDT")
            interval: Интервал (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Количество свечей (максимум 1000)
            start_time: Начальное время (timestamp в миллисекундах)
            end_time: Конечное время (timestamp в миллисекундах)
        
        Returns:
            Список словарей с данными свечей
        """
        symbol = symbol.upper()
        url = f"{self.base_url}/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # Binance максимум 1000
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Парсим данные
            # Формат: [timestamp, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            result = []
            for kline in data:
                timestamp_ms = kline[0]
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                
                result.append({
                    "timestamp": timestamp,
                    "open": Decimal(str(kline[1])),
                    "high": Decimal(str(kline[2])),
                    "low": Decimal(str(kline[3])),
                    "close": Decimal(str(kline[4])),
                    "volume": int(Decimal(str(kline[5]))),
                    "quote_volume": Decimal(str(kline[7])),
                    "trades": int(kline[8]),
                })
            
            logger.info(f"Retrieved {len(result)} klines for {symbol} @ {interval}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Binance klines: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Binance data: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30
    ) -> Optional[List[Dict]]:
        """
        Получает исторические данные за указанный период
        
        Args:
            symbol: Символ
            interval: Интервал
            days: Количество дней истории
        
        Returns:
            Список данных свечей
        """
        # Вычисляем временные границы
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Binance ограничивает 1000 свечей за запрос
        # Нужно делать несколько запросов если нужно больше
        max_candles_per_request = 1000
        
        # Оцениваем сколько свечей нужно
        interval_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
            "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
        }
        minutes_per_candle = interval_minutes.get(interval, 60)
        total_candles = (days * 24 * 60) // minutes_per_candle
        
        all_data = []
        current_end = int(end_time.timestamp() * 1000)
        
        # Если нужно больше 1000 свечей, делаем несколько запросов
        while len(all_data) < total_candles:
            limit = min(max_candles_per_request, total_candles - len(all_data))
            
            data = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                end_time=current_end
            )
            
            if not data:
                break
            
            all_data = data + all_data
            
            # Обновляем end_time для следующего запроса
            if len(data) > 0:
                current_end = int((data[0]["timestamp"] - timedelta(seconds=1)).timestamp() * 1000)
            
            # Если получили меньше чем запрашивали, значит достигли начала периода
            if len(data) < limit:
                break
        
        logger.info(f"Total retrieved: {len(all_data)} candles for {symbol} over {days} days")
        return all_data if all_data else None
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Получает текущий тикер для символа
        
        Args:
            symbol: Символ
        
        Returns:
            dict с данными тикера
        """
        symbol = symbol.upper()
        url = f"{self.base_url}/ticker/24hr"
        
        params = {"symbol": symbol}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "symbol": data.get("symbol"),
                "price": Decimal(str(data.get("lastPrice", "0"))),
                "volume": Decimal(str(data.get("volume", "0"))),
                "high": Decimal(str(data.get("highPrice", "0"))),
                "low": Decimal(str(data.get("lowPrice", "0"))),
                "open": Decimal(str(data.get("openPrice", "0"))),
                "change": Decimal(str(data.get("priceChange", "0"))),
                "change_percent": Decimal(str(data.get("priceChangePercent", "0"))),
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return None

