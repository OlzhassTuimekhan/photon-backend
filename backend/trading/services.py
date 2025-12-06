"""
Сервисы для работы с данными рынка через yfinance и Bybit API
"""
import logging
import os
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import yfinance as yf
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class BybitDataService:
    """Сервис для получения данных рынка через Bybit API"""

    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = False):
        """
        Инициализация сервиса Bybit
        api_key и secret_key опциональны для публичных данных
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"

    def _generate_signature(self, params: dict, timestamp: str) -> str:
        """Генерирует подпись для приватных запросов"""
        if not self.secret_key:
            return ""
        
        # Сортируем параметры
        sorted_params = sorted(params.items())
        param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Создаем строку для подписи
        sign_str = timestamp + self.api_key + param_str
        
        # Генерируем HMAC SHA256
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def _make_request(self, endpoint: str, params: dict = None, private: bool = False) -> Optional[dict]:
        """Выполняет HTTP запрос к Bybit API"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        # Для приватных запросов добавляем подпись
        if private and self.api_key and self.secret_key:
            timestamp = str(int(time.time() * 1000))
            params["api_key"] = self.api_key
            params["timestamp"] = timestamp
            params["recv_window"] = "5000"
            
            signature = self._generate_signature(params, timestamp)
            params["sign"] = signature
            
            headers["X-BAPI-API-KEY"] = self.api_key
            headers["X-BAPI-SIGN"] = signature
            headers["X-BAPI-TIMESTAMP"] = timestamp
            headers["X-BAPI-RECV-WINDOW"] = "5000"
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Bybit возвращает данные в формате {"retCode": 0, "retMsg": "OK", "result": {...}}
            if data.get("retCode") == 0:
                return data.get("result")
            else:
                logger.error(f"Bybit API error: {data.get('retMsg')} (code: {data.get('retCode')})")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Bybit API request error: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error processing Bybit response: {str(e)}", exc_info=True)
            return None

    def get_latest_data(self, symbol: str, category: str = "spot") -> Optional[Dict]:
        """
        Получает последние данные по символу с Bybit
        symbol должен быть в формате Bybit (например: BTCUSDT, ETHUSDT)
        category: spot, linear, inverse
        """
        try:
            # Получаем тикер
            result = self._make_request("/v5/market/tickers", {
                "category": category,
                "symbol": symbol.upper(),
            })
            
            if not result or "list" not in result or not result["list"]:
                logger.warning(f"No ticker data for {symbol}")
                return None
            
            ticker_data = result["list"][0]
            
            # Парсим данные
            current_price = Decimal(ticker_data.get("lastPrice", "0"))
            open_price_24h = Decimal(ticker_data.get("prevPrice24h", current_price))
            high_24h = Decimal(ticker_data.get("highPrice24h", current_price))
            low_24h = Decimal(ticker_data.get("lowPrice24h", current_price))
            volume_24h = Decimal(ticker_data.get("volume24h", "0"))
            change_24h = current_price - open_price_24h
            change_percent_24h = (change_24h / open_price_24h * 100) if open_price_24h > 0 else Decimal("0")
            
            return {
                "price": current_price,
                "volume": int(volume_24h),
                "high": high_24h,
                "low": low_24h,
                "open_price": open_price_24h,
                "change": change_24h,
                "change_percent": change_percent_24h,
                "timestamp": datetime.now(),
                # 'name' не нужен для MarketData - это поле только в Symbol
            }
        except Exception as e:
            logger.error(f"Error fetching Bybit data for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_historical_data(
        self, symbol: str, category: str = "spot", interval: str = "1", limit: int = 200
    ) -> Optional[List[Dict]]:
        """
        Получает исторические данные с Bybit
        interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M (в минутах, кроме D/W/M)
        limit: количество свечей (максимум 200)
        """
        try:
            result = self._make_request("/v5/market/kline", {
                "category": category,
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": str(limit),
            })
            
            if not result or "list" not in result or not result["list"]:
                return None
            
            klines = result["list"]
            result_list = []
            
            for kline in klines:
                # Формат: [timestamp, open, high, low, close, volume, turnover]
                timestamp_ms = int(kline[0])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                
                result_list.append({
                    "timestamp": timestamp,
                    "open": Decimal(kline[1]),
                    "high": Decimal(kline[2]),
                    "low": Decimal(kline[3]),
                    "close": Decimal(kline[4]),
                    "volume": int(Decimal(kline[5])),
                })
            
            # Сортируем по времени (от старых к новым)
            result_list.sort(key=lambda x: x["timestamp"])
            return result_list
        except Exception as e:
            logger.error(f"Error fetching Bybit historical data for {symbol}: {str(e)}", exc_info=True)
            return None

    def validate_symbol(self, symbol: str, category: str = "spot") -> bool:
        """Проверяет, существует ли символ на Bybit"""
        try:
            result = self._make_request("/v5/market/tickers", {
                "category": category,
                "symbol": symbol.upper(),
            })
            return bool(result and "list" in result and result["list"])
        except Exception:
            return False

    def normalize_symbol(self, symbol: str) -> str:
        """
        Нормализует символ для Bybit
        Например: BTC -> BTCUSDT, ETH -> ETHUSDT
        """
        symbol = symbol.upper().strip()
        # Если уже в формате Bybit (содержит USDT, USDC и т.д.), возвращаем как есть
        if any(suffix in symbol for suffix in ["USDT", "USDC", "BTC", "ETH", "BNB", "BUSD"]):
            return symbol
        # Иначе добавляем USDT (самая популярная пара)
        return f"{symbol}USDT"


class MarketDataService:
    """Сервис для получения данных рынка (поддерживает yfinance и Bybit)"""

    def __init__(self, data_source: str = "auto"):
        """
        Инициализация сервиса
        data_source: "yfinance", "bybit", или "auto" (автоматический выбор)
        """
        self.data_source = data_source
        # Инициализируем Bybit сервис если нужно
        if data_source in ["bybit", "auto"]:
            from django.conf import settings
            self.bybit_service = BybitDataService(
                api_key=getattr(settings, "BYBIT_API_KEY", ""),
                secret_key=getattr(settings, "BYBIT_SECRET_KEY", ""),
                testnet=getattr(settings, "BYBIT_TESTNET", False),
            )
        else:
            self.bybit_service = None

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """
        Получает последние данные по символу
        Returns: dict с данными или None при ошибке
        """
        # Определяем источник данных
        use_bybit = False
        if self.data_source == "bybit":
            use_bybit = True
        elif self.data_source == "auto":
            # Автоматически определяем: если символ похож на криптовалюту, используем Bybit
            # Или если символ в формате Bybit (содержит USDT, USDC и т.д.)
            if any(suffix in symbol.upper() for suffix in ["USDT", "USDC", "BTC", "ETH", "BNB", "BUSD"]):
                use_bybit = True

        # Пробуем Bybit если нужно
        if use_bybit and self.bybit_service:
            # Нормализуем символ для Bybit
            bybit_symbol = self.bybit_service.normalize_symbol(symbol)
            data = self.bybit_service.get_latest_data(bybit_symbol)
            if data:
                return data
            # Если Bybit не сработал, пробуем yfinance

        # Используем yfinance
        try:
            ticker = yf.Ticker(symbol)
            
            # Пробуем получить info с таймаутом
            try:
                info = ticker.info
                if not info or len(info) == 0:
                    logger.warning(f"Empty info for {symbol}, trying alternative method")
                    info = {}
            except Exception as info_error:
                logger.warning(f"Could not get info for {symbol}: {info_error}, continuing with history data")
                info = {}

            # Получаем исторические данные за последний день
            try:
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty:
                    # Если нет минутных данных, берем дневные
                    hist = ticker.history(period="5d", interval="1d")
            except Exception as hist_error:
                logger.warning(f"Could not get 1d history for {symbol}: {hist_error}, trying 5d")
                try:
                    hist = ticker.history(period="5d", interval="1d")
                except Exception:
                    logger.error(f"Could not get any history for {symbol}")
                    return None

            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Берем последнюю строку
            latest = hist.iloc[-1]

            # Получаем текущую цену из info или из исторических данных
            try:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                if current_price is None:
                    current_price = float(latest["Close"])
                else:
                    current_price = float(current_price)
            except (ValueError, TypeError):
                current_price = float(latest["Close"])

            # Вычисляем изменение
            if len(hist) > 1:
                try:
                    prev_close = float(hist.iloc[-2]["Close"])
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                except (IndexError, ValueError, TypeError):
                    change = 0
                    change_percent = 0
            else:
                change = 0
                change_percent = 0

            return {
                "price": Decimal(str(current_price)),
                "volume": int(float(latest["Volume"])) if not hist.empty and not pd.isna(latest["Volume"]) else None,
                "high": Decimal(str(float(latest["High"]))) if not hist.empty and not pd.isna(latest["High"]) else None,
                "low": Decimal(str(float(latest["Low"]))) if not hist.empty and not pd.isna(latest["Low"]) else None,
                "open_price": Decimal(str(float(latest["Open"]))) if not hist.empty and not pd.isna(latest["Open"]) else None,
                "change": Decimal(str(change)),
                "change_percent": Decimal(str(round(change_percent, 4))),
                "timestamp": datetime.now(),
                # 'name' не нужен для MarketData - это поле только в Symbol
            }
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[List[Dict]]:
        """
        Получает исторические данные
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (только для yfinance)
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (yfinance)
                  1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M (Bybit)
        """
        # Определяем источник данных
        use_bybit = False
        if self.data_source == "bybit":
            use_bybit = True
        elif self.data_source == "auto":
            if any(suffix in symbol.upper() for suffix in ["USDT", "USDC", "BTC", "ETH", "BNB", "BUSD"]):
                use_bybit = True

        # Пробуем Bybit если нужно
        if use_bybit and self.bybit_service:
            bybit_symbol = self.bybit_service.normalize_symbol(symbol)
            # Маппинг period в limit для Bybit
            period_map = {"1d": 1440, "5d": 7200, "1mo": 43200, "3mo": 129600, "1y": 525600}
            limit = period_map.get(period, 200)
            # Маппинг interval для Bybit
            interval_map = {
                "1m": "1", "5m": "5", "15m": "15", "30m": "30",
                "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1mo": "M"
            }
            bybit_interval = interval_map.get(interval, "60")
            data = self.bybit_service.get_historical_data(bybit_symbol, interval=bybit_interval, limit=limit)
            if data:
                return data

        # Используем yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return None

            result = []
            for idx, row in hist.iterrows():
                result.append({
                    "timestamp": idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

            return result
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}", exc_info=True)
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Проверяет, существует ли символ"""
        # Пробуем Bybit если нужно
        if self.bybit_service:
            bybit_symbol = self.bybit_service.normalize_symbol(symbol)
            if self.bybit_service.validate_symbol(bybit_symbol):
                return True

        # Пробуем yfinance
        try:
            ticker = yf.Ticker(symbol)
            # Пробуем получить исторические данные (более надежно чем info)
            try:
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    return True
            except Exception as hist_error:
                logger.debug(f"Could not get history for {symbol}: {hist_error}")
            
            # Если история не сработала, пробуем info
            try:
                info = ticker.info
                # Если есть хотя бы какая-то информация, считаем символ валидным
                if info and len(info) > 0:
                    return True
            except Exception as info_error:
                logger.debug(f"Could not get info for {symbol}: {info_error}")
            
            return False
        except Exception as e:
            logger.debug(f"Error validating symbol {symbol}: {str(e)}")
            return False


def get_market_data_service(data_source: Optional[str] = None) -> MarketDataService:
    """
    Helper функция для получения MarketDataService с настройками из Django settings
    """
    if data_source is None:
        from django.conf import settings
        data_source = getattr(settings, "MARKET_DATA_SOURCE", "auto")
    return MarketDataService(data_source=data_source)
