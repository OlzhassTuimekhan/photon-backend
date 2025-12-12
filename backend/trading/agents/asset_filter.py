"""
Фильтр активов для мета-модели
Определяет, какие активы подходят для торговли с мета-моделью
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class AssetFilter:
    """Фильтрует активы на основе их характеристик и исторической производительности"""
    
    def __init__(self):
        # Белый список активов, которые показали хорошие результаты
        # Основано на результатах тестирования мета-модели
        self.approved_assets = {
            'LINKUSDT': {
                'score': 21.58,  # Доходность в тестах
                'win_rate': 37.50,
                'trades': 16,
                'category': 'top_performer'
            },
            'BTCUSDT': {
                'score': 6.98,
                'win_rate': 40.00,
                'trades': 10,
                'category': 'stable'
            },
            'AVAXUSDT': {
                'score': 2.03,
                'win_rate': 21.43,
                'trades': 14,
                'category': 'stable'
            },
            'ETHUSDT': {
                'score': 0.04,
                'win_rate': 40.00,
                'trades': 20,
                'category': 'stable'
            },
            # MATICUSDT и SOLUSDT близки к нулю, можно добавить с консервативными настройками
            'MATICUSDT': {
                'score': -0.67,
                'win_rate': 27.78,
                'trades': 18,
                'category': 'marginal'
            },
            'SOLUSDT': {
                'score': -0.96,
                'win_rate': 33.33,
                'trades': 12,
                'category': 'marginal'
            }
        }
        
        # Черный список активов, которые показали плохие результаты
        self.blacklisted_assets = {
            'XRPUSDT': {
                'reason': 'Very low win rate (5.56%)',
                'score': -13.96
            },
            'DOGEUSDT': {
                'reason': 'Low win rate (20.00%)',
                'score': -9.71
            },
            'BNBUSDT': {
                'reason': 'Low win rate (25.00%)',
                'score': -5.01
            },
            'ADAUSDT': {
                'reason': 'Low win rate (25.00%)',
                'score': -4.75
            }
        }
        
        # Минимальные требования для одобрения актива
        self.min_requirements = {
            'min_win_rate': 30.0,  # Минимальный винрейт 30%
            'min_score': -2.0,      # Минимальная доходность -2%
            'min_trades': 5         # Минимальное количество сделок для статистики
        }
    
    def is_approved(self, symbol: str) -> bool:
        """
        Проверяет, одобрен ли актив для торговли с мета-моделью
        
        Args:
            symbol: Символ актива (например, 'BTCUSDT')
        
        Returns:
            True если актив одобрен, False если нет
        """
        # Проверяем черный список
        if symbol in self.blacklisted_assets:
            logger.info(f"Asset {symbol} is blacklisted: {self.blacklisted_assets[symbol]['reason']}")
            return False
        
        # Проверяем белый список
        if symbol in self.approved_assets:
            asset_info = self.approved_assets[symbol]
            logger.debug(f"Asset {symbol} is approved (category: {asset_info['category']}, score: {asset_info['score']}%)")
            return True
        
        # Если актив не в списках, по умолчанию не одобряем
        logger.warning(f"Asset {symbol} is not in approved list, defaulting to NOT approved")
        return False
    
    def get_trading_config(self, symbol: str) -> Dict:
        """
        Возвращает конфигурацию торговли для актива
        
        Args:
            symbol: Символ актива
        
        Returns:
            Словарь с параметрами торговли
        """
        if symbol not in self.approved_assets:
            return {
                'enabled': False,
                'reason': 'Asset not approved'
            }
        
        asset_info = self.approved_assets[symbol]
        category = asset_info['category']
        
        # Конфигурация в зависимости от категории
        if category == 'top_performer':
            # Для топ-перформеров - агрессивные настройки
            return {
                'enabled': True,
                'max_position_size': 0.9,  # 90% баланса
                'min_confidence': 0.5,      # Минимальная уверенность 50%
                'use_meta_model': True,     # Использовать мета-модель
                'risk_level': 'medium'
            }
        elif category == 'stable':
            # Для стабильных - умеренные настройки
            return {
                'enabled': True,
                'max_position_size': 0.8,    # 80% баланса
                'min_confidence': 0.55,     # Минимальная уверенность 55%
                'use_meta_model': True,
                'risk_level': 'low'
            }
        elif category == 'marginal':
            # Для маржинальных - консервативные настройки
            return {
                'enabled': True,
                'max_position_size': 0.6,    # 60% баланса
                'min_confidence': 0.6,       # Минимальная уверенность 60%
                'use_meta_model': True,
                'risk_level': 'low'
            }
        else:
            # По умолчанию - консервативные настройки
            return {
                'enabled': True,
                'max_position_size': 0.5,
                'min_confidence': 0.65,
                'use_meta_model': True,
                'risk_level': 'low'
            }
    
    def evaluate_asset(self, symbol: str, historical_performance: Dict) -> Tuple[bool, str]:
        """
        Оценивает актив на основе исторической производительности
        
        Args:
            symbol: Символ актива
            historical_performance: Словарь с метриками производительности
                {
                    'return_pct': float,  # Доходность в процентах
                    'win_rate': float,   # Винрейт в процентах
                    'trades': int,       # Количество сделок
                    'max_drawdown': float # Максимальная просадка (опционально)
                }
        
        Returns:
            (is_approved, reason) - одобрен ли актив и причина
        """
        return_pct = historical_performance.get('return_pct', 0.0)
        win_rate = historical_performance.get('win_rate', 0.0)
        trades = historical_performance.get('trades', 0)
        
        # Проверяем минимальные требования
        if trades < self.min_requirements['min_trades']:
            return False, f"Not enough trades ({trades} < {self.min_requirements['min_trades']})"
        
        if win_rate < self.min_requirements['min_win_rate']:
            return False, f"Win rate too low ({win_rate:.2f}% < {self.min_requirements['min_win_rate']}%)"
        
        if return_pct < self.min_requirements['min_score']:
            return False, f"Return too low ({return_pct:.2f}% < {self.min_requirements['min_score']}%)"
        
        # Если все требования выполнены
        return True, f"Asset meets requirements (return: {return_pct:.2f}%, win_rate: {win_rate:.2f}%)"
    
    def get_approved_list(self) -> List[str]:
        """Возвращает список одобренных активов"""
        return list(self.approved_assets.keys())
    
    def get_blacklisted_list(self) -> List[str]:
        """Возвращает список заблокированных активов"""
        return list(self.blacklisted_assets.keys())
    
    def add_to_approved(self, symbol: str, performance_data: Dict):
        """Добавляет актив в белый список"""
        self.approved_assets[symbol] = performance_data
        logger.info(f"Added {symbol} to approved assets")
    
    def add_to_blacklist(self, symbol: str, reason: str, score: float = None):
        """Добавляет актив в черный список"""
        self.blacklisted_assets[symbol] = {
            'reason': reason,
            'score': score
        }
        logger.info(f"Added {symbol} to blacklist: {reason}")


# Глобальный экземпляр фильтра
_asset_filter = None

def get_asset_filter() -> AssetFilter:
    """Возвращает глобальный экземпляр фильтра активов"""
    global _asset_filter
    if _asset_filter is None:
        _asset_filter = AssetFilter()
    return _asset_filter

