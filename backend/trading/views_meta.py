"""
Новые эндпоинты для мета-модели с фильтром активов
"""
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from django.db import transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from trading.models import Symbol, MarketData, TradingDecision, AgentStatus, Account, Position, Trade, AgentLog
from trading.serializers import TradingDecisionSerializer, TradeSerializer
from trading.agents.meta_model_selector import MetaModelSelector
from trading.agents.asset_filter import get_asset_filter
from trading.agents import MarketMonitoringAgent
from trading.services.binance_api import BinanceAPIService
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetaModelAgentView(APIView):
    """Эндпоинт для работы с мета-моделью (один агент - полный pipeline)"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        Запускает полный pipeline: Market Monitoring → Decision Making → Execution
        
        Body:
        {
            "symbol": "BTCUSDT",  # Обязательно
            "execute": true/false  # Выполнять ли сделку (по умолчанию false)
        }
        """
        try:
            symbol_code = request.data.get("symbol", "").upper()
            execute = request.data.get("execute", False)
            
            if not symbol_code:
                return Response(
                    {"detail": "symbol is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Проверяем фильтр активов
            asset_filter = get_asset_filter()
            if not asset_filter.is_approved(symbol_code):
                return Response(
                    {
                        "detail": f"Asset {symbol_code} is not approved for trading",
                        "reason": asset_filter.blacklisted_assets.get(symbol_code, {}).get('reason', 'Not in approved list'),
                        "approved_assets": asset_filter.get_approved_list()
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Получаем или создаем символ
            symbol, _ = Symbol.objects.get_or_create(
                user=request.user,
                symbol=symbol_code,
                defaults={"name": symbol_code, "is_active": True}
            )
            
            # Получаем данные рынка
            binance_service = BinanceAPIService()
            historical_data = binance_service.get_historical_data(
                symbol=symbol_code,
                interval="1h",
                days=30
            )
            
            if not historical_data:
                return Response(
                    {"detail": f"Could not fetch data for {symbol_code}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Конвертируем в DataFrame
            df_data = []
            for candle in historical_data:
                df_data.append({
                    "Open": float(candle["open"]),
                    "High": float(candle["high"]),
                    "Low": float(candle["low"]),
                    "Close": float(candle["close"]),
                    "Volume": float(candle["volume"]),
                })
            
            df = pd.DataFrame(df_data)
            df.index = [candle["timestamp"] for candle in historical_data]
            
            # Market Monitoring Agent
            market_agent = MarketMonitoringAgent(
                ticker=symbol_code,
                interval="1h",
                period="1mo",
                enable_cache=False
            )
            market_agent.raw_data = df
            data_with_indicators = market_agent.compute_indicators(df)
            preprocessed_data = market_agent.preprocess(data_with_indicators)
            
            if preprocessed_data.empty:
                return Response(
                    {"detail": "No data after preprocessing"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Мета-модель для принятия решения
            meta_selector = MetaModelSelector()
            
            # Подготавливаем данные для обучения (используем все доступные данные)
            X, y = self._prepare_training_data(preprocessed_data)
            
            if X is None or len(X) < 20:
                return Response(
                    {"detail": "Not enough data for training"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Обучаем модели
            meta_selector.train_base_models(symbol_code, X, y)
            
            # Получаем последнюю свечу для предсказания
            last_row = preprocessed_data.iloc[-1]
            prev_row = preprocessed_data.iloc[-2] if len(preprocessed_data) > 1 else None
            
            # Извлекаем фичи
            features = self._extract_features(last_row, prev_row)
            
            # Предсказание через мета-модель
            prediction, confidence, regime = meta_selector.predict_ensemble_with_regime(
                symbol_code, features, preprocessed_data
            )
            
            # Конвертируем предсказание в действие
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = action_map.get(prediction, "HOLD")
            
            # Сохраняем решение
            latest_market_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()
            
            decision = TradingDecision.objects.create(
                user=request.user,
                symbol=symbol,
                decision=action,
                confidence=Decimal(str(confidence * 100)),
                market_data=latest_market_data,
                reasoning=f"Meta-model prediction (regime: {regime}, confidence: {confidence:.2%})",
                metadata={
                    "model_type": "meta_model",
                    "regime": regime,
                    "confidence": float(confidence),
                    "price": float(last_row.get('close', 0.0)),
                }
            )
            
            result = {
                "success": True,
                "symbol": symbol_code,
                "decision": {
                    "action": action,
                    "confidence": float(confidence),
                    "regime": regime,
                    "price": float(last_row.get('close', 0.0)),
                    "decision_id": decision.id
                },
                "market_data": {
                    "timestamp": preprocessed_data.index[-1].isoformat() if hasattr(preprocessed_data.index[-1], 'isoformat') else str(preprocessed_data.index[-1]),
                    "close": float(last_row.get('close', 0.0)),
                    "volume": float(last_row.get('volume', 0.0)),
                }
            }
            
            # Если нужно выполнить сделку
            if execute and action != "HOLD":
                execution_result = self._execute_trade(request.user, symbol, decision, action, float(last_row.get('close', 0.0)))
                result["execution"] = execution_result
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in MetaModelAgentView: {e}", exc_info=True)
            return Response(
                {"detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Подготавливает данные для обучения"""
        if len(data) < 200:
            return None, None
        
        X = []
        y = []
        lookahead_periods = 6
        min_profit_threshold = 0.5
        
        for i in range(len(data) - lookahead_periods):
            current_row = data.iloc[i]
            
            features = []
            features.append(float(current_row.get('close', 0.0)))
            features.append(float(current_row.get('volume', 0.0)))
            features.append(float(current_row.get('price_change', 0.0)))
            features.append(float(current_row.get('sma10', 0.0)))
            features.append(float(current_row.get('sma20', 0.0)))
            features.append(float(current_row.get('rsi14', 50.0)))
            features.append(float(current_row.get('macd', 0.0)))
            features.append(float(current_row.get('macd_hist', 0.0)))
            features.append(float(current_row.get('volatility', 0.0)))
            
            sma10 = current_row.get('sma10', 0.0)
            sma20 = current_row.get('sma20', 0.0)
            if sma10 > sma20:
                trend_encoded = 1.0
            elif sma10 < sma20:
                trend_encoded = -1.0
            else:
                trend_encoded = 0.0
            features.append(trend_encoded)
            
            rsi = current_row.get('rsi14', 50.0)
            strength = abs(rsi - 50) / 50
            features.append(float(strength))
            
            if rsi > 70:
                rsi_encoded = 1.0
            elif rsi < 30:
                rsi_encoded = -1.0
            else:
                rsi_encoded = 0.0
            features.append(rsi_encoded)
            
            sma_cross = 0.0
            if i > 0:
                prev_row = data.iloc[i - 1]
                prev_sma10 = prev_row.get('sma10', 0.0)
                prev_sma20 = prev_row.get('sma20', 0.0)
                if (prev_sma10 <= prev_sma20 and sma10 > sma20) or \
                   (prev_sma10 >= prev_sma20 and sma10 < sma20):
                    sma_cross = 1.0
            features.append(sma_cross)
            
            # Генерация метки
            current_price = current_row.get('close', 0.0)
            if current_price > 0 and i + lookahead_periods < len(data):
                future_row = data.iloc[i + lookahead_periods]
                future_price = future_row.get('close', 0.0)
                
                if future_price > 0:
                    price_change_pct = ((future_price - current_price) / current_price) * 100
                    if price_change_pct > min_profit_threshold:
                        label = 2  # BUY
                    elif price_change_pct < -min_profit_threshold:
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                else:
                    label = 1
            else:
                label = 1
            
            X.append(features)
            y.append(label)
        
        if len(X) < 20:
            return None, None
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, row: pd.Series, prev_row: pd.Series = None) -> np.ndarray:
        """Извлекает фичи из строки данных"""
        features = []
        features.append(float(row.get('close', 0.0)))
        features.append(float(row.get('volume', 0.0)))
        features.append(float(row.get('price_change', 0.0)))
        features.append(float(row.get('sma10', 0.0)))
        features.append(float(row.get('sma20', 0.0)))
        features.append(float(row.get('rsi14', 50.0)))
        features.append(float(row.get('macd', 0.0)))
        features.append(float(row.get('macd_hist', 0.0)))
        features.append(float(row.get('volatility', 0.0)))
        
        sma10 = row.get('sma10', 0.0)
        sma20 = row.get('sma20', 0.0)
        if sma10 > sma20:
            trend_encoded = 1.0
        elif sma10 < sma20:
            trend_encoded = -1.0
        else:
            trend_encoded = 0.0
        features.append(trend_encoded)
        
        rsi = row.get('rsi14', 50.0)
        strength = abs(rsi - 50) / 50
        features.append(float(strength))
        
        if rsi > 70:
            rsi_encoded = 1.0
        elif rsi < 30:
            rsi_encoded = -1.0
        else:
            rsi_encoded = 0.0
        features.append(rsi_encoded)
        
        sma_cross = 0.0
        if prev_row is not None:
            prev_sma10 = prev_row.get('sma10', 0.0)
            prev_sma20 = prev_row.get('sma20', 0.0)
            if (prev_sma10 <= prev_sma20 and sma10 > sma20) or \
               (prev_sma10 >= prev_sma20 and sma10 < sma20):
                sma_cross = 1.0
        features.append(sma_cross)
        
        return np.array(features).reshape(1, -1)
    
    def _execute_trade(self, user, symbol: Symbol, decision: TradingDecision, action: str, price: float):
        """Выполняет сделку"""
        try:
            account, _ = Account.objects.get_or_create(
                user=user,
                defaults={
                    "balance": Decimal("10000.00"),
                    "free_cash": Decimal("10000.00"),
                    "initial_balance": Decimal("10000.00"),
                }
            )
            
            if action == "BUY":
                # Проверяем баланс
                if account.free_cash <= 0:
                    return {"status": "failed", "reason": "Insufficient balance"}
                
                # Используем 90% доступного баланса
                trade_amount = account.free_cash * Decimal("0.9")
                quantity = trade_amount / Decimal(str(price))
                
                # Создаем позицию
                position = Position.objects.create(
                    user=user,
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=Decimal(str(price)),
                    current_price=Decimal(str(price)),
                    side="LONG"
                )
                
                # Обновляем баланс
                account.free_cash -= trade_amount
                account.save()
                
                # Создаем сделку
                trade = Trade.objects.create(
                    user=user,
                    symbol=symbol,
                    decision=decision,
                    side="BUY",
                    quantity=quantity,
                    price=Decimal(str(price)),
                    executed_at=timezone.now()
                )
                
                return {
                    "status": "executed",
                    "action": "BUY",
                    "quantity": float(quantity),
                    "price": price,
                    "trade_id": trade.id,
                    "position_id": position.id
                }
            
            elif action == "SELL":
                # Ищем открытую позицию
                position = Position.objects.filter(
                    user=user,
                    symbol=symbol,
                    is_open=True
                ).first()
                
                if not position:
                    return {"status": "failed", "reason": "No open position"}
                
                # Закрываем позицию
                sell_amount = position.quantity * Decimal(str(price))
                account.free_cash += sell_amount
                account.save()
                
                # Создаем сделку
                trade = Trade.objects.create(
                    user=user,
                    symbol=symbol,
                    decision=decision,
                    side="SELL",
                    quantity=position.quantity,
                    price=Decimal(str(price)),
                    executed_at=timezone.now()
                )
                
                # Закрываем позицию
                position.is_open = False
                position.exit_price = Decimal(str(price))
                position.exit_time = timezone.now()
                position.save()
                
                return {
                    "status": "executed",
                    "action": "SELL",
                    "quantity": float(position.quantity),
                    "price": price,
                    "trade_id": trade.id,
                    "position_id": position.id
                }
            
            return {"status": "skipped", "reason": "HOLD action"}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return {"status": "error", "detail": str(e)}


class TradingChartDataView(APIView):
    """Эндпоинт для получения данных графика (покупки/продажи)"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Возвращает данные для графика торговли
        
        Query params:
        - symbol: код криптовалюты (обязательно)
        - days: количество дней истории (по умолчанию 30)
        """
        try:
            symbol_code = request.query_params.get("symbol", "").upper()
            days = int(request.query_params.get("days", 30))
            
            if not symbol_code:
                return Response(
                    {"detail": "symbol parameter is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Получаем символ
            try:
                symbol = Symbol.objects.get(user=request.user, symbol=symbol_code)
            except Symbol.DoesNotExist:
                return Response(
                    {"detail": f"Symbol {symbol_code} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Получаем исторические данные рынка
            binance_service = BinanceAPIService()
            historical_data = binance_service.get_historical_data(
                symbol=symbol_code,
                interval="1h",
                days=days
            )
            
            if not historical_data:
                return Response(
                    {"detail": f"Could not fetch market data for {symbol_code}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Формируем данные свечей
            candles = []
            for candle in historical_data:
                candles.append({
                    "timestamp": candle["timestamp"].isoformat() if hasattr(candle["timestamp"], 'isoformat') else str(candle["timestamp"]),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                })
            
            # Получаем сделки пользователя
            since = timezone.now() - timedelta(days=days)
            trades = Trade.objects.filter(
                user=request.user,
                symbol=symbol,
                executed_at__gte=since
            ).order_by("executed_at")
            
            # Формируем данные о сделках
            trade_markers = []
            for trade in trades:
                trade_markers.append({
                    "timestamp": trade.executed_at.isoformat(),
                    "side": trade.side,
                    "price": float(trade.price),
                    "quantity": float(trade.quantity),
                    "trade_id": trade.id,
                    "decision_id": trade.decision.id if trade.decision else None,
                    "confidence": float(trade.decision.confidence) if trade.decision else None,
                })
            
            # Получаем решения (включая HOLD)
            decisions = TradingDecision.objects.filter(
                user=request.user,
                symbol=symbol,
                created_at__gte=since
            ).order_by("created_at")
            
            decision_markers = []
            for decision in decisions:
                decision_markers.append({
                    "timestamp": decision.created_at.isoformat(),
                    "action": decision.decision,
                    "confidence": float(decision.confidence),
                    "decision_id": decision.id,
                    "regime": decision.metadata.get("regime") if decision.metadata else None,
                })
            
            return Response({
                "symbol": symbol_code,
                "candles": candles,
                "trades": trade_markers,
                "decisions": decision_markers,
                "summary": {
                    "total_trades": len(trade_markers),
                    "buy_trades": len([t for t in trade_markers if t["side"] == "BUY"]),
                    "sell_trades": len([t for t in trade_markers if t["side"] == "SELL"]),
                    "total_decisions": len(decision_markers),
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in TradingChartDataView: {e}", exc_info=True)
            return Response(
                {"detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ApprovedAssetsView(APIView):
    """Эндпоинт для получения списка одобренных активов"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Возвращает список одобренных и заблокированных активов"""
        asset_filter = get_asset_filter()
        
        approved = []
        for symbol in asset_filter.get_approved_list():
            config = asset_filter.get_trading_config(symbol)
            asset_info = asset_filter.approved_assets[symbol]
            approved.append({
                "symbol": symbol,
                "category": asset_info['category'],
                "historical_score": asset_info['score'],
                "win_rate": asset_info.get('win_rate'),
                "trades": asset_info.get('trades'),
                "config": config
            })
        
        blacklisted = []
        for symbol in asset_filter.get_blacklisted_list():
            info = asset_filter.blacklisted_assets[symbol]
            blacklisted.append({
                "symbol": symbol,
                "reason": info['reason'],
                "score": info.get('score')
            })
        
        return Response({
            "approved": approved,
            "blacklisted": blacklisted,
            "total_approved": len(approved),
            "total_blacklisted": len(blacklisted)
        }, status=status.HTTP_200_OK)

