"""
Decision-Making Agent

This module implements a decision-making agent that:
- Receives market data from MarketMonitoringAgent
- Uses AI model to make trading decisions (BUY/SELL/HOLD)
- Applies risk management rules
- Sends decisions to ExecutionAgent
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json
from collections import deque
import pickle
import os

# AI/ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using rule-based fallback.")

# Configure logging
logger = logging.getLogger(__name__)


class DecisionMakingAgent:
    """
    Decision-making agent that uses AI to make trading decisions.
    
    Receives market data from MarketMonitoringAgent and makes decisions
    based on technical indicators, market analysis, and risk management rules.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        risk_tolerance: str = "medium",  # "low", "medium", "high"
        max_position_size: float = 0.1,  # Max 10% of portfolio per trade
        min_confidence: float = 0.6,  # Minimum confidence for action
        enable_ai: bool = True,
        model_path: Optional[str] = None,
        history_size: int = 1000,
        use_historical_training: bool = True,  # Use real historical data for training
        training_ticker: Optional[str] = None,  # Ticker for historical training
        training_period: str = "1y",  # Period for historical training data
        user_id: Optional[int] = None,  # ID пользователя для доступа к БД для обучения
        enable_continuous_learning: bool = True  # Включить постоянное обучение
    ):
        """
        Initialize decision-making agent.
        
        Args:
            model_type: Type of AI model ("random_forest", "gradient_boosting", "rule_based")
            risk_tolerance: Risk tolerance level ("low", "medium", "high")
            max_position_size: Maximum position size as fraction of portfolio
            min_confidence: Minimum confidence threshold for taking action
            enable_ai: Whether to use AI model (if False, uses rule-based)
            model_path: Path to saved model (if None, trains new model)
            history_size: Size of decision history
            use_historical_training: If True, train on real historical data (default: True)
            training_ticker: Ticker for historical training data (default: "SPY")
            training_period: Period for historical training ("1y", "6mo", "3mo", "1mo")
            user_id: ID пользователя для доступа к БД для обучения на реальных данных
        """
        self.model_type = model_type
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.enable_ai = enable_ai and SKLEARN_AVAILABLE
        self.use_historical_training = use_historical_training
        self.training_ticker = training_ticker
        self.training_period = training_period
        self.user_id = user_id  # Сохраняем user_id для доступа к БД
        
        # AI Model
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model_path = model_path
        self.is_trained = False
        
        # Continuous learning settings
        self.enable_continuous_learning = enable_continuous_learning  # Включить постоянное обучение
        self.retrain_interval = 10  # Переобучать каждые N новых решений
        self.retrain_min_samples = 50  # Минимум новых samples для переобучения
        self.decisions_since_retrain = 0  # Счетчик решений с последнего переобучения
        self.last_retrain_time = None  # Время последнего переобучения
        
        # Decision history
        self.history_size = history_size
        self.decision_history: deque = deque(maxlen=history_size)
        
        # Portfolio state (simulated)
        self.portfolio = {
            "cash": 10000.0,  # Starting cash
            "positions": {},  # {ticker: {"quantity": int, "avg_price": float}}
            "total_value": 10000.0
        }
        
        # Risk management parameters
        self.risk_params = self._get_risk_params()
        
        # Initialize model
        if self.enable_ai:
            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                logger.info("AI model will be trained on first decision")
        else:
            logger.info("Using rule-based decision making (AI not available)")
    
    def _get_risk_params(self) -> Dict:
        """Get risk management parameters based on risk tolerance."""
        params = {
            "low": {
                "max_loss_per_trade": 0.01,  # 1% max loss
                "stop_loss": 0.02,  # 2% stop loss
                "take_profit": 0.05,  # 5% take profit
                "max_drawdown": 0.1  # 10% max drawdown
            },
            "medium": {
                "max_loss_per_trade": 0.02,  # 2% max loss
                "stop_loss": 0.03,  # 3% stop loss
                "take_profit": 0.08,  # 8% take profit
                "max_drawdown": 0.15  # 15% max drawdown
            },
            "high": {
                "max_loss_per_trade": 0.03,  # 3% max loss
                "stop_loss": 0.05,  # 5% stop loss
                "take_profit": 0.12,  # 12% take profit
                "max_drawdown": 0.25  # 25% max drawdown
            }
        }
        return params.get(self.risk_tolerance, params["medium"])
    
    def receive_market_data(self, market_data: Dict) -> Dict:
        """
        Receives market data from MarketMonitoringAgent and makes decision.
        
        Args:
            market_data: Dictionary with market data in standardized format:
                {
                    "timestamp": str,
                    "ticker": str,
                    "ohlcv": {...},
                    "indicators": {...},
                    "analysis": {...},
                    "meta": {...}
                }
        
        Returns:
            Dictionary with trading decision:
                {
                    "action": "BUY" | "SELL" | "HOLD",
                    "ticker": str,
                    "confidence": float (0.0-1.0),
                    "reasoning": str,
                    "quantity": int (if action is BUY/SELL),
                    "price": float,
                    "timestamp": str,
                    "risk_score": float
                }
        """
        try:
            logger.info(f"Received market data for {market_data.get('ticker', 'UNKNOWN')}")
            
            # Extract features for AI model
            features = self._extract_features(market_data)
            
            # Make decision using AI or rules
            # ВАЖНО: AI модель используется по умолчанию, rule-based только как fallback
            if self.enable_ai and self.is_trained:
                logger.info(f"Using AI model ({self.model_type}) for decision making")
                decision = self._make_ai_decision(features, market_data)
            else:
                if not self.enable_ai:
                    logger.warning("AI disabled, using rule-based fallback")
                elif not self.is_trained:
                    logger.warning("AI model not trained yet, using rule-based fallback (will train after this decision)")
                decision = self._make_rule_based_decision(features, market_data)
            
            # Apply risk management
            decision = self._apply_risk_management(decision, market_data)
            
            # ПРИМЕЧАНИЕ: Для симуляции разрешаем SELL даже без открытых позиций
            # Это ускоряет сбор данных для обучения модели
            # Проверка на открытые позиции убрана - ExecutionAgent обработает SELL в любом случае
            
            # Store in history
            self.decision_history.append({
                "timestamp": decision["timestamp"],
                "ticker": decision["ticker"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "price": decision["price"]
            })
            
            # Train model if needed (initial training)
            # ВАЖНО: Модель обучается автоматически при первом использовании
            if self.enable_ai and not self.is_trained:
                logger.info("Training AI model on first use with historical data...")
                self._train_initial_model()
                logger.info(f"AI model trained successfully. Model type: {self.model_type}, Trained: {self.is_trained}")
                # После обучения пересчитываем решение с использованием AI модели
                if self.is_trained:
                    logger.info("Recomputing decision with trained AI model...")
                    decision = self._make_ai_decision(features, market_data)
                    # Применяем risk management к новому решению
                    decision = self._apply_risk_management(decision, market_data)
            
            # Continuous learning: retrain periodically
            if self.enable_ai and self.is_trained and self.enable_continuous_learning:
                self.decisions_since_retrain += 1
                if self.decisions_since_retrain >= self.retrain_interval:
                    logger.info(f"Triggering continuous learning retrain (decisions since last retrain: {self.decisions_since_retrain})")
                    try:
                        self._retrain_with_real_data()
                        self.decisions_since_retrain = 0
                    except Exception as e:
                        logger.warning(f"Continuous learning retrain failed: {e}, continuing with current model")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return self._create_hold_decision(market_data, f"Error: {str(e)}")
    
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features from market data for AI model."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        ohlcv = market_data.get("ohlcv", {})
        
        # Feature vector
        features = []
        
        # Price features
        features.append(ohlcv.get("close", 0.0))
        features.append(ohlcv.get("volume", 0.0))
        features.append(indicators.get("price_change", 0.0))
        
        # Technical indicators
        features.append(indicators.get("sma10", 0.0))
        features.append(indicators.get("sma20", 0.0))
        features.append(indicators.get("rsi14", 50.0))
        features.append(indicators.get("macd", 0.0))
        features.append(indicators.get("macd_hist", 0.0))
        features.append(indicators.get("volatility", 0.0))
        
        # Analysis features
        trend = analysis.get("trend", "sideways")
        trend_encoded = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}.get(trend, 0.0)
        features.append(trend_encoded)
        features.append(analysis.get("strength", 0.5))
        
        # RSI state
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        rsi_encoded = {"overbought": 1.0, "oversold": -1.0, "neutral": 0.0}.get(rsi_state, 0.0)
        features.append(rsi_encoded)
        
        # SMA crossover
        sma_cross = analysis.get("signals", {}).get("sma_cross", 0)
        features.append(float(sma_cross))
        
        return np.array(features).reshape(1, -1)
    
    def _make_ai_decision(self, features: np.ndarray, market_data: Dict) -> Dict:
        """Make decision using trained AI model."""
        try:
            # Scale features
            if self.scaler is None:
                raise ValueError("Scaler not initialized")
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to action
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = action_map.get(prediction, "HOLD")
            
            # Get confidence (probability of predicted class)
            confidence = float(max(probabilities))
            
            # Generate reasoning
            reasoning = self._generate_reasoning(market_data, action, confidence)
            
            ticker = market_data.get("ticker", "UNKNOWN")
            price = market_data.get("ohlcv", {}).get("close", 0.0)
            
            return {
                "action": action,
                "ticker": ticker,
                "confidence": confidence,
                "reasoning": reasoning,
                "quantity": self._calculate_quantity(action, price, confidence),
                "price": price,
                "timestamp": datetime.now().isoformat() + "Z",
                "risk_score": self._calculate_risk_score(market_data),
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error in AI decision: {e}")
            return self._make_rule_based_decision(features, market_data)
    
    def _make_rule_based_decision(self, features: np.ndarray, market_data: Dict) -> Dict:
        """Make decision using rule-based logic (fallback or when AI disabled)."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        ohlcv = market_data.get("ohlcv", {})
        
        ticker = market_data.get("ticker", "UNKNOWN")
        price = ohlcv.get("close", 0.0)
        
        # Extract key signals
        trend = analysis.get("trend", "sideways")
        rsi = indicators.get("rsi14", 50.0)
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        macd_hist = indicators.get("macd_hist", 0.0)
        sma_cross = analysis.get("signals", {}).get("sma_cross", 0)
        strength = analysis.get("strength", 0.5)
        
        # Decision logic
        buy_signals = 0
        sell_signals = 0
        
        # Trend signals
        if trend == "bull":
            buy_signals += 2
        elif trend == "bear":
            sell_signals += 2
        
        # RSI signals
        if rsi_state == "oversold" and rsi < 30:
            buy_signals += 2
        elif rsi_state == "overbought" and rsi > 70:
            sell_signals += 2
        
        # MACD signals
        if macd_hist > 0:
            buy_signals += 1
        elif macd_hist < 0:
            sell_signals += 1
        
        # SMA crossover
        if sma_cross:
            if trend == "bull":
                buy_signals += 1
            elif trend == "bear":
                sell_signals += 1
        
        # Strength multiplier
        strength_mult = strength
        
        # Calculate confidence
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            action = "HOLD"
            confidence = 0.5
        elif buy_signals > sell_signals:
            action = "BUY"
            confidence = min(0.9, 0.5 + (buy_signals / 10.0) * strength_mult)
        elif sell_signals > buy_signals:
            action = "SELL"
            confidence = min(0.9, 0.5 + (sell_signals / 10.0) * strength_mult)
        else:
            action = "HOLD"
            confidence = 0.5
        
        reasoning = self._generate_reasoning(market_data, action, confidence)
        
        return {
            "action": action,
            "ticker": ticker,
            "confidence": confidence,
            "reasoning": reasoning,
            "quantity": self._calculate_quantity(action, price, confidence),
            "price": price,
            "timestamp": datetime.now().isoformat() + "Z",
            "risk_score": self._calculate_risk_score(market_data),
            "model_type": "rule_based"
        }
    
    def _generate_reasoning(self, market_data: Dict, action: str, confidence: float) -> str:
        """Generate human-readable reasoning for decision."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        
        trend = analysis.get("trend", "sideways")
        rsi = indicators.get("rsi14", 50.0)
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        strength = analysis.get("strength", 0.5)
        
        reasons = []
        
        if action == "BUY":
            reasons.append(f"Trend: {trend.upper()}")
            if rsi_state == "oversold":
                reasons.append(f"RSI oversold ({rsi:.1f})")
            if strength > 0.6:
                reasons.append(f"Strong trend (strength: {strength:.2f})")
        elif action == "SELL":
            reasons.append(f"Trend: {trend.upper()}")
            if rsi_state == "overbought":
                reasons.append(f"RSI overbought ({rsi:.1f})")
            if strength > 0.6:
                reasons.append(f"Strong trend (strength: {strength:.2f})")
        else:
            reasons.append("Mixed signals or low confidence")
            reasons.append(f"Trend: {trend}, RSI: {rsi:.1f}")
        
        reasoning = f"{action} decision (confidence: {confidence:.2f}). " + ". ".join(reasons)
        return reasoning
    
    def _calculate_quantity(self, action: str, price: float, confidence: float) -> int:
        """Calculate quantity to trade based on confidence and risk management."""
        if action == "HOLD":
            return 0
        
        # Calculate position size based on confidence and risk tolerance
        portfolio_value = self.portfolio.get("total_value", 10000.0)
        max_position_value = portfolio_value * self.max_position_size
        
        # Adjust by confidence
        confidence_multiplier = confidence
        position_value = max_position_value * confidence_multiplier
        
        # Calculate quantity
        if price > 0:
            quantity = int(position_value / price)
            return max(1, quantity)  # At least 1 share
        
        return 0
    
    def _calculate_risk_score(self, market_data: Dict) -> float:
        """Calculate risk score (0.0-1.0, higher = riskier)."""
        indicators = market_data.get("indicators", {})
        volatility = indicators.get("volatility", 0.0)
        price = market_data.get("ohlcv", {}).get("close", 0.0)
        
        # Normalize volatility
        if price > 0:
            volatility_pct = (volatility / price) * 100
        else:
            volatility_pct = 0.0
        
        # Risk score based on volatility (0-1 scale)
        risk_score = min(1.0, volatility_pct / 5.0)  # 5% volatility = max risk
        
        return risk_score
    
    def _apply_risk_management(self, decision: Dict, market_data: Dict) -> Dict:
        """Apply risk management rules to decision."""
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0.0)
        risk_score = decision.get("risk_score", 0.5)
        
        # Check minimum confidence (более мягкая проверка для низких порогов)
        if confidence < self.min_confidence and action != "HOLD":
            # Если min_confidence очень низкий (< 0.1), разрешаем действия с еще меньшей уверенностью
            if self.min_confidence < 0.1 and confidence >= self.min_confidence * 0.5:
                logger.debug(f"Allowing low confidence action: {confidence:.2f} >= {self.min_confidence * 0.5:.2f}")
            else:
                logger.info(f"Decision rejected: confidence {confidence:.2f} < min {self.min_confidence}")
                return self._create_hold_decision(market_data, "Low confidence")
        
        # Check risk score (более мягкая проверка для симуляции)
        max_risk = self.risk_params.get("max_drawdown", 0.15)
        # Если min_confidence очень низкий, увеличиваем допустимый риск
        if self.min_confidence < 0.1:
            max_risk = max_risk * 1.5  # Увеличиваем допустимый риск на 50%
        if risk_score > max_risk and action != "HOLD":
            logger.info(f"Decision rejected: risk score {risk_score:.2f} > max {max_risk}")
            return self._create_hold_decision(market_data, "Risk too high")
        
        # Adjust quantity based on risk
        if action in ["BUY", "SELL"]:
            risk_multiplier = 1.0 - (risk_score * 0.5)  # Reduce position by up to 50% based on risk
            decision["quantity"] = int(decision["quantity"] * risk_multiplier)
            decision["quantity"] = max(1, decision["quantity"])
        
        # Add stop loss and take profit
        if action in ["BUY", "SELL"]:
            price = decision.get("price", 0.0)
            decision["stop_loss"] = price * (1 - self.risk_params["stop_loss"])
            decision["take_profit"] = price * (1 + self.risk_params["take_profit"])
        
        return decision
    
    def _create_hold_decision(self, market_data: Dict, reason: str = "") -> Dict:
        """Create a HOLD decision."""
        ticker = market_data.get("ticker", "UNKNOWN")
        price = market_data.get("ohlcv", {}).get("close", 0.0)
        
        return {
            "action": "HOLD",
            "ticker": ticker,
            "confidence": 0.5,
            "reasoning": f"HOLD: {reason}" if reason else "HOLD: No clear signal",
            "quantity": 0,
            "price": price,
            "timestamp": datetime.now().isoformat() + "Z",
            "risk_score": self._calculate_risk_score(market_data),
            "model_type": "rule_based" if not self.enable_ai else self.model_type
        }
    
    def _train_initial_model(self):
        """Train initial AI model using historical data only."""
        if not self.enable_ai:
            return
        
        logger.info("Training initial AI model on historical data...")
        
        # Use only historical data - no synthetic data
        if not self.use_historical_training:
            logger.warning("Historical training disabled. Model will use rule-based logic until historical data is available.")
            self.is_trained = False
            return
        
        try:
            logger.info("Attempting to train on real historical data...")
            X, y = self._prepare_historical_training_data()
            # Уменьшено минимальное требование для малого количества данных (например, Bybit 200 свечек)
            min_samples = 20  # Минимум 20 samples для обучения
            if X is not None and len(X) >= min_samples:
                logger.info(f"Using {len(X)} historical samples for training")
                # Если данных мало, используем меньше деревьев и меньшую глубину
                if len(X) < 100:
                    logger.info(f"Small dataset ({len(X)} samples), using reduced model complexity")
            else:
                logger.warning(f"Not enough historical data ({len(X) if X is not None else 0} samples, need {min_samples}). Model will use rule-based logic.")
                self.is_trained = False
                return
        except Exception as e:
            logger.error(f"Error preparing historical data: {e}. Model will use rule-based logic.")
            self.is_trained = False
            return
        
        # Проверяем что данные есть
        if X is None or len(X) == 0:
            logger.error("No training data available. Model will use rule-based logic.")
            self.is_trained = False
            return
        
        # Для малого количества данных используем меньший test_size
        test_size = 0.2 if len(X) > 50 else 0.1  # Меньший test_size для малых датасетов
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            # Если данных очень мало, используем все для обучения
            logger.warning(f"Very small dataset ({len(X)} samples), using all data for training")
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features
        if self.scaler is None:
            raise ValueError("Scaler not initialized - sklearn not available")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else X_train_scaled
        
        # Train model - адаптируем параметры под размер данных
        n_samples = len(X_train)
        if n_samples < 50:
            # Очень мало данных - простая модель
            n_estimators = 20
            max_depth = 3
        elif n_samples < 100:
            # Мало данных - средняя модель
            n_estimators = 50
            max_depth = 5
        else:
            # Достаточно данных - полная модель
            n_estimators = 100
            max_depth = 10
        
        if self.model_type == "random_forest":
            # Используем class_weight для балансировки классов (больше BUY/SELL, меньше HOLD)
            # 'balanced' автоматически взвешивает классы обратно пропорционально их частоте
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=42, 
                max_depth=max_depth,
                min_samples_split=2,  # Минимум для малых датасетов
                min_samples_leaf=1,
                class_weight='balanced'  # Балансировка классов для уменьшения HOLD
            )
        elif self.model_type == "gradient_boosting":
            # GradientBoosting не поддерживает class_weight напрямую, но можем использовать sample_weight
            # Для простоты используем те же параметры, но с более агрессивным learning_rate
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, 
                random_state=42, 
                max_depth=max_depth,
                learning_rate=0.15 if n_samples >= 100 else 0.25  # Больше learning rate для малых датасетов
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=42,
                max_depth=max_depth
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save model if path provided
        if self.model_path:
            self._save_model(self.model_path)
    
    def _prepare_historical_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from real historical market data.
        
        Returns:
            Tuple of (features, labels) arrays
        """
        try:
            from .market_monitor import MarketMonitoringAgent
            
            # Use provided ticker or default to SPY (broad market index)
            ticker = self.training_ticker or "SPY"
            
            logger.info(f"Fetching historical data for {ticker} (period: {self.training_period})")
            
            # Пробуем использовать Binance REST API для криптовалют (больше данных)
            use_binance = False
            try:
                # Проверяем, является ли символ криптовалютой
                from trading.agents.market_monitor import is_cryptocurrency
                if is_cryptocurrency(ticker):
                    from trading.services.binance_api import BinanceAPIService
                    binance_service = BinanceAPIService()
                    
                    # Маппинг period в days
                    period_days = {
                        "1mo": 30,
                        "3mo": 90,
                        "6mo": 180,
                        "1y": 365,
                    }
                    days = period_days.get(self.training_period, 30)
                    
                    logger.info(f"Using Binance REST API for {ticker} (up to 1000 candles)")
                    historical_data = binance_service.get_historical_data(
                        symbol=ticker,
                        interval="1d",  # Daily data for training
                        days=days
                    )
                    
                    if historical_data and len(historical_data) >= 20:
                        use_binance = True
                        logger.info(f"Retrieved {len(historical_data)} candles from Binance")
                        
                        # Конвертируем в DataFrame
                        import pandas as pd
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
                        
                        # Используем MarketMonitoringAgent для вычисления индикаторов
                        market_agent = MarketMonitoringAgent(
                            ticker=ticker,
                            interval="1d",
                            period=self.training_period,
                            enable_cache=False
                        )
                        market_agent.raw_data = df
                        data_with_indicators = market_agent.compute_indicators(df)
                        data = market_agent.preprocess(data_with_indicators)
                        
                        logger.info(f"Processed {len(data)} records with indicators from Binance")
                    else:
                        logger.warning(f"Not enough Binance data ({len(historical_data) if historical_data else 0}), falling back to MarketMonitoringAgent")
                        use_binance = False
            except Exception as binance_error:
                logger.debug(f"Binance API not available: {binance_error}, using MarketMonitoringAgent")
                use_binance = False
            
            if not use_binance:
                # Get historical data через MarketMonitoringAgent (Bybit/yfinance)
                market_agent = MarketMonitoringAgent(
                    ticker=ticker,
                    interval="1d",  # Daily data for training
                    period=self.training_period,
                    enable_cache=True,
                    request_delay=5.0,  # Задержка для обхода блокировок
                    max_retries=5,  # Больше попыток
                    backoff_factor=3.0  # Больше времени между попытками
                )
                
                # Get processed data with indicators
                data = market_agent.get_processed_data(analyze=False)
            
            # Уменьшено минимальное требование для малого количества данных
            if data.empty or len(data) < 20:
                logger.warning(f"Insufficient historical data: {len(data)} records (need at least 20)")
                return None, None
            
            # Prepare features and labels
            X = []
            y = []
            
            # For each day, extract features and determine label based on future price movement
            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                next_row = data.iloc[i + 1]
                
                # Extract features (same as _extract_features but from DataFrame row)
                features = []
                
                # Price features
                features.append(float(current_row.get('close', 0.0)))
                features.append(float(current_row.get('volume', 0.0)))
                features.append(float(current_row.get('price_change', 0.0)))
                
                # Technical indicators
                features.append(float(current_row.get('sma10', 0.0)))
                features.append(float(current_row.get('sma20', 0.0)))
                features.append(float(current_row.get('rsi14', 50.0)))
                features.append(float(current_row.get('macd', 0.0)))
                features.append(float(current_row.get('macd_hist', 0.0)))
                features.append(float(current_row.get('volatility', 0.0)))
                
                # Analysis features (need to compute)
                sma10 = current_row.get('sma10', 0.0)
                sma20 = current_row.get('sma20', 0.0)
                if sma10 > sma20:
                    trend_encoded = 1.0  # bull
                elif sma10 < sma20:
                    trend_encoded = -1.0  # bear
                else:
                    trend_encoded = 0.0  # sideways
                features.append(trend_encoded)
                
                # Strength (based on RSI distance from 50)
                rsi = current_row.get('rsi14', 50.0)
                strength = abs(rsi - 50) / 50
                features.append(float(strength))
                
                # RSI state
                if rsi > 70:
                    rsi_encoded = 1.0  # overbought
                elif rsi < 30:
                    rsi_encoded = -1.0  # oversold
                else:
                    rsi_encoded = 0.0  # neutral
                features.append(rsi_encoded)
                
                # SMA crossover (simplified)
                sma_cross = 0.0
                if i > 0:
                    prev_row = data.iloc[i - 1]
                    prev_sma10 = prev_row.get('sma10', 0.0)
                    prev_sma20 = prev_row.get('sma20', 0.0)
                    if (prev_sma10 <= prev_sma20 and sma10 > sma20) or \
                       (prev_sma10 >= prev_sma20 and sma10 < sma20):
                        sma_cross = 1.0
                features.append(sma_cross)
                
                # Determine label based on future price movement
                # Правильная логика: смотрим на цену через несколько периодов вперед
                # и учитываем транзакционные издержки
                lookahead_periods = 3  # Смотрим на 3 свечи вперед (для daily = 3 дня)
                transaction_cost_pct = 0.1  # Комиссия 0.1% (типично для криптобирж)
                min_profit_threshold = 0.5  # Минимальная прибыль 0.5% (с учетом комиссии)
                
                current_price = current_row.get('close', 0.0)
                
                if current_price > 0 and i + lookahead_periods < len(data):
                    # Берем цену через N периодов вперед
                    future_row = data.iloc[i + lookahead_periods]
                    future_price = future_row.get('close', 0.0)
                    
                    if future_price > 0:
                        # Вычисляем изменение цены в процентах
                        price_change_pct = ((future_price - current_price) / current_price) * 100
                        
                        # Учитываем транзакционные издержки
                        # Для BUY: нужно покрыть комиссию на вход и выход (0.1% + 0.1% = 0.2%)
                        # Для SELL: аналогично
                        net_profit_pct = abs(price_change_pct) - (transaction_cost_pct * 2)
                        
                        # Label: 0=SELL, 1=HOLD, 2=BUY
                        # Простая логика: если цена выросла достаточно (с учетом комиссии) -> BUY
                        # Если упала достаточно -> SELL, иначе HOLD
                        # НЕ используем индикаторы в логике меток - они только в фичах!
                        if price_change_pct > min_profit_threshold and net_profit_pct > 0:
                            label = 2  # BUY - цена выросла достаточно для прибыли
                        elif price_change_pct < -min_profit_threshold and net_profit_pct > 0:
                            label = 0  # SELL - цена упала достаточно (шорт или продажа)
                        else:
                            label = 1  # HOLD - недостаточное движение или не покрывает комиссию
                    else:
                        label = 1  # HOLD if no future price data
                else:
                    label = 1  # HOLD if not enough future data
                
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} training samples from historical data")
            logger.info(f"Label distribution: BUY={np.sum(y==2)}, HOLD={np.sum(y==1)}, SELL={np.sum(y==0)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing historical training data: {e}")
            return None, None
    
    # УДАЛЕНО: _prepare_synthetic_training_data() - синтетические данные больше не используются
    # Модель обучается только на реальных исторических данных
    
    def _retrain_with_real_data(self):
        """
        Переобучает модель на реальных данных из БД (решения + результаты сделок).
        
        ВАЖНО: Логика обучения на реальных данных:
        
        1. НАЧАЛЬНОЕ ОБУЧЕНИЕ:
           - Модель обучается на исторических данных (месяц до текущего момента)
           - Используются пары: (фичи на день X) → (метка на основе цены дня X+1)
           - Метка определяется по будущей цене: если цена выросла → BUY, упала → SELL
        
        2. ПРИНЯТИЕ РЕШЕНИЯ:
           - Модель принимает решение на основе текущих данных
           - Открывается позиция (BUY) или закрывается (SELL)
           - НО: результат еще неизвестен (нужно ждать закрытия позиции)
        
        3. ПЕРЕОБУЧЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ:
           - Используются только ЗАВЕРШЕННЫЕ сделки (BUY → SELL пары)
           - PnL известен только после закрытия позиции (SELL)
           - Только сделки старше 1 дня (чтобы избежать обучения на "свежих" данных)
           - Фичи берутся из решения на момент открытия позиции (BUY)
           - Метка определяется по результату: PnL > 0 → решение было правильным
        
        4. FALLBACK:
           - Если реальных данных недостаточно → переобучение на исторических данных
           - Исторические данные всегда доступны (месяц до текущего момента)
        """
        if not self.enable_ai:
            return
        
        try:
            # Пробуем получить данные из БД через Django ORM
            # Это работает только если вызывается из Django контекста
            try:
                from django.contrib.auth import get_user_model
                from trading.models import TradingDecision, Trade, Position
                from django.db.models import Q, F
                from datetime import timedelta
                from django.utils import timezone as tz
                
                # Получаем пользователя из контекста (если доступен)
                # Если нет - используем исторические данные
                user = getattr(self, '_django_user', None)
                
                if not user and self.user_id:
                    User = get_user_model()
                    try:
                        user = User.objects.get(id=self.user_id)
                    except User.DoesNotExist:
                        user = None
                
                if user:
                    logger.info("Collecting real trading data from database for retraining...")
                    
                    # ВАЖНО: Используем только ЗАВЕРШЕННЫЕ сделки (SELL после BUY) с известным PnL
                    # И только те, которые старше 1 дня (чтобы избежать обучения на "свежих" данных)
                    min_age = tz.now() - timedelta(days=1)
                    
                    # Получаем все SELL сделки (закрытие позиций) с известным PnL
                    # Это означает, что позиция была открыта (BUY) и закрыта (SELL)
                    completed_trades = Trade.objects.filter(
                        user=user,
                        action="SELL",  # Только закрывающие сделки
                        pnl__isnull=False,  # Только с рассчитанным PnL
                        executed_at__lt=min_age  # Только старые данные (старше 1 дня)
                    ).select_related('symbol', 'position').order_by('-executed_at')[:self.retrain_min_samples * 2]
                    
                    logger.info(f"Found {completed_trades.count()} completed trades (SELL with PnL) for retraining")
                    
                    training_samples = []
                    
                    for sell_trade in completed_trades:
                        # Находим соответствующую BUY сделку (открытие позиции)
                        # Ищем решение, которое привело к открытию этой позиции
                        position = sell_trade.position
                        
                        if not position:
                            # Если позиция не связана, ищем по символу и времени
                            buy_trade = Trade.objects.filter(
                                user=user,
                                symbol=sell_trade.symbol,
                                action="BUY",
                                executed_at__lt=sell_trade.executed_at,
                                executed_at__gte=sell_trade.executed_at - timedelta(days=7)  # BUY в течение недели до SELL
                            ).order_by('-executed_at').first()
                            
                            if not buy_trade:
                                continue
                            
                            # Ищем решение, которое привело к BUY
                            decision = TradingDecision.objects.filter(
                                user=user,
                                symbol=sell_trade.symbol,
                                decision="BUY",
                                created_at__lte=buy_trade.executed_at,
                                created_at__gte=buy_trade.executed_at - timedelta(hours=1)
                            ).order_by('-created_at').first()
                        else:
                            # Ищем решение, которое привело к открытию позиции
                            decision = TradingDecision.objects.filter(
                                user=user,
                                symbol=sell_trade.symbol,
                                decision="BUY",
                                created_at__lte=position.opened_at,
                                created_at__gte=position.opened_at - timedelta(hours=1)
                            ).order_by('-created_at').first()
                        
                        if not decision or not decision.market_data:
                            continue
                        
                        trade = sell_trade  # Используем SELL сделку (закрытие позиции)
                        
                        # Извлекаем фичи из метаданных решения (на момент открытия позиции)
                        market_data_obj = decision.market_data
                        metadata = decision.metadata or {}
                        
                        features = []
                        
                        # Price features (на момент принятия решения BUY)
                        features.append(float(market_data_obj.price))
                        features.append(float(market_data_obj.volume or 0))
                        change_pct = float(market_data_obj.change_percent or 0)
                        features.append(change_pct)
                        
                        # Technical indicators (из metadata решения)
                        indicators = metadata.get("indicators", {})
                        features.append(float(indicators.get("sma10", 0.0)))
                        features.append(float(indicators.get("sma20", 0.0)))
                        features.append(float(indicators.get("rsi14", 50.0)))
                        features.append(float(indicators.get("macd", 0.0)))
                        features.append(float(indicators.get("macd_hist", 0.0)))
                        features.append(float(indicators.get("volatility", 0.0)))
                        
                        # Analysis features
                        analysis = metadata.get("analysis", {})
                        trend = analysis.get("trend", "sideways")
                        trend_encoded = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}.get(trend, 0.0)
                        features.append(trend_encoded)
                        features.append(float(analysis.get("strength", 0.5)))
                        
                        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
                        rsi_encoded = {"overbought": 1.0, "oversold": -1.0, "neutral": 0.0}.get(rsi_state, 0.0)
                        features.append(rsi_encoded)
                        
                        sma_cross = analysis.get("signals", {}).get("sma_cross", 0)
                        features.append(float(sma_cross))
                        
                        # Определяем label на основе результата ЗАВЕРШЕННОЙ сделки
                        # PnL известен, т.к. это SELL сделка (закрытие позиции)
                        pnl = float(trade.pnl)
                        
                        # Логика: если BUY привел к прибыли (PnL > 0) - решение было правильным
                        # Если BUY привел к убытку (PnL < 0) - решение было неправильным
                        if decision.decision == "BUY":
                            if pnl > 0:
                                label = 2  # BUY было правильным решением
                            elif pnl < 0:
                                label = 0  # SELL было бы лучше (противоположное действие)
                            else:
                                label = 1  # HOLD если PnL = 0
                        else:
                            # Если решение было SELL (что маловероятно для открытия позиции)
                            label_map = {"BUY": 2, "SELL": 0, "HOLD": 1}
                            label = label_map.get(decision.decision, 1)
                        
                        training_samples.append((features, label))
                        
                        logger.debug(
                            f"Training sample: Decision={decision.decision}, "
                            f"PnL={pnl:.2f}, Label={label}, "
                            f"Age={(tz.now() - trade.executed_at).days} days"
                        )
                    
                    if len(training_samples) >= self.retrain_min_samples:
                        logger.info(f"Collected {len(training_samples)} real trading samples for retraining")
                        
                        # Объединяем с историческими данными
                        X_new = np.array([s[0] for s in training_samples])
                        y_new = np.array([s[1] for s in training_samples])
                        
                        # Получаем старые данные (исторические)
                        X_hist, y_hist = self._prepare_historical_training_data()
                        
                        if X_hist is not None and len(X_hist) > 0:
                            # Объединяем старые и новые данные
                            X_combined = np.vstack([X_hist, X_new])
                            y_combined = np.hstack([y_hist, y_new])
                            logger.info(f"Combined training data: {len(X_hist)} historical + {len(X_new)} real = {len(X_combined)} total")
                        else:
                            # Используем только новые данные
                            X_combined = X_new
                            y_combined = y_new
                            logger.info(f"Using only real trading data: {len(X_combined)} samples")
                        
                        # Переобучаем модель
                        self._retrain_model(X_combined, y_combined)
                        from datetime import datetime
                        self.last_retrain_time = datetime.now()
                        logger.info("Model retrained successfully with real trading data")
                    else:
                        logger.debug(f"Not enough real trading samples ({len(training_samples)} < {self.retrain_min_samples})")
                        # Если реальных данных недостаточно, переобучаем на исторических данных
                        logger.info("Retraining on historical data instead (not enough real trading samples)")
                        X_hist, y_hist = self._prepare_historical_training_data()
                        if X_hist is not None and len(X_hist) >= 50:
                            self._retrain_model(X_hist, y_hist)
                            from datetime import datetime
                            self.last_retrain_time = datetime.now()
                            logger.info("Model retrained on historical data")
                        else:
                            logger.debug("Not enough historical data for retrain either, skipping")
                        return
                        
                else:
                    # Нет доступа к Django контексту - используем только исторические данные
                    logger.info("No Django user context, retraining on historical data only")
                    X, y = self._prepare_historical_training_data()
                    if X is not None and len(X) >= self.retrain_min_samples:
                        self._retrain_model(X, y)
                        from datetime import datetime
                        self.last_retrain_time = datetime.now()
                    else:
                        logger.warning("Not enough historical data for retraining")
                        return
                        
            except ImportError:
                # Django не доступен - используем только исторические данные
                logger.info("Django not available, retraining on historical data only")
                X, y = self._prepare_historical_training_data()
                if X is not None and len(X) >= self.retrain_min_samples:
                    self._retrain_model(X, y)
                    self.last_retrain_time = datetime.now()
                else:
                    logger.warning("Not enough historical data for retraining")
                    return
                    
        except Exception as e:
            logger.error(f"Error in continuous learning retrain: {e}", exc_info=True)
            # Не прерываем работу, просто логируем ошибку
    
    def _retrain_model(self, X: np.ndarray, y: np.ndarray):
        """
        Переобучает модель на новых данных.
        
        Args:
            X: Массив фичей
            y: Массив меток
        """
        if not self.enable_ai or self.scaler is None:
            return
        
        logger.info(f"Retraining model on {len(X)} samples...")
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Переобучаем scaler на всех данных
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Переобучаем модель
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                class_weight='balanced'  # Балансировка классов
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=5,
                learning_rate=0.15
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Оцениваем
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model retrained. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        # Сохраняем модель если путь указан
        if self.model_path:
            self._save_model(self.model_path)
    
    def _save_model(self, path: str):
        """Save trained model to disk."""
        try:
            if self.scaler is None:
                logger.warning("Cannot save model: scaler not initialized")
                return
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type
                }, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self, path: str):
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.model_type = data.get("model_type", self.model_type)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def update_portfolio(self, ticker: str, action: str, quantity: int, price: float):
        """Update portfolio state after trade execution."""
        if action == "BUY":
            cost = quantity * price
            if cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= cost
                if ticker in self.portfolio["positions"]:
                    # Average price calculation
                    old_qty = self.portfolio["positions"][ticker]["quantity"]
                    old_price = self.portfolio["positions"][ticker]["avg_price"]
                    total_cost = (old_qty * old_price) + cost
                    total_qty = old_qty + quantity
                    self.portfolio["positions"][ticker] = {
                        "quantity": total_qty,
                        "avg_price": total_cost / total_qty
                    }
                else:
                    self.portfolio["positions"][ticker] = {
                        "quantity": quantity,
                        "avg_price": price
                    }
        elif action == "SELL":
            if ticker in self.portfolio["positions"]:
                qty = self.portfolio["positions"][ticker]["quantity"]
                if quantity <= qty:
                    revenue = quantity * price
                    self.portfolio["cash"] += revenue
                    if quantity == qty:
                        del self.portfolio["positions"][ticker]
                    else:
                        self.portfolio["positions"][ticker]["quantity"] -= quantity
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        return self.portfolio.copy()
    
    def get_decision_history(self, n: Optional[int] = None) -> List[Dict]:
        """Get decision history."""
        history = list(self.decision_history)
        if n:
            return history[-n:]
        return history

