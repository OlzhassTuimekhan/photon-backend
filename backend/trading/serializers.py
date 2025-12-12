from datetime import timedelta

from django.db.models import Q
from django.utils import timezone
from rest_framework import serializers
from trading.models import (
    Symbol,
    MarketData,
    TradingDecision,
    AgentStatus,
    Account,
    Position,
    Trade,
    AgentLog,
    Message,
    UserSettings,
)


class SymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symbol
        fields = ["id", "symbol", "name", "is_active", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class MarketDataSerializer(serializers.ModelSerializer):
    symbol_name = serializers.CharField(source="symbol.symbol", read_only=True)
    # Явно указываем DecimalField для правильной сериализации
    price = serializers.DecimalField(max_digits=20, decimal_places=8, coerce_to_string=False)
    high = serializers.DecimalField(max_digits=20, decimal_places=8, coerce_to_string=False, allow_null=True, required=False)
    low = serializers.DecimalField(max_digits=20, decimal_places=8, coerce_to_string=False, allow_null=True, required=False)
    open_price = serializers.DecimalField(max_digits=20, decimal_places=8, coerce_to_string=False, allow_null=True, required=False)
    change = serializers.DecimalField(max_digits=20, decimal_places=8, coerce_to_string=False, allow_null=True, required=False)
    change_percent = serializers.DecimalField(max_digits=10, decimal_places=4, coerce_to_string=False, allow_null=True, required=False)

    class Meta:
        model = MarketData
        fields = [
            "id",
            "symbol",
            "symbol_name",
            "price",
            "volume",
            "high",
            "low",
            "open_price",
            "change",
            "change_percent",
            "timestamp",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class TradingDecisionSerializer(serializers.ModelSerializer):
    symbol_name = serializers.CharField(source="symbol.symbol", read_only=True)
    symbol_id = serializers.IntegerField(source="symbol.id", read_only=True)
    # Явно указываем DecimalField для правильной сериализации
    confidence = serializers.DecimalField(max_digits=5, decimal_places=2, coerce_to_string=False, allow_null=True, required=False)
    status = serializers.SerializerMethodField()
    executed_at = serializers.SerializerMethodField()

    class Meta:
        model = TradingDecision
        fields = [
            "id",
            "symbol",
            "symbol_id",
            "symbol_name",
            "decision",
            "confidence",
            "market_data",
            "reasoning",
            "metadata",
            "created_at",
            "status",
            "executed_at",
        ]
        read_only_fields = ["id", "created_at", "status", "executed_at"]
    
    def get_status(self, obj):
        """
        Определяет статус решения:
        - executed: если есть соответствующая сделка (Trade)
        - pending: если решение HOLD
        - pending: если решение BUY/SELL но нет сделки
        """
        if obj.decision == "HOLD":
            return "completed"  # HOLD решения всегда завершены
        
        # Ищем соответствующую сделку
        from trading.models import Trade
        trade = Trade.objects.filter(
            user=obj.user,
            symbol=obj.symbol,
            action=obj.decision,
            executed_at__gte=obj.created_at,
            executed_at__lte=obj.created_at + timedelta(minutes=5)  # В пределах 5 минут
        ).first()
        
        return "executed" if trade else "pending"
    
    def get_executed_at(self, obj):
        """Возвращает время исполнения если есть соответствующая сделка"""
        if obj.decision == "HOLD":
            return obj.created_at  # HOLD "исполняется" сразу
        
        # Ищем соответствующую сделку
        from trading.models import Trade
        trade = Trade.objects.filter(
            user=obj.user,
            symbol=obj.symbol,
            action=obj.decision,
            executed_at__gte=obj.created_at,
            executed_at__lte=obj.created_at + timezone.timedelta(minutes=5)
        ).first()
        
        return trade.executed_at if trade else None


class AgentStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentStatus
        fields = [
            "id",
            "agent_type",
            "status",
            "last_activity",
            "error_message",
            "metadata",
            "updated_at",
        ]
        read_only_fields = ["id", "updated_at"]


class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = [
            "id",
            "balance",
            "free_cash",
            "used_margin",
            "initial_balance",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class PositionSerializer(serializers.ModelSerializer):
    symbol = serializers.CharField(source="symbol.symbol", read_only=True)
    entryPrice = serializers.DecimalField(source="entry_price", max_digits=20, decimal_places=8, coerce_to_string=False)
    currentPrice = serializers.DecimalField(
        source="current_price",
        max_digits=20,
        decimal_places=8,
        coerce_to_string=False,
        allow_null=True,
        required=False,
    )
    pnl = serializers.SerializerMethodField()
    pnlPercent = serializers.SerializerMethodField()
    openedAt = serializers.DateTimeField(source="opened_at", read_only=True)
    closedAt = serializers.DateTimeField(source="closed_at", read_only=True)

    class Meta:
        model = Position
        fields = [
            "id",
            "symbol",
            "quantity",
            "entryPrice",
            "currentPrice",
            "pnl",
            "pnlPercent",
            "openedAt",
            "closedAt",
            "is_open",
        ]
        read_only_fields = ["id", "openedAt", "closedAt"]

    def get_pnl(self, obj):
        """Рассчитывает P&L"""
        if obj.is_open and obj.current_price:
            return float((obj.current_price - obj.entry_price) * obj.quantity)
        return None

    def get_pnlPercent(self, obj):
        """Рассчитывает процент P&L"""
        if obj.is_open and obj.current_price and obj.entry_price:
            return float(((obj.current_price - obj.entry_price) / obj.entry_price) * 100)
        return None


class TradeSerializer(serializers.ModelSerializer):
    symbol_name = serializers.CharField(source="symbol.symbol", read_only=True)
    agent = serializers.CharField(source="agent_type", read_only=True)
    timestamp = serializers.DateTimeField(source="executed_at", read_only=True)

    class Meta:
        model = Trade
        fields = [
            "id",
            "symbol",
            "symbol_name",
            "action",
            "price",
            "quantity",
            "agent",
            "pnl",
            "timestamp",
        ]
        read_only_fields = ["id", "timestamp"]


class AgentLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentLog
        fields = [
            "id",
            "timestamp",
            "level",
            "message",
        ]
        read_only_fields = ["id", "timestamp"]


class AgentDetailSerializer(serializers.ModelSerializer):
    """Расширенный сериализатор для детальной информации об агенте"""
    type = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    lastAction = serializers.SerializerMethodField()
    lastUpdated = serializers.DateTimeField(source="last_activity", read_only=True)
    messagesProcessed = serializers.SerializerMethodField()
    logs = serializers.SerializerMethodField()
    recentDecisions = serializers.SerializerMethodField()
    currentState = serializers.SerializerMethodField()
    explorationMode = serializers.SerializerMethodField()

    class Meta:
        model = AgentStatus
        fields = [
            "id",
            "type",
            "name",
            "status",
            "lastAction",
            "lastUpdated",
            "messagesProcessed",
            "logs",
            "recentDecisions",
            "currentState",
            "explorationMode",
        ]

    def get_type(self, obj):
        """Преобразует AGENT_TYPE в формат фронтенда"""
        mapping = {
            "MARKET_MONITOR": "market",
            "DECISION_MAKER": "decision",
            "EXECUTION": "execution",
        }
        return mapping.get(obj.agent_type, "market")

    def get_name(self, obj):
        """Возвращает название агента"""
        return obj.get_agent_type_display()

    def get_status(self, obj):
        """Преобразует STATUS в формат фронтенда"""
        mapping = {
            "RUNNING": "active",
            "IDLE": "idle",
            "ERROR": "error",
            "STOPPED": "idle",
        }
        return mapping.get(obj.status, "idle")

    def get_lastAction(self, obj):
        """Получает последнее действие в зависимости от типа агента"""
        # Для DECISION_MAKER - показываем последнее решение
        if obj.agent_type == "DECISION_MAKER":
            last_decision = TradingDecision.objects.filter(
                user=obj.user
            ).order_by("-created_at").first()
            if last_decision:
                action = last_decision.decision
                confidence = float(last_decision.confidence) if last_decision.confidence else 0.0
                return f"{action} (confidence: {confidence:.1f}%)"
        
        # Для EXECUTION - показываем последнюю сделку
        elif obj.agent_type == "EXECUTION":
            last_trade = Trade.objects.filter(
                user=obj.user
            ).order_by("-executed_at").first()
            if last_trade:
                action = last_trade.action
                symbol = last_trade.symbol.symbol
                qty = float(last_trade.quantity)
                return f"{action} {qty} {symbol}"
        
        # Для MARKET_MONITOR - показываем последнее обновление
        elif obj.agent_type == "MARKET_MONITOR":
            from trading.models import UserSettings
            settings = UserSettings.objects.filter(user=obj.user).first()
            if settings:
                return f"Monitoring {settings.symbol}"
        
        # Fallback: ищем в metadata или логах
        if obj.metadata and "last_action" in obj.metadata:
            return obj.metadata["last_action"]
        last_log = obj.logs.order_by("-timestamp").first()
        if last_log:
            # Обрезаем длинные сообщения
            msg = last_log.message
            return msg[:80] + "..." if len(msg) > 80 else msg
        return "No actions yet"

    def get_messagesProcessed(self, obj):
        """Считает количество сообщений, обработанных агентом"""
        from trading.models import Message
        return Message.objects.filter(
            user=obj.user,
        ).filter(
            Q(from_agent=obj.agent_type) | Q(to_agent=obj.agent_type)
        ).count()

    def get_logs(self, obj):
        """Получает последние логи агента"""
        logs = obj.logs.order_by("-timestamp")[:10]  # Последние 10 логов
        return AgentLogSerializer(logs, many=True).data
    
    def get_recentDecisions(self, obj):
        """Получает последние решения для Decision Maker"""
        if obj.agent_type != "DECISION_MAKER":
            return []
        
        decisions = TradingDecision.objects.filter(
            user=obj.user
        ).order_by("-created_at")[:5]
        
        result = []
        for decision in decisions:
            result.append({
                "id": decision.id,
                "action": decision.decision,
                "confidence": float(decision.confidence) if decision.confidence else 0.0,
                "reasoning": decision.reasoning,
                "symbol": decision.symbol.symbol,
                "timestamp": decision.created_at,
                "metadata": decision.metadata or {},
            })
        return result
    
    def get_currentState(self, obj):
        """Получает текущее состояние агента"""
        from trading.models import UserSettings, Trade
        
        state = {
            "isActive": obj.status == "RUNNING",
            "error": obj.error_message if obj.error_message else None,
        }
        
        # Для DECISION_MAKER добавляем информацию о модели
        if obj.agent_type == "DECISION_MAKER":
            settings = UserSettings.objects.filter(user=obj.user).first()
            if settings:
                state["modelType"] = settings.model_type
                state["confidenceThreshold"] = float(settings.confidence_threshold)
                state["riskLevel"] = settings.risk_level
            
            # Проверяем exploration mode
            completed_trades = Trade.objects.filter(
                user=obj.user,
                action="SELL",
                pnl__isnull=False
            ).count()
            state["completedTrades"] = completed_trades
            state["needsMoreData"] = completed_trades < 10
        
        # Для EXECUTION добавляем статистику
        elif obj.agent_type == "EXECUTION":
            trades_count = Trade.objects.filter(user=obj.user).count()
            state["totalTrades"] = trades_count
        
        # Для MARKET_MONITOR добавляем информацию о символе
        elif obj.agent_type == "MARKET_MONITOR":
            settings = UserSettings.objects.filter(user=obj.user).first()
            if settings:
                state["symbol"] = settings.symbol
                state["timeframe"] = settings.timeframe
        
        return state
    
    def get_explorationMode(self, obj):
        """Проверяет включен ли exploration mode"""
        if obj.agent_type != "DECISION_MAKER":
            return None
        
        # Exploration mode активен если < 10 завершенных сделок
        completed_trades = Trade.objects.filter(
            user=obj.user,
            action="SELL",
            pnl__isnull=False
        ).count()
        
        is_exploration = completed_trades < 10
        
        return {
            "enabled": is_exploration,
            "reason": f"Collecting data for training ({completed_trades}/10 completed trades)" if is_exploration else None,
            "confidenceThreshold": 0.35 if is_exploration else None,
        }


class MessageSerializer(serializers.ModelSerializer):
    from_agent_type = serializers.SerializerMethodField()
    to_agent_type = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = [
            "id",
            "timestamp",
            "from_agent",
            "to_agent",
            "from_agent_type",
            "to_agent_type",
            "message_type",
            "payload",
        ]
        read_only_fields = ["id", "timestamp", "from_agent_type", "to_agent_type"]

    def get_from_agent_type(self, obj):
        """Преобразует from_agent в формат фронтенда"""
        mapping = {
            "MARKET_MONITOR": "market",
            "DECISION_MAKER": "decision",
            "EXECUTION": "execution",
        }
        return mapping.get(obj.from_agent, "market")

    def get_to_agent_type(self, obj):
        """Преобразует to_agent в формат фронтенда"""
        mapping = {
            "MARKET_MONITOR": "market",
            "DECISION_MAKER": "decision",
            "EXECUTION": "execution",
        }
        return mapping.get(obj.to_agent, "market")

    def to_representation(self, instance):
        """Кастомное представление для соответствия фронтенду"""
        return {
            "id": str(instance.id),
            "timestamp": instance.timestamp,
            "from": self.get_from_agent_type(instance),
            "to": self.get_to_agent_type(instance),
            "type": instance.message_type,
            "payload": instance.payload,
        }


class UserSettingsSerializer(serializers.ModelSerializer):
    """Сериализатор для настроек пользователя"""
    # Преобразуем названия полей для соответствия фронтенду
    dataProvider = serializers.CharField(source="data_provider", required=False)
    historyLength = serializers.CharField(source="history_length", required=False)
    modelType = serializers.CharField(source="model_type", required=False)
    predictionHorizon = serializers.CharField(source="prediction_horizon", required=False)
    confidenceThreshold = serializers.DecimalField(
        source="confidence_threshold",
        max_digits=5,
        decimal_places=2,
        coerce_to_string=False,
        required=False
    )
    initialBalance = serializers.DecimalField(
        source="initial_balance",
        max_digits=20,
        decimal_places=2,
        coerce_to_string=False,
        required=False
    )
    maxPositionSize = serializers.IntegerField(source="max_position_size", required=False)
    riskLevel = serializers.CharField(source="risk_level", required=False)
    stopLoss = serializers.DecimalField(
        source="stop_loss",
        max_digits=5,
        decimal_places=2,
        coerce_to_string=False,
        required=False
    )
    takeProfit = serializers.DecimalField(
        source="take_profit",
        max_digits=5,
        decimal_places=2,
        coerce_to_string=False,
        required=False
    )
    maxLeverage = serializers.DecimalField(
        source="max_leverage",
        max_digits=5,
        decimal_places=2,
        coerce_to_string=False,
        required=False
    )

    class Meta:
        model = UserSettings
        fields = [
            "id",
            "status",
            "speed",
            "symbol",
            "timeframe",
            "dataProvider",
            "historyLength",
            "modelType",
            "predictionHorizon",
            "confidenceThreshold",
            "initialBalance",
            "maxPositionSize",
            "riskLevel",
            "stopLoss",
            "takeProfit",
            "maxLeverage",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def to_representation(self, instance):
        """Преобразует данные для соответствия фронтенду"""
        data = super().to_representation(instance)
        return {
            "status": data["status"],
            "speed": float(data["speed"]),
            "symbol": data["symbol"],
            "timeframe": data["timeframe"],
            "dataProvider": data["dataProvider"],
            "historyLength": data["historyLength"],
            "modelType": data["modelType"],
            "predictionHorizon": data["predictionHorizon"],
            "confidenceThreshold": float(data["confidenceThreshold"]),
            "initialBalance": float(data["initialBalance"]),
            "maxPositionSize": data["maxPositionSize"],
            "riskLevel": data["riskLevel"],
            "stopLoss": float(data["stopLoss"]),
            "takeProfit": float(data["takeProfit"]),
            "maxLeverage": float(data["maxLeverage"]),
        }

