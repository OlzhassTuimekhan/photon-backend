from django.db.models import Q
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
        ]
        read_only_fields = ["id", "created_at"]


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
        """Получает последнее действие из metadata или логов"""
        if obj.metadata and "last_action" in obj.metadata:
            return obj.metadata["last_action"]
        # Пробуем получить из последнего лога
        last_log = obj.logs.order_by("-timestamp").first()
        if last_log:
            return last_log.message
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

