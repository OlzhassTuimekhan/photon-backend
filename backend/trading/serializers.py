from rest_framework import serializers
from trading.models import (
    Symbol,
    MarketData,
    TradingDecision,
    AgentStatus,
    Account,
    Position,
    Trade,
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
    symbol_name = serializers.CharField(source="symbol.symbol", read_only=True)
    pnl = serializers.SerializerMethodField()
    pnl_percent = serializers.SerializerMethodField()

    class Meta:
        model = Position
        fields = [
            "id",
            "symbol",
            "symbol_name",
            "quantity",
            "entry_price",
            "current_price",
            "pnl",
            "pnl_percent",
            "opened_at",
            "closed_at",
            "is_open",
        ]
        read_only_fields = ["id", "opened_at", "closed_at"]

    def get_pnl(self, obj):
        """Рассчитывает P&L"""
        if obj.is_open and obj.current_price:
            return float((obj.current_price - obj.entry_price) * obj.quantity)
        return None

    def get_pnl_percent(self, obj):
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

