from rest_framework import serializers
from trading.models import Symbol, MarketData, TradingDecision, AgentStatus


class SymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symbol
        fields = ["id", "symbol", "name", "is_active", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class MarketDataSerializer(serializers.ModelSerializer):
    symbol_name = serializers.CharField(source="symbol.symbol", read_only=True)

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

