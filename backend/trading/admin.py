from django.contrib import admin
from trading.models import Symbol, MarketData, TradingDecision, AgentStatus


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ["symbol", "name", "user", "is_active", "created_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["symbol", "name", "user__email"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ["symbol", "price", "change_percent", "timestamp", "created_at"]
    list_filter = ["timestamp", "created_at"]
    search_fields = ["symbol__symbol"]
    readonly_fields = ["created_at"]
    date_hierarchy = "timestamp"


@admin.register(TradingDecision)
class TradingDecisionAdmin(admin.ModelAdmin):
    list_display = ["symbol", "decision", "confidence", "user", "created_at"]
    list_filter = ["decision", "created_at"]
    search_fields = ["symbol__symbol", "user__email"]
    readonly_fields = ["created_at"]


@admin.register(AgentStatus)
class AgentStatusAdmin(admin.ModelAdmin):
    list_display = ["agent_type", "status", "user", "last_activity", "updated_at"]
    list_filter = ["agent_type", "status", "updated_at"]
    search_fields = ["user__email"]
    readonly_fields = ["updated_at"]

