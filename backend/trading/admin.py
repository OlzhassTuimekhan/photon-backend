from django.contrib import admin
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


@admin.register(Account)
class AccountAdmin(admin.ModelAdmin):
    list_display = ["user", "balance", "free_cash", "used_margin", "updated_at"]
    list_filter = ["updated_at"]
    search_fields = ["user__email"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    list_display = ["symbol", "user", "quantity", "entry_price", "current_price", "is_open", "opened_at"]
    list_filter = ["is_open", "opened_at"]
    search_fields = ["symbol__symbol", "user__email"]
    readonly_fields = ["opened_at", "closed_at"]


@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ["symbol", "user", "action", "price", "quantity", "pnl", "executed_at"]
    list_filter = ["action", "agent_type", "executed_at"]
    search_fields = ["symbol__symbol", "user__email"]
    readonly_fields = ["executed_at"]


@admin.register(AgentLog)
class AgentLogAdmin(admin.ModelAdmin):
    list_display = ["agent_status", "level", "message", "timestamp"]
    list_filter = ["level", "timestamp"]
    search_fields = ["message", "agent_status__user__email"]
    readonly_fields = ["timestamp"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["from_agent", "to_agent", "message_type", "user", "timestamp"]
    list_filter = ["message_type", "from_agent", "to_agent", "timestamp"]
    search_fields = ["user__email"]
    readonly_fields = ["timestamp"]


@admin.register(UserSettings)
class UserSettingsAdmin(admin.ModelAdmin):
    list_display = ["user", "status", "symbol", "model_type", "risk_level", "updated_at"]
    list_filter = ["status", "risk_level", "model_type", "updated_at"]
    search_fields = ["user__email", "symbol"]
    readonly_fields = ["created_at", "updated_at"]

