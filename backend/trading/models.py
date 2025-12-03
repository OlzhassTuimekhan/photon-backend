from django.conf import settings
from django.db import models


class Symbol(models.Model):
    """Модель для отслеживаемых торговых символов (акции, криптовалюты и т.д.)"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="symbols",
    )
    symbol = models.CharField(max_length=20, db_index=True)  # AAPL, BTC-USD и т.д.
    name = models.CharField(max_length=255, blank=True)  # Название компании/актива
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["user", "symbol"]]
        ordering = ["symbol"]

    def __str__(self):
        return f"{self.symbol} ({self.user.email})"


class MarketData(models.Model):
    """Модель для хранения данных рынка в реальном времени"""
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name="market_data",
    )
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.BigIntegerField(null=True, blank=True)
    high = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    low = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    open_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    change = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)  # Изменение цены
    change_percent = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)  # % изменения
    timestamp = models.DateTimeField(db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["symbol", "-timestamp"]),
        ]

    def __str__(self):
        return f"{self.symbol.symbol} @ {self.price} ({self.timestamp})"


class TradingDecision(models.Model):
    """Модель для хранения решений агента Decision-Making Agent"""
    DECISION_CHOICES = [
        ("BUY", "Купить"),
        ("SELL", "Продать"),
        ("HOLD", "Держать"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="trading_decisions",
    )
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name="decisions",
    )
    decision = models.CharField(max_length=10, choices=DECISION_CHOICES)
    confidence = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)  # 0-100%
    market_data = models.ForeignKey(
        MarketData,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="decisions",
    )
    reasoning = models.TextField(blank=True)  # Объяснение решения (от AI модели)
    metadata = models.JSONField(default=dict, blank=True)  # Дополнительные данные от модели
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["symbol", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.symbol.symbol} - {self.decision} ({self.created_at})"


class AgentStatus(models.Model):
    """Модель для отслеживания статуса агентов"""
    AGENT_TYPES = [
        ("MARKET_MONITOR", "Market Monitoring Agent"),
        ("DECISION_MAKER", "Decision-Making Agent"),
        ("EXECUTION", "Execution Agent"),
    ]

    STATUS_CHOICES = [
        ("RUNNING", "Работает"),
        ("STOPPED", "Остановлен"),
        ("ERROR", "Ошибка"),
        ("IDLE", "Ожидает"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_statuses",
    )
    agent_type = models.CharField(max_length=20, choices=AGENT_TYPES, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="IDLE")
    last_activity = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)  # Дополнительная информация о работе агента
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["user", "agent_type"]]
        ordering = ["agent_type"]

    def __str__(self):
        return f"{self.get_agent_type_display()} - {self.get_status_display()} ({self.user.email})"

