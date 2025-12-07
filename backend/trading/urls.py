from django.urls import path, include
from rest_framework.routers import DefaultRouter

from trading.views import (
    SymbolViewSet,
    MarketDataViewSet,
    TradingDecisionViewSet,
    AgentStatusViewSet,
    MarketMonitorAgentView,
    DecisionMakerAgentView,
    ExecutionAgentView,
    PortfolioView,
    PositionViewSet,
    TradeViewSet,
    EquityCurveView,
)

router = DefaultRouter()
router.register(r"symbols", SymbolViewSet, basename="symbol")
router.register(r"market-data", MarketDataViewSet, basename="market-data")
router.register(r"decisions", TradingDecisionViewSet, basename="decision")
router.register(r"agents/status", AgentStatusViewSet, basename="agent-status")
router.register(r"positions", PositionViewSet, basename="position")
router.register(r"trades", TradeViewSet, basename="trade")

urlpatterns = [
    path("", include(router.urls)),
    path("agents/market-monitor/", MarketMonitorAgentView.as_view(), name="market-monitor"),
    path("agents/decision-maker/", DecisionMakerAgentView.as_view(), name="decision-maker"),
    path("agents/execution/", ExecutionAgentView.as_view(), name="execution"),
    path("portfolio/", PortfolioView.as_view(), name="portfolio"),
    path("portfolio/equity-curve/", EquityCurveView.as_view(), name="equity-curve"),
]
