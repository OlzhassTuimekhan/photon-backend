import logging
from datetime import datetime, timedelta

from django.db.models import Q
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from trading.models import Symbol, MarketData, TradingDecision, AgentStatus
from trading.serializers import (
    SymbolSerializer,
    MarketDataSerializer,
    TradingDecisionSerializer,
    AgentStatusSerializer,
)
from trading.services import MarketDataService, get_market_data_service
from trading.tasks import start_market_monitoring, stop_market_monitoring

logger = logging.getLogger(__name__)


class SymbolViewSet(viewsets.ModelViewSet):
    """ViewSet для управления символами"""
    permission_classes = [IsAuthenticated]
    serializer_class = SymbolSerializer

    def get_queryset(self):
        return Symbol.objects.filter(user=self.request.user, is_active=True)

    def create(self, request, *args, **kwargs):
        """Создание символа с валидацией через yfinance/Bybit"""
        symbol_code = request.data.get("symbol", "").upper().strip()
        if not symbol_code:
            return Response({"detail": "Символ не указан"}, status=status.HTTP_400_BAD_REQUEST)

        # Создаем сервис с настройками из settings
        market_service = get_market_data_service()

        # Проверяем, существует ли символ
        if not market_service.validate_symbol(symbol_code):
            return Response(
                {"detail": f"Символ {symbol_code} не найден или недоступен"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Получаем название символа
        data = market_service.get_latest_data(symbol_code)
        symbol_name = data.get("name", symbol_code) if data else symbol_code

        # Создаем или обновляем символ
        symbol, created = Symbol.objects.get_or_create(
            user=request.user,
            symbol=symbol_code,
            defaults={"name": symbol_name, "is_active": True},
        )

        if not created:
            symbol.is_active = True
            symbol.name = symbol_name
            symbol.save()

        serializer = self.get_serializer(symbol)
        return Response(serializer.data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class MarketDataViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для получения данных рынка"""
    permission_classes = [IsAuthenticated]
    serializer_class = MarketDataSerializer

    def get_queryset(self):
        user = self.request.user
        queryset = MarketData.objects.filter(symbol__user=user)

        # Фильтр по символу
        symbol_id = self.request.query_params.get("symbol_id")
        if symbol_id:
            queryset = queryset.filter(symbol_id=symbol_id)

        # Фильтр по символу (код)
        symbol_code = self.request.query_params.get("symbol")
        if symbol_code:
            queryset = queryset.filter(symbol__symbol=symbol_code.upper())

        # Фильтр по времени (последние N часов)
        hours = self.request.query_params.get("hours")
        if hours:
            try:
                hours = int(hours)
                since = timezone.now() - timedelta(hours=hours)
                queryset = queryset.filter(timestamp__gte=since)
            except ValueError:
                pass

        return queryset.order_by("-timestamp")

    @action(detail=False, methods=["get"])
    def latest(self, request):
        """Получить последние данные для всех символов пользователя"""
        symbols = Symbol.objects.filter(user=request.user, is_active=True)
        result = []

        for symbol in symbols:
            latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()
            if latest_data:
                result.append(MarketDataSerializer(latest_data).data)
            else:
                # Если нет данных в БД, получаем напрямую из API
                market_service = get_market_data_service()
                data = market_service.get_latest_data(symbol.symbol)
                if data:
                    market_data = MarketData.objects.create(symbol=symbol, **data)
                    result.append(MarketDataSerializer(market_data).data)

        return Response(result)

    @action(detail=False, methods=["post"])
    def refresh(self, request):
        """Обновить данные для указанных символов"""
        symbol_ids = request.data.get("symbol_ids", [])
        if not symbol_ids:
            # Обновляем все активные символы пользователя
            symbols = Symbol.objects.filter(user=request.user, is_active=True)
        else:
            symbols = Symbol.objects.filter(user=request.user, id__in=symbol_ids, is_active=True)

        updated = []
        errors = []

        market_service = get_market_data_service()
        for symbol in symbols:
            data = market_service.get_latest_data(symbol.symbol)
            if data:
                market_data = MarketData.objects.create(symbol=symbol, **data)
                updated.append(MarketDataSerializer(market_data).data)
            else:
                errors.append(f"Не удалось получить данные для {symbol.symbol}")

        return Response({"updated": updated, "errors": errors})


class TradingDecisionViewSet(viewsets.ModelViewSet):
    """ViewSet для управления решениями"""
    permission_classes = [IsAuthenticated]
    serializer_class = TradingDecisionSerializer

    def get_queryset(self):
        queryset = TradingDecision.objects.filter(user=self.request.user)

        # Фильтр по символу
        symbol_id = self.request.query_params.get("symbol_id")
        if symbol_id:
            queryset = queryset.filter(symbol_id=symbol_id)

        # Фильтр по решению
        decision = self.request.query_params.get("decision")
        if decision:
            queryset = queryset.filter(decision=decision.upper())

        return queryset.order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=["get"])
    def statistics(self, request):
        """Статистика по решениям"""
        decisions = TradingDecision.objects.filter(user=request.user)

        total = decisions.count()
        by_decision = {}
        for choice, label in TradingDecision.DECISION_CHOICES:
            count = decisions.filter(decision=choice).count()
            by_decision[choice] = {"count": count, "percentage": (count / total * 100) if total > 0 else 0}

        # Последние решения
        recent = decisions[:10]

        return Response({
            "total": total,
            "by_decision": by_decision,
            "recent": TradingDecisionSerializer(recent, many=True).data,
        })


class AgentStatusViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для статусов агентов"""
    permission_classes = [IsAuthenticated]
    serializer_class = AgentStatusSerializer

    def get_queryset(self):
        return AgentStatus.objects.filter(user=self.request.user)


class MarketMonitorAgentView(APIView):
    """Управление Market Monitoring Agent"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить статус агента"""
        status_obj, _ = AgentStatus.objects.get_or_create(
            user=request.user,
            agent_type="MARKET_MONITOR",
            defaults={"status": "IDLE"},
        )
        return Response(AgentStatusSerializer(status_obj).data)

    def post(self, request):
        """Запустить/остановить агента"""
        action_type = request.data.get("action", "start")  # start или stop

        status_obj, _ = AgentStatus.objects.get_or_create(
            user=request.user,
            agent_type="MARKET_MONITOR",
            defaults={"status": "IDLE"},
        )

        if action_type == "start":
            # Запускаем Celery задачу
            task = start_market_monitoring.delay(request.user.id)
            status_obj.status = "RUNNING"
            status_obj.metadata = {"task_id": task.id}
            status_obj.last_activity = timezone.now()
            status_obj.save()
            return Response({
                "status": "started",
                "message": "Market monitoring agent started",
                "task_id": task.id,
            })
        elif action_type == "stop":
            # Останавливаем задачу
            if status_obj.metadata.get("task_id"):
                stop_market_monitoring(request.user.id)
            status_obj.status = "STOPPED"
            status_obj.metadata = {}
            status_obj.save()
            return Response({
                "status": "stopped",
                "message": "Market monitoring agent stopped",
            })
        else:
            return Response({"detail": "Invalid action"}, status=status.HTTP_400_BAD_REQUEST)


class DecisionMakerAgentView(APIView):
    """Управление Decision-Making Agent"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить статус агента"""
        status_obj, _ = AgentStatus.objects.get_or_create(
            user=request.user,
            agent_type="DECISION_MAKER",
            defaults={"status": "IDLE"},
        )
        return Response(AgentStatusSerializer(status_obj).data)

    def post(self, request):
        """Запросить анализ и решение для символа"""
        from decimal import Decimal

        symbol_id = request.data.get("symbol_id")
        if not symbol_id:
            return Response({"detail": "symbol_id required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            symbol = Symbol.objects.get(id=symbol_id, user=request.user, is_active=True)
        except Symbol.DoesNotExist:
            return Response({"detail": "Symbol not found"}, status=status.HTTP_404_NOT_FOUND)

        # Получаем последние данные рынка
        latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()
        if not latest_data:
            # Получаем данные напрямую
            market_service = get_market_data_service()
            data = market_service.get_latest_data(symbol.symbol)
            if not data:
                return Response({"detail": "No market data available"}, status=status.HTTP_400_BAD_REQUEST)
            latest_data = MarketData.objects.create(symbol=symbol, **data)

        # TODO: Здесь будет вызов AI модели для принятия решения
        # Пока возвращаем заглушку
        decision = TradingDecision.objects.create(
            user=request.user,
            symbol=symbol,
            decision="HOLD",  # Заглушка
            confidence=Decimal("50.0"),
            market_data=latest_data,
            reasoning="AI model not implemented yet. This is a placeholder decision.",
            metadata={},
        )

        return Response(TradingDecisionSerializer(decision).data)


class ExecutionAgentView(APIView):
    """Управление Execution Agent (пока только статус, т.к. не выполняем реальные сделки)"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить статус агента"""
        status_obj, _ = AgentStatus.objects.get_or_create(
            user=request.user,
            agent_type="EXECUTION",
            defaults={"status": "IDLE"},
        )
        return Response(AgentStatusSerializer(status_obj).data)

