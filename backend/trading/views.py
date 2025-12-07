import logging
from datetime import datetime, timedelta

from django.db.models import Q
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from trading.models import Symbol, MarketData, TradingDecision, AgentStatus, Account, Position, Trade
from trading.serializers import (
    SymbolSerializer,
    MarketDataSerializer,
    TradingDecisionSerializer,
    AgentStatusSerializer,
    AccountSerializer,
    PositionSerializer,
    TradeSerializer,
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
        try:
            symbols = Symbol.objects.filter(user=request.user, is_active=True)
            result = []
            errors = []

            for symbol in symbols:
                try:
                    latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()
                    if latest_data:
                        serializer = MarketDataSerializer(latest_data)
                        result.append(serializer.data)
                    else:
                        # Если нет данных в БД, получаем напрямую из API
                        market_service = get_market_data_service()
                        data = market_service.get_latest_data(symbol.symbol)
                        if data:
                            try:
                                market_data = MarketData.objects.create(symbol=symbol, **data)
                                serializer = MarketDataSerializer(market_data)
                                result.append(serializer.data)
                            except Exception as create_error:
                                logger.error(f"Error creating MarketData for {symbol.symbol}: {str(create_error)}", exc_info=True)
                                errors.append(f"Ошибка создания данных для {symbol.symbol}: {str(create_error)}")
                        else:
                            errors.append(f"Не удалось получить данные для {symbol.symbol}")
                except Exception as e:
                    logger.error(f"Error getting latest data for {symbol.symbol}: {str(e)}", exc_info=True)
                    errors.append(f"Ошибка для {symbol.symbol}: {str(e)}")

            response_data = {"data": result}
            if errors:
                response_data["errors"] = errors
            return Response(response_data)
        except Exception as e:
            logger.error(f"Error in latest endpoint: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Внутренняя ошибка сервера: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

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

        try:
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
                try:
                    latest_data = MarketData.objects.create(symbol=symbol, **data)
                except Exception as create_error:
                    logger.error(f"Error creating MarketData: {str(create_error)}", exc_info=True)
                    return Response(
                        {"detail": f"Ошибка создания данных рынка: {str(create_error)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

            # TODO: Здесь будет вызов AI модели для принятия решения
            # Пока возвращаем заглушку
            try:
                decision = TradingDecision.objects.create(
                    user=request.user,
                    symbol=symbol,
                    decision="HOLD",  # Заглушка
                    confidence=Decimal("50.0"),
                    market_data=latest_data,
                    reasoning="AI model not implemented yet. This is a placeholder decision.",
                    metadata={},
                )
                serializer = TradingDecisionSerializer(decision)
                return Response(serializer.data)
            except Exception as decision_error:
                logger.error(f"Error creating TradingDecision: {str(decision_error)}", exc_info=True)
                return Response(
                    {"detail": f"Ошибка создания решения: {str(decision_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error in decision-maker endpoint: {str(e)}", exc_info=True)
            return Response(
                {"detail": f"Внутренняя ошибка сервера: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


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


class PortfolioView(APIView):
    """Эндпойнт для получения данных портфеля"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить сводку портфеля"""
        from decimal import Decimal
        from django.db.models import Sum, Count, Q
        from django.utils import timezone as tz

        # Получаем или создаем счет пользователя
        account, _ = Account.objects.get_or_create(
            user=request.user,
            defaults={"balance": Decimal("10000.00"), "free_cash": Decimal("10000.00")}
        )

        # Обновляем текущие цены для открытых позиций
        open_positions = Position.objects.filter(user=request.user, is_open=True)
        for position in open_positions:
            # Получаем последнюю цену из MarketData
            latest_data = MarketData.objects.filter(symbol=position.symbol).order_by("-timestamp").first()
            if latest_data:
                position.current_price = latest_data.price
                position.save(update_fields=["current_price"])

        # Рассчитываем использованную маржу (сумма всех открытых позиций)
        used_margin = Decimal("0.00")
        for position in open_positions:
            if position.current_price:
                used_margin += position.current_price * position.quantity

        # Обновляем счет
        account.used_margin = used_margin
        account.free_cash = account.balance - used_margin
        account.save(update_fields=["used_margin", "free_cash"])

        # Статистика по сделкам
        total_trades = Trade.objects.filter(user=request.user).count()
        
        # P&L за сегодня
        today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = Trade.objects.filter(
            user=request.user,
            executed_at__gte=today_start
        )
        today_pnl = today_trades.aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")

        # Общий P&L (из всех закрытых позиций и сделок)
        total_pnl = Trade.objects.filter(user=request.user).aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")
        # Также добавляем P&L от открытых позиций
        for position in open_positions:
            if position.current_price:
                position_pnl = (position.current_price - position.entry_price) * position.quantity
                total_pnl += position_pnl

        return Response({
            "balance": float(account.balance),
            "freeCash": float(account.free_cash),
            "usedMargin": float(account.used_margin),
            "totalTrades": total_trades,
            "todayPnL": float(today_pnl),
            "totalPnL": float(total_pnl),
        })


class PositionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для открытых позиций"""
    permission_classes = [IsAuthenticated]
    serializer_class = PositionSerializer

    def get_queryset(self):
        """Получить только открытые позиции пользователя"""
        queryset = Position.objects.filter(user=self.request.user, is_open=True)
        
        # Обновляем текущие цены
        for position in queryset:
            latest_data = MarketData.objects.filter(symbol=position.symbol).order_by("-timestamp").first()
            if latest_data:
                position.current_price = latest_data.price
                position.save(update_fields=["current_price"])
        
        return queryset.select_related("symbol").order_by("-opened_at")


class TradeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для истории сделок"""
    permission_classes = [IsAuthenticated]
    serializer_class = TradeSerializer

    def get_queryset(self):
        """Получить последние 20 сделок пользователя"""
        return Trade.objects.filter(user=self.request.user).select_related("symbol").order_by("-executed_at")[:20]


class EquityCurveView(APIView):
    """Эндпойнт для данных equity curve"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить данные для графика equity curve"""
        from decimal import Decimal
        from django.db.models import Sum, Min, Max
        from django.utils import timezone as tz
        from datetime import timedelta

        # Получаем счет
        account, _ = Account.objects.get_or_create(
            user=request.user,
            defaults={"balance": Decimal("10000.00"), "initial_balance": Decimal("10000.00")}
        )

        initial_balance = float(account.initial_balance)
        current_balance = float(account.balance)

        # Рассчитываем max drawdown
        # Получаем все сделки с P&L
        trades = Trade.objects.filter(user=request.user, pnl__isnull=False).order_by("executed_at")
        
        max_drawdown = Decimal("0.00")
        peak_balance = initial_balance
        running_balance = initial_balance

        for trade in trades:
            running_balance += float(trade.pnl)
            if running_balance > peak_balance:
                peak_balance = running_balance
            drawdown = running_balance - peak_balance
            if drawdown < max_drawdown:
                max_drawdown = Decimal(str(drawdown))

        # Рассчитываем Sharpe Ratio (упрощенная версия)
        # Для реального расчета нужны более сложные вычисления
        sharpe_ratio = Decimal("1.24")  # Заглушка, можно улучшить позже

        # Генерируем данные для графика (последние 30 дней)
        equity_data = []
        days = 30
        today = tz.now().date()
        
        for i in range(days + 1):
            date = today - timedelta(days=days - i)
            # Рассчитываем баланс на эту дату
            trades_until_date = Trade.objects.filter(
                user=request.user,
                executed_at__date__lte=date
            ).aggregate(total_pnl=Sum("pnl"))["total_pnl"] or Decimal("0.00")
            
            balance_on_date = initial_balance + float(trades_until_date)
            equity_data.append({
                "day": i,
                "balance": balance_on_date,
                "date": date.strftime("%b %d"),
            })

        return Response({
            "initialBalance": initial_balance,
            "currentBalance": current_balance,
            "maxDrawdown": float(max_drawdown),
            "sharpeRatio": float(sharpe_ratio),
            "equityData": equity_data,
        })
