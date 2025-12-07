import logging
from datetime import datetime, timedelta

from django.db.models import Q
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from trading.models import Symbol, MarketData, TradingDecision, AgentStatus, Account, Position, Trade, AgentLog, Message, UserSettings
from trading.serializers import (
    SymbolSerializer,
    MarketDataSerializer,
    TradingDecisionSerializer,
    AgentStatusSerializer,
    AccountSerializer,
    PositionSerializer,
    TradeSerializer,
    AgentDetailSerializer,
    MessageSerializer,
    UserSettingsSerializer,
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
        from trading.agents import MarketMonitoringAgent, DecisionMakingAgent
        from trading.agents.integration import MarketAgentIntegration, DecisionAgentIntegration

        try:
            symbol_id = request.data.get("symbol_id")
            if not symbol_id:
                return Response({"detail": "symbol_id required"}, status=status.HTTP_400_BAD_REQUEST)

            try:
                symbol = Symbol.objects.get(id=symbol_id, user=request.user, is_active=True)
            except Symbol.DoesNotExist:
                return Response({"detail": "Symbol not found"}, status=status.HTTP_404_NOT_FOUND)

            # Получаем настройки пользователя для конфигурации агентов
            user_settings, _ = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={"timeframe": "1h", "risk_level": "medium"}
            )
            
            # Определяем параметры из настроек
            timeframe = user_settings.timeframe or "1h"
            # Маппинг timeframe в period для агента
            period_map = {
                "5m": "1d",
                "15m": "3d",
                "1h": "1mo",
                "4h": "3mo",
                "1d": "1y",
            }
            period = period_map.get(timeframe, "1mo")
            
            # Шаг 1: Используем MarketMonitoringAgent для получения данных с индикаторами
            try:
                market_integration = MarketAgentIntegration(request.user)
                market_agent = MarketMonitoringAgent(
                    ticker=symbol.symbol,
                    interval=timeframe,
                    period=period,
                    enable_cache=True,
                    request_delay=5.0,  # Увеличенная задержка для обхода блокировок Yahoo Finance
                    max_retries=5,  # Больше попыток
                    backoff_factor=3.0  # Больше времени между попытками
                )
                
                # Получаем обработанные данные с анализом
                market_message = market_integration.process_and_save(
                    symbol=symbol,
                    market_agent=market_agent,
                    save_to_db=True
                )
                
                # Получаем последние данные для связи с решением
                latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp").first()
                
            except Exception as market_error:
                logger.error(f"Error getting market data with agent: {str(market_error)}", exc_info=True)
                return Response(
                    {"detail": f"Ошибка получения данных рынка через MarketMonitoringAgent: {str(market_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Шаг 2: Используем DecisionMakingAgent для принятия решения
            try:
                decision_integration = DecisionAgentIntegration(request.user)
                
                # Получаем настройки для агента
                risk_tolerance = user_settings.risk_level or "medium"
                # Снижаем порог уверенности по умолчанию для получения больше решений (для обучения)
                confidence_threshold = float(user_settings.confidence_threshold or 0.50)
                model_type = user_settings.model_type or "Random Forest"
                
                # Маппинг названия модели
                model_type_map = {
                    "Random Forest": "random_forest",
                    "Gradient Boosting": "gradient_boosting",
                }
                agent_model_type = model_type_map.get(model_type, "random_forest")
                
                # Создаем DecisionMakingAgent с настройками пользователя
                decision_agent = DecisionMakingAgent(
                    model_type=agent_model_type,
                    risk_tolerance=risk_tolerance,
                    min_confidence=confidence_threshold,
                    enable_ai=True
                )
                
                # Принимаем решение
                decision = decision_integration.make_decision(
                    symbol=symbol,
                    market_data_obj=latest_data,
                    market_message=market_message,
                    decision_agent=decision_agent
                )
                
                serializer = TradingDecisionSerializer(decision)
                return Response(serializer.data)
                
            except Exception as decision_error:
                logger.error(f"Error in decision-making agent: {str(decision_error)}", exc_info=True)
                return Response(
                    {"detail": f"Ошибка принятия решения: {str(decision_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error in decision-maker endpoint: {str(e)}", exc_info=True)
            return Response(
                {"detail": f"Внутренняя ошибка сервера: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExecutionAgentView(APIView):
    """Управление Execution Agent"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить статус агента"""
        status_obj, _ = AgentStatus.objects.get_or_create(
            user=request.user,
            agent_type="EXECUTION",
            defaults={"status": "IDLE"},
        )
        return Response(AgentStatusSerializer(status_obj).data)
    
    def post(self, request):
        """Выполнить решение (сделку)"""
        from trading.agents import ExecutionAgent
        from trading.agents.integration import ExecutionAgentIntegration
        
        try:
            decision_id = request.data.get("decision_id")
            if not decision_id:
                return Response({"detail": "decision_id required"}, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                decision = TradingDecision.objects.get(id=decision_id, user=request.user)
            except TradingDecision.DoesNotExist:
                return Response({"detail": "Decision not found"}, status=status.HTTP_404_NOT_FOUND)
            
            # Если решение HOLD, не выполняем
            if decision.decision == "HOLD":
                return Response({
                    "detail": "Decision is HOLD, no execution needed",
                    "decision_id": decision.id,
                    "status": "skipped"
                })
            
            # Получаем настройки пользователя
            user_settings, _ = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={"risk_level": "medium"}
            )
            
            # Создаем ExecutionAgent
            execution_agent = ExecutionAgent(
                execution_mode="simulated",  # Всегда симулируем для безопасности
                enable_slippage=True,
                slippage_factor=0.001,  # 0.1% slippage
                commission_rate=0.001,  # 0.1% commission
            )
            
            # Формируем решение для агента из TradingDecision
            decision_dict = {
                "action": decision.decision,
                "ticker": decision.symbol.symbol,
                "quantity": decision.metadata.get("quantity", 1),
                "price": decision.metadata.get("price", float(decision.market_data.price) if decision.market_data else 0.0),
                "confidence": float(decision.confidence / 100) if decision.confidence else 0.5,  # Конвертируем обратно в 0-1
                "timestamp": decision.created_at.isoformat(),
                "reasoning": decision.reasoning,
            }
            
            # Выполняем сделку через агента
            execution_result = execution_agent.receive_decision(decision_dict)
            
            # Сохраняем в БД через интеграцию
            execution_integration = ExecutionAgentIntegration(request.user)
            trade = execution_integration.execute_trade(
                symbol=decision.symbol,
                decision_obj=decision,
                execution_agent=execution_agent,
                execution_result=execution_result
            )
            
            if trade:
                from trading.serializers import TradeSerializer
                serializer = TradeSerializer(trade)
                return Response({
                    "status": "executed",
                    "trade": serializer.data,
                    "execution_result": execution_result
                })
            else:
                return Response({
                    "status": execution_result.get("status", "rejected"),
                    "message": execution_result.get("message", "Trade not executed"),
                    "execution_result": execution_result
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            logger.error(f"Error in execution agent endpoint: {str(e)}", exc_info=True)
            return Response(
                {"detail": f"Ошибка выполнения сделки: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


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


class AgentsDetailView(APIView):
    """Эндпойнт для получения детальной информации об агентах"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить список всех агентов с детальной информацией"""
        # Получаем или создаем статусы для всех типов агентов
        agent_types = ["MARKET_MONITOR", "DECISION_MAKER", "EXECUTION"]
        agents = []
        
        for agent_type in agent_types:
            status_obj, _ = AgentStatus.objects.get_or_create(
                user=request.user,
                agent_type=agent_type,
                defaults={"status": "IDLE"},
            )
            serializer = AgentDetailSerializer(status_obj)
            agent_data = serializer.data
            # Преобразуем id в строку для соответствия фронтенду
            agent_data["id"] = str(status_obj.id)
            agents.append(agent_data)
        
        return Response(agents)


class MessagesViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для сообщений между агентами"""
    permission_classes = [IsAuthenticated]
    serializer_class = MessageSerializer

    def get_queryset(self):
        """Получить все сообщения пользователя, отсортированные по времени"""
        return Message.objects.filter(user=self.request.user).order_by("-timestamp")


class PerformanceMetricsView(APIView):
    """Эндпойнт для метрик производительности"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Рассчитывает все метрики производительности на основе сделок и позиций"""
        from decimal import Decimal
        from django.db.models import Sum, Count, Avg, Q
        from django.utils import timezone as tz

        # Получаем счет
        account, _ = Account.objects.get_or_create(
            user=request.user,
            defaults={"balance": Decimal("10000.00"), "initial_balance": Decimal("10000.00")}
        )

        initial_balance = float(account.initial_balance)
        current_balance = float(account.balance)
        total_return = current_balance - initial_balance
        return_percent = (total_return / initial_balance * 100) if initial_balance > 0 else 0

        # Статистика по сделкам
        all_trades = Trade.objects.filter(user=request.user, pnl__isnull=False)
        total_trades = all_trades.count()

        # Выигрышные и проигрышные сделки
        winning_trades = all_trades.filter(pnl__gt=0)
        losing_trades = all_trades.filter(pnl__lt=0)
        winning_count = winning_trades.count()
        losing_count = losing_trades.count()

        # Средние значения
        avg_win = float(winning_trades.aggregate(avg=Avg("pnl"))["avg"] or Decimal("0.00"))
        avg_loss = float(losing_trades.aggregate(avg=Avg("pnl"))["avg"] or Decimal("0.00"))
        avg_loss = abs(avg_loss) if avg_loss < 0 else avg_loss  # Делаем положительным для расчета

        # Win Rate
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0

        # Win/Loss Ratio
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

        # Profit Factor
        total_wins = float(winning_trades.aggregate(total=Sum("pnl"))["total"] or Decimal("0.00"))
        total_losses = abs(float(losing_trades.aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Max Drawdown
        trades_ordered = all_trades.order_by("executed_at")
        max_drawdown = Decimal("0.00")
        peak_balance = initial_balance
        running_balance = initial_balance

        for trade in trades_ordered:
            running_balance += float(trade.pnl)
            if running_balance > peak_balance:
                peak_balance = running_balance
            drawdown = running_balance - peak_balance
            if drawdown < max_drawdown:
                max_drawdown = Decimal(str(drawdown))

        # Sharpe Ratio (упрощенная версия)
        # Для реального расчета нужны более сложные вычисления с волатильностью
        sharpe_ratio = Decimal("1.24")  # Заглушка, можно улучшить позже

        return Response({
            "totalReturn": float(total_return),
            "sharpeRatio": float(sharpe_ratio),
            "winRate": round(win_rate, 1),
            "profitFactor": round(profit_factor, 2),
            "maxDrawdown": float(max_drawdown),
            "totalTrades": total_trades,
            "winningTrades": winning_count,
            "losingTrades": losing_count,
            "avgWin": round(avg_win, 2),
            "avgLoss": round(-avg_loss, 2) if avg_loss > 0 else 0,  # Возвращаем отрицательным
            "winLossRatio": round(win_loss_ratio, 2),
            "returnPercent": round(return_percent, 1),
        })


class PnLCurveView(APIView):
    """Эндпойнт для данных P&L Curve"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить данные для графика P&L Curve (последние 30 дней)"""
        from decimal import Decimal
        from django.db.models import Sum
        from django.utils import timezone as tz
        from datetime import timedelta

        # Получаем счет
        account, _ = Account.objects.get_or_create(
            user=request.user,
            defaults={"balance": Decimal("10000.00"), "initial_balance": Decimal("10000.00")}
        )

        initial_balance = float(account.initial_balance)
        current_balance = float(account.balance)

        # Генерируем данные для графика (последние 30 дней)
        pnl_data = []
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
            pnl_data.append({
                "day": i,
                "balance": balance_on_date,
                "date": date.strftime("%b %d"),
            })

        return Response({
            "initialBalance": initial_balance,
            "currentBalance": current_balance,
            "pnlData": pnl_data,
        })


class MonthlyBreakdownView(APIView):
    """Эндпойнт для разбивки P&L по периодам"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить P&L разбивку по периодам (Today, Yesterday, This Week, по месяцам)"""
        from decimal import Decimal
        from django.db.models import Sum
        from django.utils import timezone as tz
        from datetime import timedelta, datetime

        breakdown = []

        # Сегодня
        today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_pnl = Trade.objects.filter(
            user=request.user,
            executed_at__gte=today_start,
            pnl__isnull=False
        ).aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")
        breakdown.append({
            "period": "Today",
            "pnl": float(today_pnl),
        })

        # Вчера
        yesterday_start = today_start - timedelta(days=1)
        yesterday_pnl = Trade.objects.filter(
            user=request.user,
            executed_at__gte=yesterday_start,
            executed_at__lt=today_start,
            pnl__isnull=False
        ).aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")
        breakdown.append({
            "period": "Yesterday",
            "pnl": float(yesterday_pnl),
        })

        # Эта неделя
        week_start = today_start - timedelta(days=today_start.weekday())
        week_pnl = Trade.objects.filter(
            user=request.user,
            executed_at__gte=week_start,
            pnl__isnull=False
        ).aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")
        breakdown.append({
            "period": "This Week",
            "pnl": float(week_pnl),
        })

        # Последние 3 месяца
        now = tz.now()
        for i in range(3):
            month_date = now - timedelta(days=30 * (i + 1))
            month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if i == 0:
                month_end = now
            else:
                next_month = month_start + timedelta(days=32)
                month_end = next_month.replace(day=1) - timedelta(days=1)

            month_pnl = Trade.objects.filter(
                user=request.user,
                executed_at__gte=month_start,
                executed_at__lte=month_end,
                pnl__isnull=False
            ).aggregate(total=Sum("pnl"))["total"] or Decimal("0.00")

            month_name = month_start.strftime("%b %Y")
            breakdown.append({
            "period": month_name,
            "pnl": float(month_pnl),
            })

        return Response(breakdown)


class SettingsView(APIView):
    """Эндпойнт для получения и обновления настроек пользователя"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить настройки пользователя"""
        settings, created = UserSettings.objects.get_or_create(
            user=request.user,
            defaults={
                "status": "stopped",
                "speed": 1.0,
                "symbol": "AAPL",
                "timeframe": "1h",
                "data_provider": "Yahoo Finance",
                "history_length": "Last 1 year",
                "model_type": "Random Forest",
                "prediction_horizon": "1 hour",
                "confidence_threshold": 0.55,
                "initial_balance": 10000.00,
                "max_position_size": 50,
                "risk_level": "medium",
                "stop_loss": -2.0,
                "take_profit": 5.0,
                "max_leverage": 1.0,
            }
        )
        serializer = UserSettingsSerializer(settings)
        return Response(serializer.data)

    def put(self, request):
        """Обновить настройки пользователя"""
        from decimal import Decimal

        settings, created = UserSettings.objects.get_or_create(
            user=request.user,
            defaults={
                "status": "stopped",
                "speed": 1.0,
                "symbol": "AAPL",
                "timeframe": "1h",
                "data_provider": "Yahoo Finance",
                "history_length": "Last 1 year",
                "model_type": "Random Forest",
                "prediction_horizon": "1 hour",
                "confidence_threshold": 0.55,
                "initial_balance": 10000.00,
                "max_position_size": 50,
                "risk_level": "medium",
                "stop_loss": -2.0,
                "take_profit": 5.0,
                "max_leverage": 1.0,
            }
        )

        # Обновляем поля из запроса
        data = request.data

        # Преобразуем названия полей из фронтенда в названия модели
        if "status" in data:
            settings.status = data["status"]
        if "speed" in data:
            settings.speed = float(data["speed"])
        if "symbol" in data:
            settings.symbol = data["symbol"]
        if "timeframe" in data:
            settings.timeframe = data["timeframe"]
        if "dataProvider" in data:
            settings.data_provider = data["dataProvider"]
        if "historyLength" in data:
            settings.history_length = data["historyLength"]
        if "modelType" in data:
            settings.model_type = data["modelType"]
        if "predictionHorizon" in data:
            settings.prediction_horizon = data["predictionHorizon"]
        if "confidenceThreshold" in data:
            settings.confidence_threshold = Decimal(str(data["confidenceThreshold"]))
        if "initialBalance" in data:
            settings.initial_balance = Decimal(str(data["initialBalance"]))
        if "maxPositionSize" in data:
            settings.max_position_size = int(data["maxPositionSize"])
        if "riskLevel" in data:
            settings.risk_level = data["riskLevel"]
        if "stopLoss" in data:
            settings.stop_loss = Decimal(str(data["stopLoss"]))
        if "takeProfit" in data:
            settings.take_profit = Decimal(str(data["takeProfit"]))
        if "maxLeverage" in data:
            settings.max_leverage = Decimal(str(data["maxLeverage"]))

        settings.save()

        serializer = UserSettingsSerializer(settings)
        return Response(serializer.data)

    def patch(self, request):
        """Частичное обновление настроек (аналогично PUT)"""
        return self.put(request)


class DashboardOverviewView(APIView):
    """Эндпойнт для получения данных Dashboard Overview"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить данные для Dashboard Overview (KPICards)"""
        from decimal import Decimal
        from django.db.models import Sum, Count, Q
        from django.utils import timezone as tz

        # Получаем счет
        account, _ = Account.objects.get_or_create(
            user=request.user,
            defaults={"balance": Decimal("10000.00"), "initial_balance": Decimal("10000.00")}
        )

        balance = float(account.balance)

        # P&L за сегодня
        today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = Trade.objects.filter(
            user=request.user,
            executed_at__gte=today_start,
            pnl__isnull=False
        )
        today_pnl = float(today_trades.aggregate(total=Sum("pnl"))["total"] or Decimal("0.00"))
        today_trades_count = today_trades.count()

        # Win Rate
        all_trades = Trade.objects.filter(user=request.user, pnl__isnull=False)
        total_trades = all_trades.count()
        winning_trades = all_trades.filter(pnl__gt=0).count()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Статус агентов
        agent_statuses = AgentStatus.objects.filter(user=request.user)
        active_count = agent_statuses.filter(status="RUNNING").count()
        total_agents = agent_statuses.count()
        agents_status = "All Active" if active_count == total_agents and total_agents > 0 else f"{active_count}/{total_agents} Active"

        return Response({
            "balance": balance,
            "todayPnL": today_pnl,
            "todayTradesCount": today_trades_count,
            "winRate": round(win_rate, 1),
            "agentsStatus": agents_status,
            "activeAgentsCount": active_count,
        })


class MarketChartView(APIView):
    """Эндпойнт для получения данных графика рынка"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить данные для графика по символу"""
        symbol_param = request.query_params.get("symbol", "AAPL")
        timeframe = request.query_params.get("timeframe", "1h")  # 15m, 1h, 4h, 1d

        # Получаем символ пользователя
        try:
            symbol = Symbol.objects.get(user=request.user, symbol=symbol_param, is_active=True)
        except Symbol.DoesNotExist:
            return Response(
                {"detail": f"Symbol {symbol_param} not found or not active"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Определяем период для данных в зависимости от timeframe
        from datetime import timedelta
        from django.utils import timezone as tz

        timeframe_map = {
            "15m": timedelta(hours=6),  # 6 часов данных для 15-минутного графика
            "1h": timedelta(days=7),    # 7 дней для часового графика
            "4h": timedelta(days=30),   # 30 дней для 4-часового графика
            "1d": timedelta(days=90),    # 90 дней для дневного графика
        }

        period = timeframe_map.get(timeframe, timedelta(days=7))
        start_time = tz.now() - period

        # Получаем данные рынка
        market_data = MarketData.objects.filter(
            symbol=symbol,
            timestamp__gte=start_time
        ).order_by("timestamp")

        # Формируем данные для графика
        chart_data = []
        for data in market_data:
            chart_data.append({
                "timestamp": data.timestamp,
                "price": float(data.price),
                "volume": int(data.volume) if data.volume else None,
            })

        # Получаем текущую цену (последняя запись)
        current_price = float(market_data.last().price) if market_data.exists() else 0.0

        return Response({
            "symbol": symbol_param,
            "currentPrice": current_price,
            "data": chart_data,
        })


class MarketHeatmapView(APIView):
    """Эндпойнт для получения данных Market Heatmap"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Получить данные для heatmap (все активные символы с изменениями)"""
        # Получаем все активные символы пользователя
        symbols = Symbol.objects.filter(user=request.user, is_active=True)

        market_data_list = []
        for symbol in symbols:
            # Получаем последние 2 записи для расчета изменения
            latest_data = MarketData.objects.filter(symbol=symbol).order_by("-timestamp")[:2]
            
            if latest_data.count() >= 2:
                current = latest_data[0]
                previous = latest_data[1]
                
                # Рассчитываем процент изменения
                change_percent = 0.0
                if previous.price > 0:
                    change_percent = ((current.price - previous.price) / previous.price) * 100
                
                market_data_list.append({
                    "symbol": symbol.symbol,
                    "price": float(current.price),
                    "previousPrice": float(previous.price),
                    "volume": int(current.volume) if current.volume else 0,
                    "timestamp": current.timestamp,
                    "changePercent": round(change_percent, 2),
                })
            elif latest_data.count() == 1:
                # Если только одна запись, используем её как текущую и предыдущую
                current = latest_data[0]
                market_data_list.append({
                    "symbol": symbol.symbol,
                    "price": float(current.price),
                    "previousPrice": float(current.price),  # Нет изменения
                    "volume": int(current.volume) if current.volume else 0,
                    "timestamp": current.timestamp,
                    "changePercent": 0.0,
                })
            else:
                # Если нет данных, все равно добавляем символ с нулевыми значениями
                market_data_list.append({
                    "symbol": symbol.symbol,
                    "price": 0.0,
                    "previousPrice": 0.0,
                    "volume": 0,
                    "timestamp": None,
                    "changePercent": 0.0,
                })

        # Сортируем по изменению (от большего к меньшему)
        market_data_list.sort(
            key=lambda x: x.get("changePercent", 0.0),
            reverse=True
        )

        return Response({
            "symbols": market_data_list,
            "total": len(market_data_list),
            "gainers": [s for s in market_data_list if s.get("changePercent", 0) > 0],
            "losers": [s for s in market_data_list if s.get("changePercent", 0) < 0],
            "unchanged": [s for s in market_data_list if s.get("changePercent", 0) == 0],
        })
