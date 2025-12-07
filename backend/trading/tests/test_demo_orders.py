from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from trading.models import Account, Position, Trade


class DemoOrderTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            email="demo@test.com",
            password="pass1234",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.tick = {
            "price": Decimal("100.00"),
            "volume": 1000,
            "high": Decimal("101.00"),
            "low": Decimal("99.00"),
            "open_price": Decimal("99.50"),
            "change": Decimal("0.50"),
            "change_percent": Decimal("0.50"),
            "timestamp": timezone.now(),
        }

    @mock.patch("trading.views.get_market_data_service")
    def test_buy_creates_position_and_trade(self, mock_market_service):
        mock_market_service.return_value.get_latest_data.return_value = self.tick

        response = self.client.post(
            "/api/trading/demo/orders/",
            {"action": "BUY", "symbol": "BTCUSDT", "quantity": "0.01"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        position = Position.objects.get(user=self.user, symbol__symbol="BTCUSDT")
        self.assertEqual(position.quantity, Decimal("0.01"))
        self.assertEqual(Trade.objects.filter(user=self.user).count(), 1)

        account = Account.objects.get(user=self.user)
        self.assertLess(account.balance, Decimal("10000.00"))

    @mock.patch("trading.views.get_market_data_service")
    def test_sell_realizes_pnl(self, mock_market_service):
        mock_market_service.return_value.get_latest_data.return_value = self.tick

        # Open a position first
        buy_resp = self.client.post(
            "/api/trading/demo/orders/",
            {"action": "BUY", "symbol": "BTCUSDT", "quantity": "0.02"},
            format="json",
        )
        self.assertEqual(buy_resp.status_code, status.HTTP_200_OK)

        # Move price higher to realize profit on sell
        higher_tick = self.tick.copy()
        higher_tick["price"] = Decimal("105.00")
        mock_market_service.return_value.get_latest_data.return_value = higher_tick

        sell_resp = self.client.post(
            "/api/trading/demo/orders/",
            {"action": "SELL", "symbol": "BTCUSDT", "quantity": "0.02"},
            format="json",
        )
        self.assertEqual(sell_resp.status_code, status.HTTP_200_OK)

        trade = Trade.objects.filter(user=self.user, action="SELL").first()
        self.assertIsNotNone(trade)
        self.assertGreater(trade.pnl, Decimal("0"))
        self.assertEqual(Position.objects.filter(user=self.user, is_open=True).count(), 0)
