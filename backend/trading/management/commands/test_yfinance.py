"""
Команда для проверки работы yfinance

Проверяет, может ли yfinance получать данные с Yahoo Finance
"""
import sys
from django.core.management.base import BaseCommand

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class Command(BaseCommand):
    help = "Проверяет работу yfinance для получения данных рынка"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default="AAPL",
            help="Символ для тестирования (по умолчанию: AAPL)",
        )

    def handle(self, *args, **options):
        symbol = options.get("symbol", "AAPL")
        
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(self.style.SUCCESS("ПРОВЕРКА YFINANCE"))
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write(f"Символ: {symbol}\n")
        
        if not YFINANCE_AVAILABLE:
            self.stdout.write(self.style.ERROR("✗ yfinance не установлен!"))
            self.stdout.write("Установите: pip install yfinance")
            return
        
        self.stdout.write("✓ yfinance установлен\n")
        
        # Тест 1: Базовое получение тикера
        self.stdout.write("[1/4] Создание тикера...")
        try:
            ticker = yf.Ticker(symbol)
            self.stdout.write(self.style.SUCCESS("✓ Тикер создан"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"✗ Ошибка создания тикера: {str(e)}"))
            return
        
        # Тест 2: Получение info
        self.stdout.write("\n[2/4] Получение info...")
        try:
            info = ticker.info
            if info and len(info) > 0:
                self.stdout.write(self.style.SUCCESS(f"✓ Info получен ({len(info)} полей)"))
                if "longName" in info:
                    self.stdout.write(f"  Название: {info.get('longName', 'N/A')}")
                if "currentPrice" in info:
                    self.stdout.write(f"  Текущая цена: ${info.get('currentPrice', 'N/A')}")
            else:
                self.stdout.write(self.style.WARNING("⚠ Info пустой"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"⚠ Не удалось получить info: {str(e)}"))
        
        # Тест 3: Получение исторических данных (1 день)
        self.stdout.write("\n[3/4] Получение исторических данных (1 день)...")
        try:
            hist = ticker.history(period="1d", interval="1h")
            if not hist.empty:
                self.stdout.write(self.style.SUCCESS(f"✓ Данные получены ({len(hist)} записей)"))
                latest = hist.iloc[-1]
                self.stdout.write(f"  Последняя цена: ${latest['Close']:.2f}")
                self.stdout.write(f"  Объем: {int(latest['Volume'])}")
            else:
                self.stdout.write(self.style.WARNING("⚠ Данные пустые"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"✗ Ошибка получения данных: {str(e)}"))
            self.stdout.write("\nВозможные причины:")
            self.stdout.write("  - Нет интернет-соединения")
            self.stdout.write("  - Yahoo Finance недоступен")
            self.stdout.write("  - Проблемы с прокси/файрволом")
            self.stdout.write("  - Символ не существует")
            return
        
        # Тест 4: Получение данных за месяц (как в MarketMonitoringAgent)
        self.stdout.write("\n[4/4] Получение данных за месяц (1h интервал)...")
        try:
            hist = ticker.history(period="1mo", interval="1h")
            if not hist.empty:
                self.stdout.write(self.style.SUCCESS(f"✓ Данные получены ({len(hist)} записей)"))
                self.stdout.write(f"  Период: {hist.index[0]} - {hist.index[-1]}")
                self.stdout.write(f"  Последняя цена: ${hist.iloc[-1]['Close']:.2f}")
            else:
                self.stdout.write(self.style.WARNING("⚠ Данные пустые"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"✗ Ошибка получения данных: {str(e)}"))
            return
        
        # Итог
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("✓ YFINANCE РАБОТАЕТ КОРРЕКТНО"))
        self.stdout.write(self.style.SUCCESS("="*70))
        self.stdout.write("\nMarketMonitoringAgent должен работать нормально.")

