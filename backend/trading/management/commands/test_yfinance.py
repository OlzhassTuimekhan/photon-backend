"""
Команда для проверки работы yfinance

Проверяет, может ли yfinance получать данные с Yahoo Finance
"""
import sys
import requests
from django.core.management.base import BaseCommand

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Настройка User-Agent для обхода блокировок
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def _setup_yfinance_headers():
    """Настраивает заголовки для yfinance запросов"""
    original_get = requests.get
    original_post = requests.post
    
    def patched_get(url, **kwargs):
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].setdefault("User-Agent", _DEFAULT_USER_AGENT)
        kwargs["headers"].setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        kwargs["headers"].setdefault("Accept-Language", "en-US,en;q=0.5")
        return original_get(url, **kwargs)
    
    def patched_post(url, **kwargs):
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].setdefault("User-Agent", _DEFAULT_USER_AGENT)
        kwargs["headers"].setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        kwargs["headers"].setdefault("Accept-Language", "en-US,en;q=0.5")
        return original_post(url, **kwargs)
    
    if not hasattr(requests, '_yfinance_patched'):
        requests.get = patched_get
        requests.post = patched_post
        requests._yfinance_patched = True


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
        
        self.stdout.write("✓ yfinance установлен")
        
        # Настраиваем User-Agent заголовки
        _setup_yfinance_headers()
        self.stdout.write("✓ User-Agent заголовки настроены\n")
        
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
        
        # Проверяем результаты
        tests_passed = 0
        if YFINANCE_AVAILABLE:
            tests_passed += 1
        try:
            if ticker and not hist.empty:
                tests_passed += 1
        except:
            pass
        
        if tests_passed >= 2:
            self.stdout.write(self.style.SUCCESS("✓ YFINANCE РАБОТАЕТ КОРРЕКТНО"))
            self.stdout.write(self.style.SUCCESS("="*70))
            self.stdout.write("\nMarketMonitoringAgent должен работать нормально.")
        else:
            self.stdout.write(self.style.ERROR("✗ YFINANCE НЕ РАБОТАЕТ"))
            self.stdout.write(self.style.ERROR("="*70))
            self.stdout.write("\nПРОБЛЕМЫ:")
            self.stdout.write("  - Yahoo Finance блокирует запросы (429 Too Many Requests)")
            self.stdout.write("  - Нет интернет-соединения")
            self.stdout.write("  - Проблемы с DNS/прокси")
            self.stdout.write("\nРЕШЕНИЯ:")
            self.stdout.write("  1. Добавить задержки между запросами")
            self.stdout.write("  2. Использовать прокси для обхода блокировки")
            self.stdout.write("  3. Использовать альтернативные источники (Bybit для крипты)")
            self.stdout.write("  4. Увеличить кеширование данных")
            self.stdout.write("  5. Использовать user-agent заголовки")

