"""
Команда для скачивания исторических данных в CSV файлы для backtest.

Использование:
    python manage.py download_historical_data --symbol AAPL --period 1y --interval 1h
    python manage.py download_historical_data --symbol BTCUSDT --period 1mo --interval 1h
"""
import os
from django.core.management.base import BaseCommand
from trading.agents.market_monitor import MarketMonitoringAgent


class Command(BaseCommand):
    help = "Скачивает исторические данные в CSV файл для backtest"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            required=True,
            help="Символ для скачивания (например, AAPL, BTCUSDT)",
        )
        parser.add_argument(
            "--period",
            type=str,
            default="1y",
            help="Период данных (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            help="Интервал данных (1m, 5m, 15m, 30m, 1h, 4h, 1d)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./data",
            help="Директория для сохранения CSV файлов (по умолчанию: ./data)",
        )

    def handle(self, *args, **options):
        symbol = options["symbol"].upper()
        period = options["period"]
        interval = options["interval"]
        output_dir = options["output_dir"]

        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(self.style.SUCCESS("СКАЧИВАНИЕ ИСТОРИЧЕСКИХ ДАННЫХ"))
        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(f"Символ: {symbol}")
        self.stdout.write(f"Период: {period}")
        self.stdout.write(f"Интервал: {interval}")
        self.stdout.write(f"Выходная директория: {output_dir}\n")

        # Создаем директорию если не существует
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Инициализируем агент
            self.stdout.write("[1/3] Инициализация агента...")
            market_agent = MarketMonitoringAgent(
                ticker=symbol,
                interval=interval,
                period=period,
                enable_cache=True,
                request_delay=5.0,  # Задержка для обхода блокировок
                max_retries=5,
                backoff_factor=3.0,
            )
            self.stdout.write(self.style.SUCCESS("✓ Агент инициализирован"))

            # Загружаем данные
            self.stdout.write("\n[2/3] Загрузка данных...")
            self.stdout.write("Это может занять некоторое время, особенно если yfinance блокируется...")
            
            try:
                data, analysis = market_agent.get_processed_data(analyze=True)
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Ошибка при получении данных: {e}"))
                self.stdout.write("Пробуем получить данные без анализа...")
                data = market_agent.get_processed_data(analyze=False)
                analysis = None

            if data is None or data.empty:
                self.stdout.write(self.style.ERROR("Не удалось загрузить данные"))
                return

            self.stdout.write(self.style.SUCCESS(f"✓ Загружено {len(data)} записей"))
            if len(data) > 0:
                self.stdout.write(f"  Первая дата: {data.index[0]}")
                self.stdout.write(f"  Последняя дата: {data.index[-1]}")

            # Сохраняем в CSV
            self.stdout.write("\n[3/3] Сохранение в CSV...")
            filename = f"{symbol}_{interval}.csv"
            filepath = os.path.join(output_dir, filename)

            # Убеждаемся, что timestamp в колонках и правильно сохранен
            # Если индекс - это DatetimeIndex, используем его напрямую
            if isinstance(data.index, pd.DatetimeIndex):
                data_to_save = data.copy()
            elif 'timestamp' in data.columns:
                # Если timestamp в колонках, используем его как индекс
                data_to_save = data.copy()
                if not isinstance(data_to_save.index, pd.DatetimeIndex):
                    data_to_save['timestamp'] = pd.to_datetime(data_to_save['timestamp'], errors='coerce')
                    data_to_save = data_to_save.set_index('timestamp')
            else:
                # Иначе создаем timestamp из индекса
                data_to_save = data.reset_index()
                if 'timestamp' not in data_to_save.columns:
                    # Если индекс был DatetimeIndex, используем его
                    if isinstance(data.index, pd.DatetimeIndex):
                        data_to_save['timestamp'] = data.index
                    else:
                        # Пробуем конвертировать индекс в datetime
                        data_to_save['timestamp'] = pd.to_datetime(data_to_save.index, errors='coerce')
                else:
                    # Убеждаемся, что timestamp - это datetime
                    data_to_save['timestamp'] = pd.to_datetime(data_to_save['timestamp'], errors='coerce')
                data_to_save = data_to_save.set_index('timestamp')
            
            # Убеждаемся, что индекс - это DatetimeIndex
            if not isinstance(data_to_save.index, pd.DatetimeIndex):
                data_to_save.index = pd.to_datetime(data_to_save.index, errors='coerce')
            
            # Удаляем строки с невалидными датами
            data_to_save = data_to_save.dropna(subset=[data_to_save.index.name if data_to_save.index.name else None])
            
            # Сохраняем с правильным форматом дат
            data_to_save.to_csv(filepath, index=True, date_format='%Y-%m-%d %H:%M:%S')
            self.stdout.write(self.style.SUCCESS(f"✓ Данные сохранены в: {filepath}"))

            # Показываем статистику
            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.SUCCESS("СТАТИСТИКА"))
            self.stdout.write("=" * 70)
            self.stdout.write(f"Всего записей: {len(data)}")
            self.stdout.write(f"Колонки: {', '.join(data.columns)}")
            
            if analysis:
                self.stdout.write(f"\nАнализ рынка:")
                self.stdout.write(f"  Тренд: {analysis.get('trend', 'unknown')}")
                self.stdout.write(f"  Сила: {analysis.get('strength', 0.0):.2f}")
            
            self.stdout.write(f"\nФайл готов для использования в backtest:")
            self.stdout.write(self.style.SUCCESS(f"  {filepath}"))
            self.stdout.write("\nИспользование:")
            self.stdout.write(f"  docker compose exec backend python manage.py backtest_simulation \\")
            self.stdout.write(f"    --email your@email.com \\")
            self.stdout.write(f"    --symbol {symbol} \\")
            self.stdout.write(f"    --start-date 2024-11-01 \\")
            self.stdout.write(f"    --end-date 2024-12-01 \\")
            self.stdout.write(f"    --interval {interval}")
            self.stdout.write("=" * 70)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Ошибка: {e}"))
            import traceback
            self.stdout.write(traceback.format_exc())

