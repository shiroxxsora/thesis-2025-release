# Модуль экспорта документов по участкам

Рефакторинг монолитного `per_parcel_export.py` (1170 строк) в модульную архитектуру.

## Архитектура

### Модули

```
src/export/
├── __init__.py                  # Экспорты модуля
├── parcel_exporter.py           # Главный оркестратор (250 строк)
├── report_loader.py             # Загрузка Excel данных (70 строк)
├── pdf_builder.py               # Генерация PDF документов (250 строк)
├── docx_builder.py              # Генерация DOCX документов (180 строк)
├── map_generator.py             # Создание карт (450 строк)
├── coordinate_presenter.py      # Утилиты координат (130 строк)
└── README.md                    # Эта документация
```

**Итого:** ~1330 строк модульного кода vs 1170 строк монолитного

### Принципы

- **Single Responsibility Principle (SRP):** Каждый модуль отвечает за одну задачу
- **Dependency Injection (DI):** Зависимости передаются через конструкторы
- **Separation of Concerns:** Бизнес-логика отделена от представления
- **Testability:** Каждый модуль можно тестировать независимо

## Использование

### Через главный пайплайн

```python
from src.main import ComprehensiveAnalyzer

# Автоматически выполняется на шаге 9/9
analyzer = ComprehensiveAnalyzer()
result = analyzer.run_analysis()
```

### Standalone (через обёртку)

```bash
# Базовый запуск
python per_parcel_export_NEW.py

# С параметрами
python per_parcel_export_NEW.py \
    --output output/comprehensive \
    --format pdf \
    --geotiff geotiffs/input.tiff \
    --limit 5
```

### Программный API

```python
from src.export import ParcelExporter

exporter = ParcelExporter(
    output_dir='output/comprehensive',
    font_path='C:\\Windows\\Fonts\\arial.ttf',
    geotiff_path='geotiffs/input.tiff',
    proj_string='+proj=tmerc ...',
    min_point_spacing=3.0,
    max_points=100
)

output_path = exporter.export(
    only_with_violations=True,
    limit=None,
    out_format='pdf'
)
```

## Модули (детально)

### 1. `parcel_exporter.py`

**Главный оркестратор** - координирует работу всех модулей.

**Ключевые методы:**
- `export()` - главная точка входа
- `_load_background()` - предзагрузка GeoTIFF подложки
- `_process_one_parcel()` - обработка одного участка (многопоточно)
- `_create_zoom_map()` - создание детальной карты
- `_create_overview_map()` - создание обзорной карты

**Использует:**
- `ReportLoader` для загрузки Excel
- `PDFBuilder` / `DOCXBuilder` для создания документов
- `MapGenerator` для создания карт
- `ThreadPoolExecutor` для параллельной обработки

### 2. `report_loader.py`

**Загрузчик Excel данных**.

**Ключевые методы:**
- `load_frames()` - читает 4 листа из report.xlsx

**Читает листы:**
- "2. Кадастровые участки"
- "3. Нарушения"
- "4. Координаты участков"
- "5. Координаты нарушений"

### 3. `pdf_builder.py`

**Генератор PDF документов** через ReportLab.

**Ключевые методы:**
- `generate()` - создание PDF файла
- `build_story()` - формирование содержимого
- `_setup_font()` - регистрация кириллического шрифта
- `_add_violations_summary()` - таблица нарушений
- `_add_violation_coords_tables()` - координаты точек

**Структура документа:**
1. Заголовок (СХЕМА обмера)
2. Кадастровый номер
3. Сводка по нарушениям (количество, площадь)
4. Таблица нарушений (№, площадь, центроиды)
5. Таблицы координат по каждому нарушению
6. Обзорная карта
7. Детальная карта

### 4. `docx_builder.py`

**Генератор DOCX документов** через python-docx.

**Аналогичная структура** как PDF, но в формате Word.

**Особенности:**
- Настройка шрифтов для кириллицы (Times New Roman)
- Вставка изображений (Inches)
- Таблицы через `Document.add_table()`

### 5. `map_generator.py`

**Генератор карт** через matplotlib.

**Ключевые методы:**
- `create_zoom_map()` - детальная карта нарушений
- `create_overview_map()` - обзорная карта участка с окружением
- `_plot_shapely()` - отрисовка геометрии Shapely
- `_draw_violation_points()` - нумерация точек нарушений
- `_save_figure()` - сохранение с контролем размера

**Возможности:**
- Подложка из GeoTIFF
- Перепроецирование координат (pyproj)
- Автоматическое масштабирование
- Контроль размера изображений (Pillow)
- Нумерация точек на карте
- Подписи нарушений с площадью

### 6. `coordinate_presenter.py`

**Утилиты координат** - чистые функции.

**Функции:**
- `present_xy()` - трансформация координат для документов
- `format_float()` - форматирование чисел
- `compute_distances()` - расстояния между точками
- `simplify_points()` - упрощение контуров для визуализации
- `safe_filename()` - безопасные имена файлов

## Переиспользование компонентов

### Из существующих модулей:
- ✅ `GeoTiffReader` - чтение растра (не нужен, используем GDAL напрямую)
- ✅ `logging_config` - настройка логирования
- ✅ `shapely`, `pandas`, `matplotlib` - библиотеки

### Новые зависимости:
- `reportlab` - PDF генерация
- `python-docx` - DOCX генерация (опционально)
- `pyproj` - перепроецирование координат (опционально)
- `PIL/Pillow` - контроль размера изображений (опционально)

## Сравнение со старым кодом

| Аспект | Старый (`per_parcel_export.py`) | Новый (`src/export/`) |
|--------|----------------------------------|------------------------|
| **Строк кода** | 1170 (1 файл) | ~1330 (6 модулей) |
| **Читаемость** | Монолит, сложно ориентироваться | Модули по ответственности |
| **Тестируемость** | Низкая (всё в одном файле) | Высокая (каждый модуль независим) |
| **Переиспользование** | Невозможно | Модули можно использовать отдельно |
| **Расширяемость** | Сложно добавлять функции | Легко добавить новый формат экспорта |
| **Зависимости** | Внутри функций | Явные через DI |
| **Логирование** | `print()` | `logging` module |

## Обратная совместимость

Старый `per_parcel_export.py` сохранён без изменений для обратной совместимости:

```bash
# Старая версия (монолитная)
python per_parcel_export.py

# Новая версия (модульная)
python per_parcel_export_NEW.py
```

Оба работают с одним и тем же Excel-отчётом.

## Миграция

### Автоматическая (рекомендуется)

```bash
python comprehensive_analysis_NEW.py
# → автоматически вызывает per_parcel_export на шаге 9/9
```

### Ручная

```python
from src.export import ParcelExporter

exporter = ParcelExporter('output/comprehensive')
exporter.export()
```

## Производительность

- **Многопоточность:** 4 параллельных потока (настраиваемо)
- **Предзагрузка:** GeoTIFF загружается 1 раз для всех участков
- **Кэширование:** Шрифты регистрируются 1 раз
- **Оптимизация изображений:** Автоматическое уменьшение размера при необходимости

## Логирование

Все модули используют стандартный `logging`:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Информация")
logger.warning("Предупреждение")
logger.error("Ошибка")
logger.debug("Отладка")
```

Настройка через `src/utils/logging_config.py`.

## Известные ограничения

1. **DOCX:** Требует `python-docx` (опционально)
2. **Подложка GeoTIFF:** Требует GDAL (опционально)
3. **Перепроецирование:** Требует `pyproj` (опционально)
4. **Размер изображений:** Ограничен 65536 пикселями (matplotlib)

## Будущие улучшения

- [ ] Асинхронная генерация (asyncio)
- [ ] Кэширование карт (если участок не изменился)
- [ ] Поддержка HTML/Markdown экспорта
- [ ] Шаблоны документов (настраиваемые)
- [ ] Batch processing (группы участков)
- [ ] Прогресс-бар (tqdm)

## Тестирование

```bash
# Тест на 5 участках
python per_parcel_export_NEW.py --limit 5

# Только PDF
python per_parcel_export_NEW.py --format pdf

# PDF + DOCX
python per_parcel_export_NEW.py --format both
```

## Авторы рефакторинга

- **Исходный код:** `per_parcel_export.py` (1170 строк)
- **Рефакторинг:** Разбивка на модули по принципам SOLID
- **Дата:** Январь 2026
