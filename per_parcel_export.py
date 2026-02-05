"""
Обёртка для экспорта документов по участкам (ОТРЕФАКТОРЕННАЯ версия).

Использует новый модульный код из src/export/ вместо монолитного per_parcel_export.py.

Архитектура:
- src/export/parcel_exporter.py - главный оркестратор
- src/export/report_loader.py - загрузка данных из Excel
- src/export/pdf_builder.py - генерация PDF
- src/export/docx_builder.py - генерация DOCX
- src/export/map_generator.py - создание карт (zoom + overview)
- src/export/coordinate_presenter.py - утилиты координат

СТАРЫЙ МОНОЛИТНЫЙ КОД: per_parcel_export.py (1170 строк)
НОВЫЙ МОДУЛЬНЫЙ КОД: src/export/ (6 модулей, ~700 строк)
"""

import sys
import os
import argparse

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.export import ParcelExporter
from src.utils.logging_config import setup_logging
import logging


def main():
    """Точка входа для экспорта документов по участкам."""
    setup_logging(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description='Экспорт отдельных документов по каждому участку (PDF/DOCX).'
    )
    parser.add_argument(
        '--output',
        default=os.path.join('output', 'comprehensive'),
        help='Директория с результатами комплексного анализа'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Экспорт для всех участков, а не только имеющих нарушения'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Ограничить число обрабатываемых участков (для проверки)'
    )
    parser.add_argument(
        '--font',
        type=str,
        default=None,
        help='Путь к TTF-шрифту с поддержкой кириллицы'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['pdf', 'docx', 'both'],
        default='pdf',
        help='Формат выходных файлов'
    )
    parser.add_argument(
        '--geotiff',
        type=str,
        default=None,
        help='Путь к GeoTIFF для подложки (фоновая карта)'
    )
    parser.add_argument(
        '--proj-string',
        type=str,
        default=None,
        help='PROJ-строка для исходной системы координат векторов'
    )
    parser.add_argument(
        '--min-point-spacing',
        type=float,
        default=3.0,
        help='Минимальное расстояние между точками нарушений (м) для упрощения'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        default=None,
        help='Максимальное количество точек для отображения на одно нарушение'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ЭКСПОРТ ДОКУМЕНТОВ ПО УЧАСТКАМ (REFACTORED)")
    print("="*80)
    print()
    
    try:
        exporter = ParcelExporter(
            output_dir=args.output,
            font_path=args.font,
            geotiff_path=args.geotiff,
            proj_string=args.proj_string,
            min_point_spacing=args.min_point_spacing,
            max_points=args.max_points
        )
        
        output_path = exporter.export(
            only_with_violations=not args.all,
            limit=args.limit,
            out_format=args.format
        )
        
        print(f"\n✓ Документы успешно созданы: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Экспорт прерван: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
