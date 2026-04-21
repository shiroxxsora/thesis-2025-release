"""Главный модуль для комплексного анализа геопространственных данных."""

import logging
from pathlib import Path
from typing import Optional

from .config.settings import AnalysisConfig
from .domain.models import AnalysisResult, GeoTiffData
from .utils.logging_config import setup_logging, get_logger
from .data.readers import GeoTiffReader, CadastralReader
from .data.writers import ShapefileWriter, ExcelWriter, JSONWriter
from .detection import ObjectDetector, Deduplicator
from .analysis import ViolationAnalyzer
from .visualization import MapVisualizer
from .reporting import PDFReportGenerator
from .export import ParcelExporter
from .utils.document_geometry import (
    cadastral_parcels_for_maps_and_documents,
    violations_for_maps_and_documents,
)
from .export.coordinate_presenter import parse_false_easting_x0_from_proj4


class ComprehensiveAnalyzer:
    """
    Главный класс для комплексного анализа.
    
    Оркестрирует работу всех компонентов системы:
    - Чтение GeoTIFF и кадастровых данных
    - Детекция объектов нейросетью
    - Анализ нарушений
    - Экспорт результатов
    - Визуализация и отчёты
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Инициализация анализатора.
        
        Args:
            config: Конфигурация анализа (если None, используется default)
        """
        self.config = config or AnalysisConfig()
        self.logger = get_logger(__name__)
        
        # Инициализируем компоненты
        self._init_components()
    
    def _init_components(self):
        """Инициализирует все компоненты системы."""
        self.logger.info("Инициализация компонентов...")
        
        # Readers
        self.geotiff_reader = GeoTiffReader()
        self.cadastral_reader = CadastralReader()
        
        # Detector
        self.detector = ObjectDetector(self.config.detector)
        self.deduplicator = Deduplicator(self.config.overlap_threshold)
        
        # Analyzer
        self.analyzer = ViolationAnalyzer(self.config)
        
        # Writers
        self.shapefile_writer = ShapefileWriter()
        self.excel_writer = ExcelWriter()
        self.json_writer = JSONWriter()
        
        # Visualization
        self.visualizer = MapVisualizer(
            self.config.output_dir,
            dpi=self.config.map_figure_dpi,
            max_raster_edge=self.config.map_max_raster_edge,
            figure_size=self.config.map_figure_size,
        )
        
        # Reporting
        self.pdf_generator = PDFReportGenerator(self.config.output_dir)
        
        # Export
        self.parcel_exporter = ParcelExporter(self.config.output_dir)
    
    def run_analysis(self) -> AnalysisResult:
        """
        Выполняет полный цикл анализа.
        
        Returns:
            Результаты анализа
        """
        self.logger.info("="*80)
        self.logger.info("НАЧАЛО КОМПЛЕКСНОГО АНАЛИЗА")
        self.logger.info("="*80)
        
        # 1. Загрузка GeoTIFF
        self.logger.info("\n[1/9] Загрузка GeoTIFF...")
        geotiff_data = self.geotiff_reader.read(self.config.input_geotiff)
        
        # 2. Обнаружение объектов
        self.logger.info("\n[2/9] Обнаружение объектов нейросетью...")
        detected_objects = self.detector.detect(geotiff_data)
        
        # 3. Дедупликация
        self.logger.info("\n[3/9] Дедупликация объектов...")
        detected_objects = self.deduplicator.deduplicate(detected_objects)
        
        # 4. Загрузка кадастровых данных
        self.logger.info("\n[4/9] Загрузка кадастровых данных...")
        cadastral_parcels = self.cadastral_reader.read(
            self.config.cadastral_data,
            target_crs=geotiff_data.crs,
            target_bounds=geotiff_data.bounds
        )
        
        # 5. Анализ нарушений
        self.logger.info("\n[5/9] Анализ нарушений...")
        violations = self.analyzer.analyze(
            detected_objects,
            cadastral_parcels,
            geotiff_data
        )

        # Упрощение контуров кадастра и нарушений — одни и те же данные: Excel, per_parcel, карта
        cadastral_docs = cadastral_parcels_for_maps_and_documents(
            cadastral_parcels, self.config.simplify_tolerance_m
        )
        violations_docs = violations_for_maps_and_documents(
            violations, self.config.simplify_tolerance_m
        )
        
        # 6. Экспорт результатов
        self.logger.info("\n[6/9] Экспорт результатов...")
        self._export_results(
            detected_objects,
            cadastral_parcels,
            cadastral_docs,
            violations,
            violations_docs,
            geotiff_data
        )
        
        # 7. Визуализация
        self.logger.info("\n[7/9] Создание визуализации...")
        self.visualizer.create_overview_map(
            geotiff_data,
            detected_objects,
            cadastral_docs,
            violations_docs
        )
        
        # 8. Генерация отчётов
        self.logger.info("\n[8/9] Генерация общего PDF отчёта...")
        self.pdf_generator.generate(
            detected_objects,
            cadastral_docs,
            violations_docs,
            geotiff_data
        )
        
        # 9. Экспорт документов по участкам
        self.logger.info("\n[9/9] Экспорт документов по каждому участку...")
        self.parcel_exporter.export()
        
        # Создаём результат
        result = AnalysisResult(
            detected_objects=detected_objects,
            cadastral_parcels=cadastral_parcels,
            violations=violations,
            geotiff_data=geotiff_data
        )
        
        self.logger.info("\n" + "="*80)
        self.logger.info("АНАЛИЗ ЗАВЕРШЁН УСПЕШНО")
        self.logger.info(f"Обнаружено объектов: {len(detected_objects)}")
        self.logger.info(f"Кадастровых участков: {len(cadastral_parcels)}")
        self.logger.info(f"Найдено нарушений: {len(violations)}")
        self.logger.info("="*80)
        
        return result
    
    def _export_results(
        self,
        detected_objects,
        cadastral_parcels,
        cadastral_docs,
        violations,
        violations_docs,
        geotiff_data
    ):
        """Экспортирует результаты во все форматы."""
        output_dir = Path(self.config.output_dir)
        
        # Shapefile — полная геометрия анализа
        self.shapefile_writer.write(
            detected_objects,
            str(output_dir / 'detected_objects.shp'),
            crs=geotiff_data.crs
        )
        self.shapefile_writer.write(
            cadastral_parcels,
            str(output_dir / 'cadastral_parcels.shp'),
            crs=geotiff_data.crs
        )
        self.shapefile_writer.write(
            violations,
            str(output_dir / 'violations.shp'),
            crs=geotiff_data.crs
        )
        
        # Excel: те же упрощённые контуры, что на глобальной карте и в per_parcel
        doc_x0 = parse_false_easting_x0_from_proj4(
            geotiff_data.proj4_projection or ""
        )
        self.excel_writer.write(
            detected_objects,
            cadastral_docs,
            violations_docs,
            str(output_dir / 'report.xlsx'),
            proj_false_easting_x0=doc_x0,
        )
        
        # JSON
        result_dict = {
            'detected_objects_count': len(detected_objects),
            'cadastral_objects_count': len(cadastral_parcels),
            'violations_count': len(violations),
            'total_detected_area': sum(obj.area_sqm for obj in detected_objects),
            'total_cadastral_area': sum(obj.area_sqm for obj in cadastral_parcels),
            'total_violation_area': sum(v.violation_area for v in violations),
        }
        self.json_writer.write(
            result_dict,
            str(output_dir / 'analysis_results.json')
        )


def main():
    """Точка входа для запуска анализа."""
    # Настраиваем логирование
    setup_logging(level=logging.INFO)
    
    # Создаём конфигурацию
    config = AnalysisConfig()
    
    # Запускаем анализ
    analyzer = ComprehensiveAnalyzer(config)
    result = analyzer.run_analysis()
    
    return result


if __name__ == "__main__":
    main()
