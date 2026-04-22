"""
ОБЁРТКА для обратной совместимости со старым API.

Этот файл является тонкой обёрткой вокруг отрефакторенного кода из /src.
Сохраняет старый API для совместимости с существующими скриптами.

СТАРЫЙ КОД сохранён в comprehensive_analysis_legacy.py.bak
"""

import os
import sys
from typing import Optional, Dict
from datetime import datetime

# Настройка пути для импорта из src/
sys.path.insert(0, os.path.dirname(__file__))

# Импорты из отрефакторенного кода
from src.main import ComprehensiveAnalyzer as RefactoredAnalyzer
from src.config.settings import AnalysisConfig
from src.config.constants import DEFAULT_MAP_FIGURE_DPI, DEFAULT_MAP_FIGURE_SIZE
from src.utils.logging_config import setup_logging
import logging


class ComprehensiveAnalyzer:
    """
    ОБЁРТКА для старого API comprehensive_analysis.py.
    
    Использует отрефакторенный код из src/, но предоставляет
    старый интерфейс для обратной совместимости.
    """
    
    def __init__(self):
        """Инициализация анализатора (совместимость со старым API)."""
        # Старая конфигурация в виде словаря (для совместимости)
        self.config = {
            'input_geotiff': "geotiffs/input.tiff",
            'cadastral_data': "cadastr/ЗУ все2.MIF",
            'detectron_config': "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            'model_weights': "models/pavel-01-07-25/model_final.pth",
            'output_dir': "output/comprehensive",
            'chunk_size': 4960,
            'overlap': 1536,
            'model_input_size': 1024,
            'score_threshold': 0.3,
            'min_polygon_area': 100.0,
            'min_violation_area': 0.5,
            'simplify_tolerance_m': 0.5,
            'cv_eps_factor': 0.004,
            'binding_min_intersection_ratio': 0.1,
            'binding_boundary_buffer_m': 3.0,
            'binding_max_nearest_distance_m': 25.0,
            # True: в Excel и per_parcel одна геометрия на ЗУ (union фрагментов по привязке к кадастру)
            'merge_violations_per_parcel': True,

            # Резкость подложки карты: чем выше, тем больше деталей (и расход RAM/время).
            # None = полный растр (тайловая отрисовка снижает пик памяти, но всё равно может быть тяжело).
            'map_max_raster_edge': None,
        }
        
        # Результаты в старом формате
        self.results = {
            'detected_objects': [],
            'cadastral_parcels': [],
            'violations': [],
            'statistics': {}
        }
        
        # Создаём выходную директорию
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Настраиваем логирование (совместимо с print из старого кода)
        setup_logging(level=logging.INFO)
        
        # Создаём новый анализатор внутри
        new_config = AnalysisConfig(
            input_geotiff=self.config['input_geotiff'],
            cadastral_data=self.config['cadastral_data'],
            output_dir=self.config['output_dir']
        )
        new_config.detector.config_file = self.config['detectron_config']
        new_config.detector.model_weights = self.config['model_weights']
        new_config.detector.chunk_size = self.config['chunk_size']
        new_config.detector.overlap = self.config['overlap']
        new_config.detector.model_input_size = self.config['model_input_size']
        new_config.detector.score_threshold = self.config['score_threshold']
        new_config.min_polygon_area = self.config['min_polygon_area']
        new_config.min_violation_area = self.config['min_violation_area']
        new_config.simplify_tolerance_m = self.config['simplify_tolerance_m']
        new_config.cv_eps_factor = self.config['cv_eps_factor']
        new_config.binding_min_intersection_ratio = self.config['binding_min_intersection_ratio']
        new_config.binding_boundary_buffer_m = self.config['binding_boundary_buffer_m']
        new_config.binding_max_nearest_distance_m = self.config['binding_max_nearest_distance_m']
        new_config.merge_violations_per_parcel = bool(
            self.config.get('merge_violations_per_parcel', True)
        )

        new_config.map_figure_size = DEFAULT_MAP_FIGURE_SIZE
        new_config.map_figure_dpi = DEFAULT_MAP_FIGURE_DPI
        new_config.map_max_raster_edge = self.config.get('map_max_raster_edge', new_config.map_max_raster_edge)

        self._analyzer = RefactoredAnalyzer(new_config)
        
        print("="*80)
        print("ИСПОЛЬЗУЕТСЯ ОТРЕФАКТОРЕННАЯ ВЕРСИЯ (из src/)")
        print("Старый код сохранён в comprehensive_analysis_legacy.py.bak")
        print("="*80)
    
    def run_analysis(self):
        """
        Запускает анализ (совместимость со старым API).
        
        Внутри вызывает отрефакторенный код, но возвращает результат
        в старом формате для совместимости.
        """
        print("\nЗапуск анализа через отрефакторенный код...")
        
        # Вызываем новый анализатор
        result = self._analyzer.run_analysis()
        
        # Конвертируем результат в старый формат
        self.results = result.to_legacy_format()
        
        print("\n" + "="*80)
        print("АНАЛИЗ ЗАВЕРШЁН!")
        print(f"Результаты сохранены в: {self.config['output_dir']}")
        print("="*80)
    
    # СТАРЫЕ МЕТОДЫ - ОСТАВЛЕНЫ ДЛЯ СОВМЕСТИМОСТИ (не используются)
    # Если кто-то вызывает их напрямую, выдаём предупреждение
    
    def setup_model(self):
        """[DEPRECATED] Используйте новый API."""
        print("[ПРЕДУПРЕЖДЕНИЕ] setup_model() устарел. Используется автоматически.")
        return self._analyzer.detector.predictor
    
    def normalize_to_uint8(self, ch):
        """[DEPRECATED] Используйте новый API."""
        print("[ПРЕДУПРЕЖДЕНИЕ] normalize_to_uint8() устарел.")
        from src.utils.image_utils import normalize_to_uint8
        return normalize_to_uint8(ch)
    
    def read_geotiff(self, file_path: str) -> Optional[Dict]:
        """[DEPRECATED] Используйте новый API."""
        print("[ПРЕДУПРЕЖДЕНИЕ] read_geotiff() устарел.")
        from src.data.readers import GeoTiffReader
        reader = GeoTiffReader()
        data = reader.read(file_path)
        # Возвращаем в старом формате
        return {
            'width': data.width,
            'height': data.height,
            'num_bands': data.num_bands,
            'data': data.data,
            'transform': data.transform,
            'crs': data.crs,
            'projection_wkt': data.projection_wkt,
            'proj4_projection': data.proj4_projection,
            'bounds': data.bounds
        }


def main():
    """Точка входа (совместимость со старым кодом)."""
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
