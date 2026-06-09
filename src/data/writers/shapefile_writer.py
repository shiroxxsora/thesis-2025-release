"""Writer для экспорта в Shapefile."""

import logging
from typing import List, Any
import geopandas as gpd
from shapely.geometry import mapping

from .base import DataWriter

logger = logging.getLogger(__name__)


class ShapefileWriter(DataWriter):
    """Класс для записи в Shapefile."""
    
    def write(self, objects: List[Any], path: str, crs: Any = None) -> None:
        """
        Записывает объекты в Shapefile.
        
        Args:
            objects: Список объектов с атрибутом geometry
            path: Путь к выходному файлу
            crs: Система координат
        """
        if not objects:
            logger.warning(f"Нет данных для записи в {path}")
            return
        
        # Создаём список геометрий
        geometries = [obj.geometry for obj in objects]
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        
        # Добавляем атрибуты с короткими именами (ограничение Shapefile - 10 символов)
        # Определяем тип объектов по первому элементу
        first_obj = objects[0]
        
        if hasattr(first_obj, 'violation_area'):
            # Для нарушений
            gdf['cad_num'] = [obj.cadastral_number for obj in objects]
            gdf['viol_area'] = [obj.violation_area for obj in objects]
            gdf['orig_area'] = [obj.original_object_area for obj in objects]
        elif hasattr(first_obj, 'cadastral_number'):
            # Для кадастровых участков
            gdf['cad_num'] = [obj.cadastral_number for obj in objects]
            gdf['area_m2'] = [obj.area_sqm for obj in objects]
        else:
            # Для обнаруженных объектов
            gdf['area_m2'] = [obj.area_sqm for obj in objects]
            gdf['centroid_x'] = [obj.centroid[0] for obj in objects]
            gdf['centroid_y'] = [obj.centroid[1] for obj in objects]
        
        # Записываем с кодировкой UTF-8
        try:
            gdf.to_file(path, encoding='utf-8')
        except Exception:
            gdf.to_file(path)
        
        logger.info(f"Записано {len(objects)} объектов в {path}")
