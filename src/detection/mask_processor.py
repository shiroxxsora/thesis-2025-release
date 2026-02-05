"""Процессор для обработки масок из нейросети."""

import logging
import numpy as np
from typing import List
from shapely.geometry import Polygon

from ..domain.models import DetectedObject
from ..domain.types import GeoTransform
from ..processing.geometry_processor import GeometryProcessor

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Класс для обработки масок, полученных от нейросети."""
    
    def __init__(
        self,
        geometry_processor: GeometryProcessor,
        min_polygon_area: float = 100.0
    ):
        """
        Инициализация процессора масок.
        
        Args:
            geometry_processor: Процессор геометрии
            min_polygon_area: Минимальная площадь полигона в пикселях
        """
        self.geometry_processor = geometry_processor
        self.min_polygon_area = min_polygon_area
    
    def process_masks(
        self,
        masks: np.ndarray,
        transform: GeoTransform,
        chunk_offset: tuple,
        chunk_id: str
    ) -> List[DetectedObject]:
        """
        Обрабатывает маски и создаёт объекты DetectedObject.
        
        Args:
            masks: Массив масок (N, H, W) от Detectron2
            transform: GeoTransform для преобразования координат
            chunk_offset: Смещение чанка (x_offset, y_offset)
            chunk_id: ID чанка
            
        Returns:
            Список обнаруженных объектов
        """
        if masks is None or len(masks) == 0:
            return []
        
        detected_objects = []
        
        # Обрабатываем каждую маску отдельно
        for mask_idx, mask in enumerate(masks):
            objects = self._process_single_mask(
                mask,
                transform,
                chunk_offset,
                chunk_id,
                mask_idx
            )
            detected_objects.extend(objects)
        
        logger.debug(
            f"Обработано {len(masks)} масок, "
            f"создано {len(detected_objects)} объектов"
        )
        
        return detected_objects
    
    def _process_single_mask(
        self,
        mask: np.ndarray,
        transform: GeoTransform,
        chunk_offset: tuple,
        chunk_id: str,
        mask_idx: int
    ) -> List[DetectedObject]:
        """Обрабатывает одну маску."""
        # Извлекаем полигоны из маски
        pixel_polygons = self.geometry_processor.extract_polygons_from_mask(
            mask,
            self.min_polygon_area
        )
        
        detected_objects = []
        
        for poly_idx, pixel_poly in enumerate(pixel_polygons):
            # Преобразуем в географические координаты
            geo_polygon = self.geometry_processor.convert_to_geo_polygon(
                pixel_poly,
                transform,
                offset=chunk_offset
            )
            
            # Проверяем валидность
            if not geo_polygon or geo_polygon.is_empty or geo_polygon.area == 0:
                continue
            
            # Создаём объект
            instance_id = f"{chunk_id}_mask_{mask_idx}_poly_{poly_idx}"
            
            obj = DetectedObject(
                geometry=geo_polygon,
                area_sqm=geo_polygon.area,
                centroid=geo_polygon.centroid.coords[0],
                instance_id=instance_id,
                chunk_id=chunk_id,
                mask_id=mask_idx,
                mask_area_pixels=int(np.sum(mask > 0))
            )
            
            detected_objects.append(obj)
        
        return detected_objects
