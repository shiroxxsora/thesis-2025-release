"""Процессор для работы с геометрическими объектами."""

import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
import logging

from ..domain.types import PolygonCoords, GeoTransform
from ..utils.geometry_utils import (
    pixel_to_geo_coords,
    simplify_polygon,
    validate_and_fix_polygon
)

logger = logging.getLogger(__name__)


class GeometryProcessor:
    """Класс для обработки геометрических данных."""
    
    def __init__(self, simplify_tolerance: float = 0.5, cv_eps_factor: float = 0.004):
        """
        Инициализация процессора.
        
        Args:
            simplify_tolerance: Толерантность упрощения полигонов в метрах
            cv_eps_factor: Фактор epsilon для cv2.approxPolyDP
        """
        self.simplify_tolerance = simplify_tolerance
        self.cv_eps_factor = cv_eps_factor
    
    def extract_polygons_from_mask(
        self,
        mask: np.ndarray,
        min_area: float = 100.0
    ) -> List[np.ndarray]:
        """
        Извлекает полигоны из бинарной маски.
        
        Args:
            mask: Бинарная маска (H, W)
            min_area: Минимальная площадь контура в пикселях
            
        Returns:
            Список полигонов в пиксельных координатах
        """
        # Находим контуры
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        
        for contour in contours:
            # Проверяем площадь
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Упрощаем контур
            epsilon = self.cv_eps_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Проверяем, что это полигон (минимум 3 точки)
            if len(approx) >= 3:
                poly = approx.reshape(-1, 2)
                polygons.append(poly)
        
        logger.debug(f"Извлечено {len(polygons)} полигонов из маски")
        return polygons
    
    def convert_to_geo_polygon(
        self,
        pixel_polygon: np.ndarray,
        transform: GeoTransform,
        offset: Tuple[int, int] = (0, 0)
    ) -> Polygon:
        """
        Конвертирует пиксельный полигон в географический.
        
        Args:
            pixel_polygon: Полигон в пиксельных координатах (N, 2)
            transform: GeoTransform для преобразования координат
            offset: Смещение координат (x_offset, y_offset)
            
        Returns:
            Географический полигон (Shapely Polygon)
        """
        # Применяем смещение если есть
        if offset != (0, 0):
            pixel_polygon = pixel_polygon + np.array(offset)
        
        # Преобразуем в географические координаты
        geo_coords = pixel_to_geo_coords(pixel_polygon, transform)
        
        # Создаём полигон
        if len(geo_coords) < 3:
            return Polygon()
        
        try:
            polygon = Polygon(geo_coords)
            
            # Упрощаем если нужно
            if self.simplify_tolerance > 0:
                polygon = simplify_polygon(
                    polygon,
                    self.simplify_tolerance,
                    preserve_topology=True
                )
            
            # Валидируем и исправляем если нужно
            if not polygon.is_valid:
                polygon = validate_and_fix_polygon(polygon)
            
            return polygon
        except Exception as e:
            logger.warning(f"Ошибка создания полигона: {e}")
            return Polygon()
    
    def merge_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Объединяет пересекающиеся полигоны.
        
        Args:
            polygons: Список полигонов для объединения
            
        Returns:
            Список объединённых полигонов
        """
        from shapely.ops import unary_union
        
        if not polygons:
            return []
        
        # Фильтруем валидные полигоны
        valid_polygons = [p for p in polygons if p.is_valid and not p.is_empty]
        
        if not valid_polygons:
            return []
        
        try:
            # Объединяем все полигоны
            merged = unary_union(valid_polygons)
            
            # Результат может быть Polygon или MultiPolygon
            if merged.geom_type == 'Polygon':
                return [merged]
            elif merged.geom_type == 'MultiPolygon':
                return list(merged.geoms)
            else:
                return valid_polygons
        except Exception as e:
            logger.warning(f"Ошибка объединения полигонов: {e}")
            return valid_polygons
