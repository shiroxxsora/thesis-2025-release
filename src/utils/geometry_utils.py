"""Утилиты для работы с геометрией."""

import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from shapely.validation import make_valid

from ..domain.types import Coordinate, PolygonCoords, GeoTransform


def pixel_to_geo_coords(
    pixel_coords: np.ndarray,
    transform: GeoTransform
) -> PolygonCoords:
    """
    Преобразует пиксельные координаты в географические.
    
    Args:
        pixel_coords: Массив пиксельных координат (N, 2)
        transform: GeoTransform из rasterio/GDAL
        
    Returns:
        Список географических координат
    """
    geo_coords = []
    for x, y in pixel_coords:
        geo_x = transform[0] + x * transform[1] + y * transform[2]
        geo_y = transform[3] + x * transform[4] + y * transform[5]
        geo_coords.append((geo_x, geo_y))
    return geo_coords


def geo_to_pixel_coords(
    geo_coords: PolygonCoords,
    transform: GeoTransform
) -> np.ndarray:
    """
    Преобразует географические координаты в пиксельные.
    
    Args:
        geo_coords: Список географических координат
        transform: GeoTransform из rasterio/GDAL
        
    Returns:
        Массив пиксельных координат (N, 2)
    """
    # Обратная трансформация
    # x_pixel = (geo_x - gt[0]) / gt[1]
    # y_pixel = (geo_y - gt[3]) / gt[5]
    
    pixel_coords = []
    for geo_x, geo_y in geo_coords:
        x_pixel = (geo_x - transform[0]) / transform[1] if transform[1] != 0 else 0
        y_pixel = (geo_y - transform[3]) / transform[5] if transform[5] != 0 else 0
        pixel_coords.append([x_pixel, y_pixel])
    
    return np.array(pixel_coords)


def simplify_polygon(
    polygon: Polygon,
    tolerance: float = 0.5,
    preserve_topology: bool = True
) -> Polygon:
    """
    Упрощает полигон, сохраняя его форму.
    
    Args:
        polygon: Исходный полигон
        tolerance: Толерантность упрощения в единицах CRS
        preserve_topology: Сохранять топологию
        
    Returns:
        Упрощённый полигон
    """
    if tolerance <= 0:
        return polygon
    
    try:
        simplified = polygon.simplify(tolerance, preserve_topology=preserve_topology)
        
        # Проверка валидности
        if not simplified.is_valid:
            simplified = make_valid(simplified)
        
        # Если упрощение слишком агрессивное, возвращаем оригинал
        if simplified.is_empty or simplified.area < polygon.area * 0.1:
            return polygon
        
        return simplified
    except Exception:
        return polygon


def validate_and_fix_polygon(polygon: Polygon) -> Polygon:
    """
    Проверяет и исправляет невалидный полигон.
    
    Args:
        polygon: Полигон для проверки
        
    Returns:
        Валидный полигон
    """
    if polygon.is_valid:
        return polygon
    
    try:
        fixed = make_valid(polygon)
        
        # make_valid может вернуть GeometryCollection
        if fixed.geom_type == 'Polygon':
            return fixed
        elif fixed.geom_type == 'MultiPolygon':
            # Берём самый большой полигон
            return max(fixed.geoms, key=lambda p: p.area)
        elif fixed.geom_type == 'GeometryCollection':
            # Ищем полигоны в коллекции
            polygons = [g for g in fixed.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
            if polygons:
                return polygons[0] if len(polygons) == 1 else max(polygons, key=lambda p: p.area)
        
        # Если ничего не помогло, возвращаем пустой полигон
        return Polygon()
    except Exception:
        return Polygon()


def calculate_overlap_ratio(poly1: Polygon, poly2: Polygon) -> float:
    """
    Вычисляет долю пересечения двух полигонов.
    
    Args:
        poly1: Первый полигон
        poly2: Второй полигон
        
    Returns:
        Доля пересечения (intersection area / union area)
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    if poly1.is_empty or poly2.is_empty:
        return 0.0
    
    try:
        intersection = poly1.intersection(poly2)
        union = poly1.union(poly2)
        
        if union.area == 0:
            return 0.0
        
        return intersection.area / union.area
    except Exception:
        return 0.0
