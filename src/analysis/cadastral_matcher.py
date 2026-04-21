"""Сопоставление нарушений с кадастровыми участками."""

import logging
from typing import List, Optional, Tuple
from shapely.geometry import Point, Polygon

from ..domain.models import CadastralParcel, DetectedObject

logger = logging.getLogger(__name__)


class CadastralMatcher:
    """Класс для привязки нарушений к кадастровым участкам."""
    
    def __init__(
        self,
        min_intersection_ratio: float = 0.1,
        boundary_buffer_m: float = 3.0,
        max_nearest_distance_m: float = 25.0
    ):
        """
        Инициализация matcher.
        
        Args:
            min_intersection_ratio: Минимальная доля пересечения
            boundary_buffer_m: Буфер для касания границы (м)
            max_nearest_distance_m: Максимальное расстояние для привязки
        """
        self.min_intersection_ratio = min_intersection_ratio
        self.boundary_buffer_m = boundary_buffer_m
        self.max_nearest_distance_m = max_nearest_distance_m
    
    def match(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel]
    ) -> Tuple[Optional[CadastralParcel], str, float, float]:
        """
        Находит кадастровый участок для нарушения.
        
        Args:
            violation_geometry: Геометрия нарушения
            cadastral_parcels: Список кадастровых участков
            
        Returns:
            Кортеж (parcel, binding_type, distance, intersection_ratio)
        """
        if not cadastral_parcels:
            return None, "none", 0.0, 0.0
        
        # Вариант A: в приоритете расстояние до ближайшего участка.
        # 1) nearest (в пределах max_nearest_distance_m)
        parcel, distance = self._match_by_nearest(violation_geometry, cadastral_parcels)
        if parcel:
            return parcel, "nearest", distance, 0.0

        # 2) boundary (в пределах boundary_buffer_m)
        parcel, distance = self._match_by_boundary(violation_geometry, cadastral_parcels)
        if parcel:
            return parcel, "boundary", distance, 0.0

        # 3) intersection (fallback)
        parcel, ratio = self._match_by_intersection(violation_geometry, cadastral_parcels)
        if parcel:
            return parcel, "intersection", 0.0, ratio

        return None, "none", 0.0, 0.0
    
    def _match_by_intersection(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel]
    ) -> Tuple[Optional[CadastralParcel], float]:
        """Привязка по пересечению."""
        best_parcel = None
        best_ratio = 0.0
        
        for parcel in cadastral_parcels:
            if not violation_geometry.intersects(parcel.geometry):
                continue
            
            try:
                intersection = violation_geometry.intersection(parcel.geometry)
                ratio = intersection.area / violation_geometry.area
                
                if ratio >= self.min_intersection_ratio and ratio > best_ratio:
                    best_ratio = ratio
                    best_parcel = parcel
            except Exception:
                continue
        
        return best_parcel, best_ratio
    
    def _match_by_boundary(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel]
    ) -> Tuple[Optional[CadastralParcel], float]:
        """Привязка по касанию границы: ближайший по расстоянию, при ничьей — по центроиду."""
        rp = violation_geometry.representative_point()
        candidates: List[Tuple[float, float, CadastralParcel]] = []
        for parcel in cadastral_parcels:
            try:
                distance = violation_geometry.distance(parcel.geometry)
                if distance <= self.boundary_buffer_m:
                    tie = rp.distance(Point(parcel.centroid))
                    candidates.append((distance, tie, parcel))
            except Exception:
                continue
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2], candidates[0][0]
    
    def _match_by_nearest(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel]
    ) -> Tuple[Optional[CadastralParcel], float]:
        """Привязка к ближайшему участку в пределах max_dist; при равных d — по близости центроидов."""
        rp = violation_geometry.representative_point()
        candidates: List[Tuple[float, float, CadastralParcel]] = []
        for parcel in cadastral_parcels:
            try:
                distance = violation_geometry.distance(parcel.geometry)
                if distance <= self.max_nearest_distance_m:
                    tie = rp.distance(Point(parcel.centroid))
                    candidates.append((distance, tie, parcel))
            except Exception:
                continue
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2], candidates[0][0]

    def match_nearest_unlimited(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel],
    ) -> Tuple[Optional[CadastralParcel], float]:
        """
        Ближайший ЗУ по расстоянию до полигона без порога max_nearest_distance_m
        (fallback, если match() вернул None).
        """
        if not cadastral_parcels:
            return None, 0.0
        rp = violation_geometry.representative_point()
        candidates: List[Tuple[float, float, CadastralParcel]] = []
        for parcel in cadastral_parcels:
            try:
                distance = violation_geometry.distance(parcel.geometry)
                tie = rp.distance(Point(parcel.centroid))
                candidates.append((distance, tie, parcel))
            except Exception:
                continue
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2], candidates[0][0]
