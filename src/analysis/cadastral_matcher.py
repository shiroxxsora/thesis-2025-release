"""Сопоставление нарушений с кадастровыми участками."""

import logging
from typing import List, Optional, Tuple
from shapely.geometry import Polygon
from shapely.strtree import STRtree

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
        
        # Метод 1: Пересечение
        parcel, ratio = self._match_by_intersection(
            violation_geometry,
            cadastral_parcels
        )
        if parcel:
            return parcel, "intersection", 0.0, ratio
        
        # Метод 2: Касание границы
        parcel, distance = self._match_by_boundary(
            violation_geometry,
            cadastral_parcels
        )
        if parcel:
            return parcel, "boundary", distance, 0.0
        
        # Метод 3: Ближайший участок
        parcel, distance = self._match_by_nearest(
            violation_geometry,
            cadastral_parcels
        )
        if parcel:
            return parcel, "nearest", distance, 0.0
        
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
        """Привязка по касанию границы."""
        for parcel in cadastral_parcels:
            try:
                distance = violation_geometry.distance(parcel.geometry)
                
                if distance <= self.boundary_buffer_m:
                    return parcel, distance
            except Exception:
                continue
        
        return None, 0.0
    
    def _match_by_nearest(
        self,
        violation_geometry: Polygon,
        cadastral_parcels: List[CadastralParcel]
    ) -> Tuple[Optional[CadastralParcel], float]:
        """Привязка к ближайшему участку."""
        nearest_parcel = None
        min_distance = float('inf')
        
        for parcel in cadastral_parcels:
            try:
                distance = violation_geometry.distance(parcel.geometry)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_parcel = parcel
            except Exception:
                continue
        
        if nearest_parcel and min_distance <= self.max_nearest_distance_m:
            return nearest_parcel, min_distance
        
        return None, 0.0
