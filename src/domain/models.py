"""Доменные модели для представления данных."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime


@dataclass
class GeoTiffData:
    """Модель данных GeoTIFF файла."""
    
    width: int
    height: int
    num_bands: int
    data: np.ndarray
    transform: Tuple[float, ...]
    crs: Any
    projection_wkt: str
    proj4_projection: str
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    
    @property
    def pixel_width(self) -> float:
        """Ширина пикселя в единицах CRS."""
        return abs(self.transform[1])
    
    @property
    def pixel_height(self) -> float:
        """Высота пикселя в единицах CRS."""
        return abs(self.transform[5])
    
    @property
    def pixel_area_sqm(self) -> float:
        """Площадь одного пикселя в квадратных метрах."""
        return self.pixel_width * self.pixel_height


@dataclass
class DetectedObject:
    """Обнаруженный объект нейросетью."""
    
    geometry: Polygon
    area_sqm: float
    centroid: Tuple[float, float]
    instance_id: str
    chunk_id: str = ""
    mask_id: int = 0
    mask_area_pixels: int = 0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации."""
        return {
            'geometry': self.geometry,
            'area_sqm': self.area_sqm,
            'centroid': self.centroid,
            'instance_id': self.instance_id,
            'chunk_id': self.chunk_id,
            'mask_id': self.mask_id,
            'mask_area_pixels': self.mask_area_pixels,
            'confidence': self.confidence,
        }


@dataclass
class CadastralParcel:
    """Кадастровый участок."""
    
    geometry: Polygon
    cadastral_number: str
    area_sqm: float
    centroid: Tuple[float, float]
    attributes: Dict[str, Any] = field(default_factory=dict)
    object_id: Optional[Any] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    exterior_coords: Optional[List[Tuple[float, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации."""
        result = {
            'geometry': self.geometry,
            'cadastral_number': self.cadastral_number,
            'area_sqm': self.area_sqm,
            'centroid': self.centroid,
            'attributes': self.attributes,
        }
        if self.object_id is not None:
            result['object_id'] = self.object_id
        if self.bounds is not None:
            result['bounds'] = self.bounds
        if self.exterior_coords is not None:
            result['exterior_coords'] = self.exterior_coords
        return result


@dataclass
class Violation:
    """Нарушение (обнаруженный объект вне кадастра)."""
    
    geometry: Union[Polygon, MultiPolygon]
    violation_area: float
    detected_object: DetectedObject
    parcel: Optional[CadastralParcel] = None
    binding_type: str = "none"  # "intersection", "boundary", "nearest", "none"
    binding_distance: float = 0.0
    intersection_ratio: float = 0.0
    centroid: Optional[Tuple[float, float]] = None
    original_object_area: float = 0.0  # Площадь исходного обнаруженного объекта
    
    def __post_init__(self):
        """Вычисляет центроид и площадь исходного объекта если не заданы."""
        if self.centroid is None:
            self.centroid = self.geometry.centroid.coords[0]
        if self.original_object_area == 0.0:
            self.original_object_area = self.detected_object.area_sqm
    
    @property
    def cadastral_number(self) -> str:
        """Возвращает кадастровый номер привязанного участка."""
        return self.parcel.cadastral_number if self.parcel else "Не привязано"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации."""
        return {
            'geometry': self.geometry,
            'violation_area': self.violation_area,
            'detected_object': self.detected_object.to_dict(),
            'parcel': self.parcel.to_dict() if self.parcel else None,
            'binding_type': self.binding_type,
            'binding_distance': self.binding_distance,
            'intersection_ratio': self.intersection_ratio,
            'centroid': self.centroid,
            'cadastral_number': self.cadastral_number,
        }


@dataclass
class AnalysisResult:
    """Результат комплексного анализа."""
    
    detected_objects: List[DetectedObject]
    cadastral_parcels: List[CadastralParcel]
    violations: List[Violation]
    geotiff_data: Optional[GeoTiffData] = None
    
    # Статистика
    analysis_date: datetime = field(default_factory=datetime.now)
    
    @property
    def total_detected_area(self) -> float:
        """Общая площадь обнаруженных объектов."""
        return sum(obj.area_sqm for obj in self.detected_objects)
    
    @property
    def total_cadastral_area(self) -> float:
        """Общая площадь кадастровых участков."""
        return sum(obj.area_sqm for obj in self.cadastral_parcels)
    
    @property
    def total_violation_area(self) -> float:
        """Общая площадь нарушений."""
        return sum(v.violation_area for v in self.violations)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Возвращает статистику анализа."""
        return {
            'detected_objects_count': len(self.detected_objects),
            'cadastral_objects_count': len(self.cadastral_parcels),
            'violations_count': len(self.violations),
            'total_detected_area': self.total_detected_area,
            'total_cadastral_area': self.total_cadastral_area,
            'total_violation_area': self.total_violation_area,
            'analysis_date': self.analysis_date.isoformat(),
        }
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Преобразует результат в формат старого API для совместимости.
        
        Returns:
            Словарь в формате старого comprehensive_analysis.py
        """
        return {
            'detected_objects': [obj.to_dict() for obj in self.detected_objects],
            'cadastral_parcels': [p.to_dict() for p in self.cadastral_parcels],
            'violations': [v.to_dict() for v in self.violations],
            'statistics': self.statistics,
        }
