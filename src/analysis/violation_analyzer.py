"""Анализатор нарушений."""

import logging
from collections import defaultdict
import numpy as np
import cv2
from typing import List, Optional, Tuple
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData
from ..analysis.cadastral_matcher import CadastralMatcher
from ..config.settings import AnalysisConfig

logger = logging.getLogger(__name__)


class ViolationAnalyzer:
    """Класс для анализа нарушений."""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация анализатора.
        
        Args:
            config: Конфигурация анализа
        """
        self.config = config
        self.matcher = CadastralMatcher(
            min_intersection_ratio=config.binding_min_intersection_ratio,
            boundary_buffer_m=config.binding_boundary_buffer_m,
            max_nearest_distance_m=config.binding_max_nearest_distance_m
        )
    
    def analyze(
        self,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel],
        geotiff_data: Optional[GeoTiffData] = None
    ) -> List[Violation]:
        """
        Анализирует нарушения.
        
        Args:
            detected_objects: Обнаруженные объекты
            cadastral_parcels: Кадастровые участки
            geotiff_data: Данные GeoTIFF (для mask-based анализа)
            
        Returns:
            Список нарушений: по одному на кадастровый участок (геометрия — unary_union
            фрагментов вне кадастра), без суммирования пересекающихся частей.
        """
        logger.info("Начинаю анализ нарушений...")
        
        if not detected_objects:
            logger.warning("Нет обнаруженных объектов")
            return []
        
        if not cadastral_parcels:
            logger.warning("Нет кадастровых участков")
            # Все объекты считаем нарушениями без привязки
            return self._create_unmatched_violations(detected_objects)
        
        # Используем векторный метод (более надёжный)
        violations = self._analyze_vector(
            detected_objects,
            cadastral_parcels
        )
        merged = self._merge_violations_one_per_cadastral_parcel(violations)
        logger.info(
            "Нарушений после union по кадастру: %s (фрагментов по объектам до объединения: %s)",
            len(merged),
            len(violations),
        )
        return merged
    
    def _analyze_with_mask(
        self,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel],
        geotiff_data: GeoTiffData
    ) -> List[Violation]:
        """Анализ с использованием растровой маски кадастра."""
        # Создаём маску кадастра
        cadastral_mask = self._create_cadastral_mask(
            cadastral_parcels,
            geotiff_data
        )
        
        violations = []
        
        for obj in detected_objects:
            # Находим область нарушения (вне кадастра)
            violation_geom = self._find_violation_geometry(
                obj.geometry,
                cadastral_mask,
                geotiff_data.transform
            )
            
            if not violation_geom or violation_geom.is_empty:
                continue
            
            violation_area = violation_geom.area
            
            if violation_area < self.config.min_violation_area:
                continue
            
            # Привязываем к участку
            parcel, binding_type, distance, ratio = self.matcher.match(
                violation_geom,
                cadastral_parcels
            )
            
            violation = Violation(
                geometry=violation_geom,
                violation_area=violation_area,
                detected_object=obj,
                parcel=parcel,
                binding_type=binding_type,
                binding_distance=distance,
                intersection_ratio=ratio,
                original_object_area=obj.area_sqm
            )
            
            violations.append(violation)
        
        return violations
    
    def _analyze_vector(
        self,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel]
    ) -> List[Violation]:
        """Анализ с использованием векторных операций Shapely."""
        # Объединяем все кадастровые участки
        try:
            cadastral_union = unary_union([p.geometry for p in cadastral_parcels])
        except Exception as e:
            logger.error(f"Ошибка объединения кадастра: {e}")
            return []
        
        violations = []
        
        for obj in detected_objects:
            # Вычитаем кадастр из обнаруженного объекта
            try:
                violation_geom = obj.geometry.difference(cadastral_union)
                
                if violation_geom.is_empty:
                    continue
                
                # MultiPolygon -> берём самый большой
                if violation_geom.geom_type == 'MultiPolygon':
                    violation_geom = max(violation_geom.geoms, key=lambda p: p.area)
                
                violation_area = violation_geom.area
                
                if violation_area < self.config.min_violation_area:
                    continue
                
                # Привязываем к участку
                parcel, binding_type, distance, ratio = self.matcher.match(
                    violation_geom,
                    cadastral_parcels
                )
                
                violation = Violation(
                    geometry=violation_geom,
                    violation_area=violation_area,
                    detected_object=obj,
                    parcel=parcel,
                    binding_type=binding_type,
                    binding_distance=distance,
                    intersection_ratio=ratio,
                    original_object_area=obj.area_sqm
                )
                
                violations.append(violation)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки объекта {obj.instance_id}: {e}")
                continue
        
        return violations
    
    def _merge_violations_one_per_cadastral_parcel(
        self, violations: List[Violation]
    ) -> List[Violation]:
        """
        Один кадастровый участок — одно нарушение: геометрия = unary_union фрагментов,
        площадь = площадь объединения (не сумма частей).
        Без привязки к участку записи не объединяются.
        """
        bound = defaultdict(list)
        unbound: List[Violation] = []
        for v in violations:
            if v.parcel is None:
                unbound.append(v)
                continue
            bound[v.parcel.cadastral_number].append(v)
        
        out: List[Violation] = []
        for _cn, vs in bound.items():
            if len(vs) == 1:
                out.append(vs[0])
                continue
            merged_geom = unary_union([v.geometry for v in vs])
            if merged_geom.is_empty:
                continue
            if merged_geom.geom_type == 'GeometryCollection':
                merged_geom = unary_union(
                    [
                        g
                        for g in merged_geom.geoms
                        if g.geom_type in ('Polygon', 'MultiPolygon') and not g.is_empty
                    ]
                )
            if merged_geom.geom_type not in ('Polygon', 'MultiPolygon'):
                continue
            area = merged_geom.area
            if area < self.config.min_violation_area:
                continue
            primary = max(vs, key=lambda x: x.violation_area)
            sum_orig = sum(v.original_object_area for v in vs)
            out.append(
                Violation(
                    geometry=merged_geom,
                    violation_area=area,
                    detected_object=primary.detected_object,
                    parcel=primary.parcel,
                    binding_type=primary.binding_type,
                    binding_distance=min(v.binding_distance for v in vs),
                    intersection_ratio=max(v.intersection_ratio for v in vs),
                    original_object_area=sum_orig,
                )
            )
        out.extend(unbound)
        return out
    
    def _create_cadastral_mask(
        self,
        cadastral_parcels: List[CadastralParcel],
        geotiff_data: GeoTiffData
    ) -> np.ndarray:
        """Создаёт растровую маску кадастра."""
        mask = np.zeros((geotiff_data.height, geotiff_data.width), dtype=np.uint8)
        
        from ..utils.geometry_utils import geo_to_pixel_coords
        
        for parcel in cadastral_parcels:
            try:
                # Конвертируем в пиксельные координаты
                exterior_coords = list(parcel.geometry.exterior.coords)
                pixel_coords = geo_to_pixel_coords(exterior_coords, geotiff_data.transform)
                
                # Рисуем на маске
                cv2.fillPoly(mask, [pixel_coords.astype(np.int32)], 1)
            except Exception as e:
                logger.warning(f"Ошибка создания маски для участка: {e}")
                continue
        
        return mask
    
    def _find_violation_geometry(
        self,
        obj_geometry: Polygon,
        cadastral_mask: np.ndarray,
        transform: tuple
    ) -> Optional[Polygon]:
        """
        Находит геометрию нарушения используя маску.
        
        Растеризует объект, вычитает маску кадастра, векторизует обратно.
        """
        from ..utils.geometry_utils import geo_to_pixel_coords, pixel_to_geo_coords
        from ..processing.geometry_processor import GeometryProcessor
        
        try:
            # Конвертируем в пиксельные координаты
            exterior_coords = list(obj_geometry.exterior.coords)
            pixel_coords = geo_to_pixel_coords(exterior_coords, transform)
            
            # Создаём маску объекта
            h, w = cadastral_mask.shape
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [pixel_coords.astype(np.int32)], 1)
            
            # Вычитаем кадастр (нарушение = объект НЕ на кадастре)
            violation_mask = cv2.bitwise_and(obj_mask, cv2.bitwise_not(cadastral_mask))
            
            # Если вся маска нулевая - нет нарушения
            if not violation_mask.any():
                return None
            
            # Векторизуем маску обратно в полигон
            geom_processor = GeometryProcessor()
            violation_polygons = geom_processor.extract_polygons_from_mask(
                violation_mask,
                min_area=self.config.min_violation_area / (abs(transform[0]) * abs(transform[4]))  # пиксели
            )
            
            if not violation_polygons:
                return None
            
            # Берём самый большой полигон (в пиксельных координатах)
            largest_pixel_poly = max(violation_polygons, key=lambda p: cv2.contourArea(p))
            
            # Конвертируем обратно в гео-координаты
            pixel_coords_list = largest_pixel_poly.reshape(-1, 2).tolist()
            geo_coords = pixel_to_geo_coords(pixel_coords_list, transform)
            
            return Polygon(geo_coords)
            
        except Exception as e:
            logger.debug(f"Ошибка в _find_violation_geometry: {e}, используем весь объект")
            return obj_geometry
    
    def _create_unmatched_violations(
        self,
        detected_objects: List[DetectedObject]
    ) -> List[Violation]:
        """Создаёт нарушения без привязки к кадастру."""
        return [
            Violation(
                geometry=obj.geometry,
                violation_area=obj.area_sqm,
                detected_object=obj,
                parcel=None,
                binding_type="none",
                original_object_area=obj.area_sqm
            )
            for obj in detected_objects
        ]
