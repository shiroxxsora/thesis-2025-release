"""Анализатор нарушений."""

import logging
from collections import defaultdict
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, unary_union

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
            max_nearest_distance_m=config.binding_max_nearest_distance_m,
        )

    def _bind_parcel_with_relaxed_fallback(
        self,
        geom: Polygon,
        cadastral_parcels: List[CadastralParcel],
    ) -> Tuple[Optional[CadastralParcel], str, float, float]:
        """Сначала обычный match; если участок не найден — ближайший без лимита расстояния."""
        parcel, binding_type, distance, ratio = self.matcher.match(
            geom, cadastral_parcels
        )
        if parcel is not None:
            return parcel, binding_type, distance, ratio
        parcel, distance = self.matcher.match_nearest_unlimited(geom, cadastral_parcels)
        if parcel is not None:
            return parcel, "nearest_relaxed", distance, 0.0
        return None, "none", 0.0, 0.0

    @staticmethod
    def _merge_group_key(parcel: CadastralParcel) -> Tuple[Any, ...]:
        """Ключ для union по ЗУ: различаем участки с одним КН при наличии object_id."""
        if parcel.object_id is not None:
            return ("id", parcel.object_id, parcel.cadastral_number)
        return ("cn", parcel.cadastral_number)
    
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
            Список нарушений. Если ``merge_violations_per_parcel`` — по одному на кадастровый
            номер с unary_union фрагментов; иначе отдельная запись на каждый полигональный
            фрагмент (в т.ч. части MultiPolygon после difference).
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
        if self.config.merge_violations_per_parcel:
            merged = self._merge_violations_one_per_cadastral_parcel(violations)
            logger.info(
                "Нарушений после union по кадастру: %s (фрагментов до объединения: %s)",
                len(merged),
                len(violations),
            )
            return merged
        logger.info(
            "Нарушений (без union по участку): %s записей",
            len(violations),
        )
        return violations
    
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

        boundary_splitter: Optional[BaseGeometry] = None
        if len(cadastral_parcels) >= 2:
            try:
                boundary_splitter = unary_union(
                    [p.geometry.boundary for p in cadastral_parcels]
                )
            except Exception as e:
                logger.debug("Границы ЗУ для split: %s", e)

        violations = []
        
        for obj in detected_objects:
            # Вычитаем кадастр из обнаруженного объекта
            try:
                violation_geom = obj.geometry.difference(cadastral_union)
                
                if violation_geom.is_empty:
                    continue

                parts = self._polygon_parts_from_difference(violation_geom)
                if not parts:
                    continue

                for part in parts:
                    for sub in self._split_violation_by_cadastral_boundaries(
                        part, boundary_splitter
                    ):
                        grid_assigned = self._assign_violation_centroid_grid(
                            sub, cadastral_parcels
                        )
                        if grid_assigned:
                            emitted_geoms: List[Polygon] = []
                            for gpoly, parcel in grid_assigned:
                                va = float(gpoly.area)
                                if va < self.config.min_violation_area:
                                    continue
                                if parcel is None:
                                    continue
                                try:
                                    d_bind = float(gpoly.distance(parcel.geometry))
                                except Exception:
                                    d_bind = 0.0
                                emitted_geoms.append(gpoly)
                                violations.append(
                                    Violation(
                                        geometry=gpoly,
                                        violation_area=va,
                                        detected_object=obj,
                                        parcel=parcel,
                                        binding_type="centroid_grid",
                                        binding_distance=d_bind,
                                        intersection_ratio=0.0,
                                        original_object_area=obj.area_sqm,
                                    )
                                )
                            if emitted_geoms:
                                try:
                                    cover = unary_union(emitted_geoms)
                                    rem = sub.difference(cover).buffer(0)
                                    for rp in self._polygon_parts_from_difference(rem):
                                        if rp.area < self.config.min_violation_area:
                                            continue
                                        (
                                            parcel_r,
                                            bt_r,
                                            dist_r,
                                            ratio_r,
                                        ) = self._bind_parcel_with_relaxed_fallback(
                                            rp, cadastral_parcels
                                        )
                                        if parcel_r is None:
                                            continue
                                        violations.append(
                                            Violation(
                                                geometry=rp,
                                                violation_area=float(rp.area),
                                                detected_object=obj,
                                                parcel=parcel_r,
                                                binding_type=bt_r,
                                                binding_distance=dist_r,
                                                intersection_ratio=ratio_r,
                                                original_object_area=obj.area_sqm,
                                            )
                                        )
                                except Exception as e:
                                    logger.debug("Остаток нарушения после сетки: %s", e)
                            continue

                        violation_area = sub.area
                        if violation_area < self.config.min_violation_area:
                            continue

                        parcel, binding_type, distance, ratio = (
                            self._bind_parcel_with_relaxed_fallback(
                                sub, cadastral_parcels
                            )
                        )
                        if parcel is None:
                            continue

                        violation = Violation(
                            geometry=sub,
                            violation_area=violation_area,
                            detected_object=obj,
                            parcel=parcel,
                            binding_type=binding_type,
                            binding_distance=distance,
                            intersection_ratio=ratio,
                            original_object_area=obj.area_sqm,
                        )

                        violations.append(violation)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки объекта {obj.instance_id}: {e}")
                continue
        
        return violations

    @staticmethod
    def _polygon_parts_from_difference(geom: BaseGeometry) -> List[Polygon]:
        """Из результата difference извлекает отдельные полигоны (MultiPolygon, вложенные коллекции)."""
        if geom.is_empty:
            return []
        gt = geom.geom_type
        if gt == "Polygon":
            return [geom]
        if gt == "MultiPolygon":
            return [g for g in geom.geoms if not g.is_empty and g.geom_type == "Polygon"]
        if gt == "GeometryCollection":
            out: List[Polygon] = []
            for g in geom.geoms:
                out.extend(ViolationAnalyzer._polygon_parts_from_difference(g))
            return out
        logger.debug(
            "difference/split: неожиданный тип геометрии %s, полигоны не извлечены", gt
        )
        return []

    def _grid_cells_union_polygon(
        self,
        xs_i: np.ndarray,
        ys_i: np.ndarray,
        minx: float,
        miny: float,
        step: float,
        sub: Polygon,
    ) -> BaseGeometry:
        """Объединение квадратных ячеек сетки в полигон и обрезка по исходному нарушению."""
        pairs = np.unique(np.column_stack([ys_i, xs_i]), axis=0)
        boxes = [
            box(
                float(minx + float(c) * step),
                float(miny + float(r) * step),
                float(minx + (float(c) + 1.0) * step),
                float(miny + (float(r) + 1.0) * step),
            )
            for r, c in pairs
        ]
        u = unary_union(boxes)
        return u.buffer(0).intersection(sub.buffer(0))

    def _assign_violation_centroid_grid(
        self,
        sub: Polygon,
        parcels: List[CadastralParcel],
    ) -> Optional[List[Tuple[Polygon, CadastralParcel]]]:
        """
        Сетка по bounds нарушения: каждая ячейка — класс «ближайший» ЗУ (как шаг weighted k-means
        с фиксированными центроидами). Вес prior по площади: (area/median)^exponent; опционально
        RBF-ядро exp(-d²/(2σ²)) на расстояние до центроида — score ∝ prior · K(d).
        Связные области одного класса → полигоны. None при сбое → CadastralMatcher.
        """
        if sub.is_empty or not parcels:
            return None
        if len(parcels) == 1:
            return [(sub, parcels[0])]

        minx, miny, maxx, maxy = sub.bounds
        w, h = maxx - minx, maxy - miny
        if w <= 0 or h <= 0:
            return None

        step = float(self.config.violation_binding_grid_step_m)
        max_c = int(self.config.violation_binding_grid_max_cells)
        nx = max(1, int(np.ceil(w / step)))
        ny = max(1, int(np.ceil(h / step)))
        if max(nx, ny) > max_c:
            step = max(w / max_c, h / max_c)
            nx = max(1, int(np.ceil(w / step)))
            ny = max(1, int(np.ceil(h / step)))

        gx = np.linspace(minx + step / 2.0, maxx - step / 2.0, nx)
        gy = np.linspace(miny + step / 2.0, maxy - step / 2.0, ny)
        MGx, MGy = np.meshgrid(gx, gy)

        try:
            import shapely.vectorized as sv

            inside = sv.contains(sub, MGx, MGy)
        except Exception:
            try:
                from matplotlib.path import Path as MplPath

                mpath = MplPath(np.asarray(sub.exterior.coords))
                flat = np.column_stack([MGx.ravel(), MGy.ravel()])
                inside = mpath.contains_points(flat).reshape(ny, nx)
            except Exception as e:
                logger.debug("Сетка привязки: нет shapely.vectorized/matplotlib: %s", e)
                return None

        centroids = np.array(
            [
                [float(p.geometry.centroid.x), float(p.geometry.centroid.y)]
                for p in parcels
            ],
            dtype=np.float64,
        )
        labels = np.full((ny, nx), -1, dtype=np.int32)
        flat_in = inside.ravel()
        idx = np.flatnonzero(flat_in)
        if idx.size == 0:
            return None

        pts = np.column_stack([MGx.ravel()[idx], MGy.ravel()[idx]])
        d2 = np.sum((pts[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

        areas_sqm = np.array(
            [max(float(p.area_sqm), 1.0) for p in parcels], dtype=np.float64
        )
        a_ref = float(np.median(areas_sqm))
        exp_a = float(self.config.violation_binding_area_exponent)
        prior = np.power(
            np.maximum(areas_sqm / max(a_ref, 1e-9), 1e-18), exp_a
        )

        sigma = self.config.violation_binding_rbf_sigma_m
        if sigma is not None and float(sigma) > 0:
            s2 = float(sigma) ** 2
            score = prior[None, :] * np.exp(-0.5 * d2 / s2)
            lid = np.argmax(score, axis=1).astype(np.int32)
        else:
            d2_eff = d2 / np.maximum(prior, 1e-18)
            lid = np.argmin(d2_eff, axis=1).astype(np.int32)

        labels.ravel()[idx] = lid

        out: List[Tuple[Polygon, CadastralParcel]] = []
        for pid, parcel in enumerate(parcels):
            m = (labels == pid).astype(np.uint8)
            if not m.any():
                continue
            num, cc = cv2.connectedComponents(m)
            for lab in range(1, num):
                comp = cc == lab
                ys_i, xs_i = np.where(comp)
                if ys_i.size == 0:
                    continue
                area_est = float(ys_i.size) * step * step
                if area_est < self.config.min_violation_area * 0.15:
                    continue
                try:
                    gpoly = self._grid_cells_union_polygon(
                        xs_i, ys_i, minx, miny, step, sub
                    )
                except Exception:
                    continue
                if gpoly.is_empty:
                    continue
                if gpoly.geom_type == "Polygon":
                    polys = [gpoly]
                elif gpoly.geom_type == "MultiPolygon":
                    polys = list(gpoly.geoms)
                else:
                    continue
                for pg in polys:
                    if pg.is_empty:
                        continue
                    a = float(pg.area)
                    if a < self.config.min_violation_area:
                        continue
                    out.append((pg, parcel))

        return out if out else None

    def _split_violation_by_cadastral_boundaries(
        self,
        part: Polygon,
        boundary_splitter: Optional[BaseGeometry],
    ) -> List[Polygon]:
        """
        Режет полигон нарушения по линиям границ ЗУ (shapely.ops.split).

        Длинная полоса, пересекающая несколько участков, иначе получает одну привязку
        (ближайший участок) и при merge — один КН; после разреза куски привязываются локально.

        Порог min_violation_area применяется только в _analyze_vector, не здесь — иначе
        теряется площадь при разрезе на крупный и мелкие осколки.
        """
        if boundary_splitter is None or boundary_splitter.is_empty:
            return [part]
        try:
            spl = split(part, boundary_splitter)
            out = self._polygon_parts_from_difference(spl)
            out = [p for p in out if not p.is_empty]
            return out if out else [part]
        except Exception as e:
            logger.debug("Разрез нарушения по границам ЗУ: %s", e)
            return [part]

    @staticmethod
    def _sum_original_object_area_unique(violations: List[Violation]) -> float:
        """Площадь исходных детекций без повторного суммирования одного объекта."""
        by_det: Dict[Tuple[str, str, int], float] = {}
        for v in violations:
            o = v.detected_object
            key = (o.instance_id, o.chunk_id, int(o.mask_id))
            if key not in by_det:
                by_det[key] = float(o.area_sqm)
        return float(sum(by_det.values()))

    def _merge_violations_one_per_cadastral_parcel(
        self, violations: List[Violation]
    ) -> List[Violation]:
        """
        Один кадастровый участок — одно нарушение: геометрия = unary_union фрагментов,
        площадь = площадь объединения (не сумма частей).
        Без привязки к участку записи не объединяются.
        """
        bound: Dict[Any, List[Violation]] = defaultdict(list)
        unbound: List[Violation] = []
        for v in violations:
            if v.parcel is None:
                unbound.append(v)
                continue
            bound[self._merge_group_key(v.parcel)].append(v)
        
        out: List[Violation] = []
        for _, vs in bound.items():
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
            sum_orig = self._sum_original_object_area_unique(vs)
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
