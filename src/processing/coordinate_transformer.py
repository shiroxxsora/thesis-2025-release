"""Трансформация координат между системами."""

import logging
from typing import Optional, Tuple
from osgeo import osr, ogr
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
import json
from shapely.affinity import translate as shapely_translate

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """Класс для трансформации координат между CRS."""
    
    def __init__(
        self,
        source_crs,
        target_crs,
        check_zone_offset: bool = True
    ):
        """
        Инициализация трансформера.
        
        Args:
            source_crs: Исходная система координат (OSR SpatialReference)
            target_crs: Целевая система координат (может быть rasterio.crs.CRS или OSR)
            check_zone_offset: Проверять смещение зон (для МСК)
        """
        self.source_crs = source_crs
        self.target_crs = self._convert_to_osr(target_crs)
        self.transformation = None
        self.needs_transformation = True
        self.zone_offset = 0.0
        # Ручной сдвиг по X (используется для МСК tmerc с разными x_0, когда OSR трансформацию
        # сознательно отключаем, но по bbox видно, что нужен именно зонный сдвиг на ~4e6).
        self.manual_shift_x = 0.0
        self._same_projection_zone_mismatch = False
        
        self._setup_transformation(check_zone_offset)
    
    def _convert_to_osr(self, crs):
        """
        Конвертирует CRS в osr.SpatialReference.
        
        Args:
            crs: CRS (может быть rasterio.crs.CRS или osr.SpatialReference)
            
        Returns:
            osr.SpatialReference
        """
        # Если уже osr.SpatialReference - возвращаем как есть
        if isinstance(crs, osr.SpatialReference):
            return crs
        
        # Если это rasterio.crs.CRS - конвертируем
        try:
            if hasattr(crs, 'to_wkt'):
                wkt = crs.to_wkt()
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                return srs
            elif hasattr(crs, 'wkt'):
                srs = osr.SpatialReference()
                srs.ImportFromWkt(crs.wkt)
                return srs
        except Exception as e:
            logger.warning(f"Не удалось конвертировать CRS в OSR: {e}")
        
        # Если ничего не помогло, возвращаем как есть
        return crs
    
    def _setup_transformation(self, check_zone_offset: bool):
        """Настраивает трансформацию координат."""
        # Настраиваем порядок осей
        if hasattr(self.source_crs, 'SetAxisMappingStrategy'):
            self.source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        if hasattr(self.target_crs, 'SetAxisMappingStrategy'):
            self.target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # Проверяем, нужна ли трансформация
        if self.source_crs.IsSame(self.target_crs):
            logger.info("CRS идентичны, трансформация не требуется")
            self.needs_transformation = False
            return
        
        # Проверяем смещение зон для МСК
        if check_zone_offset:
            self._check_zone_offset()
        
        # Создаём трансформацию только если она нужна
        if not self.needs_transformation:
            logger.info("Трансформация не требуется")
            return
        
        try:
            self.transformation = osr.CoordinateTransformation(
                self.source_crs,
                self.target_crs
            )
            logger.info("Создана трансформация координат")
        except Exception as e:
            logger.error(f"Ошибка создания трансформации: {e}")
            raise
    
    def _check_zone_offset(self):
        """Проверяет смещение между зонами МСК."""
        try:
            source_proj4 = self.source_crs.ExportToProj4()
            target_proj4 = self.target_crs.ExportToProj4()
            
            source_parts = dict(p.split("=", 1) for p in source_proj4.split() if "=" in p)
            target_parts = dict(p.split("=", 1) for p in target_proj4.split() if "=" in p)
            
            source_x0 = float(source_parts.get("+x_0", "0"))
            target_x0 = float(target_parts.get("+x_0", "0"))
            
            self.zone_offset = source_x0 - target_x0
            
            if abs(self.zone_offset) > 1000000:  # Больше 1 млн метров
                logger.info(
                    f"Обнаружено смещение зон: {self.zone_offset / 1000000:.1f} млн м "
                    f"(x_0: {source_x0} -> {target_x0})"
                )
                
                # Проверяем - это одна и та же проекция МСК с разными x_0?
                # Сравниваем ключевые параметры (proj, lat_0, lon_0)
                # ellps не проверяем, так как для МСК он по умолчанию krass
                key_params = ['+proj', '+lat_0', '+lon_0']
                
                logger.info("Сравнение параметров CRS для проверки совпадения проекции:")
                for param in key_params:
                    src_val = source_parts.get(param)
                    tgt_val = target_parts.get(param)
                    matches = src_val == tgt_val
                    logger.info(f"  {param}: source={src_val}, target={tgt_val}, match={matches}")
                
                same_projection = all(
                    source_parts.get(param) == target_parts.get(param)
                    for param in key_params
                )
                
                if same_projection:
                    logger.warning(
                        "Обнаружена одна и та же проекция МСК с разными x_0! "
                        "Вероятно координаты в исходных данных уже приведены к целевой системе. "
                        "Трансформация ОТКЛЮЧЕНА."
                    )
                    self.needs_transformation = False
                    self.transformation = None
                    self._same_projection_zone_mismatch = True
        except Exception as e:
            logger.debug(f"Не удалось проверить смещение зон: {e}")

    def enable_manual_zone_shift(self) -> None:
        """
        Включает ручной сдвиг по X на величину, компенсирующую разницу x_0.

        Пример: source x_0=4_250_000, target x_0=250_000 => zone_offset=4_000_000,
        чтобы привести источник к цели, нужно manual_shift_x = -zone_offset.
        """
        # Не совмещаем с полноценной OSR-трансформацией: ручной сдвиг — альтернатива ей.
        if self.needs_transformation and self.transformation is not None:
            logger.warning("Ручной зонный сдвиг не включён: активна OSR-трансформация.")
            self.manual_shift_x = 0.0
            return
        if abs(self.zone_offset) > 1_000_000:
            self.manual_shift_x = -float(self.zone_offset)
            logger.info("Включён ручной зонный сдвиг по X: %s м", self.manual_shift_x)
        else:
            self.manual_shift_x = 0.0
    
    def transform_geometry(self, geometry) -> Optional[Polygon]:
        """
        Трансформирует геометрию из исходной CRS в целевую.
        
        Args:
            geometry: OGR Geometry объект
            
        Returns:
            Shapely Polygon в целевой CRS или None
        """
        if geometry is None:
            return None
        
        # Клонируем геометрию чтобы не изменять оригинал
        geom_clone = geometry.Clone()
        
        # Трансформируем если нужно
        if self.needs_transformation and self.transformation:
            try:
                geom_clone.Transform(self.transformation)
            except Exception as e:
                logger.warning(f"Ошибка трансформации геометрии: {e}")
                return None
        
        # Конвертируем в Shapely
        try:
            from shapely.geometry import shape
            
            if geom_clone.GetGeometryType() == ogr.wkbPolygon:
                shp = shape(json.loads(geom_clone.ExportToJson()))
                if shp.is_valid and shp.area > 0:
                    if self.manual_shift_x:
                        shp = shapely_translate(shp, xoff=self.manual_shift_x, yoff=0.0)
                    return shp
            
            elif geom_clone.GetGeometryType() == ogr.wkbMultiPolygon:
                # Для MultiPolygon возвращаем самый большой полигон
                multipolygon = shape(json.loads(geom_clone.ExportToJson()))
                if isinstance(multipolygon, MultiPolygon) and len(multipolygon.geoms) > 0:
                    shp = max(multipolygon.geoms, key=lambda p: p.area)
                    if self.manual_shift_x:
                        shp = shapely_translate(shp, xoff=self.manual_shift_x, yoff=0.0)
                    return shp
        
        except Exception as e:
            logger.warning(f"Ошибка конвертации в Shapely: {e}")
        
        return None
    
    def transform_point(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
        """
        Трансформирует точку.
        
        Args:
            x: Координата X
            y: Координата Y
            z: Координата Z (высота)
            
        Returns:
            Кортеж (x, y) в целевой CRS
        """
        if not self.needs_transformation:
            if self.manual_shift_x:
                return (x + self.manual_shift_x, y)
            return (x, y)
        
        if self.transformation is None:
            if self.manual_shift_x:
                return (x + self.manual_shift_x, y)
            return (x, y)
        
        try:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(x, y, z)
            point.Transform(self.transformation)
            px, py = (point.GetX(), point.GetY())
            if self.manual_shift_x:
                px += self.manual_shift_x
            return (px, py)
        except Exception as e:
            logger.warning(f"Ошибка трансформации точки: {e}")
            return (x, y)
