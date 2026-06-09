"""Reader для чтения кадастровых данных из MIF/MID файлов."""

import logging
from typing import List, Optional, Dict, Any
from osgeo import ogr, osr
from shapely.geometry import box
from shapely.affinity import translate as shapely_translate

from .base import DataReader
from ...domain.models import CadastralParcel
from ...processing.coordinate_transformer import CoordinateTransformer

logger = logging.getLogger(__name__)


class CadastralReader(DataReader[List[CadastralParcel]]):
    """Класс для чтения кадастровых данных из MIF/MID файлов."""
    
    def __init__(self):
        """Инициализация reader."""
        self.transformer: Optional[CoordinateTransformer] = None
        self._crs_same_as_target = False
    
    def read(
        self,
        path: str,
        target_crs=None,
        target_bounds: Optional[tuple] = None,
        manual_offset_x_m: float = 0.0,
        manual_offset_y_m: float = 0.0,
    ) -> List[CadastralParcel]:
        """
        Читает кадастровые данные из MIF/MID файла.
        
        Args:
            path: Путь к MIF файлу
            target_crs: Целевая система координат (для трансформации)
            target_bounds: Границы области интереса (min_x, min_y, max_x, max_y)
            manual_offset_x_m: Ручной сдвиг кадастра по X (м)
            manual_offset_y_m: Ручной сдвиг кадастра по Y (м)
            
        Returns:
            Список кадастровых участков
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если не удалось прочитать файл
        """
        self._check_file_exists(path)
        
        logger.info("Чтение кадастровых данных...")
        
        try:
            # Открываем MIF/MID файл
            driver = ogr.GetDriverByName('MapInfo File')
            datasource = driver.Open(path, 0)
            
            if datasource is None:
                raise ValueError(f"Не удалось открыть MIF/MID файл: {path}")
            
            layer = datasource.GetLayer()
            layer_defn = layer.GetLayerDefn()
            
            # Получаем поля
            field_names = self._extract_field_names(layer_defn)
            logger.info(f"Найдены поля: {field_names}")
            
            # Настраиваем трансформацию координат
            if target_crs:
                source_crs = self._determine_source_crs(layer, target_crs)
                self.transformer = CoordinateTransformer(source_crs, target_crs)
                self._crs_same_as_target = not self.transformer.needs_transformation

                # Авто-фикс для МСК: если x_0 отличается на ~4e6 и по bbox видно, что кадастр уехал,
                # включаем ручной сдвиг по X, чтобы объекты попали в область растра.
                if target_bounds and self.transformer and abs(getattr(self.transformer, "zone_offset", 0.0)) > 1_000_000:
                    try:
                        raster_cx = (float(target_bounds[0]) + float(target_bounds[2])) / 2.0
                        layer.ResetReading()
                        sample_centers = []
                        for i, f in enumerate(layer):
                            if i >= 3:
                                break
                            g = f.GetGeometryRef()
                            if not g:
                                continue
                            env = g.GetEnvelope()  # (minx, maxx, miny, maxy)
                            sample_centers.append((env[0] + env[1]) / 2.0)
                        layer.ResetReading()
                        if sample_centers:
                            cad_cx = float(sum(sample_centers) / len(sample_centers))
                            diff = cad_cx - raster_cx
                            z = float(self.transformer.zone_offset)
                            # Если разница по X близка к зонному сдвигу — применяем его.
                            if abs(diff - z) < 500_000.0:
                                self.transformer.enable_manual_zone_shift()
                                self._crs_same_as_target = False
                    except Exception:
                        try:
                            layer.ResetReading()
                        except Exception:
                            pass
            
            # Создаём bbox для фильтрации
            target_bbox = None
            if target_bounds:
                target_bbox = box(*target_bounds)
                logger.info(
                    f"BBox растра: "
                    f"X=[{target_bounds[0]:.2f},{target_bounds[2]:.2f}] "
                    f"Y=[{target_bounds[1]:.2f},{target_bounds[3]:.2f}]"
                )
            
            # Читаем объекты
            cadastral_objects = self._read_features(
                layer,
                field_names,
                target_bbox,
                manual_offset_x_m=manual_offset_x_m,
                manual_offset_y_m=manual_offset_y_m,
            )
            
            logger.info(
                f"Прочитано {len(cadastral_objects)} кадастровых объектов "
                f"в области растра"
            )
            
            return cadastral_objects
            
        except Exception as e:
            logger.error(f"Ошибка чтения кадастровых данных: {e}")
            raise
    
    def _extract_field_names(self, layer_defn) -> List[str]:
        """Извлекает имена полей из определения слоя."""
        field_names = []
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            field_names.append(field_defn.GetName())
        return field_names
    
    def _determine_source_crs(self, layer, target_crs):
        """
        Определяет исходную CRS для кадастровых данных.
        
        Пытается использовать CRS из слоя, если не получается - использует fallback.
        """
        source_crs = layer.GetSpatialRef()
        
        # Проверяем CRS из слоя
        if source_crs is not None:
            if self._validate_layer_crs(source_crs, layer):
                logger.info("Использую CRS из слоя MIF/MID")
                return source_crs.Clone()
            else:
                logger.warning("CRS из слоя некорректна, использую fallback")
        
        # Fallback: жёстко заданная проекция для МСК-03
        return self._create_fallback_crs(target_crs)
    
    def _validate_layer_crs(self, crs, layer) -> bool:
        """
        Проверяет, что CRS из слоя корректна.
        
        Если CRS географическая (lat/lon), но координаты в метрах - это ошибка.
        """
        try:
            if not crs.IsGeographic():
                return True  # Проецированная CRS - OK
            
            # Проверяем первые несколько координат
            layer.ResetReading()
            for i, feature in enumerate(layer):
                if i >= 3:  # Проверяем первые 3 объекта
                    break
                
                geom = feature.GetGeometryRef()
                if geom:
                    env = geom.GetEnvelope()
                    # Если координаты > 180 градусов, это метры
                    if any(abs(coord) > 180 for coord in env):
                        logger.warning(
                            f"CRS географическая, но координаты в метрах "
                            f"(X={env[0]:.2f})"
                        )
                        return False
            
            layer.ResetReading()
            return True
            
        except Exception as e:
            logger.warning(f"Ошибка проверки CRS: {e}")
            return False
    
    def _create_fallback_crs(self, target_crs):
        """
        Создаёт fallback CRS (МСК-03).
        
        Для цели WGS/UTM не копируем +ellps/+towgs84 с UTM: исходник MIF в Пулково-42,
        иначе «фиктивный» tmerc с WGS-эллипсоидом даёт неверный перенос кадастра.
        Для цели tmerc/МСК (как 1..3.tiff) по-прежнему берём хвосты из target, чтобы
        совпадать с растром.
        """
        # Пулково-42 / МСК-03 по умолчанию (три параметра, как в BOUNDCRS подложек)
        ellps_param = "krass"
        towgs84_params = "23.57,-140.95,-79.8"
        target_is_utm = False
        if target_crs and hasattr(target_crs, "ExportToProj4"):
            try:
                tp = target_crs.ExportToProj4() or ""
                tlow = tp.lower()
                if "+proj=utm" in tlow:
                    target_is_utm = True
            except Exception:
                pass
        if not target_is_utm and target_crs and hasattr(target_crs, "ExportToProj4"):
            try:
                target_proj4 = target_crs.ExportToProj4()
                parts = dict(
                    p.split("=", 1)
                    for p in target_proj4.split()
                    if "=" in p
                )
                e = (parts.get("+ellps", "") or "krass").lower()
                if e and e not in ("wgs84", "grs80"):
                    ellps_param = parts.get("+ellps", "krass")
                else:
                    ellps_param = "krass"
                tw = parts.get("+towgs84", "")
                if tw:
                    towgs84_params = tw
            except Exception:
                pass
        if target_is_utm:
            ellps_param = "krass"
            towgs84_params = "23.57,-140.95,-79.8"
        
        # Создаём PROJ-строку для МСК-03
        proj4_str = (
            "+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 "
            "+x_0=4250000 +y_0=-5211057.63 "
            f"+ellps={ellps_param} "
            f"+towgs84={towgs84_params} "
            "+units=m +no_defs"
        )
        
        logger.info(f"Использую fallback CRS: {proj4_str[:100]}...")
        
        source_crs = osr.SpatialReference()
        if source_crs.ImportFromProj4(proj4_str) != 0:
            logger.warning("Не удалось импортировать fallback PROJ-строку")
            # Последний fallback: EPSG:32648
            source_crs.ImportFromEPSG(32648)
        
        return source_crs
    
    def _read_features(
        self,
        layer,
        field_names: List[str],
        target_bbox,
        manual_offset_x_m: float = 0.0,
        manual_offset_y_m: float = 0.0,
    ) -> List[CadastralParcel]:
        """Читает объекты из слоя."""
        cadastral_objects = []
        total_features = 0
        use_manual_offset = (
            abs(float(manual_offset_x_m)) > 1e-12 or abs(float(manual_offset_y_m)) > 1e-12
        )
        if use_manual_offset:
            logger.info(
                "Применяется ручной сдвиг кадастра: dX=%s м, dY=%s м",
                manual_offset_x_m,
                manual_offset_y_m,
            )
        
        # Диагностика координат
        sample_bounds_original = []
        sample_bounds_transformed = []
        
        for feature in layer:
            total_features += 1
            geom = feature.GetGeometryRef()
            
            if geom is None:
                continue
            
            # Сохраняем исходные координаты для диагностики
            if len(sample_bounds_original) < 5:
                try:
                    env = geom.GetEnvelope()
                    sample_bounds_original.append(
                        (env[0], env[2], env[1], env[3])
                    )
                except Exception:
                    pass
            
            # Трансформируем геометрию
            if self.transformer:
                shapely_geom = self.transformer.transform_geometry(geom)
            else:
                # Без трансформации (если CRS совпадают)
                from shapely.geometry import shape
                import json
                try:
                    shapely_geom = shape(json.loads(geom.ExportToJson()))
                except Exception:
                    continue
            
            if not shapely_geom or shapely_geom.is_empty or shapely_geom.area == 0:
                continue

            if use_manual_offset:
                shapely_geom = shapely_translate(
                    shapely_geom,
                    xoff=float(manual_offset_x_m),
                    yoff=float(manual_offset_y_m),
                )
            
            # Диагностика трансформированных координат
            if len(sample_bounds_transformed) < 5:
                try:
                    sample_bounds_transformed.append(shapely_geom.bounds)
                except Exception:
                    pass
            
            # Извлекаем атрибуты
            attributes = self._extract_attributes(feature, field_names)
            cadastral_number = self._extract_cadastral_number(attributes)
            object_id = self._extract_object_id(attributes, len(cadastral_objects))
            
            # Фильтрация по bbox
            if target_bbox and not shapely_geom.intersects(target_bbox):
                continue
            
            # Создаём объект CadastralParcel
            parcel = CadastralParcel(
                geometry=shapely_geom,
                cadastral_number=cadastral_number,
                area_sqm=shapely_geom.area,
                centroid=shapely_geom.centroid.coords[0],
                attributes=attributes,
                object_id=object_id,
                bounds=shapely_geom.bounds,
                exterior_coords=list(shapely_geom.exterior.coords)
            )
            
            cadastral_objects.append(parcel)
        
        # Диагностика
        target_bounds_tuple = target_bbox.bounds if target_bbox else None
        self._print_diagnostics(
            total_features,
            len(cadastral_objects),
            sample_bounds_original,
            sample_bounds_transformed,
            target_bbox,
            target_bounds_tuple
        )
        
        return cadastral_objects
    
    def _extract_attributes(
        self,
        feature,
        field_names: List[str]
    ) -> Dict[str, Any]:
        """Извлекает атрибуты из feature."""
        attributes = {}
        for field_name in field_names:
            try:
                attributes[field_name] = feature.GetField(field_name)
            except Exception:
                attributes[field_name] = None
        return attributes
    
    def _extract_cadastral_number(self, attributes: Dict[str, Any]) -> str:
        """Извлекает кадастровый номер из атрибутов."""
        # Пробуем разные варианты названий полей
        for field_name in [
            "Fed_KN",              # Федеральный кадастровый номер (ваш формат!)
            "CADASTRAL_NUMBER",
            "CAD_NUMBER",
            "NUMBER",
            "CADNUM",
            "КАД_НОМЕР"
        ]:
            if field_name in attributes and attributes[field_name]:
                return str(attributes[field_name])
        
        # Если не нашли, генерируем
        return f"Parcel_Unknown"
    
    def _extract_object_id(self, attributes: Dict[str, Any], index: int) -> Any:
        """Извлекает ID объекта из атрибутов."""
        for field_name in ["ID", "OBJECTID", "FID", "FEATURE_ID"]:
            if field_name in attributes and attributes[field_name] is not None:
                return attributes[field_name]
        return index
    
    def _print_diagnostics(
        self,
        total_features: int,
        objects_count: int,
        sample_bounds_original: List[tuple],
        sample_bounds_transformed: List[tuple],
        target_bbox,
        target_bounds: Optional[tuple] = None
    ):
        """Выводит диагностическую информацию."""
        logger.info(f"Всего обработано features из MIF/MID: {total_features}")
        
        if sample_bounds_original:
            logger.info("Примеры bounds полигонов БЕЗ трансформации (первые 3):")
            for i, (minx, miny, maxx, maxy) in enumerate(sample_bounds_original[:3], 1):
                logger.info(f"  [{i}] X=[{minx:.2f},{maxx:.2f}] Y=[{miny:.2f},{maxy:.2f}]")
        
        if sample_bounds_transformed:
            logger.info("Примеры bounds полигонов ПОСЛЕ трансформации (первые 3):")
            for i, (minx, miny, maxx, maxy) in enumerate(sample_bounds_transformed[:3], 1):
                logger.info(f"  [{i}] X=[{minx:.2f},{maxx:.2f}] Y=[{miny:.2f},{maxy:.2f}]")
        
        # Проверка на проблемы с подробной диагностикой
        if objects_count == 0 and total_features > 0 and target_bbox is not None:
            logger.warning("="*80)
            logger.warning("ПРОБЛЕМА: Не найдено кадастровых объектов в области растра!")
            logger.warning("="*80)
            
            if target_bounds:
                logger.warning(f"Границы растра: X=[{target_bounds[0]:.2f},{target_bounds[2]:.2f}] Y=[{target_bounds[1]:.2f},{target_bounds[3]:.2f}]")
            
            if sample_bounds_transformed:
                logger.warning("Координаты кадастра ПОСЛЕ трансформации НЕ попадают в область растра!")
                logger.warning("Возможные причины:")
                logger.warning("  1. Неправильная трансформация координат (проблема с x_0 или y_0)")
                logger.warning("  2. Кадастр и растр в разных зонах МСК")
                logger.warning("  3. Нужно применить ручное смещение координат")
                
                # Вычисляем разницу
                if sample_bounds_transformed and target_bounds:
                    cad_center_x = (sample_bounds_transformed[0][0] + sample_bounds_transformed[0][2]) / 2
                    raster_center_x = (target_bounds[0] + target_bounds[2]) / 2
                    diff_x = cad_center_x - raster_center_x
                    
                    logger.warning(f"Разница по X: ~{diff_x:.0f} метров")
                    logger.warning(f"Центр кадастра: X={cad_center_x:.2f}")
                    logger.warning(f"Центр растра: X={raster_center_x:.2f}")
            
            logger.warning("="*80)
    
    @property
    def crs_same_as_target(self) -> bool:
        """Возвращает True если CRS кадастра совпадает с целевой."""
        return self._crs_same_as_target
    
    def validate(self, data: List[CadastralParcel]) -> bool:
        """
        Валидирует прочитанные данные.
        
        Args:
            data: Данные для валидации
            
        Returns:
            True если данные валидны
        """
        if data is None:
            return False
        
        if not isinstance(data, list):
            return False
        
        # Проверяем, что есть хотя бы один объект
        if len(data) == 0:
            logger.warning("Список кадастровых объектов пуст")
            return True  # Пустой список - технически валиден
        
        # Проверяем первый объект
        first = data[0]
        if not isinstance(first, CadastralParcel):
            return False
        
        return True
