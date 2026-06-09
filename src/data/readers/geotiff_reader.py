"""Reader для чтения GeoTIFF файлов."""

import logging
from typing import Optional
from osgeo import gdal, osr
from osgeo.gdal import UseExceptions

from .base import DataReader
from ...domain.models import GeoTiffData

UseExceptions()
logger = logging.getLogger(__name__)


class GeoTiffReader(DataReader[GeoTiffData]):
    """Класс для чтения GeoTIFF файлов с использованием GDAL."""
    
    def __init__(self):
        """Инициализация reader."""
        pass
    
    def read(self, path: str) -> GeoTiffData:
        """
        Читает GeoTIFF файл.
        
        Args:
            path: Путь к GeoTIFF файлу
            
        Returns:
            GeoTiffData объект с данными растра
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если не удалось прочитать или распарсить файл
        """
        self._check_file_exists(path)
        
        dataset = None
        try:
            # Открываем растровый файл
            dataset = gdal.Open(path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть GeoTIFF файл: {path}")
            
            # Извлекаем основные характеристики
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            num_bands = dataset.RasterCount
            
            logger.info(f"Растр: {width}x{height}, {num_bands} каналов")
            
            # Читаем данные
            data = dataset.ReadAsArray()
            
            # Геотрансформация
            transform = dataset.GetGeoTransform()
            
            # Извлекаем проекцию
            projection_wkt, proj4_projection = self._extract_projection(dataset, path)
            
            if not projection_wkt:
                raise ValueError(
                    f"Не удалось определить проекцию растра ({path}). "
                    "Убедитесь, что GeoTIFF имеет корректно определенную проекцию."
                )
            
            logger.info(f"Проекция растра: {projection_wkt[:80]}...")
            logger.info(f"Proj4: {proj4_projection}")
            
            # Вычисляем границы
            bounds = self._calculate_bounds(width, height, transform)
            
            logger.info(
                f"Границы растра: "
                f"X=[{bounds[0]:.2f}, {bounds[2]:.2f}], "
                f"Y=[{bounds[1]:.2f}, {bounds[3]:.2f}]"
            )
            
            # Создаем CRS объект для совместимости с rasterio
            crs = self._create_crs(projection_wkt)
            
            # Создаем объект GeoTiffData
            geotiff_data = GeoTiffData(
                width=width,
                height=height,
                num_bands=num_bands,
                data=data,
                transform=transform,
                crs=crs,
                projection_wkt=projection_wkt,
                proj4_projection=proj4_projection,
                bounds=bounds
            )
            
            # Валидируем
            if not self.validate(geotiff_data):
                raise ValueError("Прочитанные данные невалидны")
            
            return geotiff_data
            
        except Exception as e:
            logger.error(f"Ошибка чтения растра: {e}")
            raise
        finally:
            # Освобождаем ресурсы
            if dataset:
                dataset = None
    
    def _extract_projection(self, dataset, file_path: str) -> tuple:
        """
        Извлекает информацию о проекции из dataset.
        
        Returns:
            Кортеж (projection_wkt, proj4_projection)
        """
        projection_wkt = None
        proj4_projection = None
        
        srs_from_dataset = dataset.GetSpatialRef()
        if not srs_from_dataset:
            return None, None
        
        try:
            # Экспортируем в Proj4
            proj4_candidate = srs_from_dataset.ExportToProj4()
            
            # Обрабатываем кодировку (для русских МСК)
            if isinstance(proj4_candidate, bytes):
                proj4_candidate = self._decode_projection_string(
                    proj4_candidate,
                    "Proj4"
                )
            
            if not proj4_candidate:
                return None, None
            
            # Нормализуем через GDAL
            temp_srs = osr.SpatialReference()
            if temp_srs.ImportFromProj4(proj4_candidate) != 0:
                logger.warning("Не удалось импортировать Proj4 в SRS объект")
                return None, None
            
            # Экспортируем в WKT
            projection_wkt = temp_srs.ExportToWkt()
            proj4_projection = proj4_candidate
            
            # Проверяем кодировку WKT
            if isinstance(projection_wkt, bytes):
                projection_wkt = self._decode_projection_string(
                    projection_wkt,
                    "WKT"
                )
            
            return projection_wkt, proj4_projection
            
        except Exception as e:
            logger.warning(f"Ошибка при обработке проекции: {e}")
            return None, None
    
    def _decode_projection_string(
        self,
        proj_bytes: bytes,
        proj_type: str
    ) -> Optional[str]:
        """
        Декодирует строку проекции из bytes.
        
        Args:
            proj_bytes: Байты для декодирования
            proj_type: Тип проекции ("Proj4" или "WKT")
            
        Returns:
            Декодированная строка или None
        """
        # Пробуем UTF-8
        try:
            return proj_bytes.decode('utf-8')
        except UnicodeDecodeError:
            pass
        
        # Пробуем CP1251 (русская кодировка)
        try:
            return proj_bytes.decode('cp1251')
        except UnicodeDecodeError:
            logger.warning(
                f"Не удалось декодировать строку {proj_type} "
                "ни в UTF-8, ни в CP1251"
            )
            return None
    
    def _calculate_bounds(self, width: int, height: int, transform: tuple) -> tuple:
        """
        Вычисляет географические границы растра.
        
        Returns:
            Кортеж (min_x, min_y, max_x, max_y)
        """
        min_x = transform[0]
        max_y = transform[3]
        max_x = transform[0] + width * transform[1]
        min_y = transform[3] + height * transform[5]
        
        return (min_x, min_y, max_x, max_y)
    
    def _create_crs(self, projection_wkt: str):
        """
        Создает CRS объект для совместимости с rasterio.
        
        Returns:
            rasterio CRS объект или None
        """
        try:
            import rasterio.crs
            return rasterio.crs.CRS.from_wkt(projection_wkt)
        except Exception as e:
            logger.debug(f"Не удалось создать rasterio CRS: {e}")
            return None
    
    def validate(self, data: GeoTiffData) -> bool:
        """
        Валидирует прочитанные данные.
        
        Args:
            data: Данные для валидации
            
        Returns:
            True если данные валидны
        """
        if data is None:
            return False
        
        if data.width <= 0 or data.height <= 0:
            return False
        
        if data.num_bands <= 0:
            return False
        
        if data.data is None or data.data.size == 0:
            return False
        
        if not data.projection_wkt:
            return False
        
        return True
