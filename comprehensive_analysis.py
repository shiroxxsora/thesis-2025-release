import os
import sys
import json
import math
import numpy as np
import cv2
import torch
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from shapely.strtree import STRtree

# Геопространственные библиотеки
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import transform as shapely_transform
from osgeo import gdal, ogr, osr
from osgeo.gdal import UseExceptions
import geopandas as gpd

# Detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import pandas as pd

UseExceptions()

class ComprehensiveAnalyzer:
    def __init__(self):
        self.config = {
            'input_geotiff': "geotiffs/input.tiff",
            'cadastral_data': "cadastr/ЗУ все2.MIF",
            'detectron_config': "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            'model_weights': "models/pavel-01-07-25/model_final.pth",
            'output_dir': "output/comprehensive",
            'chunk_size': 5000,
            'overlap': 1536,
            'model_input_size': 1024,
            'score_threshold': 0.3,
            'min_polygon_area': 100.0,
            'min_violation_area': 0.5
        }
        
        self.results = {
            'detected_objects': [],
            'cadastral_parcels': [],
            'violations': [],
            'statistics': {}
        }
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def setup_model(self):
        """Инициализация модели Detectron2"""
        cfg = get_cfg()
        cfg.merge_from_file(self.config['detectron_config'])
        cfg.MODEL.WEIGHTS = self.config['model_weights']
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config['score_threshold']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)
    
    def normalize_to_uint8(self, ch: np.ndarray) -> np.ndarray:
        """Нормализация канала к uint8"""
        if ch.dtype == np.uint8:
            return ch
        ch_min = float(np.nanmin(ch))
        ch_max = float(np.nanmax(ch))
        if np.isfinite(ch_min) and np.isfinite(ch_max) and ch_max > ch_min:
            ch_norm = (255.0 * (ch.astype(np.float32) - ch_min) / (ch_max - ch_min))
            return ch_norm.astype(np.uint8)
        else:
            return ch.astype(np.uint8)
    
    def process_mask_to_polygons(self, mask: np.ndarray, min_area: float = 100.0) -> List[np.ndarray]:
        """Извлечение полигонов из маски"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                poly = approx.reshape(-1, 2)
                polygons.append(poly)
                
        return polygons
    
    def pixel_to_geo_coords(self, pixel_coords: np.ndarray, transform: tuple) -> List[Tuple[float, float]]:
        """Преобразование пиксельных координат в географические"""
        geo_coords = []
        for x, y in pixel_coords:
            geo_x = transform[0] + x * transform[1] + y * transform[2]
            geo_y = transform[3] + x * transform[4] + y * transform[5]
            geo_coords.append((geo_x, geo_y))
        return geo_coords
    
    def read_geotiff(self, file_path: str) -> Optional[Dict]:
        """Чтение GeoTIFF файла с использованием GDAL"""
        dataset = None
        try:
            # Открываем растровый файл через GDAL
            dataset = gdal.Open(file_path)
            if dataset is None:
                print(f"Не удалось открыть файл: {file_path}")
                return None

            # Извлекаем основные характеристики растра
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            num_bands = dataset.RasterCount
            
            # Читаем все пиксельные данные в память как numpy массив
            data = dataset.ReadAsArray()
            
            # Получаем геотрансформацию
            transform = dataset.GetGeoTransform()

            # Переменные для хранения информации о проекции
            projection_wkt = None
            proj4_projection = None
            
            # Получаем информацию о системе координат из файла
            srs_from_dataset = dataset.GetSpatialRef()
            if srs_from_dataset:
                try:
                    # Экспортируем проекцию в формат Proj4 для дальнейшей обработки
                    proj4_candidate = srs_from_dataset.ExportToProj4()
                    
                    # Обрабатываем возможную проблему с кодировкой (особенно для русских МСК)
                    if isinstance(proj4_candidate, bytes):
                        try:
                            # Сначала пробуем стандартную UTF-8 кодировку
                            proj4_candidate = proj4_candidate.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                # Если не получилось, пробуем CP1251 (часто используется в России)
                                proj4_candidate = proj4_candidate.decode('cp1251')
                            except UnicodeDecodeError:
                                print(f"Предупреждение: Не удалось декодировать строку Proj4 (bytes) ни в UTF-8, ни в CP1251.")
                                proj4_candidate = None

                    # Если удалось получить Proj4 строку, нормализуем её через GDAL
                    if proj4_candidate:
                        # Создаем временный SRS объект для нормализации проекции
                        temp_srs = osr.SpatialReference()
                        
                        # Импортируем Proj4 строку и проверяем успешность операции
                        if temp_srs.ImportFromProj4(proj4_candidate) == 0:
                            # Экспортируем обратно в WKT для использования в GDAL операциях
                            projection_wkt = temp_srs.ExportToWkt()
                            proj4_projection = proj4_candidate
                            
                            # Проверяем кодировку WKT (тоже может быть в bytes)
                            if isinstance(projection_wkt, bytes):
                                try:
                                    projection_wkt = projection_wkt.decode('utf-8')
                                except UnicodeDecodeError:
                                    print(f"Предупреждение: Не удалось декодировать WKT после ImportFromProj4.")
                                    projection_wkt = None
                        else:
                            print(f"Предупреждение: Не удалось импортировать Proj4 в SRS объект.")
                            projection_wkt = None
                    else:
                        print(f"Предупреждение: Не удалось получить Proj4 строку из проекции растра.")
                        projection_wkt = None
                except Exception as e:
                    print(f"Предупреждение: Ошибка при обработке проекции (Proj4/WKT): {e}")
                    projection_wkt = None

            # КРИТИЧЕСКАЯ ПРОВЕРКА: Если проекция не определена, останавливаем обработку
            if not projection_wkt:
                print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: Не удалось определить или импортировать проекцию входного растра ({file_path}).")
                print("Это может быть связано с тем, что растр использует локальную систему координат (например, МСК 03),")
                print("которая не может быть автоматически распознана GDAL.")
                print("Пожалуйста, убедитесь, что ваш GeoTIFF имеет корректно определенную проекцию.")
                return None
            else:
                print(f"Определена проекция растра (начало): {projection_wkt[:50]}...")
                print(f"Proj4 проекции растра: {proj4_projection}")

            # Вычисляем географические границы растра в его системе координат
            min_x_raster = transform[0]
            max_y_raster = transform[3]
            max_x_raster = transform[0] + width * transform[1]
            min_y_raster = transform[3] + height * transform[5]
            print(f"Границы растра в его CRS: X=[{min_x_raster:.2f}, {max_x_raster:.2f}], Y=[{min_y_raster:.2f}, {max_y_raster:.2f}]")

            # Выводим детальную информацию о геотрансформации
            print('GeoTransform:', transform)
            
            # Вычисляем размер пикселя в единицах системы координат
            pixel_width = abs(transform[1])
            pixel_height = abs(transform[5])
            print('Pixel width:', pixel_width)
            print('Pixel height:', pixel_height)

            # Создаем объект CRS для совместимости с rasterio
            crs = None
            try:
                import rasterio.crs
                crs = rasterio.crs.CRS.from_wkt(projection_wkt)
            except:
                # Если не удалось создать CRS, используем None
                pass

            return {
                'width': width,
                'height': height,
                'num_bands': num_bands,
                'data': data,
                'transform': transform,
                'crs': crs,
                'projection_wkt': projection_wkt,
                'proj4_projection': proj4_projection,
                'bounds': (min_x_raster, min_y_raster, max_x_raster, max_y_raster)
            }
        except Exception as e:
            print(f"Ошибка при чтении растра: {e}")
            return None
        finally:
            # Освобождаем ресурсы GDAL
            if dataset:
                dataset = None
    
    def process_chunk(self, chunk: np.ndarray, predictor, model_input_dims: Tuple[int, int]) -> np.ndarray:
        """Обработка чанка нейросетью"""
        # Подготовка изображения
        if chunk.shape[0] > 3:
            chunk = chunk[:3]
        
        # Нормализация каналов
        channels = [self.normalize_to_uint8(chunk[i]) for i in range(chunk.shape[0])]
        if len(channels) < 3:
            channels.extend([channels[-1]] * (3 - len(channels)))
        
        img_array = np.stack(channels, axis=0)
        img_vis = np.transpose(img_array, (1, 2, 0))
        
        # Изменение размера для модели
        img_resized = cv2.resize(img_vis, model_input_dims, interpolation=cv2.INTER_LINEAR)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        # Предсказание
        with torch.no_grad():
            outputs = predictor(img_bgr)
        
        # Получение масок
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy()
            return masks
        return np.array([])
    
    def detect_objects(self, geotiff_data: Dict) -> List[Dict]:
        """Обнаружение объектов - каждая маска Mask R-CNN как отдельный участок БЕЗ объединения"""
        print("Начинаю обнаружение объектов...")
        predictor = self.setup_model()
        
        height = geotiff_data['height']
        width = geotiff_data['width']
        chunk_size = self.config['chunk_size']
        overlap = self.config['overlap']
        step_size = chunk_size - overlap
        num_chunks_y = math.ceil(height / step_size)
        num_chunks_x = math.ceil(width / step_size)
        total_chunks = num_chunks_x * num_chunks_y
        processed_chunks = 0
        
        all_instances = []  # Список всех отдельных инстансов
        
        print(f"Начинаю обработку {total_chunks} чанков...")
        
        for y_idx in range(num_chunks_y):
            for x_idx in range(num_chunks_x):
                processed_chunks += 1
                print(f"Обработка чанка {processed_chunks}/{total_chunks}")
                
                x_start = x_idx * step_size
                y_start = y_idx * step_size
                x_end = min(x_start + chunk_size, width)
                y_end = min(y_start + chunk_size, height)
                
                if x_end - x_start <= 0 or y_end - y_start <= 0:
                    continue
                
                chunk = geotiff_data['data'][:, y_start:y_end, x_start:x_end]
                chunk_height, chunk_width = y_end - y_start, x_end - x_start
                
                # Получаем маски для чанка
                masks = self.process_chunk(chunk, predictor, (self.config['model_input_size'], self.config['model_input_size']))
                
                # КАЖДАЯ маска обрабатывается как ОТДЕЛЬНЫЙ инстанс
                for mask_idx, mask in enumerate(masks):
                    # Масштабируем маску к размеру чанка
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), 
                        (chunk_width, chunk_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # Извлекаем полигоны из ЭТОЙ конкретной маски
                    polygons = self.process_mask_to_polygons(mask_resized, self.config['min_polygon_area'])
                    
                    for poly_pixels in polygons:
                        # Преобразуем в глобальные координаты
                        global_poly_pixels = poly_pixels + np.array([x_start, y_start])
                        geo_coords = self.pixel_to_geo_coords(global_poly_pixels, geotiff_data['transform'])
                        
                        if len(geo_coords) >= 3:
                            poly = Polygon(geo_coords)
                            if poly.is_valid and poly.area > 0:
                                # Каждая маска = отдельный инстанс (БЕЗ объединения!)
                                instance = {
                                    'geometry': poly,
                                    'area_sqm': poly.area,
                                    'centroid': poly.centroid.coords[0],
                                    'chunk_id': f"{x_idx}_{y_idx}",
                                    'mask_id': mask_idx,
                                    'instance_id': f"chunk_{x_idx}_{y_idx}_mask_{mask_idx}",
                                    'mask_area_pixels': np.sum(mask_resized > 0)
                                }
                                all_instances.append(instance)
        
        print(f"Обнаружено {len(all_instances)} отдельных инстансов Mask R-CNN")
        
        # Фильтрация по площади
        filtered_instances = []
        for instance in all_instances:
            if instance['area_sqm'] > self.config['min_polygon_area']:
                filtered_instances.append(instance)
        
        print(f"После фильтрации по площади осталось {len(filtered_instances)} инстансов")
        
        # НЕ создаем объединенную маску - работаем с отдельными инстансами
        return filtered_instances
    
    def read_cadastral_data(self, mif_path: str, raster_crs, raster_bounds=None) -> List[Dict]:
        """Чтение кадастровых данных с фильтрацией по bounds изображения"""
        print("Чтение кадастровых данных...")
        cadastral_objects = []
        try:
            driver = ogr.GetDriverByName('MapInfo File')
            datasource = driver.Open(mif_path, 0)
            if datasource is None:
                print(f"Ошибка: Не удалось открыть MIF/MID файл: {mif_path}")
                return []
            layer = datasource.GetLayer()
            raster_srs_wkt = None
            raster_proj4 = None
            if hasattr(raster_crs, 'to_wkt'):
                raster_srs_wkt = raster_crs.to_wkt()
            elif hasattr(raster_crs, 'wkt'):
                raster_srs_wkt = raster_crs.wkt
            if raster_srs_wkt:
                temp_srs = osr.SpatialReference()
                temp_srs.ImportFromWkt(raster_srs_wkt)
                raster_proj4 = temp_srs.ExportToProj4()
            if raster_proj4:
                raster_proj4_parts = dict(p.split('=', 1) for p in raster_proj4.split() if '=' in p)
                towgs84_params = raster_proj4_parts.get('+towgs84', '')
                ellps_param = raster_proj4_parts.get('+ellps', 'krass')
            else:
                towgs84_params = ''
                ellps_param = 'krass'
            source_cadastr_proj4_str = (
                f"+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 "
                f"+x_0=4250000 +y_0=-5211057.63 "
                f"+ellps={ellps_param} "
                f"+towgs84={towgs84_params} "
                f"+units=m +no_defs"
            )
            source_srs_cadastr = osr.SpatialReference()
            try:
                if source_srs_cadastr.ImportFromProj4(source_cadastr_proj4_str) == 0:
                    pass
                else:
                    source_srs_cadastr = layer.GetSpatialRef()
                    if source_srs_cadastr is None:
                        source_srs_cadastr = osr.SpatialReference()
                        source_srs_cadastr.ImportFromEPSG(32648)
            except Exception as e:
                source_srs_cadastr = layer.GetSpatialRef()
                if source_srs_cadastr is None:
                    source_srs_cadastr = osr.SpatialReference()
                    source_srs_cadastr.ImportFromEPSG(32648)
            target_srs_raster = osr.SpatialReference()
            if raster_srs_wkt:
                target_srs_raster.ImportFromWkt(raster_srs_wkt)
            else:
                print("Предупреждение: Не удалось определить проекцию растра")
                return []
            source_srs_cadastr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target_srs_raster.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transform_cadastr_to_raster = osr.CoordinateTransformation(source_srs_cadastr, target_srs_raster)
            # --- Фильтрация по bounds изображения ---
            if raster_bounds is not None:
                from shapely.geometry import box
                min_x, min_y, max_x, max_y = raster_bounds
                raster_bbox = box(min_x, min_y, max_x, max_y)
            else:
                raster_bbox = None
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None:
                    continue
                geom_clone = geom.Clone()
                try:
                    geom_clone.Transform(transform_cadastr_to_raster)
                except Exception as e:
                    continue
                try:
                    cadastral_number = None
                    for field_name in ['CADASTRAL_NUMBER', 'CAD_NUMBER', 'NUMBER']:
                        if feature.GetFieldIndex(field_name) >= 0:
                            cadastral_number = feature.GetField(field_name)
                            break
                    if not cadastral_number:
                        cadastral_number = f"Parcel_{len(cadastral_objects)}"
                    if geom_clone.GetGeometryType() == ogr.wkbPolygon:
                        poly = Polygon(json.loads(geom_clone.ExportToJson())['coordinates'][0])
                        if poly.is_valid and poly.area > 0:
                            # Фильтрация по пересечению с bbox изображения
                            if raster_bbox is None or poly.intersects(raster_bbox):
                                cadastral_objects.append({
                                    'geometry': poly,
                                    'cadastral_number': cadastral_number,
                                    'area_sqm': poly.area,
                                    'centroid': poly.centroid.coords[0]
                                })
                    elif geom_clone.GetGeometryType() == ogr.wkbMultiPolygon:
                        for i in range(geom_clone.GetGeometryCount()):
                            subgeom = geom_clone.GetGeometryRef(i)
                            poly = Polygon(json.loads(subgeom.ExportToJson())['coordinates'][0])
                            if poly.is_valid and poly.area > 0:
                                if raster_bbox is None or poly.intersects(raster_bbox):
                                    cadastral_objects.append({
                                        'geometry': poly,
                                        'cadastral_number': f"{cadastral_number}_part_{i}",
                                        'area_sqm': poly.area,
                                        'centroid': poly.centroid.coords[0]
                                    })
                except Exception as e:
                    continue
            print(f"Успешно прочитано и трансформировано {len(cadastral_objects)} кадастровых полигонов (только по изображению)")
        except Exception as e:
            print(f"Ошибка чтения кадастровых данных: {e}")
        return cadastral_objects
    
    def read_cadastral_data_enhanced(self, mif_path: str, raster_crs, raster_bounds=None) -> List[Dict]:
        """Улучшенное чтение кадастровых данных с извлечением всех доступных полей"""
        print("Чтение кадастровых данных (расширенная версия)...")
        cadastral_objects = []
        try:
            driver = ogr.GetDriverByName('MapInfo File')
            datasource = driver.Open(mif_path, 0)
            if datasource is None:
                print(f"Ошибка: Не удалось открыть MIF/MID файл: {mif_path}")
                return []
            
            layer = datasource.GetLayer()
            layer_defn = layer.GetLayerDefn()
            
            # Извлекаем информацию о всех полях
            field_names = []
            for i in range(layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(i)
                field_names.append(field_defn.GetName())
            
            print(f"Найдены поля в MID/MIF файле: {field_names}")
            
            # Настройка трансформации координат
            raster_srs_wkt = None
            raster_proj4 = None
            if hasattr(raster_crs, 'to_wkt'):
                raster_srs_wkt = raster_crs.to_wkt()
            elif hasattr(raster_crs, 'wkt'):
                raster_srs_wkt = raster_crs.wkt
            
            if raster_srs_wkt:
                temp_srs = osr.SpatialReference()
                temp_srs.ImportFromWkt(raster_srs_wkt)
                raster_proj4 = temp_srs.ExportToProj4()
            
            if raster_proj4:
                raster_proj4_parts = dict(p.split('=', 1) for p in raster_proj4.split() if '=' in p)
                towgs84_params = raster_proj4_parts.get('+towgs84', '')
                ellps_param = raster_proj4_parts.get('+ellps', 'krass')
            else:
                towgs84_params = ''
                ellps_param = 'krass'
            
            source_cadastr_proj4_str = (
                f"+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 "
                f"+x_0=4250000 +y_0=-5211057.63 "
                f"+ellps={ellps_param} "
                f"+towgs84={towgs84_params} "
                f"+units=m +no_defs"
            )
            
            source_srs_cadastr = osr.SpatialReference()
            try:
                if source_srs_cadastr.ImportFromProj4(source_cadastr_proj4_str) == 0:
                    pass
                else:
                    source_srs_cadastr = layer.GetSpatialRef()
                    if source_srs_cadastr is None:
                        source_srs_cadastr = osr.SpatialReference()
                        source_srs_cadastr.ImportFromEPSG(32648)
            except Exception as e:
                source_srs_cadastr = layer.GetSpatialRef()
                if source_srs_cadastr is None:
                    source_srs_cadastr = osr.SpatialReference()
                    source_srs_cadastr.ImportFromEPSG(32648)
            
            target_srs_raster = osr.SpatialReference()
            if raster_srs_wkt:
                target_srs_raster.ImportFromWkt(raster_srs_wkt)
            else:
                print("Предупреждение: Не удалось определить проекцию растра")
                return []
            
            source_srs_cadastr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target_srs_raster.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transform_cadastr_to_raster = osr.CoordinateTransformation(source_srs_cadastr, target_srs_raster)
            
            # Фильтрация по bounds изображения
            if raster_bounds is not None:
                from shapely.geometry import box
                min_x, min_y, max_x, max_y = raster_bounds
                raster_bbox = box(min_x, min_y, max_x, max_y)
            else:
                raster_bbox = None
            
            # Читаем каждый объект
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None:
                    continue
                
                geom_clone = geom.Clone()
                try:
                    geom_clone.Transform(transform_cadastr_to_raster)
                except Exception as e:
                    continue
                
                try:
                    # Извлекаем все доступные атрибуты
                    attributes = {}
                    for field_name in field_names:
                        try:
                            field_value = feature.GetField(field_name)
                            attributes[field_name] = field_value
                        except:
                            attributes[field_name] = None
                    
                    # Определяем кадастровый номер
                    cadastral_number = None
                    for field_name in ['CADASTRAL_NUMBER', 'CAD_NUMBER', 'NUMBER', 'CADNUM', 'КАД_НОМЕР']:
                        if field_name in attributes and attributes[field_name]:
                            cadastral_number = str(attributes[field_name])
                            break
                    
                    if not cadastral_number:
                        cadastral_number = f"Parcel_{len(cadastral_objects)}"
                    
                    # Извлекаем идентификаторы
                    object_id = None
                    for field_name in ['ID', 'OBJECTID', 'FID', 'FEATURE_ID']:
                        if field_name in attributes and attributes[field_name] is not None:
                            object_id = attributes[field_name]
                            break
                    
                    if object_id is None:
                        object_id = len(cadastral_objects)
                    
                    # Обрабатываем геометрию
                    if geom_clone.GetGeometryType() == ogr.wkbPolygon:
                        poly = Polygon(json.loads(geom_clone.ExportToJson())['coordinates'][0])
                        if poly.is_valid and poly.area > 0:
                            if raster_bbox is None or poly.intersects(raster_bbox):
                                # Вычисляем координаты углов участка
                                bounds = poly.bounds
                                exterior_coords = list(poly.exterior.coords)
                                
                                cadastral_objects.append({
                                    'geometry': poly,
                                    'cadastral_number': cadastral_number,
                                    'object_id': object_id,
                                    'area_sqm': poly.area,
                                    'centroid': poly.centroid.coords[0],
                                    'bounds': bounds,
                                    'exterior_coords': exterior_coords,
                                    'attributes': attributes
                                })
                    
                    elif geom_clone.GetGeometryType() == ogr.wkbMultiPolygon:
                        for i in range(geom_clone.GetGeometryCount()):
                            subgeom = geom_clone.GetGeometryRef(i)
                            poly = Polygon(json.loads(subgeom.ExportToJson())['coordinates'][0])
                            if poly.is_valid and poly.area > 0:
                                if raster_bbox is None or poly.intersects(raster_bbox):
                                    bounds = poly.bounds
                                    exterior_coords = list(poly.exterior.coords)
                                    
                                    cadastral_objects.append({
                                        'geometry': poly,
                                        'cadastral_number': f"{cadastral_number}_part_{i}",
                                        'object_id': f"{object_id}_part_{i}",
                                        'area_sqm': poly.area,
                                        'centroid': poly.centroid.coords[0],
                                        'bounds': bounds,
                                        'exterior_coords': exterior_coords,
                                        'attributes': attributes
                                    })
                
                except Exception as e:
                    print(f"Ошибка обработки объекта: {e}")
                    continue
            
            print(f"Успешно прочитано {len(cadastral_objects)} кадастровых объектов с расширенными атрибутами")
            
        except Exception as e:
            print(f"Ошибка чтения кадастровых данных: {e}")
        
        return cadastral_objects
    
    def calculate_pixel_area_sqm(self, raster_transform: tuple) -> float:
        """Вычисляет площадь одного пикселя в квадратных метрах"""
        pixel_width = abs(raster_transform[1])
        pixel_height = abs(raster_transform[5])
        pixel_area_sqm = pixel_width * pixel_height
        return pixel_area_sqm
    
    def calculate_violation_areas_from_mask(self, mask_after_subtraction: np.ndarray, 
                                          pixel_area_sqm: float, 
                                          raster_transform: tuple) -> List[Dict]:
        """Вычисляет площади нарушений из маски (аналогично geotiff_processor)"""
        contours, _ = cv2.findContours(
            mask_after_subtraction.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        violations = []
        
        for i, contour in enumerate(contours):
            # Вычисляем площадь в пикселях
            contour_area_pixels = cv2.contourArea(contour)
            contour_area_sqm = contour_area_pixels * pixel_area_sqm
            
            # Фильтруем слишком маленькие области
            if contour_area_sqm < self.config['min_violation_area']:
                continue
            
            # Преобразуем контур в географические координаты
            contour_pixels = contour.reshape(-1, 2)
            geo_coords = self.pixel_to_geo_coords(contour_pixels, raster_transform)
            
            if len(geo_coords) >= 3:
                try:
                    poly = Polygon(geo_coords)
                    if poly.is_valid and poly.area > 0:
                        violation = {
                            'geometry': poly,
                            'violation_area': contour_area_sqm,
                            'violation_area_ha': contour_area_sqm / 10000,
                            'original_object_area': contour_area_sqm,  # Для совместимости
                            'centroid': poly.centroid.coords[0],
                            'bounds': poly.bounds,
                            'perimeter': poly.length,
                            'exterior_coords': list(poly.exterior.coords),
                            'area_pixels': contour_area_pixels,
                            'violation_id': i + 1
                        }
                        violations.append(violation)
                except Exception as e:
                    print(f"Ошибка создания полигона нарушения {i}: {e}")
                    continue
        
        return violations
    
    def create_cadastral_mask(self, cadastral_objects: List[Dict], 
                            raster_shape: tuple, 
                            raster_transform: tuple) -> np.ndarray:
        """Создает маску кадастровых участков"""
        mask = np.zeros(raster_shape, dtype=np.uint8)
        height, width = raster_shape
        
        for cadastral in cadastral_objects:
            poly = cadastral['geometry']
            
            if poly.exterior:
                coords = np.array(list(poly.exterior.coords))
                
                # Преобразуем в пиксельные координаты
                px = ((coords[:, 0] - raster_transform[0]) / raster_transform[1]).round().astype(int)
                py = ((raster_transform[3] - coords[:, 1]) / abs(raster_transform[5])).round().astype(int)
                
                # Обрезаем координаты до размеров изображения
                px = np.clip(px, 0, width - 1)
                py = np.clip(py, 0, height - 1)
                
                pts = np.stack([px, py], axis=1)
                
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
        
        return mask

    def analyze_violations(self, detected_objects: List[Dict], cadastral_objects: List[Dict]) -> List[Dict]:
        """Анализ нарушений - вычитание кадастровых участков из обнаруженных объектов"""
        print("Анализ нарушений с вычитанием кадастровых участков...")
        
        violations = []
        total_objects = len(detected_objects)
        
        # Создаем пространственный индекс для кадастровых участков
        print("Создание пространственного индекса для кадастровых участков...")
        cadastral_geometries = [cadastral['geometry'] for cadastral in cadastral_objects]
        cadastral_tree = STRtree(cadastral_geometries)
        
        # Анализируем каждый отдельный инстанс
        for obj_idx, obj in enumerate(detected_objects):
            if obj_idx % 100 == 0:
                print(f"Обработано инстансов: {obj_idx}/{total_objects}")
            
            obj_poly = obj['geometry']
            remaining_poly = obj_poly  # Начинаем с полного объекта
            
            # Находим пересекающиеся кадастровые участки
            intersecting_cadastrals = cadastral_tree.query(obj_poly)
            intersecting_cadastral_objects = [cadastral_objects[idx] for idx in intersecting_cadastrals]
            
            # Вычитаем все пересекающиеся кадастровые участки
            for cadastral_idx in intersecting_cadastrals:
                cadastral = cadastral_objects[cadastral_idx]
                cadastral_poly = cadastral['geometry']
                
                if remaining_poly.intersects(cadastral_poly):
                    # Вычитаем кадастровый участок из оставшейся части объекта
                    difference = remaining_poly.difference(cadastral_poly)
                    
                    if difference.is_empty:
                        # Объект полностью покрыт кадастровым участком
                        remaining_poly = None
                        break
                    else:
                        # Обновляем оставшуюся часть
                        remaining_poly = difference
            
            # Если осталась часть объекта после вычитания всех кадастровых участков
            if remaining_poly and not remaining_poly.is_empty:
                # Обрабатываем результат (может быть MultiPolygon)
                if remaining_poly.geom_type == 'Polygon':
                    violation_parts = [remaining_poly]
                elif remaining_poly.geom_type == 'MultiPolygon':
                    violation_parts = list(remaining_poly.geoms)
                else:
                    continue
                
                # Создаем отдельные нарушения для каждой части
                for part in violation_parts:
                    # Валидация геометрии
                    if not part.is_valid:
                        part = make_valid(part)
                        if part.is_empty:
                            continue
                    
                    # Обработка multipolygon после валидации
                    if part.geom_type == 'MultiPolygon':
                        sub_parts = list(part.geoms)
                    else:
                        sub_parts = [part]
                    
                    for sub_part in sub_parts:
                        if sub_part.area > self.config['min_violation_area']:
                            # Находим ближайший кадастровый участок (улучшенный алгоритм)
                            closest_cadastral = None
                            min_distance = float('inf')
                            
                            # Сначала ищем среди пересекающихся участков (они более релевантны)
                            for cadastral in intersecting_cadastral_objects:
                                distance = sub_part.distance(cadastral['geometry'])
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_cadastral = cadastral
                            
                            # Если среди пересекающихся не нашли или расстояние большое,
                            # ищем среди всех участков
                            if min_distance > 50.0:  # 50 метров - порог
                                for cadastral in cadastral_objects:
                                    if cadastral not in intersecting_cadastral_objects:  # Исключаем уже проверенные
                                        distance = sub_part.distance(cadastral['geometry'])
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_cadastral = cadastral
                            
                            # Создаем детальную информацию о нарушении
                            violation = {
                                'geometry': sub_part,
                                'violation_area': sub_part.area,
                                'violation_area_ha': sub_part.area / 10000,
                                'original_object_area': obj_poly.area,
                                'centroid': sub_part.centroid.coords[0],
                                'bounds': sub_part.bounds,
                                'perimeter': sub_part.length,
                                'exterior_coords': list(sub_part.exterior.coords),
                                'instance_id': obj.get('instance_id', f'instance_{obj_idx}'),
                                'mask_area_pixels': obj.get('mask_area_pixels', 0),
                                'cadastral_number': closest_cadastral['cadastral_number'] if closest_cadastral else 'Unknown',
                                'distance_to_cadastral': min_distance,
                                'closest_cadastral_id': closest_cadastral.get('object_id') if closest_cadastral else None
                            }
                            
                            if closest_cadastral:
                                violation['closest_cadastral_area'] = closest_cadastral.get('area_sqm', 0)
                                violation['closest_cadastral_centroid'] = closest_cadastral.get('centroid', (0, 0))
                            
                            violations.append(violation)
        
        print(f"Найдено {len(violations)} нарушений из {len(detected_objects)} инстансов")
        
        # Статистика
        if violations:
            total_violation_area = sum(v['violation_area'] for v in violations)
            total_detected_area = sum(obj['area_sqm'] for obj in detected_objects)
            
            print(f"\n=== ДИАГНОСТИКА АНАЛИЗА НАРУШЕНИЙ ===")
            print(f"Всего обнаружено инстансов: {len(detected_objects)}")
            print(f"Всего кадастровых участков: {len(cadastral_objects)}")
            print(f"Найдено нарушений: {len(violations)}")
            print(f"Общая площадь обнаруженных объектов: {total_detected_area:.2f} м²")
            print(f"Общая площадь нарушений: {total_violation_area:.2f} м²")
            print(f"Процент нарушений: {(total_violation_area/total_detected_area*100) if total_detected_area > 0 else 0:.1f}%")
            
            # Статистика по размерам нарушений
            areas = [v['violation_area'] for v in violations]
            print(f"Минимальная площадь нарушения: {min(areas):.2f} м²")
            print(f"Максимальная площадь нарушения: {max(areas):.2f} м²")
            print(f"Средняя площадь нарушения: {sum(areas)/len(areas):.2f} м²")
            
            # Диагностика поиска ближайших кадастровых участков
            distances = [v['distance_to_cadastral'] for v in violations]
            close_violations = [v for v in violations if v['distance_to_cadastral'] <= 50.0]
            far_violations = [v for v in violations if v['distance_to_cadastral'] > 50.0]
            
            print(f"\nДИАГНОСТИКА ПОИСКА БЛИЖАЙШИХ УЧАСТКОВ:")
            print(f"Нарушения рядом с кадастром (≤50м): {len(close_violations)}")
            print(f"Нарушения далеко от кадастра (>50м): {len(far_violations)}")
            if distances:
                print(f"Минимальное расстояние до кадастра: {min(distances):.2f} м")
                print(f"Максимальное расстояние до кадастра: {max(distances):.2f} м")
                print(f"Среднее расстояние до кадастра: {sum(distances)/len(distances):.2f} м")
                
                # Показываем примеры ближайших и дальних нарушений
                if close_violations:
                    closest = min(close_violations, key=lambda x: x['distance_to_cadastral'])
                    print(f"Самое близкое нарушение: {closest['distance_to_cadastral']:.2f}м до участка {closest['cadastral_number']}")
                
                if far_violations:
                    farthest = max(far_violations, key=lambda x: x['distance_to_cadastral'])
                    print(f"Самое далекое нарушение: {farthest['distance_to_cadastral']:.2f}м до участка {farthest['cadastral_number']}")
            
            print(f"===================================\n")
        
        return violations

    def analyze_violations_old(self, detected_objects: List[Dict], cadastral_objects: List[Dict]) -> List[Dict]:
        """Старый метод анализа нарушений - оставлен для совместимости"""
        print("Анализ нарушений (старый метод)...")
        
        violations = []
        total_objects = len(detected_objects)
        
        # Создаем пространственный индекс для кадастровых участков для быстрого поиска
        print("Создание пространственного индекса для кадастровых участков...")
        cadastral_geometries = [cadastral['geometry'] for cadastral in cadastral_objects]
        cadastral_tree = STRtree(cadastral_geometries)
        
        # Анализируем каждый обнаруженный объект
        for obj_idx, obj in enumerate(detected_objects):
            if obj_idx % 100 == 0:
                print(f"Обработано объектов: {obj_idx}/{total_objects}")
            
            obj_poly = obj['geometry']
            remaining_poly = obj_poly  # Начинаем с полного объекта
            
            # Находим только те кадастровые участки, которые пересекаются с объектом
            intersecting_cadastrals = cadastral_tree.query(obj_poly)
            
            # Проверяем пересечения только с релевантными кадастровыми участками
            for cadastral_idx in intersecting_cadastrals:
                cadastral = cadastral_objects[cadastral_idx]
                cadastral_poly = cadastral['geometry']
                
                if remaining_poly.intersects(cadastral_poly):
                    # Вычитаем кадастровый участок из оставшейся части объекта
                    difference = remaining_poly.difference(cadastral_poly)
                    
                    if difference.is_empty:
                        # Объект полностью покрыт кадастровым участком
                        remaining_poly = None
                        break
                    else:
                        # Обновляем оставшуюся часть
                        remaining_poly = difference
            
            # Если осталась часть объекта после вычитания всех кадастровых участков
            if remaining_poly and not remaining_poly.is_empty:
                # Обрабатываем результат (может быть MultiPolygon)
                if remaining_poly.geom_type == 'Polygon':
                    violation_parts = [remaining_poly]
                elif remaining_poly.geom_type == 'MultiPolygon':
                    violation_parts = list(remaining_poly.geoms)
                else:
                    continue
                
                # Создаем отдельные нарушения для каждой части
                for part in violation_parts:
                    # Валидация геометрии
                    if not part.is_valid:
                        part = make_valid(part)
                        if part.is_empty:
                            continue
                    
                    # Обработка multipolygon после валидации
                    if part.geom_type == 'MultiPolygon':
                        sub_parts = list(part.geoms)
                    else:
                        sub_parts = [part]
                    
                    for sub_part in sub_parts:
                        if sub_part.area > self.config['min_violation_area']:
                            # Находим ближайший кадастровый участок для определения номера
                            closest_cadastral = None
                            min_distance = float('inf')
                            
                            for cadastral in cadastral_objects:
                                distance = sub_part.distance(cadastral['geometry'])
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_cadastral = cadastral
                            
                            # Создаем детальную информацию о нарушении
                            violation = {
                                'geometry': sub_part,
                                'cadastral_number': closest_cadastral['cadastral_number'] if closest_cadastral else 'Unknown',
                                'violation_area': sub_part.area,
                                'violation_area_ha': sub_part.area / 10000,
                                'original_object_area': obj_poly.area,
                                'centroid': sub_part.centroid.coords[0],
                                'bounds': sub_part.bounds,
                                'perimeter': sub_part.length,
                                'exterior_coords': list(sub_part.exterior.coords),
                                'distance_to_cadastral': min_distance,
                                'closest_cadastral_id': closest_cadastral.get('object_id') if closest_cadastral else None
                            }
                            
                            # Добавляем информацию о ближайшем кадастровом участке
                            if closest_cadastral:
                                violation['closest_cadastral_area'] = closest_cadastral.get('area_sqm', 0)
                                violation['closest_cadastral_centroid'] = closest_cadastral.get('centroid', (0, 0))
                            
                            violations.append(violation)
        
        # Дополнительная проверка: ищем объекты, которые полностью находятся вне кадастровых участков
        print("Дополнительный поиск объектов вне кадастровых участков...")
        processed_objects = set()
        
        # Собираем ID уже обработанных объектов (по геометрии)
        for violation in violations:
            for i, obj in enumerate(detected_objects):
                if obj['geometry'].equals(violation['geometry']) or obj['geometry'].contains(violation['geometry']):
                    processed_objects.add(i)
        
        for obj_idx, obj in enumerate(detected_objects):
            if obj_idx in processed_objects:
                continue  # Пропускаем уже обработанные объекты
                
            obj_poly = obj['geometry']
            
            # Проверяем, есть ли пересечения с кадастровыми участками
            has_intersection = False
            for cadastral in cadastral_objects:
                if obj_poly.intersects(cadastral['geometry']):
                    has_intersection = True
                    break
            
            # Если объект полностью вне кадастровых участков
            if not has_intersection and obj_poly.area > self.config['min_violation_area']:
                # Находим ближайший кадастровый участок
                closest_cadastral = None
                min_distance = float('inf')
                
                for cadastral in cadastral_objects:
                    distance = obj_poly.distance(cadastral['geometry'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_cadastral = cadastral
                
                # Создаем детальную информацию о нарушении
                violation = {
                    'geometry': obj_poly,
                    'cadastral_number': closest_cadastral['cadastral_number'] if closest_cadastral else 'Unknown',
                    'violation_area': obj_poly.area,
                    'violation_area_ha': obj_poly.area / 10000,
                    'original_object_area': obj_poly.area,
                    'centroid': obj_poly.centroid.coords[0],
                    'bounds': obj_poly.bounds,
                    'perimeter': obj_poly.length,
                    'exterior_coords': list(obj_poly.exterior.coords),
                    'distance_to_cadastral': min_distance,
                    'closest_cadastral_id': closest_cadastral.get('object_id') if closest_cadastral else None
                }
                
                # Добавляем информацию о ближайшем кадастровом участке
                if closest_cadastral:
                    violation['closest_cadastral_area'] = closest_cadastral.get('area_sqm', 0)
                    violation['closest_cadastral_centroid'] = closest_cadastral.get('centroid', (0, 0))
                
                violations.append(violation)
        
        print(f"Найдено {len(violations)} нарушений")
        
        # ДИАГНОСТИКА: Подробная информация о результатах анализа
        print(f"\n=== ДИАГНОСТИКА АНАЛИЗА НАРУШЕНИЙ ===")
        print(f"Всего обнаружено объектов: {len(detected_objects)}")
        print(f"Всего кадастровых участков: {len(cadastral_objects)}")
        print(f"Найдено нарушений: {len(violations)}")
        
        # Статистика по площадям
        total_detected_area = sum(obj['area_sqm'] for obj in detected_objects)
        total_violation_area = sum(v['violation_area'] for v in violations)
        print(f"Общая площадь обнаруженных объектов: {total_detected_area:.2f} м²")
        print(f"Общая площадь нарушений: {total_violation_area:.2f} м²")
        print(f"Процент нарушений: {(total_violation_area/total_detected_area*100) if total_detected_area > 0 else 0:.1f}%")
        
        # Статистика по размерам нарушений
        if violations:
            areas = [v['violation_area'] for v in violations]
            print(f"Минимальная площадь нарушения: {min(areas):.2f} м²")
            print(f"Максимальная площадь нарушения: {max(areas):.2f} м²")
            print(f"Средняя площадь нарушения: {sum(areas)/len(areas):.2f} м²")
        
        # Проверка: сколько объектов полностью внутри кадастра
        objects_inside_cadastre = 0
        objects_outside_cadastre = 0
        objects_intersecting_cadastre = 0
        
        for obj in detected_objects:
            obj_poly = obj['geometry']
            is_inside = False
            is_intersecting = False
            
            for cadastral in cadastral_objects:
                if cadastral['geometry'].contains(obj_poly):
                    is_inside = True
                    break
                elif cadastral['geometry'].intersects(obj_poly):
                    is_intersecting = True
            
            if is_inside:
                objects_inside_cadastre += 1
            elif is_intersecting:
                objects_intersecting_cadastre += 1
            else:
                objects_outside_cadastre += 1
        
        print(f"Объекты полностью внутри кадастра: {objects_inside_cadastre}")
        print(f"Объекты пересекающие кадастр: {objects_intersecting_cadastre}")
        print(f"Объекты полностью вне кадастра: {objects_outside_cadastre}")
        print(f"===================================\n")
        
        return violations
    
    def save_to_shapefile(self, objects: List[Dict], filename: str, crs):
        """Сохранение объектов в Shapefile"""
        if not objects:
            print(f"Нет объектов для сохранения в {filename}")
            return
        
        geometries = [obj['geometry'] for obj in objects]
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        
        # Добавление атрибутов с короткими именами (ограничение Shapefile - 10 символов)
        if 'cadastral_number' in objects[0]:
            gdf['cad_num'] = [obj.get('cadastral_number', '') for obj in objects]
            gdf['area_m2'] = [obj.get('area_sqm', 0) for obj in objects]
        elif 'violation_area' in objects[0]:
            # Для нарушений
            gdf['cad_num'] = [obj.get('cadastral_number', '') for obj in objects]
            gdf['viol_area'] = [obj.get('violation_area', 0) for obj in objects]
            gdf['orig_area'] = [obj.get('original_object_area', 0) for obj in objects]
        else:
            # Для обнаруженных объектов
            gdf['area_m2'] = [obj.get('area_sqm', 0) for obj in objects]
            gdf['centroid_x'] = [obj.get('centroid', (0, 0))[0] for obj in objects]
            gdf['centroid_y'] = [obj.get('centroid', (0, 0))[1] for obj in objects]
        
        output_path = os.path.join(self.config['output_dir'], filename)
        gdf.to_file(output_path)
        print(f"Сохранено в {output_path}")
    
    def create_visualization(self, geotiff_data: Dict, detected_objects: List[Dict], 
                           cadastral_objects: List[Dict], violations: List[Dict]):
        print("Создание визуализации...")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        if geotiff_data['data'].shape[0] >= 3:
            rgb_data = np.transpose(geotiff_data['data'][:3], (1, 2, 0))
            rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
            bounds = geotiff_data['bounds']
            ax.imshow(rgb_data, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        # Кадастровые границы (синие, очень тонкие)
        for cadastral in cadastral_objects:
            coords = list(cadastral['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='none', edgecolor='#0050ff', linewidth=0.5)
            ax.add_patch(poly)
        # Обнаруженные объекты (светло-желтый, очень тонкая оранжевая граница)
        for obj in detected_objects:
            coords = list(obj['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='#fff700', alpha=0.25, edgecolor='#ffb300', linewidth=0.5)
            ax.add_patch(poly)
        # Нарушения (ярко-красный, тонкая граница)
        for violation in violations:
            coords = list(violation['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='#ff0000', alpha=0.8, edgecolor='#800000', linewidth=0.7)
            ax.add_patch(poly)
            centroid = violation['centroid']
            ax.text(centroid[0], centroid[1], f"{violation['violation_area']:.1f} м²", fontsize=8, ha='center', va='center', color='white', weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
        bounds = geotiff_data['bounds']
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title('Анализ нарушений землепользования', fontsize=12)
        ax.set_xlabel('Координата X', fontsize=8)
        ax.set_ylabel('Координата Y', fontsize=8)
        
        # Добавление легенды
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#fff700', alpha=0.25, edgecolor='#ffb300', label='Обнаруженные объекты'),
            Patch(facecolor='none', edgecolor='#0050ff', label='Кадастровые границы'),
            Patch(facecolor='#ff0000', alpha=0.8, edgecolor='#800000', label='Нарушения')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        output_path = os.path.join(self.config['output_dir'], 'visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Визуализация сохранена в {output_path}")
    
    def generate_report(self, detected_objects: List[Dict], cadastral_objects: List[Dict], 
                       violations: List[Dict], geotiff_data: Dict):
        """Генерация отчета"""
        print("Генерация отчета...")
        
        # Подсчет статистики
        total_detected_area = sum(obj['area_sqm'] for obj in detected_objects)
        total_cadastral_area = sum(obj['area_sqm'] for obj in cadastral_objects)
        total_violation_area = sum(v['violation_area'] for v in violations)
        
        # Создание документа
        report_path = os.path.join(self.config['output_dir'], 'comprehensive_report.pdf')
        doc = SimpleDocTemplate(report_path, pagesize=A4)
        
        # Регистрация шрифта с поддержкой кириллицы
        try:
            # Пробуем загрузить системный шрифт с поддержкой кириллицы
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'C:/Windows/Fonts/dejavusans.ttf'))
            font_name = 'DejaVuSans'
        except:
            try:
                # Альтернативный шрифт
                pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
                font_name = 'Arial'
            except:
                # Если не удалось загрузить шрифт, используем стандартный
                font_name = 'Helvetica'
                print("Предупреждение: Не удалось загрузить шрифт с поддержкой кириллицы")
        
        styles = getSampleStyleSheet()
        
        # Содержимое отчета
        story = []
        
        # Заголовок
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=font_name
        )
        story.append(Paragraph("Отчет по анализу нарушений землепользования", title_style))
        story.append(Spacer(1, 20))
        
        # Информация о дате
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            fontName=font_name
        )
        story.append(Paragraph(f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}", date_style))
        story.append(Spacer(1, 30))
        
        # Общая статистика
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name
        )
        story.append(Paragraph("Общая статистика", heading_style))
        story.append(Spacer(1, 12))
        
        stats_data = [
            ['Параметр', 'Значение'],
            ['Обнаружено объектов', str(len(detected_objects))],
            ['Кадастровых участков', str(len(cadastral_objects))],
            ['Нарушений', str(len(violations))],
            ['Общая площадь объектов (м²)', f"{total_detected_area:.2f}"],
            ['Общая площадь кадастровых участков (м²)', f"{total_cadastral_area:.2f}"],
            ['Общая площадь нарушений (м²)', f"{total_violation_area:.2f}"],
            ['Площадь нарушений (га)', f"{total_violation_area/10000:.4f}"]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Детализация нарушений
        if violations:
            story.append(Paragraph("Детализация нарушений", heading_style))
            story.append(Spacer(1, 12))
            
            violation_data = [['№', 'Кадастровый номер', 'Площадь нарушения (м²)', 'Координаты центра']]
            for i, violation in enumerate(violations, 1):
                centroid = violation['centroid']
                violation_data.append([
                    str(i),
                    violation['cadastral_number'],
                    f"{violation['violation_area']:.2f}",
                    f"({centroid[0]:.6f}, {centroid[1]:.6f})"
                ])
            
            violation_table = Table(violation_data)
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(violation_table)
            story.append(Spacer(1, 20))
        
        # Информация о кадастровых участках
        if cadastral_objects:
            story.append(Paragraph("Кадастровые участки", heading_style))
            story.append(Spacer(1, 12))
            
            cadastral_data = [['№', 'Кадастровый номер', 'Площадь (м²)', 'Координаты центра']]
            for i, cadastral in enumerate(cadastral_objects, 1):
                centroid = cadastral['centroid']
                cadastral_data.append([
                    str(i),
                    cadastral['cadastral_number'],
                    f"{cadastral['area_sqm']:.2f}",
                    f"({centroid[0]:.6f}, {centroid[1]:.6f})"
                ])
            
            cadastral_table = Table(cadastral_data)
            cadastral_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(cadastral_table)
        
        # Сборка документа
        doc.build(story)
        print(f"Отчет сохранен в {report_path}")
    
    def export_to_excel(self, cadastral_objects: List[Dict], violations: List[Dict]):
        """Экспорт результатов в Excel (XLSX) с детальной информацией о координатах"""
        output_path = os.path.join(self.config['output_dir'], 'report.xlsx')
        
        # Лист 1: Сводная информация
        summary_data = []
        total_cadastral_area = sum(obj.get('area_sqm', 0) for obj in cadastral_objects)
        total_violation_area = sum(v.get('violation_area', 0) for v in violations)
        
        summary_data = [
            ['Параметр', 'Значение'],
            ['Дата анализа', datetime.now().strftime('%d.%m.%Y %H:%M')],
            ['Кадастровых участков', len(cadastral_objects)],
            ['Найдено нарушений', len(violations)],
            ['Общая площадь кадастровых участков, м²', round(total_cadastral_area, 2)],
            ['Общая площадь кадастровых участков, га', round(total_cadastral_area / 10000, 6)],
            ['Общая площадь нарушений, м²', round(total_violation_area, 2)],
            ['Общая площадь нарушений, га', round(total_violation_area / 10000, 6)],
            ['Процент нарушений от общей площади, %', round((total_violation_area / total_cadastral_area * 100) if total_cadastral_area > 0 else 0, 2)]
        ]
        summary_df = pd.DataFrame(summary_data, columns=['Параметр', 'Значение'])
        
        # Лист 2: Детальная информация о кадастровых участках
        cadastral_data = []
        for i, obj in enumerate(cadastral_objects, 1):
            # Основные данные
            row = {
                '№ п/п': i,
                'Кадастровый номер': obj.get('cadastral_number', ''),
                'Идентификатор объекта': obj.get('object_id', ''),
                'Площадь, м²': round(obj.get('area_sqm', 0), 2),
                'Площадь, га': round(obj.get('area_sqm', 0) / 10000, 6),
                'Центроид X': round(obj.get('centroid', (0, 0))[0], 6),
                'Центроид Y': round(obj.get('centroid', (0, 0))[1], 6),
                'Периметр, м': round(obj['geometry'].length, 2) if obj.get('geometry') else 0
            }
            
            # Координаты углов участка
            if obj.get('bounds'):
                bounds = obj['bounds']
                row.update({
                    'Мин X': round(bounds[0], 6),
                    'Мин Y': round(bounds[1], 6),
                    'Макс X': round(bounds[2], 6),
                    'Макс Y': round(bounds[3], 6)
                })
            
            # Координаты первых 10 точек контура
            if obj.get('exterior_coords'):
                coords = obj['exterior_coords'][:10]  # Берем первые 10 точек
                for j, (x, y) in enumerate(coords, 1):
                    row[f'Точка {j} X'] = round(x, 6)
                    row[f'Точка {j} Y'] = round(y, 6)
            
            # Дополнительные атрибуты из MID/MIF
            if obj.get('attributes'):
                for attr_name, attr_value in obj['attributes'].items():
                    if attr_value is not None:
                        row[f'Атрибут: {attr_name}'] = attr_value
            
            # WKT геометрии
            row['WKT геометрии'] = obj['geometry'].wkt if obj.get('geometry') else ''
            
            cadastral_data.append(row)
        
        cadastral_df = pd.DataFrame(cadastral_data)
        
        # Лист 3: Детальная информация о нарушениях
        violation_data = []
        for i, v in enumerate(violations, 1):
            row = {
                '№ нарушения': i,
                'Площадь нарушения, м²': round(v.get('violation_area', 0), 2),
                'Площадь нарушения, га': round(v.get('violation_area', 0) / 10000, 6),
                'Площадь исходного объекта, м²': round(v.get('original_object_area', 0), 2),
                'Центроид X': round(v.get('centroid', (0, 0))[0], 6),
                'Центроид Y': round(v.get('centroid', (0, 0))[1], 6),
                'Ближайший кадастровый номер': v.get('cadastral_number', ''),
                'Периметр нарушения, м': round(v['geometry'].length, 2) if v.get('geometry') else 0
            }
            
            # Координаты углов нарушения
            if v.get('geometry'):
                bounds = v['geometry'].bounds
                row.update({
                    'Мин X': round(bounds[0], 6),
                    'Мин Y': round(bounds[1], 6),
                    'Макс X': round(bounds[2], 6),
                    'Макс Y': round(bounds[3], 6)
                })
                
                # Координаты контура нарушения (первые 10 точек)
                try:
                    coords = list(v['geometry'].exterior.coords)[:10]
                    for j, (x, y) in enumerate(coords, 1):
                        row[f'Точка {j} X'] = round(x, 6)
                        row[f'Точка {j} Y'] = round(y, 6)
                except:
                    pass
            
            # WKT геометрии
            row['WKT геометрии'] = v['geometry'].wkt if v.get('geometry') else ''
            
            violation_data.append(row)
        
        violation_df = pd.DataFrame(violation_data)
        
        # Лист 4: Координаты всех точек кадастровых участков
        all_coords_data = []
        for i, obj in enumerate(cadastral_objects, 1):
            if obj.get('exterior_coords'):
                for j, (x, y) in enumerate(obj['exterior_coords'], 1):
                    all_coords_data.append({
                        'Кадастровый номер': obj.get('cadastral_number', ''),
                        'Номер точки': j,
                        'X': round(x, 6),
                        'Y': round(y, 6)
                    })
        
        all_coords_df = pd.DataFrame(all_coords_data)
        
        # Лист 5: Координаты всех точек нарушений
        violation_coords_data = []
        for i, v in enumerate(violations, 1):
            if v.get('geometry'):
                try:
                    coords = list(v['geometry'].exterior.coords)
                    for j, (x, y) in enumerate(coords, 1):
                        violation_coords_data.append({
                            '№ нарушения': i,
                            'Номер точки': j,
                            'X': round(x, 6),
                            'Y': round(y, 6)
                        })
                except:
                    pass
        
        violation_coords_df = pd.DataFrame(violation_coords_data)
        
        # Запись в Excel
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='1. Сводка', index=False)
                cadastral_df.to_excel(writer, sheet_name='2. Кадастровые участки', index=False)
                violation_df.to_excel(writer, sheet_name='3. Нарушения', index=False)
                all_coords_df.to_excel(writer, sheet_name='4. Координаты участков', index=False)
                violation_coords_df.to_excel(writer, sheet_name='5. Координаты нарушений', index=False)
                
                # Настройка форматирования
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                        
        except ImportError:
            print("Предупреждение: openpyxl не установлен, сохраняю в CSV")
            summary_df.to_csv(output_path.replace('.xlsx', '_summary.csv'), index=False, encoding='utf-8')
            cadastral_df.to_csv(output_path.replace('.xlsx', '_cadastral.csv'), index=False, encoding='utf-8')
            violation_df.to_csv(output_path.replace('.xlsx', '_violations.csv'), index=False, encoding='utf-8')
            all_coords_df.to_csv(output_path.replace('.xlsx', '_all_coords.csv'), index=False, encoding='utf-8')
            violation_coords_df.to_csv(output_path.replace('.xlsx', '_violation_coords.csv'), index=False, encoding='utf-8')
            return
        
        print(f'Детальный Excel-отчет сохранён: {output_path}')
        print(f'Отчет содержит {len(cadastral_objects)} кадастровых участков и {len(violations)} нарушений')
        
        # Вывод статистики
        print("\nСТАТИСТИКА ОТЧЕТА:")
        print(f"- Общая площадь кадастровых участков: {total_cadastral_area:.2f} м² ({total_cadastral_area/10000:.4f} га)")
        print(f"- Общая площадь нарушений: {total_violation_area:.2f} м² ({total_violation_area/10000:.4f} га)")
        print(f"- Процент нарушений: {(total_violation_area/total_cadastral_area*100) if total_cadastral_area > 0 else 0:.2f}%")
        
        # Информация о листах Excel
        print("\nСТРУКТУРА ОТЧЕТА:")
        print("1. Сводка - общая информация")
        print("2. Кадастровые участки - детальная информация с координатами")
        print("3. Нарушения - детальная информация о самозахватах")
        print("4. Координаты участков - все точки контуров участков")
        print("5. Координаты нарушений - все точки контуров нарушений")
    
    def deduplicate_instances(self, instances: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """Удаление дублей объектов в пересечениях чанков"""
        if not instances:
            return instances
        
        print(f"Дедупликация {len(instances)} инстансов...")
        
        # Создаем пространственный индекс
        geometries = [inst['geometry'] for inst in instances]
        tree = STRtree(geometries)
        
        duplicates = set()
        
        for i, instance in enumerate(instances):
            if i in duplicates:
                continue
                
            geom = instance['geometry']
            # Находим потенциальные дубли (пересекающиеся объекты)
            intersecting_indices = tree.query(geom)
            
            for j in intersecting_indices:
                if j <= i or j in duplicates:  # Пропускаем уже проверенные и самого себя
                    continue
                    
                other_geom = instances[j]['geometry']
                
                # Вычисляем площадь пересечения
                if geom.intersects(other_geom):
                    intersection = geom.intersection(other_geom)
                    if intersection.is_empty:
                        continue
                        
                    # Процент пересечения относительно меньшего объекта
                    min_area = min(geom.area, other_geom.area)
                    overlap_ratio = intersection.area / min_area
                    
                    if overlap_ratio > overlap_threshold:
                        # Это дубль - оставляем объект с большей площадью
                        if geom.area >= other_geom.area:
                            duplicates.add(j)
                        else:
                            duplicates.add(i)
                            break  # Текущий объект помечен как дубль
        
        # Удаляем дубли
        deduplicated = [inst for i, inst in enumerate(instances) if i not in duplicates]
        
        print(f"Удалено {len(duplicates)} дублей, осталось {len(deduplicated)} уникальных инстансов")
        return deduplicated
    
    def create_visualization_report(self, geotiff_data: Dict, detected_objects: List[Dict], 
                                  cadastral_objects: List[Dict], violations: List[Dict]):
        """Создание детальной визуализации отчета"""
        print("Создание детальной визуализации отчета...")
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        import matplotlib.patches as mpatches
        
        # Создаем фигуру с несколькими подграфиками
        fig = plt.figure(figsize=(20, 12))
        
        # Основная карта
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        
        # Подготовка фонового изображения
        if geotiff_data['data'].shape[0] >= 3:
            rgb_data = np.transpose(geotiff_data['data'][:3], (1, 2, 0))
            # Нормализация для отображения
            for i in range(3):
                channel = rgb_data[:, :, i]
                if channel.max() > channel.min():
                    rgb_data[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
            
            bounds = geotiff_data['bounds']
            ax_main.imshow(rgb_data, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], alpha=0.7)
        
        # Отображение кадастровых участков (синие контуры)
        for i, cadastral in enumerate(cadastral_objects):
            coords = list(cadastral['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='none', edgecolor='blue', linewidth=1, alpha=0.8)
            ax_main.add_patch(poly)
            
            # Подписи для первых нескольких участков
            if i < 5:
                centroid = cadastral['centroid']
                ax_main.text(centroid[0], centroid[1], cadastral['cadastral_number'][:8], 
                           fontsize=6, ha='center', va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # Отображение обнаруженных объектов (зеленые)
        for obj in detected_objects:
            coords = list(obj['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='green', alpha=0.3, edgecolor='darkgreen', linewidth=0.5)
            ax_main.add_patch(poly)
        
        # Отображение нарушений (красные)
        for i, violation in enumerate(violations):
            coords = list(violation['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='red', alpha=0.7, edgecolor='darkred', linewidth=1)
            ax_main.add_patch(poly)
            
            # Подписи площадей нарушений
            centroid = violation['centroid']
            area_text = f"{violation['violation_area']:.1f}м²"
            ax_main.text(centroid[0], centroid[1], area_text, 
                       fontsize=7, ha='center', va='center', color='white', weight='bold')
        
        # Настройка основной карты
        bounds = geotiff_data['bounds']
        ax_main.set_xlim(bounds[0], bounds[2])
        ax_main.set_ylim(bounds[1], bounds[3])
        ax_main.set_title('Карта нарушений землепользования', fontsize=14, weight='bold')
        ax_main.set_xlabel('Координата X (м)', fontsize=10)
        ax_main.set_ylabel('Координата Y (м)', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Легенда
        legend_elements = [
            mpatches.Patch(facecolor='green', alpha=0.3, edgecolor='darkgreen', label='Обнаруженные объекты'),
            mpatches.Patch(facecolor='none', edgecolor='blue', label='Кадастровые границы'),
            mpatches.Patch(facecolor='red', alpha=0.7, edgecolor='darkred', label='Нарушения')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # График распределения площадей нарушений
        ax_hist = plt.subplot2grid((3, 4), (0, 3))
        if violations:
            areas = [v['violation_area'] for v in violations]
            ax_hist.hist(areas, bins=min(10, len(violations)), color='red', alpha=0.7, edgecolor='black')
            ax_hist.set_title('Распределение площадей\nнарушений', fontsize=10, weight='bold')
            ax_hist.set_xlabel('Площадь (м²)', fontsize=8)
            ax_hist.set_ylabel('Количество', fontsize=8)
            ax_hist.grid(True, alpha=0.3)
        
        # Статистическая таблица
        ax_stats = plt.subplot2grid((3, 4), (1, 3))
        ax_stats.axis('off')
        
        # Подготовка статистических данных
        total_detected_area = sum(obj['area_sqm'] for obj in detected_objects)
        total_cadastral_area = sum(obj['area_sqm'] for obj in cadastral_objects)
        total_violation_area = sum(v['violation_area'] for v in violations)
        
        stats_text = [
            "СТАТИСТИКА АНАЛИЗА",
            "=" * 25,
            f"Обнаружено объектов: {len(detected_objects)}",
            f"Кадастровых участков: {len(cadastral_objects)}",
            f"Нарушений: {len(violations)}",
            "",
            "ПЛОЩАДИ:",
            f"Объекты: {total_detected_area:.1f} м²",
            f"         {total_detected_area/10000:.3f} га",
            f"Кадастр: {total_cadastral_area:.1f} м²",
            f"         {total_cadastral_area/10000:.3f} га",
            f"Нарушения: {total_violation_area:.1f} м²",
            f"          {total_violation_area/10000:.3f} га",
            "",
            f"Процент нарушений:",
            f"{(total_violation_area/total_detected_area*100) if total_detected_area > 0 else 0:.1f}%"
        ]
        
        for i, line in enumerate(stats_text):
            weight = 'bold' if line.startswith('СТАТИСТИКА') or line.startswith('ПЛОЩАДИ') else 'normal'
            size = 10 if weight == 'bold' else 8
            ax_stats.text(0.05, 0.95 - i*0.055, line, transform=ax_stats.transAxes, 
                         fontsize=size, weight=weight, verticalalignment='top',
                         fontfamily='monospace')
        
        # График типов нарушений по размерам
        ax_types = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        if violations:
            # Классифицируем нарушения по размерам
            small = [v for v in violations if v['violation_area'] < 100]
            medium = [v for v in violations if 100 <= v['violation_area'] < 1000]
            large = [v for v in violations if v['violation_area'] >= 1000]
            
            categories = ['Малые\n(<100м²)', 'Средние\n(100-1000м²)', 'Крупные\n(≥1000м²)']
            counts = [len(small), len(medium), len(large)]
            colors = ['orange', 'red', 'darkred']
            
            bars = ax_types.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            ax_types.set_title('Распределение нарушений по размерам', fontsize=11, weight='bold')
            ax_types.set_ylabel('Количество нарушений', fontsize=9)
            ax_types.grid(True, alpha=0.3, axis='y')
            
            # Подписи на столбцах
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax_types.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(count), ha='center', va='bottom', fontweight='bold')
        
        # Информация о дате и параметрах анализа
        ax_info = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        ax_info.axis('off')
        
        from datetime import datetime
        info_text = [
            f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"Размер изображения: {geotiff_data['width']}×{geotiff_data['height']} пикселей",
            f"Размер чанка: {self.config['chunk_size']} пикселей",
            f"Перекрытие: {self.config['overlap']} пикселей", 
            f"Порог уверенности: {self.config['score_threshold']}",
            f"Мин. площадь объекта: {self.config['min_polygon_area']} пикс²",
            f"Мин. площадь нарушения: {self.config['min_violation_area']} м²",
        ]
        
        for i, line in enumerate(info_text):
            ax_info.text(0.05, 0.9 - i*0.12, line, transform=ax_info.transAxes, 
                        fontsize=8, verticalalignment='top')
        
        # Сохранение
        plt.tight_layout()
        output_path = os.path.join(self.config['output_dir'], 'detailed_visualization_report.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Детальная визуализация сохранена: {output_path}")
        
        # Создаем также упрощенную версию
        self.create_simple_visualization(geotiff_data, detected_objects, cadastral_objects, violations)
    
    def create_simple_visualization(self, geotiff_data: Dict, detected_objects: List[Dict], 
                                  cadastral_objects: List[Dict], violations: List[Dict]):
        """Создание простой визуализации для быстрого просмотра"""
        print("Создание простой визуализации...")
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Фоновое изображение
        if geotiff_data['data'].shape[0] >= 3:
            rgb_data = np.transpose(geotiff_data['data'][:3], (1, 2, 0))
            for i in range(3):
                channel = rgb_data[:, :, i]
                if channel.max() > channel.min():
                    rgb_data[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
            
            bounds = geotiff_data['bounds']
            ax.imshow(rgb_data, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], alpha=0.6)
        
        # Кадастровые участки
        for cadastral in cadastral_objects:
            coords = list(cadastral['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='none', edgecolor='blue', linewidth=0.8, alpha=0.9)
            ax.add_patch(poly)
        
        # Обнаруженные объекты
        for obj in detected_objects:
            coords = list(obj['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='green', alpha=0.4, edgecolor='darkgreen', linewidth=0.5)
            ax.add_patch(poly)
        
        # Нарушения
        for violation in violations:
            coords = list(violation['geometry'].exterior.coords)
            poly = MplPolygon(coords, facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=1)
            ax.add_patch(poly)
        
        # Настройка
        bounds = geotiff_data['bounds']
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f'Анализ нарушений: {len(violations)} нарушений из {len(detected_objects)} объектов', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('X (м)', fontsize=10)
        ax.set_ylabel('Y (м)', fontsize=10)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label=f'Объекты ({len(detected_objects)})'),
            Patch(facecolor='none', edgecolor='blue', label=f'Кадастр ({len(cadastral_objects)})'),
            Patch(facecolor='red', alpha=0.8, edgecolor='darkred', label=f'Нарушения ({len(violations)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(self.config['output_dir'], 'simple_visualization.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Простая визуализация сохранена: {output_path}")

    def run_analysis(self):
        """Основной метод выполнения анализа"""
        print("Начинаю комплексный анализ...")
        
        # 1. Загрузка GeoTIFF
        print("1. Загрузка GeoTIFF...")
        geotiff_data = self.read_geotiff(self.config['input_geotiff'])
        if not geotiff_data:
            print("Ошибка: Не удалось загрузить GeoTIFF")
            return
        
        # 2. Обнаружение объектов нейросетью
        print("2. Обнаружение объектов...")
        detected_objects = self.detect_objects(geotiff_data)
        
        # 3. Дедупликация объектов (удаление дублей на пересечениях чанков)
        print("3. Дедупликация объектов...")
        detected_objects = self.deduplicate_instances(detected_objects, overlap_threshold=0.5)
        
        # 4. Загрузка кадастровых данных
        print("4. Загрузка кадастровых данных...")
        cadastral_objects = self.read_cadastral_data_enhanced(self.config['cadastral_data'], geotiff_data['crs'], geotiff_data['bounds'])
        
        # 5. Анализ нарушений
        print("5. Анализ нарушений...")
        violations = self.analyze_violations(detected_objects, cadastral_objects)
        
        # 6. Сохранение результатов в Shapefile
        print("6. Сохранение результатов...")
        self.save_to_shapefile(detected_objects, 'detected_objects.shp', geotiff_data['crs'])
        self.save_to_shapefile(cadastral_objects, 'cadastral_parcels.shp', geotiff_data['crs'])
        self.save_to_shapefile(violations, 'violations.shp', geotiff_data['crs'])
        
        # 7. Создание визуализации
        print("7. Создание визуализации...")
        self.create_visualization(geotiff_data, detected_objects, cadastral_objects, violations)
        self.create_visualization_report(geotiff_data, detected_objects, cadastral_objects, violations)
        
        # 8. Генерация отчета
        print("8. Генерация отчета...")
        self.generate_report(detected_objects, cadastral_objects, violations, geotiff_data)
        
        # 9. Экспорт в Excel
        print("9. Экспорт в Excel...")
        self.export_to_excel(cadastral_objects, violations)
        
        # Сохранение результатов в JSON
        results = {
            'detected_objects_count': len(detected_objects),
            'cadastral_objects_count': len(cadastral_objects),
            'violations_count': len(violations),
            'total_detected_area': sum(obj['area_sqm'] for obj in detected_objects),
            'total_cadastral_area': sum(obj['area_sqm'] for obj in cadastral_objects),
            'total_violation_area': sum(v['violation_area'] for v in violations),
            'analysis_date': datetime.now().isoformat()
        }
        
        json_path = os.path.join(self.config['output_dir'], 'analysis_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в {json_path}")
        print("Анализ завершен!")

def main():
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 