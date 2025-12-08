import os
import sys
import json
import math
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import transform as shapely_transform
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from osgeo import gdal, ogr, osr
from rasterio.transform import Affine
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')

gdal.UseExceptions()

PATHS_CONFIG = {
    'input_geotiff': "geotiffs/input.tiff",
    'cadastral_data': "cadastr/ЗУ все2.MIF",
    'detectron_config': "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    'model_weights': "models/pavel-01-07-25/model_final.pth",
    'output_dir': "simple_output",
}

PROCESSING_CONFIG = {
    'chunk_size': 5000,
    'overlap': 1536,
    'model_input_size': 1024,
    'score_threshold': 0.5,
    'min_polygon_area': 500.0,
    'min_violation_area': 1.5,
}

def setup_cfg(config_file: str, weights_path: str, score_thresh: float = 0.5, num_classes: int = 1):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def normalize_to_uint8(ch: np.ndarray) -> np.ndarray:
    if ch.dtype == np.uint8:
        return ch
    
    ch_min = float(np.nanmin(ch))
    ch_max = float(np.nanmax(ch))
    
    if np.isfinite(ch_min) and np.isfinite(ch_max) and ch_max > ch_min:
        ch_norm = (255.0 * (ch.astype(np.float32) - ch_min) / (ch_max - ch_min))
        return ch_norm.astype(np.uint8)
    else:
        return ch.astype(np.uint8)

def process_mask_to_polygon(mask: np.ndarray, min_area: float = 500.0) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            contour = contour.reshape(-1, 2)
            if len(contour) >= 3:
                polygons.append(contour)
    
    return polygons

def process_image_chunk(image_chunk: np.ndarray, predictor, model_input_dims: Tuple[int, int]) -> np.ndarray:
    if len(image_chunk.shape) == 3 and image_chunk.shape[0] >= 3:
        image_rgb = np.transpose(image_chunk[:3], (1, 2, 0))
    else:
        image_rgb = image_chunk
    
    image_rgb = normalize_to_uint8(image_rgb)
    
    if len(image_rgb.shape) == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    
    height, width = image_rgb.shape[:2]
    target_height, target_width = model_input_dims
    
    if height != target_height or width != target_width:
        image_rgb = cv2.resize(image_rgb, (target_width, target_height))
    
    outputs = predictor(image_rgb)
    
    if "instances" in outputs:
        instances = outputs["instances"]
        if len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            combined_mask = np.zeros((target_height, target_width), dtype=np.uint8)
            
            for mask, score in zip(masks, scores):
                if score >= PROCESSING_CONFIG['score_threshold']:
                    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height))
                    combined_mask = np.maximum(combined_mask, mask_resized)
            
            return combined_mask
    
    return np.zeros((height, width), dtype=np.uint8)

def read_raster_with_gdal(file_path: str) -> Optional[Dict]:
    try:
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"Ошибка: Не удалось открыть файл {file_path}")
            return None
        
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        num_bands = dataset.RasterCount
        
        geotransform = dataset.GetGeoTransform()
        projection_wkt = dataset.GetProjection()
        
        data = []
        for band_idx in range(1, min(num_bands + 1, 4)):
            band = dataset.GetRasterBand(band_idx)
            band_data = band.ReadAsArray()
            data.append(band_data)
        
        if len(data) == 0:
            print("Ошибка: Не удалось прочитать данные из растра")
            return None
        
        if len(data) == 1:
            data = [data[0]] * 3
        
        data = np.array(data)
        
        dataset = None
        
        return {
            'width': width,
            'height': height,
            'data': data,
            'transform': geotransform,
            'projection_wkt': projection_wkt,
            'proj4_projection': projection_wkt
        }
    
    except Exception as e:
        print(f"Ошибка при чтении растра: {e}")
        return None

def read_cadastral_polygons_mif_mid(mif_path: str, raster_srs_wkt: str, raster_proj4: str) -> list:
    try:
        driver = ogr.GetDriverByName("MapInfo File")
        dataset = driver.Open(mif_path, 0)
        if dataset is None:
            print(f"Ошибка: Не удалось открыть кадастровый файл {mif_path}")
            return []
        
        layer = dataset.GetLayer(0)
        if layer is None:
            print("Ошибка: Не удалось получить слой из кадастрового файла")
            return []
        
        source_srs = layer.GetSpatialRef()
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(raster_srs_wkt)
        
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        
        polygons = []
        feature = layer.GetNextFeature()
        
        while feature:
            geometry = feature.GetGeometryRef()
            if geometry is not None:
                geometry.Transform(transform)
                wkt = geometry.ExportToWkt()
                try:
                    shapely_polygon = Polygon(wkt)
                    if shapely_polygon.is_valid and shapely_polygon.area > 0:
                        polygons.append(shapely_polygon)
                except:
                    pass
            feature = layer.GetNextFeature()
        
        dataset = None
        print(f"Загружено {len(polygons)} кадастровых полигонов")
        return polygons
    
    except Exception as e:
        print(f"Ошибка при чтении кадастровых данных: {e}")
        return []

def create_mask_from_polygons(polygons: list, raster_shape: tuple, raster_transform: tuple) -> np.ndarray:
    height, width = raster_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in polygons:
        if polygon.is_valid:
            coords = list(polygon.exterior.coords)
            pixel_coords = []
            
            for x, y in coords:
                pixel_x = int((x - raster_transform[0]) / raster_transform[1])
                pixel_y = int((y - raster_transform[3]) / raster_transform[5])
                
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    pixel_coords.append([pixel_x, pixel_y])
            
            if len(pixel_coords) >= 3:
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                cv2.fillPoly(mask, [pixel_coords], 1)
    
    return mask

def calculate_pixel_area_sqm(raster_transform: tuple) -> float:
    pixel_width = abs(raster_transform[1])
    pixel_height = abs(raster_transform[5])
    return pixel_width * pixel_height

def calculate_violation_areas(mask_after_subtraction: np.ndarray, pixel_area_sqm: float, min_violation_area_sqm: float = 1.5) -> Dict:
    contours, _ = cv2.findContours(mask_after_subtraction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    violations = []
    total_area_pixels = 0
    
    for i, contour in enumerate(contours):
        area_pixels = cv2.contourArea(contour)
        area_sqm = area_pixels * pixel_area_sqm
        
        if area_sqm >= min_violation_area_sqm:
            violations.append({
                'id': i + 1,
                'area_pixels': area_pixels,
                'area_sqm': area_sqm,
                'area_hectares': area_sqm / 10000.0
            })
            total_area_pixels += area_pixels
    
    total_area_sqm = total_area_pixels * pixel_area_sqm
    total_area_hectares = total_area_sqm / 10000.0
    
    return {
        'total_violations': len(violations),
        'total_area_pixels': total_area_pixels,
        'total_area_sqm': total_area_sqm,
        'total_area_hectares': total_area_hectares,
        'individual_violations': violations
    }

def visualize_results(output_path: str, raster_path: str, full_blended_mask: np.ndarray, 
                     mask_after_subtraction: np.ndarray, cadastral_polygons: list,
                     violation_areas: dict, pixel_area_sqm: float):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    with rasterio.open(raster_path) as src:
        image = src.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        image = normalize_to_uint8(image)
    
    ax1.imshow(image)
    ax1.set_title('Обнаруженные объекты', fontsize=14, fontweight='bold')
    
    contours, _ = cv2.findContours(full_blended_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) * pixel_area_sqm >= PROCESSING_CONFIG['min_violation_area']:
            contour = contour.reshape(-1, 2)
            polygon = MplPolygon(contour, facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
            ax1.add_patch(polygon)
    
    for polygon in cadastral_polygons:
        if polygon.is_valid:
            coords = list(polygon.exterior.coords)
            pixel_coords = []
            
            for x, y in coords:
                pixel_x = int((x - src.transform[0]) / src.transform[1])
                pixel_y = int((y - src.transform[3]) / src.transform[5])
                pixel_coords.append([pixel_x, pixel_y])
            
            if len(pixel_coords) >= 3:
                polygon_patch = MplPolygon(pixel_coords, facecolor='yellow', alpha=0.3, edgecolor='yellow', linewidth=1)
                ax1.add_patch(polygon_patch)
    
    ax2.imshow(image)
    ax2.set_title('Нарушения (вне кадастра)', fontsize=14, fontweight='bold')
    
    contours, _ = cv2.findContours(mask_after_subtraction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) * pixel_area_sqm >= PROCESSING_CONFIG['min_violation_area']:
            contour = contour.reshape(-1, 2)
            polygon = MplPolygon(contour, facecolor='blue', alpha=0.5, edgecolor='blue', linewidth=2)
            ax2.add_patch(polygon)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_pdf_report(output_path: str, results: dict, violation_areas: dict):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1
    )
    
    story.append(Paragraph("Отчет по анализу нарушений", title_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Общая статистика:", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    data = [
        ['Параметр', 'Значение'],
        ['Общее количество нарушений', str(violation_areas['total_violations'])],
        ['Общая площадь нарушений (кв.м)', f"{violation_areas['total_area_sqm']:.2f}"],
        ['Общая площадь нарушений (га)', f"{violation_areas['total_area_hectares']:.4f}"],
        ['Площадь обнаруженных объектов (кв.м)', f"{results['total_detected_area_sqm']:.2f}"],
        ['Площадь кадастровых участков (кв.м)', f"{results['cadastral_area_sqm']:.2f}"],
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    if violation_areas['individual_violations']:
        story.append(Paragraph("Детализация нарушений:", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        violation_data = [['№', 'Площадь (кв.м)', 'Площадь (га)']]
        for violation in violation_areas['individual_violations']:
            violation_data.append([
                str(violation['id']),
                f"{violation['area_sqm']:.2f}",
                f"{violation['area_hectares']:.4f}"
            ])
        
        violation_table = Table(violation_data)
        violation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(violation_table)
    
    doc.build(story)

def process_geotiff_simple(input_path: str, output_dir: str) -> Dict:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Чтение растра...")
    raster_data = read_raster_with_gdal(input_path)
    if raster_data is None:
        raise RuntimeError("Не удалось прочитать растр")
    
    width, height = raster_data['width'], raster_data['height']
    original_transform = raster_data['transform']
    print(f"Размер растра: {width}x{height}")
    
    full_blended_mask = np.zeros((height, width), dtype=np.uint8)
    
    print("Инициализация модели...")
    cfg = setup_cfg(PATHS_CONFIG['detectron_config'], PATHS_CONFIG['model_weights'])
    predictor = DefaultPredictor(cfg)
    print(f"Модель на устройстве: {cfg.MODEL.DEVICE}")
    
    step_size = PROCESSING_CONFIG['chunk_size'] - PROCESSING_CONFIG['overlap']
    num_chunks_y = math.ceil(height / step_size)
    num_chunks_x = math.ceil(width / step_size)
    
    print("Обработка чанков...")
    for y_idx in range(num_chunks_y):
        for x_idx in range(num_chunks_x):
            x_start, y_start = x_idx * step_size, y_idx * step_size
            x_end, y_end = min(x_start + PROCESSING_CONFIG['chunk_size'], width), min(y_start + PROCESSING_CONFIG['chunk_size'], height)
            
            if x_end - x_start <= 0 or y_end - y_start <= 0:
                continue
            
            print(f"Чанк ({x_idx+1}/{num_chunks_x}, {y_idx+1}/{num_chunks_y})")
            chunk_data = raster_data['data'][:, y_start:y_end, x_start:x_end]
            masks_in_chunk = process_image_chunk(chunk_data, predictor, (PROCESSING_CONFIG['model_input_size'], PROCESSING_CONFIG['model_input_size']))
            
            for local_mask in masks_in_chunk:
                full_blended_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    full_blended_mask[y_start:y_end, x_start:x_end], local_mask
                )
    
    print("Обработка кадастровых данных...")
    cadastral_polygons = []
    mask_after_subtraction = full_blended_mask.copy()
    
    if os.path.exists(PATHS_CONFIG['cadastral_data']):
        cadastral_polygons = read_cadastral_polygons_mif_mid(
            PATHS_CONFIG['cadastral_data'], 
            raster_data['projection_wkt'], 
            raster_data['proj4_projection']
        )
        
        if cadastral_polygons:
            cadastral_mask = create_mask_from_polygons(cadastral_polygons, (height, width), original_transform)
            mask_after_subtraction = np.where(cadastral_mask > 0, 0, full_blended_mask)
    
    print("Вычисление площадей...")
    pixel_area_sqm = calculate_pixel_area_sqm(original_transform)
    violation_areas = calculate_violation_areas(mask_after_subtraction, pixel_area_sqm)
    
    total_detected_area_pixels = np.sum(full_blended_mask > 0)
    total_detected_area_sqm = total_detected_area_pixels * pixel_area_sqm
    total_detected_area_hectares = total_detected_area_sqm / 10000.0
    
    cadastral_area_pixels = np.sum(cadastral_mask) if 'cadastral_mask' in locals() else 0
    cadastral_area_sqm = cadastral_area_pixels * pixel_area_sqm
    cadastral_area_hectares = cadastral_area_sqm / 10000.0
    
    print("Создание визуализации...")
    visualization_path = os.path.join(output_dir, "visualization.png")
    visualize_results(visualization_path, input_path, full_blended_mask, mask_after_subtraction, 
                     cadastral_polygons, violation_areas, pixel_area_sqm)
    
    print("Создание отчета...")
    report_path = os.path.join(output_dir, "report.pdf")
    create_pdf_report(report_path, {
        'total_detected_area_sqm': total_detected_area_sqm,
        'total_detected_area_hectares': total_detected_area_hectares,
        'cadastral_area_sqm': cadastral_area_sqm,
        'cadastral_area_hectares': cadastral_area_hectares,
    }, violation_areas)
    
    print("\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Обнаружено нарушений: {violation_areas['total_violations']}")
    print(f"Площадь нарушений: {violation_areas['total_area_sqm']:.2f} кв.м ({violation_areas['total_area_hectares']:.4f} га)")
    print(f"Площадь объектов: {total_detected_area_sqm:.2f} кв.м")
    print(f"Площадь кадастра: {cadastral_area_sqm:.2f} кв.м")
    
    return {
        'violation_areas': violation_areas,
        'total_detected_area_sqm': total_detected_area_sqm,
        'cadastral_area_sqm': cadastral_area_sqm,
        'visualization_path': visualization_path,
        'report_path': report_path
    }

if __name__ == "__main__":
    input_file = "geotiffs/input.tiff"
    output_directory = "simple_output"
    
    try:
        results = process_geotiff_simple(input_file, output_directory)
        print(f"\nРезультаты сохранены в папке: {output_directory}")
        print(f"Визуализация: {results['visualization_path']}")
        print(f"Отчет: {results['report_path']}")
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1) 