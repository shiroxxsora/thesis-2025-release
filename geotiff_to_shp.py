import os
import sys
import math
import numpy as np
from osgeo import gdal, osr
import torch
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import networkx as nx

# =====================
# ПАРАМЕТРЫ
# =====================
INPUT_TIFF = "geotiffs/input.tiff"  # путь к входному GeoTIFF
MODEL_CONFIG = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
MODEL_WEIGHTS = "models/pavel-01-07-25/model_final.pth"
OUTPUT_SHP = "output/polygons_from_mask.shp"
OUTPUT_KML = "output/polygons_from_mask.kml"
SCORE_THRESH = 0.5
MIN_POLY_POINTS = 5
MAX_POLY_POINTS = 20
MODEL_INPUT_SIZE = 1024
CHUNK_SIZE = 5000
OVERLAP = 1536

# =====================
# ФУНКЦИИ
# =====================
def setup_cfg(config_file, weights_path, score_thresh=0.5, num_classes=1):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def normalize_to_uint8(ch):
    if ch.dtype == np.uint8:
        return ch
    ch_min = float(np.nanmin(ch))
    ch_max = float(np.nanmax(ch))
    if np.isfinite(ch_min) and np.isfinite(ch_max) and ch_max > ch_min:
        ch_norm = (255.0 * (ch.astype(np.float32) - ch_min) / (ch_max - ch_min))
        return ch_norm.astype(np.uint8)
    else:
        return ch.astype(np.uint8)

def mask_to_polygons(mask, min_points=5, max_points=20):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        epsilon = 0.001 * cv2.arcLength(contour, True)
        for _ in range(10):
            approx = cv2.approxPolyDP(contour, epsilon, True)
            n_points = len(approx)
            if n_points < min_points:
                epsilon *= 0.7
            elif n_points > max_points:
                epsilon *= 1.5
            else:
                break
        if min_points <= len(approx) <= max_points:
            poly = approx.reshape(-1, 2)
            polygons.append(poly)
    return polygons

def pixel_to_geo(coords, transform):
    return [(
        transform[0] + x * transform[1] + y * transform[2],
        transform[3] + x * transform[4] + y * transform[5]
    ) for x, y in coords]

def read_geotiff_gdal(path):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Не удалось открыть {path}")
    arr = ds.ReadAsArray()
    transform = ds.GetGeoTransform()
    
    # Получаем SRS из dataset
    srs_from_dataset = ds.GetSpatialRef()
    crs_wkt = None
    
    if srs_from_dataset:
        try:
            # Экспортируем проекцию в Proj4
            proj4_candidate = srs_from_dataset.ExportToProj4()
            
            # Декодируем Proj4 строку
            if isinstance(proj4_candidate, bytes):
                try:
                    proj4_candidate = proj4_candidate.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        proj4_candidate = proj4_candidate.decode('cp1251')
                    except UnicodeDecodeError:
                        print(f"Предупреждение: Не удалось декодировать Proj4 строку")
                        proj4_candidate = None
            
            # Нормализуем через импорт/экспорт
            if proj4_candidate:
                temp_srs = osr.SpatialReference()
                if temp_srs.ImportFromProj4(proj4_candidate) == 0:
                    crs_wkt = temp_srs.ExportToWkt()
                    if isinstance(crs_wkt, bytes):
                        try:
                            crs_wkt = crs_wkt.decode('utf-8')
                        except UnicodeDecodeError:
                            print(f"Предупреждение: Не удалось декодировать WKT")
                            crs_wkt = None
                else:
                    print(f"Предупреждение: Не удалось импортировать Proj4")
            else:
                print(f"Предупреждение: Не удалось получить Proj4 строку")
        except Exception as e:
            print(f"Предупреждение: Ошибка при обработке проекции: {e}")
    
    # Получаем размер
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    height, width = arr.shape[1:]
    
    return arr, transform, crs_wkt, height, width

def main():
    img, transform, crs_wkt, height, width = read_geotiff_gdal(INPUT_TIFF)
    all_polygons_geo = []
    
    cfg = setup_cfg(MODEL_CONFIG, MODEL_WEIGHTS, SCORE_THRESH)
    predictor = DefaultPredictor(cfg)
    
    step_size = CHUNK_SIZE - OVERLAP
    num_chunks_y = math.ceil(height / step_size)
    num_chunks_x = math.ceil(width / step_size)

    for y_idx in range(num_chunks_y):
        for x_idx in range(num_chunks_x):
            print(f"Обработка чанка ({y_idx*num_chunks_x + x_idx + 1}/{num_chunks_x*num_chunks_y})")
            x_start, y_start = x_idx * step_size, y_idx * step_size
            x_end, y_end = min(x_start + CHUNK_SIZE, width), min(y_start + CHUNK_SIZE, height)
            
            if x_end - x_start <= 0 or y_end - y_start <= 0:
                continue

            chunk = img[:, y_start:y_end, x_start:x_end]
            img3 = chunk[:3] if chunk.shape[0] > 3 else chunk
            chans = [normalize_to_uint8(img3[i]) for i in range(img3.shape[0])]
            if len(chans) < 3:
                chans.extend([chans[-1]] * (3 - len(chans)))
            img_arr = np.stack(chans, axis=0)
            img_vis = np.transpose(img_arr, (1, 2, 0))
            
            img_for_predictor = cv2.resize(img_vis, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            img_for_predictor = cv2.cvtColor(img_for_predictor, cv2.COLOR_RGB2BGR)
            img_for_predictor = np.require(img_for_predictor, dtype=np.uint8, requirements=['C'])
            
            with torch.no_grad():
                outputs = predictor(img_for_predictor)
            
            instances = outputs["instances"].to("cpu")
            raw_masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else np.array([])
            
            for mask_pred in raw_masks:
                mask_resized = cv2.resize(mask_pred.astype(np.uint8), (x_end - x_start, y_end - y_start), interpolation=cv2.INTER_NEAREST)
                polys_in_chunk = mask_to_polygons(mask_resized, MIN_POLY_POINTS, MAX_POLY_POINTS)
                for poly_pixels in polys_in_chunk:
                    global_poly_pixels = poly_pixels + np.array([x_start, y_start])
                    geo_coords = pixel_to_geo(global_poly_pixels, transform)
                    if len(geo_coords) >= 3:
                        all_polygons_geo.append(Polygon(geo_coords))

    if not all_polygons_geo:
        print("Полигоны не найдены")
        return

    print(f"Найдено {len(all_polygons_geo)} полигонов-фрагментов. Объединение...")
    
    gdf = gpd.GeoDataFrame(geometry=all_polygons_geo, crs=crs_wkt)
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.is_empty].reset_index(drop=True)

    if gdf.empty:
        print("Полигоны не найдены после очистки.")
        return

    touching_pairs = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    touching_pairs = touching_pairs[touching_pairs.index < touching_pairs['index_right']]

    graph = nx.from_edgelist(touching_pairs[['index_right']].reset_index().values)
    graph.add_nodes_from(gdf.index)

    connected_components = list(nx.connected_components(graph))
    
    merged_polys = []
    for component in connected_components:
        geoms_to_merge = gdf.iloc[list(component)]['geometry'].tolist()
        merged = unary_union(geoms_to_merge)
        if isinstance(merged, Polygon):
            merged_polys.append(merged)
        elif isinstance(merged, MultiPolygon):
            merged_polys.extend(list(merged.geoms))

    final_gdf = gpd.GeoDataFrame(geometry=merged_polys, crs=crs_wkt)
    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    final_gdf.to_file(OUTPUT_SHP)
    print(f"Сохранено {len(final_gdf)} объединённых полигонов в {OUTPUT_SHP}")

    # --- Экспорт в KML ---
    if not final_gdf.empty:
        print("Экспорт в KML...")
        try:
            # KML требует систему координат WGS 84 (EPSG:4326)
            gdf_kml = final_gdf.to_crs("EPSG:4326")
            
            # Сохраняем в KML
            gdf_kml.to_file(OUTPUT_KML, driver='KML')
            print(f"Сохранено {len(gdf_kml)} полигонов в {OUTPUT_KML}")
        except Exception as e:
            print(f"Не удалось сохранить в KML. Убедитесь, что у вас установлен драйвер 'libkml'. Ошибка: {e}")

if __name__ == "__main__":
    main() 