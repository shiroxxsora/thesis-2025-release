# Импорт стандартных библиотек Python
import os       # Работа с файловой системой и путями
import sys      # Системные функции, включая управление выходом из программы
import json     # Работа с JSON файлами для сохранения результатов
import math     # Математические функции для вычислений с чанками

# Импорт библиотек для обработки изображений и компьютерного зрения
import cv2      # OpenCV для обработки изображений и поиска контуров
import numpy as np  # NumPy для работы с многомерными массивами (матрицы изображений)

# Импорт библиотек для работы с геопространственными данными
import rasterio  # Основная библиотека для чтения геопривязанных растровых файлов
from rasterio.windows import Window  # Для чтения частей больших растров (чанкинг)
from rasterio.features import shapes # Преобразование растровых масок в векторные полигоны
from rasterio.warp import transform_geom  # Трансформация геометрии между системами координат

# Импорт Shapely для работы с геометрическими объектами
from shapely.geometry import Polygon  # Класс для представления полигонов
from shapely.validation import make_valid  # Исправление невалидных геометрий
from shapely.ops import transform as shapely_transform  # Трансформация координат в Shapely

# Импорт PyTorch для работы с нейронными сетями
import torch  # Основная библиотека PyTorch

# Импорт Detectron2 для сегментации объектов
from detectron2.config import get_cfg  # Загрузка конфигурации модели
from detectron2.engine import DefaultPredictor  # Класс для выполнения предсказаний

# Импорт GDAL/OGR для низкоуровневой работы с геоданными
from osgeo import gdal  # Чтение и запись растровых данных с геопривязкой
from osgeo import ogr   # Чтение векторных данных (кадастровые границы)
from osgeo import osr   # Работа с системами координат и проекциями

# Включение обработки исключений в GDAL для более информативных ошибок
gdal.UseExceptions()

# Импорт дополнительных компонентов rasterio
from rasterio.transform import Affine  # Класс для аффинных преобразований координат

# Импорт типов для аннотации типов (улучшает читаемость кода)
from typing import Optional, Dict, Tuple, List

# ============================================================================
# БЛОК КОНФИГУРАЦИИ ПУТЕЙ И ПАРАМЕТРОВ
# ============================================================================

# Основные пути к файлам и папкам (относительно расположения этого скрипта)
PATHS_CONFIG = {
    # Входные данные
    'input_geotiff': "geotiffs/input.tiff",           # Путь к входному GeoTIFF файлу
    'cadastral_data': "cadastr/ЗУ все2.MIF",          # Путь к кадастровым данным (MIF/MID)
    
    # Модель Detectron2
    'detectron_config': "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    'model_weights': "models/pavel-01-07-25/model_final.pth",  # Веса обученной модели
    
    # Выходные данные
    'output_dir': "output_results",                   # Папка для сохранения результатов
}

# Параметры обработки
PROCESSING_CONFIG = {
    'chunk_size': 5000,              # Размер чанка для обработки больших изображений (пиксели)
    'overlap': 1536,                 # Перекрытие между чанками (пиксели)
    'model_input_size': 1024,        # Размер входа модели (пиксели)
    'score_threshold': 0.5,          # Минимальная уверенность для обнаружения объектов (0.0-1.0)
    'min_polygon_area': 500.0,       # Минимальная площадь полигона (пиксели)
    'min_violation_area': 1.5,       # Минимальная площадь нарушения (кв.м)
}

# ============================================================================
# БЛОК ПОЛЬЗОВАТЕЛЬСКИХ НАСТРОЕК
# ============================================================================

# Пользователь может изменить эти настройки перед запуском
USER_CONFIG = {
    # Изменить входной файл (раскомментировать и указать нужный путь)
    # 'custom_input': "geotiffs/my_custom_input.tiff",
    
    # Изменить модель (раскомментировать и указать нужный путь)  
    # 'custom_model': "models/my_model/model_final.pth",
    
    # Настройки качества обработки
    'high_quality_mode': False,       # True - медленнее, но точнее; False - быстрее
    'gpu_acceleration': True,         # Использовать GPU если доступен
    
    # Настройки вывода
    'verbose_output': True,           # Подробный вывод в консоль
    'save_intermediate': False,       # Сохранять промежуточные результаты
}

def setup_cfg(config_file: str, weights_path: str, score_thresh: float = 0.5, num_classes: int = 1):
    """
    Инициализирует и настраивает конфигурацию для модели Detectron2.
    
    Эта функция создает объект конфигурации Detectron2 и настраивает его параметры
    для выполнения сегментации объектов на спутниковых изображениях.
    
    Args:
        config_file (str): Путь к YAML файлу с архитектурой модели
        weights_path (str): Путь к файлу с весами обученной модели  
        score_thresh (float): Минимальная уверенность для обнаружения объекта (0.0-1.0)
        num_classes (int): Количество классов объектов для распознавания
    
    Returns:
        DictConfig: Настроенная конфигурация Detectron2
    """
    # Создаем базовый объект конфигурации Detectron2
    cfg = get_cfg()
    
    # Загружаем архитектуру модели из YAML файла (например, Mask R-CNN)
    cfg.merge_from_file(config_file)
    
    # Указываем путь к файлу с весами нашей дообученной модели
    cfg.MODEL.WEIGHTS = weights_path
    
    # Устанавливаем порог уверенности - объекты с меньшей уверенностью отбрасываются
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    
    # Указываем количество классов (в нашем случае 1 - только здания/сооружения)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Автоматически выбираем устройство: GPU если доступен, иначе CPU
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Возвращаем настроенную конфигурацию
    return cfg

def normalize_to_uint8(ch: np.ndarray) -> np.ndarray:
    """
    Приводит одномерный массив к типу uint8, масштабируя от min..max к 0..255.
    
    Эта функция критически важна для подготовки данных растра для нейросети,
    так как модели обучены на изображениях в диапазоне 0-255.
    
    Спутниковые снимки часто имеют значения в диапазоне 0-65535 (16-bit) или другие,
    поэтому требуется нормализация для корректной работы модели.
    
    Args:
        ch (np.ndarray): Входной канал изображения (2D массив)
    
    Returns:
        np.ndarray: Нормализованный канал в диапазоне 0-255 (uint8)
    """
    # Если данные уже в формате uint8, возвращаем как есть
    if ch.dtype == np.uint8:
        return ch
    
    # Находим минимальное значение в канале, игнорируя NaN значения
    ch_min = float(np.nanmin(ch))
    
    # Находим максимальное значение в канале, игнорируя NaN значения  
    ch_max = float(np.nanmax(ch))
    
    # Проверяем, что значения корректны и есть динамический диапазон
    if np.isfinite(ch_min) and np.isfinite(ch_max) and ch_max > ch_min:
        # Выполняем линейную нормализацию: (значение - мин) / (макс - мин) * 255
        # Сначала приводим к float32 для избежания переполнения
        ch_norm = (255.0 * (ch.astype(np.float32) - ch_min) / (ch_max - ch_min))
        
        # Приводим результат к uint8 (автоматически обрезается до 0-255)
        return ch_norm.astype(np.uint8)
    else:
        # Если min == max или есть некорректные значения, просто приводим к uint8
        return ch.astype(np.uint8)

def process_mask_to_polygon(mask: np.ndarray, min_area: float = 500.0) -> List[np.ndarray]:
    """
    Извлекает полигоны из бинарной маски с фильтрацией и упрощением.
    
    Эта функция преобразует пиксельную маску объектов в векторные полигоны,
    подходящие для дальнейшего анализа и экспорта в геоформаты.
    
    Процесс включает:
    1. Поиск контуров объектов в маске
    2. Фильтрацию слишком маленьких объектов
    3. Упрощение геометрии для уменьшения количества точек
    4. Проверку валидности полигонов
    
    Args:
        mask (np.ndarray): Бинарная маска (0 или 1) размером HxW
        min_area (float): Минимальная площадь полигона в пикселях
    
    Returns:
        List[np.ndarray]: Список полигонов, каждый как массив точек (N, 2)
    """
    # Находим контуры объектов в бинарной маске
    # RETR_EXTERNAL - только внешние контуры (без дырок)
    # CHAIN_APPROX_SIMPLE - сжимает горизонтальные, вертикальные и диагональные сегменты
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Список для хранения валидных полигонов
    polygons = []
    
    # Обрабатываем каждый найденный контур
    for contour in contours:
        # Вычисляем площадь контура в пикселях
        area = cv2.contourArea(contour)
        
        # Пропускаем слишком маленькие объекты (могут быть шумом)
        if area < min_area:
            continue

        # Вычисляем параметр для упрощения геометрии (0.075% от периметра)
        # Чем больше epsilon, тем сильнее упрощение
        epsilon = 0.00075 * cv2.arcLength(contour, True)
        
        # Упрощаем контур методом Дугласа-Пекера для уменьшения количества точек
        # True означает, что контур замкнутый
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Преобразуем из формата OpenCV (N, 1, 2) в обычный массив (N, 2)
        polygon = simplified.reshape(-1, 2)
        
        # Проверяем, что полигон имеет минимум 3 точки (треугольник)
        if len(polygon) >= 3:
            polygons.append(polygon)

    # Возвращаем список обработанных полигонов
    return polygons

def process_image_chunk(image_chunk: np.ndarray, predictor, model_input_dims: Tuple[int, int]) -> np.ndarray:
    """
    Обрабатывает часть изображения через нейронную сеть Detectron2.
    
    Эта функция является ядром системы - она выполняет сегментацию объектов
    на отдельном фрагменте большого спутникового снимка.
    
    Алгоритм работы:
    1. Подготовка данных: извлечение RGB каналов и нормализация
    2. Масштабирование под размер входа модели (обычно 1024x1024)
    3. Выполнение предсказания нейронной сетью
    4. Масштабирование результатов обратно к исходному размеру чанка
    
    Args:
        image_chunk (np.ndarray): Фрагмент изображения формата (C, H, W)
        predictor: Инициализированный предиктор Detectron2
        model_input_dims (Tuple[int, int]): Размер входа модели (ширина, высота)
    
    Returns:
        np.ndarray: Массив масок найденных объектов (N, H_original, W_original)
    """
    # Сохраняем оригинальные размеры чанка для масштабирования результатов обратно
    original_chunk_height, original_chunk_width = image_chunk.shape[1], image_chunk.shape[2]

    # Берем только первые 3 канала (RGB) из многоканального изображения
    # Спутниковые снимки могут иметь больше каналов (NIR, SWIR и т.д.)
    if image_chunk.shape[0] > 3:
        img3 = image_chunk[:3, :, :]  # Берем каналы 0, 1, 2 (обычно RGB)
    else:
        img3 = image_chunk  # Если каналов 3 или меньше, используем все

    # Нормализуем каждый канал к диапазону 0-255
    chans = []
    for i in range(img3.shape[0]):
        # Применяем нормализацию к каждому каналу отдельно
        chans.append(normalize_to_uint8(img3[i]))

    # Если каналов меньше 3, дублируем последний канал до получения 3 каналов
    # Это необходимо для корректной работы модели, обученной на RGB
    if len(chans) < 3:
        chans.extend([chans[-1]] * (3 - len(chans)))

    # Объединяем каналы в трехмерный массив (3, H, W)
    img_arr = np.stack(chans, axis=0)
    
    # Переводим из формата (C, H, W) в (H, W, C) для OpenCV и Detectron2
    img_vis = np.transpose(img_arr, (1, 2, 0))  # Теперь HxWx3

    # ЭТАП 1: Масштабирование изображения для подачи в модель
    # Модель ожидает определенный размер входа (например, 1024x1024)
    img_for_predictor = cv2.resize(img_vis, model_input_dims, interpolation=cv2.INTER_LINEAR)
    
    # Конвертируем из RGB в BGR (OpenCV и Detectron2 ожидают BGR)
    img_for_predictor = cv2.cvtColor(img_for_predictor, cv2.COLOR_RGB2BGR)
    
    # Обеспечиваем непрерывность массива в памяти (требование Detectron2)
    img_for_predictor = np.require(img_for_predictor, dtype=np.uint8, requirements=['C'])

    # ЭТАП 2: Выполнение предсказания нейронной сетью
    # Отключаем вычисление градиентов для экономии памяти (только инференс)
    with torch.no_grad():
        # Основной вызов модели - получаем обнаруженные объекты
        outputs = predictor(img_for_predictor)
    
    # Переносим результаты на CPU для дальнейшей обработки
    instances = outputs["instances"].to("cpu")

    # Извлекаем маски предсказанных объектов
    # Если объекты найдены, получаем их маски, иначе пустой массив
    raw_masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else np.array([])

    # ЭТАП 3: Масштабирование масок обратно к оригинальному размеру чанка
    resized_masks = []
    for mask_pred in raw_masks:
        # Используем INTER_NEAREST для бинарных масок (сохраняет четкие границы)
        # Масштабируем каждую маску с размера модели обратно к размеру чанка
        resized_mask = cv2.resize(
            mask_pred.astype(np.uint8), 
            (original_chunk_width, original_chunk_height), 
            interpolation=cv2.INTER_NEAREST
        )
        resized_masks.append(resized_mask)

    # Возвращаем массив масок или пустой массив, если объекты не найдены
    return np.array(resized_masks) if resized_masks else np.array([])

def read_raster_with_gdal(file_path: str) -> Optional[Dict]:
    """
    Чтение растрового файла с использованием GDAL и извлечение его метаданных и данных.
    
    Эта функция является входной точкой для загрузки спутниковых снимков.
    Она не только читает пиксельные данные, но и извлекает критически важную
    информацию о геопривязке и системе координат.
    
    Особое внимание уделяется обработке локальных систем координат (МСК),
    которые часто используются в российской геодезии.
    
    Args:
        file_path (str): Путь к GeoTIFF файлу
    
    Returns:
        Optional[Dict]: Словарь с данными растра и метаданными или None при ошибке
    """
    dataset = None  # Переменная для GDAL dataset (будет освобождена в finally)
    try:
        # Открываем растровый файл через GDAL
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"Не удалось открыть файл: {file_path}")
            return None

        # Извлекаем основные характеристики растра
        width = dataset.RasterXSize        # Ширина в пикселях
        height = dataset.RasterYSize       # Высота в пикселях  
        num_bands = dataset.RasterCount    # Количество каналов (обычно 3-4 для спутниковых снимков)
        
        # Читаем все пиксельные данные в память как numpy массив
        # Формат: (bands, height, width) для многоканальных изображений
        data = dataset.ReadAsArray()
        
        # Получаем геотрансформацию - связь между пиксельными и географическими координатами
        # Формат: [x_origin, pixel_width, rotation_x, y_origin, rotation_y, pixel_height]
        transform = dataset.GetGeoTransform()

        # Переменные для хранения информации о проекции
        projection_wkt = None       # WKT (Well-Known Text) представление проекции
        proj4_projection = None     # Proj4 строка проекции
        
        # Получаем информацию о системе координат из файла
        srs_from_dataset = dataset.GetSpatialRef()
        if srs_from_dataset:
            try:
                # КРИТИЧЕСКИЙ БЛОК: Обработка проекций (особенно МСК)
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
                                print(f"Предупреждение: Не удалось декодировать WKT после ImportFromProj4. Используем значение по умолчанию.")
                                projection_wkt = None
                    else:
                        print(f"Предупреждение: Не удалось импортировать Proj4 в SRS объект. Проекция не будет определена.")
                        projection_wkt = None
                else:
                    print(f"Предупреждение: Не удалось получить Proj4 строку из проекции растра. Проекция не будет определена.")
                    projection_wkt = None
            except Exception as e:
                print(f"Предупреждение: Ошибка при обработке проекции (Proj4/WKT): {e}. Проекция не будет определена.")
                projection_wkt = None

        # КРИТИЧЕСКАЯ ПРОВЕРКА: Если проекция не определена, останавливаем обработку
        if not projection_wkt:
            # Без корректной проекции невозможно выполнить точный анализ нарушений
            # Особенно критично для местных систем координат (МСК)
            print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: Не удалось определить или импортировать проекцию входного растра ({file_path}).")
            print("Это может быть связано с тем, что растр использует локальную систему координат (например, МСК 03),")
            print("которая не может быть автоматически распознана GDAL.")
            print("Пожалуйста, убедитесь, что ваш GeoTIFF имеет корректно определенную проекцию.")
            print("Если вы используете МСК 03, вам может потребоваться явно указать её WKT/Proj4.")
            return None # Возвращаем None, чтобы сигнализировать об ошибке
        else:
            # Выводим информацию об успешно определенной проекции
            print(f"Определена проекция растра (начало): {projection_wkt[:50]}...")
            print(f"Proj4 проекции растра: {proj4_projection}")

        # Вычисляем географические границы растра в его системе координат
        # Используем геотрансформацию для преобразования пиксельных координат
        min_x_raster = transform[0]                           # Левая граница (X минимум)
        max_y_raster = transform[3]                           # Верхняя граница (Y максимум)  
        max_x_raster = transform[0] + width * transform[1]    # Правая граница (X максимум)
        min_y_raster = transform[3] + height * transform[5]   # Нижняя граница (Y минимум)
        print(f"Границы растра в его CRS: X=[{min_x_raster:.2f}, {max_x_raster:.2f}], Y=[{min_y_raster:.2f}, {max_y_raster:.2f}]")

        # Выводим детальную информацию о геотрансформации
        print('GeoTransform:', transform)
        
        # Вычисляем размер пикселя в единицах системы координат (обычно метры)
        pixel_width = abs(transform[1])   # Ширина пикселя (абсолютное значение)
        pixel_height = abs(transform[5])  # Высота пикселя (абсолютное значение)
        print('Pixel width:', pixel_width)
        print('Pixel height:', pixel_height)

        # Возвращаем словарь со всеми загруженными данными и метаданными
        return {
            'width': width,                      # Ширина растра в пикселях
            'height': height,                    # Высота растра в пикселях
            'num_bands': num_bands,              # Количество каналов
            'data': data,                        # Массив пиксельных данных
            'transform': transform,              # Геотрансформация
            'projection_wkt': projection_wkt,    # WKT представление проекции
            'proj4_projection': proj4_projection # Proj4 строка проекции
        }
    except Exception as e:
        # Обработка любых непредвиденных ошибок при чтении файла
        print(f"Ошибка при чтении растра: {e}")
        return None
    finally:
        # Освобождаем ресурсы GDAL (важно для предотвращения утечек памяти)
        if dataset:
            dataset = None

def save_mask_to_geotiff(mask: np.ndarray, output_path: str, transform: tuple, projection_wkt: str) -> None:
    """
    Сохранение маски в GeoTIFF с сохранением геопривязки.
    """
    dataset = None
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        driver = gdal.GetDriverByName('GTiff')

        dataset = driver.Create(
            output_path,
            mask.shape[1],
            mask.shape[0],
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )

        if dataset is None:
            raise RuntimeError("Не удалось создать выходной файл")

        dataset.SetGeoTransform(transform)

        srs = osr.SpatialReference()
        try:
            srs.ImportFromWkt(projection_wkt)
        except Exception as e:
            print(f"Предупреждение: Не удалось импортировать проекцию из WKT в save_mask_to_geotiff: {e}. Используем EPSG:32648.")
            srs.ImportFromEPSG(32648)

        dataset.SetProjection(srs.ExportToWkt())

        band = dataset.GetRasterBand(1)
        band.WriteArray(mask)
        band.SetNoDataValue(0)

        print(f"Маска сохранена в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении маски: {e}")
        raise
    finally:
        if dataset:
            dataset = None

def visualize_polygons_on_raster(output_path: str, raster_path: str, blended_mask_path: str, 
                                 raster_projection_wkt: str, raster_projection_proj4: str, # Добавил proj4 для кадастра
                                 mask_after_subtraction: Optional[np.ndarray] = None, 
                                 cadastral_polygons: Optional[List[Polygon]] = None,
                                 violation_areas: Optional[Dict] = None) -> None:
    """
    Создает визуализацию.
    - Накладывает растр.
    - Накладывает полупрозрачную маску всех обнаружений (желтый).
    - Рисует контуры кадастровых границ (красный).
    - Обводит синим контуры областей, оставшихся после вычитания кадастра.
    - Добавляет подписи площадей нарушений на PNG (только для PNG).
    """
    raster_dataset = None
    blended_mask_dataset = None
    cadastr_layer = None # Добавил для очистки в finally
    cadastr_datasource = None # Добавил для очистки в finally

    try:
        gdal.AllRegister()
        ogr.RegisterAll()

        raster_dataset = gdal.Open(raster_path)
        if raster_dataset is None:
            raise RuntimeError(f"Не удалось открыть исходный растр: {raster_path}")

        original_raster_transform = raster_dataset.GetGeoTransform()
        raster_width = raster_dataset.RasterXSize
        raster_height = raster_dataset.RasterYSize

        rgb = np.zeros((raster_height, raster_width, 3), dtype=np.uint8)
        for i in range(min(3, raster_dataset.RasterCount)):
            band = raster_dataset.GetRasterBand(i + 1)
            band_array = band.ReadAsArray()
            rgb[:, :, i] = normalize_to_uint8(band_array)

        print("Исходное RGB изображение загружено для визуализации.")

        # --- НАЛОЖЕНИЕ ПОЛУПРОЗРАЧНОЙ МАСКИ ВСЕХ ОБНАРУЖЕНИЙ (светло-желтый) ---
        if os.path.exists(blended_mask_path):
            blended_mask_dataset = gdal.Open(blended_mask_path)
            if blended_mask_dataset is not None:
                blended_mask = blended_mask_dataset.GetRasterBand(1).ReadAsArray()
                # Используем светло-желтый цвет для общей маски обнаружения (RGB: 255, 255, 0)
                mask_overlay_color = np.array([255, 255, 0], dtype=np.uint8) # RGB: светло-желтый
                alpha = 0.3
                rgb_copy = rgb.copy().astype(np.float32)
                mask_pixels = blended_mask > 0
                rgb_copy[mask_pixels] = (rgb_copy[mask_pixels] * (1 - alpha) + mask_overlay_color * alpha)
                rgb = rgb_copy.astype(np.uint8)
                print(f"Объединенная маска наложена на RGB изображение с прозрачностью {alpha*100:.0f}%.")
        else:
            print(f"Файл объединенной маски не найден: {blended_mask_path}. Пропуск наложения маски.")

        # --- РИСОВАНИЕ СИНИХ КОНТУРОВ КАДАСТРОВЫХ ГРАНИЦ ---
        # Теперь используем кадастровые полигоны, которые уже трансформированы
        if cadastral_polygons:
            print("Рисование кадастровых границ...")
            # Получаем обратную трансформацию из географических координат в пиксельные
            gt = original_raster_transform
            inv_gt = gdal.InvGeoTransform(gt)
            
            for i, poly in enumerate(cadastral_polygons):
                if not poly.is_valid or poly.area == 0:
                    continue

                if poly.geom_type == 'MultiPolygon':
                    parts = list(poly.geoms)
                else:
                    parts = [poly]

                for part_poly in parts:
                    if part_poly.exterior:
                        coords = np.array(list(part_poly.exterior.coords))
                        
                        px = (coords[:, 0] * inv_gt[1] + coords[:, 1] * inv_gt[2] + inv_gt[0]).round().astype(int)
                        py = (coords[:, 0] * inv_gt[4] + coords[:, 1] * inv_gt[5] + inv_gt[3]).round().astype(int)
                        
                        pixel_coords = np.stack([px, py], axis=1)

                        # Обрезаем координаты до размеров изображения
                        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, raster_width - 1)
                        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, raster_height - 1)

                        if pixel_coords.size > 0 and len(pixel_coords) >= 3:
                            # RGB: синий цвет для кадастровых границ
                            cv2.polylines(rgb, [pixel_coords], isClosed=True, color=(0, 0, 255), thickness=2)
            print("Кадастровые границы нарисованы.")
        else:
            print("Кадастровые полигоны не предоставлены для отрисовки.")

        # --- КРАСНАЯ ОБВОДКА ОБЛАСТЕЙ ЗА ПРЕДЕЛАМИ КАДАСТРА И ИЗМЕНЕНИЕ ЦВЕТА МАСКИ ---
        violation_contours = []
        if mask_after_subtraction is not None:
            print("Поиск и отрисовка контуров областей за пределами кадастра...")
            # Используем только отфильтрованные контуры нарушений
            if violation_areas and 'contours' in violation_areas:
                violation_contours = violation_areas['contours']
            else:
                # fallback: ищем все контуры (старое поведение)
                contours, _ = cv2.findContours(mask_after_subtraction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                violation_contours = contours
            
            # Создаем маску для областей нарушений
            violation_mask = np.zeros_like(mask_after_subtraction, dtype=np.uint8)
            cv2.drawContours(violation_mask, violation_contours, -1, 1, -1)  # Заполняем области нарушений
            
            # Изменяем цвет маски в областях нарушений на красноватый
            if os.path.exists(blended_mask_path):
                blended_mask_dataset = gdal.Open(blended_mask_path)
                if blended_mask_dataset is not None:
                    blended_mask = blended_mask_dataset.GetRasterBand(1).ReadAsArray()
                    # Красноватый цвет для областей нарушений (RGB: 255, 100, 100)
                    violation_color = np.array([255, 100, 100], dtype=np.uint8) # RGB: красноватый
                    alpha_violation = 0.4
                    
                    # Применяем красноватый цвет только в областях нарушений
                    violation_pixels = (blended_mask > 0) & (violation_mask > 0)
                    rgb_copy = rgb.copy().astype(np.float32)
                    rgb_copy[violation_pixels] = (rgb_copy[violation_pixels] * (1 - alpha_violation) + violation_color * alpha_violation)
                    rgb = rgb_copy.astype(np.uint8)
                    print(f"Маска в областях нарушений изменена на красноватый цвет с прозрачностью {alpha_violation*100:.0f}%.")
            
            # RGB: красный цвет для контуров нарушений
            if len(violation_contours) > 0:
                cv2.drawContours(rgb, violation_contours, -1, (255, 0, 0), 3)
            print(f"Найдено и отрисовано {len(violation_contours)} контуров красным цветом.")
        else:
            print("Маска для красной обводки не предоставлена.")

        # Сохранение результата
        png_path = output_path.replace('.tif', '.png')
        
        # Создаем копию для PNG с подписями
        rgb_with_labels = rgb.copy()
        
        # --- ДОБАВЛЕНИЕ ПОДПИСЕЙ ПЛОЩАДЕЙ НА PNG ---
        if violation_areas and violation_contours and violation_areas.get('individual_violations'):
            print("Добавление подписей площадей нарушений на PNG...")
            
            # Настройки шрифта для OpenCV - увеличены для лучшей видимости
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Уменьшено для лучшего размещения
            font_thickness = 3
            font_color = (255, 255, 255)  # Белый цвет для текста
            outline_color = (0, 0, 0)     # Черный цвет для обводки
            
            # Получаем информацию о нарушениях
            violations = violation_areas['individual_violations']
            
            # Добавляем подписи для всех нарушений
            print(f"Добавление подписей для {len(violations)} нарушений...")
            print(f"Количество контуров: {len(violation_contours)}")
            
            for i in range(len(violations)):
                if i < len(violation_contours):
                    violation = violations[i]
                    contour = violation_contours[i]
                    
                    # Проверяем размер контура
                    if len(contour) < 3:
                        continue
                    
                    # Вычисляем центр контура для размещения текста
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Обрезаем координаты до размеров изображения с учетом размера текста
                        # Формируем текст с площадью
                        area_sqm = violation['area_sqm']
                        if area_sqm >= 1.0:
                            text = f"{area_sqm:.1f} sq.m"
                        else:
                            text = f"{area_sqm:.2f} sq.m"
                        
                        # Получаем размер текста
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                        
                        # Обрезаем координаты с учетом размера текста
                        min_x = text_width // 2 + 10
                        max_x = raster_width - text_width // 2 - 10
                        min_y = text_height + 10
                        max_y = raster_height - text_height - 10
                        
                        cx = max(min_x, min(cx, max_x))
                        cy = max(min_y, min(cy, max_y))
                        
                        # Рисуем черную обводку текста
                        cv2.putText(rgb_with_labels, text, (cx - text_width//2, cy), 
                                  font, font_scale, outline_color, font_thickness + 1)
                        
                        # Рисуем белый текст
                        cv2.putText(rgb_with_labels, text, (cx - text_width//2, cy), 
                                  font, font_scale, font_color, font_thickness)
                        
                        # Отладочная информация для первых 10 подписей
                        if i < 10:
                            print(f"  Подпись #{i+1}: '{text}' в позиции ({cx}, {cy})")
                    else:
                        if i < 10:
                            print(f"  Пропуск подписи #{i+1}: контур слишком мал")
            
            print(f"Добавлены подписи площадей для {len(violations)} нарушений.")

        # Сохраняем PNG с подписями
        # Конвертируем RGB в BGR для OpenCV
        rgb_bgr = cv2.cvtColor(rgb_with_labels, cv2.COLOR_RGB2BGR)
        cv2.imwrite(png_path, rgb_bgr)
        print(f"Визуализация сохранена как PNG: {png_path}")

        # Сохраняем GeoTIFF без подписей (используем оригинальный rgb)
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, rgb.shape[1], rgb.shape[0], 3, gdal.GDT_Byte, options=['COMPRESS=LZW'])
        if dataset is None: raise RuntimeError("Не удалось создать GeoTIFF файл")
        dataset.SetGeoTransform(original_raster_transform)
        srs = osr.SpatialReference(); srs.ImportFromWkt(raster_projection_wkt)
        dataset.SetProjection(srs.ExportToWkt())
        for i in range(3):
            dataset.GetRasterBand(i + 1).WriteArray(rgb[:, :, i])
        print(f"Визуализация сохранена как GeoTIFF: {output_path}")

    except Exception as e:
        print(f"Общая ошибка при создании визуализации: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if raster_dataset: raster_dataset = None
        if blended_mask_dataset: blended_mask_dataset = None
        if cadastr_layer: cadastr_layer = None
        if cadastr_datasource: cadastr_datasource = None

def read_cadastral_polygons_mif_mid(mif_path: str, raster_srs_wkt: str, raster_proj4: str) -> list:
    """
    Читает полигоны из MIF/MID файла и возвращает их в CRS растра (WKT).
    Возвращает список Shapely Polygon в CRS растра.
    Использует подход из старого рабочего кода.
    """
    driver = ogr.GetDriverByName('MapInfo File')
    datasource = driver.Open(mif_path, 0)
    if datasource is None:
        print(f"Ошибка: Не удалось открыть MIF/MID файл: {mif_path}")
        return []
    layer = datasource.GetLayer()

    print(f"=== ОТЛАДКА КАДАСТРОВОЙ ПРОЕКЦИИ ===")
    print(f"Проекция растра (целевая): {raster_srs_wkt[:100]}...")
    print(f"Proj4 растра: {raster_proj4}")
    
    # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ИСПОЛЬЗУЕМ ПОДХОД ИЗ СТАРОГО РАБОЧЕГО КОДА ---
    # Извлекаем towgs84 из проекции растра для использования в кадастровой
    raster_proj4_parts = dict(p.split('=', 1) for p in raster_proj4.split() if '=' in p)
    towgs84_params = raster_proj4_parts.get('+towgs84', '')
    ellps_param = raster_proj4_parts.get('+ellps', 'krass') # Используем эллипсоид растра

    # Создаем Proj4-строку для кадастра (как в старом коде)
    source_cadastr_proj4_str = (
        f"+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 "
        f"+x_0=4250000 +y_0=-5211057.63 " # Эти значения из старого рабочего кода
        f"+ellps={ellps_param} " # Используем эллипсоид растра
        f"+towgs84={towgs84_params} " # Используем towgs84 растра
        f"+units=m +no_defs"
    )

    source_srs_cadastr = osr.SpatialReference()
    try:
        if source_srs_cadastr.ImportFromProj4(source_cadastr_proj4_str) == 0:
            print(f"Исходная SRS кадастра (сконструирована): {source_srs_cadastr.ExportToProj4()}")
        else:
            print(f"Ошибка: Не удалось сконструировать SRS для кадастра. Используем SRS из файла.")
            source_srs_cadastr = layer.GetSpatialRef()
            if source_srs_cadastr is None:
                print("Внимание: Проекция кадастрового слоя не определена в файле MIF. Предполагается EPSG:32648.")
                source_srs_cadastr = osr.SpatialReference()
                source_srs_cadastr.ImportFromEPSG(32648)
    except Exception as e:
        print(f"Критическая ошибка при построении SRS для кадастра: {e}")
        source_srs_cadastr = layer.GetSpatialRef()
        if source_srs_cadastr is None:
            source_srs_cadastr = osr.SpatialReference()
            source_srs_cadastr.ImportFromEPSG(32648)

    target_srs_raster = osr.SpatialReference()
    target_srs_raster.ImportFromWkt(raster_srs_wkt)
    
    # Устанавливаем стратегию порядка осей для корректной работы в GDAL 3+
    source_srs_cadastr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs_raster.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform_cadastr_to_raster = osr.CoordinateTransformation(source_srs_cadastr, target_srs_raster)
    
    # Добавляем отладочную информацию о границах
    extent = layer.GetExtent()
    print(f"Границы кадастрового файла в его CRS: X=[{extent[0]:.2f}, {extent[1]:.2f}], Y=[{extent[2]:.2f}, {extent[3]:.2f}]")
    
    polygons = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None: continue
        
        # Клонируем геометрию перед трансформацией
        geom_clone = geom.Clone()
        
        # Добавляем отладочную информацию для первых нескольких полигонов
        if len(polygons) < 3:
            original_extent = geom_clone.GetEnvelope()
            print(f"Полигон {len(polygons)} до трансформации: X=[{original_extent[0]:.2f}, {original_extent[1]:.2f}], Y=[{original_extent[2]:.2f}, {original_extent[3]:.2f}]")
        
        # Трансформация геометрии в систему координат растра
        try:
            geom_clone.Transform(transform_cadastr_to_raster)
        except Exception as e:
            print(f"Предупреждение: Не удалось трансформировать геометрию кадастрового объекта. FID: {feature.GetFID()}")
            continue

        # Добавляем отладочную информацию для первых нескольких полигонов после трансформации
        if len(polygons) < 3:
            transformed_extent = geom_clone.GetEnvelope()
            print(f"Полигон {len(polygons)} после трансформации: X=[{transformed_extent[0]:.2f}, {transformed_extent[1]:.2f}], Y=[{transformed_extent[2]:.2f}, {transformed_extent[3]:.2f}]")

        try:
            # Обрабатываем Polygon и MultiPolygon
            if geom_clone.GetGeometryType() == ogr.wkbPolygon:
                poly = Polygon(json.loads(geom_clone.ExportToJson())['coordinates'][0])
                if poly.is_valid and poly.area > 0: polygons.append(poly)
            elif geom_clone.GetGeometryType() == ogr.wkbMultiPolygon:
                for i in range(geom_clone.GetGeometryCount()):
                    subgeom = geom_clone.GetGeometryRef(i)
                    poly = Polygon(json.loads(subgeom.ExportToJson())['coordinates'][0])
                    if poly.is_valid and poly.area > 0: polygons.append(poly)
        except Exception as e:
            print(f"Предупреждение: Ошибка при конвертации геометрии в полигон Shapely. FID: {feature.GetFID()}. Ошибка: {e}")
            continue

    print(f"Успешно прочитано и трансформировано {len(polygons)} кадастровых полигонов из файла {mif_path}")
    print("=== КОНЕЦ ОТЛАДКИ КАДАСТРОВОЙ ПРОЕКЦИИ ===")
    return polygons

def create_mask_from_polygons(polygons: list, raster_shape: tuple, raster_transform: tuple) -> np.ndarray:
    """
    Создаёт бинарную маску по списку полигонов (в CRS растра).
    """
    mask = np.zeros(raster_shape, dtype=np.uint8)
    h, w = raster_shape
    
    print(f"=== ОТЛАДКА СОЗДАНИЯ МАСКИ ===")
    print(f"Размер растра: {w}x{h}")
    print(f"Трансформация растра: {raster_transform}")
    
    # Вычисляем границы растра в его CRS
    min_x_raster = raster_transform[0]
    max_y_raster = raster_transform[3]
    max_x_raster = raster_transform[0] + w * raster_transform[1]
    min_y_raster = raster_transform[3] + h * raster_transform[5]
    print(f"Границы растра в его CRS: X=[{min_x_raster:.2f}, {max_x_raster:.2f}], Y=[{min_y_raster:.2f}, {max_y_raster:.2f}]")
    
    for i, poly in enumerate(polygons):
        # Для каждого полигона (или части мультиполигона)
        if poly.geom_type == 'MultiPolygon':
            parts = list(poly.geoms)
        else:
            parts = [poly]
        
        for part_poly in parts:
            if part_poly.exterior:
                coords = np.array(list(part_poly.exterior.coords))
                
                # Отладочная информация для первых нескольких полигонов
                if i < 3:
                    print(f"  Полигон {i} координаты в CRS (первые 5): {coords[:5]}")

                # Прямое преобразование координат в пиксели (без использования InvGeoTransform)
                # Формула: px = (x - x_origin) / pixel_width
                #          py = (y_origin - y) / pixel_height
                px = ((coords[:, 0] - raster_transform[0]) / raster_transform[1]).round().astype(int)
                py = ((raster_transform[3] - coords[:, 1]) / abs(raster_transform[5])).round().astype(int)
                
                # Обрезаем координаты до размеров изображения
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                
                pts = np.stack([px, py], axis=1)

                # Отладочная информация для первых нескольких полигонов
                if i < 3:
                    print(f"  Полигон {i} пиксельные координаты (первые 5): {pts[:5]}")
                    print(f"  Диапазон X: [{np.min(px)}, {np.max(px)}], Y: [{np.min(py)}, {np.max(py)}]")

                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
    
    print(f"Площадь созданной маски (пикселей): {np.sum(mask > 0)}")
    print("=== КОНЕЦ ОТЛАДКИ СОЗДАНИЯ МАСКИ ===")
    return mask

def calculate_pixel_area_sqm(raster_transform: tuple) -> float:
    """
    Вычисляет площадь одного пикселя в квадратных метрах на основе геотрансформации растра.
    
    Args:
        raster_transform: Геотрансформация растра (6 параметров GDAL)
    
    Returns:
        float: Площадь пикселя в квадратных метрах
    """
    # Извлекаем размеры пикселя из геотрансформации
    pixel_width = abs(raster_transform[1])  # Размер пикселя по X (обычно положительный)
    pixel_height = abs(raster_transform[5])  # Размер пикселя по Y (обычно отрицательный)
    
    # Площадь пикселя = ширина * высота
    pixel_area_sqm = pixel_width * pixel_height
    
    print(f"Размер пикселя: {pixel_width:.2f} x {pixel_height:.2f} метров")
    print(f"Площадь одного пикселя: {pixel_area_sqm:.2f} кв. метров")
    
    print('GeoTransform:', raster_transform)
    print('Pixel width:', pixel_width)
    print('Pixel height:', pixel_height)
    
    return pixel_area_sqm

def calculate_violation_areas(mask_after_subtraction: np.ndarray, pixel_area_sqm: float, min_violation_area_sqm: float = 1.5) -> Dict:
    """
    Вычисляет площади нарушений землепользования на основе анализа масок.
    
    Эта функция является ключевой для анализа нарушений - она обрабатывает маску
    объектов, которые находятся за пределами официальных кадастровых границ,
    и вычисляет детальную статистику по каждому нарушению.
    
    Алгоритм работы:
    1. Поиск отдельных областей нарушений в маске
    2. Фильтрация слишком мелких объектов (возможный шум)
    3. Вычисление площадей в различных единицах измерения
    4. Формирование детальной статистики
    
    Args:
        mask_after_subtraction (np.ndarray): Маска объектов вне кадастра (HxW)
        pixel_area_sqm (float): Площадь одного пикселя в квадратных метрах
        min_violation_area_sqm (float): Минимальная площадь нарушения в кв.м
    
    Returns:
        Dict: Подробная статистика нарушений с площадями и контурами
    """
    # Находим контуры отдельных областей нарушений в бинарной маске
    # RETR_EXTERNAL - только внешние границы (без внутренних дырок)
    # CHAIN_APPROX_SIMPLE - упрощение контуров для экономии памяти
    contours, _ = cv2.findContours(mask_after_subtraction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Списки для хранения информации о валидных нарушениях
    violation_areas = []        # Метаданные каждого нарушения
    violation_contours = []     # Геометрия контуров для визуализации
    total_violation_area_sqm = 0.0  # Суммарная площадь всех нарушений
    
    # Обрабатываем каждый найденный контур (потенциальное нарушение)
    for i, contour in enumerate(contours):
        # Вычисляем площадь текущего контура в пикселях
        contour_area_pixels = cv2.contourArea(contour)
        
        # Конвертируем площадь из пикселей в квадратные метры
        contour_area_sqm = contour_area_pixels * pixel_area_sqm
        
        # Фильтруем слишком маленькие области (могут быть артефактами или шумом)
        if contour_area_sqm < min_violation_area_sqm:
            continue  # Пропускаем нарушение, если оно меньше минимального порога
        
        # Создаем запись о нарушении с подробной информацией
        violation_areas.append({
            'id': len(violation_areas) + 1,                    # Уникальный номер нарушения
            'area_pixels': contour_area_pixels,                # Площадь в пикселях
            'area_sqm': contour_area_sqm,                      # Площадь в квадратных метрах
            'area_hectares': contour_area_sqm / 10000.0        # Площадь в гектарах (1 га = 10000 кв.м)
        })
        
        # Сохраняем контур для визуализации
        violation_contours.append(contour)
        
        # Добавляем к общей площади нарушений
        total_violation_area_sqm += contour_area_sqm
    
    # Вычисляем общую площадь нарушений альтернативным способом для проверки
    total_violation_area_pixels = int(np.sum([cv2.contourArea(c) for c in violation_contours]))
    total_violation_area_sqm_calculated = total_violation_area_pixels * pixel_area_sqm
    
    # Формируем итоговый словарь с полной статистикой
    return {
        'total_violations': len(violation_areas),                           # Общее количество нарушений
        'total_area_pixels': total_violation_area_pixels,                   # Общая площадь в пикселях
        'total_area_sqm': total_violation_area_sqm_calculated,             # Общая площадь в кв.м
        'total_area_hectares': total_violation_area_sqm_calculated / 10000.0, # Общая площадь в гектарах
        'individual_violations': violation_areas,                           # Детали каждого нарушения
        'contours': violation_contours                                      # Контуры для визуализации
    }

def process_geotiff(input_path: str, json_path: str, visualization_output_path: str, blended_mask_output_path: str,
                    config_file: str, weights_path: str,
                    chunk_size: int = 5000, overlap: int = 1536) -> Dict:
    """
    Основная функция для обработки GeoTIFF.
    Возвращает словарь с информацией о результатах обработки, включая площади нарушений.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    raster_data = read_raster_with_gdal(input_path)
    if raster_data is None:
        # Если read_raster_with_gdal вернула None, это означает критическую ошибку с проекцией
        raise RuntimeError("Прерывание работы: Не удалось определить или обработать проекцию входного растра.")

    width, height = raster_data['width'], raster_data['height']
    original_transform = raster_data['transform']
    print(f"Размер входного растра: {width}x{height}")

    full_blended_mask = np.zeros((height, width), dtype=np.uint8)
    # all_extracted_individual_features собирается для GeoJSON, но не используется для визуализации в visualize_polygons_on_raster
    all_extracted_individual_features = [] 

    print("Инициализация модели Detectron2...")
    cfg = setup_cfg(config_file, weights_path)
    predictor = DefaultPredictor(cfg)
    print(f"Модель на устройстве: {cfg.MODEL.DEVICE}")

    step_size = chunk_size - overlap
    num_chunks_y = math.ceil(height / step_size)
    num_chunks_x = math.ceil(width / step_size)
    model_input_size = 1024

    source_srs_gdal = osr.SpatialReference()
    try:
        source_srs_gdal.ImportFromWkt(raster_data['projection_wkt'])
    except Exception as e:
        # Если проекция растра корректно не импортировалась ранее, но WKT есть, это тоже проблема
        raise RuntimeError(f"Ошибка при импорте WKT проекции растра в osr.SpatialReference: {e}")
    
    target_srs_wgs84 = osr.SpatialReference(); target_srs_wgs84.ImportFromEPSG(4326)
    transform_to_wgs84 = osr.CoordinateTransformation(source_srs_gdal, target_srs_wgs84)

    for y_idx in range(num_chunks_y):
        for x_idx in range(num_chunks_x):
            x_start, y_start = x_idx * step_size, y_idx * step_size
            x_end, y_end = min(x_start + chunk_size, width), min(y_start + chunk_size, height)
            if x_end - x_start <= 0 or y_end - y_start <= 0: continue

            print(f"Обработка чанка ({x_idx+1}/{num_chunks_x}, {y_idx+1}/{num_chunks_y})")
            chunk_data = raster_data['data'][:, y_start:y_end, x_start:x_end]
            masks_in_chunk = process_image_chunk(chunk_data, predictor, (model_input_size, model_input_size))

            chunk_transform = Affine.translation(x_start, y_start) * Affine.from_gdal(*original_transform)

            for local_mask in masks_in_chunk:
                full_blended_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    full_blended_mask[y_start:y_end, x_start:x_end], local_mask
                )
                # Сбор полигонов для возможного экспорта в GeoJSON, но без дедупликации
                polygons_pixels = process_mask_to_polygon(local_mask.astype(np.uint8))
                for poly_pixels in polygons_pixels:
                    coords_wgs84 = []
                    for px, py in poly_pixels:
                        x_crs, y_crs = chunk_transform * (px, py)
                        try:
                            lon, lat, _ = transform_to_wgs84.TransformPoint(x_crs, y_crs)
                            coords_wgs84.append([lon, lat])
                        except Exception: continue
                    if len(coords_wgs84) >= 3:
                        all_extracted_individual_features.append({
                            'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [coords_wgs84]},
                            'properties': {}
                        })

    # Дедупликация отключена по запросу пользователя.
    final_unique_features = all_extracted_individual_features # Используем все обнаруженные полигоны без дедупликации
    
    print(f"Всего полигонов (без дедупликации): {len(final_unique_features)}")

    feature_collection = {'type': 'FeatureCollection', 'features': final_unique_features}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_collection, f, ensure_ascii=False, indent=2)
    print(f"GeoJSON сохранён в: {json_path}")

    print("Сохранение объединенной маски всех обнаружений...")
    save_mask_to_geotiff(full_blended_mask, blended_mask_output_path, original_transform, raster_data['projection_wkt'])

    print("\n--- ОБРАБОТКА КАДАСТРОВЫХ ГРАНИЦ ---")
    # Получаем путь к кадастровым данным из конфигурации
    base_dir = os.path.abspath(os.path.dirname(__file__)) # Убедимся, что base_dir корректен
    cadastral_mif_path = os.path.join(base_dir, PATHS_CONFIG['cadastral_data'])
    cadastral_mif_path = os.path.abspath(cadastral_mif_path) # Убедимся, что путь абсолютный и корректный

    cadastral_polygons = []
    mask_after_subtraction = full_blended_mask.copy() # Инициализируем маску для вычитания
    cadastral_mask = None  # Инициализируем переменную

    if not os.path.exists(cadastral_mif_path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Кадастровый файл не найден: {cadastral_mif_path}. Кадастровые слои не будут обработаны.")
    else:
        cadastral_polygons = read_cadastral_polygons_mif_mid(cadastral_mif_path, raster_data['projection_wkt'], raster_data['proj4_projection'])
        if cadastral_polygons:
            cadastral_mask = create_mask_from_polygons(cadastral_polygons, (height, width), original_transform)
            print(f"Площадь кадастровой маски (пикселей): {np.sum(cadastral_mask > 0)}")
            # Вычитаем кадастровую маску из маски обнаружения
            mask_after_subtraction = np.where(cadastral_mask > 0, 0, full_blended_mask)
            print(f"Площадь итоговой маски после вычитания кадастра (пикселей): {np.sum(mask_after_subtraction > 0)}")

    print("\nСоздание итоговой визуализации...")
    visualize_polygons_on_raster(
        visualization_output_path,
        input_path,
        blended_mask_output_path,
        raster_data['projection_wkt'],
        raster_data['proj4_projection'], # Передаем Proj4 растра
        mask_after_subtraction, # Передаем маску после вычитания для отрисовки синих контуров
        cadastral_polygons=cadastral_polygons, # Передаем трансформированные Shapely полигоны
        violation_areas=calculate_violation_areas(mask_after_subtraction, calculate_pixel_area_sqm(original_transform)) # Передаем анализ нарушений
    )

    # Вычисляем площади нарушений
    pixel_area_sqm = calculate_pixel_area_sqm(original_transform)
    violation_areas = calculate_violation_areas(mask_after_subtraction, pixel_area_sqm)
    
    print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА НАРУШЕНИЙ ===")
    print(f"Общее количество нарушений: {violation_areas['total_violations']}")
    print(f"Общая площадь нарушений: {violation_areas['total_area_sqm']:.2f} кв. метров ({violation_areas['total_area_hectares']:.4f} га)")
    print(f"Площадь нарушений в пикселях: {violation_areas['total_area_pixels']}")
    
    # Вычисляем общую площадь распознанных объектов
    total_detected_area_pixels = np.sum(full_blended_mask > 0)
    total_detected_area_sqm = total_detected_area_pixels * pixel_area_sqm
    total_detected_area_hectares = total_detected_area_sqm / 10000.0
    
    # Вычисляем площадь кадастровых участков
    if cadastral_mask is not None:
        cadastral_area_pixels = np.sum(cadastral_mask > 0)
        cadastral_area_sqm = cadastral_area_pixels * pixel_area_sqm
        cadastral_area_hectares = cadastral_area_sqm / 10000.0
    else:
        cadastral_area_pixels = 0
        cadastral_area_sqm = 0
        cadastral_area_hectares = 0
    
    print("\n=== ПОДРОБНАЯ СТАТИСТИКА ПЛОЩАДЕЙ ===")
    print(f"Общая площадь распознанных объектов:")
    print(f"   • {total_detected_area_sqm:.2f} кв. метров")
    print(f"   • {total_detected_area_hectares:.4f} гектаров")
    print(f"   • {total_detected_area_pixels:,} пикселей")
    
    print(f"\nПлощадь кадастровых участков:")
    print(f"   • {cadastral_area_sqm:.2f} кв. метров")
    print(f"   • {cadastral_area_hectares:.4f} гектаров")
    print(f"   • {cadastral_area_pixels:,} пикселей")
    
    print(f"\nПлощадь нарушений (объекты вне кадастра):")
    print(f"   • {violation_areas['total_area_sqm']:.2f} кв. метров")
    print(f"   • {violation_areas['total_area_hectares']:.4f} гектаров")
    print(f"   • {violation_areas['total_area_pixels']:,} пикселей")
    
    # Вычисляем процент нарушений
    if total_detected_area_sqm > 0:
        violation_percentage = (violation_areas['total_area_sqm'] / total_detected_area_sqm) * 100
        print(f"\nПроцент нарушений от общей площади объектов: {violation_percentage:.2f}%")
    
    # Вычисляем площадь объектов в кадастре
    objects_in_cadastre_area_sqm = total_detected_area_sqm - violation_areas['total_area_sqm']
    objects_in_cadastre_area_hectares = objects_in_cadastre_area_sqm / 10000.0
    print(f"\nПлощадь объектов в кадастре:")
    print(f"   • {objects_in_cadastre_area_sqm:.2f} кв. метров")
    print(f"   • {objects_in_cadastre_area_hectares:.4f} гектаров")
    
    if violation_areas['individual_violations']:
        print("\nДетализация по нарушениям:")
        for violation in violation_areas['individual_violations']:
            print(f"  Нарушение #{violation['id']}: {violation['area_sqm']:.2f} кв.м ({violation['area_hectares']:.4f} га)")
    
    # Формируем результат для возврата
    result = {
        'total_polygons': len(final_unique_features),
        'violation_analysis': violation_areas,
        'total_detected_area_sqm': total_detected_area_sqm,
        'total_detected_area_hectares': total_detected_area_hectares,
        'cadastral_area_sqm': cadastral_area_sqm,
        'cadastral_area_hectares': cadastral_area_hectares,
        'raster_info': {
            'width': width,
            'height': height,
            'pixel_area_sqm': pixel_area_sqm,
            'projection': raster_data['projection_wkt'][:100] + "..." if len(raster_data['projection_wkt']) > 100 else raster_data['projection_wkt']
        }
    }

    return result

# ============================================================================
# ТОЧКА ВХОДА СКРИПТА
# ============================================================================
if __name__ == "__main__":
    try:
        # Приветственное сообщение
        print("="*80)
        print("  СИСТЕМА ОБНАРУЖЕНИЯ НАРУШЕНИЙ ЗЕМЛЕПОЛЬЗОВАНИЯ")
        print("  Анализ спутниковых снимков с помощью нейронных сетей")
        print("="*80)
        
        # Определяем базовую директорию скрипта
        base_dir = os.path.abspath(os.path.dirname(__file__))
        print(f"\n📁 Рабочая директория: {base_dir}")
        
        # Формируем пути на основе конфигурации
        input_tiff = USER_CONFIG.get('custom_input') or PATHS_CONFIG['input_geotiff']
        input_tiff = os.path.join(base_dir, input_tiff)
        
        output_dir = os.path.join(base_dir, PATHS_CONFIG['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        # Пути к выходным файлам
        output_json = os.path.join(output_dir, "polygons_result.json")
        visualization_output_path = os.path.join(output_dir, "visualization_result.tif")
        blended_mask_output_path = os.path.join(output_dir, "blended_mask_all_detections.tif")
        violation_analysis_path = os.path.join(output_dir, "violation_analysis.json")

        # Красивый вывод информации о файлах
        print(f"\n📊 ВХОДНЫЕ И ВЫХОДНЫЕ ФАЙЛЫ:")
        print(f"   🗂️  Входной GeoTIFF: {input_tiff}")
        print(f"   📄 Выходной GeoJSON: {output_json}")
        print(f"   🖼️  Визуализация: {visualization_output_path}")
        print(f"   🎭 Маска объектов: {blended_mask_output_path}")
        print(f"   📈 Анализ нарушений: {violation_analysis_path}")

        # Пути к модели
        detectron_cfg_path = os.path.join(base_dir, PATHS_CONFIG['detectron_config'])
        weights_path = USER_CONFIG.get('custom_model') or PATHS_CONFIG['model_weights']
        weights_path = os.path.join(base_dir, weights_path)
        
        print(f"\n🤖 КОНФИГУРАЦИЯ МОДЕЛИ:")
        print(f"   ⚙️  Конфигурация: {detectron_cfg_path}")
        print(f"   🧠 Веса модели: {weights_path}")
        print(f"   🎯 Порог уверенности: {PROCESSING_CONFIG['score_threshold']}")
        print(f"   🔧 Устройство: {'GPU (CUDA)' if USER_CONFIG['gpu_acceleration'] and torch.cuda.is_available() else 'CPU'}")
        
        print(f"\n⚡ ПАРАМЕТРЫ ОБРАБОТКИ:")
        print(f"   📐 Размер чанка: {PROCESSING_CONFIG['chunk_size']} пикселей")
        print(f"   🔄 Перекрытие: {PROCESSING_CONFIG['overlap']} пикселей")
        print(f"   🎪 Режим качества: {'Высокий' if USER_CONFIG['high_quality_mode'] else 'Обычный'}")

        # Проверка существования файлов
        print(f"\n🔍 ПРОВЕРКА ФАЙЛОВ:")
        if not os.path.exists(detectron_cfg_path):
            print(f"   ❌ Файл конфигурации Detectron2 не найден: {detectron_cfg_path}")
            sys.exit(1)
        else:
            print(f"   ✅ Конфигурация Detectron2 найдена")
            
        if not os.path.exists(weights_path):
            print(f"   ❌ Файл весов модели не найден: {weights_path}")
            sys.exit(1)
        else:
            print(f"   ✅ Веса модели найдены")
            
        if not os.path.exists(input_tiff):
            print(f"   ❌ Входной GeoTIFF не найден: {input_tiff}")
            sys.exit(1)
        else:
            print(f"   ✅ Входной GeoTIFF найден")
        
        print(f"\n🚀 ЗАПУСК ОБРАБОТКИ...")
        print("-" * 80)

        # Запуск основной обработки с параметрами из конфигурации
        result = process_geotiff(
            input_tiff, output_json, visualization_output_path, blended_mask_output_path,
            config_file=detectron_cfg_path, 
            weights_path=weights_path,
            chunk_size=PROCESSING_CONFIG['chunk_size'],
            overlap=PROCESSING_CONFIG['overlap']
        )

        # Сохраняем результаты анализа нарушений в отдельный файл
        with open(violation_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Результаты анализа сохранены в: {violation_analysis_path}")

        print("\n" + "="*80)
        if result['total_polygons'] > 0:
            print("🎉 ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            print("="*80)
            
            # Выводим итоговую сводку с красивым форматированием
            violation_analysis = result['violation_analysis']
            
            print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   🏗️  Всего объектов найдено: {result['total_polygons']:,}")
            print(f"   📐 Общая площадь объектов: {result['total_detected_area_sqm']:,.2f} кв.м ({result['total_detected_area_hectares']:.4f} га)")
            print(f"   🗺️  Площадь кадастра: {result['cadastral_area_sqm']:,.2f} кв.м ({result['cadastral_area_hectares']:.4f} га)")
            
            print(f"\n🚨 АНАЛИЗ НАРУШЕНИЙ:")
            print(f"   ⚠️  Количество нарушений: {violation_analysis['total_violations']}")
            print(f"   📏 Площадь нарушений: {violation_analysis['total_area_sqm']:,.2f} кв.м")
            print(f"   🏞️  Площадь нарушений: {violation_analysis['total_area_hectares']:.4f} га")
            
            if violation_analysis['total_violations'] > 0:
                total_area = result['total_detected_area_sqm']
                violation_percent = (violation_analysis['total_area_sqm'] / total_area * 100) if total_area > 0 else 0
                print(f"   📊 Процент нарушений: {violation_percent:.2f}%")
            
            print(f"\n📁 Все результаты сохранены в папке: {output_dir}")
            print(f"   • Полигоны объектов: polygons_result.json")
            print(f"   • Визуализация: visualization_result.tif/.png")
            print(f"   • Маска объектов: blended_mask_all_detections.tif")
            print(f"   • Анализ нарушений: violation_analysis.json")
            
        else:
            print("⚠️  ОБРАБОТКА ЗАВЕРШЕНА - ОБЪЕКТЫ НЕ НАЙДЕНЫ")
            print("="*80)
            print("\nВозможные причины:")
            print("   • Низкий порог уверенности модели")
            print("   • Отсутствие целевых объектов на снимке")
            print("   • Проблемы с качеством входного изображения")
            print("   • Несоответствие модели типу объектов")

        print("\n" + "="*80)
        print("  СИСТЕМА ЗАВЕРШИЛА РАБОТУ")
        print("="*80)

    except Exception as e:
        print(f"\n💥 [КРИТИЧЕСКАЯ ОШИБКА]: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        print("\n🔧 Проверьте:")
        print("   • Корректность путей к файлам")
        print("   • Наличие всех зависимостей")
        print("   • Достаточно ли свободной памяти")
        print("   • Права доступа к файлам")
        sys.exit(1)
