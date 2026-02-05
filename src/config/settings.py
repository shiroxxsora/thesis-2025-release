"""Настройки и конфигурация приложения."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .constants import *


@dataclass
class DetectorConfig:
    """Конфигурация детектора объектов."""
    
    config_file: str = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    model_weights: str = "models/pavel-01-07-25/model_final.pth"
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    num_classes: int = 1
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Параметры обработки
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE


@dataclass
class AnalysisConfig:
    """Основная конфигурация анализа."""
    
    # Входные данные
    input_geotiff: str = "geotiffs/input.tiff"
    cadastral_data: str = "cadastr/ЗУ все2.MIF"
    
    # Выходная директория
    output_dir: str = "output/comprehensive"
    
    # Параметры детекции
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    # Фильтрация объектов
    min_polygon_area: float = MIN_POLYGON_AREA_SQM
    min_violation_area: float = MIN_VIOLATION_AREA_SQM
    
    # Упрощение геометрии
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOLERANCE_M
    cv_eps_factor: float = DEFAULT_CV_EPS_FACTOR
    
    # Привязка к кадастру
    binding_min_intersection_ratio: float = DEFAULT_INTERSECTION_RATIO
    binding_boundary_buffer_m: float = DEFAULT_BOUNDARY_BUFFER_M
    binding_max_nearest_distance_m: float = DEFAULT_MAX_DISTANCE_M
    
    # Дедупликация
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    
    def __post_init__(self):
        """Создаёт выходную директорию если её нет."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class VisualizationConfig:
    """Конфигурация визуализации."""
    
    dpi: int = DEFAULT_DPI
    figure_size: tuple = DEFAULT_FIGURE_SIZE
    
    # Цвета
    cadastral_edge_color: str = 'blue'
    detected_face_color: str = 'green'
    detected_alpha: float = 0.4
    violation_face_color: str = 'red'
    violation_alpha: float = 0.8
    
    # Размеры линий
    cadastral_linewidth: float = 0.8
    detected_linewidth: float = 0.5
    violation_linewidth: float = 1.0
