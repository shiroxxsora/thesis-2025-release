"""Главный класс для детекции объектов с использованием Detectron2."""

import logging
import torch
from typing import List
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from ..domain.models import GeoTiffData, DetectedObject
from ..config.settings import DetectorConfig
from ..processing.image_processor import ImageProcessor
from ..processing.geometry_processor import GeometryProcessor
from ..detection.mask_processor import MaskProcessor
from ..detection.chunk_processor import ChunkProcessor

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Класс для обнаружения объектов на геопространственных изображениях."""
    
    def __init__(self, config: DetectorConfig):
        """
        Инициализация детектора.
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config
        self.predictor = self._setup_model()
        
        # Инициализируем процессоры
        self.image_processor = ImageProcessor()
        self.geometry_processor = GeometryProcessor(
            simplify_tolerance=0.5,  # Можно вынести в конфиг
            cv_eps_factor=0.004
        )
        self.mask_processor = MaskProcessor(
            self.geometry_processor,
            min_polygon_area=100.0
        )
        self.chunk_processor = ChunkProcessor(
            self.image_processor,
            self.mask_processor,
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
    
    def _setup_model(self) -> DefaultPredictor:
        """
        Настраивает модель Detectron2.
        
        Returns:
            Настроенный предиктор
        """
        logger.info("Настройка модели Detectron2...")
        
        cfg = get_cfg()
        cfg.merge_from_file(self.config.config_file)
        cfg.MODEL.WEIGHTS = self.config.model_weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.score_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.num_classes
        
        # Определяем устройство
        if self.config.device == "cuda" and torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            logger.info("Используется CUDA")
        else:
            cfg.MODEL.DEVICE = "cpu"
            logger.info("Используется CPU")
        
        predictor = DefaultPredictor(cfg)
        logger.info("Модель загружена успешно")
        
        return predictor
    
    def detect(self, geotiff_data: GeoTiffData) -> List[DetectedObject]:
        """
        Обнаруживает объекты на изображении.
        
        Args:
            geotiff_data: Данные GeoTIFF для обработки
            
        Returns:
            Список обнаруженных объектов
        """
        logger.info("Начинаю обнаружение объектов...")
        
        # Обрабатываем чанками
        detected_objects = self.chunk_processor.process_chunks(
            geotiff_data,
            self.predictor,
            (self.config.model_input_size, self.config.model_input_size)
        )
        
        logger.info(f"Всего обнаружено объектов: {len(detected_objects)}")
        
        # Фильтрация по минимальной площади (если нужно)
        # filtered_objects = [
        #     obj for obj in detected_objects
        #     if obj.area_sqm >= min_area
        # ]
        
        return detected_objects
