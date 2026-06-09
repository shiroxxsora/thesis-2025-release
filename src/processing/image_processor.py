"""Процессор для обработки изображений."""

import numpy as np
import cv2
from typing import Tuple, List
import logging

from ..domain.types import ImageArray
from ..utils.image_utils import (
    normalize_to_uint8,
    prepare_image_for_model,
    convert_rgb_to_bgr,
    ensure_3_channels
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс для обработки изображений перед подачей в нейросеть."""
    
    def __init__(self):
        """Инициализация процессора."""
        pass
    
    def normalize_channel(self, channel: np.ndarray) -> ImageArray:
        """
        Нормализует канал к uint8.
        
        Args:
            channel: Входной канал
            
        Returns:
            Нормализованный канал uint8
        """
        return normalize_to_uint8(channel)
    
    def prepare_chunk(
        self,
        chunk: np.ndarray,
        model_input_size: Tuple[int, int]
    ) -> ImageArray:
        """
        Подготавливает чанк для подачи в модель.
        
        Args:
            chunk: Чанк изображения (C, H, W) или (H, W, C)
            model_input_size: Размер входа модели (width, height)
            
        Returns:
            Подготовленное изображение BGR для Detectron2
        """
        # Если формат (C, H, W), оставляем первые 3 канала
        if chunk.ndim == 3 and chunk.shape[0] <= 4:
            if chunk.shape[0] > 3:
                chunk = chunk[:3]
            
            # Нормализуем каждый канал
            channels = [self.normalize_channel(chunk[i]) for i in range(chunk.shape[0])]
            
            # Дополняем до 3 каналов если нужно
            while len(channels) < 3:
                channels.append(channels[-1])
            
            # Собираем в (C, H, W) и транспонируем в (H, W, C)
            img_array = np.stack(channels, axis=0)
            img_vis = np.transpose(img_array, (1, 2, 0))
        else:
            # Формат уже (H, W, C) или (H, W)
            img_vis = ensure_3_channels(chunk)
            
            # Нормализуем если нужно
            if img_vis.dtype != np.uint8:
                channels = [self.normalize_channel(img_vis[:, :, i]) for i in range(3)]
                img_vis = np.stack(channels, axis=-1)
        
        # Изменяем размер для модели
        img_resized = cv2.resize(
            img_vis,
            model_input_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Конвертируем RGB -> BGR для Detectron2
        img_bgr = convert_rgb_to_bgr(img_resized)
        
        return img_bgr
    
    def extract_chunk(
        self,
        data: np.ndarray,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int
    ) -> np.ndarray:
        """
        Извлекает чанк из большого изображения.
        
        Args:
            data: Полное изображение (C, H, W) или (H, W, C)
            x_start: Начало по X
            y_start: Начало по Y
            x_end: Конец по X
            y_end: Конец по Y
            
        Returns:
            Чанк изображения
        """
        # Определяем формат
        if data.ndim == 3 and data.shape[0] <= 4:
            # Формат (C, H, W)
            return data[:, y_start:y_end, x_start:x_end]
        else:
            # Формат (H, W, C) или (H, W)
            return data[y_start:y_end, x_start:x_end]
