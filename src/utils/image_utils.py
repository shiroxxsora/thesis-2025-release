"""Утилиты для работы с изображениями."""

import numpy as np
from typing import Tuple
from ..domain.types import ImageArray, FloatArray


def normalize_to_uint8(channel: np.ndarray) -> ImageArray:
    """
    Нормализует канал изображения к uint8 [0, 255].
    
    Args:
        channel: Входной массив
        
    Returns:
        Нормализованный массив uint8
    """
    # Уже uint8 - возвращаем как есть
    if channel.dtype == np.uint8:
        return channel
    
    # Вычисляем min/max
    ch_min = float(np.nanmin(channel))
    ch_max = float(np.nanmax(channel))
    
    # Проверка на валидность диапазона
    if not (np.isfinite(ch_min) and np.isfinite(ch_max)):
        # Если данные невалидны, возвращаем нули
        return np.zeros_like(channel, dtype=np.uint8)
    
    if ch_max <= ch_min:
        # Если нет диапазона, возвращаем константу
        return np.full_like(channel, 128, dtype=np.uint8)
    
    # Нормализация к [0, 255]
    normalized = 255.0 * (channel.astype(np.float32) - ch_min) / (ch_max - ch_min)
    return normalized.astype(np.uint8)


def prepare_image_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int],
    normalize: bool = True
) -> ImageArray:
    """
    Подготавливает изображение для подачи в модель.
    
    Args:
        image: Входное изображение (H, W, C) или (C, H, W)
        target_size: Целевой размер (width, height)
        normalize: Нормализовать ли к uint8
        
    Returns:
        Подготовленное изображение
    """
    import cv2
    
    # Преобразуем к формату (H, W, C) если нужно
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        # Формат (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    # Нормализация каналов
    if normalize and image.dtype != np.uint8:
        if image.shape[-1] == 1:
            image = normalize_to_uint8(image[:, :, 0])
            image = np.stack([image] * 3, axis=-1)
        else:
            channels = [normalize_to_uint8(image[:, :, i]) for i in range(min(3, image.shape[-1]))]
            if len(channels) < 3:
                channels.extend([channels[-1]] * (3 - len(channels)))
            image = np.stack(channels, axis=-1)
    
    # Изменение размера
    if image.shape[:2] != target_size[::-1]:  # target_size это (W, H)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    return image


def convert_rgb_to_bgr(image: ImageArray) -> ImageArray:
    """
    Конвертирует RGB в BGR (для OpenCV/Detectron2).
    
    Args:
        image: RGB изображение
        
    Returns:
        BGR изображение
    """
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_3_channels(image: np.ndarray) -> ImageArray:
    """
    Гарантирует, что изображение имеет 3 канала.
    
    Args:
        image: Входное изображение
        
    Returns:
        Изображение с 3 каналами
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    
    if image.ndim == 3:
        if image.shape[-1] == 1:
            return np.concatenate([image] * 3, axis=-1)
        elif image.shape[-1] == 3:
            return image
        elif image.shape[-1] > 3:
            return image[:, :, :3]
    
    raise ValueError(f"Неожиданная форма изображения: {image.shape}")
