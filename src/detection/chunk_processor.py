"""Процессор для разбиения изображения на чанки и их обработки."""

import math
import logging
from typing import List, Tuple, Iterator

import cv2
import numpy as np
import torch

from ..domain.models import GeoTiffData, DetectedObject
from ..processing.image_processor import ImageProcessor
from ..detection.mask_processor import MaskProcessor

logger = logging.getLogger(__name__)


def _release_cuda_after_chunk() -> None:
    """Снижает фрагментацию VRAM: outputs Detectron2 держит большие тензоры до del/GC."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class ChunkProcessor:
    """Класс для обработки больших изображений по частям (чанкам)."""
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        mask_processor: MaskProcessor,
        chunk_size: int = 5000,
        overlap: int = 1536
    ):
        """
        Инициализация процессора чанков.
        
        Args:
            image_processor: Процессор изображений
            mask_processor: Процессор масок
            chunk_size: Размер чанка в пикселях
            overlap: Перекрытие между чанками
        """
        self.image_processor = image_processor
        self.mask_processor = mask_processor
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step_size = chunk_size - overlap
    
    def generate_chunks(
        self,
        geotiff_data: GeoTiffData
    ) -> Iterator[Tuple[np.ndarray, tuple, str]]:
        """
        Генерирует чанки из GeoTIFF данных.
        
        Args:
            geotiff_data: Данные GeoTIFF
            
        Yields:
            Кортеж (chunk_data, chunk_offset, chunk_id)
            где chunk_offset это (x_start, y_start)
        """
        height = geotiff_data.height
        width = geotiff_data.width
        
        num_chunks_y = math.ceil(height / self.step_size)
        num_chunks_x = math.ceil(width / self.step_size)
        
        logger.info(
            f"Разбиение на {num_chunks_x}x{num_chunks_y} = "
            f"{num_chunks_x * num_chunks_y} чанков"
        )
        
        for y_idx in range(num_chunks_y):
            for x_idx in range(num_chunks_x):
                # Вычисляем границы чанка
                x_start = x_idx * self.step_size
                y_start = y_idx * self.step_size
                x_end = min(x_start + self.chunk_size, width)
                y_end = min(y_start + self.chunk_size, height)
                
                # Проверяем валидность
                if x_end - x_start <= 0 or y_end - y_start <= 0:
                    continue
                
                # Извлекаем чанк
                chunk = self.image_processor.extract_chunk(
                    geotiff_data.data,
                    x_start,
                    y_start,
                    x_end,
                    y_end
                )
                
                chunk_offset = (x_start, y_start)
                chunk_id = f"chunk_{x_idx}_{y_idx}"
                
                yield chunk, chunk_offset, chunk_id
    
    def process_chunks(
        self,
        geotiff_data: GeoTiffData,
        predictor,
        model_input_size: Tuple[int, int]
    ) -> List[DetectedObject]:
        """
        Обрабатывает все чанки изображения.
        
        Args:
            geotiff_data: Данные GeoTIFF
            predictor: Предиктор Detectron2
            model_input_size: Размер входа модели (width, height)
            
        Returns:
            Список всех обнаруженных объектов
        """
        all_objects = []
        total_chunks = math.ceil(geotiff_data.height / self.step_size) * \
                      math.ceil(geotiff_data.width / self.step_size)
        
        processed = 0
        
        for chunk, chunk_offset, chunk_id in self.generate_chunks(geotiff_data):
            processed += 1
            
            if processed % 10 == 0 or processed == total_chunks:
                logger.info(f"Обработка чанка {processed}/{total_chunks}")
            
            # Подготавливаем чанк для модели
            prepared_chunk = self.image_processor.prepare_chunk(
                chunk,
                model_input_size
            )
            outputs = None
            try:
                with torch.no_grad():
                    outputs = predictor(prepared_chunk)
                instances = outputs["instances"].to("cpu")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "CUDA OOM на чанке %s/%s (%s). Уменьшите chunk_size/overlap в конфиге "
                        "или освободите VRAM.",
                        processed,
                        total_chunks,
                        chunk_id,
                    )
                    _release_cuda_after_chunk()
                raise
            finally:
                del prepared_chunk
                if outputs is not None:
                    del outputs

            if not instances.has("pred_masks"):
                del instances
                _release_cuda_after_chunk()
                continue

            masks = instances.pred_masks.cpu().numpy()
            del instances

            if len(masks) == 0:
                del masks
                _release_cuda_after_chunk()
                continue

            # Масштабируем маски к размеру чанка
            chunk_height = chunk.shape[1] if chunk.ndim == 3 and chunk.shape[0] <= 4 else chunk.shape[0]
            chunk_width = chunk.shape[2] if chunk.ndim == 3 and chunk.shape[0] <= 4 else chunk.shape[1]
            
            # Обрабатываем каждую маску
            for mask_idx, mask in enumerate(masks):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (chunk_width, chunk_height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Обрабатываем маску
                objects = self.mask_processor._process_single_mask(
                    mask_resized,
                    geotiff_data.transform,
                    chunk_offset,
                    chunk_id,
                    mask_idx
                )
                
                all_objects.extend(objects)

            del masks
            _release_cuda_after_chunk()
        
        logger.info(f"Обнаружено {len(all_objects)} объектов во всех чанках")
        return all_objects
