"""Модули детекции объектов с использованием нейросетей."""

from .mask_processor import MaskProcessor
from .chunk_processor import ChunkProcessor
from .detector import ObjectDetector
from .deduplicator import Deduplicator

__all__ = [
    'MaskProcessor',
    'ChunkProcessor',
    'ObjectDetector',
    'Deduplicator',
]
