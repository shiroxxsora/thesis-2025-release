"""Дедупликация обнаруженных объектов."""

import logging
from typing import List
from shapely.geometry import Polygon

from ..domain.models import DetectedObject
from ..utils.geometry_utils import calculate_overlap_ratio

logger = logging.getLogger(__name__)


class Deduplicator:
    """Класс для удаления дубликатов обнаруженных объектов."""
    
    def __init__(self, overlap_threshold: float = 0.5):
        """
        Инициализация дедупликатора.
        
        Args:
            overlap_threshold: Порог перекрытия для считания дубликатом (IoU)
        """
        self.overlap_threshold = overlap_threshold
    
    def deduplicate(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Удаляет дублирующиеся объекты.
        
        Объекты считаются дубликатами если их IoU (Intersection over Union)
        превышает порог overlap_threshold.
        
        Args:
            objects: Список обнаруженных объектов
            
        Returns:
            Список уникальных объектов
        """
        if not objects:
            return []
        
        logger.info(f"Дедупликация: исходно {len(objects)} объектов")
        
        # Сортируем по площади (больше сначала)
        sorted_objects = sorted(objects, key=lambda x: x.area_sqm, reverse=True)
        
        unique_objects = []
        removed_count = 0
        
        for obj in sorted_objects:
            # Проверяем, не является ли объект дубликатом уже добавленных
            is_duplicate = False
            
            for unique_obj in unique_objects:
                overlap = calculate_overlap_ratio(
                    obj.geometry,
                    unique_obj.geometry
                )
                
                if overlap >= self.overlap_threshold:
                    is_duplicate = True
                    removed_count += 1
                    logger.debug(
                        f"Объект {obj.instance_id} является дубликатом "
                        f"{unique_obj.instance_id} (IoU={overlap:.2f})"
                    )
                    break
            
            if not is_duplicate:
                unique_objects.append(obj)
        
        logger.info(
            f"Дедупликация: удалено {removed_count} дубликатов, "
            f"осталось {len(unique_objects)} уникальных объектов"
        )
        
        return unique_objects
