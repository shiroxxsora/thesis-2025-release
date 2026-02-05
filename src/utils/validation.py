"""Функции валидации данных."""

import os
from pathlib import Path
from typing import Union
from shapely.geometry import Polygon, MultiPolygon


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Проверяет существование файла.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Path объект
        
    Raises:
        FileNotFoundError: Если файл не существует
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    if not path.is_file():
        raise ValueError(f"Путь не является файлом: {file_path}")
    return path


def validate_polygon(polygon: Polygon, min_area: float = 0.0) -> bool:
    """
    Проверяет валидность полигона.
    
    Args:
        polygon: Полигон для проверки
        min_area: Минимальная площадь
        
    Returns:
        True если полигон валиден
    """
    if polygon is None:
        return False
    
    if not polygon.is_valid:
        return False
    
    if polygon.is_empty:
        return False
    
    if polygon.area < min_area:
        return False
    
    return True


def validate_crs_compatibility(crs1, crs2) -> bool:
    """
    Проверяет совместимость систем координат.
    
    Args:
        crs1: Первая CRS
        crs2: Вторая CRS
        
    Returns:
        True если CRS совместимы или идентичны
    """
    if crs1 is None or crs2 is None:
        return False
    
    # Проверка через метод IsSame (для OSR)
    if hasattr(crs1, 'IsSame'):
        try:
            return crs1.IsSame(crs2)
        except:
            pass
    
    # Проверка через сравнение строк
    try:
        return str(crs1) == str(crs2)
    except:
        return False
