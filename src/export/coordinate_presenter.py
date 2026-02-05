"""
Утилиты для форматирования и трансформации координат в документах.
"""

import math
from typing import List, Tuple, Optional


def present_xy(x: float, y: float) -> Tuple[float, float]:
    """
    Трансформирует координаты для представления в документах.
    
    Применяет специальную трансформацию:
    - Меняет местами X и Y
    - Применяет смещение по Y
    
    Args:
        x: Исходная координата X
        y: Исходная координата Y
        
    Returns:
        Кортеж (new_x, new_y) трансформированных координат
    """
    try:
        xf = float(x)
        yf = float(y)
    except Exception:
        return x, y
    
    new_x = yf
    sign = -1.0 if xf < 0 else 1.0
    new_y = sign * (4000000.0 + abs(xf))
    return new_x, new_y


def format_float(value: float, ndigits: int = 2) -> str:
    """
    Форматирует число с заданной точностью.
    
    Args:
        value: Число для форматирования
        ndigits: Количество знаков после запятой
        
    Returns:
        Отформатированная строка или пустая строка при ошибке
    """
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return ""


def compute_distances(points_xy: List[Tuple[float, float]]) -> List[float]:
    """
    Вычисляет расстояния между последовательными точками.
    
    Замыкает контур на первую точку (последняя дистанция - от последней к первой).
    
    Args:
        points_xy: Список точек [(x1, y1), (x2, y2), ...]
        
    Returns:
        Список расстояний (в единицах CRS, ожидаются метры)
    """
    if not points_xy:
        return []
    
    distances: List[float] = []
    for i in range(len(points_xy)):
        x1, y1 = points_xy[i]
        x2, y2 = points_xy[(i + 1) % len(points_xy)]
        distances.append(math.hypot(x2 - x1, y2 - y1))
    
    return distances


def simplify_points(
    points: List[Tuple[float, float]], 
    min_spacing: float = 3.0, 
    max_points: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Упрощает список точек для визуализации.
    
    Удаляет точки, которые расположены слишком близко друг к другу,
    и опционально ограничивает общее количество точек.
    
    Args:
        points: Исходный список точек
        min_spacing: Минимальное расстояние между точками (м)
        max_points: Максимальное количество точек (опционально)
        
    Returns:
        Упрощённый список точек
    """
    if len(points) <= 3:
        return points
    
    simplified = [points[0]]
    
    for i in range(1, len(points)):
        x_prev, y_prev = simplified[-1]
        x_curr, y_curr = points[i]
        dist = math.hypot(x_curr - x_prev, y_curr - y_prev)
        
        if dist >= min_spacing:
            simplified.append(points[i])
    
    if max_points and len(simplified) > max_points:
        step = len(simplified) / max_points
        indices = [int(i * step) for i in range(max_points)]
        simplified = [simplified[i] for i in indices]
    
    return simplified


def safe_filename(name: str) -> str:
    """
    Приводит строку к безопасному имени файла.
    
    Args:
        name: Исходное имя (например, кадастровый номер)
        
    Returns:
        Безопасное имя файла без запрещённых символов
    """
    invalid = '\\/:*?"<>|'  # Windows-символы
    result = name
    for ch in invalid:
        result = result.replace(ch, '_')
    return result.replace(' ', '_')
