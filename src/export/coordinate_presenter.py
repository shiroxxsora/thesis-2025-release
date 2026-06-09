"""
Утилиты для форматирования и трансформации координат в документах.
"""

import math
import re
from typing import List, Optional, Tuple

# Типовое ложное востокание для «зонной» записи МСК в кадастровых документах (м)
MSK_FALSE_EASTING_CANONICAL = 4_250_000.0
# Сдвиг сетки между вариантами метаданных (+x_0=250000 vs ожидаемая зонная запись)
MSK_GRID_OFFSET_M = 4_000_000.0
# Нординг ниже порога не трогаем (уже в виде ~4.1e6); выше — вероятен лишний +4e6 (напр. 8.1e6)
_NORTHING_ADJUST_MIN = 6_500_000.0


def projection_uses_msk_excel_convention(proj4: str) -> bool:
    """
    True — применить историческую схему МСК-03/Улан-Удэ в Excel (present_xy: перестановка
    осей и смещение 4_000_000). False — сцена в WGS/UTM и т.п.: писать (X, Y) как в
    исходной проекции растра, в метрах, без 4M-схемы.

    Пустой proj4: True (сохраняем прежнее предположение, чтобы не ломать нестандартные
    поставки, где калибруют вручную).
    """
    if not proj4 or not isinstance(proj4, str):
        return True
    s = proj4.lower()
    s_compact = s.replace(" ", "")
    if "+proj=utm" in s_compact:
        return False
    if any(
        t in s_compact
        for t in (
            "+proj=longlat",
            "+proj=latlong",
            "+proj=lonlat",
            "+proj=latlon",
        )
    ):
        return False
    return True


def parse_false_easting_x0_from_proj4(proj4: str) -> Optional[float]:
    """Извлекает +x_0 из PROJ4-строки."""
    if not proj4 or not isinstance(proj4, str):
        return None
    for token in proj4.strip().split():
        if token.startswith("+x_0="):
            try:
                return float(token.split("=", 1)[1])
            except ValueError:
                return None
    # Иногда без пробелов
    m = re.search(r"\+x_0=([+-]?\d+\.?\d*)", proj4)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def adjust_xy_before_document_present(
    x: float, y: float, proj_false_easting_x0: Optional[float]
) -> Tuple[float, float]:
    """
    При +x_0 в метаданных не как в типовой зонной записи (4250000), а например 250000,
    нординг в метрах иногда приходит с лишними 4e6 (8148012 вместо 4148012). Перед
    present_xy приводим Y к виду, согласованному с ожидаемой «зонной» записью МСК.
    """
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return (x, y)
    if proj_false_easting_x0 is None:
        return xf, yf
    try:
        x0 = float(proj_false_easting_x0)
    except (TypeError, ValueError):
        return xf, yf
    if abs(x0 - MSK_FALSE_EASTING_CANONICAL) < 200_000.0:
        return xf, yf
    if yf >= _NORTHING_ADJUST_MIN:
        yf -= MSK_GRID_OFFSET_M
    return xf, yf


def present_xy(
    x: float,
    y: float,
    proj_false_easting_x0: Optional[float] = None,
    use_msk_coordinate_presentation: bool = True,
) -> Tuple[float, float]:
    """
    Трансформирует координаты для представления в документах.

    Сначала при необходимости сдвигает север (см. adjust_xy_before_document_present),
    затем: меняет местами X и Y, применяет смещение по второй координате (историческая схема).

    Args:
        x: Исходная координата X (восток, м)
        y: Исходная координата Y (север, м)
        proj_false_easting_x0: +x_0 из PROJ растра/сцены; None — без сдвига по сетке МСК.
        use_msk_coordinate_presentation: False — WGS/UTM и др.: (X, Y) в метрах сцены.
    """
    if not use_msk_coordinate_presentation:
        try:
            return (float(x), float(y))
        except (TypeError, ValueError):
            return (x, y)  # type: ignore[return-value]

    xf, yf = adjust_xy_before_document_present(x, y, proj_false_easting_x0)
    try:
        xf = float(xf)
        yf = float(yf)
    except Exception:
        return x, y

    # Документная схема (как ожидают в отчёте МСК): X_doc = Y_src, Y_doc = 4_000_000 + X_src,
    # но если X_src уже «в зонной записи» (x_0≈4_250_000), то перед добавкой 4_000_000
    # снимаем лишние 4_000_000, иначе получится 8_***_***.
    x0 = None
    if proj_false_easting_x0 is not None:
        try:
            x0 = float(proj_false_easting_x0)
        except (TypeError, ValueError):
            x0 = None

    # Приведение X к «малому» виду (около 0..1e6) для формулы 4_000_000 + X_small.
    x_small = xf
    if x0 is not None and abs(x0 - MSK_FALSE_EASTING_CANONICAL) < 200_000.0:
        # Канонический x_0≈4_250_000: easting обычно уже содержит +4_000_000.
        # Но если easting уже «малый» (например 151k), вычитать нельзя — иначе уйдём в отрицательные.
        if xf >= 3_000_000.0:
            x_small = xf - MSK_GRID_OFFSET_M

    new_x = yf
    sign = -1.0 if x_small < 0 else 1.0
    new_y = sign * (MSK_GRID_OFFSET_M + abs(x_small))
    return new_x, new_y


def present_xy_extrema_from_bounds(
    bounds: Tuple[float, float, float, float],
    proj_false_easting_x0: Optional[float],
    use_msk_coordinate_presentation: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Мин/макс в «документных» осях после present_xy по четырём углам осевого bbox.
    """
    minx, miny, maxx, maxy = bounds
    corners = ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))
    pxys = [
        present_xy(a, b, proj_false_easting_x0, use_msk_coordinate_presentation)
        for a, b in corners
    ]
    xs = [p[0] for p in pxys]
    ys = [p[1] for p in pxys]
    return (min(xs), min(ys), max(xs), max(ys))


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
