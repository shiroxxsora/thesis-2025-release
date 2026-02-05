"""Кастомные типы для type hints."""

from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray

# Типы для координат
Coordinate = Tuple[float, float]
CoordinateList = List[Coordinate]
BoundingBox = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)

# Типы для трансформаций
GeoTransform = Tuple[float, float, float, float, float, float]

# Типы для массивов
ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
MaskArray = NDArray[np.bool_]

# Типы для геометрии
PolygonCoords = List[Coordinate]

# Типы для конфигурации
ConfigDict = Dict[str, Any]

# Типы для CRS
CRSType = Any  # rasterio.crs.CRS или osr.SpatialReference
