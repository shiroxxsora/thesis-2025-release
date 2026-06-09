"""Константы приложения."""

# Минимальные значения
MIN_POLYGON_AREA_SQM = 100.0
MIN_VIOLATION_AREA_SQM = 0.5

# Пороги детекции
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_OVERLAP_THRESHOLD = 0.5

# Размеры обработки
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_OVERLAP = 1536
DEFAULT_MODEL_INPUT_SIZE = 1024

# Пороги привязки к кадастру
DEFAULT_INTERSECTION_RATIO = 0.1
DEFAULT_BOUNDARY_BUFFER_M = 3.0
DEFAULT_MAX_DISTANCE_M = 25.0
DEFAULT_CADASTRAL_OFFSET_X_M = 29.5
DEFAULT_CADASTRAL_OFFSET_Y_M = -31.5

# Упрощение геометрии
DEFAULT_SIMPLIFY_TOLERANCE_M = 0.5
DEFAULT_CV_EPS_FACTOR = 0.004

# Обзорная карта visualization.png: пиксели ≈ map_figure_size[in] × map_figure_dpi (минус поля от tight_layout).
DEFAULT_MAP_FIGURE_SIZE = (22.0, 14.67)
DEFAULT_MAP_FIGURE_DPI = 350

# Подложка overview: длина длинной стороны растра в px (прореживание). None = без лимита (риск OOM на больших GeoTIFF).
DEFAULT_MAP_MAX_RASTER_EDGE = 16384
