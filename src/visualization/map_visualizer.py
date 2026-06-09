"""Визуализация карт."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry.base import BaseGeometry

from ..config.constants import DEFAULT_BOUNDARY_BUFFER_M, DEFAULT_INTERSECTION_RATIO, DEFAULT_MAX_DISTANCE_M
from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Объекты детекции на карте (заливка + контур; плашка — янтарь, тёмный текст для контраста)
_DETECTED_FACE_COLOR = "#ffeb3b"
_DETECTED_EDGE_COLOR = "#e65100"
_DETECTED_FACE_ALPHA = 0.45
_LABEL_BG_DETECTED = "#fbc02d"
_LABEL_TEXT_ON_DETECTED = "#212121"
_LABEL_BG_VIOLATION = "#c62828"

# Базовые размеры плашек (pt); итог = база × _map_label_font_scale(dpi)
# Пользовательская настройка: делаем плашки заметно меньше, чтобы не перекрывали карту.
_STICKER_FONT_MAIN = 4.25
_STICKER_FONT_CAD = 3.5


def _map_label_font_scale(dpi: float) -> float:
    """Множитель к pt: безопасный диапазон, чтобы при 350 dpi плашки не перекрывали всю карту."""
    r = float(dpi) / 200.0
    return float(max(0.58, min(0.88, 0.54 * (r ** 0.2))))


def _sticker_half_extents_data(
    map_bounds: Tuple[float, float, float, float],
    line1: str,
    line2: str,
) -> Tuple[float, float]:
    """Половины ширины/высоты плашки в координатах карты (эвристика против наложений)."""
    minx, miny, maxx, maxy = map_bounds
    span_x = max(maxx - minx, 1e-9)
    span_y = max(maxy - miny, 1e-9)
    nch = max(len(line1), len(line2), 6)
    # Эвристика габаритов плашки в метрах карты: держим согласованной с реальным размером шрифта.
    hw = span_x * (0.0031 + min(nch, 48) * 0.00016)
    hh = span_y * 0.0046
    return hw, hh


class _StickerPlacer:
    """Не даёт ставить плашки, если их прямоугольники в метрах пересекаются."""

    def __init__(self, map_bounds: Tuple[float, float, float, float]):
        self._bounds = map_bounds
        self._items: List[Tuple[float, float, float, float]] = []

    def try_place(
        self,
        cx: float,
        cy: float,
        line1: str,
        line2: str,
        clearance: float = 1.08,
    ) -> bool:
        hw, hh = _sticker_half_extents_data(self._bounds, line1, line2)
        hw *= clearance
        hh *= clearance
        for px, py, phw, phh in self._items:
            if abs(cx - px) < hw + phw and abs(cy - py) < hh + phh:
                return False
        self._items.append((cx, cy, hw, hh))
        return True

    def force_place(
        self,
        cx: float,
        cy: float,
        line1: str,
        line2: str,
        clearance: float = 1.08,
    ) -> None:
        """Регистрирует плашку даже при коллизии (fallback-резервирование)."""
        hw, hh = _sticker_half_extents_data(self._bounds, line1, line2)
        self._items.append((cx, cy, hw * clearance, hh * clearance))


def _resolve_sticker_xy(
    placer: _StickerPlacer,
    x0: float,
    y0: float,
    line1: str,
    line2: str,
    map_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Ищет позицию плашки без пересечения с уже размещёнными (спираль в метрах карты)."""
    if placer.try_place(x0, y0, line1, line2):
        return x0, y0
    minx, miny, maxx, maxy = map_bounds
    span = max(maxx - minx, maxy - miny, 1e-9)
    step = span * 0.007
    for k in range(1, 32):
        ang = k * 2.3999632297286533
        r = step * float(np.sqrt(k))
        nx = x0 + r * float(np.cos(ang))
        ny = y0 + r * float(np.sin(ang))
        if nx < minx or nx > maxx or ny < miny or ny > maxy:
            continue
        if placer.try_place(nx, ny, line1, line2):
            return nx, ny
    # Если не нашли свободного места — всё равно резервируем исходную точку,
    # иначе последующие плашки будут считать её свободной и наложения усилятся.
    placer.force_place(x0, y0, line1, line2)
    return x0, y0


def _geometry_label_xy(geom: BaseGeometry) -> Tuple[float, float]:
    c = geom.centroid
    return float(c.x), float(c.y)


#
# NOTE: CadastralMatcher / centroid-under-violation were used when labels were drawn per detected object.
# Now cadastral labels are placed per parcel centroid; keep the constants import for defaults used elsewhere.


def _cadastral_line(parcel: Optional[CadastralParcel]) -> str:
    if parcel is None:
        return "—"
    cn = (parcel.cadastral_number or "").strip()
    return cn if cn else "—"


def _draw_sticker_label(
    ax,
    x: float,
    y: float,
    line_primary: str,
    line_secondary: str,
    facecolor: str,
    zorder: int = 25,
    fontsize_main: float = _STICKER_FONT_MAIN,
    fontsize_cad: float = _STICKER_FONT_CAD,
    text_color: str = "white",
    font_scale: float = 1.0,
) -> None:
    """Скруглённая плашка: одна или две строки."""
    # FreeType в matplotlib не поддерживает fontsize < 1.0 pt (иначе спамит логами и всё равно ставит 1).
    fs_m = max(1.0, float(fontsize_main) * float(font_scale))
    fs_c = max(1.0, float(fontsize_cad) * float(font_scale))
    _tp = dict(
        color=text_color,
        family="sans-serif",
        antialiased=True,
    )
    ta1 = TextArea(
        line_primary,
        textprops=dict(
            fontsize=fs_m,
            fontweight="bold",
            **_tp,
        ),
    )
    if (line_secondary or "").strip():
        ta2 = TextArea(
            line_secondary,
            textprops=dict(fontsize=fs_c, **_tp),
        )
        children = [ta1, ta2]
    else:
        children = [ta1]
    pack = VPacker(children=children, align="center", pad=0, sep=1)
    ab = AnnotationBbox(
        pack,
        (x, y),
        xycoords="data",
        boxcoords="data",
        box_alignment=(0.5, 0.5),
        frameon=True,
        bboxprops=dict(
            boxstyle="round,pad=0.03",
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.35,
        ),
        zorder=zorder,
    )
    patch = getattr(ab, "patch", None)
    if patch is not None:
        patch.set_antialiased(True)
    ax.add_artist(ab)


def _sample_len(n: int, step: int) -> int:
    """Число строк/столбцов при выборке [::step] по длине n."""
    return (n + step - 1) // step if step >= 1 else n


def _stride_for_max_hw(h: int, w: int, max_edge: Optional[int]) -> int:
    """
    Целый шаг s: после data[..., ::s, ::s] обе стороны <= max_edge.
    Простой ceil(max(h,w)/max_edge) недостаточен для вытянутых растров.
    max_edge is None — без прореживания (шаг 1).
    """
    if max_edge is None:
        return 1
    if max(h, w) <= max_edge:
        return 1
    s = max(1, int(np.ceil(max(h, w) / float(max_edge))))
    while max(_sample_len(h, s), _sample_len(w, s)) > max_edge:
        s += 1
    return s


def _default_max_raster_edge(dpi: int, fig_w: float, fig_h: float) -> int:
    """Верхняя граница пикселей подложки: ~запас к DPI фигуры, без гигантских буферов."""
    v = int(max(fig_w, fig_h) * dpi * 2.5)
    return max(4096, min(16384, v))


def _pil_resize_hw3_uint8(rgb: np.ndarray, max_edge: Optional[int]) -> np.ndarray:
    """Дожимает H×W×3 uint8 до max_edge по длинной стороне (LANCZOS). max_edge None — без изменений."""
    if max_edge is None:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    m = max(h, w)
    if m <= max_edge or not _PIL_AVAILABLE:
        return rgb
    scale = max_edge / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    im = PILImage.fromarray(rgb, mode="RGB")
    try:
        im = im.resize((nw, nh), PILImage.Resampling.LANCZOS)
    except AttributeError:
        im = im.resize((nw, nh), PILImage.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _stretch_to_uint8_hw3(hw3: np.ndarray) -> np.ndarray:
    """Поканальная линейная нормализация в uint8; один канал float32 за раз."""
    out = np.empty(hw3.shape[:2] + (3,), dtype=np.uint8)
    for i in range(3):
        ch = hw3[:, :, i]
        lo = float(np.min(ch))
        hi = float(np.max(ch))
        if hi > lo:
            tmp = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
            out[:, :, i] = np.clip(tmp, 0, 255).astype(np.uint8)
        else:
            out[:, :, i] = 0
    return out


def _stretch_to_uint8_hw3_global(
    hw3: np.ndarray, lo3: Tuple[float, float, float], hi3: Tuple[float, float, float]
) -> np.ndarray:
    """Та же нормализация, но с общими пределами по каналам (для тайлов без швов)."""
    out = np.empty(hw3.shape[:2] + (3,), dtype=np.uint8)
    for i in range(3):
        ch = hw3[:, :, i]
        lo, hi = float(lo3[i]), float(hi3[i])
        if hi > lo:
            tmp = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
            out[:, :, i] = np.clip(tmp, 0, 255).astype(np.uint8)
        else:
            out[:, :, i] = 0
    return out


# Макс. исходных пикселей по строке/столбцу на один тайл (до прореживания); снижает пик RAM.
_ORTHOPHOTO_TILE_SRC_PX = 4096


def _extent_for_pixel_window(
    transform: Tuple[float, ...], c0: int, c1: int, r0: int, r1: int
) -> Tuple[float, float, float, float]:
    """Границы окна пикселей [c0,c1)×[r0,r1) в CRS: (min_x, max_x, min_y, max_y).

    Для общего affine с поворотом/сдвигом прямоугольник не совпадает с истинной сеткой пикселей.
    """
    t = transform
    xs: list[float] = []
    ys: list[float] = []
    for c in (c0, c1):
        for r in (r0, r1):
            xs.append(t[0] + c * t[1] + r * t[2])
            ys.append(t[3] + c * t[4] + r * t[5])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Убираем видимые "швы" между соседними imshow: слегка перекрываем extent.
    # Полпикселя по affine обычно достаточно, при этом геометрия не меняется заметно.
    # (matplotlib может рисовать тонкие зазоры из-за округления в экранных пикселях)
    px = max(abs(float(t[1])), abs(float(t[2])), abs(float(t[4])), abs(float(t[5])), 0.0)
    eps = 0.51 * px  # ~полпикселя
    if eps > 0:
        min_x -= eps
        max_x += eps
        min_y -= eps
        max_y += eps

    return (min_x, max_x, min_y, max_y)


def _accumulate_rgb_minmax_tiled(
    chw: np.ndarray, h0: int, w0: int, step: int, tile: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Min/max по R,G,B по всему растру без сборки полного H×W×3 в памяти."""
    cmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    cmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for r0 in range(0, h0, tile):
        r1 = min(r0 + tile, h0)
        for c0 in range(0, w0, tile):
            c1 = min(c0 + tile, w0)
            blk = chw[:, r0:r1:step, c0:c1:step]
            if blk.size == 0:
                continue
            for i in range(3):
                cmin[i] = min(cmin[i], float(np.min(blk[i])))
                cmax[i] = max(cmax[i], float(np.max(blk[i])))
    return cmin, cmax


def _accumulate_gray_minmax_tiled(
    g2d: np.ndarray, h0: int, w0: int, step: int, tile: int
) -> Tuple[float, float]:
    """Min/max по одноканальному растру без полного np.min(весь массив)."""
    glo, ghi = float("inf"), float("-inf")
    for r0 in range(0, h0, tile):
        r1 = min(r0 + tile, h0)
        for c0 in range(0, w0, tile):
            c1 = min(c0 + tile, w0)
            blk = g2d[r0:r1:step, c0:c1:step]
            if blk.size == 0:
                continue
            glo = min(glo, float(np.min(blk)))
            ghi = max(ghi, float(np.max(blk)))
    return glo, ghi


def _pil_resize_hw3_uint8_by_factor(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Уменьшение H×W при factor < 1 (общий масштаб как у монолитной подложки после stride)."""
    if factor >= 1.0 - 1e-12 or not _PIL_AVAILABLE:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    if nh == h and nw == w:
        return rgb
    im = PILImage.fromarray(rgb, mode="RGB")
    try:
        im = im.resize((nw, nh), PILImage.Resampling.LANCZOS)
    except AttributeError:
        im = im.resize((nw, nh), PILImage.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _try_draw_orthophoto_tiled(
    ax,
    geotiff_data: GeoTiffData,
    max_raster_edge: Optional[int],
) -> bool:
    """
    Рисует ортофото тайлами: не держит в RAM целый H×W×3 после прореживания.
    Для float-RGB сначала собирает общие min/max по каналам (без швов).

    Предполагается «север вверх» без сдвига по affine (t[2], t[4] ≈ 0); иначе
    прямоугольный extent на тайл не совпадает с параллелограммом сетки — см. GDAL.
    """
    data = geotiff_data.data
    if data is None or data.size == 0:
        return False
    t = geotiff_data.transform
    tile = _ORTHOPHOTO_TILE_SRC_PX

    try:
        if data.ndim == 3 and data.shape[0] >= 3:
            h0, w0 = int(data.shape[1]), int(data.shape[2])
            step = _stride_for_max_hw(h0, w0, max_raster_edge)
            h1, w1 = _sample_len(h0, step), _sample_len(w0, step)
            if step > 1:
                logger.info(
                    "Подложка (тайлы): шаг %s → виртуально %s×%s; плитка исх. до %s px по исходу",
                    step,
                    w1,
                    h1,
                    tile,
                )
            else:
                logger.info(
                    "Подложка (тайлы): без прореживания; плитка исх. до %s px по исходу", tile
                )
            pil_scale = 1.0
            if max_raster_edge is not None and max(h1, w1) > max_raster_edge:
                pil_scale = max_raster_edge / float(max(h1, w1))

            chw = data[:3, :, :]
            # uint8 не выходит за [0,255]; без полного chw.min/max по гигарастру
            use_u8 = chw.dtype == np.uint8 and chw.size > 0
            if use_u8:
                lo3 = hi3 = None
            else:
                cmin, cmax = _accumulate_rgb_minmax_tiled(chw, h0, w0, step, tile)
                lo3 = (float(cmin[0]), float(cmin[1]), float(cmin[2]))
                hi3 = (float(cmax[0]), float(cmax[1]), float(cmax[2]))

            for r0 in range(0, h0, tile):
                r1 = min(r0 + tile, h0)
                for c0 in range(0, w0, tile):
                    c1 = min(c0 + tile, w0)
                    raw = np.ascontiguousarray(chw[:, r0:r1:step, c0:c1:step])
                    if raw.size == 0:
                        continue
                    hw3 = np.transpose(raw, (1, 2, 0))
                    if use_u8:
                        rgb8 = np.ascontiguousarray(hw3)
                    else:
                        assert lo3 is not None and hi3 is not None
                        rgb8 = _stretch_to_uint8_hw3_global(hw3, lo3, hi3)
                    rgb8 = _pil_resize_hw3_uint8_by_factor(rgb8, pil_scale)
                    ex = _extent_for_pixel_window(t, c0, c1, r0, r1)
                    ax.imshow(
                        rgb8,
                        extent=[ex[0], ex[1], ex[2], ex[3]],
                        interpolation="nearest",
                        zorder=0,
                    )
            return True

        if data.ndim == 3 and data.shape[0] == 1:
            gray2d = data[0]
        elif data.ndim == 2:
            gray2d = data
        else:
            return False

        h0, w0 = int(gray2d.shape[0]), int(gray2d.shape[1])
        step = _stride_for_max_hw(h0, w0, max_raster_edge)
        h1, w1 = _sample_len(h0, step), _sample_len(w0, step)
        glo, ghi = _accumulate_gray_minmax_tiled(gray2d, h0, w0, step, tile)
        pil_scale = 1.0
        if max_raster_edge is not None and max(h1, w1) > max_raster_edge:
            pil_scale = max_raster_edge / float(max(h1, w1))

        for r0 in range(0, h0, tile):
            r1 = min(r0 + tile, h0)
            for c0 in range(0, w0, tile):
                c1 = min(c0 + tile, w0)
                g = gray2d[r0:r1:step, c0:c1:step]
                if g.size == 0:
                    continue
                if ghi > glo:
                    u8 = np.clip(
                        ((g.astype(np.float32) - glo) * (255.0 / (ghi - glo))), 0, 255
                    ).astype(np.uint8)
                else:
                    u8 = np.zeros(g.shape, dtype=np.uint8)
                hw3 = np.stack([u8, u8, u8], axis=-1)
                hw3 = _pil_resize_hw3_uint8_by_factor(hw3, pil_scale)
                ex = _extent_for_pixel_window(t, c0, c1, r0, r1)
                ax.imshow(
                    hw3,
                    extent=[ex[0], ex[1], ex[2], ex[3]],
                    interpolation="nearest",
                    zorder=0,
                )
        return True
    except MemoryError:
        logger.error(
            "Тайловая подложка: нехватка памяти; монолитный режим не используется (избегаем повторного OOM)."
        )
        return False
    except Exception as e:
        logger.warning("Тайловая подложка: %s — пробуем монолитный режим", e)
        return False


def _orthophoto_rgb_uint8_for_axes(
    geotiff_data: GeoTiffData,
    max_raster_edge: Optional[int],
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """
    Подложка для imshow: RGB uint8 H×W×3 и extent в метрах.

    Сначала прореживание по осям (C,H,W), затем transpose в H×W×3 — без
    полного float64(H,W,3) на исходный размер. extent полный.
    """
    data = geotiff_data.data
    if data is None or data.size == 0:
        return None

    bounds = geotiff_data.bounds
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])

    try:
        if data.ndim == 3 and data.shape[0] >= 3:
            h0, w0 = int(data.shape[1]), int(data.shape[2])
            step = _stride_for_max_hw(h0, w0, max_raster_edge)
            raw = np.ascontiguousarray(data[:3, ::step, ::step])
            hw3 = np.transpose(raw, (1, 2, 0))
            h1, w1 = hw3.shape[0], hw3.shape[1]
            if step > 1:
                logger.info(
                    "Подложка: прореживание шаг %s → %s×%s px перед нормализацией",
                    step,
                    w1,
                    h1,
                )
            if hw3.dtype == np.uint8 and hw3.size and hw3.min() >= 0 and hw3.max() <= 255:
                rgb8 = np.ascontiguousarray(hw3)
            else:
                rgb8 = _stretch_to_uint8_hw3(hw3)
            h2, w2 = rgb8.shape[0], rgb8.shape[1]
            rgb8 = _pil_resize_hw3_uint8(rgb8, max_raster_edge)
            if rgb8.shape[0] != h2 or rgb8.shape[1] != w2:
                logger.info(
                    "Подложка: LANCZOS %s×%s → %s×%s px (лимит %s)",
                    w2,
                    h2,
                    rgb8.shape[1],
                    rgb8.shape[0],
                    max_raster_edge,
                )
            return rgb8, extent

        if data.ndim == 3 and data.shape[0] == 1:
            gray2d = np.ascontiguousarray(data[0])
        elif data.ndim == 2:
            gray2d = np.ascontiguousarray(data)
        else:
            return None

        h0, w0 = gray2d.shape[0], gray2d.shape[1]
        step = _stride_for_max_hw(h0, w0, max_raster_edge)
        g = gray2d[::step, ::step]
        lo, hi = float(np.min(g)), float(np.max(g))
        if hi > lo:
            u8 = np.clip(((g.astype(np.float32) - lo) * (255.0 / (hi - lo))), 0, 255).astype(np.uint8)
        else:
            u8 = np.zeros(g.shape, dtype=np.uint8)
        hw3 = np.stack([u8, u8, u8], axis=-1)
        hw3 = _pil_resize_hw3_uint8(hw3, max_raster_edge)
        return hw3, extent
    except Exception as e:
        logger.warning("Не удалось подготовить подложку из GeoTIFF: %s", e)
        return None


def _orthophoto_rgba_for_axes(
    geotiff_data: GeoTiffData,
    max_raster_edge: Optional[int],
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """RGB uint8 + непрозрачный альфа для imshow (без лишнего float на полный растр)."""
    got = _orthophoto_rgb_uint8_for_axes(geotiff_data, max_raster_edge)
    if got is None:
        return None
    rgb, extent = got
    h, w = rgb.shape[0], rgb.shape[1]
    a = np.full((h, w, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, a], axis=2)
    return rgba, extent


class MapVisualizer:
    """Класс для визуализации карт и результатов анализа."""
    
    def __init__(
        self,
        output_dir: str,
        dpi: int = 200,
        max_raster_edge: Optional[int] = None,
        figure_size: Tuple[float, float] = (12.0, 8.0),
    ):
        """
        Инициализация визуализатора.
        
        Args:
            output_dir: Директория для сохранения изображений
            dpi: DPI фигуры и растеризации при savefig (итоговые px ≈ figure_size[in] × dpi).
            max_raster_edge: Верхний предел длины стороны подложки в px (прореживание + сжатие).
                None — исходный размер растра (без потери деталей на этапе подложки).
                Число — экономия памяти (например 8192 или см. _default_max_raster_edge).
            figure_size: Размер фигуры в дюймах (ширина, высота). Больше — выше разрешение PNG.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.max_raster_edge = max_raster_edge
        self.figure_size = figure_size
    
    def create_overview_map(
        self,
        geotiff_data: GeoTiffData,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel],
        violations: List[Violation]
    ) -> str:
        """
        Создаёт обзорную карту.
        
        Returns:
            Путь к созданному файлу
        """
        logger.info("Создание обзорной карты...")
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        pt_scale = self.dpi / 200.0
        sticker_scale = _map_label_font_scale(self.dpi)

        max_edge = self.max_raster_edge
        if max_edge is None:
            logger.warning(
                "Подложка карты: без лимита по длинной стороне — тайловая отрисовка снижает пик RAM; "
                "при нехватке памяти задайте AnalysisConfig.map_max_raster_edge (например %s).",
                8192,
            )
        else:
            logger.info("Подложка карты: max_edge=%s px по длинной стороне", max_edge)
        if not _try_draw_orthophoto_tiled(ax, geotiff_data, max_edge):
            bg = _orthophoto_rgba_for_axes(geotiff_data, max_edge)
            if bg is not None:
                img, extent = bg
                ax.imshow(
                    img,
                    extent=[extent[0], extent[1], extent[2], extent[3]],
                    interpolation="nearest",
                    zorder=0,
                )
        
        # Рисуем кадастр
        for parcel in cadastral_parcels:
            coords = list(parcel.geometry.exterior.coords)
            poly = MplPolygon(
                coords, facecolor='none', edgecolor='blue', linewidth=0.8, alpha=0.9, zorder=2
            )
            ax.add_patch(poly)
        
        # Рисуем обнаружения
        for obj in detected_objects:
            coords = list(obj.geometry.exterior.coords)
            poly = MplPolygon(
                coords,
                facecolor=_DETECTED_FACE_COLOR,
                alpha=_DETECTED_FACE_ALPHA,
                edgecolor=_DETECTED_EDGE_COLOR,
                linewidth=0.55,
                zorder=3,
            )
            ax.add_patch(poly)
        
        # Рисуем нарушения (Polygon или MultiPolygon после union по участку)
        for violation in violations:
            geom = violation.geometry
            parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
            for part in parts:
                coords = list(part.exterior.coords)
                poly = MplPolygon(
                    coords,
                    facecolor='red',
                    alpha=0.65,
                    edgecolor='darkred',
                    linewidth=1,
                    zorder=4,
                )
                ax.add_patch(poly)

        # Оси до подписей — для корректных преобразований и единых границ
        bounds = geotiff_data.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal", adjustable="box")

        # Плашки: смещение при пересечении (_StickerPlacer + спираль)
        sticker_placer = _StickerPlacer(bounds)

        for violation in violations:
            try:
                x, y = _geometry_label_xy(violation.geometry)
                l1 = f"{violation.violation_area:.1f} м²"
                l2 = _cadastral_line(violation.parcel)
                px, py = _resolve_sticker_xy(sticker_placer, x, y, l1, l2, bounds)
                _draw_sticker_label(
                    ax,
                    px,
                    py,
                    l1,
                    l2,
                    _LABEL_BG_VIOLATION,
                    zorder=20,
                    font_scale=sticker_scale,
                )
            except Exception as e:
                logger.debug("Подпись нарушения пропущена: %s", e)

        # Подписи кадастра: 1 плашка на 1 КН. Ставим по центроиду участка (так не пропадают,
        # даже если внутри ЗУ нет детекций/все центроиды попали под нарушение).
        parcel_labels: Dict[str, Tuple[float, float]] = {}
        for p in cadastral_parcels:
            try:
                cn = _cadastral_line(p)
                if cn == "—" or cn in parcel_labels:
                    continue
                cx, cy = float(p.centroid[0]), float(p.centroid[1])
                parcel_labels[cn] = (cx, cy)
            except Exception as e:
                logger.debug("Подпись кадастра пропущена: %s", e)

        for cn, (cx, cy) in parcel_labels.items():
            try:
                l1 = cn
                l2 = ""
                px, py = _resolve_sticker_xy(sticker_placer, cx, cy, l1, l2, bounds)
                _draw_sticker_label(
                    ax,
                    px,
                    py,
                    l1,
                    l2,
                    _LABEL_BG_DETECTED,
                    zorder=19,
                    font_scale=sticker_scale,
                    text_color=_LABEL_TEXT_ON_DETECTED,
                )
            except Exception as e:
                logger.debug("Подпись объекта пропущена: %s", e)

        # Настройка
        title_fs = min(24, round(13 * pt_scale))
        axis_fs = min(18, round(11 * pt_scale))
        leg_fs = max(9, round(10 * pt_scale))
        ax.set_title(
            f'Анализ нарушений: {len(violations)} нарушений из {len(detected_objects)} объектов',
            fontsize=title_fs,
            weight='bold'
        )
        ax.set_xlabel('X (м)', fontsize=axis_fs)
        ax.set_ylabel('Y (м)', fontsize=axis_fs)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(
                facecolor=_DETECTED_FACE_COLOR,
                alpha=_DETECTED_FACE_ALPHA,
                edgecolor=_DETECTED_EDGE_COLOR,
                label=f'Объекты ({len(detected_objects)})',
            ),
            Patch(facecolor='none', edgecolor='blue', label=f'Кадастр ({len(cadastral_parcels)})'),
            Patch(facecolor='red', alpha=0.8, edgecolor='darkred', label=f'Нарушения ({len(violations)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=leg_fs)
        
        plt.tight_layout()
        output_path = self.output_dir / 'visualization.png'
        plt.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor='white',
            pad_inches=0.15,
        )
        plt.close()

        try:
            from PIL import Image as PILImage

            with PILImage.open(output_path) as im:
                logger.info(
                    "Визуализация сохранена: %s (%s×%s px, фигура %.1f×%.1f in @ %s dpi)",
                    output_path,
                    im.width,
                    im.height,
                    self.figure_size[0],
                    self.figure_size[1],
                    self.dpi,
                )
        except Exception:
            logger.info("Визуализация сохранена: %s", output_path)
        return str(output_path)
