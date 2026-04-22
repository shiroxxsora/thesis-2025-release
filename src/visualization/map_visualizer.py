"""Визуализация карт."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ..analysis.cadastral_matcher import CadastralMatcher
from ..config.constants import (
    DEFAULT_BOUNDARY_BUFFER_M,
    DEFAULT_INTERSECTION_RATIO,
    DEFAULT_MAX_DISTANCE_M,
)
from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Плашки-подписи: цвет под слой (как на карте)
_LABEL_BG_DETECTED = "#2e7d32"
_LABEL_BG_VIOLATION = "#c62828"

# Размеры текста на плашках (pt): при высоком dpi не умножать линейно — иначе перекрывают карту
_STICKER_FONT_MAIN = 4.0
_STICKER_FONT_CAD = 2.75


def _sticker_font_scale(dpi: float) -> float:
    """Слабая зависимость от dpi (в отличие от заголовка), плашки остаются компактными."""
    t = (dpi / 200.0) ** 0.32
    return float(min(0.92, max(0.62, t)) * 0.65)


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
    hw = span_x * (0.0085 + min(nch, 48) * 0.00042)
    hh = span_y * 0.0118
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


def _geometry_label_xy(geom: BaseGeometry) -> Tuple[float, float]:
    c = geom.centroid
    return float(c.x), float(c.y)


def _centroid_under_violation(x: float, y: float, violations: List[Violation]) -> bool:
    """True, если точка попадает в полигон нарушения (чтобы не дублировать зелёные подписи)."""
    pt = Point(x, y)
    for v in violations:
        try:
            g = v.geometry
            if g.covers(pt) or g.distance(pt) <= 0.08:
                return True
        except Exception:
            continue
    return False


def _matcher_default() -> CadastralMatcher:
    return CadastralMatcher(
        min_intersection_ratio=DEFAULT_INTERSECTION_RATIO,
        boundary_buffer_m=DEFAULT_BOUNDARY_BUFFER_M,
        max_nearest_distance_m=DEFAULT_MAX_DISTANCE_M,
    )


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
    ta1 = TextArea(
        line_primary,
        textprops=dict(
            color=text_color,
            fontsize=fs_m,
            fontweight="bold",
            family="sans-serif",
        ),
    )
    if (line_secondary or "").strip():
        ta2 = TextArea(
            line_secondary,
            textprops=dict(color=text_color, fontsize=fs_c, family="sans-serif"),
        )
        children = [ta1, ta2]
    else:
        children = [ta1]
    pack = VPacker(children=children, align="center", pad=0, sep=0)
    ab = AnnotationBbox(
        pack,
        (x, y),
        xycoords="data",
        boxcoords="data",
        box_alignment=(0.5, 0.5),
        frameon=True,
        bboxprops=dict(
            boxstyle="round,pad=0.06",
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.38,
        ),
        zorder=zorder,
    )
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
        sticker_scale = _sticker_font_scale(self.dpi) * (1.0 / 3.0)

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
                facecolor='green',
                alpha=0.4,
                edgecolor='darkgreen',
                linewidth=0.5,
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

        # Плашки: площадь + КН (рисуем все, даже если накладываются)
        matcher = _matcher_default()

        for violation in violations:
            try:
                x, y = _geometry_label_xy(violation.geometry)
                l1 = f"{violation.violation_area:.1f} м²"
                l2 = _cadastral_line(violation.parcel)
                _draw_sticker_label(
                    ax,
                    x,
                    y,
                    l1,
                    l2,
                    _LABEL_BG_VIOLATION,
                    zorder=20,
                    font_scale=sticker_scale,
                )
            except Exception as e:
                logger.debug("Подпись нарушения пропущена: %s", e)

        for obj in detected_objects:
            try:
                cx, cy = float(obj.centroid[0]), float(obj.centroid[1])
                if _centroid_under_violation(cx, cy, violations):
                    continue
                parcel, _, _, _ = matcher.match(obj.geometry, cadastral_parcels)
                # Зелёные подписи делаем компактнее: только кадастровый номер (без площади)
                l1 = _cadastral_line(parcel)
                l2 = ""
                _draw_sticker_label(
                    ax,
                    cx,
                    cy,
                    l1,
                    l2,
                    _LABEL_BG_DETECTED,
                    zorder=19,
                    font_scale=sticker_scale * 0.72,
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
            Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label=f'Объекты ({len(detected_objects)})'),
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
