"""Визуализация карт."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _sample_len(n: int, step: int) -> int:
    """Число строк/столбцов при выборке [::step] по длине n."""
    return (n + step - 1) // step if step >= 1 else n


def _stride_for_max_hw(h: int, w: int, max_edge: int) -> int:
    """
    Целый шаг s: после data[..., ::s, ::s] обе стороны <= max_edge.
    Простой ceil(max(h,w)/max_edge) недостаточен для вытянутых растров.
    """
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


def _pil_resize_hw3_uint8(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    """Дожимает H×W×3 uint8 до max_edge по длинной стороне (LANCZOS)."""
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


def _orthophoto_rgb_uint8_for_axes(
    geotiff_data: GeoTiffData,
    max_raster_edge: int,
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
    max_raster_edge: int,
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
    
    def __init__(self, output_dir: str, dpi: int = 200):
        """
        Инициализация визуализатора.
        
        Args:
            output_dir: Директория для сохранения изображений
            dpi: DPI для сохранения
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
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
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)

        max_edge = _default_max_raster_edge(self.dpi, 12.0, 8.0)
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
                    alpha=0.8,
                    edgecolor='darkred',
                    linewidth=1,
                    zorder=4,
                )
                ax.add_patch(poly)
        
        # Настройка
        bounds = geotiff_data.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal', adjustable='box')  # Правильные пропорции, без искажений
        ax.set_title(
            f'Анализ нарушений: {len(violations)} нарушений из {len(detected_objects)} объектов',
            fontsize=12,
            weight='bold'
        )
        ax.set_xlabel('X (м)', fontsize=10)
        ax.set_ylabel('Y (м)', fontsize=10)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label=f'Объекты ({len(detected_objects)})'),
            Patch(facecolor='none', edgecolor='blue', label=f'Кадастр ({len(cadastral_parcels)})'),
            Patch(facecolor='red', alpha=0.8, edgecolor='darkred', label=f'Нарушения ({len(violations)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'visualization.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Визуализация сохранена: {output_path}")
        return str(output_path)
