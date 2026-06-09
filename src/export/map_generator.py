"""
Генерация карт для документов по участкам (zoom и overview карты).
"""

import logging
import warnings
from io import BytesIO
from typing import Optional, Tuple, Any, List
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import transform, unary_union

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

logger = logging.getLogger(__name__)


def _proj4_tokens(proj4: str) -> dict:
    return dict(p.split("=", 1) for p in proj4.strip().split() if "=" in p)


def _normalize_map_dest_crs(dest_crs: "pyproj.CRS"):
    """
    WKT из GeoTIFF часто BOUNDCRS(проекция → WGS84). Для сопоставления с векторами
    нужна именно проектируемая часть (как в GDAL SOURCECRS).
    """
    try:
        if getattr(dest_crs, "is_bound", False):
            return dest_crs.source_crs
    except Exception:
        pass
    return dest_crs


def _same_tmerc_false_easting_mismatch(src_crs: "pyproj.CRS", dest_crs: "pyproj.CRS") -> bool:
    """
    Та же эвристика, что в CoordinateTransformer._check_zone_offset:
    один и тот же tmerc (lon_0, lat_0, …), но +x_0 в PROJ-строке отличается (часто 250000 в
    ExportToProj4/WKT растра и 4250000 в строке для кадастра/Excel). Координаты пикселей
    и WKT уже в одной метрической сетке геотрансформа — pyproj-переход даст сдвиг ~4 млн м.
    """
    try:
        # pyproj предупреждает при to_proj4() для некоторых CRS-представлений (BOUNDCRS и т.п.).
        # Здесь нам нужны только базовые параметры tmerc (+lat_0/+lon_0/+x_0), поэтому
        # подавляем этот warning локально, чтобы не зашумлять пакетный экспорт.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information when converting to a PROJ string.*",
                category=UserWarning,
            )
            pa = _proj4_tokens(src_crs.to_proj4())
            pb = _proj4_tokens(dest_crs.to_proj4())
        if pa.get("+proj") != "tmerc" or pb.get("+proj") != "tmerc":
            return False
        for key in ("+lat_0", "+lon_0"):
            if pa.get(key) != pb.get(key):
                return False
        xa = float(pa.get("+x_0", "0"))
        xb = float(pb.get("+x_0", "0"))
        if abs(xa - xb) < 1000.0:
            return True
        if abs(abs(xa - xb) - 4_000_000.0) < 500_000.0:
            return True
        return False
    except Exception:
        return False


class MapGenerator:
    """Генератор карт для документов по участкам."""
    
    def __init__(self, background: Optional[Tuple[Any, Optional[Tuple[float, float, float, float]], Any]] = None):
        """
        Args:
            background: Предзагруженный фон (img_array, extent, crs)
        """
        self.background = background
    
    def create_zoom_map(
        self,
        cadastral_number: str,
        cadastral_df: pd.DataFrame,
        violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame,
        save_path: str,
        proj_string: Optional[str] = None,
        merge_violations_per_parcel: bool = True,
    ) -> None:
        """
        Создаёт zoom-карту с увеличением на нарушения конкретного участка.
        
        Args:
            cadastral_number: Кадастровый номер участка
            cadastral_df: DataFrame с кадастровыми участками
            violations_df: DataFrame с нарушениями
            viol_coords_df: DataFrame с координатами нарушений
            save_path: Путь для сохранения PNG
            proj_string: PROJ-строка исходной СК (опционально)
            merge_violations_per_parcel: True — один union на КН (как при merge в анализе);
                False — отдельный полигон на каждую строку отчёта.
        """
        v = violations_df[violations_df['Ближайший кадастровый номер'] == cadastral_number]
        if v.empty:
            return

        # Попытка взять WKT геометрий нарушений
        geoms = []
        if 'WKT геометрии' in v.columns:
            for w in v['WKT геометрии'].dropna().astype(str).tolist():
                try:
                    geom = shapely_wkt.loads(w)
                    geoms.append(geom)
                except Exception:
                    pass

        if not geoms:
            return

        # Контур участка
        parcel_geom = self._get_parcel_geometry(cadastral_number, cadastral_df)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=130)

        # Перепроецирование
        project = self._setup_projection(proj_string)
        plot_geoms: List = []
        plot_parcel_geom = parcel_geom
        if project:
            try:
                if plot_parcel_geom:
                    plot_parcel_geom = transform(project, plot_parcel_geom)
                if merge_violations_per_parcel:
                    merged = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                    plot_geoms = [transform(project, merged)]
                else:
                    plot_geoms = [transform(project, g) for g in geoms]
            except Exception as e:
                logger.warning(f"Ошибка при перепроецировании геометрий: {e}")
                return
        else:
            if merge_violations_per_parcel:
                plot_geoms = [geoms[0] if len(geoms) == 1 else unary_union(geoms)]
            else:
                plot_geoms = list(geoms)

        # Если после перепроекции участок ушёл из области подложки — откатываемся в исходные координаты.
        if project and not self._is_geometry_near_background(plot_parcel_geom):
            logger.warning(
                "Zoom-карта: после перепроекции геометрия вне области подложки; "
                "использую исходные координаты без pyproj-трансформации."
            )
            project = None
            plot_parcel_geom = parcel_geom
            if merge_violations_per_parcel:
                plot_geoms = [geoms[0] if len(geoms) == 1 else unary_union(geoms)]
            else:
                plot_geoms = list(geoms)

        # Подложка
        self._draw_background(ax)

        # Контур участка
        if plot_parcel_geom is not None and not plot_parcel_geom.is_empty:
            self._plot_shapely(ax, plot_parcel_geom, facecolor='none', edgecolor='blue', linewidth=0.8, alpha=0.9)

        # Нарушения: union на КН или по строкам отчёта
        for g in plot_geoms:
            if g is None or g.is_empty:
                continue
            self._plot_shapely(ax, g, facecolor='red', edgecolor='darkred', linewidth=0.5, alpha=0.6)

        # Центроиды нарушений
        self._draw_centroids(ax, v, project)

        # Нумерация точек (координаты уже упрощены в Excel)
        self._draw_violation_points(
            ax, cadastral_number, violations_df, viol_coords_df,
            project,
        )

        # Масштаб
        self._set_zoom_scale(ax, [x for x in plot_geoms if x is not None and not x.is_empty], plot_parcel_geom)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Сохранение
        self._save_figure(fig, save_path, cadastral_number, 'zoom')
    
    def create_overview_map(
        self,
        cadastral_number: str,
        cadastral_df: pd.DataFrame,
        violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame,
        save_path: str,
        proj_string: Optional[str] = None,
        merge_violations_per_parcel: bool = True,
    ) -> None:
        """
        Создаёт обзорную карту участка с окружением.
        
        Args:
            cadastral_number: Кадастровый номер участка
            cadastral_df: DataFrame с кадастровыми участками
            violations_df: DataFrame с нарушениями
            viol_coords_df: DataFrame с координатами нарушений
            save_path: Путь для сохранения PNG
            proj_string: PROJ-строка исходной СК (опционально)
            merge_violations_per_parcel: True — один union на карте; False — отдельные полигоны,
                площадь на подписи = сумма столбца «Площадь нарушения, м²» (как в Excel).
        """
        v = violations_df[violations_df['Ближайший кадастровый номер'] == cadastral_number]
        
        cad_rows = cadastral_df[cadastral_df['Кадастровый номер'] == cadastral_number]
        if cad_rows.empty:
            return
        
        cad_row = cad_rows.iloc[0]
        parcel_geom = self._get_parcel_geometry(cadastral_number, cadastral_df)
        
        if parcel_geom is None or parcel_geom.is_empty:
            return
        
        # Геометрии нарушений
        geoms = []
        if not v.empty and 'WKT геометрии' in v.columns:
            for w in v['WKT геометрии'].dropna().astype(str).tolist():
                try:
                    geom = shapely_wkt.loads(w)
                    geoms.append(geom)
                except Exception:
                    pass

        viol_union_for_aoi = unary_union(geoms) if geoms else None
        
        # Область интереса с буфером
        area_of_interest = parcel_geom
        buffer_distance = 0
        if viol_union_for_aoi is not None and not viol_union_for_aoi.is_empty:
            all_geoms_combined = unary_union([parcel_geom, viol_union_for_aoi])
            buffer_distance = max(
                parcel_geom.bounds[2] - parcel_geom.bounds[0],
                parcel_geom.bounds[3] - parcel_geom.bounds[1]
            ) * 0.3
            area_of_interest = all_geoms_combined.buffer(buffer_distance)
        
        # Соседние участки
        all_cadastral_info = self._get_neighboring_parcels(
            cadastral_number, cadastral_df, area_of_interest, buffer_distance
        )
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        # Перепроецирование и список геометрий для отрисовки
        project = self._setup_projection(proj_string)
        plot_geoms: List = []
        plot_parcel_geom = parcel_geom
        plot_cadastral_info = all_cadastral_info
        if project:
            try:
                plot_parcel_geom = transform(project, plot_parcel_geom)
                if geoms:
                    if merge_violations_per_parcel:
                        u = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                        if not u.is_empty:
                            plot_geoms = [transform(project, u)]
                    else:
                        plot_geoms = [
                            transform(project, g) for g in geoms if g is not None and not g.is_empty
                        ]
                plot_cadastral_info = []
                for info in all_cadastral_info:
                    ni = dict(info)
                    ni['geom'] = transform(project, info['geom'])
                    plot_cadastral_info.append(ni)
            except Exception as e:
                logger.exception("Обзорная карта: ошибка перепроецирования: %s", e)
                return
        else:
            if geoms:
                if merge_violations_per_parcel:
                    u = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                    plot_geoms = [u] if not u.is_empty else []
                else:
                    plot_geoms = [g for g in geoms if g is not None and not g.is_empty]
        
        # Если после перепроекции участок ушёл из области подложки — откатываемся в исходные координаты.
        if project and not self._is_geometry_near_background(plot_parcel_geom):
            logger.warning(
                "Overview-карта: после перепроекции геометрия вне области подложки; "
                "использую исходные координаты без pyproj-трансформации."
            )
            project = None
            plot_parcel_geom = parcel_geom
            plot_cadastral_info = all_cadastral_info
            if geoms:
                if merge_violations_per_parcel:
                    u = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                    plot_geoms = [u] if not u.is_empty else []
                else:
                    plot_geoms = [g for g in geoms if g is not None and not g.is_empty]
        
        # Подложка
        self._draw_background(ax)
        
        # Все участки
        for info in plot_cadastral_info:
            if info['is_main']:
                self._plot_shapely(ax, info['geom'], facecolor='lightblue', edgecolor='blue', linewidth=2.0, alpha=0.4)
            else:
                self._plot_shapely(ax, info['geom'], facecolor='lightyellow', edgecolor='orange', linewidth=1.5, alpha=0.3)
        
        # Подпись участка
        cadastral_area = self._get_cadastral_area(cad_row)
        if merge_violations_per_parcel and plot_geoms:
            total_violation_area = float(plot_geoms[0].area)
        elif not v.empty and 'Площадь нарушения, м²' in v.columns:
            total_violation_area = float(v['Площадь нарушения, м²'].fillna(0).sum())
        else:
            total_violation_area = 0.0
        
        # Большую сводку участка фиксируем в углу осей, чтобы не перекрывала геометрию/точки.
        ax.text(
            0.03,
            0.05,
            f"КН: {cadastral_number}\nПлощадь участка: {cadastral_area:.2f} м²\nПлощадь нарушений: {total_violation_area:.2f} м²",
            transform=ax.transAxes,
            fontsize=8.5,
            color='blue',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='blue', alpha=0.88),
            zorder=12,
        )
        
        for g in plot_geoms:
            if g is None or g.is_empty:
                continue
            self._plot_shapely(ax, g, facecolor='red', edgecolor='darkred', linewidth=1.0, alpha=0.5)
        
        # Нумерация точек нарушений (координаты уже упрощены в Excel)
        self._draw_violation_labels(
            ax, cadastral_number, violations_df, viol_coords_df,
            project,
        )
        
        # Масштаб
        self._set_overview_scale(ax, plot_cadastral_info, plot_geoms)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Обзорная карта участка {cadastral_number}', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Сохранение
        self._save_figure(fig, save_path, cadastral_number, 'overview', figsize=(6.0, 6.0))
    
    # === Вспомогательные методы ===
    
    def _get_parcel_geometry(self, cadastral_number: str, cadastral_df: pd.DataFrame) -> Optional[Polygon]:
        """Извлекает геометрию участка из DataFrame."""
        if 'WKT геометрии' not in cadastral_df.columns:
            return None
        
        cad_rows = cadastral_df[cadastral_df['Кадастровый номер'] == cadastral_number]
        if cad_rows.empty:
            return None
        
        try:
            return shapely_wkt.loads(str(cad_rows.iloc[0]['WKT геометрии']))
        except Exception:
            return None
    
    def _setup_projection(self, proj_string: Optional[str]):
        """Настраивает трансформацию координат."""
        if self.background is None or proj_string is None:
            return None
        
        if not PYPROJ_AVAILABLE:
            logger.warning("Пакет 'pyproj' не найден. Пропускаю перепроецирование.")
            return None
        
        try:
            _, _, dest_crs = self.background
            if dest_crs is None:
                return None
            dest_crs = _normalize_map_dest_crs(dest_crs)
            src_crs = pyproj.CRS.from_proj4(proj_string)
            if _same_tmerc_false_easting_mismatch(src_crs, dest_crs):
                logger.info(
                    "Карты: tmerc с разным +x_0 у векторов и подложки (как 4250000 vs 250000 в "
                    "метаданных) — координаты уже согласованы с geotransform, перепроецирование отключено."
                )
                return None
            transformer = pyproj.Transformer.from_crs(src_crs, dest_crs, always_xy=True)
            return transformer.transform
        except Exception as e:
            logger.warning(f"Не удалось создать трансформацию координат: {e}")
            return None
    
    def _draw_background(self, ax):
        """Рисует фоновое изображение."""
        if self.background is None:
            return
        
        try:
            bg_img, bg_extent, _ = self.background
            if bg_img is not None:
                if bg_extent is not None:
                    left, right, bottom, top = bg_extent
                    ax.imshow(
                        bg_img,
                        extent=[left, right, bottom, top],
                        alpha=0.85,
                        interpolation='nearest',
                        zorder=0,
                    )
                else:
                    ax.imshow(bg_img, interpolation='nearest', zorder=0)
        except Exception:
            pass

    def _is_geometry_near_background(self, geom, tolerance_ratio: float = 0.02) -> bool:
        """Проверяет, пересекается ли геометрия с extent подложки (с небольшим допуском)."""
        if geom is None or geom.is_empty or self.background is None:
            return True
        try:
            _, bg_extent, _ = self.background
            if bg_extent is None:
                return True
            left, right, bottom, top = bg_extent
            bg_box = box(left, bottom, right, top)
            if bg_box.is_empty:
                return True
            span = max(right - left, top - bottom, 1.0)
            tol = span * float(max(0.0, tolerance_ratio))
            return bool(geom.intersects(bg_box.buffer(tol)))
        except Exception:
            return True
    
    def _draw_centroids(self, ax, violations_df: pd.DataFrame, project):
        """Рисует центроиды нарушений."""
        if 'Центроид X' not in violations_df.columns or 'Центроид Y' not in violations_df.columns:
            return
        
        for _, row in violations_df.iterrows():
            try:
                cx = pd.to_numeric(row['Центроид X'], errors='coerce')
                cy = pd.to_numeric(row['Центроид Y'], errors='coerce')
                if pd.notna(cx) and pd.notna(cy):
                    if project:
                        cx, cy = project(cx, cy)
                    ax.plot(cx, cy, marker='+', markersize=6, color='yellow', markeredgewidth=1.5)
            except Exception:
                pass
    
    def _draw_violation_points(
        self, ax, cadastral_number: str, violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame, project,
    ):
        """Рисует нумерованные точки нарушений по WKT геометрии (независимо от формата координат Excel)."""
        vs_all = violations_df[violations_df['Ближайший кадастровый номер'] == cadastral_number]
        if vs_all.empty or 'WKT геометрии' not in vs_all.columns:
            return
        
        try:
            vs = vs_all.reset_index(drop=True)
            for local_idx, (idx_row, row_v) in enumerate(vs.iterrows(), 1):
                try:
                    geom = shapely_wkt.loads(str(row_v.get('WKT геометрии', '')))
                except Exception:
                    continue
                if geom is None or geom.is_empty:
                    continue
                g = geom if geom.geom_type == 'Polygon' else (list(geom.geoms)[0] if geom.geom_type == 'MultiPolygon' and len(geom.geoms) > 0 else None)
                if g is None or g.is_empty:
                    continue
                pts = list(g.exterior.coords)
                
                if project and pts:
                    try:
                        pts = [project(x, y) for (x, y) in pts]
                    except Exception:
                        continue
                
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                
                for i, (x, y) in enumerate(pts, 1):
                    ax.plot([x], [y], marker='o', markersize=2, color='darkred')
                    ax.text(
                        x, y, str(i), fontsize=8, color='darkred', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)
                    )
        except Exception:
            pass
    
    def _draw_violation_labels(
        self, ax, cadastral_number: str, violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame, project,
    ):
        """Рисует подписи нарушений на обзорной карте."""
        vs_all = violations_df[violations_df['Ближайший кадастровый номер'] == cadastral_number]
        if vs_all.empty or 'WKT геометрии' not in vs_all.columns:
            return
        
        try:
            vs = vs_all.reset_index(drop=True)
            for local_idx, (idx_row, row_v) in enumerate(vs.iterrows(), 1):
                try:
                    geom = shapely_wkt.loads(str(row_v.get('WKT геометрии', '')))
                except Exception:
                    continue
                if geom is None or geom.is_empty:
                    continue
                g = geom if geom.geom_type == 'Polygon' else (list(geom.geoms)[0] if geom.geom_type == 'MultiPolygon' and len(geom.geoms) > 0 else None)
                if g is None or g.is_empty:
                    continue
                pts = list(g.exterior.coords)
                
                if project and pts:
                    try:
                        pts = [project(x, y) for (x, y) in pts]
                    except Exception:
                        continue
                
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                
                violation_area = row_v.get('Площадь нарушения, м²', 0)
                try:
                    violation_area = float(pd.to_numeric(violation_area, errors='coerce'))
                except Exception:
                    violation_area = 0
                
                if pts:
                    centroid_x = sum(p[0] for p in pts) / len(pts)
                    centroid_y = sum(p[1] for p in pts) / len(pts)
                    minx = min(p[0] for p in pts)
                    miny = min(p[1] for p in pts)
                    maxx = max(p[0] for p in pts)
                    maxy = max(p[1] for p in pts)
                    span = max(maxx - minx, maxy - miny, 1.0)
                    # Для мелких нарушений выносим плашку ненамного наружу, чтобы не закрывать полигон.
                    # Дистанцию ограничиваем сверху, чтобы плашка не улетала слишком далеко.
                    ax_minx, ax_maxx = ax.get_xlim()
                    ax_miny, ax_maxy = ax.get_ylim()
                    ax_span = max(ax_maxx - ax_minx, ax_maxy - ax_miny, 1.0)
                    offset = min(max(0.18 * span, 8.0), 0.07 * ax_span)
                    label_x = centroid_x + offset
                    label_y = centroid_y + 0.65 * offset
                    # Небольшой отступ от края осей, чтобы плашка не обрезалась.
                    pad = 0.01 * ax_span
                    label_x = min(max(label_x, ax_minx + pad), ax_maxx - pad)
                    label_y = min(max(label_y, ax_miny + pad), ax_maxy - pad)
                    ax.text(
                        label_x, label_y,
                        f"№{local_idx}\n{violation_area:.1f} м²",
                        fontsize=8.5, color='white', ha='left', va='bottom', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', edgecolor='darkred', alpha=0.78),
                        zorder=9,
                    )
                    ax.plot(
                        [centroid_x, label_x],
                        [centroid_y, label_y],
                        color='darkred',
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=8,
                    )
                
                for i, (x, y) in enumerate(pts, 1):
                    ax.plot([x], [y], marker='o', markersize=3.2, color='yellow', markeredgecolor='darkred', markeredgewidth=0.6, zorder=10)
                    ax.text(x, y, str(i), fontsize=7, color='black', ha='center', va='center', weight='bold', zorder=11)
        except Exception:
            pass
    
    def _get_neighboring_parcels(
        self, cadastral_number: str, cadastral_df: pd.DataFrame, 
        area_of_interest, buffer_distance: float
    ) -> List[dict]:
        """Получает список соседних участков."""
        all_cadastral_info = []
        
        if 'WKT геометрии' not in cadastral_df.columns:
            return all_cadastral_info
        
        for idx, row in cadastral_df.iterrows():
            try:
                cad_num = row.get('Кадастровый номер', '')
                if pd.isna(cad_num):
                    continue
                
                geom = shapely_wkt.loads(str(row['WKT геометрии']))
                if geom and not geom.is_empty:
                    is_main = (str(cad_num) == str(cadastral_number))
                    if is_main or geom.intersects(area_of_interest) or geom.distance(area_of_interest) < buffer_distance * 0.5:
                        all_cadastral_info.append({
                            'number': str(cad_num),
                            'geom': geom,
                            'is_main': is_main
                        })
            except Exception:
                pass
        
        return all_cadastral_info
    
    def _get_cadastral_area(self, cad_row) -> float:
        """Извлекает площадь участка."""
        cadastral_area = cad_row.get('Площадь, м²', 0)
        try:
            return float(pd.to_numeric(cadastral_area, errors='coerce'))
        except Exception:
            return 0.0
    
    def _set_zoom_scale(self, ax, geoms: List, parcel_geom):
        """Устанавливает масштаб для zoom-карты."""
        all_bounds = [g.bounds for g in geoms if g and not g.is_empty]
        if parcel_geom is not None and not parcel_geom.is_empty:
            all_bounds.append(parcel_geom.bounds)
        
        if not all_bounds:
            return
        
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        padding_x = (maxx - minx) * 1.0 if maxx > minx else 50
        padding_y = (maxy - miny) * 1.0 if maxy > miny else 50
        ax.set_xlim(minx - padding_x, maxx + padding_x)
        ax.set_ylim(miny - padding_y, maxy + padding_y)
    
    def _set_overview_scale(self, ax, all_cadastral_info: List[dict], geoms: List):
        """Устанавливает масштаб для обзорной карты."""
        all_bounds = []
        for info in all_cadastral_info:
            if info['geom'] and not info['geom'].is_empty:
                all_bounds.append(info['geom'].bounds)
        for g in geoms:
            if g and not g.is_empty:
                all_bounds.append(g.bounds)
        
        if not all_bounds:
            return
        
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        padding_x = (maxx - minx) * 0.2
        padding_y = (maxy - miny) * 0.2
        ax.set_xlim(minx - padding_x, maxx + padding_x)
        ax.set_ylim(miny - padding_y, maxy + padding_y)
    
    def _save_figure(
        self, fig, save_path: str, cadastral_number: str, map_type: str,
        figsize: Tuple[float, float] = (4.0, 4.0), dpi: int = 200
    ):
        """Сохраняет фигуру с контролем размера."""
        max_pixels = 65536
        fig.set_size_inches(*figsize)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                plt.tight_layout(pad=0.5 if map_type == 'zoom' else 0.7)
            except Exception:
                pass
        
        try:
            if PIL_AVAILABLE:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=dpi)
                buf.seek(0)
                
                img = PILImage.open(buf)
                width, height = img.size
                
                if width > max_pixels or height > max_pixels:
                    scale = min(max_pixels / width, max_pixels / height)
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    logger.debug(f"Pillow {map_type}: уменьшаю {width}x{height} до {new_size[0]}x{new_size[1]}")
                    try:
                        img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                    except AttributeError:
                        img = img.resize(new_size, PILImage.LANCZOS)
                
                img.save(save_path, 'PNG', optimize=True)
                buf.close()
            else:
                fig.savefig(save_path, dpi=dpi)
        except Exception as e:
            logger.warning(f"Не удалось сохранить {map_type}-карту для {cadastral_number}: {e}")
        finally:
            plt.close(fig)
    
    @staticmethod
    def _plot_shapely(ax, geom, facecolor='none', edgecolor='black', linewidth=1.0, alpha=1.0):
        """Рисует геометрию Shapely на matplotlib axes."""
        if geom is None or geom.is_empty:
            return
        
        if isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                MapGenerator._plot_shapely(ax, g, facecolor, edgecolor, linewidth, alpha)
            return

        if isinstance(geom, GeometryCollection):
            polys = [
                g for g in geom.geoms
                if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty
            ]
            if not polys:
                return
            MapGenerator._plot_shapely(
                ax, unary_union(polys), facecolor, edgecolor, linewidth, alpha
            )
            return
        
        if isinstance(geom, Polygon):
            exterior = np.asarray(geom.exterior.coords)
            verts = exterior.tolist()
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(exterior) - 2) + [MplPath.CLOSEPOLY]
            
            for interior in geom.interiors:
                ring = np.asarray(interior.coords)
                verts.extend(ring.tolist())
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
            
            path = MplPath(verts, codes)
            patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
            ax.add_patch(patch)
