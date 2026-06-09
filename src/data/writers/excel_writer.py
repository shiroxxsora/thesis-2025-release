"""Writer для экспорта в Excel."""

import logging
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .base import DataWriter
from ...domain.models import DetectedObject, CadastralParcel, Violation
from ...export.coordinate_presenter import present_xy, present_xy_extrema_from_bounds

logger = logging.getLogger(__name__)

try:
    import pyproj
    _PYPROJ_AVAILABLE = True
except Exception:
    _PYPROJ_AVAILABLE = False

# МСК-03 (Улан-Удэ), "малый" false easting (x_0=250000) для согласованного документного вида:
# после present_xy получаем ожидаемые координаты порядка X~5xx xxx, Y~4 1xx xxx.
_DOC_MSK_SCENE_PROJ4 = (
    "+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 +x_0=250000 "
    "+y_0=-5211057.63 +ellps=krass +towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 "
    "+units=m +no_defs"
)

def _violation_coord_rows(geom):
    """Точки контура нарушения: один или несколько полигонов (MultiPolygon)."""
    if geom.geom_type == 'Polygon':
        return list(geom.exterior.coords)
    if geom.geom_type == 'MultiPolygon':
        pts = []
        for p in geom.geoms:
            pts.extend(list(p.exterior.coords))
        return pts
    return []


class ExcelWriter(DataWriter):
    """Класс для записи отчётов в Excel."""
    
    SHEET_NAMES = {
        'summary': '1. Сводка',
        'cadastral': '2. Кадастровые участки',
        'violations': '3. Нарушения',
        'parcel_coords': '4. Координаты участков',
        'violation_coords': '5. Координаты нарушений'
    }
    
    def write(
        self,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel],
        violations: List[Violation],
        path: str,
        proj_false_easting_x0: Optional[float] = None,
        use_msk_coordinate_presentation: bool = True,
        source_proj4: Optional[str] = None,
        force_msk_for_all_inputs: bool = False,
    ) -> None:
        """
        Записывает отчёт в Excel.

        Ожидается, что ``cadastral_parcels`` и ``violations`` уже приведены к виду
        для карт и per_parcel (упрощённые контуры тем же tolerance, что в пайплайне).

        proj_false_easting_x0: +x_0 из PROJ растра; для МСК с x_0≠4250000 подправляет
        север для колонок в «зонной» записи документов.
        use_msk_coordinate_presentation: False — UTM/географ. СК: колонки X/Y = метры/градусы
        в системе сцены без 4M-схемы МСК.
        source_proj4: PROJ4 исходной сцены/растра (нужен для UTM->МСК в принудительном режиме).
        force_msk_for_all_inputs: True — всегда выдавать документные МСК-координаты, даже для UTM.
        """
        logger.info(f"Создание Excel отчёта: {path}")
        self._proj_x0 = proj_false_easting_x0
        self._use_msk_docs = use_msk_coordinate_presentation
        self._coord_transform = None
        if force_msk_for_all_inputs:
            self._use_msk_docs = True
            # Для формулы present_xy ориентируемся на "малый" x0.
            self._proj_x0 = 250000.0
            self._coord_transform = self._build_transform_to_doc_msk(source_proj4)
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                # 1. Сводка
                self._write_summary(writer, detected_objects, cadastral_parcels, violations)
                
                # 2. Кадастровые участки
                self._write_cadastral(writer, cadastral_parcels)
                
                # 3. Нарушения
                self._write_violations(writer, violations)
                
                # 4. Координаты участков
                self._write_parcel_coords(writer, cadastral_parcels)
                
                # 5. Координаты нарушений
                self._write_violation_coords(writer, violations)
                
                # Форматирование столбцов (автоширина)
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except Exception:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Статистика
            total_cadastral_area = sum(obj.area_sqm for obj in cadastral_parcels)
            total_violation_area = sum(v.violation_area for v in violations)
            
            logger.info(f"Excel отчёт создан: {path}")
            logger.info(f"Отчёт содержит {len(cadastral_parcels)} кадастровых участков и {len(violations)} нарушений")
            logger.info(f"Общая площадь кадастровых участков: {total_cadastral_area:.2f} м² ({total_cadastral_area/10000:.4f} га)")
            logger.info(
                f"Общая площадь нарушений (union по участку): {total_violation_area:.2f} м² "
                f"({total_violation_area/10000:.4f} га)"
            )
            if total_cadastral_area > 0:
                logger.info(f"Процент нарушений: {(total_violation_area/total_cadastral_area*100):.2f}%")
        finally:
            self._proj_x0 = None
            self._use_msk_docs = True
            self._coord_transform = None

    def _build_transform_to_doc_msk(self, source_proj4: Optional[str]):
        """Создаёт трансформер source->MSK(document scene)."""
        if not _PYPROJ_AVAILABLE:
            logger.warning("pyproj недоступен: принудительный МСК-режим работает без преобразования CRS.")
            return None
        if not source_proj4:
            return None
        try:
            src = pyproj.CRS.from_proj4(source_proj4)
            dst = pyproj.CRS.from_proj4(_DOC_MSK_SCENE_PROJ4)
            if src == dst:
                return None
            tr = pyproj.Transformer.from_crs(src, dst, always_xy=True)
            logger.info("Excel: включено преобразование координат source CRS -> МСК document.")
            return tr.transform
        except Exception as e:
            logger.warning(f"Не удалось включить source->MSK преобразование для Excel: {e}")
            return None

    def _present(self, x: float, y: float) -> Tuple[float, float]:
        """Преобразует точку к документному представлению (с опциональным source->MSK)."""
        xx, yy = x, y
        if self._coord_transform is not None:
            try:
                xx, yy = self._coord_transform(float(x), float(y))
            except Exception:
                xx, yy = x, y
        return present_xy(xx, yy, self._proj_x0, self._use_msk_docs)

    def _present_bounds(self, bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Экстремумы bbox в документных осях с учетом source->MSK при необходимости."""
        if self._coord_transform is None:
            return present_xy_extrema_from_bounds(bounds, self._proj_x0, self._use_msk_docs)
        minx, miny, maxx, maxy = bounds
        corners = ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))
        pts = [self._present(x, y) for (x, y) in corners]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def _write_summary(self, writer, detected, cadastral, violations):
        """Записывает сводку."""
        total_cadastral_area = sum(obj.area_sqm for obj in cadastral)
        total_violation_area = sum(v.violation_area for v in violations)
        
        summary_data = [
            ['Параметр', 'Значение'],
            ['Дата анализа', datetime.now().strftime('%d.%m.%Y %H:%M')],
            ['Кадастровых участков', len(cadastral)],
            ['Найдено нарушений', len(violations)],
            ['Общая площадь кадастровых участков, м²', round(total_cadastral_area, 2)],
            ['Общая площадь кадастровых участков, га', round(total_cadastral_area / 10000, 6)],
            ['Общая площадь нарушений, м² (union по участку)', round(total_violation_area, 2)],
            ['Общая площадь нарушений, га (union по участку)', round(total_violation_area / 10000, 6)],
            ['Процент нарушений от общей площади, %', 
             round((total_violation_area / total_cadastral_area * 100) if total_cadastral_area > 0 else 0, 2)]
        ]
        
        df = pd.DataFrame(summary_data, columns=['Параметр', 'Значение'])
        df.to_excel(writer, sheet_name=self.SHEET_NAMES['summary'], index=False)
    
    def _write_cadastral(self, writer, cadastral_parcels):
        """Записывает кадастровые участки (геометрия без упрощения)."""
        if not cadastral_parcels:
            return
        
        data = []
        for i, p in enumerate(cadastral_parcels, 1):
            g = p.geometry
            cxy = g.centroid.coords[0]
            ccx, ccy = self._present(cxy[0], cxy[1])
            row = {
                '№ п/п': i,
                'Кадастровый номер': p.cadastral_number,
                'Площадь, м²': round(p.area_sqm, 2),
                'Площадь, га': round(p.area_sqm / 10000, 6),
                'Центроид X': round(ccx, 6),
                'Центроид Y': round(ccy, 6),
                'Периметр, м': round(g.length, 2)
            }
            
            # Bounds: экстремумы по четырём углам bbox после present_xy
            bounds = g.bounds
            mix, miy, mxx, mxy = self._present_bounds(bounds)
            row.update({
                'Мин X': round(mix, 6),
                'Мин Y': round(miy, 6),
                'Макс X': round(mxx, 6),
                'Макс Y': round(mxy, 6),
            })
            
            # Первые 10 точек контура
            coords = list(g.exterior.coords)[:10]
            for j, (x, y) in enumerate(coords, 1):
                px, py = self._present(x, y)
                row[f'Точка {j} X'] = round(px, 6)
                row[f'Точка {j} Y'] = round(py, 6)
            
            # WKT геометрии
            row['WKT геометрии'] = g.wkt
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=self.SHEET_NAMES['cadastral'], index=False)
    
    def _write_violations(self, writer, violations):
        """Записывает нарушения (геометрия как передано — уже упрощённая для документов)."""
        if not violations:
            return
        
        data = []
        for i, v in enumerate(violations, 1):
            g = v.geometry
            cxy = v.centroid if v.centroid is not None else g.centroid.coords[0]
            vcx, vcy = self._present(cxy[0], cxy[1])
            row = {
                '№ нарушения': i,
                'Площадь нарушения, м²': round(v.violation_area, 2),
                'Площадь нарушения, га': round(v.violation_area / 10000, 6),
                'Площадь исходного объекта, м²': round(v.original_object_area, 2),
                'Центроид X': round(vcx, 6),
                'Центроид Y': round(vcy, 6),
                'Ближайший кадастровый номер': v.cadastral_number,
                'Периметр нарушения, м': round(g.length, 2)
            }
            
            bounds = g.bounds
            mix, miy, mxx, mxy = self._present_bounds(bounds)
            row.update({
                'Мин X': round(mix, 6),
                'Мин Y': round(miy, 6),
                'Макс X': round(mxx, 6),
                'Макс Y': round(mxy, 6),
            })
            
            # Координаты контура (первые 10 точек)
            try:
                coords = _violation_coord_rows(g)[:10]
                for j, (x, y) in enumerate(coords, 1):
                    px, py = self._present(x, y)
                    row[f'Точка {j} X'] = round(px, 6)
                    row[f'Точка {j} Y'] = round(py, 6)
            except Exception:
                pass
            
            # WKT геометрии
            row['WKT геометрии'] = g.wkt
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=self.SHEET_NAMES['violations'], index=False)
    
    def _write_parcel_coords(self, writer, cadastral_parcels):
        """Записывает координаты участков."""
        if not cadastral_parcels:
            return
        
        data = []
        for p in cadastral_parcels:
            coords = list(p.geometry.exterior.coords)
            for i, (x, y) in enumerate(coords, 1):
                px, py = self._present(x, y)
                data.append({
                    'Кадастровый номер': p.cadastral_number,
                    'Номер точки': i,
                    'X': round(px, 6),
                    'Y': round(py, 6)
                })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=self.SHEET_NAMES['parcel_coords'], index=False)
    
    def _write_violation_coords(self, writer, violations):
        """Записывает координаты нарушений."""
        if not violations:
            return
        
        data = []
        for i, v in enumerate(violations, 1):
            try:
                coords = _violation_coord_rows(v.geometry)
                for j, (x, y) in enumerate(coords, 1):
                    px, py = self._present(x, y)
                    data.append({
                        '№ нарушения': i,
                        'Номер точки': j,
                        'X': round(px, 6),
                        'Y': round(py, 6)
                    })
            except Exception:
                pass
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=self.SHEET_NAMES['violation_coords'], index=False)
