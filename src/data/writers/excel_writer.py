"""Writer для экспорта в Excel."""

import logging
import pandas as pd
from typing import List, Tuple
from pathlib import Path
from datetime import datetime

from .base import DataWriter
from ...domain.models import DetectedObject, CadastralParcel, Violation

logger = logging.getLogger(__name__)


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


def _present_xy(x: float, y: float) -> Tuple[float, float]:
    """
    Преобразует координаты для отображения.
    Меняет X и Y местами и добавляет смещение.
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
    ) -> None:
        """
        Записывает отчёт в Excel.

        Ожидается, что ``cadastral_parcels`` и ``violations`` уже приведены к виду
        для карт и per_parcel (упрощённые контуры тем же tolerance, что в пайплайне).
        """
        logger.info(f"Создание Excel отчёта: {path}")
        
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
            row = {
                '№ п/п': i,
                'Кадастровый номер': p.cadastral_number,
                'Площадь, м²': round(p.area_sqm, 2),
                'Площадь, га': round(p.area_sqm / 10000, 6),
                'Центроид X': round(_present_xy(cxy[0], cxy[1])[0], 6),
                'Центроид Y': round(_present_xy(cxy[0], cxy[1])[1], 6),
                'Периметр, м': round(g.length, 2)
            }
            
            # Bounds
            bounds = g.bounds
            row.update({
                'Мин X': round(bounds[0], 6),
                'Мин Y': round(bounds[1], 6),
                'Макс X': round(bounds[2], 6),
                'Макс Y': round(bounds[3], 6)
            })
            
            # Первые 10 точек контура
            coords = list(g.exterior.coords)[:10]
            for j, (x, y) in enumerate(coords, 1):
                px, py = _present_xy(x, y)
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
            row = {
                '№ нарушения': i,
                'Площадь нарушения, м²': round(v.violation_area, 2),
                'Площадь нарушения, га': round(v.violation_area / 10000, 6),
                'Площадь исходного объекта, м²': round(v.original_object_area, 2),
                'Центроид X': round(_present_xy(cxy[0], cxy[1])[0], 6),
                'Центроид Y': round(_present_xy(cxy[0], cxy[1])[1], 6),
                'Ближайший кадастровый номер': v.cadastral_number,
                'Периметр нарушения, м': round(g.length, 2)
            }
            
            # Bounds с преобразованием координат
            bounds = g.bounds
            row.update({
                'Мин X': round(_present_xy(bounds[0], bounds[1])[0], 6),
                'Мин Y': round(_present_xy(bounds[0], bounds[1])[1], 6),
                'Макс X': round(_present_xy(bounds[2], bounds[3])[0], 6),
                'Макс Y': round(_present_xy(bounds[2], bounds[3])[1], 6)
            })
            
            # Координаты контура (первые 10 точек)
            try:
                coords = _violation_coord_rows(g)[:10]
                for j, (x, y) in enumerate(coords, 1):
                    px, py = _present_xy(x, y)
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
                px, py = _present_xy(x, y)
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
                    px, py = _present_xy(x, y)
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
