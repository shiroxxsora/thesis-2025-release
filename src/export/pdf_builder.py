"""
Генерация PDF-документов по отдельным участкам.
"""

import logging
import os
from typing import List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .coordinate_presenter import format_float, compute_distances

logger = logging.getLogger(__name__)


class PDFBuilder:
    """Построитель PDF-документов для отдельных участков."""
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Args:
            font_path: Путь к TTF-шрифту с кириллицей (опционально)
        """
        self.font_name = self._setup_font(font_path)
    
    def _setup_font(self, font_path: Optional[str] = None) -> str:
        """Регистрирует шрифт с поддержкой кириллицы."""
        font_candidates: List[str] = []
        if font_path:
            font_candidates.append(font_path)
        
        # Типичные пути Windows/Linux
        font_candidates += [
            os.path.join('C:\\Windows\\Fonts', 'arial.ttf'),
            os.path.join('C:\\Windows\\Fonts', 'calibri.ttf'),
            os.path.join('C:\\Windows\\Fonts', 'times.ttf'),
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        ]
        
        for candidate in font_candidates:
            if candidate and os.path.exists(candidate):
                try:
                    pdfmetrics.registerFont(TTFont('CyrillicTT', candidate))
                    logger.info(f"Использую шрифт: {candidate}")
                    return 'CyrillicTT'
                except Exception as e:
                    logger.warning(f"Не удалось зарегистрировать шрифт {candidate}: {e}")
        
        logger.warning("Кириллический TTF не найден. Используется стандартный шрифт (может не поддерживать кириллицу)")
        return 'Helvetica'
    
    def build_story(
        self,
        cadastral_number: str,
        cadastral_df: pd.DataFrame,
        coords_df: pd.DataFrame,
        violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame,
        image_path: Optional[str] = None,
        overview_path: Optional[str] = None,
    ) -> List:
        """
        Формирует содержимое PDF-документа для участка.
        
        Args:
            cadastral_number: Кадастровый номер участка
            cadastral_df: DataFrame с кадастровыми участками
            coords_df: DataFrame с координатами участков
            violations_df: DataFrame с нарушениями
            viol_coords_df: DataFrame с координатами нарушений
            image_path: Путь к zoom-карте (опционально)
            overview_path: Путь к обзорной карте (опционально)
            
        Returns:
            Список элементов ReportLab для построения PDF
        """
        styles = getSampleStyleSheet()
        heading = ParagraphStyle(
            name='HeadingCenter',
            parent=styles['Heading2'],
            alignment=1,
            fontName=self.font_name
        )
        normal = ParagraphStyle(
            name='NormalCyr',
            parent=styles['Normal'],
            fontName=self.font_name
        )
        
        story: List = []
        
        # Заголовок
        story.append(Paragraph('СХЕМА', heading))
        story.append(Paragraph(
            'обмера земельного участка',
            ParagraphStyle(name='Sub', parent=normal, alignment=1)
        ))
        story.append(Spacer(1, 8))
        
        # Данные по участку
        story.append(Paragraph(f"Кадастровый номер: <b>{cadastral_number}</b>", normal))
        story.append(Spacer(1, 6))
        
        # Сводка по нарушениям
        v = violations_df[violations_df['Ближайший кадастровый номер'] == cadastral_number]
        if not v.empty:
            self._add_violations_summary(story, v, normal, heading, viol_coords_df)
        
        # Обзорная карта
        if overview_path and os.path.exists(overview_path):
            self._add_image(story, overview_path, 'Обзорная карта участка', heading)
        
        # Детальная карта
        if image_path and os.path.exists(image_path):
            self._add_image(story, image_path, 'Детальная карта нарушений', heading)
        
        return story
    
    def _add_violations_summary(
        self,
        story: List,
        violations_df: pd.DataFrame,
        normal: ParagraphStyle,
        heading: ParagraphStyle,
        viol_coords_df: pd.DataFrame,
    ):
        """Добавляет сводку по нарушениям (координаты как в Excel — уже после present_xy)."""
        sum_violation = float(violations_df['Площадь нарушения, м²'].fillna(0).sum())
        story.append(Paragraph(f"Количество нарушений: <b>{len(violations_df)}</b>", normal))
        story.append(Paragraph(
            f"Площадь нарушений по участку (объединение контуров): <b>{format_float(sum_violation, 2)}</b> м²",
            normal,
        ))
        story.append(Spacer(1, 6))
        
        # Краткая таблица нарушений
        v_table = [['№', 'Площадь, м²', 'Центроид X', 'Центроид Y']]
        for i, (_, r) in enumerate(violations_df.iterrows(), 1):
            try:
                area_val = pd.to_numeric(r.get('Площадь нарушения, м²', 0), errors='coerce')
            except Exception:
                area_val = 0
            try:
                cx = pd.to_numeric(r.get('Центроид X', 0), errors='coerce')
            except Exception:
                cx = 0
            try:
                cy = pd.to_numeric(r.get('Центроид Y', 0), errors='coerce')
            except Exception:
                cy = 0
            cx_p = float(cx) if pd.notna(cx) else 0.0
            cy_p = float(cy) if pd.notna(cy) else 0.0
            v_table.append([
                str(i),
                format_float(area_val if pd.notna(area_val) else 0, 2),
                format_float(cx_p, 2),
                format_float(cy_p, 2),
            ])
        
        v_tbl = Table(v_table, repeatRows=1)
        v_tbl.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f7f7ff')),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
        ]))
        story.append(v_tbl)
        story.append(Spacer(1, 8))
        
        # Таблицы координат по каждому нарушению
        if not viol_coords_df.empty and '№ нарушения' in viol_coords_df.columns:
            self._add_violation_coords_tables(story, violations_df, viol_coords_df, heading)
    
    def _add_violation_coords_tables(
        self,
        story: List,
        violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame,
        heading: ParagraphStyle,
    ):
        """Добавляет таблицы координат (как в Excel, без повторного present_xy)."""
        v_reset = violations_df.reset_index(drop=True)
        for local_idx, (idx_row, row_v) in enumerate(v_reset.iterrows(), 1):
            global_v_num = row_v.get('№ нарушения', idx_row + 1)
            try:
                global_v_num = int(global_v_num)
            except Exception:
                global_v_num = idx_row + 1
            
            sub = viol_coords_df[viol_coords_df['№ нарушения'] == global_v_num].copy()
            if sub.empty:
                continue
            
            try:
                sub.sort_values(by='Номер точки', inplace=True)
                pts = list(zip(
                    pd.to_numeric(sub['X'], errors='coerce').astype(float),
                    pd.to_numeric(sub['Y'], errors='coerce').astype(float)
                ))
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                
                logger.debug(f"PDF: Нарушение №{local_idx} (глобальное #{global_v_num}): {len(pts)} точек")
                dists = compute_distances(pts)
                
                story.append(Paragraph(f"Координаты нарушения № {local_idx}", heading))
                vt = [['Обозначение\nточки', 'X', 'Y', 'Расстояние до точки, м.']]
                for i, (x, y) in enumerate(pts, 1):
                    dist_idx = (i - 2) % len(dists)
                    px, py = float(x), float(y)
                    vt.append([
                        str(i),
                        format_float(px, 2),
                        format_float(py, 2),
                        format_float(dists[dist_idx], 2)
                    ])
                
                vt_tbl = Table(vt, repeatRows=1)
                vt_tbl.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef7ee')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), self.font_name),
                ]))
                story.append(vt_tbl)
                story.append(Spacer(1, 8))
            except Exception:
                continue
    
    def _add_image(self, story: List, image_path: str, title: str, heading: ParagraphStyle):
        """Добавляет изображение в документ."""
        try:
            story.append(Paragraph(title, heading))
            story.append(Spacer(1, 6))
            img = Image(image_path)
            max_w, max_h = 520, 520
            img._restrictSize(max_w, max_h)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception:
            pass
    
    def generate(
        self,
        output_path: str,
        cadastral_number: str,
        cadastral_df: pd.DataFrame,
        coords_df: pd.DataFrame,
        violations_df: pd.DataFrame,
        viol_coords_df: pd.DataFrame,
        image_path: Optional[str] = None,
        overview_path: Optional[str] = None,
    ) -> str:
        """
        Генерирует PDF-документ для участка.
        
        Args:
            output_path: Путь для сохранения PDF
            cadastral_number: Кадастровый номер участка
            cadastral_df: DataFrame с кадастровыми участками
            coords_df: DataFrame с координатами участков
            violations_df: DataFrame с нарушениями
            viol_coords_df: DataFrame с координатами нарушений
            image_path: Путь к zoom-карте (опционально)
            overview_path: Путь к обзорной карте (опционально)
            
        Returns:
            Путь к созданному PDF-файлу
        """
        story = self.build_story(
            cadastral_number,
            cadastral_df,
            coords_df,
            violations_df,
            viol_coords_df,
            image_path,
            overview_path,
        )
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        doc.build(story)
        
        logger.info(f"PDF создан: {output_path}")
        return output_path
