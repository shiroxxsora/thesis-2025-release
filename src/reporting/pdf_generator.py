"""Генератор PDF отчётов."""

import logging
from typing import List
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Класс для генерации PDF отчётов."""
    
    def __init__(self, output_dir: str):
        """
        Инициализация генератора.
        
        Args:
            output_dir: Директория для сохранения отчётов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_fonts()
    
    def _setup_fonts(self):
        """Настройка шрифтов с поддержкой кириллицы."""
        try:
            # Пробуем загрузить DejaVuSans
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'C:/Windows/Fonts/dejavusans.ttf'))
            self.font_name = 'DejaVuSans'
            logger.debug("Использован шрифт DejaVuSans")
        except Exception:
            try:
                # Альтернативный шрифт Arial
                pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
                self.font_name = 'Arial'
                logger.debug("Использован шрифт Arial")
            except Exception:
                # Если не удалось загрузить шрифт, используем стандартный
                self.font_name = 'Helvetica'
                logger.warning("Не удалось загрузить шрифт с поддержкой кириллицы, используется Helvetica")
    
    def generate(
        self,
        detected_objects: List[DetectedObject],
        cadastral_parcels: List[CadastralParcel],
        violations: List[Violation],
        geotiff_data: GeoTiffData
    ) -> str:
        """
        Генерирует PDF отчёт.
        
        Args:
            detected_objects: Обнаруженные объекты
            cadastral_parcels: Кадастровые участки
            violations: Нарушения
            geotiff_data: Данные GeoTIFF
            
        Returns:
            Путь к созданному файлу
        """
        logger.info("Генерация PDF отчёта...")
        
        # Подсчет статистики
        total_detected_area = sum(obj.area_sqm for obj in detected_objects)
        total_cadastral_area = sum(obj.area_sqm for obj in cadastral_parcels)
        total_violation_area = sum(v.violation_area for v in violations)
        
        # Создание документа
        report_path = self.output_dir / 'comprehensive_report.pdf'
        doc = SimpleDocTemplate(str(report_path), pagesize=A4)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Заголовок
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=self.font_name
        )
        story.append(Paragraph("Отчет по анализу нарушений землепользования", title_style))
        story.append(Spacer(1, 20))
        
        # Дата
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            fontName=self.font_name
        )
        story.append(Paragraph(f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}", date_style))
        story.append(Spacer(1, 30))
        
        # Общая статистика
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=self.font_name
        )
        story.append(Paragraph("Общая статистика", heading_style))
        story.append(Spacer(1, 12))
        
        stats_data = [
            ['Параметр', 'Значение'],
            ['Обнаружено объектов', str(len(detected_objects))],
            ['Кадастровых участков', str(len(cadastral_parcels))],
            ['Нарушений', str(len(violations))],
            ['Общая площадь объектов (м²)', f"{total_detected_area:.2f}"],
            ['Общая площадь кадастровых участков (м²)', f"{total_cadastral_area:.2f}"],
            ['Общая площадь нарушений (м²), union по участку', f"{total_violation_area:.2f}"],
            ['Площадь нарушений (га), union по участку', f"{total_violation_area/10000:.4f}"]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), self.font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Детализация нарушений
        if violations:
            story.append(Paragraph("Детализация нарушений", heading_style))
            story.append(Spacer(1, 12))
            
            violation_data = [['№', 'Кадастровый номер', 'Площадь нарушения (м²)', 'Координаты центра']]
            for i, violation in enumerate(violations, 1):
                centroid = violation.centroid
                violation_data.append([
                    str(i),
                    violation.cadastral_number,
                    f"{violation.violation_area:.2f}",
                    f"({centroid[0]:.6f}, {centroid[1]:.6f})"
                ])
            
            violation_table = Table(violation_data)
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self.font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(violation_table)
            story.append(Spacer(1, 20))
        
        # Информация о кадастровых участках
        if cadastral_parcels:
            story.append(Paragraph("Кадастровые участки", heading_style))
            story.append(Spacer(1, 12))
            
            cadastral_data = [['№', 'Кадастровый номер', 'Площадь (м²)', 'Координаты центра']]
            for i, cadastral in enumerate(cadastral_parcels, 1):
                centroid = cadastral.centroid
                cadastral_data.append([
                    str(i),
                    cadastral.cadastral_number,
                    f"{cadastral.area_sqm:.2f}",
                    f"({centroid[0]:.6f}, {centroid[1]:.6f})"
                ])
            
            cadastral_table = Table(cadastral_data)
            cadastral_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self.font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(cadastral_table)
        
        # Сборка документа
        doc.build(story)
        logger.info(f"PDF отчёт сохранён: {report_path}")
        return str(report_path)
