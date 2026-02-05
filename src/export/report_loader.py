"""
Загрузка данных из Excel-отчёта для генерации документов по участкам.
"""

import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class ReportLoader:
    """Загрузчик данных из Excel-отчёта комплексного анализа."""
    
    @staticmethod
    def load_frames(report_path: str) -> Dict[str, pd.DataFrame]:
        """
        Читает необходимые листы из Excel-отчета.
        
        Args:
            report_path: Путь к файлу report.xlsx
            
        Returns:
            Словарь с DataFrame из разных листов отчёта:
            - cadastral_df: Кадастровые участки
            - violations_df: Нарушения
            - coords_df: Координаты участков
            - viol_coords_df: Координаты нарушений
        """
        logger.info(f"Чтение отчёта: {report_path}")
        xls = pd.ExcelFile(report_path)
        logger.info(f"Найдены листы: {xls.sheet_names}")

        # Листы по именам из ExcelWriter
        try:
            cadastral_df = pd.read_excel(xls, sheet_name='2. Кадастровые участки')
            logger.info(f"'2. Кадастровые участки': {len(cadastral_df)} строк")
        except Exception as e:
            logger.error(f"Не удалось прочитать лист '2. Кадастровые участки': {e}")
            raise

        try:
            violations_df = pd.read_excel(xls, sheet_name='3. Нарушения')
            logger.info(f"'3. Нарушения': {len(violations_df)} строк")
        except Exception as e:
            logger.error(f"Не удалось прочитать лист '3. Нарушения': {e}")
            raise

        try:
            coords_df = pd.read_excel(xls, sheet_name='4. Координаты участков')
            logger.info(f"'4. Координаты участков': {len(coords_df)} строк")
        except Exception as e:
            logger.error(f"Не удалось прочитать лист '4. Координаты участков': {e}")
            raise

        try:
            viol_coords_df = pd.read_excel(xls, sheet_name='5. Координаты нарушений')
            logger.info(f"'5. Координаты нарушений': {len(viol_coords_df)} строк")
        except Exception as e:
            logger.error(f"Не удалось прочитать лист '5. Координаты нарушений': {e}")
            raise

        return {
            'cadastral_df': cadastral_df,
            'violations_df': violations_df,
            'coords_df': coords_df,
            'viol_coords_df': viol_coords_df,
        }
