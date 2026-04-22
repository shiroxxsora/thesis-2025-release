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

        # Листы по именам из ExcelWriter (некоторые могут отсутствовать, если данных 0)
        def _read_or_empty(name: str) -> pd.DataFrame:
            try:
                df = pd.read_excel(xls, sheet_name=name)
                logger.info("'%s': %s строк", name, len(df))
                return df
            except Exception as e:
                logger.warning("Лист '%s' не прочитан (%s). Возвращаю пустой DataFrame.", name, e)
                return pd.DataFrame()

        cadastral_df = _read_or_empty('2. Кадастровые участки')
        violations_df = _read_or_empty('3. Нарушения')
        coords_df = _read_or_empty('4. Координаты участков')
        viol_coords_df = _read_or_empty('5. Координаты нарушений')

        return {
            'cadastral_df': cadastral_df,
            'violations_df': violations_df,
            'coords_df': coords_df,
            'viol_coords_df': viol_coords_df,
        }
