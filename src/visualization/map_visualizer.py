"""Визуализация карт."""

import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import List
from pathlib import Path

from ..domain.models import DetectedObject, CadastralParcel, Violation, GeoTiffData

logger = logging.getLogger(__name__)


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
        
        # Рисуем кадастр
        for parcel in cadastral_parcels:
            coords = list(parcel.geometry.exterior.coords)
            poly = MplPolygon(coords, facecolor='none', edgecolor='blue', linewidth=0.8, alpha=0.9)
            ax.add_patch(poly)
        
        # Рисуем обнаружения
        for obj in detected_objects:
            coords = list(obj.geometry.exterior.coords)
            poly = MplPolygon(coords, facecolor='green', alpha=0.4, edgecolor='darkgreen', linewidth=0.5)
            ax.add_patch(poly)
        
        # Рисуем нарушения
        for violation in violations:
            coords = list(violation.geometry.exterior.coords)
            poly = MplPolygon(coords, facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=1)
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
