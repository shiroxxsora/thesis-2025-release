"""Writer для экспорта в JSON."""

import json
import logging
from typing import Any
from pathlib import Path

from .base import DataWriter

logger = logging.getLogger(__name__)


class JSONWriter(DataWriter):
    """Класс для записи в JSON."""
    
    def write(self, data: Any, path: str, **kwargs) -> None:
        """
        Записывает данные в JSON.
        
        Args:
            data: Данные для записи (должны быть сериализуемы)
            path: Путь к выходному файлу
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Записан JSON: {path}")
