"""Базовый класс для writers."""

from abc import ABC, abstractmethod
from typing import Any, List


class DataWriter(ABC):
    """Абстрактный класс для записи данных."""
    
    @abstractmethod
    def write(self, data: Any, path: str, **kwargs) -> None:
        """
        Записывает данные в файл.
        
        Args:
            data: Данные для записи
            path: Путь к выходному файлу
            **kwargs: Дополнительные параметры
        """
        pass
