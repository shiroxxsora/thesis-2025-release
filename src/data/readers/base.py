"""Базовые классы для чтения данных."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
from pathlib import Path

T = TypeVar('T')


class DataReader(ABC, Generic[T]):
    """Абстрактный базовый класс для чтения данных."""
    
    @abstractmethod
    def read(self, path: str) -> T:
        """
        Читает данные из файла.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Прочитанные данные
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если данные невалидны
        """
        pass
    
    @abstractmethod
    def validate(self, data: T) -> bool:
        """
        Валидирует прочитанные данные.
        
        Args:
            data: Данные для валидации
            
        Returns:
            True если данные валидны
        """
        pass
    
    def _check_file_exists(self, path: str) -> Path:
        """
        Проверяет существование файла.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Path объект
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return file_path
