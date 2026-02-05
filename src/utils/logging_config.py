"""Конфигурация логирования."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Настраивает систему логирования.
    
    Args:
        level: Уровень логирования (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу лога (опционально)
        format_string: Формат сообщений (опционально)
    """
    if format_string is None:
        format_string = '[%(asctime)s] %(levelname)-8s %(name)s - %(message)s'
    
    # Базовая конфигурация
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Добавляем файловый handler если указан путь
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Перезаписываем существующую конфигурацию
    )


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер с заданным именем.
    
    Args:
        name: Имя логгера (обычно __name__ модуля)
        
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)
