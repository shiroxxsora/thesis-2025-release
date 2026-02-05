"""Readers для чтения различных форматов данных."""

from .base import DataReader
from .geotiff_reader import GeoTiffReader
from .cadastral_reader import CadastralReader

__all__ = [
    'DataReader',
    'GeoTiffReader',
    'CadastralReader',
]
