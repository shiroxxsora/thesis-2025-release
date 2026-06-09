"""
Экспорт детальных документов (PDF/DOCX) по каждому кадастровому участку.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

from .report_loader import ReportLoader
from .pdf_builder import PDFBuilder
from .map_generator import MapGenerator
from .coordinate_presenter import safe_filename

# Проверяем доступность python-docx (ленивый импорт - только при необходимости)
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParcelExporter:
    """
    Экспорт детальных документов (PDF/DOCX) по каждому кадастровому участку.
    
    Использует данные из Excel отчёта для создания индивидуальных документов
    с картами, таблицами координат и визуализацией нарушений.
    """
    
    def __init__(
        self,
        output_dir: str,
        font_path: Optional[str] = None,
        geotiff_path: Optional[str] = None,
        proj_string: Optional[str] = None,
        merge_violations_per_parcel: bool = True,
    ):
        """
        Args:
            output_dir: Директория с результатами комплексного анализа
            font_path: Путь к TTF-шрифту с кириллицей (опционально)
            geotiff_path: Путь к GeoTIFF для подложки (опционально)
            proj_string: PROJ-строка для исходной СК векторов
            merge_violations_per_parcel: как в анализе — True: на картах один union на КН;
                False: каждая строка отчёта отдельным полигоном, площадь на подписи = сумма из Excel.
        """
        self.output_dir = Path(output_dir)
        self.report_path = self.output_dir / "report.xlsx"
        self.per_parcel_dir = self.output_dir / "per_parcel_docs"
        
        self.font_path = font_path
        self.geotiff_path = geotiff_path or (self.output_dir.parent.parent / "geotiffs" / "input.tiff")
        # +x_0=4250000 — та же МСК-03/Улан-Удэ, что и fallback в cadastral_reader: кадастр и отчёт
        # в «зонной» записи. У GeoTIFF в WKT часто False easting 250000, но координаты уже
        # согласованы с geotransform; MapGenerator не делает лишний pyproj-сдвиг между ними.
        self.proj_string = proj_string or (
            '+proj=tmerc +lat_0=0 +lon_0=109.03333333333 +k=1 +x_0=4250000 '
            '+y_0=-5211057.63 +ellps=krass +towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 '
            '+units=m +no_defs'
        )
        
        self.merge_violations_per_parcel = merge_violations_per_parcel

        # Компоненты
        self.pdf_builder = PDFBuilder(font_path=font_path)
        self.background = None
    
    def export(
        self,
        only_with_violations: bool = True,
        limit: Optional[int] = None,
        out_format: str = 'both'
    ) -> str:
        """
        Экспортирует документы по каждому участку с нарушениями.
        
        Args:
            only_with_violations: Экспортировать только участки с нарушениями
            limit: Ограничить количество участков (для тестирования)
            out_format: Формат выходных файлов ('pdf', 'docx', 'both')
            
        Returns:
            Путь к директории с созданными документами.
        """
        if not self.report_path.exists():
            logger.error(f"Excel отчёт не найден: {self.report_path}")
            raise FileNotFoundError(f"Требуется файл {self.report_path}")
        
        logger.info(f"Запуск экспорта документов по участкам (формат: {out_format})")
        
        # Проверка доступности DOCX (только информационное сообщение)
        if out_format in ('docx', 'both') and not DOCX_AVAILABLE:
            logger.warning(
                "DOCX формат запрошен, но 'python-docx' не установлен. "
                "DOCX документы не будут созданы. "
                "Установите: pip install python-docx"
            )
        
        # Загрузка данных
        frames = ReportLoader.load_frames(str(self.report_path))
        cadastral_df = frames['cadastral_df']
        violations_df = frames['violations_df']
        coords_df = frames['coords_df']
        viol_coords_df = frames['viol_coords_df']
        
        # Если каких-то листов/колонок нет (например кадастра 0), не падаем.
        if only_with_violations:
            if 'Ближайший кадастровый номер' not in violations_df.columns:
                logger.warning(
                    "report.xlsx не содержит лист/колонку нарушений ('Ближайший кадастровый номер'). "
                    "Экспорт per_parcel пропущен."
                )
                self.per_parcel_dir.mkdir(parents=True, exist_ok=True)
                return str(self.per_parcel_dir)
        else:
            if 'Кадастровый номер' not in cadastral_df.columns:
                logger.warning(
                    "report.xlsx не содержит лист/колонку кадастра ('Кадастровый номер'). "
                    "Экспорт per_parcel пропущен."
                )
                self.per_parcel_dir.mkdir(parents=True, exist_ok=True)
                return str(self.per_parcel_dir)

        # Определение списка участков
        if only_with_violations:
            parcel_ids = sorted(set(violations_df['Ближайший кадастровый номер'].dropna().astype(str)))
        else:
            parcel_ids = sorted(set(cadastral_df['Кадастровый номер'].dropna().astype(str)))
        
        if limit is not None:
            parcel_ids = parcel_ids[:max(0, int(limit))]
        
        logger.info(f"Всего участков к выгрузке: {len(parcel_ids)}")
        self.per_parcel_dir.mkdir(parents=True, exist_ok=True)
        
        # Предзагрузка подложки
        self.background = self._load_background()
        
        # Обработка участков: последовательно (matplotlib/pyplot не потокобезопасны).
        total = len(parcel_ids)
        for idx, cad_number in enumerate(parcel_ids, 1):
            self._process_one_parcel(
                cad_number, idx, total,
                cadastral_df, violations_df, coords_df, viol_coords_df,
                out_format,
            )
        
        logger.info(f"Готово. Файлы: {self.per_parcel_dir}")
        return str(self.per_parcel_dir)
    
    def _load_background(self) -> Optional[Tuple[Any, Optional[Tuple[float, float, float, float]], Any]]:
        """Загружает фоновое изображение (GeoTIFF) один раз."""
        if not self.geotiff_path:
            logger.info("GeoTIFF подложка не указана")
            return None
        geotiff_path = Path(self.geotiff_path)
        if not geotiff_path.exists():
            logger.warning(f"GeoTIFF подложка не найдена: {geotiff_path.resolve()}")
            return None
        
        if not str(geotiff_path).lower().endswith(('.tif', '.tiff')):
            logger.info("Подложка не является GeoTIFF, пропускаем")
            return None

        logger.info(f"Загрузка подложки для per_parcel: {geotiff_path.resolve()}")

        # 1) Основной путь: GDAL
        if GDAL_AVAILABLE:
            try:
                logger.info("Подложка: пробую GDAL")
                ds = gdal.Open(str(geotiff_path))
                if ds is not None:
                    arr = ds.ReadAsArray()
                    if arr.ndim == 2:
                        arr = arr[None, ...]

                    height = ds.RasterYSize
                    width = ds.RasterXSize
                    gt = ds.GetGeoTransform()
                    left = gt[0]
                    top = gt[3]
                    right = gt[0] + width * gt[1]
                    bottom = gt[3] + height * gt[5]
                    proj_wkt = ds.GetProjection()

                    # Уменьшение размера для производительности
                    if max(height, width) > 3000:
                        step = max(1, math.ceil(max(height, width) / 3000))
                        arr = arr[:, ::step, ::step]
                        height = arr.shape[1]
                        width = arr.shape[2]

                    # Нормализация в RGB
                    if arr.shape[0] >= 3:
                        rgb = arr[:3].astype('float64')
                        rgb = (rgb - rgb.min()) / max(1e-9, (rgb.max() - rgb.min()))
                        img_bg = rgb.transpose(1, 2, 0)
                    else:
                        gray = arr[0].astype('float64')
                        gray = (gray - gray.min()) / max(1e-9, (gray.max() - gray.min()))
                        img_bg = gray

                    # CRS подложки
                    dest_crs = None
                    if PYPROJ_AVAILABLE and proj_wkt:
                        try:
                            dest_crs = pyproj.CRS.from_wkt(proj_wkt)
                        except Exception:
                            dest_crs = None

                    logger.info(f"Подложка (GDAL) загружена: {width}x{height}, CRS: {dest_crs}")
                    return (img_bg, (left, right, bottom, top), dest_crs)
            except Exception as e:
                logger.warning(f"GDAL не смог загрузить подложку: {e}")
        else:
            logger.warning("GDAL не доступен, пробую fallback через rasterio")

        # 2) Fallback: rasterio (если GDAL-путь не сработал)
        try:
            import rasterio

            logger.info("Подложка: пробую rasterio fallback")
            with rasterio.open(str(geotiff_path)) as src:
                arr = src.read()
                if arr.ndim == 2:
                    arr = arr[None, ...]
                height, width = int(src.height), int(src.width)
                b = src.bounds
                left, bottom, right, top = float(b.left), float(b.bottom), float(b.right), float(b.top)

                if max(height, width) > 3000:
                    step = max(1, math.ceil(max(height, width) / 3000))
                    arr = arr[:, ::step, ::step]
                    height = int(arr.shape[1])
                    width = int(arr.shape[2])

                if arr.shape[0] >= 3:
                    rgb = arr[:3].astype('float64')
                    rgb = (rgb - rgb.min()) / max(1e-9, (rgb.max() - rgb.min()))
                    img_bg = rgb.transpose(1, 2, 0)
                else:
                    gray = arr[0].astype('float64')
                    gray = (gray - gray.min()) / max(1e-9, (gray.max() - gray.min()))
                    img_bg = gray

                dest_crs = None
                if PYPROJ_AVAILABLE and getattr(src, "crs", None) is not None:
                    try:
                        dest_crs = pyproj.CRS.from_user_input(src.crs)
                    except Exception:
                        dest_crs = None

                logger.info(f"Подложка (rasterio) загружена: {width}x{height}, CRS: {dest_crs}")
                return (img_bg, (left, right, bottom, top), dest_crs)
        except Exception as e:
            logger.warning(f"Не удалось загрузить подложку ни через GDAL, ни через rasterio: {e}")
            return None
    
    def _process_one_parcel(
        self,
        cad_number: str,
        idx: int,
        total: int,
        cadastral_df,
        violations_df,
        coords_df,
        viol_coords_df,
        out_format: str
    ) -> None:
        """Обрабатывает один участок."""
        logger.info(f"({idx}/{total}) Формирую документ для КН {cad_number}...")
        
        base_filename = f"parcel_{safe_filename(cad_number)}"
        pdf_path = self.per_parcel_dir / f"{base_filename}.pdf"
        docx_path = self.per_parcel_dir / f"{base_filename}.docx"
        
        try:
            # Генерация карт
            zoom_img_path = self._create_zoom_map(cad_number, cadastral_df, violations_df, viol_coords_df, coords_df)
            overview_img_path = self._create_overview_map(cad_number, cadastral_df, violations_df, viol_coords_df)
            
            # Генерация документов
            if out_format in ('pdf', 'both'):
                self.pdf_builder.generate(
                    str(pdf_path), cad_number, cadastral_df, coords_df,
                    violations_df, viol_coords_df, zoom_img_path, overview_img_path,
                )
            
            if out_format in ('docx', 'both'):
                if not DOCX_AVAILABLE:
                    # Как в старом коде - выбрасываем RuntimeError, который будет пойман общим except
                    raise RuntimeError("Требуется пакет 'python-docx' (pip install python-docx) для экспорта DOCX")
                # Ленивый импорт DOCXBuilder только когда нужен
                from .docx_builder import DOCXBuilder
                try:
                    docx_builder = DOCXBuilder()
                    docx_builder.generate(
                        str(docx_path), cad_number, cadastral_df, coords_df,
                        violations_df, viol_coords_df, zoom_img_path, overview_img_path,
                    )
                except RuntimeError as e:
                    # Пробрасываем RuntimeError дальше, чтобы он был пойман общим except
                    raise
        
        except KeyError as e:
            logger.warning(f"Пропускаю {cad_number}: отсутствует столбец {e}")
        except RuntimeError as e:
            # RuntimeError для DOCX (если python-docx не установлен) - логируем как warning
            if "python-docx" in str(e):
                logger.warning(f"Проблема при формировании {cad_number}: {e}")
            else:
                raise  # Пробрасываем другие RuntimeError
        except Exception as e:
            logger.warning(f"Проблема при формировании {cad_number}: {e}")
    
    def _create_zoom_map(
        self, cad_number: str, cadastral_df, violations_df, viol_coords_df, coords_df
    ) -> Optional[str]:
        """Создаёт zoom-карту для участка."""
        zoom_img_path = self.per_parcel_dir / f"zoom_{safe_filename(cad_number)}.png"
        
        try:
            map_gen = MapGenerator(background=self.background)
            map_gen.create_zoom_map(
                cad_number, cadastral_df, violations_df, viol_coords_df,
                str(zoom_img_path), self.proj_string,
                merge_violations_per_parcel=self.merge_violations_per_parcel,
            )
            if zoom_img_path.exists():
                return str(zoom_img_path)
        except Exception as e:
            logger.warning(f"Не удалось создать zoom-изображение для {cad_number}: {e}")
        
        return None
    
    def _create_overview_map(
        self, cad_number: str, cadastral_df, violations_df, viol_coords_df
    ) -> Optional[str]:
        """Создаёт обзорную карту для участка."""
        overview_img_path = self.per_parcel_dir / f"overview_{safe_filename(cad_number)}.png"
        
        try:
            map_gen = MapGenerator(background=self.background)
            map_gen.create_overview_map(
                cad_number, cadastral_df, violations_df, viol_coords_df,
                str(overview_img_path), self.proj_string,
                merge_violations_per_parcel=self.merge_violations_per_parcel,
            )
            if overview_img_path.exists():
                return str(overview_img_path)
        except Exception as e:
            logger.warning(f"Не удалось создать обзорную карту для {cad_number}: {e}")
        
        return None
