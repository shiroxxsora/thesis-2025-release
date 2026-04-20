"""Подготовка геометрий для глобальной карты, Excel и per_parcel (без изменения исходного анализа)."""

from dataclasses import replace
from typing import List

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..domain.models import CadastralParcel, Violation
from .geometry_utils import simplify_polygon


def cadastral_parcels_for_maps_and_documents(
    parcels: List[CadastralParcel], tolerance_m: float
) -> List[CadastralParcel]:
    """
    Копии участков с упрощённым контуром для глобальной карты, Excel и per_parcel.

    Площадь ``area_sqm`` из исходных данных кадастра не меняем; меняются геометрия и центроид.
    """
    if tolerance_m <= 0:
        return list(parcels)
    out: List[CadastralParcel] = []
    for p in parcels:
        g = p.geometry
        if not isinstance(g, Polygon):
            out.append(p)
            continue
        g2 = simplify_polygon(g, tolerance_m)
        cxy = g2.centroid.coords[0]
        out.append(replace(p, geometry=g2, centroid=cxy))
    return out


def violations_for_maps_and_documents(
    violations: List[Violation], tolerance_m: float
) -> List[Violation]:
    """
    Копии нарушений с упрощённой геометрией (Douglas–Peucker) для визуализации и отчётов.

    Исходные shapefile/JSON по-прежнему из полного анализа (без этого упрощения).
    """
    if tolerance_m <= 0:
        return list(violations)
    out: List[Violation] = []
    for v in violations:
        g = v.geometry
        if isinstance(g, Polygon):
            g2 = simplify_polygon(g, tolerance_m)
        elif isinstance(g, MultiPolygon):
            parts = [simplify_polygon(p, tolerance_m) for p in g.geoms if isinstance(p, Polygon)]
            if not parts:
                out.append(v)
                continue
            g2 = unary_union(parts)
            if g2.is_empty or g2.geom_type not in ('Polygon', 'MultiPolygon'):
                out.append(v)
                continue
        else:
            out.append(v)
            continue
        cxy = g2.centroid.coords[0]
        out.append(replace(v, geometry=g2, centroid=cxy))
    return out
