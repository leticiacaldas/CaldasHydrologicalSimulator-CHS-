"""
Utility functions para data preparation geospatial para simulação HydroSim.

Inclui:
- Carregamento e reamostragem de DEM
- Vector rasterization (sources, rivers, intensity)
- Preparation of backgrounds/orthomosaics
- Reprojection e resampling de rasters

Author: Letícia Caldas
License: MIT
"""

import numpy as np
import rasterio as rio  # type: ignore
from rasterio.transform import from_bounds, Affine  # type: ignore
from rasterio.enums import Resampling  # type: ignore
from rasterio.warp import reproject as warp_reproject
from rasterio.warp import Resampling as WarpResampling  # type: ignore
from rasterio.features import rasterize  # type: ignore
import geopandas as gpd  # type: ignore
import os
from typing import Optional, Tuple, Any

try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # type: ignore


def _prepare_background(
    img_path: str, target_shape: Tuple[int, int], target_crs
) -> Optional[np.ndarray]:
    """Carrega um raster de fundo (ex. DOM ortomosaico), reamostra para target_shape e retorna RGB float [0,1].

    Parameters
    ----------
    img_path : str
        Caminho do arquivo raster
    target_shape : tuple
        (H, W) alvo
    target_crs : CRS
        CRS alvo

    Returns
    -------
    np.ndarray or None
        Array RGB float32 [0, 1] ou None se falhar
    """
    try:
        H, W = target_shape
        with rio.open(img_path) as src:
            # Limite de tamanho para evitar queda do app
            if src.width > 2000 or src.height > 2000:
                msg = (
                    f"O raster DOM é muito grande para o Streamlit Cloud ({src.width}x{src.height}). "
                    f"Reduza a resolução para até 2000x2000 pixels."
                )
                if st:
                    st.warning(msg)
                else:
                    print(f"Aviso: {msg}")
                return None

            # Ignora banda 4 se houver 4 bandas (usa só RGB)
            count = min(3, src.count if src.count != 4 else 3)
            bands: list = []
            for b in range(1, count + 1):
                band = src.read(
                    b,
                    out_shape=(H, W),
                    resampling=Resampling.bilinear,
                ).astype(float)
                bands.append(band)

            if len(bands) == 1:
                rgb = np.stack([bands[0]] * 3, axis=-1)
            else:
                rgb = np.stack(bands, axis=-1)

            # Normalização robusta para [0,1]
            if not np.issubdtype(rgb.dtype, np.floating):
                rgb = rgb.astype(float)

            finite_vals = rgb[np.isfinite(rgb)]
            if finite_vals.size > 0:
                p2, p98 = np.percentile(finite_vals, (2, 98))
                p2 = max(float(p2), 0.0)
                denom = (float(p98) - p2) or 1.0
                rgb = np.clip((rgb - p2) / denom, 0, 1)
            else:
                rgb = np.zeros_like(rgb, dtype=float)

            return rgb

    except Exception as e:
        # Tenta abrir o arquivo e mostrar metadados para diagnóstico
        try:
            with rio.open(img_path) as src:
                meta = src.meta
                msg = f"Falha ao preparar fundo (DOM): {e}\nMetadados do arquivo: {meta}"
                if st:
                    st.warning(msg)
                else:
                    print(f"Aviso: {msg}")
        except Exception as e2:
            msg = f"Falha ao abrir DOM: {e}\nErro secundário: {e2}"
            if st:
                st.warning(msg)
            else:
                print(f"Aviso: {msg}")

        return None


def _read_raster_to_match(
    img_path: str,
    target_shape: Tuple[int, int],
    target_transform,
    target_crs,
) -> Optional[np.ndarray]:
    """Lê um raster (1 banda) e reamostra/reprojeta para bater com a grade alvo.

    Parameters
    ----------
    img_path : str
        Caminho do arquivo raster
    target_shape : tuple
        (H, W) alvo
    target_transform : Affine
        Transform alvo
    target_crs : CRS
        CRS alvo

    Returns
    -------
    np.ndarray or None
        Array float32 com shape target_shape, ou None se falhar
    """
    try:
        H, W = target_shape
        out = np.zeros((H, W), dtype=np.float32)
        with rio.open(img_path) as src:
            src_data = src.read(1)
            warp_reproject(
                source=src_data,
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=WarpResampling.bilinear,
            )
        return out

    except Exception as e:
        msg = f"(Falha ao ler raster '{os.path.basename(img_path)}' para grade alvo: {e})"
        if st:
            st.caption(msg)
        else:
            print(msg)
        return None


def _setup_geodata(
    dem_path: str,
    vector_path: Optional[str],
    grid_reduction_factor: int,
    river_path: Optional[str] = None,
    attribute_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Any, Any, float, np.ndarray, Optional[np.ndarray]]:
    """Carrega DEM e rasteriza sources e rio opcionalmente.

    Parameters
    ----------
    dem_path : str
        Caminho do arquivo DEM
    vector_path : str or None
        Caminho do arquivo de sources (vetor)
    grid_reduction_factor : int
        Fator de redução de grade (subsampling)
    river_path : str, optional
        Caminho do arquivo de rivers (vetor)
    attribute_name : str, optional
        Nome de atributo para intensity espacial

    Returns
    -------
    dem_data : np.ndarray
        Matriz de elevação reamostrada
    sources_mask : np.ndarray
        Máscara binária de sources
    transform : Affine
        Transform do raster
    crs : CRS
        Coordinate reference system
    cell_size : float
        Tamanho da célula em metros
    river_mask : np.ndarray
        Máscara binária de rivers
    sources_intensity : np.ndarray or None
        Mapa de intensity espacial se attribute_name fornecido
    """
    with rio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
        crs = dem_src.crs

        # Redução de grade (amostragem) se solicitado
        try:
            gf = int(max(1, grid_reduction_factor))
        except Exception:
            gf = 1

        if gf > 1:
            H = dem_src.height // gf
            W = dem_src.width // gf
            dem_data = dem_src.read(
                1, out_shape=(H, W), resampling=Resampling.bilinear
            )
            # Ajustar transform para nova resolução
            transform = Affine(
                transform.a * gf,
                transform.b,
                transform.c,
                transform.d,
                transform.e * gf,
                transform.f,
            )
        else:
            dem_data = dem

    # Tamanho de célula aproximado (m) usando escala do transform
    try:
        cell_size = float(abs(transform.a))
    except Exception:
        cell_size = 1.0

    # Rasterizar sources (vetor)
    sources_mask = np.zeros_like(dem_data, dtype=np.uint8)
    sources_intensity = None

    if vector_path and os.path.exists(vector_path):
        try:
            gdf = gpd.read_file(vector_path)
            if gdf.crs is None and crs is not None:
                gdf = gdf.set_crs(crs)
            elif crs is not None and gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            if attribute_name and (attribute_name in gdf.columns):
                # Rasterizar intensity
                sources_intensity = rasterize(
                    [
                        (geom, float(val) if np.isfinite(val) else 0.0)
                        for geom, val in zip(gdf.geometry, gdf[attribute_name])
                    ],
                    out_shape=dem_data.shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.float32,
                )
                if sources_intensity is not None:
                    sources_mask = (sources_intensity > 0).astype(np.uint8)
                else:
                    sources_mask = np.zeros_like(dem_data, dtype=np.uint8)
            else:
                sources_mask = rasterize(
                    gdf.geometry,
                    out_shape=dem_data.shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )
        except Exception:
            sources_mask = np.zeros_like(dem_data, dtype=np.uint8)
            sources_intensity = None

    # Rasterizar rio (opcional)
    river_mask = np.zeros_like(dem_data, dtype=bool)
    if river_path and os.path.exists(river_path):
        try:
            rgdf = gpd.read_file(river_path)
            if rgdf.crs is None and crs is not None:
                rgdf = rgdf.set_crs(crs)
            elif crs is not None and rgdf.crs != crs:
                rgdf = rgdf.to_crs(crs)

            river_result = rasterize(
                rgdf.geometry,
                out_shape=dem_data.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )
            if river_result is not None:
                river_mask = river_result.astype(bool)
            else:
                river_mask = np.zeros_like(dem_data, dtype=bool)
        except Exception:
            river_mask = np.zeros_like(dem_data, dtype=bool)

    return dem_data, sources_mask, transform, crs, cell_size, river_mask, sources_intensity
