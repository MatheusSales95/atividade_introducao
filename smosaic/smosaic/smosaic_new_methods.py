"""New xarray-based temporal composition methods for smosaic."""

import os

import numpy as np
import rasterio
import tqdm
import xarray as xr
from rasterio.warp import Resampling

from smosaic.smosaic_utils import get_all_cloud_configs

NEW_METHODS = {'avg', 'media', 'med', 'mediana', 'max', 'min', 'maxx', 'minx'}


# ---------------------------------------------------------------------------
# Spectral index
# ---------------------------------------------------------------------------

def calcular_nbr_cubo(cubo, nir_band='B08', swir_band='B12', eps=1e-7):
    """Add NBR = (NIR - SWIR) / (NIR + SWIR + eps) to the cube as a new variable.

    Returns a NEW Dataset (does not mutate the input).
    Healthy vegetation -> NBR near +1. Burned area -> NBR negative.
    """
    nir = cubo[nir_band].astype(np.float32)
    swir = cubo[swir_band].astype(np.float32)
    nbr = (nir - swir) / (nir + swir + eps)
    return cubo.assign(NBR=nbr)


# ---------------------------------------------------------------------------
# Composition functions
# ---------------------------------------------------------------------------

def _gerar_composicao_media(cubo):
    return cubo.mean(dim='time'), None


def _gerar_composicao_mediana(cubo):
    return cubo.median(dim='time'), None


def _gerar_composicao_max(cubo, bandas):
    """Per-band maximum + per-band provenance."""
    dict_bandas, dict_provs = {}, {}
    for banda in bandas:
        banda_temp = cubo[banda].fillna(-np.inf)
        idx = banda_temp.argmax(dim='time').compute()
        compo = cubo[banda].isel(time=idx)
        prov = cubo['time'].isel(time=idx)
        if 'time' in compo.coords:
            compo = compo.drop_vars('time')
        if 'time' in prov.coords:
            prov = prov.drop_vars('time')
        dict_bandas[banda] = compo
        dict_provs[f'prov_{banda}'] = prov
    return xr.Dataset(dict_bandas), xr.Dataset(dict_provs)


def _gerar_composicao_min(cubo, bandas):
    """Per-band minimum + per-band provenance."""
    dict_bandas, dict_provs = {}, {}
    for banda in bandas:
        banda_temp = cubo[banda].fillna(np.inf)
        idx = banda_temp.argmin(dim='time').compute()
        compo = cubo[banda].isel(time=idx)
        prov = cubo['time'].isel(time=idx)
        if 'time' in compo.coords:
            compo = compo.drop_vars('time')
        if 'time' in prov.coords:
            prov = prov.drop_vars('time')
        dict_bandas[banda] = compo
        dict_provs[f'prov_{banda}'] = prov
    return xr.Dataset(dict_bandas), xr.Dataset(dict_provs)


def _achar_indice_do_enesimo(matriz_3d, enesimo, tipo='max'):
    """Return the index of the k-th extreme value along the last (time) axis."""
    n = min(enesimo, matriz_3d.shape[-1])
    if tipo == 'max':
        ms = np.where(np.isnan(matriz_3d), -np.inf, matriz_3d)
        return np.argsort(ms, axis=-1)[..., -n]
    ms = np.where(np.isnan(matriz_3d), np.inf, matriz_3d)
    return np.argsort(ms, axis=-1)[..., n - 1]


def _gerar_composicao_extrema(cubo, banda_ref='B12', enesimo=1, tipo='max'):
    """k-th extreme of reference band; all output bands come from the same date."""
    arr_ref = cubo[banda_ref].transpose('y', 'x', 'time').values
    idx_np = _achar_indice_do_enesimo(arr_ref, enesimo=enesimo, tipo=tipo)
    idx = xr.DataArray(idx_np, dims=('y', 'x'),
                       coords={'y': cubo['y'], 'x': cubo['x']})
    compo = cubo.isel(time=idx)
    prov = cubo['time'].isel(time=idx)
    if 'time' in compo.coords:
        compo = compo.drop_vars('time')
    return compo, prov


def _aplicar_metodo(cubo, mosaic_method, bandas, k=1, banda_ref='B12'):
    method = mosaic_method.lower()
    if method in ('avg', 'media'):
        return _gerar_composicao_media(cubo)
    if method in ('med', 'mediana'):
        return _gerar_composicao_mediana(cubo)
    if method == 'max':
        return _gerar_composicao_max(cubo, bandas)
    if method == 'min':
        return _gerar_composicao_min(cubo, bandas)
    if method == 'maxx':
        return _gerar_composicao_extrema(cubo, banda_ref=banda_ref, enesimo=k, tipo='max')
    if method == 'minx':
        return _gerar_composicao_extrema(cubo, banda_ref=banda_ref, enesimo=k, tipo='min')
    raise ValueError(f"Metodo nao reconhecido: {mosaic_method}")


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------

def _proveniencia_para_int(prov_da, mask_nodata=None):
    """datetime64 -> int32 day-of-year (1-365); masked pixels become 0 (no-data)."""
    out = prov_da.dt.dayofyear.values.astype('int32')
    if mask_nodata is not None:
        out = np.where(mask_nodata, np.int32(0), out)
    return out


# ---------------------------------------------------------------------------
# Cube builder from local reprojected files
# ---------------------------------------------------------------------------

def _build_xarray_cube(all_sorted_data, cloud_sorted_data, collection_name, bands, scene):
    """Build an xarray Dataset for a single MGRS scene with cloud masking applied.

    Files are already reprojected (output of reproject_tifs).
    Returns xr.Dataset with dims (time, y, x) and float32 values (NaN = masked).
    Returns None if the scene has no data.
    """
    cloud_dict = get_all_cloud_configs()
    non_cloud_values = cloud_dict[collection_name]['non_cloud_values']

    cloud_by_date = {
        item['date']: item['file']
        for item in cloud_sorted_data
        if item.get('scene') == scene
    }

    first_band = bands[0]
    scene_items = [item for item in all_sorted_data[first_band] if item.get('scene') == scene]
    if not scene_items:
        return None

    dates = sorted({item['date'] for item in scene_items})
    datas_np = np.array(
        [np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}") for d in dates],
        dtype='datetime64[ns]',
    )

    ref_file = scene_items[0]['file']
    with rasterio.open(ref_file) as src:
        height, width = src.shape
        transform = src.transform

    file_lookup = {}
    for band in bands:
        for item in all_sorted_data[band]:
            if item.get('scene') == scene:
                file_lookup[(band, item['date'])] = item['file']

    data_vars = {}
    for band in bands:
        stack = []
        for date in dates:
            band_file = file_lookup.get((band, date))
            cloud_file = cloud_by_date.get(date)

            if (band_file and cloud_file
                    and os.path.exists(band_file)
                    and os.path.exists(cloud_file)):
                with rasterio.open(band_file) as src:
                    arr = src.read(
                        1,
                        out_shape=(height, width),
                        resampling=Resampling.nearest,
                    ).astype(np.float32)
                    src_nodata = src.nodata if src.nodata is not None else 0

                with rasterio.open(cloud_file) as csrc:
                    cloud_arr = csrc.read(
                        1,
                        out_shape=(height, width),
                        resampling=Resampling.nearest,
                    )

                clear_mask = np.isin(cloud_arr, non_cloud_values)
                arr[~clear_mask] = np.nan
                arr[arr == src_nodata] = np.nan
            else:
                arr = np.full((height, width), np.nan, dtype=np.float32)

            stack.append(arr)

        data_vars[band] = (('time', 'y', 'x'), np.stack(stack, axis=0))

    y_coords = np.array([transform.f + transform.e * i for i in range(height)])
    x_coords = np.array([transform.c + transform.a * i for i in range(width)])

    return xr.Dataset(data_vars, coords={'time': datas_np, 'y': y_coords, 'x': x_coords})


# ---------------------------------------------------------------------------
# Cloud composite helper
# ---------------------------------------------------------------------------

def _build_cloud_composite(cloud_sorted_data, scene, height, width):
    """First-valid-pixel cloud composite for a scene."""
    cloud_files = [
        item['file'] for item in cloud_sorted_data
        if item.get('scene') == scene and os.path.exists(item.get('file', ''))
    ]
    if not cloud_files:
        return np.zeros((height, width), dtype=np.int16)

    with rasterio.open(cloud_files[0]) as src:
        composite = src.read(
            1, out_shape=(height, width), resampling=Resampling.nearest
        ).astype(np.int16)

    for f in cloud_files[1:]:
        with rasterio.open(f) as src:
            arr = src.read(
                1, out_shape=(height, width), resampling=Resampling.nearest
            ).astype(np.int16)
        fill_mask = composite == 0
        composite[fill_mask] = arr[fill_mask]

    return composite


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compose_new_method(
    all_sorted_data,
    cloud_sorted_data,
    scenes,
    collection_name,
    bands,
    data_dir,
    start_date,
    end_date,
    mosaic_method,
    k=1,
    banda_ref=None,
):
    """Compose using xarray-based methods (avg/med/max/min/maxx/minx).

    Receives files already on disk (post reproject_tifs), applies the chosen
    temporal composition, and returns merge files in the same format as
    merge_scene / merge_scene_provenance_cloud so the rest of the pipeline
    (merge_tifs, clip_raster, fix_baseline_number, generate_cog) is unchanged.

    Args:
        all_sorted_data (dict): {band_name: [items...]} — reprojected, each item
            has keys: band, date, scene, file, clean_percentage.
        cloud_sorted_data (list): reprojected cloud band items.
        scenes (list): MGRS scene identifiers to process.
        collection_name (str): BDC collection ID (e.g. "S2_L2A-1").
        bands (list): spectral bands to output.
        data_dir (str): directory for intermediate files.
        start_date (str): period start "YYYY-MM-DD".
        end_date (str): period end "YYYY-MM-DD".
        mosaic_method (str): one of avg/media/med/mediana/max/min/maxx/minx.
        k (int): rank for maxx/minx (default 1 = absolute extreme).
        banda_ref (str): reference band for maxx/minx (may be "NBR").

    Returns:
        dict keyed by band name. Each value is a dict with:
            merge_files (list): one TIF per scene.
            provenance_merge_files (list): only on bands[0] when applicable.
            cloud_merge_files (list): only on bands[0].
    """
    method = mosaic_method.lower()
    needs_prov_single = method in ('minx', 'maxx')
    needs_prov_multi = method in ('min', 'max')
    needs_prov = needs_prov_single or needs_prov_multi

    if banda_ref is None:
        banda_ref = bands[0]

    nbr_needed = (banda_ref == 'NBR') or ('NBR' in bands)

    cloud_dict = get_all_cloud_configs()
    collection_prefix = collection_name.split('-')[0]
    start_str = str(start_date).replace('-', '')
    end_str = str(end_date).replace('-', '')

    results = {band: {'merge_files': []} for band in bands}
    provenance_merge_files = []
    cloud_merge_files = []

    for scene in tqdm.tqdm(scenes, desc=f'Composing ({mosaic_method})'):

        cubo = _build_xarray_cube(
            all_sorted_data, cloud_sorted_data, collection_name, bands, scene
        )
        if cubo is None:
            continue

        if nbr_needed:
            cubo = calcular_nbr_cubo(cubo)

        compo, prov = _aplicar_metodo(cubo, mosaic_method, bands, k=k, banda_ref=banda_ref)
        if 'time' in compo.coords:
            compo = compo.drop_vars('time')

        ref_item = next(
            item for item in all_sorted_data[bands[0]] if item.get('scene') == scene
        )
        with rasterio.open(ref_item['file']) as src:
            profile = src.profile.copy()
            height, width = src.shape

        profile.update(dtype='float32', count=1, nodata=float('nan'), driver='GTiff')

        # Save merge file per band
        for band in bands:
            base = f'merge_{collection_prefix}_{band}_{scene}_{start_str}_{end_str}'
            out_file = os.path.join(data_dir, f'{base}.tif')
            arr = compo[band].values.astype(np.float32)
            with rasterio.open(out_file, 'w', **profile) as dst:
                dst.write(arr, 1)
            results[band]['merge_files'].append(out_file)

        # Provenance
        if needs_prov and prov is not None:
            prov_base = f'provenance_merge_{collection_prefix}_{scene}_{start_str}_{end_str}'
            prov_file = os.path.join(data_dir, f'{prov_base}.tif')
            prov_profile = profile.copy()
            prov_profile.update(dtype='int32', nodata=0)

            if needs_prov_single:
                ref_for_mask = banda_ref if banda_ref in compo else bands[0]
                mask = np.isnan(compo[ref_for_mask].values)
                prov_int = _proveniencia_para_int(prov, mask_nodata=mask)
                prov_profile.update(count=1)
                with rasterio.open(prov_file, 'w', **prov_profile) as dst:
                    dst.write(prov_int, 1)

            elif needs_prov_multi:
                prov_profile.update(count=len(bands))
                with rasterio.open(prov_file, 'w', **prov_profile) as dst:
                    for bi, band in enumerate(bands, start=1):
                        mask = np.isnan(compo[band].values)
                        prov_int = _proveniencia_para_int(
                            prov[f'prov_{band}'], mask_nodata=mask
                        )
                        dst.write(prov_int, bi)

            provenance_merge_files.append(prov_file)

        # Cloud composite
        cloud_composite = _build_cloud_composite(
            cloud_sorted_data, scene, height, width
        )
        cloud_base = f'cloud_merge_{collection_prefix}_{scene}_{start_str}_{end_str}'
        cloud_file_path = os.path.join(data_dir, f'{cloud_base}.tif')
        cloud_profile = profile.copy()
        cloud_profile.update(
            dtype='int16',
            nodata=cloud_dict[collection_name]['no_data_value'],
            count=1,
        )
        with rasterio.open(cloud_file_path, 'w', **cloud_profile) as dst:
            dst.write(cloud_composite, 1)
        cloud_merge_files.append(cloud_file_path)

    if provenance_merge_files:
        results[bands[0]]['provenance_merge_files'] = provenance_merge_files
    if cloud_merge_files:
        results[bands[0]]['cloud_merge_files'] = cloud_merge_files

    return results
