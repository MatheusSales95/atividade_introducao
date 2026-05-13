import os
import tempfile

import numpy as np
import pytest
import rasterio
import shapely.geometry
from rasterio.transform import from_bounds

from smosaic.smosaic_count_pixels import count_pixels


def _write_tif(path, data, nodata, dtype='int16', crs='EPSG:4326'):
    h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=w, height=h, nodata=nodata,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])


# ── Contagem básica ───────────────────────────────────────────────────────────

def test_count_pixels_identifica_valores_alvo():
    """Todos os pixels são SCL=4 → count deve ser igual ao total."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((4, 4), 4, dtype=np.int16)
        path = os.path.join(tmp, 'scl.tif')
        _write_tif(path, data, nodata=0)

        geom = shapely.geometry.box(0, 0, 1, 1)
        result = count_pixels(path, target_values=[4, 5, 6], geom=geom)

        assert result['count'] == result['total']
        assert result['count'] == 16


def test_count_pixels_nuvem_nao_e_alvo():
    """Metade dos pixels é nuvem (SCL=8) e metade é válida (SCL=5)."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((4, 4), 5, dtype=np.int16)
        data[:, 2:] = 8  # direita: nuvem

        path = os.path.join(tmp, 'scl_mix.tif')
        _write_tif(path, data, nodata=0)

        geom = shapely.geometry.box(0, 0, 1, 1)
        result = count_pixels(path, target_values=[4, 5, 6], geom=geom)

        assert result['total'] == 16
        assert result['count'] == 8  # só a metade esquerda é válida


def test_count_pixels_nodata_excluido_do_total():
    """Pixels com valor nodata não devem entrar no total."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.zeros((4, 4), dtype=np.int16)
        data[0, 0] = 4  # único pixel válido

        path = os.path.join(tmp, 'mostly_nodata.tif')
        _write_tif(path, data, nodata=0)

        geom = shapely.geometry.box(0, 0, 1, 1)
        result = count_pixels(path, target_values=[4, 5, 6], geom=geom)

        assert result['total'] == 1
        assert result['count'] == 1


def test_count_pixels_retorna_dict_com_chaves_corretas():
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((2, 2), 4, dtype=np.int16)
        path = os.path.join(tmp, 'small.tif')
        _write_tif(path, data, nodata=0)

        geom = shapely.geometry.box(0, 0, 1, 1)
        result = count_pixels(path, target_values=[4], geom=geom)

        assert 'total' in result
        assert 'count' in result
        assert isinstance(result['total'], int)
        assert isinstance(result['count'], int)
