import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from smosaic.smosaic_fix_baseline_number import fix_baseline_number


def _write_tif(folder, name, data, nodata, dtype='int16', crs='EPSG:4326'):
    """Escreve um raster de teste e retorna o nome (sem .tif)."""
    path = os.path.join(folder, f'{name}.tif')
    h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=w, height=h, nodata=nodata,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])
    return name


# ── Baseline ≤ 400: sem alteração ────────────────────────────────────────────

def test_baseline_abaixo_400_nao_altera_dados():
    """Baseline 399 não deve modificar os valores do raster."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[2000, 1500], [1000, 500]], dtype=np.int16)
        name = _write_tif(tmp, 'banda', data, nodata=0)

        result = fix_baseline_number(tmp, name, baseline_number='399')

        assert result is True
        with rasterio.open(os.path.join(tmp, f'{name}.tif')) as src:
            out = src.read(1)
        np.testing.assert_array_equal(out, data)


def test_baseline_igual_a_400_nao_altera_dados():
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[3000, 2000], [1000, 0]], dtype=np.int16)
        name = _write_tif(tmp, 'banda', data, nodata=0)

        fix_baseline_number(tmp, name, baseline_number='400')

        with rasterio.open(os.path.join(tmp, f'{name}.tif')) as src:
            out = src.read(1)
        np.testing.assert_array_equal(out, data)


# ── Baseline > 400: subtrai 1000 ─────────────────────────────────────────────

def test_baseline_acima_400_subtrai_1000():
    """Baseline 401 deve subtrair 1000 de todos os valores não-nodata."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[2000, 1500], [1000, 500]], dtype=np.int16)
        name = _write_tif(tmp, 'banda', data, nodata=0)

        fix_baseline_number(tmp, name, baseline_number='401')

        with rasterio.open(os.path.join(tmp, f'{name}.tif')) as src:
            out = src.read(1)
        expected = np.array([[1000, 500], [0, -500]], dtype=np.int16)
        np.testing.assert_array_equal(out, expected)


def test_baseline_acima_400_nan_nodata_converte_nan_para_zero():
    """Baseline > 400 com nodata=NaN: pixels NaN viram 0, demais sofrem -1000."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[2000.0, np.nan], [1500.0, np.nan]], dtype=np.float32)
        name = _write_tif(tmp, 'banda_float', data, nodata=float('nan'), dtype='float32')

        fix_baseline_number(tmp, name, baseline_number='500')

        with rasterio.open(os.path.join(tmp, f'{name}.tif')) as src:
            out = src.read(1)
            assert src.nodata == 0

        assert out[0, 0] == 1000   # 2000 - 1000
        assert out[1, 0] == 500    # 1500 - 1000
        assert out[0, 1] == 0      # era NaN → 0
        assert out[1, 1] == 0      # era NaN → 0
