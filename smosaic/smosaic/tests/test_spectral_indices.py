import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from smosaic.smosaic_spectral_indices import (
    evi2_calc,
    fix_negatives,
    ndvi_calc,
    savi_calc,
)


def _write_tif(path, value, dtype='float32', crs='EPSG:4326', size=4):
    """Cria um raster de NxN pixels com um valor uniforme."""
    data = np.full((size, size), value, dtype=dtype)
    transform = from_bounds(0, 0, 1, 1, size, size)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=size, height=size, nodata=None,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])


# ── fix_negatives ─────────────────────────────────────────────────────────────

def test_fix_negatives_zera_valores_negativos():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'banda.tif')
        data = np.array([[-100, 500], [200, -1]], dtype=np.float32)
        transform = from_bounds(0, 0, 1, 1, 2, 2)
        with rasterio.open(path, 'w', driver='GTiff', dtype='float32', count=1,
                           width=2, height=2, crs='EPSG:4326',
                           transform=transform) as dst:
            dst.write(data[np.newaxis, :, :])

        fix_negatives(path)

        with rasterio.open(path) as src:
            out = src.read(1)
        assert out[0, 0] == 0    # era -100
        assert out[0, 1] == 500  # positivo preservado
        assert out[1, 0] == 200  # positivo preservado
        assert out[1, 1] == 0    # era -1


def test_fix_negatives_preserva_positivos():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'banda.tif')
        data = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
        transform = from_bounds(0, 0, 1, 1, 2, 2)
        with rasterio.open(path, 'w', driver='GTiff', dtype='float32', count=1,
                           width=2, height=2, crs='EPSG:4326',
                           transform=transform) as dst:
            dst.write(data[np.newaxis, :, :])

        fix_negatives(path)

        with rasterio.open(path) as src:
            out = src.read(1)
        np.testing.assert_array_equal(out, data)


# ── ndvi_calc ─────────────────────────────────────────────────────────────────

def test_ndvi_formula_correta():
    """NDVI = (NIR - RED) / (NIR + RED), escalado por 10000."""
    # NIR=1000, RED=500 → ndvi = 500/1500 = 0.3333 → int16 = 3333
    with tempfile.TemporaryDirectory() as tmp:
        f_red = os.path.join(tmp, 'scene_B04_20240101T000000.tif')
        f_nir = os.path.join(tmp, 'scene_B08_20240101T000000.tif')

        _write_tif(f_red, value=500.0)
        _write_tif(f_nir, value=1000.0)

        ndvi_calc(f_nir, f_red)

        f_ndvi = f_red.replace('B04', 'NDVI')
        assert os.path.exists(f_ndvi), "Arquivo NDVI não foi criado"

        with rasterio.open(f_ndvi) as src:
            out = src.read(1)

        expected = int(round((1000 - 500) / (1000 + 500) * 10000))  # 3333
        assert out[0, 0] == expected


def test_ndvi_output_dtype_int16():
    with tempfile.TemporaryDirectory() as tmp:
        f_red = os.path.join(tmp, 'x_B04_x.tif')
        f_nir = os.path.join(tmp, 'x_B08_x.tif')
        _write_tif(f_red, 500.0)
        _write_tif(f_nir, 1000.0)
        ndvi_calc(f_nir, f_red)
        with rasterio.open(f_red.replace('B04', 'NDVI')) as src:
            assert src.dtypes[0] == 'int16'


# ── evi2_calc ─────────────────────────────────────────────────────────────────

def test_evi2_formula_correta():
    """EVI2 = 2.5 * (NIR - RED) / (NIR + RED + 1), escalado por 10000."""
    # NIR=1000, RED=500 → evi2 = 2.5 * 500/1501 ≈ 0.8328 → int16 ≈ 8328
    with tempfile.TemporaryDirectory() as tmp:
        f_red = os.path.join(tmp, 'img_B04_t.tif')
        f_nir = os.path.join(tmp, 'img_B08_t.tif')
        _write_tif(f_red, 500.0)
        _write_tif(f_nir, 1000.0)

        evi2_calc(f_nir, f_red)

        f_evi2 = f_red.replace('B04', 'EVI2')
        assert os.path.exists(f_evi2)

        with rasterio.open(f_evi2) as src:
            out = src.read(1)

        expected = int(round(2.5 * (1000 - 500) / (1000 + 500 + 1) * 10000))
        assert out[0, 0] == pytest.approx(expected, abs=1)


# ── savi_calc ─────────────────────────────────────────────────────────────────

def test_savi_formula_correta():
    """SAVI = (1+L)(NIR-RED)/(NIR+RED+L) com L=0.5, escalado por 10000."""
    # NIR=1000, RED=500 → savi = 1.5*500/1500.5 ≈ 0.4998 → int16 ≈ 4998
    with tempfile.TemporaryDirectory() as tmp:
        f_red = os.path.join(tmp, 'img_B04_s.tif')
        f_nir = os.path.join(tmp, 'img_B08_s.tif')
        _write_tif(f_red, 500.0)
        _write_tif(f_nir, 1000.0)

        savi_calc(f_nir, f_red)

        f_savi = f_red.replace('B04', 'SAVI')
        assert os.path.exists(f_savi)

        with rasterio.open(f_savi) as src:
            out = src.read(1)

        L = 0.5
        expected = int(round((1 + L) * (1000 - 500) / (1000 + 500 + L) * 10000))
        assert out[0, 0] == pytest.approx(expected, abs=1)
