import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from smosaic.smosaic_generate_cog import generate_cog


def _write_tif(folder, name, data, dtype='int16', crs='EPSG:4326'):
    """Cria raster de teste e retorna (pasta, nome_sem_extensão)."""
    path = os.path.join(folder, f'{name}.tif')
    h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=w, height=h, nodata=0,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


# ── Geração de COG ────────────────────────────────────────────────────────────

def test_generate_cog_cria_arquivo_cog():
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((8, 8), 1000, dtype=np.int16)
        _write_tif(tmp, 'raster', data)

        result = generate_cog(tmp, 'raster')

        assert os.path.exists(result), "Arquivo COG não foi criado"
        assert result.endswith('_COG.tif')


def test_generate_cog_retorna_path_correto():
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((4, 4), 500, dtype=np.int16)
        _write_tif(tmp, 'banda_B04', data)

        result = generate_cog(tmp, 'banda_B04')

        expected = os.path.join(tmp, 'banda_B04_COG.tif')
        assert result == expected


def test_generate_cog_arquivo_e_raster_valido():
    """O COG gerado deve ser um raster com pelo menos 1 banda e pixels válidos."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[100, 200], [300, 400]], dtype=np.int16)
        _write_tif(tmp, 'small', data)

        result = generate_cog(tmp, 'small')

        with rasterio.open(result) as src:
            assert src.count >= 1
            out = src.read(1)
            assert out.size > 0


def test_generate_cog_output_dtype_int16():
    """generate_cog converte para int16 (comportamento documentado)."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.full((4, 4), 1500, dtype=np.int16)
        _write_tif(tmp, 'banda', data)

        result = generate_cog(tmp, 'banda')

        with rasterio.open(result) as src:
            assert src.dtypes[0] == 'int16'
