import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from smosaic.smosaic_merge_tifs import merge_tifs


def _write_tif(path, data, nodata, dtype='float32', crs='EPSG:4326'):
    h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=w, height=h, nodata=nodata,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])


# ── Novos métodos: nodata = NaN ───────────────────────────────────────────────

def test_nan_nodata_preenche_tiles_adjacentes():
    """
    Dois tiles com nodata=NaN: tile1 tem dado na esquerda, tile2 tem dado na direita.
    Após o merge, o resultado deve ter ambos os lados preenchidos sem buracos NaN.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data1 = np.full((4, 4), np.nan, dtype=np.float32)
        data1[:, :2] = 100.0

        data2 = np.full((4, 4), np.nan, dtype=np.float32)
        data2[:, 2:] = 200.0

        f1 = os.path.join(tmp, 'tile1.tif')
        f2 = os.path.join(tmp, 'tile2.tif')
        out = os.path.join(tmp, 'merged.tif')

        _write_tif(f1, data1, nodata=float('nan'))
        _write_tif(f2, data2, nodata=float('nan'))

        merge_tifs([f1, f2], out, band='B04')

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == pytest.approx(100.0), "Coluna esquerda deve vir do tile1"
        assert result[0, 3] == pytest.approx(200.0), "Coluna direita deve vir do tile2"


def test_nan_nodata_tile_unico_preserva_valores():
    """Tile único com nodata=NaN: valores válidos devem ser preservados no output."""
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[1.5, 2.5], [np.nan, 4.5]], dtype=np.float32)
        f1 = os.path.join(tmp, 'single.tif')
        out = os.path.join(tmp, 'merged.tif')

        _write_tif(f1, data, nodata=float('nan'))
        merge_tifs([f1], out, band='B08')

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == pytest.approx(1.5)
        assert result[0, 1] == pytest.approx(2.5)
        assert result[1, 1] == pytest.approx(4.5)


# ── Métodos antigos: nodata = 0 ───────────────────────────────────────────────

def test_nodata_zero_preenche_tiles_adjacentes():
    """
    Mesma lógica do teste NaN, mas com nodata=0 (métodos antigos: lcf, chrono, ctd).
    Compatibilidade regressiva não deve ter sido quebrada pela correção.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data1 = np.zeros((4, 4), dtype=np.float32)
        data1[:, :2] = 100.0

        data2 = np.zeros((4, 4), dtype=np.float32)
        data2[:, 2:] = 200.0

        f1 = os.path.join(tmp, 'tile1.tif')
        f2 = os.path.join(tmp, 'tile2.tif')
        out = os.path.join(tmp, 'merged.tif')

        _write_tif(f1, data1, nodata=0)
        _write_tif(f2, data2, nodata=0)

        merge_tifs([f1, f2], out, band='B04')

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, 3] == pytest.approx(200.0)


# ── Banda de nuvem: nodata vem do cloud_dict ──────────────────────────────────

def test_cloud_band_scl_usa_nodata_do_cloud_dict():
    """
    Para a banda SCL (S2_L2A-1), o nodata deve vir do cloud_dict (valor 0),
    independente do que está no arquivo.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data1 = np.zeros((4, 4), dtype=np.int16)
        data1[:, :2] = 4   # SCL=4 → vegetação (pixel válido)

        data2 = np.zeros((4, 4), dtype=np.int16)
        data2[:, 2:] = 5   # SCL=5 → solo exposto (pixel válido)

        f1 = os.path.join(tmp, 'scl1.tif')
        f2 = os.path.join(tmp, 'scl2.tif')
        out = os.path.join(tmp, 'scl_merged.tif')

        _write_tif(f1, data1, nodata=0, dtype='int16')
        _write_tif(f2, data2, nodata=0, dtype='int16')

        merge_tifs([f1, f2], out, band='SCL')

        with rasterio.open(out) as src:
            result = src.read(1)
            out_nodata = src.nodata

        assert result[0, 0] == 4, "Coluna esquerda deve ter SCL=4"
        assert result[0, 3] == 5, "Coluna direita deve ter SCL=5"
        assert out_nodata == 0, "Nodata da banda SCL deve ser 0"


# ── Proveniência: int32 nodata = 0 ────────────────────────────────────────────

def test_proveniencia_int32_dois_tiles():
    """
    Arquivos de proveniência são int32 com nodata=0.
    merge_tifs deve ler o 0 do arquivo e mesclá-los corretamente.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data1 = np.zeros((4, 4), dtype=np.int32)
        data1[:, :2] = 167   # dia do ano tile1

        data2 = np.zeros((4, 4), dtype=np.int32)
        data2[:, 2:] = 200   # dia do ano tile2

        f1 = os.path.join(tmp, 'prov1.tif')
        f2 = os.path.join(tmp, 'prov2.tif')
        out = os.path.join(tmp, 'prov_merged.tif')

        _write_tif(f1, data1, nodata=0, dtype='int32')
        _write_tif(f2, data2, nodata=0, dtype='int32')

        merge_tifs([f1, f2], out, band='B04')

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == 167
        assert result[0, 3] == 200
