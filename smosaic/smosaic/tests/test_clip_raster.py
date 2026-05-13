import os
import shutil
import tempfile

import numpy as np
import pytest
import rasterio
import shapely.geometry
from rasterio.transform import from_bounds

from smosaic.smosaic_clip_raster import clip_raster


def _write_tif(path, data, nodata=0, dtype='float32', crs='EPSG:4326',
               bounds=(-1, -1, 1, 1)):
    h, w = data.shape
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, w, h)
    profile = dict(
        driver='GTiff', dtype=dtype, count=1,
        width=w, height=h, nodata=nodata,
        crs=crs, transform=transform,
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data[np.newaxis, :, :])


# ── Clipping sem reprojeção ───────────────────────────────────────────────────

def test_clip_raster_recorta_area_menor():
    """
    Raster 8x8 cobrindo [-1,-1,1,1]. Clipe para [-0.5,-0.5,0.5,0.5].
    O output deve ser menor que 8x8.
    """
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        data = np.ones((8, 8), dtype=np.float32) * 100
        src_path = os.path.join(src_dir, 'raster.tif')
        _write_tif(src_path, data, bounds=(-1, -1, 1, 1))

        # clip_raster DELETA o input, então criamos uma cópia
        input_copy = os.path.join(src_dir, 'raster_copy.tif')
        shutil.copy(src_path, input_copy)

        result_path = clip_raster(
            input_raster_path=input_copy,
            output_folder=out_dir,
            clip_geometry=shapely.geometry.box(-0.5, -0.5, 0.5, 0.5),
            projection_output=4326,
            output_filename='clipped.tif',
        )

        assert os.path.exists(result_path)
        with rasterio.open(result_path) as src:
            assert src.width < 8 or src.height < 8, "Clipe não reduziu o tamanho"


def test_clip_raster_retorna_path_correto():
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        data = np.ones((4, 4), dtype=np.float32) * 50
        input_path = os.path.join(src_dir, 'input.tif')
        _write_tif(input_path, data, bounds=(-1, -1, 1, 1))

        result_path = clip_raster(
            input_raster_path=input_path,
            output_folder=out_dir,
            clip_geometry=shapely.geometry.box(-0.8, -0.8, 0.8, 0.8),
            projection_output=4326,
            output_filename='output.tif',
        )

        assert result_path == os.path.join(out_dir, 'output.tif')
        assert not os.path.exists(input_path), "Input deve ser removido após clipe"


def test_clip_raster_preserva_valores_validos():
    """Pixels com valor conhecido devem ser preservados no output."""
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        data = np.full((8, 8), 42.0, dtype=np.float32)
        input_path = os.path.join(src_dir, 'raster.tif')
        _write_tif(input_path, data, bounds=(-1, -1, 1, 1))

        result_path = clip_raster(
            input_raster_path=input_path,
            output_folder=out_dir,
            clip_geometry=shapely.geometry.box(-0.5, -0.5, 0.5, 0.5),
            projection_output=4326,
            output_filename='out.tif',
        )

        with rasterio.open(result_path) as src:
            out = src.read(1)
        valid = out[out != 0]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 42.0, rtol=1e-5)
