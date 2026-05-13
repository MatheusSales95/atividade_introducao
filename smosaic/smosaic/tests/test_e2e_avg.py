"""
Teste e2e Nível 3 — Parte 1: método avg (média temporal, sem proveniência).

Execute individualmente para não sobrecarregar a memória:
    pytest smosaic/tests/test_e2e_avg.py -v -s
"""
import os
import math
import tempfile

import numpy as np
import pytest
import requests
import rasterio

STAC_URL  = 'https://data.inpe.br/bdc/stac/v1'
COLLECTION = 'S2_L2A-1'
BBOX       = '-54.5,-12.5,-54.4,-12.4'
START      = dict(start_year=2024, start_month=6, start_day=1)
DURATION   = dict(duration_days=10)


def _stac_reachable():
    try:
        return requests.get(STAC_URL, timeout=10).status_code == 200
    except Exception:
        return False


def _cogs_em(output_dir, sufixo='_COG.tif'):
    return [os.path.join(output_dir, f)
            for f in os.listdir(output_dir) if f.endswith(sufixo)]


def _assert_cog_valido(path):
    assert os.path.exists(path), f"Arquivo não encontrado: {path}"
    with rasterio.open(path) as src:
        data   = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            validos = (~np.isnan(data) if (isinstance(nodata, float) and math.isnan(nodata))
                       else data != nodata)
            assert validos.any(), f"Raster sem pixels válidos: {path}"


@pytest.mark.skipif(not _stac_reachable(), reason="Servidor STAC não alcançável")
def test_e2e_avg():
    """
    avg: média temporal, 1 banda (B04).
    Esperado: COG da banda gerado e sem arquivo de proveniência.
    """
    from smosaic import mosaic

    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        mosaic(
            name='test',
            data_dir=data_dir,
            stac_url=STAC_URL,
            collection=COLLECTION,
            output_dir=output_dir,
            **START,
            **DURATION,
            mosaic_method='avg',
            bands=['B04'],
            bbox=BBOX,
            projection_output=4326,
        )

        cogs = _cogs_em(output_dir)
        assert len(cogs) >= 1, "Nenhum COG gerado"

        banda_cogs = [f for f in cogs
                      if 'B04' in os.path.basename(f)
                      and 'cloud' not in f.lower()
                      and 'provenance' not in f.lower()]
        assert len(banda_cogs) >= 1, "COG de B04 não encontrado"
        _assert_cog_valido(banda_cogs[0])

        prov_cogs = [f for f in cogs if 'provenance' in f.lower()]
        assert len(prov_cogs) == 0, "avg não deve gerar proveniência"
