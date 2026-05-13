"""
Teste e2e Nível 3 — Parte 3: método maxx (k-ésimo máximo pela banda de referência).

Execute individualmente para não sobrecarregar a memória:
    pytest smosaic/tests/test_e2e_maxx.py -v -s
"""
import os
import math
import tempfile

import numpy as np
import pytest
import requests
import rasterio

STAC_URL   = 'https://data.inpe.br/bdc/stac/v1'
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
def test_e2e_maxx():
    """
    maxx: k-ésimo máximo usando B12 como referência, 3 bandas (B04, B08, B12).
    Esperado: COG por banda, COG de nuvem e COG de proveniência com 1 banda
              contendo valores no intervalo 1-366 (dia do ano).
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
            mosaic_method='maxx',
            bands=['B04', 'B08', 'B12'],
            bbox=BBOX,
            projection_output=4326,
            k=1,
            banda_ref='B12',
        )

        cogs = _cogs_em(output_dir)
        assert len(cogs) >= 1, "Nenhum COG gerado"

        for banda in ['B04', 'B08', 'B12']:
            banda_cogs = [f for f in cogs
                          if banda in os.path.basename(f)
                          and 'cloud' not in f.lower()
                          and 'provenance' not in f.lower()]
            assert len(banda_cogs) >= 1, f"COG de {banda} não encontrado"
            _assert_cog_valido(banda_cogs[0])

        prov_cogs = [f for f in cogs if 'provenance' in f.lower()]
        assert len(prov_cogs) >= 1, "maxx deve gerar proveniência"

        with rasterio.open(prov_cogs[0]) as src:
            assert src.count == 1, "Proveniência do maxx deve ter 1 banda"
            prov_data = src.read(1)
            validos = prov_data[prov_data != 0]
            assert len(validos) > 0, "Proveniência sem valores de dia-do-ano"
            assert validos.min() >= 1 and validos.max() <= 366, (
                f"Valores fora do intervalo 1-366: min={validos.min()}, max={validos.max()}"
            )
