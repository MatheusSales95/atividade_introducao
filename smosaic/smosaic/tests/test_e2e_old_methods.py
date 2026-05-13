"""
Teste e2e Nível 3 — Métodos antigos: lcf, chrono e ctd.

Verifica que os métodos de composição originais continuam gerando a estrutura
de saída correta (COG de banda + COG de proveniência + COG de nuvem) após as
novas implementações.

Execute individualmente para não sobrecarregar a memória:
    pytest smosaic/tests/test_e2e_old_methods.py -v -s
"""
import math
import os
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


def _assert_proveniencia_valida(path):
    """Verifica que o COG de proveniência tem 1 banda com valores 1–366."""
    assert os.path.exists(path), f"Arquivo de proveniência não encontrado: {path}"
    with rasterio.open(path) as src:
        assert src.count == 1, "Proveniência deve ter 1 banda"
        prov_data = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0
    validos = prov_data[prov_data != nodata]
    assert len(validos) > 0, "Proveniência sem valores de dia-do-ano"
    assert validos.min() >= 1 and validos.max() <= 366, (
        f"Valores fora do intervalo 1-366: min={validos.min()}, max={validos.max()}"
    )


# ── lcf — Menor cobertura de nuvem primeiro ───────────────────────────────────

@pytest.mark.skipif(not _stac_reachable(), reason="Servidor STAC não alcançável")
def test_e2e_lcf():
    """
    lcf: menor cobertura de nuvem primeiro, 1 banda (B04).
    Esperado: COG de banda + COG de proveniência (1 banda, 1-366) + COG de nuvem (SCL).
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
            mosaic_method='lcf',
            bands=['B04'],
            bbox=BBOX,
            projection_output=4326,
        )

        cogs = _cogs_em(output_dir)
        assert len(cogs) >= 1, "Nenhum COG gerado"

        banda_cogs = [f for f in cogs
                      if 'B04' in os.path.basename(f)
                      and 'provenance' not in f.lower()]
        assert len(banda_cogs) >= 1, "COG de B04 não encontrado"
        _assert_cog_valido(banda_cogs[0])

        prov_cogs = [f for f in cogs if 'provenance' in f.lower()]
        assert len(prov_cogs) >= 1, "lcf deve gerar COG de proveniência"
        _assert_proveniencia_valida(prov_cogs[0])

        cloud_cogs = [f for f in cogs if 'SCL' in os.path.basename(f)]
        assert len(cloud_cogs) >= 1, "lcf deve gerar COG de nuvem (SCL)"
        _assert_cog_valido(cloud_cogs[0])


# ── chrono — Ordem cronológica ────────────────────────────────────────────────

@pytest.mark.skipif(not _stac_reachable(), reason="Servidor STAC não alcançável")
def test_e2e_chrono():
    """
    chrono: ordem cronológica, 2 bandas (B04, B08).
    Esperado: COG por banda + 1 COG de proveniência + 1 COG de nuvem (SCL).
    A proveniência é gerada apenas para a primeira banda; a segunda usa merge_scene.
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
            mosaic_method='chrono',
            bands=['B04', 'B08'],
            bbox=BBOX,
            projection_output=4326,
        )

        cogs = _cogs_em(output_dir)
        assert len(cogs) >= 1, "Nenhum COG gerado"

        for banda in ['B04', 'B08']:
            banda_cogs = [f for f in cogs
                          if banda in os.path.basename(f)
                          and 'provenance' not in f.lower()]
            assert len(banda_cogs) >= 1, f"COG de {banda} não encontrado"
            _assert_cog_valido(banda_cogs[0])

        prov_cogs = [f for f in cogs if 'provenance' in f.lower()]
        assert len(prov_cogs) >= 1, "chrono deve gerar COG de proveniência"
        _assert_proveniencia_valida(prov_cogs[0])

        cloud_cogs = [f for f in cogs if 'SCL' in os.path.basename(f)]
        assert len(cloud_cogs) >= 1, "chrono deve gerar COG de nuvem (SCL)"
        _assert_cog_valido(cloud_cogs[0])


# ── ctd — Mais próximo de uma data de referência ──────────────────────────────

@pytest.mark.skipif(not _stac_reachable(), reason="Servidor STAC não alcançável")
def test_e2e_ctd():
    """
    ctd: mais próximo da data de referência, 1 banda (B04).
    reference_date dentro do período (2024-06-05).
    Esperado: COG de banda + COG de proveniência (1 banda, 1-366) + COG de nuvem (SCL).
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
            mosaic_method='ctd',
            bands=['B04'],
            bbox=BBOX,
            projection_output=4326,
            reference_date='2024-06-05',
        )

        cogs = _cogs_em(output_dir)
        assert len(cogs) >= 1, "Nenhum COG gerado"

        banda_cogs = [f for f in cogs
                      if 'B04' in os.path.basename(f)
                      and 'provenance' not in f.lower()]
        assert len(banda_cogs) >= 1, "COG de B04 não encontrado"
        _assert_cog_valido(banda_cogs[0])

        prov_cogs = [f for f in cogs if 'provenance' in f.lower()]
        assert len(prov_cogs) >= 1, "ctd deve gerar COG de proveniência"
        _assert_proveniencia_valida(prov_cogs[0])

        cloud_cogs = [f for f in cogs if 'SCL' in os.path.basename(f)]
        assert len(cloud_cogs) >= 1, "ctd deve gerar COG de nuvem (SCL)"
        _assert_cog_valido(cloud_cogs[0])
