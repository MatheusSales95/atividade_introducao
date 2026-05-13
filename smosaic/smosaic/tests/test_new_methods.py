import numpy as np
import xarray as xr
import datetime
import pytest

from smosaic.smosaic_new_methods import (
    calcular_nbr_cubo,
    _gerar_composicao_media,
    _gerar_composicao_mediana,
    _gerar_composicao_max,
    _gerar_composicao_min,
    _gerar_composicao_extrema,
    _proveniencia_para_int,
    _aplicar_metodo,
)


def _make_cubo(values_b04, values_b08, values_b12):
    """Cria um Dataset (time=3, y=2, x=2) com valores controlados."""
    times = np.array([
        '2024-01-01', '2024-02-01', '2024-03-01'
    ], dtype='datetime64[ns]')
    return xr.Dataset(
        {
            'B04': (('time', 'y', 'x'), np.array(values_b04, dtype=np.float32).reshape(3, 2, 2)),
            'B08': (('time', 'y', 'x'), np.array(values_b08, dtype=np.float32).reshape(3, 2, 2)),
            'B12': (('time', 'y', 'x'), np.array(values_b12, dtype=np.float32).reshape(3, 2, 2)),
        },
        coords={'time': times, 'y': [0.0, 1.0], 'x': [0.0, 1.0]},
    )


# ── Proveniência ─────────────────────────────────────────────────────────────

def test_proveniencia_dia_do_ano():
    """Verifica que o formato é dia-do-ano (1-365), não YYYYMMDD."""
    times = xr.DataArray(
        np.array(['2024-01-01', '2024-06-15', '2024-12-31'], dtype='datetime64[ns]'),
        dims='time'
    )
    result = _proveniencia_para_int(times)
    assert result[0] == 1    # 1 de janeiro = dia 1
    assert result[1] == 167  # 15 de junho = dia 167 (2024 é bissexto)
    assert result[2] == 366  # 31 de dezembro num ano bissexto = dia 366


def test_proveniencia_nodata_zero_em_pixels_mascarados():
    times = xr.DataArray(
        np.array(['2024-06-15'], dtype='datetime64[ns]'), dims='time'
    )
    mask = np.array([True])   # pixel mascarado
    result = _proveniencia_para_int(times, mask_nodata=mask)
    assert result[0] == 0


# ── Composições ──────────────────────────────────────────────────────────────

def test_media_ignora_nan():
    cubo = _make_cubo(
        values_b04=[1, 2, np.nan, 4,   # time0: pixel(0,0)=1, pixel(0,1)=2, pixel(1,0)=nan, pixel(1,1)=4
                    3, 6,      3,  8,   # time1
                    5, 4,      9,  2],  # time2
        values_b08=[[1]*4]*3,
        values_b12=[[1]*4]*3,
    )
    compo, prov = _gerar_composicao_media(cubo)
    # pixel(0,0): mean(1,3,5) = 3.0
    assert float(compo['B04'].values[0, 0]) == pytest.approx(3.0)
    assert prov is None


def test_max_retorna_valor_maximo_por_banda():
    cubo = _make_cubo(
        values_b04=[10, 10, 10, 10,
                    50, 50, 50, 50,
                    20, 20, 20, 20],
        values_b08=[[1]*4]*3,
        values_b12=[[1]*4]*3,
    )
    compo, prov = _gerar_composicao_max(cubo, bandas=['B04'])
    assert float(compo['B04'].values[0, 0]) == 50.0


def test_min_retorna_valor_minimo_por_banda():
    cubo = _make_cubo(
        values_b04=[10, 10, 10, 10,
                    50, 50, 50, 50,
                    20, 20, 20, 20],
        values_b08=[[1]*4]*3,
        values_b12=[[1]*4]*3,
    )
    compo, prov = _gerar_composicao_min(cubo, bandas=['B04'])
    assert float(compo['B04'].values[0, 0]) == 10.0


def test_maxx_usa_banda_referencia():
    """maxx com B12 como referência: o pixel com maior B12 é selecionado para todas as bandas."""
    cubo = _make_cubo(
        values_b04=[100, 100, 100, 100,   # time 0: B04=100
                    200, 200, 200, 200,   # time 1: B04=200
                    150, 150, 150, 150],  # time 2: B04=150
        values_b08=[[1]*4]*3,
        values_b12=[ 5,  5,  5,  5,      # time 0: B12=5
                    99, 99, 99, 99,       # time 1: B12=99 ← máximo
                    10, 10, 10, 10],      # time 2: B12=10
    )
    compo, prov = _gerar_composicao_extrema(cubo, banda_ref='B12', enesimo=1, tipo='max')
    # Como B12 é maior no time 1, B04 do resultado deve ser 200
    assert float(compo['B04'].values[0, 0]) == 200.0


def test_nbr_calculo():
    cubo = _make_cubo(
        values_b04=[[0]*4]*3,
        values_b08=[1000, 1000, 1000, 1000,
                    1000, 1000, 1000, 1000,
                    1000, 1000, 1000, 1000],
        values_b12=[ 500,  500,  500,  500,
                     500,  500,  500,  500,
                     500,  500,  500,  500],
    )
    cubo_nbr = calcular_nbr_cubo(cubo, nir_band='B08', swir_band='B12')
    assert 'NBR' in cubo_nbr
    # NBR = (1000-500)/(1000+500) = 500/1500 ≈ 0.333
    assert float(cubo_nbr['NBR'].values[0, 0, 0]) == pytest.approx(500 / 1500, abs=1e-4)


# ── Roteamento ───────────────────────────────────────────────────────────────

def test_aplicar_metodo_routing():
    cubo = _make_cubo([[1]*4]*3, [[1]*4]*3, [[1]*4]*3)
    for method in ('avg', 'media'):
        compo, prov = _aplicar_metodo(cubo, method, ['B04', 'B08', 'B12'])
        assert prov is None
    for method in ('med', 'mediana'):
        compo, prov = _aplicar_metodo(cubo, method, ['B04', 'B08', 'B12'])
        assert prov is None
