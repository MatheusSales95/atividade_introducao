import datetime
import json
import os
import tempfile

import pytest
import shapely.geometry

from smosaic.smosaic_utils import (
    add_days_to_date,
    add_months_to_date,
    clean_dir,
    create_composition_json,
    days_between_dates,
    geometry_collides_with_bbox,
    get_all_cloud_configs,
)


# ── get_all_cloud_configs ─────────────────────────────────────────────────────

def test_cloud_config_colecoes_presentes():
    cfg = get_all_cloud_configs()
    assert 'S2_L2A-1' in cfg
    assert 'S2_L1C_BUNDLE-1' in cfg


def test_cloud_config_s2l2a1_banda_e_nodata():
    cfg = get_all_cloud_configs()['S2_L2A-1']
    assert cfg['cloud_band'] == 'SCL'
    assert cfg['no_data_value'] == 0
    assert 4 in cfg['non_cloud_values']
    assert 5 in cfg['non_cloud_values']
    assert 6 in cfg['non_cloud_values']


def test_cloud_config_s2l1c_banda_e_nodata():
    cfg = get_all_cloud_configs()['S2_L1C_BUNDLE-1']
    assert cfg['cloud_band'] == 'FMASK'
    assert cfg['no_data_value'] == 255
    assert 0 in cfg['non_cloud_values']
    assert 1 in cfg['non_cloud_values']


def test_cloud_config_retorna_copia_independente():
    """Modificar o retorno não deve alterar o config original."""
    cfg1 = get_all_cloud_configs()
    cfg1['S2_L2A-1']['no_data_value'] = 999
    cfg2 = get_all_cloud_configs()
    assert cfg2['S2_L2A-1']['no_data_value'] == 0


# ── add_months_to_date ────────────────────────────────────────────────────────

def test_add_months_to_date_retorna_ultimo_dia_do_mes():
    result = add_months_to_date('2024-01-01', 1)
    assert result.month == 2
    assert result.day == 29  # 2024 é bissexto


def test_add_months_to_date_aceita_datetime():
    start = datetime.datetime(2024, 3, 15)
    result = add_months_to_date(start, 1)
    assert result.month == 4
    assert result.day == 30


# ── days_between_dates ────────────────────────────────────────────────────────

def test_days_between_dates_diferenca_positiva():
    # date1 no formato YYYY-MM-DD, date2 no formato YYYYMMDD
    diff = days_between_dates('2024-01-01', '20240106')
    assert diff == 5


def test_days_between_dates_mesmo_dia():
    diff = days_between_dates('2024-06-15', '20240615')
    assert diff == 0


# ── add_days_to_date ──────────────────────────────────────────────────────────

def test_add_days_to_date_soma_dias():
    result = add_days_to_date('2024-01-01', 10)
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 11


def test_add_days_to_date_vira_mes():
    result = add_days_to_date('2024-01-28', 5)
    assert result.month == 2
    assert result.day == 2


# ── geometry_collides_with_bbox ───────────────────────────────────────────────

def test_geometry_intersecta_bbox():
    geom = shapely.geometry.box(-1, -1, 1, 1)
    assert geometry_collides_with_bbox(geom, (-0.5, -0.5, 0.5, 0.5)) is True


def test_geometry_nao_intersecta_bbox():
    geom = shapely.geometry.box(10, 10, 20, 20)
    assert geometry_collides_with_bbox(geom, (-5, -5, 5, 5)) is False


# ── clean_dir ─────────────────────────────────────────────────────────────────

def test_clean_dir_sem_args_remove_tifs_nao_cog():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, 'scene.tif'), 'w').close()
        open(os.path.join(tmp, 'scene_COG.tif'), 'w').close()
        open(os.path.join(tmp, 'keep.txt'), 'w').close()

        clean_dir(tmp)

        files = os.listdir(tmp)
        assert 'scene.tif' not in files, "TIF comum deve ser removido"
        assert 'scene_COG.tif' in files, "COG não deve ser removido"
        assert 'keep.txt' in files, "Arquivos não-TIF não devem ser removidos"


# ── create_composition_json ───────────────────────────────────────────────────

def test_create_composition_json_cria_arquivo():
    with tempfile.TemporaryDirectory() as tmp:
        path = create_composition_json(
            output_dir=tmp,
            collection='S2_L2A-1',
            input_scenes=['21LXH', '21LYH'],
            ignored_scenes=['21LZH'],
            used_scenes=['21LXH'],
        )
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data['collection'] == 'S2_L2A-1'
        assert '21LXH' in data['input_scenes']
        assert '21LZH' in data['ignored_scenes']
        assert data['used_scenes'] == ['21LXH']
