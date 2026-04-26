"""
Tests unitarios para src/utils/config.py (estructura multi-token).
"""
from __future__ import annotations

import pytest
import yaml

from src.utils.config import _validate, load_config


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def make_valid_config(**overrides) -> dict:
    """Config válido mínimo con estructura multi-token."""
    base = {
        "tokens": [
            {"symbol": "BTC/USDT", "num_levels": 6, "min_step_usdt": 600},
        ],
        "grid": {
            "total_capital_usdt": 5.0,
            "max_order_usdt": 1.0,
            "range_width_pct": 5.0,
        },
        "risk": {"max_daily_loss_usdt": 2.0},
        "loop": {"interval_seconds": 5, "max_consecutive_errors": 3},
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------
# Validación de tokens
# ------------------------------------------------------------------

class TestValidateTokens:
    def test_config_valido_no_lanza(self):
        _validate(make_valid_config())

    def test_tokens_ausente_lanza(self):
        cfg = make_valid_config()
        del cfg["tokens"]
        with pytest.raises(ValueError, match="tokens"):
            _validate(cfg)

    def test_tokens_lista_vacia_lanza(self):
        cfg = make_valid_config(tokens=[])
        with pytest.raises(ValueError, match="tokens"):
            _validate(cfg)

    def test_tokens_no_es_lista_lanza(self):
        cfg = make_valid_config(tokens="BTC/USDT")
        with pytest.raises(ValueError, match="tokens"):
            _validate(cfg)

    def test_token_sin_symbol_lanza(self):
        cfg = make_valid_config(tokens=[{"num_levels": 6, "min_step_usdt": 600}])
        with pytest.raises(ValueError, match="symbol"):
            _validate(cfg)

    def test_token_symbol_vacio_lanza(self):
        cfg = make_valid_config(tokens=[{"symbol": "", "num_levels": 6, "min_step_usdt": 600}])
        with pytest.raises(ValueError, match="symbol"):
            _validate(cfg)

    def test_token_num_levels_menor_2_lanza(self):
        cfg = make_valid_config(tokens=[{"symbol": "BTC/USDT", "num_levels": 1, "min_step_usdt": 600}])
        with pytest.raises(ValueError, match="num_levels"):
            _validate(cfg)

    def test_token_num_levels_exactamente_2_ok(self):
        cfg = make_valid_config(tokens=[{"symbol": "BTC/USDT", "num_levels": 2, "min_step_usdt": 600}])
        _validate(cfg)  # no debe lanzar

    def test_token_min_step_usdt_cero_lanza(self):
        cfg = make_valid_config(tokens=[{"symbol": "BTC/USDT", "num_levels": 6, "min_step_usdt": 0}])
        with pytest.raises(ValueError, match="min_step_usdt"):
            _validate(cfg)

    def test_token_min_step_usdt_negativo_lanza(self):
        cfg = make_valid_config(tokens=[{"symbol": "BTC/USDT", "num_levels": 6, "min_step_usdt": -1}])
        with pytest.raises(ValueError, match="min_step_usdt"):
            _validate(cfg)

    def test_multiples_tokens_validos(self):
        cfg = make_valid_config(tokens=[
            {"symbol": "BTC/USDT", "num_levels": 6, "min_step_usdt": 600},
            {"symbol": "ETH/USDT", "num_levels": 6, "min_step_usdt": 30},
            {"symbol": "SOL/USDT", "num_levels": 6, "min_step_usdt": 3},
        ])
        _validate(cfg)  # no debe lanzar

    def test_segundo_token_invalido_lanza_con_indice(self):
        cfg = make_valid_config(tokens=[
            {"symbol": "BTC/USDT", "num_levels": 6, "min_step_usdt": 600},
            {"symbol": "ETH/USDT", "num_levels": 1, "min_step_usdt": 30},  # inválido
        ])
        with pytest.raises(ValueError, match=r"tokens\[1\]"):
            _validate(cfg)


# ------------------------------------------------------------------
# Validación de grid
# ------------------------------------------------------------------

class TestValidateGrid:
    def test_total_capital_cero_lanza(self):
        cfg = make_valid_config(grid={"total_capital_usdt": 0, "max_order_usdt": 1.0, "range_width_pct": 5.0})
        with pytest.raises(ValueError, match="total_capital_usdt"):
            _validate(cfg)

    def test_max_order_cero_lanza(self):
        cfg = make_valid_config(grid={"total_capital_usdt": 5.0, "max_order_usdt": 0, "range_width_pct": 5.0})
        with pytest.raises(ValueError, match="max_order_usdt"):
            _validate(cfg)

    def test_range_width_pct_cero_lanza(self):
        cfg = make_valid_config(grid={"total_capital_usdt": 5.0, "max_order_usdt": 1.0, "range_width_pct": 0})
        with pytest.raises(ValueError, match="range_width_pct"):
            _validate(cfg)

    def test_range_width_pct_negativo_lanza(self):
        cfg = make_valid_config(grid={"total_capital_usdt": 5.0, "max_order_usdt": 1.0, "range_width_pct": -1.0})
        with pytest.raises(ValueError, match="range_width_pct"):
            _validate(cfg)


# ------------------------------------------------------------------
# Validación de risk y loop
# ------------------------------------------------------------------

class TestValidateRiskAndLoop:
    def test_max_daily_loss_cero_lanza(self):
        cfg = make_valid_config(risk={"max_daily_loss_usdt": 0})
        with pytest.raises(ValueError, match="max_daily_loss_usdt"):
            _validate(cfg)

    def test_interval_seconds_menor_5_lanza(self):
        cfg = make_valid_config(loop={"interval_seconds": 4, "max_consecutive_errors": 3})
        with pytest.raises(ValueError, match="interval_seconds"):
            _validate(cfg)

    def test_max_consecutive_errors_cero_lanza(self):
        cfg = make_valid_config(loop={"interval_seconds": 5, "max_consecutive_errors": 0})
        with pytest.raises(ValueError, match="max_consecutive_errors"):
            _validate(cfg)

    def test_trailing_recenter_cero_lanza(self):
        cfg = make_valid_config()
        cfg["trailing"] = {"recenter_every_cycles": 0}
        with pytest.raises(ValueError, match="recenter_every_cycles"):
            _validate(cfg)

    def test_trailing_ausente_no_lanza(self):
        cfg = make_valid_config()
        cfg.pop("trailing", None)
        _validate(cfg)  # trailing es opcional


# ------------------------------------------------------------------
# load_config con archivo real
# ------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_yaml_real(self):
        """Verifica que config.yaml actual (multi-token) carga y valida correctamente."""
        cfg = load_config("config.yaml")
        assert "tokens" in cfg
        assert len(cfg["tokens"]) >= 1
        assert "grid" in cfg
        assert "range_width_pct" in cfg["grid"]

    def test_load_config_archivo_inexistente_lanza(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "no_existe.yaml"))

    def test_load_config_retorna_dry_run(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        cfg = load_config("config.yaml")
        assert cfg["dry_run"] is True

    def test_load_config_dry_run_false(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "false")
        cfg = load_config("config.yaml")
        assert cfg["dry_run"] is False

    def test_load_config_log_level_env_override(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        cfg = load_config("config.yaml")
        assert cfg["logging"]["level"] == "DEBUG"

    def test_load_config_yaml_invalido_lanza(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "tokens:\n  - symbol: 'X'\n    num_levels: 1\n    min_step_usdt: 10\n"
            "grid:\n  total_capital_usdt: 5.0\n  max_order_usdt: 1.0\n  range_width_pct: 5.0\n"
            "risk:\n  max_daily_loss_usdt: 2.0\n"
            "loop:\n  interval_seconds: 5\n  max_consecutive_errors: 3\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="num_levels"):
            load_config(str(bad_yaml))
