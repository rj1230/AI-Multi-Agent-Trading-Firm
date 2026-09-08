"""Sanity checks for risk mandate loading, incl. the sector_map addition."""

from config.risk_config import load_risk_config


def test_default_mandate_loads():
    config = load_risk_config()
    assert config.per_ticker_cap_pct == 0.20
    assert config.per_sector_cap_pct == 0.40


def test_sector_map_loads_from_default_mandate():
    config = load_risk_config()
    assert config.sector_map["AAPL"] == "technology"
    assert config.sector_map["JPM"] == "financials"
    assert len(config.sector_map) > 0


def test_sector_cap_validator_rejects_inverted_caps(tmp_path):
    import yaml

    bad_mandate = {
        "position_sizing": {"equity_risk_pct": 0.01, "atr_stop_multiple": 1.5},
        "per_ticker_cap_pct": 0.40,
        "per_sector_cap_pct": 0.20,  # smaller than ticker cap -- should fail
        "max_concurrent_positions": 3,
        "circuit_breaker": {"daily_drawdown_pct": -0.03},
        "correlation": {"max_correlation": 0.7},
    }
    bad_path = tmp_path / "bad_mandate.yaml"
    bad_path.write_text(yaml.dump(bad_mandate))

    import pytest

    with pytest.raises(ValueError):
        load_risk_config(bad_path)
