"""Tests for production config validation (fail-closed checks)."""
from __future__ import annotations

import pytest

from app.config import Settings, validate_production_config


def _make_settings(**overrides) -> Settings:
    defaults = {
        "ENVIRONMENT": "production",
        "API_KEY": "test-secret-key",
        "CORS_ORIGINS": "https://app.example.com",
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestProductionConfigValidation:
    def test_valid_production_config_passes(self):
        s = _make_settings()
        validate_production_config(s)

    def test_missing_api_key_fails(self):
        s = _make_settings(API_KEY=None)
        with pytest.raises(ValueError, match="API_KEY must be set"):
            validate_production_config(s)

    def test_empty_api_key_fails(self):
        s = _make_settings(API_KEY="")
        with pytest.raises(ValueError, match="API_KEY must be set"):
            validate_production_config(s)

    def test_wildcard_cors_fails(self):
        s = _make_settings(CORS_ORIGINS="*")
        with pytest.raises(ValueError, match="CORS_ORIGINS must be an explicit origin list"):
            validate_production_config(s)

    def test_empty_cors_fails(self):
        s = _make_settings(CORS_ORIGINS="")
        with pytest.raises(ValueError, match="CORS_ORIGINS must be an explicit origin list"):
            validate_production_config(s)

    def test_sqlite_database_fails(self):
        s = _make_settings(DATABASE_URL="sqlite:///./storage/dev.db")
        with pytest.raises(ValueError, match="SQLite is not supported in production"):
            validate_production_config(s)

    def test_multiple_errors_reported(self):
        s = _make_settings(API_KEY=None, CORS_ORIGINS="*", DATABASE_URL="sqlite:///x.db")
        with pytest.raises(ValueError, match="API_KEY must be set") as exc_info:
            validate_production_config(s)
        msg = str(exc_info.value)
        assert "CORS_ORIGINS" in msg
        assert "SQLite" in msg

    def test_development_skips_all_checks(self):
        s = _make_settings(ENVIRONMENT="development", API_KEY=None, CORS_ORIGINS="*")
        validate_production_config(s)

    def test_custom_production_string_skips_checks(self):
        s = _make_settings(ENVIRONMENT="staging", API_KEY=None, CORS_ORIGINS="*")
        validate_production_config(s)


class TestDataRetentionSemantics:
    def test_negative_one_is_disabled(self):
        s = Settings(DATA_RETENTION_DAYS=-1)
        assert s.retention_enabled is False

    def test_zero_means_delete_immediately(self):
        s = Settings(DATA_RETENTION_DAYS=0)
        assert s.retention_enabled is True

    def test_positive_means_enabled(self):
        s = Settings(DATA_RETENTION_DAYS=30)
        assert s.retention_enabled is True

    def test_default_is_disabled(self):
        s = Settings()
        assert s.DATA_RETENTION_DAYS == -1
        assert s.retention_enabled is False
