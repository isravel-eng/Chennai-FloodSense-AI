"""Tests for rainfall data layer — repository upsert, duplicate prevention, CSV fallback.

These tests verify:
  - The upsert uses (locality, observed_at) as the conflict key.
  - Duplicate calls for the same locality+date do not create duplicate rows.
  - The CSV fallback activates when SUPABASE_URL is absent.
  - The read() function returns the expected DataFrame shape.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Repository upsert tests (PostgreSQL path)
# ---------------------------------------------------------------------------

class TestRainfallRepositoryUpsert:
    def test_upsert_calls_supabase_with_correct_conflict_key(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = MagicMock()

        with patch("backend.database.get_client", return_value=mock_client):
            # Re-import to pick up mocked get_client
            import importlib
            import backend.repositories.rainfall_repository as repo
            importlib.reload(repo)
            repo.upsert("Sholinganallur", date(2024, 11, 15), 12.5, "open_meteo")

        mock_client.table.assert_called_with("rainfall_logs")
        upsert_call = mock_client.table.return_value.upsert.call_args
        payload = upsert_call[0][0]
        assert payload["locality"] == "Sholinganallur"
        assert payload["observed_at"] == "2024-11-15"
        assert payload["rainfall_mm"] == 12.5
        assert payload["source"] == "open_meteo"
        # Verify conflict key
        kwargs = upsert_call[1]
        assert kwargs.get("on_conflict") == "locality,observed_at"

    def test_upsert_duplicate_does_not_call_insert_twice(self, monkeypatch):
        """Calling upsert twice with the same (locality, date) must not error."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = MagicMock()

        with patch("backend.database.get_client", return_value=mock_client):
            import importlib
            import backend.repositories.rainfall_repository as repo
            importlib.reload(repo)
            repo.upsert("Alandur", date(2024, 12, 1), 5.0)
            repo.upsert("Alandur", date(2024, 12, 1), 8.0)  # update same day

        # Both calls should succeed — the DB handles idempotency via ON CONFLICT
        assert mock_client.table.return_value.upsert.return_value.execute.call_count == 2


# ---------------------------------------------------------------------------
# Repository read tests
# ---------------------------------------------------------------------------

class TestRainfallRepositoryRead:
    def test_read_returns_expected_columns(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        rows = [
            {"observed_at": "2024-11-10", "locality": "Sholinganallur", "rainfall_mm": 5.0},
            {"observed_at": "2024-11-11", "locality": "Sholinganallur", "rainfall_mm": 8.5},
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.ilike.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=rows)

        with patch("backend.database.get_client", return_value=mock_client):
            import importlib
            import backend.repositories.rainfall_repository as repo
            importlib.reload(repo)
            df = repo.read("Sholinganallur")

        assert list(df.columns) == ["date", "locality", "rainfall_mm"]
        assert len(df) == 2

    def test_read_returns_empty_df_when_no_rows(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.ilike.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

        with patch("backend.database.get_client", return_value=mock_client):
            import importlib
            import backend.repositories.rainfall_repository as repo
            importlib.reload(repo)
            df = repo.read("UnknownLocality")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "locality", "rainfall_mm"]
        assert len(df) == 0

    def test_read_sorts_ascending_by_date(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        rows = [
            {"observed_at": "2024-11-12", "locality": "Alandur", "rainfall_mm": 3.0},
            {"observed_at": "2024-11-10", "locality": "Alandur", "rainfall_mm": 1.0},
            {"observed_at": "2024-11-11", "locality": "Alandur", "rainfall_mm": 2.0},
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.ilike.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=rows)

        with patch("backend.database.get_client", return_value=mock_client):
            import importlib
            import backend.repositories.rainfall_repository as repo
            importlib.reload(repo)
            df = repo.read("Alandur")

        dates = df["date"].tolist()
        assert dates == sorted(dates), "Read() must return dates in ascending order"


# ---------------------------------------------------------------------------
# CSV fallback path
# ---------------------------------------------------------------------------

class TestCSVFallback:
    def test_use_postgres_returns_false_without_env(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        from live import rainfall_history
        import importlib
        importlib.reload(rainfall_history)
        assert rainfall_history._use_postgres() is False

    def test_use_postgres_returns_true_with_env(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
        from live import rainfall_history
        import importlib
        importlib.reload(rainfall_history)
        assert rainfall_history._use_postgres() is True

    def test_log_observation_writes_csv_without_supabase(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

        from live import rainfall_history
        import importlib
        importlib.reload(rainfall_history)

        # Redirect log to a temp file
        log_path = tmp_path / "live_rainfall_log.csv"
        log = rainfall_history.RainfallLog(path=log_path)
        log.append("Sholinganallur", 10.0, date(2024, 11, 20))

        df = log.read("Sholinganallur")
        assert len(df) == 1
        assert float(df.iloc[0]["rainfall_mm"]) == 10.0


# ---------------------------------------------------------------------------
# Rainfall API endpoint
# ---------------------------------------------------------------------------

class TestRainfallLogsEndpoint:
    def test_rainfall_log_endpoint_requires_auth(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/rainfall-logs/Sholinganallur")
        assert resp.status_code == 401

    def test_rainfall_log_endpoint_with_valid_token(self, app_client):
        client, db = app_client
        fake_user = MagicMock()
        fake_user.id = "aaa"
        fake_user.email = "t@e.com"
        db.auth.get_user.return_value = MagicMock(user=fake_user)
        db.table.return_value.select.return_value.ilike.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"observed_at": "2024-11-15", "locality": "Sholinganallur", "rainfall_mm": 5.0, "source": "open_meteo"}]
        )
        resp = client.get(
            "/api/v1/rainfall-logs/Sholinganallur",
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["observations"][0]["rainfall_mm"] == 5.0
