"""Shared pytest fixtures for Chennai FloodSense AI tests.

All fixtures that require Supabase credentials use mocking so tests run
fully offline without any live service configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on the path for all tests.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Supabase environment stub — prevents real credential requirement in tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_supabase_env(monkeypatch):
    """Set dummy Supabase env vars so is_configured() returns True in tests."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")


@pytest.fixture
def mock_db():
    """Return a MagicMock Supabase client and patch _get_db to return it."""
    db = MagicMock()
    # Default: admin.create_user returns a user with a fixed UUID
    fake_user = MagicMock()
    fake_user.id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    fake_user.email = "test@example.com"
    db.auth.admin.create_user.return_value = MagicMock(user=fake_user)

    # Default: sign_in_with_password returns a valid session
    fake_session = MagicMock()
    fake_session.access_token = "test-access-token"
    fake_session.refresh_token = "test-refresh-token"
    fake_session.expires_in = 3600
    db.auth.sign_in_with_password.return_value = MagicMock(session=fake_session, user=fake_user)

    # Default: table().insert().execute() succeeds
    db.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{}])
    # Default: table().select().eq().single().execute() returns empty profile
    db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(data={})

    return db


@pytest.fixture
def app_client(mock_db):
    """Return a FastAPI TestClient with Supabase patched out."""
    from fastapi.testclient import TestClient
    import api.main as main_module

    with patch.object(main_module, "_get_db", return_value=mock_db), \
         patch.object(main_module, "_supabase_configured", return_value=True):
        from importlib import reload
        # Re-import to pick up patches if needed
        client = TestClient(main_module.app, raise_server_exceptions=False)
        yield client, mock_db
