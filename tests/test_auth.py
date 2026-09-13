"""Tests for authentication endpoints — register, login, logout, /users/me.

These tests use FastAPI TestClient with a mocked Supabase client so no live
Supabase credentials are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "name": "Test User",
    "email": "test@example.com",
    "password": "password123",
    "native_locality": "Sholinganallur",  # must match locality_lookup.csv exactly
}


def _make_client_and_db(mock_db, monkeypatch, require_email_confirm=False):
    """Build a TestClient where _get_db() returns mock_db."""
    import api.main as main_module
    from fastapi.testclient import TestClient

    monkeypatch.setenv(
        "REQUIRE_EMAIL_CONFIRMATION",
        "true" if require_email_confirm else "false",
    )

    client = TestClient(main_module.app, raise_server_exceptions=False)
    return client


# ---------------------------------------------------------------------------
# Registration — happy path (auto-confirm / demo mode)
# ---------------------------------------------------------------------------

class TestRegisterAutoConfirm:
    def test_register_success_returns_201(self, app_client):
        client, db = app_client
        resp = client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        assert resp.status_code == 201, resp.text

    def test_register_success_response_shape(self, app_client):
        client, db = app_client
        data = client.post("/api/v1/auth/register", json=VALID_PAYLOAD).json()
        assert "id" in data
        assert data["email"] == VALID_PAYLOAD["email"]
        assert data["native_locality"] == VALID_PAYLOAD["native_locality"]
        assert data["requires_verification"] is False

    def test_register_creates_auth_user(self, app_client):
        client, db = app_client
        client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        db.auth.admin.create_user.assert_called_once()
        call_kwargs = db.auth.admin.create_user.call_args[0][0]
        assert call_kwargs["email"] == VALID_PAYLOAD["email"]
        assert call_kwargs["email_confirm"] is True  # demo mode

    def test_register_inserts_profile_row(self, app_client):
        client, db = app_client
        client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        db.table.assert_called_with("users")
        db.table.return_value.insert.assert_called_once()
        profile = db.table.return_value.insert.call_args[0][0]
        assert profile["name"] == VALID_PAYLOAD["name"]
        assert profile["email"] == VALID_PAYLOAD["email"]
        assert profile["native_locality"] == VALID_PAYLOAD["native_locality"]

    def test_register_uuid_matches_auth_user(self, app_client):
        client, db = app_client
        resp = client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        data = resp.json()
        profile_inserted = db.table.return_value.insert.call_args[0][0]
        assert data["id"] == profile_inserted["id"]
        assert data["id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_register_session_returned_when_auto_confirm(self, app_client):
        client, _ = app_client
        data = client.post("/api/v1/auth/register", json=VALID_PAYLOAD).json()
        assert data["session"] is not None
        assert data["session"]["access_token"] == "test-access-token"


# ---------------------------------------------------------------------------
# Registration — email confirmation required mode
# ---------------------------------------------------------------------------

class TestRegisterEmailConfirm:
    def test_register_requires_verification_flag(self, mock_db, monkeypatch):
        monkeypatch.setenv("REQUIRE_EMAIL_CONFIRMATION", "true")
        import api.main as main_module
        from fastapi.testclient import TestClient
        with patch.object(main_module, "_get_db", return_value=mock_db), \
             patch.object(main_module, "_supabase_configured", return_value=True):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        assert resp.status_code == 201
        data = resp.json()
        assert data["requires_verification"] is True
        assert data["session"] is None

    def test_register_email_confirm_false_on_auth_call(self, mock_db, monkeypatch):
        monkeypatch.setenv("REQUIRE_EMAIL_CONFIRMATION", "true")
        import api.main as main_module
        from fastapi.testclient import TestClient
        with patch.object(main_module, "_get_db", return_value=mock_db), \
             patch.object(main_module, "_supabase_configured", return_value=True):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        call_kwargs = mock_db.auth.admin.create_user.call_args[0][0]
        assert call_kwargs["email_confirm"] is False


# ---------------------------------------------------------------------------
# Registration — validation failures
# ---------------------------------------------------------------------------

class TestRegisterValidation:
    def test_invalid_locality_returns_422(self, app_client):
        client, _ = app_client
        bad = {**VALID_PAYLOAD, "native_locality": "Atlantis"}
        resp = client.post("/api/v1/auth/register", json=bad)
        assert resp.status_code == 422
        assert "locality" in resp.json()["detail"].lower() or "unknown" in resp.json()["detail"].lower()

    def test_short_password_returns_422(self, app_client):
        client, _ = app_client
        bad = {**VALID_PAYLOAD, "password": "abc"}
        resp = client.post("/api/v1/auth/register", json=bad)
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, app_client):
        client, _ = app_client
        bad = {**VALID_PAYLOAD, "name": "   "}
        resp = client.post("/api/v1/auth/register", json=bad)
        assert resp.status_code == 422

    def test_duplicate_email_returns_409(self, app_client):
        client, db = app_client
        db.auth.admin.create_user.side_effect = Exception("User already registered")
        resp = client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"].lower()

    def test_profile_insert_failure_cleans_up_auth_user(self, app_client):
        """If profile INSERT fails, the auth user must be deleted (no orphan)."""
        client, db = app_client
        db.table.return_value.insert.return_value.execute.side_effect = Exception("DB error")
        resp = client.post("/api/v1/auth/register", json=VALID_PAYLOAD)
        assert resp.status_code == 500
        # Verify cleanup was attempted
        db.auth.admin.delete_user.assert_called_once_with(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_login_success_returns_200(self, app_client):
        client, db = app_client
        # Profile fetch
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
            data={"id": "aaa", "name": "Test", "email": "test@example.com", "native_locality": "Sholinganallur", "created_at": "2024-01-01"}
        )
        resp = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "password123"})
        assert resp.status_code == 200

    def test_login_response_contains_session_and_user(self, app_client):
        client, db = app_client
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
            data={"id": "aaa", "name": "Test", "email": "test@example.com", "native_locality": "Sholinganallur", "created_at": "2024-01-01"}
        )
        data = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "password123"}).json()
        assert "session" in data
        assert "user" in data
        assert data["session"]["access_token"] == "test-access-token"
        assert data["user"]["native_locality"] == "Sholinganallur"

    def test_login_wrong_password_returns_401(self, app_client):
        client, db = app_client
        db.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")
        resp = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_no_session_returns_401(self, app_client):
        client, db = app_client
        db.auth.sign_in_with_password.return_value = MagicMock(session=None)
        resp = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "password123"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /users/me — JWT-protected endpoint
# ---------------------------------------------------------------------------

class TestUsersMe:
    def test_me_without_token_returns_401(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/users/me")
        assert resp.status_code == 401

    def test_me_with_bad_token_returns_401(self, app_client):
        client, db = app_client
        db.auth.get_user.side_effect = Exception("JWT expired")
        resp = client.get("/api/v1/users/me", headers={"Authorization": "Bearer badtoken"})
        assert resp.status_code == 401

    def test_me_with_valid_token_returns_profile(self, app_client):
        client, db = app_client
        fake_user = MagicMock()
        fake_user.id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        fake_user.email = "test@example.com"
        db.auth.get_user.return_value = MagicMock(user=fake_user)
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
            data={"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "name": "Test", "email": "test@example.com",
                  "native_locality": "Sholinganallur", "created_at": "2024-01-01"}
        )
        resp = client.get("/api/v1/users/me", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["native_locality"] == "Sholinganallur"
        assert data["id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_me_token_not_trusted_for_id(self, app_client):
        """Verify the endpoint does NOT trust a user-supplied id; it uses the JWT-validated one."""
        client, db = app_client
        fake_user = MagicMock()
        fake_user.id = "server-side-uuid"
        fake_user.email = "test@example.com"
        db.auth.get_user.return_value = MagicMock(user=fake_user)
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
            data={"id": "server-side-uuid", "name": "T", "email": "t@e.com", "native_locality": "Alandur", "created_at": ""}
        )
        client.get("/api/v1/users/me", headers={"Authorization": "Bearer any-token"})
        # The .eq() call must use the server-verified UUID, not any client-supplied value
        db.table.return_value.select.return_value.eq.assert_called_with("id", "server-side-uuid")


# ---------------------------------------------------------------------------
# Locality validation helper
# ---------------------------------------------------------------------------

class TestLocalityValidation:
    def test_known_locality_validates(self, app_client):
        import api.main as m
        assert m._canonical_locality("Sholinganallur") == "Sholinganallur"

    def test_case_insensitive_match(self, app_client):
        import api.main as m
        result = m._canonical_locality("sholinganallur")
        assert result is not None
        assert result == "Sholinganallur"

    def test_unknown_locality_returns_none(self, app_client):
        import api.main as m
        assert m._canonical_locality("Mars Colony") is None

    def test_known_localities_not_empty(self, app_client):
        import api.main as m
        assert len(m._known_localities()) > 0


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
