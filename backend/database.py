"""Supabase client singleton for the Chennai FloodSense AI backend.

Uses the service-role key so the backend can bypass Row Level Security when
writing rainfall observations.  The service-role key must NEVER be exposed to
the frontend or committed to version control.
"""

from __future__ import annotations

import os
from functools import lru_cache

from supabase import Client, create_client


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Return a cached Supabase service-role client.

    Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from the environment.
    Raises RuntimeError with a helpful message if either variable is missing.
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the "
            "environment before using the database client."
        )
    return create_client(url, key)


def is_configured() -> bool:
    """Return True when Supabase environment variables are present.

    Used by persistence modules to decide between PostgreSQL and CSV fallback.
    """
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
