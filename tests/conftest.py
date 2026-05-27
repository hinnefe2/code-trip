"""Shared test fixtures.

The TTS client is used by every test that constructs a :class:`Context`.
A bare ``MagicMock`` for it would return a non-awaitable when ``speak()``
is called, breaking async dispatch code, so we always wire ``speak`` as
an :class:`AsyncMock`. Both a factory function (for use inside ``_make_ctx``
helpers that take other arguments) and a pytest fixture (for tests that
just want a TTS mock) are exposed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def make_mock_tts() -> MagicMock:
    """Build a TTS mock with ``speak`` set up as an AsyncMock.

    Use from inside test-helper functions that build a Context themselves.
    """
    tts = MagicMock()
    tts.is_playing.return_value = False
    tts.speak = AsyncMock(return_value=None)
    return tts


@pytest.fixture
def mock_tts() -> MagicMock:
    """Fixture form of :func:`make_mock_tts` for tests that want it directly."""
    return make_mock_tts()
