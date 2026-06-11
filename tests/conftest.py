"""Shared test fixtures.

The TTS client is used by every test that constructs a :class:`Context`.
A bare ``MagicMock`` for it would return a non-awaitable when ``speak()``
is called, breaking async dispatch code, so we always wire ``speak`` as
an :class:`AsyncMock`. Both a factory function (for use inside ``_make_ctx``
helpers that take other arguments) and a pytest fixture (for tests that
just want a TTS mock) are exposed.

The mock is built with :func:`create_autospec` against :class:`TTSClient`
so accessing a method name that doesn't exist on the real class raises
``AttributeError`` *and* calling a method with the wrong signature raises
``TypeError`` at test time. Plain ``MagicMock(spec=X)`` only checks the
former — not enough to catch positional/kwarg collisions like the
``kind=task.kind`` regression that ate every ``queue_turn`` event for
months. ``create_autospec`` is what we want for our own classes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, create_autospec

import pytest

from code_trip2.tts_client import TTSClient


def make_mock_tts():
    """Build a TTS mock with ``speak`` set up as an AsyncMock.

    Use from inside test-helper functions that build a Context themselves.
    """
    tts = create_autospec(TTSClient, instance=True)
    tts.is_playing.return_value = False
    tts.speak = AsyncMock(return_value=None)
    return tts


@pytest.fixture
def mock_tts() -> MagicMock:
    """Fixture form of :func:`make_mock_tts` for tests that want it directly."""
    return make_mock_tts()
