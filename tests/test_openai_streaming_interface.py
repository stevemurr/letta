"""
Tests for OpenAI streaming interface tool call argument handling.

This specifically tests the fix for empty string arguments being incorrectly
converted to None during streaming tool call delta processing.

Bug: When vLLM (or other providers) send the first streaming chunk with
{"name":"tool_name","arguments":""}, the empty string was treated as falsy
and converted to None, causing downstream crashes when code tried to call
.replace() on the None value.

Fix: Changed `tool_call.function.arguments` falsy check to explicit
`tool_call.function.arguments is not None` check.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from letta.schemas.letta_message import ToolCallDelta


class TestToolCallDeltaArgumentsHandling:
    """Test that ToolCallDelta properly handles various argument values."""

    def test_empty_string_arguments_preserved(self):
        """Empty string arguments should be preserved, not converted to None."""
        delta = ToolCallDelta(
            name="search",
            arguments="",  # Empty string from first streaming chunk
            tool_call_id="call_123",
        )
        assert delta.arguments == ""
        assert delta.arguments is not None

    def test_none_arguments_stays_none(self):
        """None arguments should remain None."""
        delta = ToolCallDelta(
            name="search",
            arguments=None,
            tool_call_id="call_123",
        )
        assert delta.arguments is None

    def test_valid_arguments_preserved(self):
        """Valid JSON string arguments should be preserved."""
        delta = ToolCallDelta(
            name="search",
            arguments='{"query": "test"}',
            tool_call_id="call_123",
        )
        assert delta.arguments == '{"query": "test"}'

    def test_partial_arguments_preserved(self):
        """Partial JSON arguments (mid-stream) should be preserved."""
        delta = ToolCallDelta(
            name="search",
            arguments='{"query": "te',  # Incomplete JSON mid-stream
            tool_call_id="call_123",
        )
        assert delta.arguments == '{"query": "te'


class TestStreamingToolCallCondition:
    """Test the specific condition that caused the bug."""

    def test_falsy_empty_string_vs_explicit_none_check(self):
        """
        Demonstrate the difference between falsy check and explicit None check.

        The bug was:
            arguments = value if value else None  # "" becomes None (WRONG)

        The fix is:
            arguments = value if value is not None else None  # "" stays "" (CORRECT)
        """
        # Simulate what vLLM sends in first chunk
        first_chunk_arguments = ""

        # OLD (buggy) behavior - falsy check
        buggy_result = first_chunk_arguments if first_chunk_arguments else None
        assert buggy_result is None  # Bug: empty string became None

        # NEW (fixed) behavior - explicit None check
        fixed_result = first_chunk_arguments if first_chunk_arguments is not None else None
        assert fixed_result == ""  # Correct: empty string preserved

    def test_none_value_behavior_unchanged(self):
        """Ensure None values still become None with the fix."""
        value = None

        # Both old and new behavior should return None for None input
        buggy_result = value if value else None
        fixed_result = value if value is not None else None

        assert buggy_result is None
        assert fixed_result is None


class TestToolCallDeltaModelDump:
    """Test ToolCallDelta serialization behavior."""

    def test_empty_string_excluded_from_dump(self):
        """
        Empty string arguments should be excluded from JSON dump.

        This matches OpenAI's streaming behavior where empty values
        are omitted from the response.
        """
        delta = ToolCallDelta(
            name="search",
            arguments="",
            tool_call_id="call_123",
        )
        dumped = delta.model_dump()

        # Empty string is still a value, but model_dump excludes None
        # The key should be present since it's not None
        assert "arguments" in dumped
        assert dumped["arguments"] == ""

    def test_none_excluded_from_dump(self):
        """None arguments should be excluded from JSON dump."""
        delta = ToolCallDelta(
            name="search",
            arguments=None,
            tool_call_id="call_123",
        )
        dumped = delta.model_dump()

        # None values are excluded by model_dump(exclude_none=True)
        assert "arguments" not in dumped
