from datetime import datetime, timezone, date
import pytest
from repo_digest.since import parse_since


def test_parse_days():
    dt = parse_since("7d")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 6 <= delta.days <= 7


def test_parse_weeks():
    dt = parse_since("2w")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 13 <= delta.days <= 14


def test_parse_months():
    dt = parse_since("1m")
    delta = datetime.now(tz=timezone.utc) - dt
    assert 29 <= delta.days <= 30


def test_parse_iso_date():
    dt = parse_since("2026-04-15")
    assert dt.date() == date(2026, 4, 15)
    assert dt.tzinfo is not None


def test_parse_invalid_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_since("yesterday")


def test_result_is_utc():
    dt = parse_since("3d")
    assert dt.tzinfo == timezone.utc
