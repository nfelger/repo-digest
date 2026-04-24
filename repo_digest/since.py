import re
from datetime import datetime, timedelta, timezone


def parse_since(since: str) -> datetime:
    """Parse --since value to a UTC-aware datetime.

    Accepts: '7d', '2w', '1m' (relative) or '2026-04-15' (ISO date).
    """
    now = datetime.now(tz=timezone.utc)

    match = re.fullmatch(r"(\d+)([dwm])", since)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        if unit == "d":
            return now - timedelta(days=n)
        elif unit == "w":
            return now - timedelta(weeks=n)
        else:  # m
            return now - timedelta(days=n * 30)

    try:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise ValueError(
            f"Cannot parse --since value: {since!r}. "
            "Use '7d', '2w', '1m', or an ISO date like '2026-04-15'."
        )
