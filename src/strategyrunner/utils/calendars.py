from __future__ import annotations

import datetime as dt

import exchange_calendars as xcals


def session_close_utc(market: str, asof: str | None = None) -> dt.datetime:
    cal = xcals.get_calendar(market)
    date = dt.date.fromisoformat(asof) if asof else dt.date.today()
    sess = cal.session_close(date)
    return sess.tz_convert("UTC").to_pydatetime()


def is_trading_day(market: str, d: dt.date) -> bool:
    cal = xcals.get_calendar(market)
    return cal.is_session(d)
