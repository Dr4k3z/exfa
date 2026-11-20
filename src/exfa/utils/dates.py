from __future__ import annotations

from datetime import date as Date
from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta

lazy import pandas_market_calendars as mcal

from exfa.config.enums import TradingCalendar


def get_last_business_date() -> Date:
    last_business_day = Date.today() - TimeDelta(days=1)
    while last_business_day.weekday() >= 5:
        last_business_day -= TimeDelta(days=1)
    return last_business_day


def get_last_trading_day(cal: str | mcal.MarketCalendar = TradingCalendar.EUREX) -> Date:
    market = mcal.get_calendar(cal) if isinstance(cal, str) else cal
    yesterday = DateTime.now(market.tz).date() - TimeDelta(days=1)
    sessions = get_trading_days(yesterday - TimeDelta(days=365), yesterday, market)
    if not sessions:
        raise ValueError(f"No trading sessions found for {market.name} in the past year")
    return sessions[-1]


def get_business_days(start_date: Date, end_date: Date) -> list[Date]:
    days = (
        start_date + TimeDelta(days=offset) for offset in range((end_date - start_date).days + 1)
    )
    return [day for day in days if day.weekday() < 5]


def get_trading_days(
    start_date: Date,
    end_date: Date,
    calendar: str | mcal.MarketCalendar = TradingCalendar.EUREX,
) -> list[Date]:
    market = mcal.get_calendar(calendar) if isinstance(calendar, str) else calendar
    return [session.date() for session in market.valid_days(start_date, end_date)]
