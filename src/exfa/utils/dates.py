from datetime import date as Date
from datetime import timedelta as TimeDelta


def get_last_business_date() -> Date:
    last_business_day = Date.today() - TimeDelta(days=1)
    while last_business_day.weekday() >= 5:
        last_business_day -= TimeDelta(days=1)
    return last_business_day
