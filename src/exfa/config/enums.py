from enum import StrEnum


class Database(StrEnum):
    EXFA = "exfa"


class Schema(StrEnum):
    LSEG = "lseg"
    BLOOMBERG = "bloomberg"


class TradingCalendar(StrEnum):
    EUREX = "EUREX"
    XETRA = "XETR"
    PARIS = "XPAR"
    AMSTERDAM = "XAMS"
    BRUSSELS = "XBRU"
    MILAN = "XMIL"
    MADRID = "XMAD"
    LONDON = "LSE"
    SIX = "SIX"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CME_EQUITY = "CME_Equity"
    TOKYO = "JPX"
    HONG_KONG = "HKEX"
