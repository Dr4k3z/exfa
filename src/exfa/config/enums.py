from enum import StrEnum


class Database(StrEnum):
    EXFA = "exfa"


class Schema(StrEnum):
    LSEG = "lseg"
    BLOOMBERG = "bloomberg"
