import pandas as pd
import psycopg

from .adapters import Adapter


class PostgresProvider[T]:
    def __init__(self, verbose: bool = False, adapter: Adapter[T] | None = None) -> None:
        self._raw_connection = psycopg.raw_cursor
        self._adapter = adapter

    def get_equities(self, names: str | list[str] | None = None) -> pd.DataFrame:
        """
        docstring
        """

        return pd.DataFrame()

    def get_historical_prices(self, names: str | list[str] | None = None) -> pd.DataFrame:
        """
        docstring
        """

        return pd.DataFrame()
