import pandas as pd
import psycopg


class PostgresProvider:
    def __init__(self, verbose: bool = False) -> None:
        self._raw_connection = psycopg.raw_cursor

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
