from typing import cast, overload

import psycopg
lazy import pandas as pd

from .adapters import Adapter, PandasAdapter


class PostgresProvider[T = pd.DataFrame]:
    @overload
    def __init__(
        self: PostgresProvider[pd.DataFrame], verbose: bool = False, adapter: None = None
    ): ...

    @overload
    def __init__(self, verbose: bool = False, *, adapter: Adapter[T]) -> None: ...

    def __init__(self, verbose: bool = False, adapter: Adapter[T] | None = None) -> None:
        self._raw_connection = psycopg.raw_cursor
        self._adapter = adapter if adapter is not None else cast(Adapter[T], PandasAdapter)

    def get_equities(self, names: str | list[str] | None = None) -> T:
        """
        docstring
        """

        raw_data = {}
        return self._adapter.shape_raw_data(raw_data)

    def get_historical_prices(self, names: str | list[str] | None = None) -> T:
        """
        docstring
        """

        raw_data = {}
        return self._adapter.shape_raw_data(raw_data)
