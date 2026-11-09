from typing import Any, Protocol

import pandas as pd
import polars as pl


class Adapter[T](Protocol):
    def shape_raw_data(raw_data: Any) -> T: ...


class PandasAdapter:
    def __init__(self) -> None:
        pass

    def shape_raw_data(raw_data: Any) -> pd.DataFrame:
        return pd.DataFrame()


class PolarsAdapter:
    def __init__(self) -> None:
        pass

    def shape_raw_data(raw_data: Any) -> pl.DataFrame:
        return pl.DataFrame()
