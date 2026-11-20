from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

lazy import pandas as pd
lazy import polars as pl
lazy import pyarrow as pa

# Decoded, normalized column values; no dataframe implementation is required.
type TableData = Mapping[str, Sequence[object]]


class Adapter[T](Protocol):
    def shape_raw_data(self, raw_data: TableData, index: Sequence[str] = ()) -> T: ...

    def shape_arrow_data(self, table: pa.Table, index: Sequence[str] = ()) -> T: ...


class PandasAdapter:
    def shape_raw_data(self, raw_data: TableData, index: Sequence[str] = ()) -> pd.DataFrame:
        """
        Build pandas output, using a MultiIndex for historical date/asset keys.
        """

        frame = pd.DataFrame(dict(raw_data))
        if "date" in index:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index(list(index)) if index else frame

    def shape_arrow_data(self, table: pa.Table, index: Sequence[str] = ()) -> pd.DataFrame:
        frame = table.to_pandas(types_mapper=pd.ArrowDtype)
        if "date" in index:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index(list(index)) if index else frame


class PolarsAdapter:
    def shape_raw_data(self, raw_data: TableData, index: Sequence[str] = ()) -> pl.DataFrame:
        """
        Build Polars output directly, retaining historical keys as ordinary columns.
        """

        frame = pl.DataFrame(dict(raw_data), strict=False, infer_schema_length=None)
        if "date" in index:
            frame = frame.with_columns(pl.col("date").cast(pl.Datetime("us")))
        if "asset" in index:
            frame = frame.with_columns(pl.col("asset").cast(pl.String))
        return frame

    def shape_arrow_data(self, table: pa.Table, index: Sequence[str] = ()) -> pl.DataFrame:
        frame = cast(pl.DataFrame, pl.from_arrow(table))
        if "date" in index:
            frame = frame.with_columns(pl.col("date").cast(pl.Datetime("us")))
        return frame
