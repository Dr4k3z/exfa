from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from enum import StrEnum
from typing import Any, Self

import lseg.data as ld
import pandas as pd
from lseg.data.content import historical_pricing
from lseg.data.session import Session

from .adapters import Adapter


class LsegProviderError(RuntimeError):
    """
    Base error raised by the LSEG ingestion client.
    """


class LsegSessionError(LsegProviderError):
    """
    Raised when an extraction is attempted without an open session.
    """


class LsegSchemaError(LsegProviderError):
    """
    Raised when LSEG returns data that violates the provider contract.
    """


class PriceField(StrEnum):
    CLOSE = "TRDPRC_1"
    HIGH = "HIGH_1"
    LOW = "LOW_1"
    BID = "BID"
    ASK = "ASK"
    VWAP = "VWAP"
    VOLUME = "ACVOL_UNS"
    TURNOVER = "TRNOVR_UNS"


class YieldField(StrEnum):
    YIELD = "YIELD"
    OPEN = "OPEN_YLD"
    HIGH = "HIGH_YLD"
    LOW = "LOW_YLD"


DEFAULT_PRICE_FIELDS: tuple[PriceField, ...] = (
    PriceField.CLOSE,
    PriceField.HIGH,
    PriceField.LOW,
    PriceField.BID,
    PriceField.ASK,
    PriceField.VWAP,
    PriceField.VOLUME,
)


@dataclass(frozen=True, slots=True)
class LsegConfig:
    app_key: str | None = None
    session_name: str | None = None
    config_name: str | None = None

    @classmethod
    def from_env(cls) -> LsegConfig:
        return cls(
            app_key=os.getenv("LSEG_APP_KEY"),
            session_name=os.getenv("LSEG_SESSION_NAME"),
            config_name=os.getenv("LSEG_CONFIG_NAME"),
        )


class LsegDataProvider[T]:
    """
    LSEG ingestion client with a stable time-series contract.

    Historical methods return a sorted, unique ``(date, asset)`` MultiIndex.
    Static/reference methods return ordinary tabular dataframes.
    """

    def __init__(self, config: LsegConfig | None = None, adapter: Adapter[T] | None = None) -> None:
        self.config = config or LsegConfig.from_env()
        self._session: Session | None = None
        self._adapter = adapter

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> Session:
        if self._session is None:
            raise LsegSessionError("Use the provider as a context manager or call open().")
        return self._session

    def open(self) -> Self:
        if self._session is None:
            self._session = ld.open_session(
                name=self.config.session_name,
                app_key=self.config.app_key,
                config_name=self.config.config_name,
            )
        return self

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def get_data(
        self,
        assets: str | Sequence[str],
        fields: str | Sequence[str],
        parameters: str | dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Call LSEG's generic endpoint without changing its output.
        """

        _ = self.session
        requested = [fields] if isinstance(fields, str) else list(fields)
        return ld.get_data(universe=_assets(assets), fields=requested, parameters=parameters)

    def get_trbc(self, assets: str | Sequence[str]) -> pd.DataFrame:
        """
        Return static country and TRBC classifications, one row per asset.
        """

        frame = self.get_data(
            assets,
            [
                "TR.HeadquartersCountry",
                "TR.TRBCEconomicSector",
                "TR.TRBCBusinessSector",
                "TR.TRBCIndustryGroup",
                "TR.TRBCIndustry",
                "TR.TRBCActivity",
            ],
        ).rename(columns={"Instrument": "asset"})
        frame.columns = [_snake_case(str(column)) for column in frame.columns]
        return frame.sort_values("asset", ignore_index=True)

    def get_prices(
        self,
        assets: str | Sequence[str],
        start_date: Date,
        end_date: Date,
        fields: Sequence[PriceField | str] = DEFAULT_PRICE_FIELDS,
    ) -> pd.DataFrame:
        """
        Return historical pricing indexed by ``(date, asset)``.
        """

        _ = self.session
        universe = _assets(assets)
        _date_range(start_date, end_date)
        requested = list(dict.fromkeys(str(field) for field in fields))

        series: list[pd.DataFrame] = []
        for field in requested:
            response = historical_pricing.summaries.Definition(
                universe=universe,
                start=start_date,
                end=end_date,
                fields=[field],
            ).get_data()
            series.append(_normalize_price(response.data.df, universe, field))

        if not series:
            raise ValueError("At least one price field is required.")
        return _join_timeseries(series)

    def get_timeseries_field(
        self,
        assets: str | Sequence[str],
        field: str,
        start_date: Date,
        end_date: Date,
        *,
        name: str | None = None,
        frequency: str = "D",
        currency: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch one TR value/date pair as a raw observed time series.
        """

        _date_range(start_date, end_date)
        parameters: dict[str, Any] = {
            "SDate": start_date.isoformat(),
            "EDate": end_date.isoformat(),
            "Frq": frequency,
        }
        if currency is not None:
            parameters["Curn"] = currency

        frame = self.get_data(
            assets,
            fields=[f"{field}.date", field],
            parameters=parameters,
        )
        return _normalize_tr_timeseries(frame, name or _field_name(field))

    def get_timeseries_fields(
        self,
        assets: str | Sequence[str],
        fields: Sequence[str] | Mapping[str, str],
        start_date: Date,
        end_date: Date,
        *,
        frequency: str = "D",
        currency: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fields independently and outer-join their observation dates.

        A mapping supplies ``LSEG field -> output column`` names. A sequence
        derives output names from the LSEG field identifiers.
        """

        items: list[tuple[str, str]]
        if isinstance(fields, Mapping):
            items = [(str(field), str(name)) for field, name in fields.items()]
        else:
            items = [(field, _field_name(field)) for field in fields]
        series = [
            self.get_timeseries_field(
                assets,
                field,
                start_date,
                end_date,
                name=name,
                frequency=frequency,
                currency=currency,
            )
            for field, name in items
        ]
        if not series:
            raise ValueError("At least one time-series field is required.")
        return _join_timeseries(series)

    def get_market_cap(
        self,
        assets: str | Sequence[str],
        start_date: Date,
        end_date: Date,
        currency: str | None = None,
    ) -> pd.DataFrame:
        return self.get_timeseries_field(
            assets,
            "TR.CompanyMarketCap",
            start_date,
            end_date,
            name="market_cap",
            currency=currency,
        )

    def get_shares_outstanding(
        self, assets: str | Sequence[str], start_date: Date, end_date: Date
    ) -> pd.DataFrame:
        return self.get_timeseries_field(
            assets,
            "TR.SharesOutstanding",
            start_date,
            end_date,
            name="shares_outstanding",
        )

    def get_value_fundamentals(
        self,
        assets: str | Sequence[str],
        start_date: Date,
        end_date: Date,
        currency: str | None = None,
    ) -> pd.DataFrame:
        return self.get_timeseries_fields(
            assets,
            {
                "TR.CompanyMarketCap": "market_cap",
                "TR.Revenue": "revenue",
                "TR.TotalEquity": "book_value",
                "TR.CashFromOperatingActivities": "operating_cash_flow",
            },
            start_date,
            end_date,
            currency=currency,
        )


def _assets(assets: str | Sequence[str]) -> list[str]:
    return [assets] if isinstance(assets, str) else list(dict.fromkeys(assets))


def _date_range(start: Date, end: Date) -> None:
    if start > end:
        raise ValueError(f"start ({start}) must be on or before end ({end}).")


def _normalize_price(frame: pd.DataFrame, assets: Sequence[str], field: str) -> pd.DataFrame:
    if set(frame.columns).issubset(assets):
        frame = (
            frame.rename_axis("date")
            .rename_axis("asset", axis="columns")
            .stack(future_stack=True)
            .rename(field)
            .reset_index()
        )
    elif isinstance(frame.columns, pd.MultiIndex):
        frame = frame.stack(level=0, future_stack=True).reset_index()
    else:
        frame = frame.reset_index()
        frame = frame.rename(columns={"Instrument": "asset"})
        if "asset" not in frame and len(assets) == 1:
            frame.insert(1, "asset", assets[0])

    names = {member.value: member.name.lower() for member in PriceField}
    frame = frame.rename(columns=names)
    frame.columns = [_snake_case(str(column)) for column in frame.columns]
    return _as_timeseries(frame)


def _normalize_tr_timeseries(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_snake_case(str(column)) for column in frame.columns]
    asset = next((column for column in frame if column in {"asset", "instrument", "ric"}), None)
    dates = [column for column in frame if column == "date" or column.endswith("_date")]
    values = [column for column in frame if column != asset and column not in dates]
    if asset is None or not dates or len(values) != 1:
        raise LsegSchemaError(f"Cannot identify date, asset and value in {list(frame.columns)}")
    return _as_timeseries(
        frame.rename(columns={asset: "asset", dates[0]: "date", values[0]: name})[
            ["date", "asset", name]
        ]
    )


def _as_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "asset"}.issubset(frame.columns):
        raise LsegSchemaError("Historical data must contain date and asset columns.")
    frame = frame.dropna(subset=["date", "asset"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    # LSEG can repeat an unchanged fundamental on every requested frequency
    # date while reporting the same associated field date. Exact repeats carry
    # no additional information; conflicting duplicates remain an error below.
    frame = frame.drop_duplicates()
    frame = frame.set_index(["date", "asset"]).sort_index()
    if not frame.index.is_unique:
        raise LsegSchemaError("Historical data contains duplicate (date, asset) rows.")
    return frame.dropna(how="all")


def _join_timeseries(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, how="outer", validate="one_to_one")
    return result.sort_index().dropna(how="all")


def _field_name(field: str) -> str:
    return _snake_case(field.removeprefix("TR."))


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
