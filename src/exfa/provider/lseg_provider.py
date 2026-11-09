from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as Date
from enum import StrEnum
from typing import Any, Self, cast, overload

import lseg.data as ld
import pandas as pd
from lseg.data.content import fundamental_and_reference, historical_pricing
from lseg.data.session import Session

from .adapters import Adapter, PandasAdapter, TableData


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


class LsegDataProvider[T = pd.DataFrame]:
    """
    LSEG ingestion client with a stable time-series contract.

    All methods construct output through the selected adapter. Historical data
    has sorted, unique (date, asset) keys: a pandas MultiIndex by default, or
    ordinary columns with PolarsAdapter. SDK pandas builders are never accessed.
    """

    @overload
    def __init__(
        self: LsegDataProvider[pd.DataFrame],
        config: LsegConfig | None = None,
        *,
        adapter: None = None,
    ) -> None: ...

    @overload
    def __init__(self, config: LsegConfig | None = None, *, adapter: Adapter[T]) -> None: ...

    def __init__(
        self, config: LsegConfig | None = None, *, adapter: Adapter[T] | None = None
    ) -> None:
        self.config = config or LsegConfig.from_env()
        self._session: Session | None = None
        self._adapter = adapter if adapter is not None else cast(Adapter[T], PandasAdapter())

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
    ) -> T:
        """
        Fetch fundamental/reference fields with LSEG column titles and adapt the output.
        """

        return self._adapter.shape_raw_data(
            self._request_reference_data(assets, fields, parameters)
        )

    def _request_reference_data(
        self,
        assets: str | Sequence[str],
        fields: str | Sequence[str],
        parameters: str | dict[str, Any] | None = None,
    ) -> dict[str, list[object]]:
        if isinstance(parameters, str):
            key, separator, value = parameters.partition("=")
            if not separator or not key.strip():
                raise ValueError("String parameters must contain a key=value pair")
            parameters = {key.strip(): value.strip()}
        response = fundamental_and_reference.Definition(
            universe=_assets(assets),
            fields=[fields] if isinstance(fields, str) else list(fields),
            parameters=parameters,
        ).get_data(session=self.session)
        if not response.is_success or response.errors:
            raise LsegProviderError(f"LSEG reference request failed: {response.errors}")
        return _parse_raw_table(response.data.raw)

    def get_trbc(self, assets: str | Sequence[str]) -> T:
        """
        Return static country and TRBC classifications, one row per asset.
        """

        table = self._request_reference_data(
            assets,
            [
                "TR.HeadquartersCountry",
                "TR.TRBCEconomicSector",
                "TR.TRBCBusinessSector",
                "TR.TRBCIndustryGroup",
                "TR.TRBCIndustry",
                "TR.TRBCActivity",
            ],
        )
        table = _rename_columns(
            table,
            {"Instrument": "asset", "instrument": "asset", "RIC": "asset", "ric": "asset"},
            normalize=True,
        )
        if "asset" not in table:
            raise LsegSchemaError("Reference data must identify the asset")
        positions = sorted(
            range(len(table["asset"])), key=lambda row: str(table["asset"][row])
        )
        return self._adapter.shape_raw_data(
            {name: [values[row] for row in positions] for name, values in table.items()}
        )

    def get_prices(
        self,
        assets: str | Sequence[str],
        start_date: Date,
        end_date: Date,
        fields: Sequence[PriceField | str] = DEFAULT_PRICE_FIELDS,
    ) -> T:
        """
        Return historical pricing indexed by ``(date, asset)``.
        """

        _ = self.session
        universe = _assets(assets)
        _date_range(start_date, end_date)
        requested = list(dict.fromkeys(str(field) for field in fields))

        series: list[dict[str, list[object]]] = []
        for field in requested:
            response = historical_pricing.summaries.Definition(
                universe=universe,
                start=start_date,
                end=end_date,
                fields=[field],
            ).get_data(session=self.session)
            if not response.is_success or response.errors:
                raise LsegProviderError(f"LSEG pricing request failed: {response.errors}")
            series.append(_normalize_prices(response.data.raw, universe, field))

        if not series:
            raise ValueError("At least one price field is required.")
        return self._adapter.shape_raw_data(_join_timeseries(series), index=("date", "asset"))

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
    ) -> T:
        """
        Fetch one TR value/date pair as a raw observed time series.
        """

        table = self._request_timeseries_field(
            assets,
            field,
            start_date,
            end_date,
            name=name or _field_name(field),
            frequency=frequency,
            currency=currency,
        )
        return self._adapter.shape_raw_data(table, index=("date", "asset"))

    def _request_timeseries_field(
        self,
        assets: str | Sequence[str],
        field: str,
        start_date: Date,
        end_date: Date,
        *,
        name: str,
        frequency: str,
        currency: str | None,
    ) -> dict[str, list[object]]:
        _date_range(start_date, end_date)
        parameters: dict[str, Any] = {
            "SDate": start_date.isoformat(),
            "EDate": end_date.isoformat(),
            "Frq": frequency,
        }
        if currency is not None:
            parameters["Curn"] = currency
        return _normalize_tr_timeseries(
            self._request_reference_data(assets, [f"{field}.date", field], parameters), name
        )

    def get_timeseries_fields(
        self,
        assets: str | Sequence[str],
        fields: Sequence[str] | Mapping[str, str],
        start_date: Date,
        end_date: Date,
        *,
        frequency: str = "D",
        currency: str | None = None,
    ) -> T:
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
            self._request_timeseries_field(
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
        return self._adapter.shape_raw_data(_join_timeseries(series), index=("date", "asset"))

    def get_market_cap(
        self,
        assets: str | Sequence[str],
        start_date: Date,
        end_date: Date,
        currency: str | None = None,
    ) -> T:
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
    ) -> T:
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
    ) -> T:
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


def _field_name(field: str) -> str:
    return _snake_case(field.removeprefix("TR."))


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _parse_raw_table(raw_data: object) -> dict[str, list[object]]:
    """
    Decode the content API's RDP or UDF headers/data JSON into columns.
    """

    if not isinstance(raw_data, Mapping):
        raise LsegSchemaError("Expected an LSEG JSON object with headers and data")
    error = raw_data.get("error")
    if error:
        raise LsegProviderError(f"LSEG returned an error: {error}")
    headers = raw_data.get("headers")
    rows = raw_data.get("data")
    if not isinstance(headers, list) or not isinstance(rows, list):
        raise LsegSchemaError("LSEG response must contain headers and data lists")
    # Desktop/UDF responses wrap the single header row in an extra list.
    if len(headers) == 1 and isinstance(headers[0], list):
        headers = headers[0]
    names = []
    date_columns = []
    for header in headers:
        if not isinstance(header, Mapping):
            raise LsegSchemaError("LSEG headers must be JSON objects")
        name = header.get("title") or header.get("displayName") or header.get("name")
        if not isinstance(name, str) or not name:
            raise LsegSchemaError("LSEG headers must identify each column")
        names.append(name)
        identifier = str(header.get("name") or header.get("field") or name).lower()
        if header.get("type") in {"date", "datetime"} or identifier.endswith(".date"):
            date_columns.append(name)
    if len(set(names)) != len(names):
        raise LsegSchemaError("LSEG response contains duplicate column titles")
    table: dict[str, list[object]] = {name: [] for name in names}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(names):
            raise LsegSchemaError("LSEG data rows must match the header width")
        for name, value in zip(names, row, strict=True):
            table[name].append(_parse_date(value) if name in date_columns else value)
    return table


def _rename_columns(
    table: TableData, names: Mapping[str, str], *, normalize: bool = False
) -> dict[str, list[object]]:
    result = {}
    for name, values in table.items():
        output_name = names.get(name, name)
        if normalize:
            output_name = _snake_case(output_name)
        if output_name in result:
            raise LsegSchemaError(f"Duplicate output column: {output_name!r}")
        result[output_name] = list(values)
    return result


def _parse_date(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, Date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise LsegSchemaError(f"Invalid LSEG date: {value!r}") from error
    else:
        raise LsegSchemaError(f"Invalid LSEG date: {value!r}")
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed


def _normalize_tr_timeseries(table: TableData, name: str) -> dict[str, list[object]]:
    table = _rename_columns(table, {}, normalize=True)
    asset = next(
        (column for column in table if column in {"asset", "instrument", "ric"}), None
    )
    dates = [column for column in table if column == "date" or column.endswith("_date")]
    values = [column for column in table if column != asset and column not in dates]
    if asset is None or len(dates) != 1 or len(values) != 1:
        raise LsegSchemaError(f"Cannot identify date, asset and value in {list(table)}")
    if not name or name in {"date", "asset"}:
        raise LsegSchemaError("Time-series value names must not shadow date or asset")
    return _as_timeseries(
        {"date": table[dates[0]], "asset": table[asset], name: table[values[0]]}
    )


def _normalize_prices(
    raw_data: object, assets: Sequence[str], field: str
) -> dict[str, list[object]]:
    payloads = raw_data if isinstance(raw_data, list) else [raw_data]
    name = next(
        (member.name.lower() for member in PriceField if member.value == field),
        _field_name(field),
    )
    tables = []
    for payload in payloads:
        table = _parse_raw_table(payload)
        table = _rename_columns(table, {field: name}, normalize=True)
        if name not in table:
            raise LsegSchemaError(f"Pricing response does not contain field {field!r}")
        if "date_time" in table and "date" not in table:
            table["date"] = table.pop("date_time")
        if not isinstance(payload, Mapping):
            raise LsegSchemaError("Pricing response must identify its instrument")
        universe = payload.get("universe")
        asset = universe.get("ric") if isinstance(universe, Mapping) else universe
        if not isinstance(asset, str):
            asset = assets[0] if len(assets) == 1 else None
        if "asset" not in table:
            if asset is None:
                raise LsegSchemaError("Pricing response must identify its instrument")
            table["asset"] = [asset] * len(table[name])
        if "date" not in table:
            raise LsegSchemaError("Pricing response must identify its dates")
        tables.append({"date": table["date"], "asset": table["asset"], name: table[name]})
    combined: dict[str, list[object]] = {"date": [], "asset": [], name: []}
    for table in tables:
        for column in combined:
            combined[column].extend(table[column])
    return _as_timeseries(combined)


def _as_timeseries(table: TableData) -> dict[str, list[object]]:
    if not {"date", "asset"}.issubset(table):
        raise LsegSchemaError("Historical data must contain date and asset columns")
    value_columns = [column for column in table if column not in {"date", "asset"}]
    observations: dict[tuple[datetime, str], tuple[object, ...]] = {}
    for row in zip(*table.values(), strict=True):
        record = dict(zip(table, row, strict=True))
        date = _parse_date(record["date"])
        asset = record["asset"]
        if date is None or asset is None:
            continue
        if not isinstance(asset, str):
            raise LsegSchemaError("Asset identifiers must be strings")
        values = tuple(record[column] for column in value_columns)
        key = date, asset
        if key in observations and observations[key] != values:
            raise LsegSchemaError("Historical data contains duplicate (date, asset) rows")
        observations[key] = values
    result: dict[str, list[object]] = {
        "date": [],
        "asset": [],
        **{name: [] for name in value_columns},
    }
    for (date, asset), values in sorted(observations.items()):
        if all(value is None for value in values):
            continue
        result["date"].append(date)
        result["asset"].append(asset)
        for name, value in zip(value_columns, values, strict=True):
            result[name].append(value)
    return result


def _join_timeseries(frames: Sequence[TableData]) -> dict[str, list[object]]:
    columns: list[str] = []
    observations: dict[tuple[object, object], dict[str, object]] = {}
    for frame in frames:
        values = [column for column in frame if column not in {"date", "asset"}]
        if set(values).intersection(columns):
            raise LsegSchemaError("Time-series fields must have distinct output column names")
        columns.extend(values)
        for row in zip(*frame.values(), strict=True):
            record = dict(zip(frame, row, strict=True))
            key = record.pop("date"), record.pop("asset")
            observations.setdefault(key, {}).update(record)
    result: dict[str, list[object]] = {
        "date": [],
        "asset": [],
        **{name: [] for name in columns},
    }
    for (date, asset), record in observations.items():
        result["date"].append(date)
        result["asset"].append(asset)
        for column in columns:
            result[column].append(record.get(column))
    return _as_timeseries(result)
