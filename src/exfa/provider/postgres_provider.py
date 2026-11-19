from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from itertools import repeat
from types import TracebackType
from typing import NamedTuple, Self, cast, overload

import psycopg
from dotenv import dotenv_values, find_dotenv
from psycopg import sql
from psycopg.abc import Params, QueryNoTemplate
lazy import pandas as pd
lazy import polars as pl
lazy import pyarrow as pa

from exfa.utils import log_method, logger

from .adapters import Adapter, PandasAdapter


@dataclass(frozen=True, slots=True)
class PostgresConfig:
    database: str | None = None
    user: str | None = None
    password: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> PostgresConfig:
        settings = dotenv_values(find_dotenv(usecwd=True))
        return cls(
            database=os.getenv("POSTGRES_DATABASE", settings.get("POSTGRES_DATABASE")),
            user=os.getenv("POSTGRES_USER", settings.get("POSTGRES_USER")),
            password=os.getenv("POSTGRES_PASSWORD", settings.get("POSTGRES_PASSWORD")),
        )


class InsertResult(NamedTuple):
    class RejectedRow(NamedTuple):
        position: int
        reason: str

    inserted: int
    rejected_rows: list[RejectedRow]


class PostgresProvider[T = pd.DataFrame]:
    @overload
    def __init__(
        self: PostgresProvider[pd.DataFrame],
        config: PostgresConfig | None = None,
        adapter: None = None,
        verbose: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        config: PostgresConfig | None = None,
        *,
        adapter: Adapter[T],
        verbose: bool = False,
    ) -> None: ...

    def __init__(
        self,
        config: PostgresConfig | None = None,
        adapter: Adapter[T] | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config if config is not None else PostgresConfig.from_env()
        self._connection: psycopg.Connection | None = None
        self._adapter = adapter if adapter is not None else cast(Adapter[T], PandasAdapter())
        self._verbose = verbose

    def __enter__(self) -> PostgresProvider[T]:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exc_type, exc_value, traceback)

    @property
    def connection(self) -> psycopg.Connection:
        if self._connection is None:
            raise RuntimeError("Use the provider as a context manager or call open().")
        return self._connection

    @property
    def verbose(self) -> bool:
        return self._verbose

    @log_method
    def open(self) -> Self:
        if self._connection is None:
            self._connection = psycopg.connect(
                dbname=self._config.database,
                user=self._config.user,
                password=self._config.password,
            )
        return self

    @log_method
    def close(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @staticmethod
    def _check_table_exists(cur: psycopg.Cursor, schema: str, table: str) -> None:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
            AND table_type = 'BASE TABLE'
            """,
            (schema, table),
        )

        if cur.fetchone() is None:
            raise ValueError(f"Table {schema}.{table} not found")

    @staticmethod
    def _check_columns_match(
        cur: psycopg.Cursor,
        schema: str,
        table: str,
        columns: list[str],
        *,
        drop_unknown_columns: bool,
    ) -> list[str]:
        cur.execute(
            """
        SELECT
            column_name,
            is_nullable,
            column_default,
            is_identity,
            is_generated,
            identity_generation
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        """,
            (schema, table),
        )

        metadata = {
            name: (nullable, default, identity, generated, identity_mode)
            for name, nullable, default, identity, generated, identity_mode in cur.fetchall()
        }

        unknown = set(columns) - metadata.keys()

        if unknown and not drop_unknown_columns:
            raise ValueError(f"Unknown columns: {sorted(unknown)}")
        elif unknown:
            logger.warning(f"Dropped the following columns: {unknown}")

        columns = [name for name in columns if name in metadata]

        missing = [
            name
            for name, (nullable, default, identity, generated, _) in metadata.items()
            if name not in columns
            and nullable == "NO"
            and default is None
            and identity != "YES"
            and generated != "ALWAYS"
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        readonly = [
            name
            for name in columns
            if metadata[name][3] == "ALWAYS" or metadata[name][4] == "ALWAYS"
        ]

        if readonly:
            raise ValueError(f"Cannot insert generated columns: {readonly}")

        return columns

    @log_method
    def insert_dataframe_into_table(
        self,
        schema: str = "public",
        *,
        table: str,
        df: pd.DataFrame | pl.DataFrame,
        drop_unknown_columns: bool = True,
        conflict_columns: str | Sequence[str] = (),
        preserve_null_columns: str | Sequence[str] = (),
    ) -> InsertResult:
        """
        Write dataframe rows and report rejected rows by zero-based position.
        """

        inserted = 0
        rejected: list[InsertResult.RejectedRow] = []
        keys = (conflict_columns,) if isinstance(conflict_columns, str) else tuple(conflict_columns)
        preserve_nulls = (
            {preserve_null_columns}
            if isinstance(preserve_null_columns, str)
            else set(preserve_null_columns)
        )
        conn = self.connection
        with conn.transaction(), conn.cursor() as cur:
            self._check_table_exists(cur, schema, table)
            columns = self._check_columns_match(
                cur, schema, table, list(df.columns), drop_unknown_columns=drop_unknown_columns
            )
            missing_keys = set(keys) - set(columns)
            if missing_keys:
                raise ValueError(
                    f"Conflict columns must be present in the insert: {sorted(missing_keys)}"
                )

            target: sql.Composable = sql.Identifier(schema, table)
            if keys:
                target = sql.SQL("{} AS existing_row").format(target)
            if columns:
                query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    target,
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                )
                if isinstance(df, pd.DataFrame):
                    selected = df.loc[:, columns]
                    normalized = selected.astype(object).where(selected.notna(), None)
                    rows = normalized.itertuples(index=False, name=None)
                else:
                    rows = df.select(columns).iter_rows()
            else:
                query = sql.SQL("INSERT INTO {} DEFAULT VALUES").format(target)
                rows = repeat((), len(df))

            if keys:
                query += sql.SQL(" ON CONFLICT ({})").format(
                    sql.SQL(", ").join(map(sql.Identifier, keys))
                )
                assignments = []
                for column in columns:
                    if column in keys:
                        continue
                    identifier = sql.Identifier(column)
                    incoming = sql.SQL("EXCLUDED.{}").format(identifier)
                    if column in preserve_nulls:
                        incoming = sql.SQL("COALESCE({}, existing_row.{})").format(
                            incoming, identifier
                        )
                    assignments.append(sql.SQL("{} = {}").format(identifier, incoming))
                if assignments:
                    query += sql.SQL(" DO UPDATE SET {}").format(sql.SQL(", ").join(assignments))
                else:
                    query += sql.SQL(" DO NOTHING")

            for position, row in enumerate(rows):
                try:
                    with conn.transaction():
                        cur.execute(query, row)
                except (psycopg.DataError, psycopg.IntegrityError) as exc:
                    reason = exc.diag.message_primary or str(exc)
                    rejected.append(InsertResult.RejectedRow(position, reason))
                else:
                    inserted += cur.rowcount

        return InsertResult(inserted, rejected)

    def read_query(
        self, query: QueryNoTemplate, params: Params | None = None, *, index: Sequence[str] = ()
    ) -> T:
        """
        Read a row-returning query through PyArrow and the configured dataframe adapter.
        """

        connection = self.connection
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description is None:
                raise ValueError("The query must return columns")
            description = cursor.description
            names = [column.name for column in description]
            if len(set(names)) != len(names):
                raise ValueError("The query must return unique column names")
            rows = cursor.fetchall()

        arrow_types = {
            "text": pa.string(),
            "varchar": pa.string(),
            "date": pa.date32(),
            "int2": pa.int16(),
            "int4": pa.int32(),
            "int8": pa.int64(),
            "float4": pa.float32(),
            "float8": pa.float64(),
            "bool": pa.bool_(),
        }
        arrays = []
        columns = list(zip(*rows, strict=True)) if rows else [()] * len(names)
        for column, values in zip(description, columns, strict=True):
            pg_type = connection.adapters.types.get(column.type_code)
            arrow_type = arrow_types.get(pg_type.name) if pg_type is not None else None
            arrays.append(pa.array(values, type=arrow_type))
        table = pa.Table.from_arrays(arrays, names=names)
        return self._adapter.shape_arrow_data(table, index=index)

    @log_method
    def get_equity_classifications(self, assets: str | Sequence[str] | None = None) -> T:
        """
        Read current equity classifications for the selected assets.
        """

        names = [assets] if isinstance(assets, str) else assets
        params = {"assets": [name.strip() for name in names] if names is not None else None}
        query = """
            SELECT *
            FROM lseg.equity_classification
            WHERE (%(assets)s::text[] IS NULL OR asset = ANY(%(assets)s))
            ORDER BY asset
        """
        return self.read_query(query, params)

    @log_method
    def get_value_fundamentals(
        self,
        assets: str | Sequence[str] | None = None,
        *,
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> T:
        """
        Read historical fundamentals, including shares outstanding, for an inclusive date range.
        """

        names = [assets] if isinstance(assets, str) else assets
        params = {
            "assets": [name.strip() for name in names] if names is not None else None,
            "start_date": start_date,
            "end_date": end_date,
        }
        query = """
            SELECT *
            FROM lseg.value_fundamentals
            WHERE (%(assets)s::text[] IS NULL OR asset = ANY(%(assets)s))
              AND (%(start_date)s::date IS NULL OR date >= %(start_date)s)
              AND (%(end_date)s::date IS NULL OR date <= %(end_date)s)
            ORDER BY date, asset
        """
        return self.read_query(query, params, index=("date", "asset"))

    @log_method
    def get_reference_prices(
        self,
        assets: str | Sequence[str] | None = None,
        *,
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> T:
        """
        Read historical market prices for the selected assets and inclusive date range.
        """

        names = [assets] if isinstance(assets, str) else assets
        params = {
            "assets": [name.strip() for name in names] if names is not None else None,
            "start_date": start_date,
            "end_date": end_date,
        }
        query = """
            SELECT *
            FROM lseg.ref_prices
            WHERE (%(assets)s::text[] IS NULL OR asset = ANY(%(assets)s))
              AND (%(start_date)s::date IS NULL OR date >= %(start_date)s)
              AND (%(end_date)s::date IS NULL OR date <= %(end_date)s)
            ORDER BY date, asset
        """
        return self.read_query(query, params, index=("date", "asset"))
