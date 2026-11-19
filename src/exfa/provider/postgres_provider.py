from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import TracebackType
from typing import NamedTuple, cast, overload

import psycopg
from dotenv import dotenv_values, find_dotenv
from psycopg import sql
lazy import pandas as pd
lazy import polars as pl

from exfa.utils import log_method

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


class RejectedRow(NamedTuple):
    position: int
    reason: str


class InsertResult(NamedTuple):
    inserted: int
    rejected_rows: list[RejectedRow]


class PostgresProvider[T = pd.DataFrame]:
    @overload
    def __init__(
        self: PostgresProvider[pd.DataFrame],
        verbose: bool = False,
        adapter: None = None,
        config: PostgresConfig | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        verbose: bool = False,
        *,
        adapter: Adapter[T],
        config: PostgresConfig | None = None,
    ) -> None: ...

    def __init__(
        self,
        verbose: bool = False,
        adapter: Adapter[T] | None = None,
        config: PostgresConfig | None = None,
    ) -> None:
        config = config if config is not None else PostgresConfig.from_env()
        self._connection = psycopg.connect(
            dbname=config.database,
            user=config.user,
            password=config.password,
        )
        self._adapter = adapter if adapter is not None else cast(Adapter[T], PandasAdapter)

    def __enter__(self) -> PostgresProvider[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._connection.close()

    @staticmethod
    def _check_table_exists(conn: psycopg.Connection, schema: str, table: str) -> None:
        with conn.transaction():
            with conn.cursor() as cur:
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
        conn: psycopg.Connection, schema: str, table: str, columns: list[str]
    ) -> None:
        with conn.transaction():
            with conn.cursor() as cur:
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

                if unknown:
                    raise ValueError(f"Unknown columns: {sorted(unknown)}")

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

    def insert_dataframe_into_table(
        self,
        schema: str = "public",
        *,
        table: str,
        df: pd.DataFrame | pl.DataFrame,
    ) -> InsertResult:
        """
        Write dataframe rows and report rejected rows by zero-based position.
        """

        if isinstance(df, pd.DataFrame):
            columns = list(df.columns)
            normalized = df.astype(object).where(df.notna(), None)
            rows = normalized.itertuples(index=False, name=None)
        else:
            columns = df.columns
            rows = df.iter_rows()

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(schema, table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )

        inserted = 0
        rejected: list[RejectedRow] = []
        conn = self._connection
        with conn.transaction():
            self._check_table_exists(conn, schema, table)
            self._check_columns_match(conn, schema, table, columns)
            with conn.cursor() as cur:
                for position, row in enumerate(rows):
                    try:
                        with conn.transaction():
                            cur.execute(query, row)
                    except (psycopg.DataError, psycopg.IntegrityError) as exc:
                        reason = exc.diag.message_primary or str(exc)
                        rejected.append(RejectedRow(position, reason))
                    else:
                        inserted += cur.rowcount

        return InsertResult(inserted, rejected)

    @log_method
    def get_equities(self, names: str | list[str] | None = None) -> T:
        """
        docstring
        """

        raw_data = {}
        return self._adapter.shape_raw_data(raw_data)

    @log_method
    def get_historical_prices(self, names: str | list[str] | None = None) -> T:
        """
        docstring
        """

        raw_data = {}
        return self._adapter.shape_raw_data(raw_data)
