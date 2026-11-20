from dataclasses import dataclass

from .enums import Database, Schema


@dataclass(frozen=True, slots=True)
class PostgresTable:
    database: Database
    schema: str
    base: str

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.base}"


EquityClassification_Table = PostgresTable(
    database=Database.EXFA, schema=Schema.LSEG, base="equity_classification"
)
RefPrices_Table = PostgresTable(database=Database.EXFA, schema=Schema.LSEG, base="ref_prices")
ValueFundamentals_Table = PostgresTable(
    database=Database.EXFA, schema=Schema.LSEG, base="value_fundamentals"
)
