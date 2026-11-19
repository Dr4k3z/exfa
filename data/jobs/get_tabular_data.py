from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider
from exfa.provider.postgres_provider import PostgresProvider

UNIVERSE_PATH = Path("data/EUROSTOXX.csv")


def get_assets(path: Path = UNIVERSE_PATH) -> list[str]:
    universe = pd.read_csv(path)
    return universe["RIC"].dropna().astype(str).drop_duplicates().tolist()


def main() -> None:
    assets = get_assets()

    with LsegDataProvider() as lseg:
        classifications = lseg.get_trbc(assets)

    with PostgresProvider() as postgres:
        result = postgres.insert_dataframe_into_table(
            "lseg",
            table="equity_classification",
            df=classifications,
        )

    print(f"Saved {result.inserted:,} rows to lseg.equity_classification")
    if result.rejected_rows:
        raise RuntimeError(f"Rejected classification rows: {result.rejected_rows[:5]}")


if __name__ == "__main__":
    main()
