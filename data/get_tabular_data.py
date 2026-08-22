from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider

UNIVERSE_PATH = Path("data/EUROSTOXX.csv")
OUTPUT_PATH = Path("data/lseg/equity_classification.parquet")


def get_assets(path: Path = UNIVERSE_PATH) -> list[str]:
    universe = pd.read_csv(path)
    return universe["RIC"].dropna().astype(str).drop_duplicates().tolist()


def main() -> None:
    assets = get_assets()

    with LsegDataProvider() as lseg:
        classifications = lseg.get_trbc(assets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    classifications.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(classifications):,} classifications -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
