from .base import CategoricalFactor


class SectorFactor(CategoricalFactor):
    def __init__(self, sector_col: str = "sector") -> None:
        super().__init__("sector", sector_col)
