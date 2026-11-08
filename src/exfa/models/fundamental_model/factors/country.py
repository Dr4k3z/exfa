from .base import CategoricalFactor


class CountryFactor(CategoricalFactor):
    def __init__(self, column: str = "country") -> None:
        super().__init__("country", column)
