from collections.abc import Mapping, Sequence
from types import MappingProxyType

DUAL_LISTINGS_EU: Mapping[str, str] = MappingProxyType(
    {
        "CRIG.DE": "CRDI.MI",  # UniCredit
        "NDASE.ST": "NDAFI.HE",  # Nordea
        "NDADK.CO": "NDAFI.HE",
        "AIR.PA": "AIR.PA1",  # Airbus
        "AIR.DE": "AIR.PA1",
        "AIR.MC": "AIR.PA1",
        "ASML.O": "ASML.AS",  # ASML
        "RACE.N": "RACE.MI",  # Ferrari
        "PRXJn.J": "PRX.AS",  # Prosus
        "TTE.N": "TTEF.PA",  # TotalEnergies
    }
)


def normalize_lseg_assets(assets: str | Sequence[str]) -> list[str]:
    """
    Resolve listing aliases and remove duplicate RICs in input order.
    """

    names = [assets] if isinstance(assets, str) else assets
    stripped = (name.strip() for name in names)
    return list(dict.fromkeys(DUAL_LISTINGS_EU.get(name, name) for name in stripped))
