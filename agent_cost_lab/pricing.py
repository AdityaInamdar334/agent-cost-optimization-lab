"""Provider-agnostic accounting for explicitly partitioned token usage.

All prices are scenario assumptions, NOT current provider prices.
The four token buckets are mutually exclusive; caller must normalize the
provider's actual usage schema before passing them here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    regular_input: float = 3.0   # USD per 1M
    cache_write: float = 3.75     # 1.25 x regular, illustrative
    cache_read: float = 0.30     # 0.1 x regular, illustrative
    output: float = 15.0

    def __post_init__(self):
        if min(self.regular_input, self.cache_write,
               self.cache_read, self.output) < 0:
            raise ValueError("Rates must be nonnegative")


@dataclass(frozen=True)
class Usage:
    regular_input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    def __post_init__(self):
        values = (self.regular_input, self.cache_write,
                  self.cache_read, self.output)
        if not all(isinstance(x, int) and not isinstance(x, bool)
                   and x >= 0 for x in values):
            raise ValueError("Token counts must be nonnegative integers")


def price(usage: Usage, rates: Rates = Rates()) -> float:
    return (usage.regular_input * rates.regular_input
            + usage.cache_write * rates.cache_write
            + usage.cache_read * rates.cache_read
            + usage.output * rates.output) / 1_000_000
