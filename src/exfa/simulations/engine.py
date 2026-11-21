from collections.abc import Callable, Iterator, Mapping

import numpy as np

from .models import Model

type Measurement = Mapping[str, float | np.ndarray]


class MCEngine:
    def __init__(
        self,
        model: Model,
        num_trials: int,
        seed: int | None = 42,
    ) -> None:
        if num_trials < 1:
            raise ValueError("num_trials must be positive")

        self.model = model
        self.num_trials = num_trials
        self._seed = np.random.SeedSequence(seed)

    def samples(self, T: int) -> Iterator[np.ndarray]:
        if T < 1:
            raise ValueError("T must be positive")

        for trial in range(self.num_trials):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    entropy=self._seed.entropy,
                    spawn_key=(T, trial),
                )
            )

            yield self.model.sample(T, rng)

    def run(
        self,
        T: int,
        measure: Callable[[np.ndarray], Measurement],
    ) -> dict[str, np.ndarray]:
        results: dict[str, list[np.ndarray]] = {}

        for trial, R in enumerate(self.samples(T)):
            values = measure(R)

            if trial == 0:
                if not values:
                    raise ValueError("Measurement must return results")

                results = {name: [] for name in values}

            if values.keys() != results.keys():
                raise ValueError("Inconsistent measurement keys")

            for name, value in values.items():
                results[name].append(np.array(value, copy=True))

        return {name: np.stack(values, axis=0) for name, values in results.items()}
