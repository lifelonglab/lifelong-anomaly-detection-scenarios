from dataclasses import dataclass
from typing import Literal

ScenarioType = Literal['random_anomalies', 'clustered_with_random_assignment', 'clustered_with_closest_assignment']


@dataclass
class ScenarioConfig:
    scenario_type: ScenarioType
    concepts_no: int
    size_per_concept: int
