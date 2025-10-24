from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

ScenarioType = Literal[
    "random_anomalies",
    "clustered_with_random_assignment",
    "clustered_with_closest_assignment",
]


@dataclass
class ScenarioConfig:
    scenario_type: ScenarioType
    concepts_no: int
    size_per_concept: int

    def save_as_md(self, path: Path) -> None:
        config_dict = {
            k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()
        }

        md_lines = ["# Scenario Config\n"]
        for key, value in config_dict.items():
            md_lines.append(f"- **{key}**: {value}")
        md_lines.append("\n")

        path.write_text("\n".join(md_lines))
