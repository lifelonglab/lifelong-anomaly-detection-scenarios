# Lifelong Learning for Anomaly Detection: New Challenges, Perspectives, and Insights

## Introduction
One of the contributions included in our paper was a scenario design procedure that applies to most datasets and aims to enable researchers and practitioners to transition their current scenarios and evaluation setup towards lifelong anomaly detection.
More details, along with description of lifelong anomaly detection, are available here (Open Access): <https://ieeexplore.ieee.org/document/10473036> 

## How to use the algorithm?

To generate the scenarios, you can use `prepare_scenario` function available in `prepare_scenario` module. The example application is available in `main.py`.

The function takes three arguments:
- `normal_data` - array containing normal data
  - `anomaly_data` - array containing anomalous data
  - `config` - a config according to which the scenario is created

There are three things to configure in `ScenarioConfig`:
1. `scenario_type` - a type of the scenario 
   1. `random_anomalies`
   1. `clustered_with_random_assignment`
   1. `clustered_with_closest_assignment`
1. `concepts_no` - number of concepts to create
1. `size_per_concept` - size (in terms of data samples) of a single concept

## Paper & Citation
The details of the algorithm are described in the paper (open access): <https://ieeexplore.ieee.org/document/10473036>

Whenever using the algorithm or generated scenarios, please cite:

K. Faber, R. Corizzo, B. Sniezynski and N. Japkowicz, "Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights," in IEEE Access, vol. 12, pp. 41364-41380, 2024, doi: 10.1109/ACCESS.2024.3377690
```
@ARTICLE{10473036,
  author={Faber, Kamil and Corizzo, Roberto and Sniezynski, Bartlomiej and Japkowicz, Nathalie},
  journal={IEEE Access}, 
  title={Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights}, 
  year={2024},
  volume={12},
  number={},
  pages={41364-41380},
  doi={10.1109/ACCESS.2024.3377690}}
```