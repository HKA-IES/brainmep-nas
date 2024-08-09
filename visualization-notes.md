# Notes for the visualization and analysis scripts
This document contains notes about the desired visualization for model study 
results, as well as their actual implementation.

## Requirements
- Convergence analysis
  - Hypervolume (from normalized objectives) vs time or trials
    - Each outer fold as a curve (F curves)
    - Mean + std dev of each outer fold (1 curve + error)
  - Other convergence metrics?
- Pareto Front
  - Inner loops front
    - Each outer fold as a curve (F curves)
    - Mean + std dev of each outer fold (1 curve + error)
  - Outer loop front
    - One curve
  - Inner loops + outer loop in one
    - Pssimistic hypervolume
    - optimistic hypervolume
    - difference between both
- Parameters importance
  - Optuna implementation
    - Mean + std dev of each outer fold
  - Shapley value (?)
- Parameters distribution
  - Simple distribution
  - Distribution w.r.t. to objectives (2D heatmap, obj on one axis, parameter 
  values on other axis, identify pareto set)

Would be nice to be able to use it both from the command-line and from 
a script, also yielding the fig object -> would enable combining and 
customizing different visualization from different studies.

## Implementation
Generate all visualizations as png and svg (stored in 
modelstudy/visualization/) at once:
```
python mymodelstudy.py visualize
```

Alternatively, the visualization module contains all relevant functions 
and can be called with a specific model:
```
from brainmepnas.visualization import plot_hypervolume
from mymodelstudy import MyModelStudy

fig = plot_hypervolume(MyModelStudy)
```