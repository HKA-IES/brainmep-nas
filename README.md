# BrainMEP NAS - PUBLIC REPO

![Python](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11|3.12-blue.svg)

This repository contains the public code used for the NAS process for the
Brain-MEP Project.

**Note: this package is a work in progress**

## Requirements
- Linux distribution
  - If you are under Windows, consider using [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL)
- [task-spooler](https://github.com/justanhduc/task-spooler)
- [SQLite 3](https://sqlite.org/)

## Install it from the repository

### Python >= 3.10
```
pip install git+https://github.com/jonathanlarochelle/brainmep-nas
```

### Python >= 3.8, <3.10
The timescoring package only officially supports Python >= 3.10. However, we 
have found that timescoring 0.0.5 works fine with Python 3.8, 3.9 if dependency
requirements are ignored. 
```
pip install timescoring --ignore-requires --no-dependencies
pip install nptyping
pip install git+https://github.com/jonathanlarochelle/brainmep-nas
```

## Usage
The package *brainmepnas* consists in utility modules to faciliate the 
implementation of a NAS process for the Brain-MEP project. Example uses are
shown in examples/.

## Development
Please note that the goal of this package is not to offer a general framework 
for NAS. It was built around the needs of the Brain-MEP project and is designed
for this specific use-case.

Please do not hesitate to submit issues if you have found bugs or if the
documentation is unclear.

## Publication
This package was used and characterized in the publication "XYZ"

## License
This project is licensed under GNU General Public License v3.0.