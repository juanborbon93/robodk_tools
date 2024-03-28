# RoboDK Tools

A collection of python programs for RoboDK

## singularity analysis

Gets the robot joint trajectories from robodk and evaluates each step to calculate the manipulability using the jacobian matrix.

```
usage: singularity_analysis.py [-h] [--program-name PROGRAM_NAME]

This module provides functions to evaluate the manipulability of a RoboDK program.
It calculates the manipulability of each move instruction in the program and plots the results.
If the robot approaches a joint singularity, the manipulability score will approach zero.

options:
  -h, --help            show this help message and exit
  --program-name PROGRAM_NAME
                        Name of the program to evaluate. You will be asked to pick a program if not specified.
```
**Sample result:**
![Example output](singularity_analysis/example_output.png)
