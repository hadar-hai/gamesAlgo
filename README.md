[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3121/)
[![NumPy](https://img.shields.io/badge/numpy-1.26.2+-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.26.2/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.8.2+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.8.2)
[![pandas](https://img.shields.io/badge/pandas-2.1.4+-green?logo=pandas&logoColor=white)](https://pandas.org/)

# Games Algorithms

#### [Hadar Hai](https://www.linkedin.com/in/hadar-hai/), Technion - Israel Institute of Technology

This codebase is a comprehensive collection of AI algorithms designed to enhance gaming experiences through intelligent decision-making. Included within are implementations of fundamental strategies like Greedy, Minimax, Alpha-Beta Pruning, and Expectimax algorithms. These algorithms serve as powerful tools for creating AI agents capable of making strategic, optimal, and adaptive choices within various game environments.

The game unfolds on a 5x5 grid, featuring 2 robots, 2 charging stations, 2 packages, and 2 destinations—one for each package. Each robot possesses battery and credit units. The objective is to outscore the opponent by transporting packages to their respective destinations. A robot earns points based on the Manhattan distance multiplied by 2 between a package's original location and its destination upon successful delivery. The game concludes when a robot depletes its battery or reaches the maximum allowed steps. Robots move up, down, left, or right, and can collect, deposit, or charge at stations. Actions like picking up or dropping off packages occur only at valid slots, while any robot can access charging stations.

## Table of Contents

* [Requirements](#requirements)
* [How to Use](#how-to-use)
* [Examples](#examples)
* [Acknowledgements](#acknowledgements)

## Requirements

Install the Python environment.

1. Create an environment.

```batch
python -m venv venv 
venv\Scripts\activate.bat
pip install wheel
```

2. Install pygame

```batch
pip install pygame
```
# How to Use

Run `main.py` using your desired path, e.g.:

```batch
python main.py greedyImproved random -t 0.5 -s 2000 -c 200 --console_print --screen_print
```
- The first argument (in this case greedyImproved) specifies the algorithm by which agent0 will
play
- The second argument (in this case random) specifies the algorithm by which agent1
will play
- Time limit for the step t- gets a value that represents the maximum number of
seconds for the step
- A seed for receiving a random value s- receives a value that helps generate an
environment randomly, when the same seed value is passed it will generate the
same environment
- The maximum number of steps for agent c-
- value console_print-- an optional flag displaying data
- value screen_print-- an optional flag displaying the game

# Examples


















# Acknowledgements

The project was created as a part of course CS236501 of Computer Science faculty, Technion.