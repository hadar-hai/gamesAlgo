[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3121/)

# Games Algorithms

![warehouse](https://github.com/hadar-hai/gamesAlgo/assets/64587231/32f37b82-e82a-43bc-99b0-4046a0e73264)

#### [Hadar Hai](https://www.linkedin.com/in/hadar-hai/), Technion - Israel Institute of Technology

This codebase is a comprehensive collection of AI algorithms designed to enhance gaming experiences through intelligent decision-making. Included within are implementations of fundamental strategies like Greedy, Minimax, Alpha-Beta Pruning, and Expectimax algorithms. These algorithms serve as powerful tools for creating AI agents capable of making strategic, optimal, and adaptive choices within a game environment.

The game unfolds on a 5x5 grid, featuring 2 robots, 2 charging stations, 2 packages, and 2 destinations—one for each package. Each robot possesses battery and credit units. The objective is to outscore the opponent by transporting packages to their respective destinations. A robot earns points based on the Manhattan distance multiplied by 2 between a package's original location and its destination upon successful delivery. The game concludes when a robot depletes its battery or reaches the maximum allowed steps. Robots move up, down, left, or right, and can collect, deposit, or charge at stations. Actions like picking up or dropping off packages occur only at valid slots, while any robot can access charging stations.

## Table of Contents

* [Requirements](#requirements)
* [How to Use](#how-to-use)
* [Algorithms Explanation](#algorithms-explanation)
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

# Algorithms Explanation

## Heuristic Function

<p>𝒉𝒆𝒖𝒓𝒊𝒔𝒕𝒊𝒄 = 𝑭𝟏 ∙ 𝑾𝟏 + 𝑭𝟑 ∙ 𝑾𝟑 + 𝑭𝟒 ∙ 𝑾𝟒 + (𝑭𝟓 ∙ 𝑾𝟓 + 𝑭𝟔 ∙ 𝑾𝟔) ∙ ¬𝑭𝟕 + (𝑾𝟕 + 𝑭𝟖 ∙ 𝑾𝟖) ∙ 𝑭𝟕 + (𝑾𝟗 + 𝑭𝟏 ∙ 𝑾𝟏 + 𝑭𝟐 ∙ 𝑾𝟐) ∙ 𝑭𝟗 + 𝑭𝟏𝟎 ∙ 𝑾𝟏𝟎 + 𝑭𝟏𝟏 ∙ 𝑾𝟏𝟏</p>
<p>𝒉𝒆𝒖𝒓𝒊𝒔𝒕𝒊𝒄 = 𝒉𝒆𝒖𝒓𝒊𝒔𝒕𝒊𝒄 ∙ ¬𝑭𝟏𝟐 ∙ ¬𝑭𝟏𝟑 + 𝑭𝟏𝟐 ∙ 𝑾𝟏𝟐 + 𝑭𝟏𝟑 ∙ 𝑾𝟏𝟑</p>

![heuristic_function](https://github.com/hadar-hai/gamesAlgo/assets/64587231/8c1a7705-8027-4057-8f1a-110758d431c3)

• Most valuable package is selected by = 𝑴𝒂𝒙 (
𝟐∙{𝑷𝒂𝒄𝒌𝒂𝒈𝒆 𝒕𝒐 𝒅𝒆𝒔𝒕𝒊𝒏𝒂𝒕𝒊𝒐𝒏 𝒅𝒊𝒔𝒕𝒂𝒏𝒄𝒆}
{𝑷𝒂𝒄𝒌𝒂𝒈𝒆 𝒕𝒐 𝒅𝒆𝒔𝒕𝒊𝒏𝒂𝒕𝒊𝒐𝒏 𝒅𝒊𝒔𝒕𝒂𝒏𝒄𝒆} + {𝑹𝒐𝒃𝒐𝒕 𝒕𝒐 𝒑𝒂𝒄𝒌𝒂𝒈𝒆 𝒅𝒊𝒔𝒕𝒂𝒏𝒄𝒆}
∙ 𝟏𝟎 −
{𝑹𝒐𝒃𝒐𝒕 𝒕𝒐 𝒑𝒂𝒄𝒌𝒂𝒈𝒆 𝒅𝒊𝒔𝒕𝒂𝒏𝒄𝒆}) 

• {Robot to destination distance} is a value from 0 to 10. So, we multiply the “credit to battery ratio” by 10 
– In this way we give higher importance to the potential of the package, and if potential are equals, we give secondary 
importance to the distance of the robot from package (Closer the better) 

## Greedy Imrpoved

Greedy Improved is a decision-making algorithm that selects the next step based on a sophisticated heuristic function. At each iteration, it picks the move that minimizes the heuristic function, aiming to make locally optimal choices in the immediate decision space.

## Minimax

Minimax is a decision-making algorithm commonly used in game theory and AI to determine the best move for a player in a turn-based game. It works by evaluating the possible moves of both players, assuming that the opponent will also make optimal moves. The algorithm aims to minimize the potential loss for the worst-case scenario (hence "min") while maximizing the potential gain for the player making the move (hence "max"). By recursively exploring the game's possible future states through a tree-like structure, Minimax computes the best move based on a defined scoring or evaluation function. This algorithm is fundamental in creating intelligent game-playing agents, particularly in scenarios where all possible moves and their outcomes are known.

## Alpha-Beta

Alpha-Beta pruning is an optimization technique applied to the Minimax algorithm, primarily used in game-playing AI. It works by significantly reducing the number of nodes evaluated in the game tree while maintaining the same optimal move determination as Minimax. By discarding certain branches of the game tree that are deemed irrelevant for the final decision, Alpha-Beta pruning efficiently narrows down the search space. This is achieved by maintaining two values, alpha and beta, representing the best achievable scores for the maximizing and minimizing players, respectively. During the search, when it's identified that a move leads to a situation worse than what's already been found for the opposing player, that branch is pruned, effectively ignoring further exploration along that path. Alpha-Beta pruning can significantly reduce the computation time, especially in complex games, by disregarding futile branches and focusing on the most promising moves, making it an essential optimization for game-playing AI.

## Expectimax

Expectimax is a probabilistic extension of the Minimax algorithm, often used in decision-making for games or problems involving uncertainty or chance elements. Unlike Minimax, which assumes that opponents make optimal moves, Expectimax considers all possible actions a player can take along with the probabilities associated with those actions. In scenarios where outcomes aren't deterministic, such as in games with random events or incomplete information, Expectimax accounts for the uncertainty by incorporating expected values.

This algorithm evaluates moves by calculating the expected value of each possible action, considering the probabilities of various outcomes occurring. It navigates through the game tree by averaging the possible scores based on the likelihood of each event happening. Expectimax is particularly useful in scenarios where uncertainty or randomness plays a significant role, allowing AI agents to make informed decisions in such stochastic environments, as seen in various games like probabilistic board games or games involving chance elements.

# Examples

## Greedy Improved

![greedy_improved_random](https://github.com/hadar-hai/gamesAlgo/assets/64587231/f6ba3213-5eb6-462d-8082-bbc6d66e5b80)

## Minimax

![minimax_random](https://github.com/hadar-hai/gamesAlgo/assets/64587231/df9ca6ef-8df2-4f97-9c73-e677736494d2)

## Alpha-Beta

## Expectimax

![expectimax_random](https://github.com/hadar-hai/gamesAlgo/assets/64587231/fdd8e64d-1290-4413-bdbc-89eae409e51e)















# Acknowledgements

The project was created as a part of course CS236501 of Computer Science faculty, Technion.
