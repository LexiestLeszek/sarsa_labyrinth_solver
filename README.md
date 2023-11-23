# SARSA (on poilicy TD control) learning algorithm
State-action-reward-state-action (SARSA) learning algorithm to solve labyrinth runs using reinforcement learning.

## Dataset 
Upload the dataset into the algorithm as a *.txt file that contains info about the labyrinth e.g. - 

sssssssssssssss
000000000000000
000000000000000
000000000000000
000000000000000
0000xxxxxxx0000
000000000000000
000000000000000
000000000000000
xxxxx00000xxxxx
000000000000000
000000000000000
000000000000000
0000xxxxxxx0000
000000000000000
000000000000000
fffffffffffffff

Where
- **x** - barrier
- **s** - start
- **f** - final

## Getting Started
- Clone the repository
</pre></code>
- run main.py with args, something like: 
<pre><code>
python3 main.py -f "data/labyrinth2.txt" -a 0.4 -g 0.9 -e 0.1
</pre></code>
-a (alpha): Represents the learning rate in the context of reinforcement learning. It determines to what extent newly acquired information overrides old information. A low learning rate means the agent is slower to learn, while a high learning rate means the agent quickly adapts to new information. In the provided code, alpha seems to represent the learning rate.

-g (gamma): Denotes the discount factor which determines the importance of future rewards in the agent's decision-making process. It discounts future rewards relative to immediate rewards. A gamma of 0 means the agent only considers immediate rewards, while a gamma close to 1 means the agent considers future rewards with higher significance.

-e (epsilon): Represents the exploration rate, usually in epsilon-greedy strategies. It determines the probability of the agent choosing a random action over the action with the highest Q-value. A higher epsilon value implies more exploration (choosing random actions), while a lower value leads to more exploitation (choosing actions based on learned Q-values).

## Inferences
**state space** are the coordinates of the labyrinth with (-1,-1) being the state when the agent goes out of bounds and 
**action space** are the two components of velocity with a constrint that it can have a absolute value of greater than 5.

The algorithm was tested for three labyrinths. We can see that the algorithm(agent) starts to find the currect path after few iteration and also the proportion of successes keep on increasing with the number of episodes as the agent finds the optimal action value function.
