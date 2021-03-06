# Reinforcement-Learning-Algorithm-Implementation
I tried to implement all algorithms and examples in the book "[Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018trimmed.pdf)" using Python individually. Please feel free to use all these codes for education or self-study. 

# Contents
## Chapter 2
|<img src="https://i.imgur.com/si2CUuM.png" width="200" height="150"><p align="center"> Figure 2.1 </p>|<img src="https://i.imgur.com/VXX3Gki.png" width="200" height="150"><p align="center"> Figure 2.2 </p>|<img src="https://i.imgur.com/BvkK1fr.png" width="200" height="150"><p align="center"> Figure 2.3 </p>|
|---|---|---|
|<img src="https://i.imgur.com/ev5Nnl1.png" width="200" height="150"><p align="center"> <b>Figure 2.4</b> </p>|<img src="https://i.imgur.com/3rJGoAr.png" width="200" height="150"><p align="center"> <b> Figure 2.5 </b> </p>|<img src="https://i.imgur.com/w3TYCSn.png" width="200" height="150"><p align="center"><b> Figure 2.6 </b> </p>|
1. [Figure 2.1: An example bandit problem from the 10-armed testbed.](https://i.imgur.com/si2CUuM.png) 
2. [Figure 2.2: Average performance of "ε-greedy action-value methods on the 10-armed testbed.](https://i.imgur.com/VXX3Gki.png)
3. [Figure 2.3: The effect of optimistic initial action-value estimates on the 10-armed testbed.](https://i.imgur.com/BvkK1fr.png) 
4. [Figure 2.4: Average performance of UCB action selection on the 10-armed testbed.](https://i.imgur.com/ev5Nnl1.png)
5. [Figure 2.5: Average performance of the gradient bandit algorithm with and without a reward baseline on the 10-armed testbed when the q_star(a) are chosen to be near +4 rather than near zero.](https://i.imgur.com/3rJGoAr.png)
6. [Figure 2.6: A parameter study of the various bandit algorithms presented in this chapter.Each point is the average reward obtained over 1000 steps with a particular algorithm at a particular setting of its parameter.](https://i.imgur.com/w3TYCSn.png)

## Chapter 3
|<img src="https://i.imgur.com/QzGFsT3.png" width="200" height="150"><p align="center"> Figure 3.2 </p>|<img src="https://i.imgur.com/yPPa3Ps.png" width="200" height="150"><p align="center"> Figure 3.5 </p>|
|---|---|

1. [Figure 3.2: Gridworld example:state-value function for the equiprobable random policy](https://i.imgur.com/QzGFsT3.png)
2. [Figure 3.5: Optimal solutions to the gridworld example.](https://i.imgur.com/yPPa3Ps.png)

## Chapter 4
|<img src="https://i.imgur.com/LOqJemG.png" width="200" height="150"><p align="center"> Figure 4.1 </p>|<img src="https://i.imgur.com/Jb1q89P.png" width="200" height="150"><p align="center"> Figure 4.2 </p>|<img src="https://i.imgur.com/ASxCemV.png" width="200" height="150"><p aign="center"> Figure 4.3 </p>|
|---|---|---|
1. [Figure 4.1: Convergence of iterative policy evaluation on a small gridworld.](https://i.imgur.com/LOqJemG.png)
1. [Figure 4.2: The sequence of policies found by policy iteration on Jack’s car rental problem, and the final state-value function.](https://i.imgur.com/Jb1q89P.png)
2. [Figure 4.3: The solution to the gambler’s problem for ph = 0.4. The upper graph shows the value function found by successive sweeps of value iteration. The lower graph shows the final policy.](https://i.imgur.com/ASxCemV.png)

## Chapter 5
|<img src="https://i.imgur.com/bfNIi8v.png" width="200" height="150"><p align="center"> Figure 5.1 </p>|<img src="https://i.imgur.com/Np7J17N.png" width="200" height="150"><p align="center"> Figure 5.2 </p>|<img src="https://i.imgur.com/G6bGNbu.png" width="200" height="150"><p aign="center"> Figure 5.3 </p>|<img src="https://i.imgur.com/wiynD78.png" width="200" height="150"><p aign="center"> Figure 5.4 </p>|
|---|---|---|---|
1. [Figure 5.1: Approximate state-value functions for the blackjack policy that sticks only on 20 or 21, computed by Monte Carlo policy evaluation.](https://i.imgur.com/bfNIi8v.png)
2. [Figure 5.2: The optimal policy and state-value function for blackjack, found by Monte Carlo ES. The state-value function shown was computed from the action-value function found by Monte Carlo ES.](https://i.imgur.com/Np7J17N.png)
3. [Figure 5.3: Weighted importance sampling produces lower error estimates of the value of a single blackjack state from off-policy episodes.](https://i.imgur.com/G6bGNbu.png)
4. [Figure 5.4: Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP shown inset (Example 5.5).](https://i.imgur.com/wiynD78.png)

## Chapter 6
|<img src="https://imgur.com/cgEorvy.png" width="200" height="150"><p align="center"> Example 6.2 </p>|<img src="https://i.imgur.com/HVvLUSw.png" width="200" height="150"><p align="center"> Figure 6.2 </p>|<img src="https://i.imgur.com/7FIjyRm.png" width="200" height="150"><p align="center"> Example 6.5 </p>|
|---|---|---|
|<img src="https://i.imgur.com/cGoWwQY.png" width="200" height="150"><p align="center"> <b>Example 6.6</b> </p>|<img src="https://i.imgur.com/lRtrYqb.png" width="200" height="150"><p align="center"> <b> Figure 6.3 </b> </p>|<img src="https://i.imgur.com/H3yWsHC.png" width="200" height="150"><p align="center"><b> Figure 6.5 </b> </p>|
1. [Example 6.2: Random walk.](https://imgur.com/cgEorvy.png) 
2. [Figure 6.2: Performance of TD(0) and constant-α MC under batch training on the random walk task.](https://i.imgur.com/HVvLUSw.png)
3. [Example 6.5: Windy gridworld.](https://i.imgur.com/7FIjyRm.png) 
4. [Example 6.6: Cliff walking.](https://i.imgur.com/cGoWwQY.png)
5. [Figure 6.3: Interim and asymptotic performance of TD control methods on the cliff-walking task as a function of α.](https://i.imgur.com/lRtrYqb.png)
6. [Figure 6.5: Comparison of Q-learning and Double Q-learning on a simple episodic MDP.](https://i.imgur.com/H3yWsHC.png)


## Chapter 7
|<img src="https://i.imgur.com/5e1IkIb.png" width="200" height="150"><p align="center"> Figure 7.2 </p>|
|---|
1. [Figure 7.2: Performance of n-step TD methods as a function of ↵, for various values of n, on a 19-state random walk task (Example 7.1)](https://i.imgur.com/5e1IkIb.png)
