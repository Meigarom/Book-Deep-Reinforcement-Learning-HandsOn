# Book-Deep-Reinforcement-Learning-HandsOn
This repository has exercises and notes about my studies of the book "Deep Reinforcement Learning Hands on" by Maxim Lapan

## Chapter 04 - The Cross-Entropy Method
- Cross-Entropy method is really simple
- Cross-Entropy has good convergence
	- don't require complex policies to be learned and discovered
	- short episodes with frequent rewards

### The taxonomy of RL methods
Cross-Entropy falls into the **model-free** and **policy-based**.
You problem specifics can influnce your decision on a particular method.
- **Model-free** or **model-based**
- **Value-based** or **policy-based**
- **On-policy** or **off-policy**

**Model-free** doesn't build a model of the environment or reward. It just directly connects observations to actions. 
Current observations -> computations -> actions

**Model-based** try to predict what the next observation and/or reward will be. Based on this prediction, the agent tries to choose the best possible action to take, very often making such predictions multiple times to look more and more steps into the future.

- Strong and Weak: Pure model-based methods are used in deterministic environments, such as board games with strict rules. On the other hands, model-free methods are usually easier to train as it's hard to build good models of complex environment with rich observations.

**Policy-based** directly approximate the policy of the agent, that is, what actions the agent should carry out at every step. The policy is usually represented by a probability distribution over the available actions.

**Value-based** instead of the probability of actions, the agent calculates the value of every possible action and choose the action with the best value.

**Off-policy** the ability of the method to learn on historical data, obtained by a previous version of the agent, recorded by human demonstration, or just seen by the same agente several episodes ago.

**Cross-entropy** is a **model-free**, **policy-based** and **on-policy**
- It doesn't build any model of the environment. It just says to the agent what to do at every step.
- It approximates the policy of the agent.
- It requires fresh data obtained from the environment.

- The core of the cross-entropy method is to throw away bad episodes and train on better ones. So, the steps of the method are as follows:
1. Play N number of episodes using our current model and environment.
2. Calculate the total reward for every episode and decide on a reward boundary. Usually, we use some percentile of all rewards, such as 50th or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.

With the preceding procedure, our NN learns how to repeat actions, which leads to a larger reward, constantly moving the boundary higher and higher. Despite the simplicity of this method, it works well in basic environments, it's easy to implement, and it's quite robust to hyperparameters changing, which makes it an ideal baseline method to try.


###  Limitations of the cross-entropy method:
1. For training, our episodes have to be finite and, preferably, short.
2. The total reward for the episodes should have enough variability to separate good episodes from bad ones.
3. There is no intermediate indication about wheter the agent has succeeded of failed



![SSR]( https://latex.codecogs.com/gif.latex?SSR&space;=&space;\sum_{i=1}^{n}(&space;\hat{y_{i}}&space;-&space;\bar{&space;y&space;}&space;)^2 ).
