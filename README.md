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



### The Coefficient of Determination, r-squared
Measure the strenght of the relationship
- SSR ( Reg Sum of Squares ): Quantifies how far the estimated sloped regression line, yhat, is from the horizontal, ybar (no relationship line).

![SSR]( https://latex.codecogs.com/gif.latex?SSR&space;=&space;\sum_{i=1}^{n}(&space;\hat{y_{i}}&space;-&space;\bar{&space;y&space;}&space;)^2 ).

- SSE ( Error Sum of Squares ): Quantifies how much the data points, yi, vary around the estimated regression line, yhat.
![SSE]( https://latex.codecogs.com/gif.latex?SSE%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28%20y_%7Bi%7D%20-%20%5Chat%7B%20y_%7Bi%7D%20%7D%20%29%5E2 ).

- SSTO ( Total Sum of Squares ): Quantifies how much the data points, yi, vary around their mean, ybar.
![SSTO]( https://latex.codecogs.com/gif.latex?SSE%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28%20y_%7Bi%7D%20-%20%5Cbar%7B%20y_%7Bi%7D%20%7D%20%29%5E2 ).

- SSTO = SSR + SSE

#### **Coefficient of determination** or **R-squared**:
Percentage of the variation in y is reduced by taking into account predictor x.
Percentage of the variation in y is 'explained by' the variation in predictor x.

![R-squared]( https://latex.codecogs.com/gif.latex?r%5E2%20%3D%20%5Cfrac%7B%20SSR%20%7D%7B%20SSTO%20%7D%20%3D%201%20-%20%5Cfrac%7BSSE%7D%7BSSTO%7D )
- **r-squared** is a number between 0 and 1.
- If **r-squared = 1**, all of the data points fall perfectly on the regression line. The predictor x accounts for all the variation in y!.
- If **r-squared = 0**, the estimated regression line is perfectly horizontal. The predictor x accounts for none of the variation in y!.

### (Pearson) Correlation Coefficient r
Measure the sign of the relationship

### R-squared Cautions
1. The r-squared quantifies the strength of a **linear** relationship. If r-squared = 0, tells us that if there is a relationship between x and y, it's not linear.

2. A large r-squared value should not be interpreted as meaning that the estimated regression line fits the data well.
Its large values does suggest that taking into account the predictor is better than not doing so. It just doesn't tell us that we could still do better.

3. The r-squared can be greatly affected by just one data point (or a few data points).

4. Correlation (or association) does not imply causation.

5. Ecological correlations, correlations that are based on rates or averages, tend to overstate the strength of an association.

6. A **statistically significant** r-squared does not imply that the slope Beta1 is meaninfully different from 0.

7. A large r-squared value does not necessarily mean that an useful prediction of the response can be made. It's still possible to get prediction intervals or confidence intervals that are too wide to be useful.

## Lesson 02 - Simple Linear Regression Evaluation
This lesson presents two alternatives methods for testing if a linear association exists between the
predictor x and the response y in a simple linear regression model.
![B0](https://latex.codecogs.com/gif.latex?H_%7B0%7D%3A%20%5Cbeta_%7B1%7D%20%3D%200 ) versus (https://latex.codecogs.com/gif.latex?H_%7BA%7D%3A%20%5Cbeta_%7B1%7D%20%5Cneq%200)
- The t-test for the slope
- Analysis of variance (ANOVA) F-test
