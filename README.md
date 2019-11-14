# marketenv
The marketenv package is a collection of stock market environments for reinforcement learning agents modeled after the OpenAI Gym environments. The environment objects are heavily modularlized so that both historical and simulated data can be used. Each market environment is comprised of two main objects: (1) a simulator, which is used to simulate the time series for each episode, and (2) an inventory object, which tracks the cash on hand and shares owned. The base environments provide very basic information, so customizable wrappers can be used for feature engineering, action mappings, and reward shaping. 

While users can use the different environment components to create customized environments, there are also out of the box environments. Users can create these environments using the make_env function and get a brief overview of the environment using the describe_env function. The MiniMarket-v1 EDA iPython Notebook looks at the first environment and the data it is based on in detail and is a good introduction to the market environment objects.

## Development Roadmap
This package was started as part of a class project, but I'm very excited to continue to expand on it. Right now, there are a few things that are top of mind:
1. Environments that support short selling
2. Generative adversarial network simulator
3. Acquiring quality historical data for out of the box historical environments

## Requirements
| Name            | Version   | Notes                       |
|-----------------|-----------|-----------------------------|
| numpy           | 1.16.5    |                             |
| pandas          | 0.25.2    |                             | 
