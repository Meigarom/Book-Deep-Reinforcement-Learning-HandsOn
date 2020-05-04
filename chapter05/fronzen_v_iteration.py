import gym
import collections

ENV_NAME='FrozenLake-v0'

class Agent:
    def __init__( self ):
        self.env = gym.make( ENV_NAME )
        self.state = self.env.reset()
        self.rewards = collections.defaultdict( float )
        self.transits = collections.defaultdict( collections.Counter )
        self.values = collections.defaultdict( float )


if __name__ == '__main__':
    test_env = gym.make( ENV_NAME )
    agent = Agent()
