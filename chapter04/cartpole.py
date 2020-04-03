import gym
import torch
import torch.nn    as nn
import torch.optim as optim

HIDDEN_SIZE=128

class Net( nn.Module ):
    def __init__( self, obs_size, hidden_size, n_actions ):
        super( Net, self ).__init__()
        self.net = nn.Sequential(
                nn.Linear( obs_size, hidden_size ),
                nn.ReLU(),
                nn.Linear( hidden_size, n_actions )
        )

    def  forward( self, x ):
        return self.net( x )

Episode = namedtuple( 'Episode', field_names=['reward', 'steps'] )
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation','action'])

if __name__ == '__main__':
    env = gym.make( 'CartPole-v0' )
    #env = gym.wrappers.Monitor( env, directory='mon', force=True )
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # NN Agent
    net = Net( obs_size, HIDDEN_SIZE, n_actions )
