import gym
import numpy       as np
import torch
import torch.nn    as nn
import torch.optim as optim

from collections import namedtuple
from tensorboardX import SummaryWriter

HIDDEN_SIZE=128
BATCH_SIZE = 16
PERCENTILE = 70

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

def iterate_batches( env, net, batch_size ):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax( dim=1 )

    while True:
        # convert inputs number to inputs tensor
        obs_v = torch.FloatTensor( [obs] )
        
        # apply: NN and then Softmax function
        act_probs_v = sm( net( obs_v ) )

        # get the probs from the tensor
        act_probs = act_probs_v.data.numpy()[0]

        # select randomly the action 0 or 1 with p probability for each.
        # np.random.choice( 2 ) -> create a list of 2 numbers ( [0,1] )
        # p=act_probs           -> probability associated with each entry
        action = np.random.choice( len( act_probs ), p=act_probs )

        # apply action over the environment
        next_obs, reward, is_done, _ = env.step( action )

        # save reward
        episode_reward += reward

        step = EpisodeStep( observation=obs, action=action )
        episode_steps.append( step )

        # Environment says if it's done
        if is_done: 
            # save the last observations and the last action for each episode
            e = Episode( reward=episode_reward, steps=episode_steps )
            batch.append( e )
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            # If we get the batch size, we get out!
            if len( batch ) == batch_size:
                yield batch
                batch = []

        obs = next_obs

def filter_batch( batch, percentile ):
    rewards = list( map( lambda s: s.reward, batch ) )
    reward_bound = np.percentile( rewards, percentile )
    reward_mean = float( np.mean( rewards ) )

    train_obs = []
    train_act = []
    for reward, steps in batch: # Get the best 30% out of 16 episode
        if reward < reward_bound:
            continue

        train_obs.extend( map( lambda step: step.observation, steps ) )
        train_act.extend( map( lambda step: step.action, steps ) )

    # convert back to tensor
    train_obs_v = torch.FloatTensor( train_obs )
    train_act_v = torch.LongTensor( train_act )

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make( 'CartPole-v0' )
    #env = gym.wrappers.Monitor( env, directory='mon', force=True )

    # observations ( environment output )
    obs_size = env.observation_space.shape[0]

    # actions ( environment input )
    n_actions = env.action_space.n

    # NN Agent
    net = Net( obs_size, HIDDEN_SIZE, n_actions )
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam( params=net.parameters(), lr=0.01 )
    writer = SummaryWriter( comment='-cartpole' )

    # For loop to play the game and get batches epsidoes with:
    # - observations
    # - actions
    # - rewards
    for iter_no, batch in enumerate( iterate_batches(env, net, BATCH_SIZE) ):
        obs_v, acts_v, reward_b, reward_m = filter_batch( batch, PERCENTILE )

        optimizer.zero_grad()

        # retrain forward with best episodes ( > 70th percentile )
        action_scores_v = net( obs_v )

        # compute the error
        loss_v = objective( action_scores_v, acts_v )

        # update the NN weights
        loss_v.backward()
        
        # apply optmizer to upste the weights
        optimizer.step()

        print( '%d: loss=%.3f, reward_mean=%1.f, rw_bound=%.1f' % ( iter_no, loss_v.item(), reward_m, reward_b ) )

        writer.add_scalar( 'loss', loss_v.item(), iter_no )
        writer.add_scalar( 'reward_bound', reward_b, iter_no )
        writer.add_scalar( 'reward_mean', reward_m, iter_no )

        if reward_m > 199:
            print( 'Solved!' )
            break

    writer.close()
