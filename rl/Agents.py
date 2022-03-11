# this is going to be the Agent that interacts with the environment given the state
# there should be 3 main functionalities
#     1. act (Agent decides the action based on the state)
#     2. observe (Agent captures the state and the target Q values)
#     3. replay (Agent trains the model based on the observation)

# libraries
import math
from rl.Memory import Memory
from rl.DQN import DQN
import numpy as np
import random
from rl.helper_functions import sel_action, sel_action_index


# Agent and Random Agent Implementations
# defining global variables

max_reward = 10
grass_penalty = 0.4
action_repeat_num = 8
max_num_episodes = 1000
memory_size = 10000
max_num_steps = action_repeat_num * 100
gamma = 0.99
max_eps = 0.1
min_eps = 0.02
EXPLORATION_STOP = int(max_num_steps * 10)
_lambda_ = - np.log(0.001) / EXPLORATION_STOP
UPDATE_TARGET_FREQUENCY = int(50)
batch_size = 128


# creating the Agent itself

class Agent:

    # constructor
    def __init__(self, num_states, num_actions, img_dim, model_path):

        # print field
        print('num_states:', num_states)
        print('num_actions:', num_actions)
        print('img_dim:', img_dim)
        print('model_path:', model_path)

        # global datafield
        self.steps = 0
        self.epsilon = max_eps
        self.memory = Memory(memory_size)

        # const passed in datafield
        self.num_states = num_states
        self.num_actions = num_actions
        self.DQN = DQN(num_states, num_actions, model_path)
        self.no_state = np.zeros(num_states)
        self.x = np.zeros((batch_size,)+img_dim)
        self.y = np.zeros([batch_size, num_actions])
        self.errors = np.zeros(batch_size)
        self.rand = False

        self.agent_type = 'Learning'
        self.maxEpsilon = max_eps


    # act function (#1)
    def act(self, s):

        print(self.epsilon)

        if random.random() < self.epsilon:
            best_act = np.random.randint(self.num_actions)
            self.rand = True
            return sel_action(best_act), sel_action(best_act)
        else:
            act_soft = self.DQN.predict_single_state(s)
            best_act = np.argmax(act_soft)
            self.rand = False
            return sel_action(best_act), act_soft


    # compute targets function used by the observe function (#2) and replay function (#3)
    def compute_targets(self, batch):

        '''
            0 -> Index for current state
            1 -> Index for action
            2 -> Index for reward
            3 -> Index for next state
        '''
        states = np.array([ rec[1][0] for rec in batch ])
        states_ = np.array([ (self.no_state if rec[1][3] is None else rec[1][3]) for rec in batch ])

        p = self.DQN.predict(states)
        p_ = self.DQN.predict(states_, target=False)
        p_t = self.DQN.predict(states_, target=True)

        # action counter
        act_ctr = np.zeros(self.num_actions)

        for i in range(len(batch)):
            rec = batch[i][1]
            s, a, r, s_ = rec[0], rec[1], rec[2], rec[3]

            a = sel_action_index(a)
            t = p[i]
            act_ctr[a] += 1

            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                # DDQN
                t[a] = r + gamma * p_t[i][ np.argmax(p_[i]) ]

            self.x[i] = s
            self.y[i] = t

            if self.steps % 20 ==0 and i == len(batch)-1:
                print('t',t[a], 'r: %.4f' % r,'mean t',np.mean(t))
                print ('act ctr: ', act_ctr)

            self.errors[i] = abs(oldVal - t[a])

        return (self.x, self.y, self.errors)


    # observe function (#2)
    # in (s, a, r, s_) format
    def observe(self, sample):
        _, _, errors = self.compute_targets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.DQN.target_model_update()
        self.steps += 1
        self.epsilon = min_eps + (self.maxEpsilone - min_eps) * np.exp(-1*_lambda_ * self.steps)


    # replay function (#3)
    def replay(self):
        batch = self.memory.sample(batch_size)
        x, y, errors = self.compute_targets(batch)
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.DQN.train(x, y)


# Random Agent class defined similar to the agent class

class RandomAgent:

    # constructor
    def __init__(self, num_actions):

        # global datafield
        self.memory = Memory(memory_size)
        self.exp = 0
        self.steps = 0

        # const passed in datafield
        self.num_actions = num_actions
        self.agent_type = 'Learning'
        self.rand = True


    # act function (#1)
    def act(self, s):
        best_act = np.random.randint(self.num_actions)
        return sel_action(best_act), sel_action(best_act)


    # observe function (#2)
    # in (s, a, r, s_) format
    def observe(self, sample):
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1
        self.steps += 1


    # replay function (#3)
    def replay(self):
        pass
