import time
from tqdm import tqdm
import numpy as np
import h5py
import json

from action_selection import eps_greedy
from network import Network
from agent_dqn import Agent
from tqdm import tqdm

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    NUM_EPISODES = 600
else:
    NUM_EPISODES = 50
# NUM_EPISODES = 100000
NUM_STEPS = 1000
NUM_RUNS = 100
multi_runs = False
# actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
actions = [FORWARD, BACKWARD, RIGHT, LEFT]
num_acts = len(actions)

# Environment dimensions
NUM_ROWS = 16
NUM_COLS = 16
row_lim = NUM_ROWS - 1
col_lim = NUM_COLS - 1

#                          r1  r2  r3  r4
adj_mat_prior = np.array([[1,  0],
                          [0,  1]], dtype=float)
exp_name = '2RS_100000episodes_'
NUM_VICTIMS = 10
# make the map from json file
# with open('data10.json') as f:
#     data = json.load(f)
#     test = data['map'][0]
#     dim = data['dimensions']
#     rows = dim[0]['rows']
#     columns = dim[0]['columns']
#
#     env_map = np.zeros((rows, columns))
#
#     for cell in data['map']:
#         if cell['isWall'] == 'true':
#             env_map[cell['x'], cell['y']] = 1

env_mat = np.zeros((NUM_ROWS, NUM_COLS))
global env_map
env_map = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# env_map=np.zeros_like(env_map)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Transition function (avoid walls)
def movement(pos, old_poses, next_poses, action, speed):
    global env_map
    row = pos[0]
    col = pos[1]
    next_pos = pos.copy()
    if action == 0:  # up
        next_pos = [max(row - speed, 0), col]
    elif action == 1:  # down
        next_pos = [min(row + speed, row_lim), col]
    elif action == 2:  # right
        next_pos = [row, min(col + speed, col_lim)]
    elif action == 3:  # left
        next_pos = [row, max(col - speed, 0)]

    obstacle_check = env_map[next_pos[0], next_pos[1]] == 0
    old_occupied_check = not(next_pos[0] in old_poses[:, 0] and next_pos[1] in old_poses[:, 1])
    next_occupied_check = not (next_pos[0] in next_poses[:, 0] and next_pos[1] in next_poses[:, 1])

    if obstacle_check and old_occupied_check and next_occupied_check:
        return next_pos
    else:
        return pos

def reward_func_ql(sensation_prime, dist2home, busy):
    perceived = (sensation_prime[0] == 0 and sensation_prime[1] == 0)
    saved = (dist2home[0] == 0 and dist2home[1] == 0)
    if perceived and not busy and not saved:
        re = 100
    elif saved and busy:
        re = 100
    else:
        re = -.1
    return re

def q_learning(q, old_idx, curr_idx, re, act, alpha=0.8, gamma=0.9):
    q[old_idx, act] += alpha * (re + gamma * np.max(q[curr_idx, :]) - q[old_idx, act])
    return q

global steps_done
steps_done = 0

# def select_action(agent, state):
#     global steps_done
#     state = torch.tensor(state, device=device)
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#                     math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return the largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return agent.policy_net(torch.DoubleTensor(state)).max().view(1, 1)
#     else:
#         idx = np.random.randint(num_acts)
#         return torch.tensor([[actions[idx]]], device=device, dtype=torch.long)


def select_action(state, epsilon, num_actions, model):

    if random.random() < epsilon:
        idx = random.randint(0, num_actions - 1)
        return torch.tensor([[actions[idx]]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return model(torch.FloatTensor(state)).argmax().item()


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def env(accuracy=1e-15):
    global adj_mat_prior
    # Define the Network and the agent objects
    network = Network
    agent = Agent

    epsilon_start = 1.0  # starting value of epsilon
    epsilon_end = 0.01   # minimum value of epsilon
    epsilon_decay = 0.995  # decay multiplier for epsilon

    # Define the rescue team
    r1 = agent(0, 'r', 3, NUM_ROWS, 1, [0, 7],
               num_acts, NUM_ROWS, NUM_COLS, LR)
    r2 = agent(1, 'r', 3, NUM_ROWS, 1, [0, 8],
               num_acts, NUM_ROWS, NUM_COLS, LR)
    r3 = agent(1, 'r', 3, NUM_ROWS, 1, [15, 7],
               num_acts, NUM_ROWS, NUM_COLS, LR)
    r4 = agent(3, 'r', 3, NUM_ROWS, 1, [15, 8],
               num_acts, NUM_ROWS, NUM_COLS, LR)
    # List of rescue team members
    rescue_team = [r1, r3]

    # Define the victims
    victim_locs = np.argwhere(env_map == 0).tolist()  # list of non-occupied locations in the map
    for member in rescue_team:
        victim_locs.remove(member.init_pos)  # remove the location of the rescue team members from the list

    victims = []  # make a list of the victims
    for victim_id in range(NUM_VICTIMS):
        loc = victim_locs[np.random.randint(len(victim_locs))]  # find a random non-occupied location
        victims.append(agent(victim_id, 'v', 0, 0, 1, loc, num_acts, NUM_ROWS, NUM_COLS, LR))  # initialize the victim
        victim_locs.remove(loc)  # remove the location of the previous victim from the list
    # v0 = agent(0, 'v', 0, 0, 1, [15, 7], num_acts, NUM_ROWS, NUM_COLS)
    # v1 = agent(1, 'v', 0, 0, 1, [15, 12], num_acts, NUM_ROWS, NUM_COLS)
    # v2 = agent(2, 'v', 0, 0, 1, [5, 5], num_acts, NUM_ROWS, NUM_COLS)
    # v3 = agent(3, 'v', 0, 0, 1, [11, 3], num_acts, NUM_ROWS, NUM_COLS)
    # v4 = agent(4, 'v', 0, 0, 1, [0, 1], num_acts, NUM_ROWS, NUM_COLS)
    # v5 = agent(5, 'v', 0, 0, 1, [14, 6], num_acts, NUM_ROWS, NUM_COLS)
    # v6 = agent(6, 'v', 0, 0, 1, [14, 7], num_acts, NUM_ROWS, NUM_COLS)
    # v7 = agent(7, 'v', 0, 0, 1, [3, 15], num_acts, NUM_ROWS, NUM_COLS)
    # v8 = agent(8, 'v', 0, 0, 1, [10, 15], num_acts, NUM_ROWS, NUM_COLS)
    # v9 = agent(9, 'v', 0, 0, 1, [0, 12], num_acts, NUM_ROWS, NUM_COLS)
    # victims = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]
    vfd_list = []  # make a list of the rescue team visual fields depths
    rescue_team_roles = []  # make a list of the rescue team roles
    muster_site = []  # make a list of the muster sites
    num_just_scouts = 0  # how many of the members in the rescue team are only scouting
    for member in rescue_team:
        rescue_team_roles.append(member.role)
        vfd_list.append(member.visual_field)
        muster_site.append(member.init_pos)
        if member.role == 's':
            num_just_scouts += 1
    # eps = -1
    tic = time.time()
    # while True:
    for eps in tqdm(range(NUM_EPISODES)):
        epsilon = epsilon_start
        rescue_team_hist = rescue_team.copy()
        victims_hist = victims.copy()
        adj_mat = adj_mat_prior.copy()

        agents_idx = []
        for member in rescue_team:
            agents_idx.append(member.id)

        victims_idx = []
        for victim in victims:
            victims_idx.append(victim.id)
        rescue_team_roles = np.array(rescue_team_roles, dtype=list)
        # eps += 1

        # Reset the agents flags, positions, etc
        for member in rescue_team:
            member.reset()
        # Reset the victims flags, positions, etc
        for victim in victims:
            victim.reset()

        t_step = 0
        # for _ in range(NUM_STEPS):
        while True:
            num_rescue_team = len(rescue_team_hist)
            num_victims = len(victims_hist)

            net = network(adj_mat, num_rescue_team, num_victims)

            t_step += 1

            rescue_team_vfd_list = []
            team_vfd_status = []
            for member in rescue_team_hist:
                # List of the Visual Fields
                rescue_team_vfd_list.append(member.visual_field)

                # Count the steps that agent could see a victim
                if member.perceived:
                    member.t_step_seen += 1

                # Keeping track of the rescue team positions
                member.traj.append(member.old_pos)

                # Update VFD status
                member.update_vfd(env_map)

                # Keep track of VFD status
                member.vfd_status_history.append(member.vfd_status)

                # vfd status for the team
                team_vfd_status.append(member.vfd_status)

                # History of Q
                # member.q_hist = member.q.copy()

            rescue_team_vfd_list = np.asarray(rescue_team_vfd_list)

            # Keep track of the victims positions
            # Make a list of the victims old positions
            victims_old_pos_list = []
            for victim in victims_hist:
                victim.traj.append(victim.old_pos)
                victims_old_pos_list.append(victim.old_pos)
            victims_old_pos_list = np.asarray(victims_old_pos_list)

            # Make a list of the agents old positions
            rescue_team_old_pos_list = []
            rescue_team_old_dist2home_list = []
            for member in rescue_team_hist:
                rescue_team_old_pos_list.append(member.old_pos)
                rescue_team_old_dist2home_list.append(member.old_dist2home)
            rescue_team_old_pos_list = np.asarray(rescue_team_old_pos_list)

            # Calculation of the distance between the agents
            old_scouts2rescuers = net.pos2pos(rescue_team_old_pos_list)

            # Calculation of the raw sensations for the rescue team
            old_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_old_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_old_sensations = net.is_seen(rescue_team_vfd_list, old_raw_sensations, team_vfd_status)

            rescue_team_curr_pos_list = []
            rescue_team_role_list = []
            seen_victim = []
            for member in rescue_team_hist:
                # Calculation of the sensations for the rescue team
                member.old_sensation, old_seen_victim = member.update_sensation(rescue_team_hist.index(member),
                                                             old_raw_sensations, eval_old_sensations,
                                                             old_scouts2rescuers, net.adj_mat, adj_mat)
                # Calculation of the indices for the rescue team
                member.old_index = member.sensation2index(member.old_sensation, member.old_dist2home)
                # member.old_index = torch.tensor(member.old_index, dtype=torch.float32, device=device).unsqueeze(0)
                member.old_state_vect = np.zeros((member.num_observations,))
                member.old_state_vect[member.old_index] = 1
                # # actions for the rescue team
                # if eps < NUM_EPISODES:
                epsilon = max(epsilon_end, epsilon_decay * epsilon)  # update epsilon
                member.action = select_action(member.old_state_vect, epsilon, member.num_actions, member.policy_net)  # select an action
                # member.action = select_action(member, member.old_state_vect)#eps_greedy(member.q[member.old_index, :], num_acts, epsilon=.05)
                # else:
                #     member.action = eps_greedy(member.q[member.old_index, :], num_acts, epsilon=0.0)
                # List of the current positions for the rescue team members
                rescue_team_curr_pos_list.append(member.curr_pos)

                # List of the roles for the rescue team members
                rescue_team_role_list.append(member.role)
                if old_seen_victim != None:
                    seen_victim.append(old_seen_victim)

            rescue_team_curr_pos_list = np.asarray(rescue_team_curr_pos_list)
            rescue_team_curr_dist2home_list = []
            for member in rescue_team_hist:
                # Next positions for the rescue team
                member.curr_pos = movement(member.old_pos, rescue_team_old_pos_list, rescue_team_curr_pos_list, member.action, member.speed)

                # Search algorithm
                # member.straight_move(member.old_index, member.were_here, env_map, member.busy)
                member.random_walk(member.old_index, member.old_pos, member.speed, env_map, member.busy)
                # member.ant_colony_move(env_mat, member.old_index, env_map, member.busy)
                # member.levy_walk(env_map, member.busy)

                member.update_dist2home()
                rescue_team_curr_dist2home_list.append(member.curr_dist2home)
            # Calculation of the distance between agents (after their movement)
            curr_scouts2rescuers = net.pos2pos(rescue_team_curr_pos_list)

            # Calculation of the new raw sensations for the rescue team (after their movement)
            curr_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_curr_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_curr_sensations = net.is_seen(rescue_team_vfd_list, curr_raw_sensations, team_vfd_status)

            # Calculation of the new sensations for the rescue team (after their movement)
            for member in rescue_team_hist:
                member.curr_sensation, curr_seen_victim = member.update_sensation(rescue_team_hist.index(member),
                                                              curr_raw_sensations, eval_curr_sensations,
                                                              curr_scouts2rescuers, net.adj_mat, adj_mat)

                # Calculation of the indices for the rescue team (after their movement)
                member.curr_index = member.sensation2index(member.curr_sensation, member.curr_dist2home)
                # member.curr_index = torch.tensor(member.curr_index, dtype=torch.float32, device=device).unsqueeze(0)
                member.curr_state_vect = np.zeros((member.num_observations,))
                member.curr_state_vect[member.curr_index] = 1
                # Rewarding the rescue team
                member.reward = reward_func_ql(member.curr_sensation, member.curr_dist2home, member.busy)
                member.reward = torch.tensor([member.reward], device=device)
                # q learning for agents in the rescue team
                # member.q = q_learning(member.q, member.old_index, member.curr_index, member.reward, member.action, alpha=0.8)
                if curr_seen_victim != None:
                    seen_victim.append(curr_seen_victim)
                rescue_team_hist, adj_mat = member.rescue_started(rescue_team_hist, member, adj_mat)
                adj_mat = member.rescue_accomplished(adj_mat, adj_mat_prior)
                # Check to see if the team rescued any victim
                for victim in victims_hist:
                    # Check to see if the victim rescued by the team
                    # Keep track of the steps
                    # Remove the victim from the list
                    if not victim.saved:
                        victims[victim.id] = victim
                        victims_hist = victim.victim_rescued(rescue_team_old_pos_list,
                                                             rescue_team_curr_pos_list,
                                                             muster_site,
                                                             rescue_team_role_list,
                                                             victim, victims_hist)
                        if victim.saved and victim.first:
                            victim.steps.append(t_step)
                            victim.first = False
                            # break  # Rescue more than one victim by an agent
                # Keeping track of the rewards
                member.rew_hist.append(member.reward)
                if member.perceived:
                    member.rew_hist_seen.append(member.reward)
                if member.busy and member.first:
                    member.steps.append(t_step)
                    member.steps_seen.append(member.t_step_seen)
                    member.rew_sum.append(torch.sum(torch.tensor(member.rew_hist, device=device)).item())
                    member.rew_sum_seen.append(torch.sum(torch.tensor(member.rew_hist_seen, device=device)).item())
                    member.first = False
                    rescue_team[member.id] = member

            if len(victims_hist) == 0 or t_step==3000:
                episode_durations.append(t_step + 1)
                plot_durations()
                # print(f'In episode {eps+1}, all of the victims were rescued in {t_step} steps')
                break

            # Update the rescue team positions
            for member in rescue_team_hist:
                member.old_pos = member.curr_pos
                # Store the transition in memory
                member.memory.push(member.old_state_vect, member.action, member.curr_state_vect, member.reward)
                # Perform one step of the optimization (on the policy network)
                member.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = member.target_net.state_dict()
                policy_net_state_dict = member.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                member.target_net.load_state_dict(target_net_state_dict)

            victims_curr_pos_list = []
            # Victims' actions and positions
            for victim in victims_hist:
                # actions for the victims
                victim.action = np.random.choice(actions)
                # Victims next positions
                victims_curr_pos_list.append(victim.curr_pos)
                if (victim.savior is not None) and (not victim.saved):
                    victim.old_pos = rescue_team_hist[victim.savior].curr_pos
                elif victim.id not in seen_victim:
                    # Update the victims position
                    victim.old_pos = victim.curr_pos
            victims_curr_pos_list = np.asarray(victims_curr_pos_list)
            for victim in victims_hist:
                victim.curr_pos = movement(victim.old_pos, victims_old_pos_list, victims_curr_pos_list, victim.action, victim.speed)
    # Add agents last pos in the trajectory
    for member in rescue_team:
        for victim in victims:
            if member.curr_pos[0] == victim.old_pos[0] and member.curr_pos[1] == victim.old_pos[1]:
                member.traj.append(member.curr_pos)
                member.vfd_status_history.append(member.vfd_status)

    rescue_team_traj = []
    vfd_status_list = []
    rescue_team_rew_sum = []
    rescue_team_steps = []
    rescue_team_rew_sum_seen = []
    rescue_team_steps_seen = []
    rescue_team_q = []
    largest = len(rescue_team[0].traj)
    for member in rescue_team:
        if len(member.traj) > largest:
            largest = len(member.traj)
        rescue_team_rew_sum.append(member.rew_sum)
        rescue_team_steps.append(member.steps)
        rescue_team_rew_sum_seen.append(member.rew_sum_seen)
        rescue_team_steps_seen.append(member.steps_seen)
        rescue_team_q.append(member.q)
    for member in rescue_team:
        while len(member.traj) < largest:
            member.traj.append(member.traj[-1])
            member.vfd_status_history.append((member.vfd_status))
        rescue_team_traj.append(member.traj)
        # List of the vfd status
        vfd_status_list.append(member.vfd_status_history)

    victims_traj = []
    for victim in victims:
        while len(victim.traj) < largest:
            victim.traj.append(victim.traj[-1])
        victims_traj.append(victim.traj)
    print(f'This experiment took {time.time() - tic} seconds')
    return (rescue_team_traj,
            rescue_team_rew_sum, rescue_team_steps,
            rescue_team_rew_sum_seen, rescue_team_steps_seen,
            rescue_team_q, victims_traj, vfd_list, vfd_status_list, rescue_team_roles)


if multi_runs:
    # Multi Runs
    rescue_team_rew_sum_run = []
    rescue_team_steps_run = []
    rescue_team_rew_sum_seen_run = []
    rescue_team_steps_seen_run = []
    for run in range(NUM_RUNS):
        print(f'Run {run + 1} of {NUM_RUNS}')
        (rescue_team_traj,
         rescue_team_rew_sum, rescue_team_steps,
         rescue_team_rew_sum_seen, rescue_team_steps_seen,
         rescue_team_q, victims_traj, vfd_list, vfd_status_list, rescue_team_roles) = env(accuracy=1e-7)

        rescue_team_rew_sum_run.append(list(filter(None, rescue_team_rew_sum)))
        rescue_team_steps_run.append(list(filter(None, rescue_team_steps)))
        rescue_team_rew_sum_seen_run.append(list(filter(None, rescue_team_rew_sum_seen)))
        rescue_team_steps_seen_run.append(list(filter(None, rescue_team_steps_seen)))

    rescue_team_rew_sum_run = np.mean(np.asarray(rescue_team_rew_sum_run), axis=0)
    rescue_team_steps_run = np.mean(np.asarray(rescue_team_steps_run), axis=0)
    rescue_team_rew_sum_seen_run = np.mean(np.asarray(rescue_team_rew_sum_seen_run), axis=0)
    rescue_team_steps_seen_run = np.mean(np.asarray(rescue_team_steps_seen_run), axis=0)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}_{str(NUM_RUNS)}Runs.hdf5', 'w') as f:
        for idx, rew_sum in enumerate(rescue_team_rew_sum_run):
            f.create_dataset(f'RS{idx}_reward', data=rew_sum)
        for idx, steps in enumerate(rescue_team_steps_run):
            f.create_dataset(f'RS{idx}_steps', data=steps)
        for idx, rew_sum_seen in enumerate(rescue_team_rew_sum_seen_run):
            f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
        for idx, steps_seen in enumerate(rescue_team_steps_seen_run):
            f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
        f.create_dataset('RS_VFD', data=vfd_list)

else:
    # Single Run
    (rescue_team_traj,
     rescue_team_rew_sum, rescue_team_steps,
     rescue_team_rew_sum_seen, rescue_team_steps_seen,
     rescue_team_q, victims_traj, vfd_list, vfd_status_list, rescue_team_roles) = env(accuracy=1e-7)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}.hdf5', 'w') as f:
        for idx, traj in enumerate(rescue_team_traj):
            f.create_dataset(f'RS{idx}_trajectory', data=traj)
        for idx, vfd_sts in enumerate(vfd_status_list):
            f.create_dataset(f'RS{idx}_VFD_status', data=vfd_sts)
        for idx, rew_sum in enumerate(rescue_team_rew_sum):
            f.create_dataset(f'RS{idx}_reward', data=rew_sum)
        for idx, steps in enumerate(rescue_team_steps):
            f.create_dataset(f'RS{idx}_steps', data=steps)
        for idx, rew_sum_seen in enumerate(rescue_team_rew_sum_seen):
            f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
        for idx, steps_seen in enumerate(rescue_team_steps_seen):
            f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
        for idx, q in enumerate(rescue_team_q):
            f.create_dataset(f'RS{idx}_Q', data=q)
        for idx, victim_traj in enumerate(victims_traj):
            f.create_dataset(f'victim{idx}_trajectory', data=victim_traj)
        f.create_dataset('victims_num', data=[len(victims_traj)])
        f.create_dataset('RS_VFD', data=vfd_list)
        f.create_dataset('RS_ROLES', data=rescue_team_roles)
