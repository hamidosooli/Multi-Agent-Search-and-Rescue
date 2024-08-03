import time
import numpy as np
import h5py
from network import Network
from agent_dqn import Agent
from tqdm import tqdm

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from gridworld_multi_agent1_1 import training_animate
import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    NUM_EPISODES = 300
else:
    NUM_EPISODES = 300
NUM_STEPS = 3000
NUM_RUNS = 100
multi_runs = False
# actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
actions = [FORWARD, BACKWARD, RIGHT, LEFT]
num_acts = len(actions)


#                          r1  r2  r3  r4
adj_mat_prior = np.array([[1,  0],
                          [0,  1]], dtype=float)

exp_name = f'{len(adj_mat_prior)}R_{NUM_EPISODES}episodes_dqn'
NUM_VICTIMS = 10

global env_map
global env_mat
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
# env_map = np.zeros((10, 10))
# env_map = np.array([[0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [1, 0, 1, 0, 1],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0]])
# Environment dimensions
NUM_ROWS, NUM_COLS = np.shape(env_map)
row_lim = NUM_ROWS - 1
col_lim = NUM_COLS - 1
env_mat = np.zeros_like(env_map, dtype=float)
env_mat = np.pad(env_mat, 1, mode='constant', constant_values=np.nan)
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
LR = 1e-3


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


# def reward_func_ql(sensation_prime, dist2home, busy):
#     perceived = (sensation_prime[0] == 0 and sensation_prime[1] == 0)
#     saved = (dist2home[0] == 0 and dist2home[1] == 0)
#     if perceived and not busy and not saved:
#         re = 1
#     elif saved and busy:
#         re = 1
#     else:
#         re = -.1
#
#     return re
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

# Function to print gradients

def select_action(state, agent, num_actions, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.tensor([[agent.policy_net(state).argmax()]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[random.randint(0, num_actions - 1)]], device=device, dtype=torch.long)

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

    plt.pause(0.00001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def env(accuracy=1e-15):
    print(f'Training started with {"gpu" if device.type=="cuda" else "cpu"}')
    global adj_mat_prior, env_mat
    # Define the Network and the agent objects
    network = Network
    agent = Agent

    r0 = agent(0, 'r', 3, NUM_ROWS, 1, [0, 7], num_acts, NUM_ROWS, NUM_COLS, LR)
    # r1 = agent(1, 'r', 3, NUM_ROWS, 1, [0, 8], num_acts, NUM_ROWS, NUM_COLS, LR)
    # r2 = agent(2, 'r', 3, NUM_ROWS, 1, [15, 7], num_acts, NUM_ROWS, NUM_COLS, LR)
    r3 = agent(1, 'r', 3, NUM_ROWS, 1, [15, 8], num_acts, NUM_ROWS, NUM_COLS, LR)
    # List of rescue team members
    rescue_team = [r0, r3]

    # Define the victims
    victim_locs = np.argwhere(env_map == 0).tolist()  # list of non-occupied locations in the map
    for member in rescue_team:
        victim_locs.remove(member.init_pos)  # remove the location of the rescue team members from the list

    victims = []  # make a list of the victims
    for victim_id in range(NUM_VICTIMS):
        loc = victim_locs[np.random.randint(len(victim_locs))]  # find a random non-occupied location
        victims.append(agent(victim_id, 'v', 0, 0, 1, loc, num_acts, NUM_ROWS, NUM_COLS, LR))  # initialize the victim
        victim_locs.remove(loc)  # remove the location of the previous victim from the list
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
    tic = time.time()
    # animate = training_animate(len(rescue_team), len(victims), Col_num=NUM_COLS, Row_num=NUM_ROWS)
    # while True:
    for eps in tqdm(range(NUM_EPISODES)):
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
        test = []
        for _ in range(NUM_STEPS):
        # while True:
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

            rescue_team_vfd_list = np.asarray(rescue_team_vfd_list)

            # Keep track of the victims positions
            # Make a list of the victims old positions
            victims_old_pos_list = []
            for victim in victims_hist:
                victim.traj.append(victim.old_pos)
                if victim.savior is None:
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
                member.old_state_vect = np.zeros((member.num_observations,))
                member.old_state_vect[member.old_index] = 1
                member.old_state_vect = torch.tensor(member.old_state_vect,
                                                     dtype=torch.float32, device=device).unsqueeze(0)
                # actions for the rescue team
                member.action = select_action(member.old_state_vect, member, member.num_actions, t_step)
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
                member.curr_pos = movement(member.old_pos, rescue_team_old_pos_list, rescue_team_curr_pos_list,
                                           member.action, member.speed)

            # Search algorithm
            # env_mat = r0.ant_colony(env_mat, r0.old_index, env_map, r0.busy, break_ties=False)
            # r1.random_walk(r1.old_index, r1.old_pos, r1.speed, env_map, r1.busy, device)
            # env_mat = r2.ant_colony(env_mat, r2.old_index, env_map, r2.busy, break_ties=True)
            # r3.straight_move(r3.old_index, r3.were_here, env_map, r3.busy)

            env_mat = r0.ant_colony(env_mat, r0.old_index, env_map, r0.busy, break_ties=True)
            # env_mat = r1.ant_colony(env_mat, r1.old_index, env_map, r1.busy, break_ties=True)
            # env_mat = r2.ant_colony(env_mat, r2.old_index, env_map, r2.busy, break_ties=True)
            env_mat = r3.ant_colony(env_mat, r3.old_index, env_map, r3.busy, break_ties=True)

            for member in rescue_team_hist:
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
                member.curr_state_vect = np.zeros((member.num_observations,))
                member.curr_state_vect[member.curr_index] = 1
                member.curr_state_vect = torch.tensor(member.curr_state_vect,
                                                      dtype=torch.float32, device=device).unsqueeze(0)
                # Rewarding the rescue team
                member.reward = reward_func_ql(member.curr_sensation, member.curr_dist2home, member.busy)
                member.reward = torch.tensor([member.reward], device=device)

                if curr_seen_victim != None:
                    seen_victim.append(curr_seen_victim)

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
                rescue_team_hist, adj_mat = member.rescue_started(rescue_team_hist, member, adj_mat)
                adj_mat = member.rescue_accomplished(adj_mat, adj_mat_prior)
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
            # test.append(torch.sum(torch.tensor(rescue_team_hist[0].rew_hist, device=device)).item())
            # plt.plot(test, color='b')
            # plt.pause(.0001)
            if len(victims_hist) == 0:
                episode_durations.append(t_step + 1)
                # plot_durations()
                # print(f'In episode {eps+1}, all of the victims were rescued in {t_step} steps')
                break
            # Update the rescue team positions, store the transitions in memory, perform one step of optimization
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
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
                member.target_net.load_state_dict(target_net_state_dict)
            # plt.plot(rescue_team_hist[0].loss, color='b')
            # plt.pause(.00001)
            victims_curr_pos_list = []
            # Victims' actions and positions
            for victim in victims_hist:
                # actions for the victims
                victim.action = np.random.choice(actions)
                # Victims next positions
                if victim.savior is None:
                    victims_curr_pos_list.append(victim.curr_pos)
                    victim.curr_pos = movement(victim.old_pos, victims_old_pos_list,
                                               np.asarray(victims_curr_pos_list), victim.action, victim.speed)
                if (victim.savior is not None) and (not victim.saved):
                    victim.old_pos = rescue_team_hist[victim.savior].curr_pos
                elif victim.id not in seen_victim:
                    # Update the victims position
                    victim.old_pos = victim.curr_pos


            # animate.animate([[member.old_pos] for member in rescue_team], [[victim.old_pos] for victim in victims],
            #                  [member.visual_field for member in rescue_team],
            #                  [member.vfd_status for member in rescue_team],
            #                  [member.role for member in rescue_team],
            #                  [member.curr_sensation for member in rescue_team],
            #                  [np.round(member.reward.item(), decimals=2) for member in rescue_team],
            #                  ['S' if member.curr_index == member.last_index else 'R' for member in rescue_team],
            #                  env_map, wait_time=0.1)
            # print([member.role for member in rescue_team])
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
    rescue_team_policy_nets = []
    rescue_team_target_nets = []
    largest = len(rescue_team[0].traj)
    for member in rescue_team:
        if len(member.traj) > largest:
            largest = len(member.traj)
        rescue_team_rew_sum.append(member.rew_sum)
        rescue_team_steps.append(member.steps)
        rescue_team_rew_sum_seen.append(member.rew_sum_seen)
        rescue_team_steps_seen.append(member.steps_seen)
        rescue_team_q.append(member.q)
        rescue_team_policy_nets.append(member.policy_net.state_dict())
        rescue_team_target_nets.append(member.target_net.state_dict())
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
            rescue_team_rew_sum_seen, rescue_team_steps_seen, rescue_team_policy_nets, rescue_team_target_nets,
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
         rescue_team_rew_sum_seen, rescue_team_steps_seen, rescue_team_policy_nets, rescue_team_target_nets,
         rescue_team_q, victims_traj, vfd_list, vfd_status_list, rescue_team_roles) = env(accuracy=1e-7)

        rescue_team_rew_sum_run.append(list(filter(None, rescue_team_rew_sum)))
        rescue_team_steps_run.append(list(filter(None, rescue_team_steps)))
        rescue_team_rew_sum_seen_run.append(list(filter(None, rescue_team_rew_sum_seen)))
        rescue_team_steps_seen_run.append(list(filter(None, rescue_team_steps_seen)))

    rescue_team_rew_sum_run = np.mean(np.asarray(rescue_team_rew_sum_run), axis=0)
    rescue_team_steps_run = np.mean(np.asarray(rescue_team_steps_run), axis=0)
    rescue_team_rew_sum_seen_run = np.mean(np.asarray(rescue_team_rew_sum_seen_run), axis=0)
    rescue_team_steps_seen_run = np.mean(np.asarray(rescue_team_steps_seen_run), axis=0)

    with h5py.File(f'{exp_name}_{str(NUM_RUNS)}Runs.hdf5', 'w') as f:
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

    for numrun in range(NUM_RUNS):
        # Single Run
        (rescue_team_traj,
         rescue_team_rew_sum, rescue_team_steps,
         rescue_team_rew_sum_seen, rescue_team_steps_seen, rescue_team_policy_nets, rescue_team_target_nets,
         rescue_team_q, victims_traj, vfd_list, vfd_status_list, rescue_team_roles) = env(accuracy=1e-7)

        with h5py.File(f'{exp_name}_{numrun}.hdf5', 'w') as f:
            f.create_dataset(f'RS0_reward', data=rescue_team_rew_sum[0])
            f.create_dataset(f'RS0_steps', data=rescue_team_rew_sum_seen[0])
            f.create_dataset(f'RS0_reward_seen', data=rescue_team_steps[0])
            f.create_dataset(f'RS0_steps_seen', data=rescue_team_steps_seen[0])

            f.create_dataset(f'RS1_reward', data=rescue_team_rew_sum[1])
            f.create_dataset(f'RS1_steps', data=rescue_team_rew_sum_seen[1])
            f.create_dataset(f'RS1_reward_seen', data=rescue_team_steps[1])
            f.create_dataset(f'RS1_steps_seen', data=rescue_team_steps_seen[1])

    # with h5py.File(f'{exp_name}.hdf5', 'w') as f:
    #     for idx, traj in enumerate(rescue_team_traj):
    #         f.create_dataset(f'RS{idx}_trajectory', data=traj)
    #     for idx, vfd_sts in enumerate(vfd_status_list):
    #         f.create_dataset(f'RS{idx}_VFD_status', data=vfd_sts)
    #     for idx, rew_sum in enumerate(rescue_team_rew_sum):
    #         f.create_dataset(f'RS{idx}_reward', data=rew_sum)
    #     for idx, steps in enumerate(rescue_team_steps):
    #         f.create_dataset(f'RS{idx}_steps', data=steps)
    #     for idx, rew_sum_seen in enumerate(rescue_team_rew_sum_seen):
    #         f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
    #     for idx, steps_seen in enumerate(rescue_team_steps_seen):
    #         f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
    #     for idx, q in enumerate(rescue_team_q):
    #         f.create_dataset(f'RS{idx}_Q', data=q)
    #     for idx, victim_traj in enumerate(victims_traj):
    #         f.create_dataset(f'victim{idx}_trajectory', data=victim_traj)
    #     f.create_dataset('victims_num', data=[len(victims_traj)])
    #     f.create_dataset('RS_VFD', data=vfd_list)
    #     f.create_dataset('RS_ROLES', data=rescue_team_roles)
    #
    # for idx, pnet in enumerate(rescue_team_policy_nets):
    #     torch.save(pnet, f'{exp_name}_RS{idx}_policy_net.pt')
    # for idx, tnet in enumerate(rescue_team_target_nets):
    #     torch.save(tnet, f'{exp_name}_RS{idx}_target_net.pt')