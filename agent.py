import numpy as np
from search_algorithms import SearchAlgorithms
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent(SearchAlgorithms):
    def __init__(self, agent_id, init_role, vfd, max_vfd, speed, init_pos, num_actions, num_rows, num_cols, LR=1):
        super(Agent, self).__init__(max_vfd, init_pos, num_actions, num_rows, num_cols)
        self.init_role = init_role
        self.role = self.init_role  # can be 'r': rescuer, 's': scout, 'rs': rescuer and scout, 'v': victim
        self.id = agent_id  # an identification for the agent
        self.visual_field = vfd
        self.max_visual_field = max_vfd
        self.vfd_status = np.ones((2 * self.visual_field + 1, 2 * self.visual_field + 1), dtype=bool)
        self.savior = None
        self.curr_sensation = [np.nan, np.nan]
        self.old_sensation = self.curr_sensation

        self.curr_dist2home = [0, 0]
        self.old_dist2home = self.curr_dist2home

        self.curr_index = None
        self.old_index = self.curr_index

        self.perceived = False
        self.saved = False
        self.busy = False
        self.first = True
        self.were_here = np.ones((num_rows, num_cols))
        self.competency_map = np.zeros((num_rows, num_cols))
        self.competency_profile = np.ones((num_rows, num_cols))
        self.competency = 1.0  # initially the agent is 100% competent
        self.speed = speed  # is the number of cells the agent can move in one time-step

        self.init_pos = init_pos
        self.curr_pos = self.init_pos
        self.old_pos = self.curr_pos

        self.traj = []  # Trajectory of the agent locations
        self.vfd_status_history = []
        self.rew_hist = []
        self.rew_hist_seen = []
        self.rew_sum = []  # Keeps track of the rewards in each step
        self.rew_sum_seen = []  # Keeps track of the rewards after receiving first data
        self.steps = []  # Keeps track of the steps in each step
        self.steps_seen = []  # Keeps track of the steps after receiving first data

        self.num_actions = num_actions
        self.num_observations = 2*((2 * self.max_visual_field + 1) ** 2) + 1  # catch, rescue, search
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.t_step_seen = 0
        self.action = None
        self.reward = None
        self.tree_status = False
        self.team_reward = 0
        self.probs = np.nan
        self.q = np.zeros((self.num_observations, self.num_actions))  # q table for q-learning
        self.q_hist = self.q
        self.policy_net = DQN(self.num_observations, self.num_actions).to(device)
        self.target_net = DQN(self.num_observations, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def reset(self):
        self.old_pos = self.init_pos
        self.curr_pos = self.init_pos
        self.role = self.init_role
        self.old_sensation = [np.nan, np.nan]
        self.curr_sensation = [np.nan, np.nan]

        self.curr_dist2home = [0, 0]
        self.old_dist2home = [0, 0]
        self.savior = None
        self.perceived = False
        self.saved = False
        self.busy = False
        self.convergence = False
        self.first = True
        self.t_step_seen = 0
        self.rew_hist = []
        self.rew_hist_seen = []
        self.traj = []
        self.vfd_status_history = []
        self.were_here = np.ones_like(self.were_here)
        self.competency_map = np.zeros_like(self.competency_map)
        self.competency_profile = np.ones_like(self.competency_profile)
        self.competency = 1.0
        self.vfd_status = np.ones((2 * self.visual_field + 1, 2 * self.visual_field + 1), dtype=bool)

    def update_vfd(self, env_map):
        # rescuer visual field depth
        self.vfd_status = np.ones((2 * self.visual_field + 1, 2 * self.visual_field + 1), dtype=bool)
        vfd_j = 0
        for j in range(int(max(self.old_pos[1] - self.visual_field, 0)),
                       int(min(self.num_cols, self.old_pos[1] + self.visual_field + 1))):
            vfd_i = 0
            for i in range(int(max(self.old_pos[0] - self.visual_field, 0)),
                           int(min(self.num_rows, self.old_pos[0] + self.visual_field + 1))):
                if env_map[i, j] == 1:
                    self.vfd_status[vfd_i, vfd_j] = False
                    # Down or Up
                    if i - self.old_pos[0] > 0 and j - self.old_pos[1] == 0:
                        self.vfd_status[vfd_i:, vfd_j] = False
                    elif i - self.old_pos[0] < 0 and j - self.old_pos[1] == 0:
                        self.vfd_status[:min(vfd_i+1, self.visual_field), vfd_j] = False
                    # Right or Left
                    elif i - self.old_pos[0] == 0 and j - self.old_pos[1] > 0:
                        self.vfd_status[vfd_i, vfd_j:] = False
                    elif i - self.old_pos[0] == 0 and j - self.old_pos[1] < 0:
                        self.vfd_status[vfd_i, :min(vfd_j+1, self.visual_field)] = False
                    # Northeast or Northwest
                    elif i - self.old_pos[0] < 0 and j - self.old_pos[1] > 0:
                        self.vfd_status[:min(vfd_i+1, self.visual_field), vfd_j:] = False
                    elif i - self.old_pos[0] < 0 and j - self.old_pos[1] < 0:
                        self.vfd_status[:min(vfd_i+1, self.visual_field), :min(vfd_j+1, self.visual_field)] = False
                    # Southeast or Southwest
                    elif i - self.old_pos[0] > 0 and j - self.old_pos[1] > 0:
                        self.vfd_status[vfd_i:, vfd_j:] = False
                    elif i - self.old_pos[0] > 0 and j - self.old_pos[1] < 0:
                        self.vfd_status[vfd_i:, :min(vfd_j+1, self.visual_field)] = False

                vfd_i += 1
            vfd_j += 1

    def update_dist2home(self):
        self.old_dist2home = np.subtract(self.init_pos, self.old_pos)
        self.curr_dist2home = np.subtract(self.init_pos, self.curr_pos)

    def update_sensation(self, index, raw_sensation, sensation_evaluate, pos2pos, net_adj_mat, adj_mat):
        next_sensation = [np.nan, np.nan]
        self.perceived = False
        which_victim = None
        if any(sensation_evaluate[index, :]):
            which_victim = np.argwhere(sensation_evaluate[index, :])[0][0]
            for victim in np.argwhere(sensation_evaluate[index, :])[0]:
                if (np.linalg.norm(raw_sensation[index, victim, :]) <
                    np.linalg.norm(raw_sensation[index, which_victim, :])):
                    which_victim = victim
            next_sensation = raw_sensation[index, which_victim, :]
            self.perceived = True

        elif not all(np.isnan(net_adj_mat[index, :])):
            temp_sensation = next_sensation.copy()
            num_scouts = np.sum(adj_mat[index, :])
            for ns in range(int(num_scouts)):
                curr_scout = np.argwhere(adj_mat[index, :])[ns]
                if any(sensation_evaluate[curr_scout, :][0].tolist()):
                    which_victim = np.argwhere(sensation_evaluate[curr_scout, :][0])[0]
                    # if you see more than one victim go for the nearest
                    for victim in np.argwhere(sensation_evaluate[curr_scout, :][0]):
                        if (np.linalg.norm(raw_sensation[curr_scout, victim, :]) <
                            np.linalg.norm(raw_sensation[curr_scout, which_victim, :])):
                            which_victim = victim

                    next_sensation[0] = (pos2pos[curr_scout, index][0][0] +
                                         raw_sensation[curr_scout, which_victim, :][0][0])
                    next_sensation[1] = (pos2pos[curr_scout, index][0][1] +
                                         raw_sensation[curr_scout, which_victim, :][0][1])
                    self.perceived = True

                    if np.linalg.norm(temp_sensation) < np.linalg.norm(next_sensation):
                        next_sensation = temp_sensation.copy()
                    else:
                        temp_sensation = next_sensation.copy()

        return next_sensation, which_victim

    def sensation2index(self, sensation, dist2home):
        if self.perceived and not self.busy:
            index = ((sensation[0] + self.max_visual_field) * (2 * self.max_visual_field + 1) +
                     (sensation[1] + self.max_visual_field))
        elif self.busy:
            index = (((2 * self.max_visual_field + 1) ** 2) +
                     ((dist2home[0] + self.max_visual_field) * (2 * self.max_visual_field + 1) +
                      (dist2home[1] + self.max_visual_field)))
        else:
            index = 2*((2 * self.max_visual_field + 1) ** 2)

        return int(index)

    def rescue_started(self, rescue_team_hist, agent, adj_mat):
        if (self.curr_sensation[0] == 0 and self.curr_sensation[1] == 0) and 'r' in self.role and not self.busy:
            self.busy = True  # set the robot mode to rescue
            self.role = 's'  # while in rescue mode robot serves as a scout for other teammates
            adj_mat[rescue_team_hist.index(agent), :] = 0  # while rescuing a victim stop receiving info from others

        return rescue_team_hist, adj_mat

    def rescue_accomplished(self, adj_mat, original_adj_mat):
        if (self.curr_dist2home[0] == 0 and self.curr_dist2home[1] == 0) and self.busy:
            self.busy = False
            self.role = 'r'
            return original_adj_mat
        else:
            return adj_mat

    def victim_rescued(self, rescue_team_old_pos_list, rescue_team_curr_pos_list,
                       muster_site,
                       rescue_team_role_list, victim, victims_hist):
        for idx, rescuer_old_pos in enumerate(rescue_team_old_pos_list):
            rescuer_found_you = ((rescuer_old_pos[0] == self.old_pos[0] and
                                  rescuer_old_pos[1] == self.old_pos[1]) or
                                 (rescue_team_curr_pos_list[idx][0] == self.old_pos[0] and
                                  rescue_team_curr_pos_list[idx][1] == self.old_pos[1]))

            if rescuer_found_you and 'r' in rescue_team_role_list[idx]:
                self.savior = idx
            # if self.savior is not None:
                if self.saved:
                    break  # You already removed this victim, no need to check the rest of the list
                elif (self.old_pos[0] == muster_site[self.savior][0] and
                        self.old_pos[1] == muster_site[self.savior][1]):
                    self.saved = True
                    victims_hist.remove(victim)

        return victims_hist
