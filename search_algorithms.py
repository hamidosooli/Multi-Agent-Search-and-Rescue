from scipy.stats import wrapcauchy, levy_stable
import numpy as np
import torch

class SearchAlgorithms:
    def __init__(self, max_vfd, init_pos, num_actions, num_rows, num_cols):
        self.action = None
        self.max_VisualField = max_vfd
        self.wereHere = np.ones((num_rows, num_cols))

        self.init_pos = init_pos
        self.curr_pos = self.init_pos
        self.old_pos = self.curr_pos

        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.up, self.down, self.left, self.right = (-1, 0),  (1, 0), (0, -1), (0, 1)
        self.actions = [self.up,  self.down,  self.left, self.right]
        # Levy Parameters
        self.prev_action = None
        self.alpha = 0.8
        self.beta = 0
        self.scale_to_var = 1/(np.sqrt(2))
        self.levy_dist = levy_stable(self.alpha, self.beta, scale=self.scale_to_var)
        self.decision_wait_time = 0

        # Turning Angle Parameters
        self.rho = 0.1  # 0 is completely fair 1 is biased to previous action
        self.wrap_dist = wrapcauchy(self.rho)
        self.curr_angle = np.random.random()*np.pi*2
        self.dest_comb = 20
        self.path_to_take = None
        self.path_taken_counter = 0

        self.last_index = 2 * ((2 * max_vfd - 1) * (2 * max_vfd + 1) + (2 * max_vfd - 1)) + 1

    def random_walk(self, idx, pos, speed, env_map, busy_agent, device):
        """
        takes current location and its relevant index in the Q table
        as well as agents speed
        If the index is for the search mode,
         it randomly selects one of the 4 possible directions
         """
        if idx != self.last_index or busy_agent:
            return
        row_lim = self.num_rows - 1
        col_lim = self.num_cols - 1
        row = pos[0]
        col = pos[1]

        self.action = torch.tensor([[np.random.randint(self.num_actions)]], device=device, dtype=torch.long)

        if self.action.item() == 0:  # up
            next_loc = [max(row - speed, 0), col]
        elif self.action.item() == 1:  # down
            next_loc = [min(row + speed, row_lim), col]
        elif self.action.item() == 2:  # right
            next_loc = [row, min(col + speed, col_lim)]
        elif self.action.item() == 3:  # left
            next_loc = [row, max(col - speed, 0)]

        if env_map[next_loc[0], next_loc[1]] == 0:
            self.curr_pos = next_loc

    def straight_move(self, idx, were_here, env_map, busy_agent):
        """
        takes index to see if the agent is in search mode
        wereHere is a matrix that tracks the visited cells
        """
        if idx == self.last_index and not busy_agent:
            if len(np.argwhere(were_here)) > 0:
                for loc in np.argwhere(were_here):
                    if np.sqrt((loc[0] - self.old_pos[0]) ** 2 + (loc[1] - self.old_pos[1]) ** 2) == 1:
                        next_loc = loc
                        self.wereHere[self.curr_pos[0], self.curr_pos[1]] = 0

                        if env_map[next_loc[0], next_loc[1]] == 0:
                            self.curr_pos = next_loc
                        break
                    else:
                        continue
            else:
                self.wereHere = np.ones_like(self.wereHere)

    # The following method was developed by Fernando Mazzoni
    def account_for_boundary(self):
        row, col = self.curr_pos
        actions = self.actions.copy()

        if row == 0:
            actions[0] = actions[1]
        elif row == self.num_rows - 1:
            actions[1] = actions[0]
        if col == 0:
            actions[2] = actions[3]
        elif col == self.num_cols - 1:
            actions[3] = actions[2]

        check_cells = [[drow+row, dcol+col] for drow, dcol in [list(action) for action in actions]]

        for n, cell_ in enumerate(check_cells):
            m = [1, 0, 3, 2]  # get the opposite action
            if ((cell_[0] == 0 or cell_[0] == self.num_cols - 1) or
                (cell_[1] == 0 or cell_[1] == self.num_rows - 1)):
                actions[n] = actions[m[n]]

        return actions

    def levy_walk(self, env_map, busy_agent):
        """
        Randomly gets next step based on set of actions.
        Boundary conditions are reflective
        """
        if busy_agent:
            return
        prev_action = self.prev_action

        # Levy Walk each decision step continues otherwise repeat last action
        if prev_action is None or self.decision_wait_time == 0:
            r = self.levy_dist.rvs()
            r = np.round(np.abs(r))
            self.decision_wait_time = int(r)

            r_angle = self.wrap_dist.rvs()
            self.curr_angle = (self.curr_angle + r_angle) % (2*np.pi)
            px = np.cos(self.curr_angle)
            py = np.sin(self.curr_angle)

            possible_actions = [[int(-1*int(np.sign(py))), 0], [0, int(np.sign(px))]]
            px1 = abs(px)
            py1 = abs(py)
            self.path_to_take = [possible_actions[i] for i in np.random.choice([0, 1],
                                                                               size=self.dest_comb,
                                                                               p=[py1**2, px1**2])]
        else:
            self.decision_wait_time -= 1

        actions_in_boundary = self.account_for_boundary()
        desired_action = self.path_to_take[self.path_taken_counter]
        self.path_taken_counter = (1+self.path_taken_counter) % self.dest_comb

        if desired_action in actions_in_boundary:
            action = desired_action
        else:
            if np.abs(desired_action[1]) == 1:
                dtheta = self.curr_angle - np.pi/2
                if dtheta > 0:
                    self.curr_angle = np.pi/2 - dtheta
                else:
                    self.curr_angle = np.pi/2 + np.abs(dtheta)
            elif np.abs(desired_action[0]) == 1:
                dtheta = self.curr_angle - np.pi
                if dtheta > 0:
                    self.curr_angle = np.pi - dtheta
                else:
                    self.curr_angle = np.pi + np.abs(dtheta)
            self.path_to_take = [(int(desired_action[0]*-1), int(desired_action[1]*-1))
                                 if x == desired_action else x for x in self.path_to_take]

            action = [int(desired_action[0]*-1), int(desired_action[1]*-1)]
        row, col = np.shape(env_map)
        next_loc = [min(max(self.curr_pos[0] + action[0], 0), row-1), min(max(self.curr_pos[1] + action[1], 0), col-1)]
        self.prev_action = action
        if env_map[next_loc[0], next_loc[1]] == 0:
            self.curr_pos = next_loc

    def ant_colony(self, cells_visited, idx, env_map, busy_agent, num_acts=4, break_ties=True):
        if idx != self.last_index or busy_agent:
            return cells_visited

        x, y = np.add(self.old_pos, [1, 1])

        # boundaries
        left = y-1
        right = y+1
        top = x-1
        down = x+1
        env_map_copy = env_map.astype(float)
        env_map_copy[env_map_copy == 1] = np.nan
        env_map_copy = np.pad(env_map_copy, 1, mode='constant', constant_values=np.nan)
        decision_map = cells_visited[top:down+1, left:right+1] + env_map_copy[top:down+1, left:right+1]
        decision_map_shape = np.shape(decision_map)
        # actions
        stay = [0, 0]
        go_right = [0, 1]
        go_left = [0, -1]
        go_up = [-1, 0]
        go_down = [1, 0]

        go_northeast = [-1, 1]
        go_northwest = [-1, -1]
        go_southeast = [1, 1]
        go_southwest = [1, -1]
        actions_map = np.array([[go_northwest, go_up, go_northeast],
                                [go_left, stay, go_right],
                                [go_southwest, go_down, go_southeast]])

        ignore_pattern = np.array([[np.nan, 0, np.nan],
                                   [0, np.nan, 0],
                                   [np.nan, 0, np.nan]])
        if num_acts == 8:
            pass
        elif num_acts == 4:
            decision_map = np.add(decision_map, ignore_pattern)
        else:
            raise ValueError('Numer of actions must be 4 or 8.')
        decision_map_min_values = np.nanmin(decision_map)

        # use the following to break the ties between min values
        if break_ties:
            min_visit_flat_index = np.random.choice(np.flatnonzero(decision_map == decision_map_min_values))
            min_visit_index = np.unravel_index(min_visit_flat_index, decision_map_shape)
        else:
            min_visit_flat_index = np.flatnonzero(decision_map == decision_map_min_values)
            min_visit_index = np.unravel_index(min_visit_flat_index[0], decision_map_shape)

        action = actions_map[min_visit_index[0], min_visit_index[1]]
        next_pos = np.add(self.old_pos, action)

        if env_map[next_pos[0], next_pos[1]] == 0:
            cells_visited[next_pos[0]+1, next_pos[1]+1] += 1
            self.curr_pos = next_pos
        else:
            cells_visited[self.old_pos[0]+1, self.old_pos[1]+1] += 1
            self.curr_pos = self.old_pos
        return cells_visited