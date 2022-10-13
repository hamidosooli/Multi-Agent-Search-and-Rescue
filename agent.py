import numpy as np


class Agent:
    def __init__(self, agent_id, role, vfd, max_vfd, speed, init_pos, num_actions, num_rows, num_cols):
        self.Role = role  # can be 'r': rescuer, 's': scout, 'rs': rescuer and scout, 'v': victim
        self.id = agent_id  # an identification for the agent
        self.VisualField = vfd
        self.max_VisualField = max_vfd

        self.curr_Sensation = [np.nan, np.nan]
        self.old_Sensation = self.curr_Sensation

        self.curr_Index = None
        self.old_Index = self.curr_Index

        self.CanSeeIt = False
        self.Finish = False
        self.Convergence = False
        self.First = True
        self.wereHere = np.ones((num_rows, num_cols))
        self.Speed = speed  # is the number of cells the agent can move in one time-step

        self.init_pos = init_pos
        self.curr_Pos = self.init_pos
        self.old_Pos = self.curr_Pos

        self.Traj = []  # Trajectory of the agent locations
        self.RewHist = []
        self.RewHist_seen = []
        self.RewSum = []  # Keeps track of the rewards in each step
        self.RewSum_seen = []  # Keeps track of the rewards after receiving first data
        self.Steps = []  # Keeps track of the steps in each step
        self.Steps_seen = []  # Keeps track of the steps after receiving first data

        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.t_step_seen = 0
        self.action = None
        self.reward = None
        self.probs = np.nan
        self.Q = np.zeros(((2 * self.max_VisualField + 1) ** 2 + 1, self.num_actions))
        self.Q_hist = self.Q

    def reset(self):
        self.old_Pos = self.init_pos
        self.curr_Pos = self.init_pos
        self.old_Sensation = [np.nan, np.nan]
        self.curr_Sensation = [np.nan, np.nan]
        self.CanSeeIt = False
        self.Finish = False
        self.Convergence = False
        self.First = True
        self.t_step_seen = 0
        self.RewHist = []
        self.RewHist_seen = []
        self.Traj = []
        self.wereHere = np.ones_like(self.wereHere)

    def smart_move(self, idx, wereHere):

        if idx == (2 * self.max_VisualField + 1) ** 2:
            if len(np.argwhere(wereHere)) > 0:
                for loc in np.argwhere(wereHere):
                    if np.sqrt((loc[0] - self.old_Pos[0]) ** 2 + (loc[1] - self.old_Pos[1]) ** 2) == 1:
                        self.curr_Pos = loc
                        self.wereHere[self.curr_Pos[0], self.curr_Pos[1]] = 0
                        break
                    else:
                        continue

    def random_walk(self, idx, pos, speed):
        row_lim = self.num_rows - 1
        col_lim = self.num_cols - 1
        row = pos[0]
        col = pos[1]
        if idx == (2 * self.max_VisualField + 1) ** 2:
            self.action = np.random.randint(self.num_actions)

            if self.action == 0:  # up
                self.curr_Pos = [max(row - speed, 0), col]
            elif self.action == 1:  # down
                self.curr_Pos = [min(row + speed, row_lim), col]
            elif self.action == 2:  # right
                self.curr_Pos = [row, min(col + speed, col_lim)]
            elif self.action == 3:  # left
                self.curr_Pos = [row, max(col - speed, 0)]

    def get_dist(self, pos1, pos2):
        """ takes two 2 element arrays representing positios in the grid
            return the distance between the two positions calculated via pythagorean theorem
        """
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_nearby_location_visits(self, grid_cells):
        """ takes the grid of cells (represented by a 2D numpy array)
            returns a  of the locations (as x,y tuples) which are 1 unit away
            (ie: is UP, DOWN, LEFT, RIGHT of current agent position) together with their
            visit count which is an integer representing the number of times the cell has been visited
        """
        nearby_cell_visits = list()
        for row in range(0, len(grid_cells)):
            for col in range(0, len(grid_cells[row, :])):
                visit_num = grid_cells[row, col]
                loc = [row, col]
                if self.get_dist(loc, self.old_Pos) == 1:
                    loc_visits = [loc, visit_num]
                    nearby_cell_visits.append(loc_visits)
        return nearby_cell_visits

    def get_minimum_visited_cells(self, location_visits):
        """ takes a list of tuples whose elements represent locations in the grid world together
            with their visit counts and returns an array of locations which have the minimum number
            of visits
        """
        min_visits = np.inf  # or any very large number (greater than any expected visit count)
        min_visited_locations = []
        # find the minimum visited number for cells corresponding with the passed locations
        for loc_visits in location_visits:
            times_visited = loc_visits[1]
            if times_visited < min_visits:
                min_visits = times_visited
        # filter the locations corresponding with this minimum visit number
        for loc in location_visits:
            if loc[1] == min_visits:
                min_visited_locations.append(loc)
        return min_visited_locations

    def ant_colony_move(self, cells_visited, idx, env_map):
        """ takes a 2D array representing the visit count for cells in the grid world
            and increments the current agents position toward the least visited neighboring cell
        """
        # increment the cell visit number
        cells_visited[self.old_Pos[0], self.old_Pos[1]] += 1
        if idx == (2 * self.max_VisualField + 1) ** 2:
            nearby_location_visits = self.get_nearby_location_visits(cells_visited)
            least_visited_locations = self.get_minimum_visited_cells(nearby_location_visits)
            # select a random location from the least visit locations nearby
            next_loc_ind = np.random.randint(0, len(least_visited_locations))
            next_loc = least_visited_locations[next_loc_ind][0]
            if env_map[next_loc[0], next_loc[1]] == 0:
                self.curr_Pos = next_loc

    def update_sensation(self, index, raw_sensation, sensation_evaluate, pos2pos, net_adj_mat, adj_mat):

        next_sensation = [np.nan, np.nan]
        self.CanSeeIt = False

        if any(sensation_evaluate[index, :]):
            which_victim = np.argwhere(sensation_evaluate[index, :])[0][0]
            for victim in np.argwhere(sensation_evaluate[index, :])[0]:
                if (np.linalg.norm(raw_sensation[index, victim, :]) <
                    np.linalg.norm(raw_sensation[index, which_victim, :])):
                    which_victim = victim
            next_sensation = raw_sensation[index, which_victim, :]
            self.CanSeeIt = True

        elif not all(np.isnan(net_adj_mat[index, :])):
            temp_sensation = next_sensation.copy()
            num_scouts = np.sum(adj_mat[index, :])
            for ns in range(int(num_scouts)):
                curr_scout = np.argwhere(adj_mat[index, :])[ns]
                if any(sensation_evaluate[curr_scout, :][0].tolist()):
                    which_victim = np.argwhere(sensation_evaluate[curr_scout, :][0])[0]
                    for victim in np.argwhere(sensation_evaluate[curr_scout, :][0]):
                        if (np.linalg.norm(raw_sensation[curr_scout, victim, :]) <
                            np.linalg.norm(raw_sensation[curr_scout, which_victim, :])):
                            which_victim = victim

                    next_sensation[0] = (pos2pos[curr_scout, index][0][0] +
                                         raw_sensation[curr_scout, which_victim, :][0][0])
                    next_sensation[1] = (pos2pos[curr_scout, index][0][1] +
                                         raw_sensation[curr_scout, which_victim, :][0][1])
                    self.CanSeeIt = True

                    if np.linalg.norm(temp_sensation) < np.linalg.norm(next_sensation):
                        next_sensation = temp_sensation.copy()
                    else:
                        temp_sensation = next_sensation.copy()

        return next_sensation

    def sensation2index(self, sensation, max_vfd):
        if self.CanSeeIt:
            index = ((sensation[0] + max_vfd) * (2 * max_vfd + 1) + (sensation[1] + max_vfd))
        else:
            index = (2 * max_vfd + 1) ** 2

        return int(index)

    def rescue_accomplished(self, rescue_team_Hist, agent, adj_mat):
        if (((self.old_Sensation[0] == 0 and self.old_Sensation[1] == 0) or
             (self.curr_Sensation[0] == 0 and self.curr_Sensation[1] == 0)) and
                'r' in self.Role):
            self.Finish = True
            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 0)
            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 1)

            rescue_team_Hist.remove(agent)

        return rescue_team_Hist, adj_mat

    def victim_rescued(self, rescue_team_old_pos_list, rescue_team_curr_pos_list,
                       rescue_team_role_list, victim, victims_Hist):
        for idx, rescuer_old_pos in enumerate(rescue_team_old_pos_list):
            if (((rescuer_old_pos[0] == self.old_Pos[0] and
                  rescuer_old_pos[1] == self.old_Pos[1]) or
                 (rescue_team_curr_pos_list[idx][0] == self.old_Pos[0] and
                  rescue_team_curr_pos_list[idx][1] == self.old_Pos[1])) and
                    'r' in rescue_team_role_list[idx]):
                self.Finish = True
                victims_Hist.remove(victim)
                break  # You already removed this victim, no need to check the rest of the list

        return victims_Hist

    def convergence_check(self, accuracy):

        if (np.abs(np.sum(self.Q - self.Q_hist) /
                   (np.shape(self.Q)[0] * np.shape(self.Q)[1])) <= accuracy):
            self.Convergence = True

        return self.Convergence
