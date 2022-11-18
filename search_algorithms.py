import numpy as np


class SearchAlgorithms:
    def __init__(self, max_vfd, init_pos, num_actions, num_rows, num_cols):
        self.max_VisualField = max_vfd
        self.wereHere = np.ones((num_rows, num_cols))

        self.init_pos = init_pos
        self.curr_Pos = self.init_pos
        self.old_Pos = self.curr_Pos

        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols

    def smart_move(self, idx, wereHere):
        '''
        takes index to see if thee agent is in search mode
        wereHere is a matrix that tracks the visited cells
        '''
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
        '''
        takes current location and its relevant index in the Q table
        as well as agents speed
        If the index is for the search mode,
         it randomly selects one of the 4 possible directions
         '''
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