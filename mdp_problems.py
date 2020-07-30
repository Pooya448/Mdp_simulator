from utilities import Actions, initialize_mdp_parameters
from copy import deepcopy

class MDPProblem:
    """

    :param grid_dim: a tuple of (height, width) which declares dimensions of the world grid.
    :param exit_locations: a dictionary with exit states as keys and rewards as values.
    example given: self.exit_locations[(0, 2)] = -1 or self.exit_locations = {(0, 2): -1, ...}


    """

    def __init__(self, grid_dim, exit_locations):
        self.grid_dim = grid_dim
        self.exit_locations = exit_locations

    def compute_policy(self, reward=-0.01, gama=1, steps=10):
        """

        :param reward: reward of moving from one cell to another. (Living reward)
        :param gama: Discount coefficient
        :param steps: depth of computation. (How many turns agent can play)
        :return:
        1-2D grid of computed V*_k(s) after each step.
        Example Given for 3x3 world after some steps.
       [ 0     0.8  1
       -0.02 -0.1 -1
        0   -0.02  0 ]
        2- A 2D grid of computed Policies. (same as v_states but filled with Actions class instances.)
        a naive policy:
      [ Actions.N Actions.N Actions.EXIT
        Actions.N Actions.N Actions.EXIT
        Actions.N Actions.N Actions.N ]
        """

        width, height = self.grid_dim

        # Use pre_v_states for keeping previous V states. (former iteration)
        v_states, pre_v_states, policy = initialize_mdp_parameters(width, height, self.exit_locations)
        for row in v_states:
            print(*row)
        print('******************')
        for row in pre_v_states:
            print(*row)
        print('******************')
        for row in policy:
            print(*row)
        print('******************')
        prev_policy = deepcopy(policy)
        actions = [Actions.N, Actions.S, Actions.E, Actions.W]

        for k in range(0, steps):
            for i in range(0, width):
                for j in range(0, height):
                    if (j, i) in self.exit_locations:
                        policy[j][i] = Actions.EXIT
                        continue
                    else:
                        max_sum = -10000
                        optimal_policy = Actions.N
                        for action in actions:
                            sum = 0
                            for x, y, p in self.get_transition((j, i), action):
                                # if (x, y) in self.exit_locations:
                                #     sum += p * (self.exit_locations[(x, y)] + gama * pre_v_states[x][y])
                                # else:
                                sum += p * (reward + gama * pre_v_states[x][y])
                            if sum >= max_sum:
                                max_sum = sum
                                optimal_policy = action
                        v_states[j][i] = max_sum
                        policy[j][i] = optimal_policy
            print("************************ ", k, " *********************")
            if self.convergence_policy(policy,prev_policy):
                print("Policy Converged\n")
            if self.convergence(v_states,pre_v_states):
                print("Values Converged\n")


            pre_v_states = deepcopy(v_states)
            prev_policy = deepcopy(policy)
            # DO NOT CHANGE yield Line. You should return V and Pi computed in each step.
            yield v_states, policy

    def convergence(self, v_states, v_prev):

        diff = 0
        x , y = self.grid_dim
        for i in range(x):
            for j in range(y):
                if abs(v_states[i][j] - v_prev[i][j]) >= 0.05:
                    diff += 1
        if diff == 0:
            return True
        return False

    def convergence_policy(self, policy, policy_prev):

        diff = 0
        x , y = self.grid_dim
        for i in range(x):
            for j in range(y):
                if policy[i][j] != policy_prev[i][j]:
                    diff += 1
        if diff == 0:
            return True
        return False
    def get_transition(self, state, action):
        """

        :param state: a tuple of (x, y) as dimensions
        :param action: object of Actions enum class. (such as:
        Actions.N)
        :return: given current state and chosen action, returns next non-determinist states with
        corresponding transition probabilities. example given: [(x, y, 0.8), (z, t, 0,2), ...] means after choosing
        action, agent goes to (x, y) with probability of 80% and goes to (z, t) with probability of 20%.

        """

        next_state_dict = {Actions.N: (-1, 0), Actions.S: (1, 0), Actions.E: (0, 1), Actions.W: (0, -1)}
        non_determinist_dict = {Actions.N: Actions.E, Actions.E: Actions.S, Actions.S: Actions.W, Actions.W: Actions.N}
        transitions = []
        next_x, next_y = tuple(map(sum, zip(next_state_dict[action], state)))
        if (0 <= next_x < self.grid_dim[0]) and (0 <= next_y < self.grid_dim[1]):
            transitions += [(next_x, next_y, 0.8)]
        next_x, next_y = tuple(map(sum, zip(next_state_dict[non_determinist_dict[action]], state)))
        if (0 <= next_x < self.grid_dim[0]) and (0 <= next_y < self.grid_dim[1]):
            transitions += [(next_x, next_y, 0.2)]
        return transitions
