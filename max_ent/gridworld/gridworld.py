"""
Adopted from https://github.com/qzed/irl-maxent/blob/master/src/gridworld.py

Grid-World Markov Decision Processes (MDPs).

The MDPs in this module are actually not complete MDPs, but rather the
sub-part of an MDP containing states, actions, and transitions (including
their probabilistic character). Reward-function and terminal-states are
supplied separately.

Some general remarks:
    - Edges act as barriers, i.e. if an agent takes an action that would cross
    an edge, the state will not change.

    - Actions are not restricted to specific states. Any action can be taken
    in any state and have a unique inteded outcome. The result of an action
    can be stochastic, but there is always exactly one that can be described
    as the intended result of the action.
"""

import numpy as np
from itertools import product


class GridWorld:
    """
    Basic deterministic grid world MDP.

    The attribute size specifies both widht and height of the world, so a
    world will have size**2 states.

    Args:
        size: The width and height of the world as integer.
        allow_diagonal_actions: Can the agent move diagonally.

    Attributes:
        n_states: The number of states of this MDP.
        n_actions: The number of actions of this MDP.
        p_transition: The transition probabilities as table. The entry
            `p_transition[from, to, a]` contains the probability of
            transitioning from state `from` to state `to` via action `a`.
        size: The width and height of the world.
        actions: The actions of this world as paris, indicating the
            direction in terms of coordinates.
    """

    def __init__(self, size, feature_list=[], allow_diagonal_actions=False):
        self.size = size
        self.diagonal = allow_diagonal_actions
        if allow_diagonal_actions:
            self.actions = Directions.ALL_DIRECTIONS
        else:
            self.actions = Directions.FOUR_DIRECTIONS

        self.n_states = size**2
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()
        self.feature_list = feature_list.copy()
        self.n_features = sum([f.size for f in feature_list])
        self.phi = np.zeros(
            (self.n_states, self.n_actions, self.n_states, self.n_features))
        self._build_phi()

    def _build_phi(self):
        if self.n_features == 0:
            return

        def get(s, a, s_):
            temp = [f.get(s, a, s_)
                    for f in self.feature_list]
            return np.concatenate(temp)

        for s in range(self.n_states):
            for s_ in range(self.n_states):
                for a in range(self.n_actions):
                    self.phi[s, a, s_] = get(s, a, s_)

    def state_index_to_point(self, state):
        """
        Convert a state index to the coordinate representing it.

        Args:
            state: Integer representing the state.

        Returns:
            The coordinate as tuple of integers representing the same state
            as the index.
        """
        return state % self.size, state // self.size

    def state_point_to_index(self, state):
        """
        Convert a state coordinate to the index representing it.

        Note:
            Does not check if coordinates lie outside of the world.

        Args:
            state: Tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate.
        """
        return state[1] * self.size + state[0]

    def state_point_to_index_clipped(self, state):
        """
        Convert a state coordinate to the index representing it, while also
        handling coordinates that would lie outside of this world.

        Coordinates that are outside of the world will be clipped to the
        world, i.e. projected onto to the nearest coordinate that lies
        inside this world.

        Useful for handling transitions that could go over an edge.

        Args:
            state: The tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate if the coordinate lies inside this world, or the
            index to the closest state that lies inside the world.
        """
        s = (max(0, min(self.size - 1, state[0])),
             max(0, min(self.size - 1, state[1])))
        return self.state_point_to_index(s)

    def state_index_transition(self, s, a):
        """
        Perform action `a` at state `s` and return the intended next state.

        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.

        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.

        Returns:
            The next state as implied by the given action and state.
        """
        s = self.state_index_to_point(s)
        s = s[0] + self.actions[a].x, s[1] + self.actions[a].y
        return self.state_point_to_index_clipped(s)

    def off_grid(self, state, action):
        sx, sy = state
        ax, ay = action.x, action.y
        return not((0 <= sx + ax < self.size) and (0 <= sy + ay < self.size))

    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
        # For each state perform all actions and add the corresponding probability to the transition
        for si in range(self.n_states):
            s = self.state_index_to_point(si)
            for a in self.actions:
                for performed_a in self.actions:
                    off = self.off_grid(s, performed_a)
                    # Hitting the edge
                    if off:
                        new_state = si
                    else:
                        new_state = self.state_point_to_index(
                            (s[0] + performed_a.x, s[1] + performed_a.y))

                    table[si][new_state][a.idx] += self._action_prob(
                        a, performed_a)
        return table

    def _action_prob(self, chosen_a, performed_a):
        """ 
        Returns the probability of agent choosing action `chosen_a` but the environment 
        performing `performed_a`

        Args:
            chosen_a: The action that should be taken.
            performed_a: The action that actually perfoermed by the environment.

        Returns:
            Probability of this
        """
        return 1.0 if chosen_a == performed_a else 0.0

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)


class IcyGridWorld(GridWorld):
    """
    Grid world MDP similar to Frozen Lake, just without the holes in the ice.

    In this worlds, agents will slip with a specified probability, causing
    the agent to end up in a random neighboring state instead of the one
    implied by the chosen action.

    Args:
        size: The width and height of the world as integer.
        p_slip: The probability of a slip.

    Attributes:
        p_slip: The probability of a slip.

    See `class GridWorld` for more information.
    """

    def __init__(self, size, feature_list=[],  allow_diagonal_actions=False, p_slip=0.2):
        self.p_slip = p_slip

        super().__init__(size, feature_list, allow_diagonal_actions)

    def _action_prob(self, chosen_a, performed_a):
        # With 1 - p_slip do the action
        p = 1 - self.p_slip if chosen_a == performed_a else 0
        # otherwise do a random action
        p += (self.p_slip / self.n_actions)
        return p

    def __repr__(self):
        return "IcyGridWorld(size={}, p_slip={})".format(self.size, self.p_slip)


def state_features(world):
    """
    Return the feature matrix assigning each state with an individual
    feature (i.e. an identity matrix of size n_states * n_states).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    return np.identity(world.n_states)


def coordinate_features(world):
    """
    Symmetric features assigning each state a vector where the respective
    coordinate indices are nonzero (i.e. a matrix of size n_states *
    world_size).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    features = np.zeros((world.n_states, world.size))

    for s in range(world.n_states):
        x, y = world.state_index_to_point(s)
        features[s, x] += 1
        features[s, y] += 1

    return features


class Directions:
    class Action:
        def __init__(self, idx, x, y, name):
            self.idx = idx
            self.x = x
            self.y = y
            self.name = name

        def __repr__(self):
            return self.name

        def __str__(self):
            return self.name

    LEFT = Action(0, -1, 0, 'LEFT')
    RIGHT = Action(1, 1, 0, 'RIGHT')
    UP = Action(2, 0, 1, 'UP')
    DOWN = Action(3, 0, -1, 'DOWN')
    UP_LEFT = Action(4, -1, 1, 'UP_LEFT')
    UP_RIGHT = Action(5, 1, 1, 'UP_RIGHT')
    DOWN_LEFT = Action(6, -1, -1, 'DOWN_LEFT')
    DOWN_RIGHT = Action(7, 1, -1, 'DOWN_RIGHT')
    FOUR_DIRECTIONS = [LEFT, RIGHT, UP, DOWN]
    ALL_DIRECTIONS = FOUR_DIRECTIONS + \
        [UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]
