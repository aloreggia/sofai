from collections import namedtuple, defaultdict
import numpy as np


class Feature:
    def __init__(self, default_value):
        self.default = default_value
        self.name2value = {default_value: default_value}
        self.key2name = defaultdict(lambda: default_value)
        self.size = 1

    def get(self, s, a, s_):
        """Returns the feature value corresponding to the transition

        Args:
            s (Integer): Source state
            a (Integer): performed action
            s_ (Integer): destination state
        """
        name = self._get_name(self._get_key(s, a, s_))
        return self._get_value(name)

    def set(self, s, a, s_, value):
        """Sets the feature value of a transition

        Args:
            s (Integer): Source state
            a (Integer): performed action
            s_ (Integer): destination state
            value(Any): the feature value
        """
        name = self._get_name(self._get_key(s, a, s_))
        self._set_value(name, value)

    def value2feature(self, v):
        return v

    def _get_key(self, s, a, a_):
        return s, a, a_

    def _get_name(self, key):
        return self.key2name[key]

    def _get_value(self, name):
        return self.name2value[name]

    def _set_value(self, name, value):
        self.name2value[name] = value


class OneHotFeature(Feature):
    def __init__(self, default_value, size):
        super().__init__(default_value)
        self.size = size

    def get(self, s, a, s_):
        val = super().get(s, a, s_)
        return self.value2feature(val)

    def value2feature(self, v):
        o = np.zeros((self.size,))
        o[v] = 1
        return o


class ColorFeature(OneHotFeature):
    def __init__(self, colors, default_color):
        super().__init__(colors[default_color], len(colors))
        self.name2value = {c: i for i, c in enumerate(colors)}

    def _get_key(self, s, a, s_):
        return s_

    def set_states_color(self, states, color):
        if color not in self.name2value:
            raise Exception(f'Color {color} is not supported.')

        for s in states:
            self.key2name[s] = color


class SourceStateFeature(OneHotFeature):
    def __init__(self, n_states):
        super().__init__('s0', n_states)
        for s in range(n_states):
            self.name2value[s] = s
            self.key2name[s] = s

    def _get_key(self, s, a, s_):
        return s


class DistinationStateFeature(OneHotFeature):
    def __init__(self, n_states):
        super().__init__(0, n_states)
        for s in range(n_states):
            self.name2value[s] = s
            self.key2name[s] = s

    def _get_key(self, s, a, s_):
        return s_


class DirectionFeature(OneHotFeature):
    def __init__(self, directions):
        super().__init__(0, len(directions))
        for d in directions:
            self.name2value[d.name] = d.idx
            self.key2name[d.idx] = d.name

    def _get_key(self, s, a, s_):
        return a
