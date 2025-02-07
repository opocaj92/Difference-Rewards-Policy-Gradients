import numpy as np
import heapq
from functools import total_ordering
import pickle
import gc
import copy
import os
from envs.gridworld.predator_prey import PredatorPrey

gc.collect()

@total_ordering
class Policy:
    def __init__(self, env, acc_return, moves, gamma):
        self.n_agents = env.n_agents
        self.n_actions = env.n_actions
        self.env_setting = env.get_current_setting()
        self.moves = moves
        self.gamma = gamma
        self.acc_return = acc_return
        # value is negative to use in minheap
        self.value = -(self.acc_return + env.get_heuristic(self.gamma))
        self.terminated = self.env_setting["n_steps"] == env.episode_limit

    def actions(self, state):
        return self.moves[state.tobytes()]

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.value >= other.value

class MAAStar:
    def __init__(self, env, gamma, savepath):
        self.env = env
        self.n_agents = env.n_agents
        self.n_preys = env.n_preys
        self.n_actions = env.n_actions
        self.joint_actions = self.__generate_all_joint_actions()
        self.gamma = gamma
        self.savepath = savepath
        os.makedirs(savepath, exist_ok = True)

    def __generate_all_joint_actions(self, actions = [[]]):
        if all([len(a) == self.n_agents for a in actions]):
            return actions
        else:
            new = []
            for a in range(self.n_actions):
                expanded = [copy.deepcopy(old_a) + [a] for old_a in actions]
                new = new + self.__generate_all_joint_actions(expanded)
            return new

    def run(self, name_tag = 0):
        steps_best_values = [float("inf") for _ in range(self.env.episode_limit)]
        steps_best_returns = [float("-inf") for _ in range(self.env.episode_limit)]
        preys_actions_sequence = np.random.choice(self.n_actions, size = (self.env.episode_limit, self.n_preys))
        pi_pool = []
        heapq.heapify(pi_pool)
        _ = self.env.reset()
        with open(self.savepath + "/env_config" + str(name_tag) + ".pkl", "wb") as f:
            pickle.dump(self.env.get_current_setting(), f)
        with open(self.savepath + "/preys_actions_sequence" + str(name_tag) + ".pkl", "wb") as f:
            pickle.dump(preys_actions_sequence, f)
        heapq.heappush(pi_pool, Policy(self.env, 0, {}, self.gamma))
        while True:
            current = heapq.heappop(pi_pool)
            if current.terminated:
                break
            state = self.env.get_state()
            prev_setting = current.env_setting
            for a in self.joint_actions:
                self.env.set_current_setting(prev_setting)
                reward, _, _ = self.env.step(a, preys_actions_sequence[prev_setting["n_steps"]])
                new_return = current.acc_return + (self.gamma ** prev_setting["n_steps"]) * reward
                new_moves = copy.deepcopy(current.moves)
                new_moves[state.tobytes()] = a
                new_pi = Policy(self.env, new_return, new_moves, self.gamma)
                #if new_pi.value < steps_best_values[prev_setting["n_steps"]] or (new_pi.value == steps_best_values[prev_setting["n_steps"]] and new_pi.acc_return >= steps_best_returns[prev_setting["n_steps"]]):
                if new_pi.value <= steps_best_values[prev_setting["n_steps"]]:
                    heapq.heappush(pi_pool, new_pi)
                    steps_best_values[prev_setting["n_steps"]] = new_pi.value
                    steps_best_returns[prev_setting["n_steps"]] = new_pi.acc_return
        print("OPTIMAL POLICY FOUND")
        print("VALUE OF THE OPTIMAL POLICY: " + str(-current.value))
        with open(self.savepath + "/opt_pi" + str(name_tag) + ".pkl", "wb") as f:
            pickle.dump(current, f)
        del pi_pool
        gc.collect()
        return current

n_agents = 3
gamma = 1#0.99
runs = 30

env = PredatorPrey(n_agents = n_agents)

savepath = "maastar_predator_prey_n" + str(n_agents)
planner = MAAStar(env, gamma, savepath)

for i in range(runs):
    opt_pi = planner.run(i + 1)