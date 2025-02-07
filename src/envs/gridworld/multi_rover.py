from __future__ import division
from gym.envs.classic_control import rendering
import numpy as np
import copy

class MultiRover:
    def __init__(self, n_agents = 3,
                 grid_size = 10,
                 episode_limit = 25,
                 obs_last_action = False,
                 obs_timestep_number = False,
                 state_last_action = False,
                 state_timestep_number = False,
                 collisions = True,
                 collisions_penalty = 1.,
                 return_all_rewards = False,
                 seed = None):        
        self._viewer = None
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        self.grid_size = grid_size
        self.episode_limit = episode_limit
        self._n_steps = 0
        self.n_actions = 5
        self.obs_last_action = obs_last_action
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.collisions = collisions
        self.collisions_penalty = collisions_penalty
        self.return_all_rewards = return_all_rewards
        self.seed = seed
        self._agents_pos = None
        self._landmarks_pos = None

    def step(self, actions):
        actions = [int(a) for a in actions]
        info = {
            "all_rewards": []
        }
        if self.return_all_rewards:
            original = copy.copy(self._agents_pos)
            for i in range(self.n_agents):
                tmp_actions = copy.copy(actions)
                info["all_rewards"].append([])
                for j in range(self.n_actions):
                    tmp_actions[i] = j
                    self._take_agents_action(tmp_actions)
                    info["all_rewards"][i].append(self._get_agents_reward())
                    self._agents_pos = copy.copy(original)
        self._take_agents_action(actions)
        rewards = self._get_agents_reward()
        self.last_action = np.eye(self.n_actions)[np.array(actions)]
        self._n_steps += 1
        return rewards, self._n_steps == self.episode_limit, info

    def get_obs(self):
        return [self.get_agent_obs(i) for i in range(self.n_agents)]

    def get_agent_obs(self, agent_id):
        agent_obs = []
        for j, q in enumerate(np.concatenate((self._agents_pos, self._landmarks_pos), axis = 0)):
            if agent_id != j:
                agent_obs.append(np.subtract(q, self._agents_pos[agent_id]))
        if self.obs_last_action:
            agent_obs = np.append(agent_obs, self.last_action[agent_id])
        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs, self._n_steps / self.episode_limit)
        return np.array(agent_obs).flatten()

    def get_obs_size(self):
        return 2 * (self.n_agents + self.n_landmarks - 1) + self.n_actions * self.obs_last_action + 1 * self.obs_timestep_number

    def get_state(self):
        state = self.get_obs()
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state, self._n_steps / self.episode_limit)
        return np.array(state).flatten()

    def get_state_size(self):
        return (2 * (self.n_agents + self.n_landmarks - 1)) * self.n_agents + self.n_actions * self.n_agents * self.state_last_action + 1 * self.state_timestep_number

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions
        
    def reset(self):
        self._n_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        pos = np.random.choice(self.grid_size ** 2, size = self.n_landmarks + self.n_agents, replace = False)
        self._landmarks_pos = np.array([[p // self.grid_size, p % self.grid_size] for p in pos[:self.n_landmarks]])
        self._agents_pos = np.array([[p // self.grid_size, p % self.grid_size] for p in pos[-self.n_agents:]])
        return self.get_obs()

    def render(self, mode = "human", close = False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return
        grid = np.multiply(np.ones((self.grid_size ** 2, 3), dtype = np.int8), np.array([32, 32, 32], dtype = np.int8))
        for p in self._landmarks_pos:
            grid[p[0] * self.grid_size + p[1]] = np.array([255, 255, 0])
        for p in self._agents_pos:
            grid[p[0] * self.grid_size + p[1]] = np.array([0, 153, 0])
        grid = grid.reshape(self.grid_size, self.grid_size, 3)
        if mode == "human":
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            grid = np.repeat(grid, 10, axis = 1)
            grid = np.repeat(grid, 10, axis = 0)
            self._viewer.imshow(grid)
            return self._viewer.isopen
        elif mode == "rgb_array":
            return grid
        else:
            return
        
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return
        
    def seed(self):
        return self.seed
        
    def get_replay(self):
        return
        
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def _take_agents_action(self, actions):
        for a, p in zip(actions, self._agents_pos):
            if a == 1:
                p[0] = max(0, p[0] - 1)
            elif a == 2:
                p[0] = min(p[0] + 1, self.grid_size - 1)
            elif a == 3:
                p[1] = min(p[1] + 1, self.grid_size - 1)
            elif a == 4:
                p[1] = max(0, p[1] - 1)

    def _get_agents_reward(self):
        reward = 0.
        for p in self._landmarks_pos:
            min_d = self.grid_size ** 2
            for q in self._agents_pos:
                min_d = min(min_d, np.sum(np.abs(np.subtract(p, q))))
            reward -= min_d
        if self.collisions:
            for i, p in enumerate(self._agents_pos):
                for j, q in enumerate(self._agents_pos[i + 1:]):
                    if np.array_equal(p, q):
                        reward -= self.collisions_penalty
        return reward

    def get_stats(self):
        stats = {}
        return stats

    # SHITTY STUFF FOR PLOTTING PURPOSES ONLY!!!
    def get_current_setting(self):
        env_setting = {"agents_pos": copy.deepcopy(self._agents_pos),
                        "landmarks_pos": copy.deepcopy(self._landmarks_pos),
                        "n_steps": self._n_steps,
                        "last_action": copy.deepcopy(self.last_action)}
        return env_setting

    def set_current_setting(self, env_setting):
        self._agents_pos = copy.deepcopy(env_setting["agents_pos"])
        self._landmarks_pos = copy.deepcopy(env_setting["landmarks_pos"])
        self._n_steps = env_setting["n_steps"]
        self.last_action = copy.deepcopy(env_setting["last_action"])

    def uniform_sample(self):
        # Needs to call reset() after this!
        actions = np.random.randint(self.n_actions, size = self.n_agents)
        pos = np.random.choice(self.grid_size ** 2, size = self.n_landmarks + self.n_agents, replace = False)
        self._landmarks_pos = np.array([[p // self.grid_size, p % self.grid_size] for p in pos[:self.n_landmarks]])
        self._agents_pos = np.array([[p // self.grid_size, p % self.grid_size] for p in pos[-self.n_agents:]])
        last_action = np.eye(self.n_actions)[np.random.randint(self.n_actions, size = self.n_agents)]
        self.last_action = last_action
        n_steps = np.random.randint(self.episode_limit)
        self._n_steps = n_steps
        obs = self.get_obs()
        state = self.get_state()
        reward, terminated, infos = self.step(actions)
        next_obs = self.get_obs()
        next_state = self.get_state()
        setting = self.get_current_setting()
        return obs, state, actions, self.get_avail_actions(), reward, next_obs, next_state, terminated, last_action, n_steps, infos, setting

    # SHITTY STUFF FOR PLANNING PURPOSES ONLY!!!
    def get_heuristic(self, gamma):
        steps_per_landmark = []
        for p in self._landmarks_pos:
            min_d = self.grid_size ** 2
            for q in self._agents_pos:
                min_d = min(min_d, np.sum(np.abs(np.subtract(p, q))))
            steps_per_landmark.append(min_d)
        heuristic = 0
        for i in range(self._n_steps, self.episode_limit):
            steps_per_landmark = [max(0, j - 1) for j in steps_per_landmark]
            heuristic -= (gamma ** i) * sum(steps_per_landmark)
        return heuristic