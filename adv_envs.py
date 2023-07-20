import gym
import numpy as np


def perturbation_swirl(x, y, params):
    px = params[0] * (y - params[2]) / np.sqrt(np.square(x - params[1]) + np.square(y - params[2]))
    py = - params[0] * (x - params[1]) / np.sqrt(np.square(x - params[1]) + np.square(y - params[2]))
    return (px, py)


def perturbation_current_h(x, y, params):
    px = params[0] * np.exp(-np.square(y - params[1]) / (params[2] ** 2))
    py = np.zeros_like(y)
    return (px, py)


def perturbation_current_v(x, y, params):
    px = np.zeros_like(x)
    py = params[0] * np.exp(-np.square(x - params[1]) / (params[2] ** 2))
    return (px, py)


def perturbation_const(x, y, params):
    px = params[0] * np.cos(params[1]) * np.ones_like(x)
    py = params[0] * np.sin(params[1]) * np.ones_like(y)
    return (px, py)


class PerturbationModel:
    def __init__(self, mode, params=None, n_params = 3):
        self.mode = mode  # swirl, current_h or current_v
        self.data = None
        self.n_params = n_params
        if params is None:
            self.params = np.random.rand(n_params) # c, x0, y0 / a, mu, sigma (swirl / current models)
        else:
            self.params = params

    def perturbation(self, x, y):
        if self.mode == 'swirl':
            return perturbation_swirl(x, y, params=self.params)
        elif self.mode == 'current_h':
            return perturbation_current_h(x, y, params=self.params)
        elif self.mode == 'current_v':
            return perturbation_current_v(x, y, params=self.params)
        elif self.mode == 'const':
            return perturbation_const(x, y, params=self.params)
        else:
            raise RuntimeError('Perturbation mode not recognized')


class UuvPertSingle(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, pert_mode='swirl', pert_params=np.array([0.5, 5, 1])):

        self.state_size = 4

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params

    def obs(self):
        return self.state  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(state, self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.n = 0
        if self.pert_mode is not None:
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!
        return self.obs()

    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
        else:
            px, py = 0, 0

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvPertSingleHC(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, pert_mode='current_h', pert_params=np.array([1, 5, 3])):

        self.state_size = 4

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params

    def obs(self):
        return self.state  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(state, self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.n = 0
        if self.pert_mode is not None:
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!
        return self.obs()

    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
        else:
            px, py = 0, 0

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvPertSingleVC(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, pert_mode='current_v', pert_params=np.array([1, 5, 3])):

        self.state_size = 4

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params

    def obs(self):
        return self.state  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(state, self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.n = 0
        if self.pert_mode is not None:
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!
        return self.obs()

    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
        else:
            px, py = 0, 0

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvPertSingleC(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, pert_mode='const', pert_params=np.array([0.1, np.pi / 4, 1])):

        self.state_size = 4

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params

    def obs(self):
        return self.state  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(state, self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.n = 0
        if self.pert_mode is not None:
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!
        return self.obs()

    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
        else:
            px, py = 0, 0

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvAdvSingle(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, max_adv_force=0.5):

        self.state_size = 4

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space_pro = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        self.ac_adv_low = np.array([-max_adv_force, -max_adv_force])
        self.ac_adv_high = np.array([max_adv_force, max_adv_force])
        self.action_space_adv = gym.spaces.Box(low=self.ac_adv_low, high=self.ac_adv_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.n = None

    def obs(self):
        return self.state  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(state, self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.n = 0
        return self.obs()

    def action_adaptation(self, action, adv=False):
        if not adv:
            action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
            return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!
        else:
            action = self.clip(action, self.ac_adv_low, self.ac_adv_high)  # Limit action values
            return self.ac_adv_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action, adv_action=None):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        if adv_action is None:
            px, py = 0, 0
        else:
            adv_action = self.action_adaptation(np.squeeze(adv_action), adv=True)
            px = adv_action[0]
            py = adv_action[1]

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvPertDouble(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, pert_mode='swirl', pert_params=np.array([0.5, 5, 1])):

        self.state_size = 4
        self.obs_size = 8

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel, -max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel, max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.prev_state = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params

    def obs(self):
        return np.hstack([self.state, self.prev_state])  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high[0: self.state_size] - self.obs_low[0: self.state_size])
                                   + self.obs_low[0: self.state_size], self.obs_low[0: self.state_size], self.obs_high[0: self.state_size])
        else:
            self.state = self.clip(state[0: self.state_size], self.obs_low[0: self.state_size], self.obs_high[0: self.state_size])

        self.state = np.reshape(self.state, [1, self.state_size])
        self.prev_state = np.copy(self.state)
        self.n = 0
        if self.pert_mode is not None:
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!
        return self.obs()

    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
        else:
            px, py = 0, 0

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.prev_state = np.copy(self.state)
        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass


class UuvAdvDouble(gym.Env):
    def __init__(self, max_force=1, max_pos=10, max_vel=2, max_adv_force=0.5):

        self.state_size = 4
        self.obs_size = 8

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel, -max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel, max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])
        self.action_space_pro = gym.spaces.Box(low=self.ac_low, high=self.ac_high, dtype=np.float32)

        self.ac_adv_low = np.array([-max_adv_force, -max_adv_force])
        self.ac_adv_high = np.array([max_adv_force, max_adv_force])
        self.action_space_adv = gym.spaces.Box(low=self.ac_adv_low, high=self.ac_adv_high, dtype=np.float32)

        # Simulation parameters
        self.n_max = 100  # Max number of time steps
        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1  # To conclude an episode: ball centered at speed and position!
        self.k = 0.5  # Friction coefficient
        # Internal simulation parameters
        self.state = None
        self.prev_state = None
        self.n = None

    def obs(self):
        return np.hstack([self.state, self.prev_state])  # By default, it is a differential state (i.e., target = 0)

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        # Initialize with random values!
        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high[0: self.state_size] - self.obs_low[0: self.state_size])
                                   + self.obs_low[0: self.state_size], self.obs_low[0: self.state_size], self.obs_high[0: self.state_size])
        else:
            self.state = self.clip(state, self.obs_low[0: self.state_size], self.obs_high[0: self.state_size])

        self.state = np.reshape(self.state, [1, self.state_size])
        self.prev_state = np.copy(self.state)
        self.n = 0
        return self.obs()

    def action_adaptation(self, action, adv=False):
        if not adv:
            action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
            return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!
        else:
            action = self.clip(action, self.ac_adv_low, self.ac_adv_high)  # Limit action values
            return self.ac_adv_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!

    def step(self, action, adv_action=None):

        action = np.squeeze(action)

        if action.size == 2:
            action = self.action_adaptation(action)
        else:
            raise RuntimeError('Action not valid')

        if adv_action is None:
            px, py = 0, 0
        else:
            adv_action = self.action_adaptation(np.squeeze(adv_action), adv=True)
            px = adv_action[0]
            py = adv_action[1]

        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (action[0] + px - self.k * u) * self.time_step + u
        v_next = (action[1] + py - self.k * v) * self.time_step + v
        # next_state = self.clip(np.vstack([x_next, y_next, u_next, v_next]).T, self.obs_low, self.obs_high)
        next_state = np.vstack([x_next, y_next, u_next, v_next]).T

        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False
        if distance_to_target <= self.distance_th or self.n >= self.n_max:
            done = True

        reward = -1
        self.n += 1

        self.prev_state = np.copy(self.state)
        self.state = next_state

        return self.obs(), reward, done, (px, py)

    def render(self, mode='human'):
        pass

