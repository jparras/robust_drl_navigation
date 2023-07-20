import numpy as np

class SIMPLE(object):
    def __init__(self, env, gamma=0.99):
        """
        Function that implements the simple baseline. Input params added for compatibility (not used anyways)
        :param env: Gym like environment
        """
        self.env = env
        self.gamma = gamma

    def plot_training(self, data_vals, tikz_name=None):
        pass

    def test(self, nn_name=None, n_episodes=100, initial_states=None):

        output_data = []
        for e in range(n_episodes):
            if initial_states is not None:
                state, done = self.env.reset(initial_states[e]), False
            else:
                state, done = self.env.reset(), False

            episode_data = {'states': [state],
                            'actions': [],
                            'rewards': [],
                            'episode_reward': 0,
                            'episode_reward_disc': 0,
                            'episode_timestaps': 0}
            while not done:
                episode_data['episode_timestaps'] += 1
                ac = np.arctan2(-state[0, 1], -state[0, 0])
                ac = np.array([np.cos(ac), np.sin(ac)])
                # Perform action
                next_state, reward, done, _ = self.env.step(ac)
                state = next_state

                # Save data
                episode_data['states'].append(state)
                episode_data['actions'].append(ac)
                episode_data['rewards'].append(reward)
                episode_data['episode_reward'] += reward
                episode_data['episode_reward_disc'] += self.gamma ** (
                        episode_data['episode_timestaps'] - 1) * reward

            output_data.append(episode_data)

        return output_data

    def plot_policy(self, nn_name=None, norm_value=False):
        pass
