import numpy as np
import torch
from torch.optim import Adam
import time
import coreppo as core
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import pickle
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt
from tikzplotlib import save


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOADV(object):

    def __init__(self, env_fn,
                 actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 gamma=0.99,
                 clip_ratio=0.2,
                 pi_lr=3e-4,
                 vf_lr=1e-3,
                 train_pi_iters=80,
                 train_v_iters=80,
                 lam=0.97,
                 target_kl=0.01
                 ):
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.gamma = gamma
        self.lam = lam
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim_pro = self.env.action_space_pro.shape  # Protagonist
        self.act_dim_adv = self.env.action_space_adv.shape  # Adversary

        # Create actor-critic module
        self.ac_pro = actor_critic(self.env.observation_space, self.env.action_space_pro, **ac_kwargs)
        self.ac_adv = actor_critic(self.env.observation_space, self.env.action_space_adv, **ac_kwargs)

        # Sync params across processes
        sync_params(self.ac_pro)
        sync_params(self.ac_adv)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac_pro.pi, self.ac_pro.v])
        print('\nNumber of parameters (prot): \t pi: %d, \t v: %d\n' % var_counts)
        var_counts = tuple(core.count_vars(module) for module in [self.ac_adv.pi, self.ac_adv.v])
        print('\nNumber of parameters (adv): \t pi: %d, \t v: %d\n' % var_counts)

        # Set up experience buffer
        self.buf_pro = None  # Buffer is initialized when calling train method
        self.buf_adv = None  # Buffer is initialized when calling train method

        # Set up optimizers for policy and value function
        self.pi_optimizer_pro = Adam(self.ac_pro.pi.parameters(), lr=pi_lr)
        self.vf_optimizer_pro = Adam(self.ac_pro.v.parameters(), lr=vf_lr)
        self.pi_optimizer_adv = Adam(self.ac_adv.pi.parameters(), lr=pi_lr)
        self.vf_optimizer_adv = Adam(self.ac_adv.v.parameters(), lr=vf_lr)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, adversary):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        if not adversary:
            pi, logp = self.ac_pro.pi(obs, act)
        else:
            pi, logp = self.ac_adv.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data, adversary):
        obs, ret = data['obs'], data['ret']
        if not adversary:
            return ((self.ac_pro.v(obs) - ret) ** 2).mean()
        else:
            return ((self.ac_adv.v(obs) - ret) ** 2).mean()

    def update(self, adversary):
        if not adversary:
            data = self.buf_pro.get()
        else:
            data = self.buf_adv.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data, adversary)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data, adversary).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            if not adversary:
                self.pi_optimizer_pro.zero_grad()
            else:
                self.pi_optimizer_adv.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, adversary)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            if not adversary:
                mpi_avg_grads(self.ac_pro.pi)  # average grads across MPI processes
                self.pi_optimizer_pro.step()
            else:
                mpi_avg_grads(self.ac_adv.pi)  # average grads across MPI processes
                self.pi_optimizer_adv.step()

        # Value function learning
        for i in range(self.train_v_iters):
            if not adversary:
                self.vf_optimizer_pro.zero_grad()
            else:
                self.vf_optimizer_adv.zero_grad()
            loss_v = self.compute_loss_v(data, adversary)
            loss_v.backward()
            if not adversary:
                mpi_avg_grads(self.ac_pro.v)  # average grads across MPI processes
                self.vf_optimizer_pro.step()
            else:
                mpi_avg_grads(self.ac_adv.v)  # average grads across MPI processes
                self.vf_optimizer_adv.step()

        return pi_info, np.squeeze(loss_pi.cpu().data.numpy().flatten()), \
               np.squeeze(loss_v.cpu().data.numpy().flatten())

    def load(self, filename):
        self.ac_pro.load_state_dict(torch.load(filename + '_pro'))
        self.ac_adv.load_state_dict(torch.load(filename + '_adv'))

    def save(self, filename):
        torch.save(self.ac_pro.state_dict(), filename + '_pro')
        torch.save(self.ac_adv.state_dict(), filename + '_adv')

    def train(self, steps_per_epoch, epochs, max_ep_len=1000, data_vals=None, nn_file=None):

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf_pro = PPOBuffer(self.obs_dim, self.act_dim_pro, local_steps_per_epoch, self.gamma, self.lam)
        self.buf_adv = PPOBuffer(self.obs_dim, self.act_dim_adv, local_steps_per_epoch, self.gamma, self.lam)
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        rv = []
        lv = []
        lens = [[], []]
        rets = [[], []]
        pi_losses = [[], []]
        v_losses = [[], []]
        pi_kl = [[], []]
        ents = [[], []]
        clipfracs = [[], []]

        for epoch in range(epochs):
            for adversary in [False, True]:  # Alternate training!!
                for t in range(local_steps_per_epoch):

                    a_pro, v_pro, logp_pro = self.ac_pro.step(torch.as_tensor(o, dtype=torch.float32))
                    a_adv, v_adv, logp_adv = self.ac_adv.step(torch.as_tensor(o, dtype=torch.float32))

                    next_o, r, d, _ = self.env.step(a_pro, adv_action=a_adv)
                    ep_ret += r
                    ep_len += 1

                    # save and log
                    if not adversary:
                        self.buf_pro.store(o, a_pro, r, v_pro, logp_pro)
                    else:
                        self.buf_adv.store(o, a_adv, -r, v_adv, logp_adv)

                    # Update obs (critical!)
                    o = next_o

                    timeout = ep_len == max_ep_len
                    terminal = d or timeout
                    epoch_ended = t == local_steps_per_epoch - 1

                    if terminal or epoch_ended:
                        if epoch_ended and not (terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            if not adversary:
                                _, v_pro, _ = self.ac_pro.step(torch.as_tensor(o, dtype=torch.float32))
                            else:
                                _, v_adv, _ = self.ac_adv.step(torch.as_tensor(o, dtype=torch.float32))
                        else:
                            v_pro = v_adv = 0
                        if not adversary:
                            self.buf_pro.finish_path(v_pro)
                        else:
                            self.buf_adv.finish_path(v_adv)
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            #print('EpRet=', ep_ret, 'EpLen=', ep_len)
                            lv.append(ep_len)
                            rv.append(ep_ret)
                        o, ep_ret, ep_len = self.env.reset(), 0, 0

                # Perform PPO update!
                pi_info, pi_loss, v_loss = self.update(adversary)

                if not adversary:
                    idx = 0
                else:
                    idx = 1

                lens[idx].append(np.mean(np.array(lv)))
                rets[idx].append(np.mean(np.array(rv)))
                rv = []
                lv = []
                pi_losses[idx].append(pi_loss)
                v_losses[idx].append(v_loss)
                ents[idx].append(pi_info['ent'])
                pi_kl[idx].append(pi_info['kl'])
                clipfracs[idx].append(pi_info['cf'])

                # Log info about epoch
                print('Epoch ', epoch)
                print('Adversary ', adversary)
                print('Time ', time.time() - start_time)
                print('Avg return ', rets[idx][-1])

        if nn_file is not None:
            self.save(nn_file)

        if data_vals is not None:
            with open(data_vals, 'wb') as handle:
                pickle.dump({'lens': lens, 'rets': rets, 'pi_loss': pi_losses, 'v_loss': v_losses, 'ent': ents,
                             'cf': clipfracs, 'kl': pi_kl}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_training(self, data_vals, tikz_name=None):
        data = []
        if isinstance(data_vals, list):
            for d in data_vals:
                with open(d, 'rb') as handle:
                    data.append(pickle.load(handle))
        else:
            raise RuntimeError('data_vals has to be a list')
        for key in ['lens', 'rets', 'pi_loss', 'v_loss', 'ent', 'cf', 'kl']:
            aux_pro = np.array([d[key][0] for d in data])
            aux_adv = np.array([d[key][1] for d in data])
            for i in range(len(aux_pro)):
                # d = uniform_filter1d(d, size=100)  # Moving average!
                plt.plot(aux_pro[i], 'b')
                plt.plot(aux_adv[i], 'r')
            plt.ylabel(key)
            plt.xlabel('Training epoch')
            plt.title('PPO ADV training ')
            if tikz_name is not None:
                save(tikz_name + str(key) + '_ppo_adv.tex')
            plt.show()

    def test(self, nn_name=None, n_episodes=100, initial_states=None, env=None):
        # Returns a list of dictionaries with the data of several runs
        if nn_name is not None:
            self.load(nn_name)  # To load a saved NN

        if initial_states is not None:
            assert len(initial_states) == n_episodes

        output_data = []
        if env is None:  # Use adv env from class
            for e in range(n_episodes):
                if initial_states is not None:
                    state, done = self.env.reset(np.squeeze(initial_states[e])), False
                else:
                    state, done = self.env.reset(), False

                episode_data = {'states': [state],
                                'actions': [],
                                'actions_adv': [],
                                'rewards': [],
                                'episode_reward': 0,
                                'episode_reward_disc': 0,
                                'episode_timestaps': 0}
                while not done:
                    episode_data['episode_timestaps'] += 1
                    ac_pro, _, _ = self.ac_pro.step(torch.as_tensor(state, dtype=torch.float32))
                    ac_adv, _, _ = self.ac_adv.step(torch.as_tensor(state, dtype=torch.float32))

                    # Perform action
                    next_state, reward, done, _ = self.env.step(ac_pro, adv_action=ac_adv)
                    state = next_state

                    # Save data
                    episode_data['states'].append(state)
                    episode_data['actions'].append(ac_pro)
                    episode_data['actions_adv'].append(ac_adv)
                    episode_data['rewards'].append(reward)
                    episode_data['episode_reward'] += reward
                    episode_data['episode_reward_disc'] += self.gamma ** (
                            episode_data['episode_timestaps'] - 1) * reward

                output_data.append(episode_data)

        else:  # Use env without adversary given as input!
            for e in range(n_episodes):
                if initial_states is not None:
                    state, done = env.reset(initial_states[e]), False
                else:
                    state, done = env.reset(), False

                episode_data = {'states': [state],
                                'actions': [],
                                'rewards': [],
                                'episode_reward': 0,
                                'episode_reward_disc': 0,
                                'episode_timestaps': 0}
                while not done:
                    episode_data['episode_timestaps'] += 1
                    ac_pro, _, _ = self.ac_pro.step(torch.as_tensor(state, dtype=torch.float32))

                    # Perform action
                    next_state, reward, done, _ = env.step(ac_pro)
                    state = next_state

                    # Save data
                    episode_data['states'].append(state)
                    episode_data['actions'].append(ac_pro)
                    episode_data['rewards'].append(reward)
                    episode_data['episode_reward'] += reward
                    episode_data['episode_reward_disc'] += self.gamma ** (
                            episode_data['episode_timestaps'] - 1) * reward

                output_data.append(episode_data)
        return output_data

    def plot_policy(self, nn_name=None, norm_value=False):
        if nn_name is not None:
            self.load(nn_name)  # To load a saved NN

        npp = 20  # Points to represent the policy
        x = y = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], npp)
        X, Y = np.meshgrid(x, y)
        observations = np.vstack([np.ravel(X), np.ravel(Y), np.zeros(npp ** 2), np.zeros(npp ** 2)]).T

        acs = np.zeros((npp * npp, 2))
        vals = np.zeros(npp * npp)
        for i in range(npp * npp):
            ac, val, _ = self.ac_pro.step(torch.as_tensor(observations[i], dtype=torch.float32))
            acs[i] = np.squeeze(self.env.action_adaptation(ac))  # Adapt action!
            vals[i] = val
        plt.quiver(np.ravel(X), np.ravel(Y), acs[:, 0], acs[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('PPO ADV prot policy')
        plt.show()

        if norm_value:
            # Normalize v between 0 and 1
            aux = vals - np.amin(vals)
            vals = aux / np.amax(aux)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, vals.reshape(X.shape), alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Value')
        plt.title('PPO value function estimated prot')
        plt.show()

        acs = np.zeros((npp * npp, 2))
        vals = np.zeros(npp * npp)
        for i in range(npp * npp):
            ac, val, _ = self.ac_adv.step(torch.as_tensor(observations[i], dtype=torch.float32))
            acs[i] = np.squeeze(self.env.action_adaptation(ac, adv=True))  # Adapt action!
            vals[i] = val
        plt.quiver(np.ravel(X), np.ravel(Y), acs[:, 0], acs[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('PPOADV adversary policy')
        plt.show()

        if norm_value:
            # Normalize v between 0 and 1
            aux = vals - np.amin(vals)
            vals = aux / np.amax(aux)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, vals.reshape(X.shape), alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Value')
        plt.title('PPOADV value function estimated adversary')
        plt.show()

