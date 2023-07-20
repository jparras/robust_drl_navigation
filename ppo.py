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
        assert self.ptr < self.max_size     # buffer has to have room so you can store
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
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPO(object):

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
        self.act_dim = self.env.action_space.shape

        # Create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)

        # Sync params across processes
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        self.buf = None # Buffer is initialized when calling train method

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self):
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        #logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        return pi_info, np.squeeze(loss_pi.cpu().data.numpy().flatten()), np.squeeze(loss_v.cpu().data.numpy().flatten())

    def load(self, filename):
        self.ac.load_state_dict(torch.load(filename ))

    def save(self, filename):
        torch.save(self.ac.state_dict(), filename)

    def train(self, steps_per_epoch, epochs, max_ep_len=1000, data_vals=None, nn_file=None):

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, local_steps_per_epoch, self.gamma, self.lam)
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        rv = []
        lv = []
        lens = []
        rets = []
        pi_losses = []
        v_losses = []
        pi_kl = []
        ents = []
        clipfracs = []

        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                #logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        #print('EpRet=', ep_ret, 'EpLen=', ep_len)
                        lv.append(ep_len)
                        rv.append(ep_ret)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            pi_info, pi_loss, v_loss = self.update()

            lens.append(np.mean(np.array(lv)))
            rets.append(np.mean(np.array(rv)))
            rv = []
            lv = []
            pi_losses.append(pi_loss)
            v_losses.append(v_loss)
            ents.append(pi_info['ent'])
            pi_kl.append(pi_info['kl'])
            clipfracs.append(pi_info['cf'])

            # Log info about epoch
            print('Epoch', epoch)
            print('Time', time.time()-start_time)
            print('Avg return ', rets[-1])

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
            aux = np.array([d[key] for d in data])
            for d in aux:
                #d = uniform_filter1d(d, size=100)  # Moving average!
                plt.plot(d)
            plt.ylabel(key)
            plt.xlabel('Training epoch')
            plt.title('PPO training ')
            if tikz_name is not None:
                save(tikz_name + str(key) + '_ppo.tex')
            plt.show()

    def test(self, nn_name=None, n_episodes=100, initial_states=None):
        # Returns a list of dictionaries with the data of several runs
        if nn_name is not None:
            self.load(nn_name)  # To load a saved NN

        if initial_states is not None:
            assert len(initial_states) == n_episodes

        output_data = []

        for e in range(n_episodes):
            if initial_states is not None:
                state, done = self.env.reset(np.squeeze(initial_states[e])), False
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
                action, _, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

                # Perform action
                next_state, reward, done, _ = self.env.step(action)
                state = next_state

                # Save data
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['episode_reward'] += reward
                episode_data['episode_reward_disc'] += self.gamma ** (episode_data['episode_timestaps'] - 1) * reward

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
        for i in range(npp*npp):
            ac, val, _ = self.ac.step(torch.as_tensor(observations[i], dtype=torch.float32))
            acs[i] = np.squeeze(self.env.action_adaptation(ac))  # Adapt action!
            vals[i] = val
        plt.quiver(np.ravel(X), np.ravel(Y), acs[:, 0], acs[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('PPO policy')
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
        plt.title('PPO value function estimated')
        plt.show()

