from joblib import Parallel, delayed
from ppo import PPO
from ppo_adv import PPOADV
from baseline import SIMPLE
import coreppo
from adv_envs import UuvPertSingle, UuvPertDouble, UuvAdvSingle, UuvAdvDouble, PerturbationModel, UuvPertSingleHC, UuvPertSingleVC, UuvPertSingleC
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from tikzplotlib import save


def default_params(adv=False, double=False):  # env, gamma
    if not adv and double:
        return UuvPertDouble(), 0.9999, UuvPertDouble
    elif not adv and not double:
        return UuvPertSingle(), 0.9999, UuvPertSingle
    elif adv and double:
        return UuvAdvDouble(), 0.9999, UuvAdvDouble
    elif adv and not double:
        return UuvAdvSingle(), 0.9999, UuvAdvSingle
    else:
        raise RuntimeError('Default params invalid')


def init_algo_train(algo, seed=0):
    if algo == 'ppo_adv_single':
        env, gamma, env_fn = default_params(adv=True, double=False)
        kwargs = {"actor_critic": coreppo.MLPActorCritic,
                  "ac_kwargs": dict(hidden_sizes=[64]*2),
                  "seed": seed,
                  "gamma": gamma,
                  "clip_ratio": 0.2,
                  "pi_lr": 3e-4,
                  "vf_lr": 1e-3,
                  "train_pi_iters": 80,
                  "train_v_iters": 80,
                  "lam": 0.97,
                  "target_kl": 0.01
                  }
        return PPOADV(env_fn, **kwargs)
    elif algo == 'ppo_adv_double':
        env, gamma, env_fn = default_params(adv=True, double=True)
        kwargs = {"actor_critic": coreppo.MLPActorCritic,
                  "ac_kwargs": dict(hidden_sizes=[64]*2),
                  "seed": seed,
                  "gamma": gamma,
                  "clip_ratio": 0.2,
                  "pi_lr": 3e-4,
                  "vf_lr": 1e-3,
                  "train_pi_iters": 80,
                  "train_v_iters": 80,
                  "lam": 0.97,
                  "target_kl": 0.01
                  }
        return PPOADV(env_fn, **kwargs)
    else:
        raise RuntimeError('Algorithm not recognized')


def test_algo(algo, seed=0, pert_mode=None, pert_params=None):
    gamma = 0.9999
    if algo is not 'ppo_adv_double':
        env = UuvPertSingle(pert_mode=pert_mode, pert_params=pert_params)
    else:
        env = UuvPertDouble(pert_mode=pert_mode, pert_params=pert_params)

    initial_states_file = os.path.join(os.getcwd(), 'res', 'initial_states.pickle')
    with open(initial_states_file, 'rb') as handle:
        initial_states = pickle.load(handle)['initial_states']
    n_episodes = len(initial_states)
    # Load algo
    nn_file = os.path.join(os.getcwd(), 'res', 'nn_data_' + str(algo) + '_' + str(seed))
    if algo == 'ppo':
        nn_file = os.path.join(os.getcwd(), 'res', 'nn_data_ppo' + str(pert_mode) + '_' + str(seed))
        kwargs = {"actor_critic": coreppo.MLPActorCritic,
                  "ac_kwargs": dict(hidden_sizes=[64]*2),
                  "seed": seed,
                  "gamma": gamma
                  }
        alg = PPO(UuvPertSingle, **kwargs)  # For testing, we do not use the PPO environment: don't care about it!
        alg.load(nn_file)
    elif algo == 'ppo_adv_single':
        kwargs = {"actor_critic": coreppo.MLPActorCritic,
                  "ac_kwargs": dict(hidden_sizes=[64]*2),
                  "seed": seed,
                  "gamma": gamma
                  }
        alg = PPOADV(UuvAdvSingle, **kwargs)
        alg.load(nn_file)
    elif algo == 'ppo_adv_double':
        kwargs = {"actor_critic": coreppo.MLPActorCritic,
                  "ac_kwargs": dict(hidden_sizes=[64]*2),
                  "seed": seed,
                  "gamma": gamma
                  }
        alg = PPOADV(UuvAdvDouble, **kwargs)
        alg.load(nn_file)
        initial_states = [np.hstack([ins, ins]) for ins in initial_states]
    elif algo == 'simple':
        pass
    else:
        raise RuntimeError('Algorithm not recognized')

    output_data = []

    for e in range(n_episodes):
        state, done = env.reset(np.squeeze(initial_states[e])), False

        episode_data = {'states': [state],
                        'actions': [],
                        'rewards': [],
                        'episode_reward': 0,
                        'episode_reward_disc': 0,
                        'episode_timestaps': 0}
        while not done:
            episode_data['episode_timestaps'] += 1
            if algo == 'simple':
                ac = np.arctan2(-state[0, 1], -state[0, 0])
                action = np.array([np.cos(ac), np.sin(ac)])
            elif algo == 'ppo':
                action, _, _ = alg.ac.step(torch.as_tensor(state, dtype=torch.float32))
            else:
                action, _, _ = alg.ac_pro.step(torch.as_tensor(state, dtype=torch.float32))

            # Perform action
            next_state, reward, done, _ = env.step(action)
            state = next_state

            # Save data
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['episode_reward'] += reward
            episode_data['episode_reward_disc'] += gamma ** (episode_data['episode_timestaps'] - 1) * reward

        output_data.append(episode_data)
    test_file = os.path.join(os.getcwd(), 'res', 'data_test_' + str(algo) + '_' + str(seed) + '_' + str(pert_mode) + '.pickle')
    with open(test_file, 'wb') as handle:
        pickle.dump({'test': output_data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Data for ', algo, ' seed ', seed, ' and ', pert_mode, ' saved successfully')


def process_train(algo, seed):
    n_itr = 2000
    data_vals = os.path.join(os.getcwd(), 'res', 'data_training_' + str(algo) + '_' + str(seed) + '.pickle')
    nn_file = os.path.join(os.getcwd(), 'res', 'nn_data_' + str(algo) + '_' + str(seed))
    alg = init_algo_train(algo, seed)
    if algo == 'ppo':
        alg.train(steps_per_epoch=10000, epochs=n_itr, data_vals=data_vals, nn_file=nn_file)
    elif algo == 'ppo_adv_single':
        alg.train(steps_per_epoch=10000, epochs=n_itr, data_vals=data_vals, nn_file=nn_file)
    elif algo == 'ppo_adv_double':
        alg.train(steps_per_epoch=10000, epochs=n_itr, data_vals=data_vals, nn_file=nn_file)
    else:
        raise RuntimeError('Algorithm not recognized')


if __name__ == '__main__':
    train = False
    test = False
    n_episodes_test = 100
    plot = True
    n_seeds = 10
    threads = 40
    algos = ['ppo_adv_single', 'ppo_adv_double']
    params_vector = [np.array([0.5, 5, 1]), np.array([1, 5, 3]), np.array([1, 5, 3]), np.array([0.1, np.pi / 4, 1])]
    pert_mode_vector = ['swirl', 'current_v', 'current_h', 'const']
    #params_vector = [np.array([0.1, np.pi / 4, 1])]
    #pert_mode_vector = ['const']
    if train:
        _ = Parallel(n_jobs=threads, verbose=5)(delayed(process_train)(algo=algo, seed=seed)
                                                for algo in algos for seed in range(n_seeds))
        algos.append('ppo')  # PPO has a different training!

        def process_train_ppo(seed, pmi):
            n_itr = 2000
            pert_mode = pert_mode_vector[pmi]
            pert_params = params_vector[pmi]
            data_vals = os.path.join(os.getcwd(), 'res', 'data_training_ppo_' + str(pert_mode) + '_' + str(seed) + '.pickle')
            nn_file = os.path.join(os.getcwd(), 'res', 'nn_data_ppo' + str(pert_mode) + '_' + str(seed))

            _, gamma, _ = default_params(adv=False, double=False)

            if pert_mode is 'swirl':
                env, env_fn = UuvPertSingle(), UuvPertSingle
            elif pert_mode is 'current_h':
                env, env_fn = UuvPertSingleHC(), UuvPertSingleHC
            elif pert_mode is 'current_v':
                env, env_fn = UuvPertSingleVC(), UuvPertSingleVC
            else:
                env, env_fn = UuvPertSingleC(), UuvPertSingleC
            print(pert_mode, pert_params, env.pert_params)
            assert np.linalg.norm(pert_params - env.pert_params) < 1e-3  # Assert params are right!
            kwargs = {"actor_critic": coreppo.MLPActorCritic,
                      "ac_kwargs": dict(hidden_sizes=[64] * 2),
                      "seed": seed,
                      "gamma": gamma,
                      "clip_ratio": 0.2,
                      "pi_lr": 3e-4,
                      "vf_lr": 1e-3,
                      "train_pi_iters": 80,
                      "train_v_iters": 80,
                      "lam": 0.97,
                      "target_kl": 0.01
                      }
            alg = PPO(env_fn, **kwargs)
            alg.train(steps_per_epoch=10000, epochs=n_itr, data_vals=data_vals, nn_file=nn_file)

        _ = Parallel(n_jobs=threads, verbose=5)(delayed(process_train_ppo)(seed=seed, pmi=pmi)
                                                for seed in range(n_seeds) for pmi in range(len(pert_mode_vector)))
    else:
        algos.append('ppo')  # PPO has a different training!

    algos.append('simple')  # Add baseline (needs not training!)
    if test:
        initial_states_file = os.path.join(os.getcwd(), 'res', 'initial_states.pickle')
        if not os.path.exists(initial_states_file):
            env, _, _ = default_params(adv=False, double=False)
            initial_states = [env.reset() for _ in range(n_episodes_test)]
            with open(initial_states_file, 'wb') as handle:
                pickle.dump({'initial_states': initial_states}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for algo in algos:
            for seed in range(n_seeds):
                for i in range(len(params_vector)):
                    test_algo(algo=algo, seed=seed, pert_mode=pert_mode_vector[i], pert_params=params_vector[i])
    if plot:
        data_comp = dict.fromkeys(algos)  # To obtain the comparison plot!
        for algo in algos:
            data_comp[algo] = {}
        ## TRAINING VALUES
        '''
        for i, algo in enumerate(algos):
            if algo is not 'simple':
                if algo is 'ppo':
                    _, gamma, env_fn = default_params(adv=False, double=False)
                    seed = 0
                    kwargs = {"actor_critic": coreppo.MLPActorCritic,
                              "ac_kwargs": dict(hidden_sizes=[64] * 2),
                              "seed": seed,
                              "gamma": gamma,
                              "clip_ratio": 0.2,
                              "pi_lr": 3e-4,
                              "vf_lr": 1e-3,
                              "train_pi_iters": 80,
                              "train_v_iters": 80,
                              "lam": 0.97,
                              "target_kl": 0.01
                              }
                    alg = PPO(env_fn, **kwargs)
                    # Training values
                    for pert_mode in pert_mode_vector:
                        data_vals = [
                            os.path.join(os.getcwd(), 'res', 'data_training_ppo_' + str(pert_mode) + '_' + str(seed) + '.pickle')
                            for seed in range(n_seeds)]
                        tikz_name = os.path.join(os.getcwd(), 'training_ppo_' + str(pert_mode) + '.tex')
                        alg.plot_training(data_vals, tikz_name=tikz_name)
                else:
                    alg = init_algo_train(algo)
                    # Training values
                    data_vals = [os.path.join(os.getcwd(), 'res', 'data_training_' + str(algo) + '_' + str(seed) + '.pickle')
                                 for seed in range(n_seeds)]
                    tikz_name = os.path.join(os.getcwd(), 'training_' + str(algo) + '.tex')
                    alg.plot_training(data_vals, tikz_name=tikz_name)
        '''
        for pmi, pert_mode in enumerate(pert_mode_vector):

            rwd = [0 for _ in algos]
            rwd_d = [0 for _ in algos]
            sample_trajs = [[] for _ in algos]

            for i, algo in enumerate(algos):

                trained = []
                for seed in range(n_seeds):
                    test_file = os.path.join(os.getcwd(), 'res', 'data_test_' + str(algo) + '_' + str(seed) + '_' + str(pert_mode) + '.pickle')
                    with open(test_file, 'rb') as handle:
                        trained.append(pickle.load(handle)['test'])
                avg_rwd = np.array([np.mean(np.array([trained[seed][j]['episode_reward']
                                                      for j in range(len(trained[seed]))])) for seed in range(n_seeds)])
                avg_rwd_d = np.array([np.mean(np.array([trained[seed][j]['episode_reward_disc']
                                                        for j in range(len(trained[0]))])) for seed in range(n_seeds)])
                rwd[i] = avg_rwd
                rwd_d[i] = avg_rwd_d
                best_rwd = int(np.argmax(avg_rwd_d))
                for j in range(len(trained[best_rwd])):
                    sample_trajs[i].append(trained[best_rwd][j]['states'])
                best_reward = np.array([trained[best_rwd][j]['episode_reward_disc']
                                        for j in range(len(trained[best_rwd]))])
                best_rwds = np.argsort(rwd[i])[-3:].tolist()
                best_rewards = np.array([np.array([trained[seed][j]['episode_reward_disc']
                                        for j in range(len(trained[best_rwd]))]) for seed in best_rwds])
                data_comp[algo][pert_mode] = {'best_reward': best_reward, 'best_rewards': best_rewards}

            for i, algo in enumerate(algos):
                # plt.plot(rwd[i], label='Rwd not disc ' + str(algo))
                plt.plot(rwd_d[i], label='Rwd disc ' + str(algo))
            plt.ylabel('Reward')
            plt.xlabel('Seed')
            plt.title('Evaluation reward ' + str(pert_mode))
            plt.legend(loc='best')
            plt.savefig('rwd_' + str(pert_mode) + '.png')
            save('rwd_' + str(pert_mode) + '.tex')
            plt.show()

            color = ['r', 'g', 'b', 'k', 'c']
            for i, algo in enumerate(algos):
                traj = np.squeeze(np.array(sample_trajs[i][0]))
                plt.plot(traj[:, 0], traj[:, 1], color=color[i], label=str(algo))
                plt.plot(traj[0, 0], traj[0, 1], 'o', color=color[i])
                for j in range(10):  # Number of trajs per algorithm
                    traj = np.squeeze(np.array(sample_trajs[i][j]))
                    plt.plot(traj[:, 0], traj[:, 1], color=color[i])
                    plt.plot(traj[0, 0], traj[0, 1], 'o', color=color[i])
            # Add perturbation
            pert = PerturbationModel(mode=pert_mode, params=params_vector[pmi])  # To extract perturbation
            env = UuvPertSingle()
            npp = 20  # Points to represent the policy
            x = y = np.linspace(env.observation_space.low[0], env.observation_space.high[0], npp)
            X, Y = np.meshgrid(x, y)
            observations = np.vstack([np.ravel(X), np.ravel(Y), np.zeros(npp ** 2), np.zeros(npp ** 2)]).T
            if env.pert_mode is not None:
                px, py = pert.perturbation(np.ravel(X), np.ravel(Y))
                plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps,
                           py.reshape(Y.shape) + np.finfo(float).eps)
            plt.ylabel('y')
            plt.xlabel('x')
            plt.title('Evaluation trajs ' + str(pert_mode))
            plt.legend(loc='best')
            plt.savefig('trajs_' + str(pert_mode) + '.png')
            save('trajs_' + str(pert_mode) + '.tex')
            plt.show()

        ## Data comp plot
        mean = np.zeros((len(algos), len(pert_mode_vector), 2))
        std = np.zeros((len(algos), len(pert_mode_vector), 2))
        for pmi, pert_mode in enumerate(pert_mode_vector):
            for i, algo in enumerate(algos):
                m = np.mean(data_comp[algo][pert_mode]['best_reward'])
                s = np.std(data_comp[algo][pert_mode]['best_reward'])
                mean[i, pmi, 0] = m
                std[i, pmi, 0] = s
                print(pert_mode, algo, ' best ', m, s)

                m = np.mean(data_comp[algo][pert_mode]['best_rewards'])
                s = np.std(data_comp[algo][pert_mode]['best_rewards'])
                mean[i, pmi, 1] = m
                std[i, pmi, 1] = s
                print(pert_mode, algo, ' best 3 ', m, s)

        x = np.arange(len(pert_mode_vector))
        colors = ['r', 'g', 'b', 'k']
        for i, algo in enumerate(algos):
            plt.errorbar(x, mean[i, :, 0], yerr=std[i, :, 0], color=colors[i], linestyle='-', label=algo)
            x = x + 0.05
            plt.errorbar(x, mean[i, :, 1], yerr=std[i, :, 1], color=colors[i], linestyle=':')
            x = x + 0.05
        plt.xlabel('Pert')
        plt.ylabel('Rwd')
        plt.title('Performance comparison')
        plt.legend(loc='best')
        plt.xticks(range(len(pert_mode_vector)), pert_mode_vector, size='small')
        save('comparison_algos.tex')
        plt.savefig('comparison_algos.png')
        plt.show()
        print('')



