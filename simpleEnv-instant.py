

import numpy as np
from gym_anm import MPCAgentConstant
from gym_anm import MPCAgentPerfect
from simpleEnv import SimpleEnv


class SimpleEnvInst(SimpleEnv):
    """The :code:`SimpleEnv-v0` task."""

    def __init__(self):
        observation = 'state'  # fully observable environment
        K = 1
        delta_t = 0.25         # 15 minutes between timesteps
        gamma = 0.995
        lamb = 100
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds,
                         costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()

    def init_state(self):

        n_dev = self.simulator.N_device              # number of devices
        n_des = self.simulator.N_des                 # number of DES units
        n_gen = self.simulator.N_non_slack_gen       # number of non-slack generators
        # N_vars = 2 * n_dev + n_des + n_gen + self.K  # size of state vectors - calculated based on number of devices, gen and storage units - integer
        # N_vars is equals to 11 in this case
        # state = np.random.rand(N_vars)  

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.randint(0, int(24 / self.delta_t))
        state[-1] = t_0

        # Load (P, Q) injections.
        for dev_id, p_load in zip([1], self.P_loads):
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = \
                p_load[t_0] * self.simulator.devices[dev_id].qp_ratio

        # Non-slack generator (P, Q) injections.
        for idx, (dev_id, p_max) in enumerate(zip([3], self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = \
                self.np_random.uniform(self.simulator.devices[dev_id].q_min,
                                       self.simulator.devices[dev_id].q_max)

        # Energy storage unit.
        for idx, dev_id in enumerate([2]):
            state[2 * n_dev + idx] = \
                self.np_random.uniform(self.simulator.devices[dev_id].soc_min,
                                       self.simulator.devices[dev_id].soc_max)

        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        vars.append(aux)

        return np.array(vars)

    def reset(self, date_init=None):
        obs = super().reset()

        # Reset the time of the day based on the auxiliary variable.
        date = self.date
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)

        return obs


def _get_load_time_series():
    """Return the fixed 24-hour time-series for the load injections."""

    # Device 1 (residential load).
    s1 = - np.ones(25)
    s12 = np.linspace(-1.5, -4.5, 7)
    s2 = - 5 * np.ones(13)
    s23 = np.linspace(-4.625, -2.375, 7)
    s3 = - 2 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P_loads = P1.reshape(1,96)

    assert P_loads.shape == (1, 96)

    return P_loads


def _get_gen_time_series():
    """Return the fixed 24-hour time-series for the generator maximum production."""

    # Device 4 (wind farm).
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P4 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P_maxs = P4.reshape(1,96)

    assert P_maxs.shape == (1, 96)

    return P_maxs


if __name__ == '__main__':
    import time

    env = SimpleEnvInst()
    env.reset()
    print('Environment reset and ready.')
    agent = MPCAgentConstant(env.simulator, env.action_space, env.gamma, # This policy assumes constant demand and generation during the optimization horizon
                             safety_margin=0.96, planning_steps=10)      #  planning_steps = The number of stages (time steps) taken into account in the optimization problem.
    
    max_r = 0

    T = 10000
    start = time.time()
    for i in range(T):
        action = agent.act(env)
        obs, reward, done, _ = env.step(action)
        if(reward > max_r):
            max_r = reward
            obs = obs
        print(f'reward at t ={i} is {reward:.3}')
    
        env.render()
        time.sleep(0.5)

    print('maximum reward = ', max_r)
    print('Done with {} steps in {} seconds!'.format(T, time.time() - start))
env.close()
