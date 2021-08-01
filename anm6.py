"""The base class for a 6-bus and 7-device :code:`gym-anm` environment."""

import datetime as dt
import numpy as np
from gym_anm import ANMEnv
from rendering import rendering
from utils import random_date

network = {
    'baseMVA': 100,
    'bus': np.array([
    [0, 0, 132, 1., 1.],
    [1, 1, 33, 1.1, 0.9],
    [2, 1, 33, 1.1, 0.9]
]),
    'device': np.array([
    [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None], #slack
    [1, 1, -1, 0.2, 0, -10,  None, None, None, None, None, None, None, None, None], # load
    [2, 1, 3, None, 50, -50, 50, -50, 30, -30, 25, -25, 100, 0, 0.9],# storage
    [3, 1, 2, None, 30, 0, 30, -30, 20, None, 15, -15, None, None, None] # wind power
]),
    'branch': np.array([
    [0, 1,  0.03,  0.022, 0., 25, 1, 0],
    [1, 2,   0.03,  0.022, 0., 25, 1, 0],
])
}

class ANM6(ANMEnv):
    

    metadata = {'render.modes': ['human']}

    def __init__(self, observation, K, delta_t, gamma, lamb,
                 aux_bounds=None, costs_clipping=(None, None), seed=None):

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

        # Rendering variables.
        self.network_specs = self.simulator.get_rendering_specs()
        self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        self.date = None
        self.date_init = None
        self.year_count = 0
        self.skipped_frames = None
        self.render_mode = None
        self.is_rendering = False


    def render(self, mode='human', skip_frames=0):

        if self.render_mode is None:
            if mode not in ['human']:
                raise NotImplementedError()

            # Render the initial image of the distribution network.
            self.render_mode = mode
            self.skipped_frames = 0
            rendered_network_specs = ['dev_type', 'dev_p', 'dev_q', 'branch_s',
                                      'bus_v', 'des_soc']
            specs = {s: self.network_specs[s] for s in rendered_network_specs}
            self._init_render(specs)

            # Render the initial state.
            self.render(mode=mode, skip_frames=skip_frames)

        else:
            self.skipped_frames = (self.skipped_frames + 1) % (skip_frames + 1)
            if self.skipped_frames:
                return

            full_state = self.simulator.state
            dev_p = list(full_state['dev_p']['MW'].values())
            dev_q = list(full_state['dev_q']['MVAr'].values())
            branch_s = list(full_state['branch_s']['MVA'].values())
            des_soc = list(full_state['des_soc']['MWh'].values())
            gen_p_max = list(full_state['gen_p_max']['MW'].values())
            bus_v_magn = list(full_state['bus_v_magn']['pu'].values())
            costs = [self.e_loss, self.penalty]
            network_collapsed = not self.simulator.pfe_converged

            self._update_render(dev_p, dev_q, branch_s, des_soc,
                                gen_p_max, bus_v_magn, costs, network_collapsed)

    def step(self, action):
        obs, r, done, info = super().step(action)

        # Increment the date (for rendering).
        self.date += self.timestep_length

        # Increment the year count.
        self.year_count = (self.date - self.date_init).days // 365

        return obs, r, done, info

    def reset(self, date_init=None):
        # Save rendering setup to restore after the reset().
        render_mode = self.render_mode

        obs = super().reset()

        # Restore the rendering setup.
        self.render_mode = render_mode

        # Reset the date (for rendering).
        self.year_count = 0
        if date_init is None:
            self.date_init = random_date(self.np_random, 2020)
        else:
            self.date_init = date_init
        self.date = self.date_init

        return obs

    def reset_date(self, date_init):
        """Reset the date displayed in the visualization (and the year count)."""
        self.date_init = date_init
        self.date = date_init

    def _init_render(self, network_specs):
        """
        Initialize the rendering of the environment state.

        Parameters
        ----------
        network_specs : dict of {str : list}
            The operating characteristics of the electricity distribution network.
        """

        # Set visualization title to class name.
        title = type(self).__name__

        # Convert dict of network specs into lists.
        dev_type = list(network_specs['dev_type'].values())
        ps, qs = [], []
        for i in network_specs['dev_p'].keys():
            p_min_max = [network_specs['dev_p'][i]['MW'][j] for j in [0, 1]]
            ps.append(np.max(np.abs(p_min_max)))
            q_min_max = [network_specs['dev_q'][i]['MVAr'][j] for j in [0, 1]]
            qs.append(np.max(np.abs(q_min_max)))
        branch_rate = []
        for br in network_specs['branch_s'].keys():
            branch_rate.append(network_specs['branch_s'][br]['MVA'][1])
        bus_v_min, bus_v_max = [], []
        for i in network_specs['bus_v'].keys():
            bus_v_min.append(network_specs['bus_v'][i]['pu'][0])
            bus_v_max.append(network_specs['bus_v'][i]['pu'][1])
        soc_max = []
        for i in network_specs['des_soc'].keys():
            soc_max.append(network_specs['des_soc'][i]['MWh'][1])

        # Set default costs range if not specified.
        c1 = 100 if self.costs_clipping[0] is None else self.costs_clipping[0]
        c2 = 10000 if self.costs_clipping[1] is None else self.costs_clipping[1]
        costs_range = (c1, c2)

        self.http_server, self.ws_server = \
            rendering.start(title, dev_type, ps, qs, branch_rate,
                            bus_v_min, bus_v_max, soc_max, costs_range)

    def _update_render(self, dev_p, dev_q, branch_s, des_soc, gen_p_max,
                       bus_v_magn, costs, network_collapsed):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        dev_p  : list of float
            The real power injection from each device (MW).
        dev_q : list of float
            The reactive power injection from each device (MW).
        branch_s : list of float
            The apparent power flow in each branch (MVA).
        des_soc : list of float
            The state of charge of each storage unit (MWh).
        gen_p_max : list of float
            The potential real power generation of each RER generator before
            curtailment (MW).
        bus_v_magn : list of float
            The voltage magnitude of each bus (pu).
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        network_collapsed : bool
            True if no load flow solution is found (possibly infeasible); False
            otherwise.
        """
        rendering.update(self.ws_server.address, self.date, self.year_count,
                         dev_p, dev_q, branch_s, des_soc, gen_p_max,
                         bus_v_magn, costs, network_collapsed)

    def close(self):
        """
        Close the rendering.
        """
        rendering.close(self.http_server, self.ws_server)
        self.render_mode = None
