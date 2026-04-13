from brian2 import*
import numpy as np

from constants import *
from equations import *

def build_network_train():

    # Input Layer
    inp_group = PoissonGroup(n_input, rates=0*Hz, name='inp')

    # Excitatory Layer
    exc_group = NeuronGroup(n_e, eqs_e, threshold='v > v_thresh', reset='v = E_reset; theta += d_theta', 
                            refractory=E_refrac, method='euler', name='exc')
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', reset='v = I_reset', 
                            refractory=I_refrac, method='euler', name='inh')

    exc_group.v = E_rest
    inh_group.v = I_rest
    exc_group.theta = init_theta
    
    # Input -> Excitatory (Using the Power-Law STDP rule)
    S_input_exc = Synapses(inp_group, exc_group, model=eqs_stdp_power,
                           on_pre=eqs_stdp_pre_power, on_post=eqs_stdp_post_power, 
                           method='euler', name='s_inp_exc')

    S_input_exc.connect(p=1.0) # Fully connected
    
    S_input_exc.w = np.load(initial_weights)
    S_input_exc.delay = 'rand() * max_delay' # delay

    # Excitatory -> Inhibitory
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i') # 1-to-1 mapping

    # Inhibitory -> Excitatory (Winner-Take-All)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='gi_post += w_ie', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitor the excitatory spikes
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')

    # Network object
    net = Network(inp_group, exc_group, inh_group, 
                  S_input_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor)
    
    print("Network built successfully!")
    return net, spike_monitor, inp_group
