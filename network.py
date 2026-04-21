from brian2 import*
import numpy as np

from constants import *
from equations import *

def build_network_train():

    # Input Layer
    # inp_group = PoissonGroup(n_input, rates=0*Hz, name='inp')
    # inp_group = SpikeGeneratorGroup(n_input, indices=np.array([], dtype=int), times=np.array([]) * ms, name='inp')
    inp_group = PoissonGroup(n_input, rates='input_rates(t - batch_start_time, i)', name='inp')
    
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
    S_inp_exc = Synapses(inp_group, exc_group, model=eqs_stdp_power,
                           on_pre=eqs_stdp_pre_power, on_post=eqs_stdp_post_power, name='s_inp_exc')
    # S_input_exc = Synapses(inp_group, exc_group, model=eqs_stdp_triplet,
    #                        on_pre=eqs_stdp_pre_triplet, on_post=eqs_stdp_post_triplet, name='s_inp_exc') 

    S_inp_exc.connect(p=1.0) # Fully connected
    
    S_inp_exc.w = np.load(initial_weights)
    S_inp_exc.delay = 'rand() * max_delay' # delay

    # Excitatory -> Inhibitory
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i') # 1-to-1 mapping

    # Inhibitory -> Excitatory (Winner-Take-All)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='gi_post += w_ie', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitor the excitatory spikes
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')
        
    # @network_operation(dt=time_per_img)
    # def normalize_weights():
    #     # We use [:] to access the actual numpy values of the weights
    #     W = S_inp_exc.w[:].reshape((784, n_e))
    #     col_sums = np.sum(W, axis=0)
    #     col_sums[col_sums == 0] = 1.0
    #     W = W * (target_weight / col_sums)
    #     S_inp_exc.w = W.flatten()
    
    @network_operation(dt=time_per_img)
    def normalize_weights():
        w_array = S_inp_exc.w[:]
        
        # Safely sum the weights entering each excitatory neuron (target index 'j')
        col_sums = np.bincount(S_inp_exc.j, weights=w_array, minlength=n_e)
        col_sums[col_sums == 0] = 1.0 # Prevent division by zero
        
        # Find scaling factors and broadcast them directly to the 1D weight array
        scale_factors = target_weight / col_sums
        S_inp_exc.w = w_array * scale_factors[S_inp_exc.j]

    # # Network object
    # net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  # spike_monitor)
    net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor, normalize_weights)
    
    print("Network built successfully!")
    return net, spike_monitor, inp_group
    # return net, spike_monitor, inp_group, normalize_weights

def build_network_test():
    # Input Layer 
    inp_group = PoissonGroup(n_input, rates='input_rates(t - batch_start_time, i)', name='inp')
    
    # Excitatory Layer : Use eqs_e_test and remove theta += d_theta
    exc_group = NeuronGroup(n_e, eqs_e_test, threshold='v > v_thresh', reset='v = E_reset', 
                            refractory=E_refrac, method='euler', name='exc')
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', reset='v = I_reset', 
                            refractory=I_refrac, method='euler', name='inh')

    exc_group.v = E_rest
    inh_group.v = I_rest
    
    # Input -> Excitatory : NO STDP
    S_inp_exc = Synapses(inp_group, exc_group, model='w : 1', on_pre='ge_post += w', name='s_inp_exc')
    S_inp_exc.connect(p=1.0) # Fully connected
    S_inp_exc.delay = 'rand() * max_delay' #  delays
    
    # Excitatory -> Inhibitory
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i') # 1-to-1 mapping

    # Inhibitory -> Excitatory (Winner-Take-All)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='gi_post += w_ie', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitor the excitatory spikes
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')
        
    # Network object
    net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor)
    
    print("Test Network built successfully!")
    return net, spike_monitor, inp_group