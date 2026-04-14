# Excitatory Neurons
eqs_e = '''
        dv/dt = ((E_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_e : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        dtheta/dt = -theta / tau_theta : volt
        v_thresh = E_thresh + theta : volt
        '''

# Disable homeostasis during the inference
eqs_e_test = '''
        dv/dt = ((E_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_e : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        theta : volt
        v_thresh = E_thresh + theta : volt
        '''

# Inhibitory Neurons
eqs_i = '''
        dv/dt = ((I_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_i : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        '''

# -----------------------------------------------------------------------------


# The fours STDP rules described in the paper:

# 1) Power-Law Weight Dependence STDP
# this is the one mainly used in the paper, others are alternatives
eqs_stdp_power = '''
        w : 1
        dpre_trace/dt = -pre_trace / tc_pre : 1 (event-driven)
        '''

eqs_stdp_pre_power = '''
        ge_post += w
        pre_trace += 1.0
        '''
eqs_stdp_post_power = '''
        is_ltp = int(pre_trace > x_tar)
        is_ltd = int(pre_trace <= x_tar)
        delta_w = nu * (pre_trace - x_tar)
        
        w = clip(w + is_ltp * delta_w * (wmax - w)**mu + is_ltd * delta_w * (w)**mu, 0.0, wmax)
        '''
# -----------------------------------------------------------------------------


# 2) Exponential Weight Dependence STDP
eqs_stdp_exp = '''
        w : 1
        dx_pre/dt = -x_pre / tc_pre : 1 (event-driven)
        '''

eqs_stdp_pre_exp = '''
        x_pre += 1.0
        '''

eqs_stdp_post_exp = '''
        w = clip(w + nu_post * (x_pre * exp(-beta * w) - x_tar * exp(-beta * (wmax - w))), 0.0, wmax)
        '''
# -----------------------------------------------------------------------------

# 3) Symmetric (Pre-and-Post) STDP
eqs_stdp_sym = '''
        w : 1
        dx_pre/dt = -x_pre / tc_pre : 1 (event-driven)
        dx_post/dt = -x_post / tc_post : 1 (event-driven)
        '''

eqs_stdp_pre_sym = '''
        x_pre += 1.0
        w = clip(w - nu_pre * x_post * (w**mu_sym), 0.0, wmax)
        '''

eqs_stdp_post_sym = '''
        x_post += 1.0
        w = clip(w + nu_post * (x_pre - x_tar) * (wmax - w)**mu_sym, 0.0, wmax)
        '''
# -----------------------------------------------------------------------------


# 4) Triplet STDP Rule
eqs_stdp_triplet = '''
        w : 1
        post2before : 1
        dpre/dt   = -pre / tc_pre : 1 (event-driven)
        dpost1/dt = -post1 / tc_post1 : 1 (event-driven)
        dpost2/dt = -post2 / tc_post2 : 1 (event-driven)
        '''

eqs_stdp_pre_triplet = '''
        pre += 1.0
        w = clip(w - nu_pre * post1, 0.0, wmax)
        '''

eqs_stdp_post_triplet = '''
        post2before = post2
        w = clip(w + nu_post * pre * post2before, 0.0, wmax)
        post1 += 1.0
        post2 += 1.0
        '''


#  Summary of variables
# -----------------------------------------------------------------------------
# 1. Neuron Membrane Dynamics
# -----------------------------------------------------------------------------
# v             : Current membrane voltage (potential) of the neuron.
# E_rest        : Resting membrane potential. The baseline voltage (-65 mV).
# I_rest        : Resting membrane potential. The baseline voltage (-60 mV).
# E_exc         : Equilibrium (reversal) potential for excitatory synapses (0 mV).
# E_inh         : Equilibrium (reversal) potential for inhibitory synapses (-100 mV or -85 mV).
# tau_e         : Membrane time constant for excitatory neurons. Intentionally long (100 ms) to integrate sparse input spikes over time.
# tau_i         : Membrane time constant for inhibitory neurons. Shorter (10 ms) so they can react and suppress other neurons instantly.

# 2. Synaptic Conductance
# -----------------------------------------------------------------------------
# ge            : Current conductance of the excitatory synapses.
# gi            : Current conductance of the inhibitory synapses.
# tau_ge        : Time constant for excitatory conductance decay.
# tau_gi        : Time constant for inhibitory conductance decay.

# 3. Homoeostasis
# -----------------------------------------------------------------------------
# E_thresh      : The static baseline firing threshold (e.g., -52 mV).
# theta         : The adaptive threshold offset. Increases slightly every time the neuron fires to prevent dominance.
# v_thresh      : The actual dynamic threshold (v_thresh_base + theta).
# tau_theta     : Time constant governing how slowly the adaptive offset (theta) decays back to zero over time.

# 4. Synaptic Weights and Traces
# -----------------------------------------------------------------------------
# w             : Current synaptic weight between the pre- and postsynaptic neuron.
# wmax          : Maximum allowable value for the synaptic weight.
# x_pre / pre   : Presynaptic trace. Jumps by 1.0 on input spike, decays exponentially.
# x_post        : Postsynaptic trace (Symmetric rule). Jumps by 1.0 on receiving spike.
# post1 / post2 : Postsynaptic traces (Triplet rule) that decay at different rates.
# post2before   : Snapshot of post2 right before the current spike updates it.
# tc_pre        : Time constant for the decay of the presynaptic trace.
# tc_post(1/2)  : Time constants for the decay of the postsynaptic traces.

# 5. STDP Learning Parameters
# -----------------------------------------------------------------------------
# nu / nu_post  : Learning rate for weight increases (LTP) triggered by a post-spike.
# nu_pre        : Learning rate for weight decreases (LTD) triggered by a pre-spike.
# x_tar         : Target average value of the presynaptic trace. Acts as a penalty offset to disconnect irrelevant inputs.
# mu            : Exponent for power-law weight dependence. Scales how stronglythe current weight restricts further weight growth.
# beta          : Scaling factor for Exponential STDP rule. Determines how sharply the weight change drops off as the weight approaches wmax.
