from brian2 import *
import os

# most of the following constants are found in 
# the orignal Diehl and Cook codebase.

n_input = 784
n_e = 400
n_i = n_e 
time_per_img = 500 * ms
target_weight = 78.0    # Magic number for normalization

# Neuron parameters
E_rest = -65. * mV 
I_rest = -60. * mV 
E_reset = -65. * mV
I_reset = -45. * mV
E_thresh = -52. * mV
I_thresh = -40. * mV
E_refrac = 5. * ms
I_refrac = 2. * ms
E_exc = 0.0 * mV
E_inh = -100.0 * mV
# init_theta = 20.0 * mV
# d_theta = 0.05 * mV
init_theta = 20.0 * mV
d_theta = 0.05 * mV

# synapse weight
w_ei = 10.4
w_ie = 17.0

# Time Constants
tau_e = 100*ms       
tau_i = 10*ms        
tau_ge = 1*ms        
tau_gi = 2*ms        
tau_theta = 1e7*ms   
tau_pre = 20*ms      

# network config
max_weight = 0.3
max_delay = 10 * ms
initial_weights = os.path.join("random", "initial.npy")

# STDP Parameters: Power-Law Weight Dependence
# 
tc_pre = 20 * ms
nu = 0.01
x_tar = 0.15
wmax = 1.0
mu = 1.0

# STDP Parameters: Exponential Weight Dependence
nu_post = 0.01
x_tar = 0.15
wmax = 1.0
beta = 3.0

# STDP Parameters: Symmetric (Pre-and-Post) STDP
tc_pre = 20 * ms
tc_post = 20 * ms
nu_pre = 0.0001
nu_post = 0.01
x_tar = 0.15
wmax = 1.0
mu_sym = 0.9

# STDP parameters (Triplet Rule)
# this was used in the original codebase
tc_pre = 20*ms
tc_post1 = 20*ms
tc_post2 = 40*ms
nu_pre =  0.0001
nu_post = 0.01
wmax = 1.0


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
