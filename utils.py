import numpy as np
from brian2 import *

def generate_poisson_spikes(image_pixels, duration_ms, max_rate_hz=63.0, dt_ms=0.1):
    """
    Fast NumPy vectorized Poisson spike generation.
    Guarantees no multiple spikes per time step (dt).
    """
    # Normalize pixels (0-255) and convert to firing rates
    rates = (image_pixels / 255.0) * max_rate_hz
    
    indices = []
    times = []
    
    # Calculate how many dt bins exist in our duration
    num_bins = int(duration_ms / dt_ms)
    
    for i, rate in enumerate(rates):
        if rate > 0:
            # Expected number of spikes
            expected_spikes = (rate * duration_ms) / 1000.0
            
            # Generate actual number of spikes from Poisson distribution
            num_spikes = np.random.poisson(expected_spikes)
            
            # Safety catch: can't have more spikes than available time bins
            num_spikes = min(num_spikes, num_bins)
            
            if num_spikes > 0:
                # randomly select unique time bins (replace=False prevents duplicates)
                chosen_bins = np.random.choice(num_bins, size=num_spikes, replace=False)
                
                # Convert bin indices back to millisecond times
                spike_times = chosen_bins * dt_ms
                
                indices.extend([i] * num_spikes)
                times.extend(spike_times)
                
    return np.array(indices), np.array(times) * ms