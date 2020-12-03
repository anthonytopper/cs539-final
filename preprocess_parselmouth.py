import numpy as np
import parselmouth

# parselmouth library https://parselmouth.readthedocs.io/en/stable/

RATIO = 10

def preproc_intensity(x,r):
    
    features = []
    

    snd = parselmouth.Sound(x,r)

    intensity = snd.to_intensity()
    features.append(signorm(intensity.values[0]))
    
    
    spec = snd.to_spectrogram()
    spec = signorm(spec.values.T)
    features.append([x.max() for x in spec][::4])
    
    harmonicity = snd.to_harmonicity()
    features.append([harmonicity.get_value(t) for t in np.arange(0,harmonicity.get_total_duration(),RATIO/r)])

    pitch = snd.to_pitch()
    features.append([pitch.get_value_at_time(t) for t in np.arange(0,pitch.get_total_duration(),RATIO/r)])
    
    length = min([len(f) for f in features])
    
    features = [filterNaN(f) for f in features]
    
    return [f[0:length] for f in features]

def signorm(x):
    return x / np.max(x)

def filterNaN(arr):
    return [x if not np.isnan(x) else 0 for x in arr]