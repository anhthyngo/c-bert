"""
Testing for implementations - Will
"""

import torch
import analyze
import numpy as np
from datetime import datetime as dt

# ============================= Testing Analyze ==============================
# Generate test data
btrivia = {"iter":np.arange(11),"val":np.arange(11)}
bsquad = {"iter":np.arange(11),"val":0.5*np.arange(11)}
mtrivia = {"iter":np.arange(11),"val":2*np.arange(11)}
msquad = {"iter":np.arange(11),"val":1.5*np.arange(11)}

data = {
        "BERT Trivia":btrivia,
        "BERT SQuAD":bsquad,
        "Meta-BERT Trivia":mtrivia,
        "Meta-BERT SQuAD":msquad
        }

# Test plotting
plot = analyze.plot_learning(data, iterations=10, max_score=20, x_tick_int=2, y_tick_int=10)

# Tryout displaying and saving plot
#
# Datetime string formatting:
# %Y = year
# %m = month
# %d = day
# %H = hour
# %M = minute
plot.show
plot.savefig("./results/test_fig_{}.png".format(dt.now().strftime("%Y%m%d_%H%M")))
# ============================= Testing Analyze ==============================