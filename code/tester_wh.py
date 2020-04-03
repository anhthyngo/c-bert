"""
Testing for implementations - Will
"""

import analyze
import numpy as np
from datetime import datetime as dt

# Generate test data
trivia = {"iter":np.arange(11),"val":np.arange(11)}
squad = {"iter":np.arange(11),"val":0.5*np.arange(11)}

print(trivia)
print(squad)

# Test plotting
plot = analyze.plot_learning(trivia, squad, "BERT", iterations=10, max_score=10, x_tick_int=2, y_tick_int=1)

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