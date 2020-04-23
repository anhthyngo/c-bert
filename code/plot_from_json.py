import analyze
import json
import os

data_dir = r"C:\Users\Willi\Documents\NYU\2020 Spring\NLU\Project\Projects\MetaLearning\Results\Baseline_20200423\\"
data_file = os.path.join(data_dir, r"baseline_baseline_24b_4gpu_10k_noprev_3e-5lr_bert-base-uncased_20200423_1644.json")

with open(data_file, 'r') as file:
    data = json.load(file)
    
print(data)

fig = analyze.plot_learning(data, x_tick_int = 2000, iterations = 10000, y_tick_int=10)
fig.show()
fig.savefig(os.path.join(data_dir, "baseline.png"))