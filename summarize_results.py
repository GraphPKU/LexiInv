import wandb
import numpy as np
import itertools

# Initialize the wandb API
api = wandb.Api()

# Specify your project and entity (team)
entity = "graphpku"  # Replace with your entity/team name
project = "lexiinv_9_16"  # Replace with your project name

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

models1 = ['hash-GIN', 'diff_one-DiffGIN']
models2 = ['hash-DeepSet', 'counts-DeepCount']
datasets = ['MUTAG', 'PROTEINS', 'PTC_MR']
combinations1 = list(itertools.product(datasets, models1))
combinations_as_strings1 = ['-'.join(combination) for combination in combinations1]
combinations2 = list(itertools.product(datasets, models2))
combinations_as_strings2 = ['-'.join(combination) for combination in combinations2]
print(combinations_as_strings1)
print(combinations_as_strings2)
combinations = combinations_as_strings1 + combinations_as_strings2

# Initialize lists to store the values
metric_names = ["Loss", "Train", "Test"]  # Replace with the name of the metric you want to calculate mean and std for
all_metrics = dict()
for com in combinations:
    all_metrics[com] = []

# Iterate through the runs and extract the desired metric
for run in runs:
    if run.name in combinations:
        results = []
        for metric_name in metric_names:
            if metric_name in run.summary:
                results.append(run.summary[metric_name])
        all_metrics[run.name].append(results)

for com in combinations:
    print(com + ' ' + ','.join(metric_names))
    results = np.array(all_metrics[com])
    mean_value = np.mean(results, 0)
    std_value = np.std(results, 0)
    for i, metric in enumerate(metric_names):
        print('%.4f$\pm$%.4f'%(np.around(mean_value[i], 4), np.around(std_value[i], 4)))

