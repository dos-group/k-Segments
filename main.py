BASE_DIR = './k-Segments-traces'

import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import pandas as pd
from tsb_resource_allocation.witt_task_model import WittTaskModel
from tsb_resource_allocation.tovar_task_model import TovarTaskModel
from tsb_resource_allocation.simulation import Simulation
from tsb_resource_allocation.k_segments_model import KSegmentsModel
from tsb_resource_allocation.file_events_model import FileEventsModel
from tsb_resource_allocation.default_model import DefaultModel

sns.set_theme(style="darkgrid")


# Helper methods

def get_file_names(directory, number_of_files=-1):
    file_names = [name.rsplit('_', 1)[0] for name in os.listdir(directory) if
                  not os.path.isdir(f"{directory}{name}") and name.endswith("_memory.csv")]
    if number_of_files != -1:
        return file_names[:number_of_files]
    return file_names


def run_simulation(directory, training, test, monotonically_increasing=True, k=4, collection_interval=2):
    # MODELS
    simulations = []

    # KSegments retry: selective
    task_model = KSegmentsModel(k=k, monotonically_increasing=monotonically_increasing)
    simulation = Simulation(task_model, directory, retry_mode='selective', provided_file_names=training)
    simulations.append(simulation)

    # KSegments retry: selective - NO UNDERPREDICTION
    task_model = KSegmentsModel(k=k, monotonically_increasing=monotonically_increasing, time_mode=-1)
    simulation = Simulation(task_model, directory, retry_mode='selective', provided_file_names=training)
    # simulations.append(simulation)

    # KSegments retry: partial
    task_model = KSegmentsModel(k=k, monotonically_increasing=monotonically_increasing)
    simulation = Simulation(task_model, directory, retry_mode='partial', provided_file_names=training)
    simulations.append(simulation)

    # WITT LR MEAN+- TASK MODEL
    task_model = WittTaskModel(mode="mean+-")
    simulation = Simulation(task_model, directory, retry_mode='full', provided_file_names=training)
    simulations.append(simulation)

    # TOVAR TASK MODEL - full retry
    task_model = TovarTaskModel()
    simulation = Simulation(task_model, directory, retry_mode='full', provided_file_names=training)
    simulations.append(simulation)

    # TOVAR TASK MODEL - tovar retry
    task_model = TovarTaskModel()
    simulation = Simulation(task_model, directory, retry_mode='tovar', provided_file_names=training)
    simulations.append(simulation)

    # Default Model
    task_model = DefaultModel()
    simulation = Simulation(task_model, directory, retry_mode='full', provided_file_names=training)
    simulations.append(simulation)

    waste, retries, runtimes = [0 for _ in range(len(simulations))], [0 for _ in range(len(simulations))], [0 for _ in
                                                                                                            range(
                                                                                                                len(simulations))]
    for file_name in test:
        for i, s in enumerate(simulations):
            result = s.execute(file_name, True)
            waste[i] += (result[0] )
            retries[i] += result[1]
            runtimes[i] += (result[2] * collection_interval)

    avg_waste = list(map(lambda w: w, waste))
    avg_retries = list(map(lambda r: r, retries))
    avg_runtime = list(map(lambda r: r, runtimes))

    return avg_waste, avg_retries, avg_runtime


import random


# OUTPUT = ( [Waste: [Witt: 25, Tovar: 25, k-segments:25], [50] , [75]], [Retries], [Runtime])


def split(train_percent, instances, seed=18181):
    r = random.Random(seed)
    trainidx = set()
    while len(trainidx) < int(train_percent * len(instances)):
        trainidx.add(int(len(instances) * r.random()))

    train = [instances[i] for i in sorted(trainidx)]
    data = [instances[i] for i in range(len(instances)) if i not in trainidx]

    return (train, data)


def benchmark_task(task_dir='/eager/markduplicates', base_directory=BASE_DIR, seed=0):
    directory = f'{base_directory}/{task_dir}'
    file_names_orig = []
    file_order = get_file_order(directory)
    if file_order != None:
        file_names_orig = file_order
    else:
        file_names_orig = get_file_names(directory)

    percentages = [0.25, 0.5, 0.75]

    x = []
    y_waste = []
    y_retries = []
    y_runtime = []

    filter_file_names = list(
        filter(lambda x: len(pd.read_csv(f'{directory}/{x}_memory.csv', skiprows=3)) >= 60, file_names_orig))
    if len(filter_file_names) < 16:
        return -1

    filter_file_names = list(
        filter(lambda x: len(pd.read_csv(f'{directory}/{x}_memory.csv', skiprows=3)) >= 4, file_names_orig))
    file_names = sorted(filter_file_names)
    print(f'Usable Data: {len(file_names)}/{len(file_names_orig)}')

    for p in percentages:
        # training = file_names[:i]
        # test = file_names[i:] # file_names[i:] - other mode
        training, test = split(p, file_names, seed)
        # TODO p
        print(f"training: {len(training)}, test: {len(test)}", end="\r", flush=True)
        avg_waste, avg_retries, avg_runtime = run_simulation(directory, training, test, k=4)
        # x.append(i)
        y_waste.append(list(map(lambda w: round(w, 2), avg_waste)))
        y_retries.append(avg_retries)
        y_runtime.append(avg_runtime)

    return (y_waste, y_retries, y_runtime)


def record_file_order(workflow_tasks, base_directory, depth):
    if depth > 1:
        return
    f = open(f"{base_directory}/file_order.txt", "w")
    for task in workflow_tasks:
        basename = os.path.basename(task)
        f.write(f"{basename}\n")
        if depth > 0:
            continue
        record_file_order(get_file_names(task), f'{base_directory}/{basename}', depth + 1)


def get_file_order(base_directory):
    try:
        with open(f'{base_directory}/file_order.txt') as f:
            return f.read().splitlines()
    except:
        return None


import csv

workflow = "eager"
seed = 0
base_directory = f'{BASE_DIR}/{workflow}'
workflow_tasks = []
file_order = get_file_order(base_directory)
file_order = None
if file_order != None:
    workflow_tasks = file_order
    
    

else:
    workflow_tasks = [os.path.join(base_directory, item) for item in os.listdir(base_directory) if
                      os.path.isdir(os.path.join(base_directory, item))]
    workflow_tasks = [task for task in workflow_tasks if len(os.listdir(task)) > 15]
    workflow_tasks = list(map(os.path.basename, workflow_tasks))

categories = ["Wastage", "Retries", "Runtime"]
percentages = ["25%", "50%", "75%"]
methods = ["k-Segments Selective", "k-Segments Partial",
           "Witt", "Tovar-Improved","Tovar", "Default"]


def write_results_csv(resultfile: str, method: str, task: str, setup: str, wastage, retries, runtime):
    if not (os.path.exists(resultfile)):
        with open(resultfile, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ["Method", "Task", "Setup", "Wastage", "Retries", "Runtime"])

    with open(resultfile, 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            [method, task, setup, wastage, retries, runtime])


# 0 = WASTE, 1 = RETRIES, 2 = RUNTIME
for task in workflow_tasks:
    print("Analyze Task: " + task)
    r = benchmark_task(task, base_directory, seed)
    if r == -1:
        continue
    task_name = os.path.basename(task)
    m = ', '.join(map(str, r[0][2]))
    print(f'{task_name}')
    for i, category in enumerate(categories):
        for j, percentage in enumerate(percentages):
            print(f'{category} {percentage}: {r[i][j]}')

    for i, method in enumerate(methods):
        for j, percentage in enumerate(percentages):
            write_results_csv(f"./results/{workflow}.csv", method, task, percentage, r[0][j][i], r[1][j][i], r[2][j][i])
