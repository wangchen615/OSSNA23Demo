import argparse
import csv
import random
import math
import time

import numpy as np
import requests
from prometheus_api_client import PrometheusConnect

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--exp-name", type=str, required=True, help="experiment name")
    group.add_argument("--dataset", type=str, required=False, default="./datasets/reddit_rarepuppers_politics_2000_context.csv", help="a csv file with list of prompts for text generation")
    group.add_argument("--num-tests", type=int, required=False, default=100, help="number of tests to run for each batch size and token length")
    group.add_argument("--metric-endpoint", type=str, required=False,
                       default="http://localhost:9090",
                       help="a csv file with list of prompts for text generation")
    group.add_argument("--output-dir", type=str, required=False,
                       default="./rsts/",
                       help="an output directory to save the results")

    return parser.parse_args()


def generate(url: str, prompts, max_tokens):
    url = url + "/generate/"

    request_body = {
        "text": prompts,
        "max_new_tokens": max_tokens,
    }
    response = requests.post(url=url, json=request_body, verify=False)
    result = response.json()
    return result

def get_datasets(data_file_path):
    all_records = []

    with open(data_file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            all_records.append(row)

    return all_records

def get_prompts(all_records, num_prompts):
    num_records = len(all_records)
    if num_records > num_prompts:
        random_idx = random.sample(range(1, num_records), num_prompts)
        print("Sampled prompts: {}".format(random_idx))
        prompts = [all_records[i][0] for i in random_idx]
    else:
        prompts = []

    return prompts

def get_metrics(prom, query):
    result = prom.custom_query(query=query)
    return result

def main():
    args = get_args()
    url = "http://{}:{}".format(args.host, args.port)
    dataset_file = args.dataset
    prom_url = args.metric_endpoint
    output_dir = args.output_dir
    exp_name = args.exp_name

    batch_sizes = [2 ** i for i in range(5)]
    num_tests = args.num_tests
    num_tokens = [20*i for i in range(1, 10)]

    # All results
    latency_results = np.zeros((len(num_tokens), len(batch_sizes), num_tests))
    energy_results = np.zeros((len(num_tokens), len(batch_sizes)))
    freq_results = np.zeros((len(num_tokens), len(batch_sizes)))
    util_results = np.zeros((len(num_tokens), len(batch_sizes)))

    # Energy Consumption Metrics from DCGM exporter
    prom = PrometheusConnect(url=prom_url, disable_ssl=True)
    container = 'bloom'
    total_energy_consumption_metric = 'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION'
    gpu_frequency_metric = 'DCGM_FI_DEV_SM_CLOCK'
    gpu_utilization_metric = 'DCGM_FI_DEV_GPU_UTIL'
    total_energy_consumption_query = f'{total_energy_consumption_metric}{{exported_container="{container}"}}'
    gpu_frequency_query = f'avg_over_time({gpu_frequency_metric}{{exported_container="{container}"}}[1m])'
    gpu_utilization_query = f'avg_over_time({gpu_utilization_metric}{{exported_container="{container}"}}[1m])'

    # Read the prompts from the dataset file
    all_records = get_datasets(dataset_file)

    for num_token in num_tokens:
        for batch_size in batch_sizes:
            batch_idx = int(math.log2(batch_size))
            token_idx = int(num_token / 20 - 1)
            before_energy = get_metrics(prom, total_energy_consumption_query)
            start_time = time.time()
            for i in range(num_tests):
                prompts = get_prompts(all_records, batch_size)
                result = generate(url, prompts, num_token)
                print("The total time taken for {} queries is {}".format(batch_size, result['total_time_taken']))
                per_query_latency_value = float(result['total_time_taken'].split(' ')[0]) / batch_size
                print("The per query latency is {}".format(per_query_latency_value))

                # Record the latency and energy consumption results
                latency_results[token_idx][batch_idx][i] = per_query_latency_value
            after_energy = get_metrics(prom, total_energy_consumption_query)
            delta_energy = int(after_energy[0]['value'][1]) - int(before_energy[0]['value'][1])
            print("The total energy consumption for {} queries is {}".format(batch_size, delta_energy))
            energy_results[token_idx][batch_idx] = delta_energy
            query_time = time.time() - start_time
            print("The total time taken for {} queries is {}".format(num_tests, query_time))

            gpu_frequency = get_metrics(prom, gpu_frequency_query)
            ave_gpu_frequency_val = float(gpu_frequency[0]['value'][1])
            gpu_utilization = get_metrics(prom, gpu_utilization_query)
            ave_gpu_utilization_val = float(gpu_utilization[0]['value'][1])
            print("The GPU frequency is {} KHz".format(ave_gpu_frequency_val))
            print("The GPU utilization is {}%".format(ave_gpu_utilization_val))
            freq_results[token_idx][batch_idx] = ave_gpu_frequency_val
            util_results[token_idx][batch_idx] = ave_gpu_utilization_val


    # Flatten out the results in the first dimension (token idx) and save the raw data
    latency_results_2D = latency_results.reshape(-1, latency_results.shape[-1])
    np.savetxt(output_dir + exp_name + '-latency_results_raw.csv', latency_results_2D)

    # Calculate the mean of num_tests for each batch size and token size and save the results
    latency_results_mean = np.mean(latency_results, axis=-1)
    np.savetxt(output_dir + exp_name + '-latency_mean.csv', latency_results_mean, delimiter=",")
    np.savetxt(output_dir + exp_name + '-energy_total.csv', energy_results, delimiter=",")
    np.savetxt(output_dir + exp_name + '-GPU_freq.csv', freq_results, delimiter=",")
    np.savetxt(output_dir + exp_name + '-GPU_util.csv', util_results, delimiter=",")




if __name__ == "__main__":
    main()