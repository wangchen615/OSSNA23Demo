import argparse
import io
import base64
import json
import random
import time
import requests

from datetime import datetime
from PIL import Image
from datasets import load_dataset
from prometheus_api_client import PrometheusConnect

image_path = "./images/"

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--exp-name", type=str, required=True, help="experiment name")
    group.add_argument("--num-tests", type=int, required=False, default=5, help="number of tests to run for configuration")
    group.add_argument("--metric-endpoint", type=str, required=False,
                       default="http://localhost:9090",
                       help="a csv file with list of prompts for text generation")
    group.add_argument("--output-dir", type=str, required=False,
                       default="./rsts/",
                       help="an output directory to save the results")

    return parser.parse_args()

def get_datasets(data_file_path):
    dataset = load_dataset(data_file_path)
    # Convert HuggingFace dataset to list of strings
    training_dataset = dataset['train'].data['Prompt']
    all_training_records = [str(element) for element in training_dataset]
    return all_training_records

def get_prompts(all_records, num_tests):
    sampled_indexes = random.sample(range(len(all_records)), num_tests)

    # Get the sampled data
    prompts = [all_records[i] for i in sampled_indexes]
    return prompts, sampled_indexes

def get_metrics(prom, query):
    result = prom.custom_query(query=query)
    return result

def main():
    args = get_args()
    url = "http://{}:{}".format(args.host, args.port)
    prom_url = args.metric_endpoint
    output_dir = args.output_dir
    exp_name = args.exp_name

    num_tests = args.num_tests

    # Energy Consumption Metrics from DCGM exporter
    prom = PrometheusConnect(url=prom_url, disable_ssl=True)
    container = 'stablediffusion'
    total_energy_consumption_metric = 'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION'
    gpu_frequency_metric = 'DCGM_FI_DEV_SM_CLOCK'
    gpu_utilization_metric = 'DCGM_FI_DEV_GPU_UTIL'
    total_energy_consumption_query = f'{total_energy_consumption_metric}{{exported_container="{container}"}}'
    gpu_frequency_query = f'avg_over_time({gpu_frequency_metric}{{exported_container="{container}"}}[1m])'
    gpu_utilization_query = f'avg_over_time({gpu_utilization_metric}{{exported_container="{container}"}}[1m])'

    # Read the prompts from the dataset file
    data_file_path = "Gustavosta/Stable-Diffusion-Prompts"
    all_records = get_datasets(data_file_path)

    prompts, prompts_indices = get_prompts(all_records, num_tests)

    # Initialize results for energy and latencies
    latency_results = []

    before_energy = get_metrics(prom, total_energy_consumption_query)
    for idx, prompt in zip(prompts_indices, prompts):
        payload = {
            "prompt": prompt,
            "steps": 40,
            "width": 1280,
            "height": 960
        }
        before_query_time = time.time()
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        r = response.json()
        query_rsp_time = time.time() - before_query_time
        print("The query time is {}".format(query_rsp_time))
        latency_results.append(query_rsp_time)
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",", 1)[0])))

        # Save the image to a local folder
        image.save(image_path + 'SD-' + str(idx) + '.jpg', 'JPEG')

    print("====================== SD results for {} queries =================".format(num_tests))
    after_energy = get_metrics(prom, total_energy_consumption_query)
    delta_energy = int(after_energy[0]['value'][1]) - int(before_energy[0]['value'][1])
    print("The total energy consumption for {} queries is {} mJ".format(num_tests, delta_energy))
    ave_query_latency = sum(latency_results)/len(latency_results)
    print("The average query latency is {} seconds.".format(ave_query_latency))

    gpu_frequency = get_metrics(prom, gpu_frequency_query)
    ave_gpu_frequency_val = float(gpu_frequency[0]['value'][1])
    gpu_utilization = get_metrics(prom, gpu_utilization_query)
    ave_gpu_utilization_val = float(gpu_utilization[0]['value'][1])
    print("The GPU frequency is {} KHz".format(ave_gpu_frequency_val))
    print("The GPU utilization is {}%".format(ave_gpu_utilization_val))

    results = {
        "energy": delta_energy,
        "ave_query_latency": ave_query_latency,
        "gpu_frequency": ave_gpu_frequency_val,
        "gpu_utilization": ave_gpu_utilization_val,
        "all_query_latencies": latency_results
    }

    # Open a file for writing
    # Get the current time
    now = datetime.now()

    # Format the time as a string in the format "DD-hh-mm-ss"
    formatted_time = now.strftime("%m-%d-%H-%M")

    results_file_path = output_dir + exp_name + "-" + formatted_time + "-" + str(num_tests) + "queries.json"
    with open(results_file_path, 'w') as file:
        # Dump JSON data to the file
        json.dump(results, file)


if __name__ == "__main__":
    main()