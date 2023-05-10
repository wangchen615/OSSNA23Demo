import matplotlib.pyplot as plt
import numpy as np

exp_folder = "../rsts/"
img_folder = "../plots/"
exp_prefix = "exp2-"
num_tests = 100

def load_and_check_rst_format(rst_file, batch_idx_length, token_idx_length):
    rsts = np.loadtxt(rst_file, delimiter=',')
    row_len, col_len = rsts.shape
    if row_len != token_idx_length:
        print("Data file {} does not have the correct row indices!".format(rst_file))
        return None

    if col_len != batch_idx_length:
        print("Data file {} does not have the correct row indices!".format(rst_file))
        return  None

    return rsts

def plot_latency_and_energy_over_batchsize(latency_rsts, energy_rsts, batch_sizes, num_tokens, power_limits, chosen_token_idx):
    # Plot latency and energy results over batch sizes
    chosen_token_length = num_tokens[chosen_token_idx]

    # Create a new figure and an axis for the first y-axis
    fig, ax1 = plt.subplots()

    # Line colors for different power limits
    line_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    latency_energy_lines = []

    # Plot the first two datasets on the first y-axis
    for i, power_limit in enumerate(power_limits):
        latency_data = latency_rsts[power_limit]
        chosen_token_latency = latency_data[chosen_token_idx]
        latency_line, = ax1.plot(batch_sizes, chosen_token_latency, label=power_limit, color=line_colors[i], linestyle='-')
        latency_energy_lines.append(latency_line)

    # Set labels for the first y-axis
    ax1.set_xlabel('Batch sizes')
    ax1.set_ylabel('Average Query Latency (seconds)'.format(chosen_token_length))

    # Set the title of the plot
    ax1.set_title('Generated Token Length {} words'.format(chosen_token_length))

    # Create the second y-axis, sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot the third and fourth datasets on the second y-axis
    for i, power_limit in enumerate(power_limits):
        energy_data = energy_rsts[power_limit]
        chosen_token_energy = energy_data[chosen_token_idx - 1] / (num_tests * 1000000)
        energy_line, = ax2.plot(batch_sizes, chosen_token_energy, label=power_limit, color=line_colors[i], marker='o')

    # Set labels for the second y-axis
    ax2.set_ylabel('Energy Consumption (kJ) per Batch')

    # Combine the legend entries from both y-axes
    labels = ["Power Limit: {}".format(l.get_label()) for l in latency_energy_lines]

    # Create a combined legend and specify its location
    ax1.legend(latency_energy_lines, labels, loc='best')

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'Token-' + str(chosen_token_length) + '-latency-energy.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'Token-' + str(chosen_token_length) + '-latency-energy.jpeg')

    # Display the plot
    plt.show()

def plot_latency_over_token_length(latency_rsts, batch_sizes, num_tokens, power_limits, chosen_batchsize_idx):
    # Plot latency and energy results over batch sizes
    chosen_batchsize = batch_sizes[chosen_batchsize_idx]

    # Create a new figure and an axis for the first y-axis
    fig, ax1 = plt.subplots()

    # Line colors for different power limits
    line_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    latency_energy_lines = []

    # Plot the first two datasets on the first y-axis
    for i, power_limit in enumerate(power_limits):
        latency_data = latency_rsts[power_limit]
        chosen_batchsize_latency = latency_data[:, chosen_batchsize_idx]
        latency_line, = ax1.plot(num_tokens, chosen_batchsize_latency, label=power_limit, color=line_colors[i], linestyle='-')
        latency_energy_lines.append(latency_line)

    # Set labels for the first y-axis
    ax1.set_xlabel('Token lengths')
    ax1.set_ylabel('Average Query Latency (seconds)')

    # Set the title of the plot
    ax1.set_title('Batch size: {}'.format(chosen_batchsize))

    # Combine the legend entries from both y-axes
    labels = ["Power Limit: {}".format(l.get_label()) for l in latency_energy_lines]

    # Create a combined legend and specify its location
    ax1.legend(latency_energy_lines, labels, loc='best')

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'Batch-' + str(chosen_batchsize) + '-latency.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'Batch-' + str(chosen_batchsize) + '-latency.jpeg')

    # Display the plot
    plt.show()

def plot_energy_over_token_length(energy_rsts, batch_sizes, num_tokens, power_limits, chosen_batchsize_idx):
    # Plot latency and energy results over batch sizes
    chosen_batchsize = batch_sizes[chosen_batchsize_idx]

    # Create a new figure and an axis for the first y-axis
    fig, ax1 = plt.subplots()

    # Line colors for different power limits
    line_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    latency_energy_lines = []

    # Plot the first two datasets on the first y-axis
    for i, power_limit in enumerate(power_limits):
        energy_data = energy_rsts[power_limit]
        chosen_batch_energy = energy_data[:, chosen_batchsize_idx] / (num_tests * 1000000)
        energy_line, = ax1.plot(num_tokens, chosen_batch_energy, label=power_limit, color=line_colors[i], marker='o')
        latency_energy_lines.append(energy_line)

    # Set labels for the first y-axis
    ax1.set_xlabel('Token lengths')
    ax1.set_ylabel('Energy Consumption (kJ) per Batch')

    # Set the title of the plot
    ax1.set_title('Batch size: {}'.format(chosen_batchsize))

    # Combine the legend entries from both y-axes
    labels = ["Power Limit: {}".format(l.get_label()) for l in latency_energy_lines]

    # Create a combined legend and specify its location
    ax1.legend(latency_energy_lines, labels, loc='best')

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'Batch-' + str(chosen_batchsize) + '-energy.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'Batch-' + str(chosen_batchsize) + '-energy.jpeg')

    # Display the plot
    plt.show()

def main():
    power_limits = ["100W", "125W", "150W", "175W", "200W", "225W", "250W"]
    energy_rst_file_suffix = "-energy_total.csv"
    latency_rst_file_suffix = "-latency_mean.csv"

    batch_sizes = [2 ** i for i in range(5)]
    num_tokens = [20 * i for i in range(1, 10)]

    energy_rsts = {}
    latency_rsts = {}

    for power_limit in power_limits:
        energy_rst_file = exp_folder + exp_prefix + power_limit + energy_rst_file_suffix
        latency_rst_file = exp_folder + exp_prefix + power_limit + latency_rst_file_suffix

        energy_rst = load_and_check_rst_format(energy_rst_file, len(batch_sizes), len(num_tokens))
        latency_rst = load_and_check_rst_format(latency_rst_file, len(batch_sizes), len(num_tokens))

        energy_rsts[power_limit] = energy_rst
        latency_rsts[power_limit] = latency_rst

    chosen_token_idx = 8
    plot_latency_and_energy_over_batchsize(latency_rsts, energy_rsts, batch_sizes, num_tokens, power_limits,
                                           chosen_token_idx)

    chosen_batchsize_idx = 4
    plot_latency_over_token_length(latency_rsts, batch_sizes, num_tokens, power_limits, chosen_batchsize_idx)
    plot_energy_over_token_length(energy_rsts, batch_sizes, num_tokens, power_limits, chosen_batchsize_idx)

if __name__ == "__main__":
    main()




