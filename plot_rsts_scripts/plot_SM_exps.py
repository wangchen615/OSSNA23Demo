import matplotlib.pyplot as plt
import numpy as np

exp_folder = "../rsts/"
img_folder = "../plots/"
exp_prefix = "SM2-"
num_tests = 20

def load_and_check_rst_format(rst_file, batch_idx_length, token_idx_length):
    rsts = np.loadtxt(rst_file, delimiter=',')
    try:
        row_len, col_len = rsts.shape
    except:
        row_len, col_len = rsts.size, rsts.size
    if row_len != token_idx_length:
        print("Data file {} does not have the correct row indices!".format(rst_file))
        return None

    if col_len != batch_idx_length:
        print("Data file {} does not have the correct row indices!".format(rst_file))
        return  None

    return rsts

def plot_energy_over_freq(GPU_Freqs, energy_rsts, batch_size, token_len, num_tests):
    # Create a plot of the x and y values
    energy_rsts_in_joules = [energy / (1000*num_tests) for energy in energy_rsts]
    plt.plot(GPU_Freqs, energy_rsts_in_joules)

    # Add a title and axis labels
    plt.title("Batch size: {}; Token Length: {};".format(batch_size, token_len))
    plt.xlabel("GPU Frequencies")
    plt.ylabel("Energy Consumption (J) per Batches")

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'energy.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'energy.jpeg')

    # Display the plot
    plt.show()

def plot_latency_over_freq(GPU_Freqs, latency_rsts, batch_size, token_len):
    # Create a plot of the x and y values
    plt.plot(GPU_Freqs, latency_rsts)

    # Add a title and axis labels
    plt.title("Batch size: {}; Token Length: {}".format(batch_size, token_len))
    plt.xlabel("GPU Frequencies")
    plt.ylabel("Per Query Latency (Seconds)")

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'latency.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'latency.jpeg')

    # Display the plot
    plt.show()

def plot_latency_over_energy(energy_rsts, latency_rsts, GPU_Freqs, batch_size, token_len, num_tests):
    energy_rsts_in_joules = [energy / (1000 * num_tests) for energy in energy_rsts]
    plt.scatter(energy_rsts_in_joules, latency_rsts)

    # Add labels to each dot
    for i, label in enumerate(GPU_Freqs):
        plt.annotate(label, (energy_rsts_in_joules[i], latency_rsts[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    # Add a title and axis labels
    plt.title("Batch size: {}; Token Length: {}".format(batch_size, token_len))
    plt.xlabel("Energy Consumption per Batch (Joules)")
    plt.ylabel("Per Query Latency (Seconds)")

    # Save the plot as a PDF file
    plt.savefig(img_folder + exp_prefix + 'energy-latency.pdf')

    # Save the plot as a JPEG file
    plt.savefig(img_folder + exp_prefix + 'energy-latency.jpeg')

    # Display the plot
    plt.show()

def main():
    GPU_Freqs = [150, 270, 390, 510, 630, 750, 870, 990, 1110, 1230, 1380]
    energy_rst_file_suffix = "-energy_total.csv"
    latency_rst_file_suffix = "-latency_mean.csv"

    batch_sizes = [32]
    num_tokens = [200]

    chosen_batchsize_idx = 0
    chosen_tokenlen_idx = 0

    energy_rsts = []
    latency_rsts = []

    for gpu_freq in GPU_Freqs:
        energy_rst_file = exp_folder + exp_prefix + str(gpu_freq) + energy_rst_file_suffix
        latency_rst_file = exp_folder + exp_prefix + str(gpu_freq) + latency_rst_file_suffix

        energy_rst = load_and_check_rst_format(energy_rst_file, len(batch_sizes), len(num_tokens))
        latency_rst = load_and_check_rst_format(latency_rst_file, len(batch_sizes), len(num_tokens))

        if (len(batch_sizes) == len(num_tokens)) and (len(batch_sizes) == 1):
            energy_rsts.append(energy_rst.item())
            latency_rsts.append(latency_rst.item())
        else:
            energy_rsts.append(energy_rst[chosen_tokenlen_idx][chosen_batchsize_idx])
            latency_rsts.append(latency_rst[chosen_tokenlen_idx][chosen_batchsize_idx])

    plot_latency_over_freq(GPU_Freqs, latency_rsts, batch_sizes[chosen_batchsize_idx], num_tokens[chosen_tokenlen_idx])
    plot_energy_over_freq(GPU_Freqs, energy_rsts, batch_sizes[chosen_batchsize_idx], num_tokens[chosen_tokenlen_idx], num_tests)
    plot_latency_over_energy(energy_rsts, latency_rsts, GPU_Freqs, batch_sizes[chosen_batchsize_idx], num_tokens[chosen_tokenlen_idx], num_tests)

if __name__ == "__main__":
    main()




