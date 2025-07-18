import sys
import os

# This line is needed to import pynncml when running from the examples folder

script_dir = os.path.dirname(os.path.abspath(__file__))
pynncml_path = os.path.join(script_dir, '..', '..', '..', 'pynncml')
if os.path.exists(pynncml_path):
    sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', '..', '..')))

import pynncml as pnc
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the dataset for a small time slice
    time_slice = slice("2015-06-01", "2015-06-02")
    dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=time_slice)

    # Take the first link as a single example
    link_example = dataset.link_set.link_list[0]
    
    # Align data to a common 15-minute time base
    rain_rate, rsl, tsl, _ = link_example.data_alignment(dataset.link_set.max_label_size)
    
    # The loader already resamples the gauge data to 15 minutes. 
    # The CML data (RSL/TSL) is still at 10s, but grouped into 90-sample blocks per 15-min interval.
    # We'll average it for a clearer plot.
    rsl_mean = np.mean(rsl, axis=1)
    tsl_mean = np.mean(tsl, axis=1)
    
    # Generate the time axes for plotting
    num_points = rain_rate.shape[0]
    
    # Get the correctly formatted time axis from the Link object
    full_time_axis = link_example.time()
    # The data alignment might shorten the series, so we'll use the generated rain_rate length
    time_axis = full_time_axis[:num_points]


    # Plot the signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Data Inspection: Single CML Example (Link 0)', fontsize=16)

    # Plot RSL and TSL
    ax1.plot(time_axis, rsl_mean, label='RSL (Received Signal Level)', color='blue')
    ax1.plot(time_axis, tsl_mean, label='TSL (Transmitted Signal Level)', color='red', linestyle='--')
    ax1.set_ylabel('Signal Level (dBm)')
    ax1.set_title('CML Signal Levels')
    ax1.legend()
    ax1.grid(True)

    # Plot Rain Rate
    ax2.plot(time_axis, rain_rate[:, 0], label='Rain Rate (Gauge Reference)', color='green')
    ax2.set_xlabel('Datetime')
    ax2.set_ylabel('Rain Rate (mm/hr)')
    ax2.set_title('Corresponding Rain Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_dir = 'examples/physics_aware_network/data_inspection/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'data_inspection_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'data_inspection_plot.png')}")
    
    plt.show()

if __name__ == '__main__':
    main() 