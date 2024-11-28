import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from dqmc_analysis_tools import Get_den_orb

def parse_filename_mu_et_V(filename):
    """
    Extract the `mu`, `et`, and `V` parameters from the filename.
    """
    filename = filename.replace('.out', '')

    # Extract mu, et, and V values
    try:
        mu = float(re.search(r'mu(-?\d+\.\d+)', filename).group(1))
        et = float(re.search(r'et(-?\d+\.\d+)', filename).group(1))
        V = float(re.search(r'V(\d+\.\d+)', filename).group(1))
    except AttributeError:
        raise ValueError(f"Filename does not contain valid mu, et, or V parameters: {filename}")

    return mu, et, V

def extract_avg_sign(file):
    """
    Extract the average sign from the file content.
    """
    avg_sign = None
    with open(file, 'r') as f:
        for line in f:
            if 'Avg sign :' in line:
                try:
                    # Extract the number following 'Avg sign :'
                    avg_sign = float(line.split('Avg sign :')[1].strip().split()[0])
                except (IndexError, ValueError):
                    print(f"Error parsing Avg sign in file: {file}")
                break
    return avg_sign

def analyze_mu_et_V_avg_sign(file_list):
    """
    Analyze files to extract mu, et, V, and average sign values.
    """
    mu_values = []
    et_values = []
    V_values = []
    avg_signs = []

    for file in file_list:
        try:
            # Extract mu, et, and V from filename
            mu, et, V = parse_filename_mu_et_V(file)
            mu_values.append(mu)
            et_values.append(et)
            V_values.append(V)
        except ValueError as e:
            print(e)
            continue

        # Extract the average sign from the file
        avg_sign = extract_avg_sign(file)
        if avg_sign is not None:
            avg_signs.append(avg_sign)

    # Convert to numpy arrays for easier handling
    mu_values = np.array(mu_values)
    et_values = np.array(et_values)
    V_values = np.array(V_values)
    avg_signs = np.array(avg_signs)

    print("Extracted data:")
    for mu, et, V, sign in zip(mu_values, et_values, V_values, avg_signs):
        print(f"mu = {mu}, et = {et}, V = {V}, Avg sign = {sign}")

    return mu_values, et_values, V_values, avg_signs

def plot_colormap(mu_values, et_values, V_values, avg_signs, axis='et'):
    """
    Create and save a colormap plot with mu on the x-axis, et or V on the y-axis,
    and average sign as the color.
    
    `axis`: 'et' or 'V' to choose which to use for the y-axis.
    """
    if axis == 'et':
        y_values = et_values
        xlabel = r'$\mu$'
        ylabel = r'$et$'
    elif axis == 'V':
        y_values = V_values
        xlabel = r'$\mu$'
        ylabel = r'$V$'
    else:
        raise ValueError("axis must be 'et' or 'V'.")

    # Create a grid for interpolation
    grid_mu, grid_y = np.meshgrid(
        np.linspace(mu_values.min(), mu_values.max(), 100),
        np.linspace(y_values.min(), y_values.max(), 100)
    )
    grid_sign = griddata(
        (mu_values, y_values), avg_signs, (grid_mu, grid_y), method='linear'
    )

    # Plot colormap
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    contour = plt.contourf(grid_mu, grid_y, grid_sign, levels=100, cmap=cmap)
    plt.colorbar(contour, label='Average Sign')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Average Sign Colormap ({ylabel} vs $\mu$)')
    plt.grid(True)

    # Save the plot
    if not os.path.exists('./results'):
        os.makedirs('./results')

    output_filename = './results/Avg_Sign_Colormap.pdf'
    plt.savefig(output_filename, format='pdf')
    plt.close()
    print(f"Colormap saved to {output_filename}")

# Main function
if __name__ == "__main__":
    # Get file names from command-line arguments
    if len(sys.argv) < 2:
        print("Please provide at least one filename as an argument.")
        sys.exit(1)

    file_list = sys.argv[1:]  # Get the list of files from command-line arguments
    file_list = [file for file in file_list if not file.endswith(".tdm.out")]

    # Check if files exist
    for file in file_list:
        if not file.endswith(".out") or not os.path.isfile(file):
            print(f"Invalid file or file does not exist: {file}")
            sys.exit(1)

    # Analyze files to extract mu, et, V, and average sign
    mu_values, et_values, V_values, avg_signs = analyze_mu_et_V_avg_sign(file_list)

    # Plot the colormap using 'et' as the y-axis
    plot_colormap(mu_values, et_values, V_values, avg_signs, axis='et')

    # Optionally, if you want to plot using 'V' as the y-axis, uncomment the following:
    # plot_colormap(mu_values, et_values, V_values, avg_signs, axis='V')

