import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
from linecache import getline
from dqmc_analysis_tools import Get_den_orb

plot_sign = 0


def extract_v_or_et(filename, key):
    """
    Extract the numerical value following 'V' or 'et' in the given filename.
    
    Parameters:
        filename (str): The input filename to parse.
        key (str): The key to search for, either 'V' or 'et'.
    
    Returns:
        float: The numerical value following the specified key.
        None: If the key is not found in the filename.
    """
    try:
        # Match pattern like 'V1.23' or 'et-4.56' and extract the numerical part
        pattern = rf'{key}(-?\d+\.\d+)'  # Regex to match key followed by a number
        match = re.search(pattern, filename)
        if match:
            return float(match.group(1))  # Convert the matched string to a float
        else:
            print(f"Key '{key}' not found in filename: {filename}")
            return None
    except Exception as e:
        print(f"Error processing filename {filename}: {e}")
        return None


def parse_filename(filename):
    """
    Extract relevant parameters for use in plot title and filename.
    Title includes all parameters, filename excludes 'mu' and 's1234567'.
    """
    filename = filename.replace('.out', '')  # Remove file extension
    
    # Use regular expressions to find parameters
    params = re.findall(r'(V\d+\.\d+|Uh\d+\.\d+|Ut\d+\.\d+|tH\d+\.\d+|tT\d+\.\d+|eh-?\d+\.\d+|et-?\d+\.\d+|N\d+|be\d+\.\d+)', filename)
    
    # Filename for saving excludes 'mu' and 's1234567'
    params_for_filename = [param for param in params if not param.startswith('s') and not param.startswith('mu') and not param.startswith('V') and not param.startswith('et')]
    filename_str = "_".join(params_for_filename)  # Name for the saved file

    # Title for the plot includes all parameters
    title_str = ", ".join(params_for_filename)

    return title_str, filename_str

def extract_avg_sign(file):
    """
    Extract the average sign from the file content based on the line 
    containing 'Avg sign :' followed by the average sign value.
    """
    avg_sign = None
    with open(file, 'r') as f:
        for line in f:
            if 'Avg sign :' in line:
                try:
                    # Extract the number following 'Avg sign :'
                    avg_sign = float(line.split('Avg sign :')[1].strip().split()[0])
                except (IndexError, ValueError):
                    print("Error parsing Avg sign in file: {}".format(file))
                break
    return avg_sign

def analyze_dqmc_data(file_list, norb, Nline):
    """
    Process a list of DQMC output files to extract chemical potential values (mu), 
    density values for each orbital, and average sign values.
    """
    mu_values = []
    densities = np.zeros((len(file_list), norb))  # Initialize array to store density data
    avg_signs = []  # Initialize list to store average sign data
    v_values = []  # To store V values extracted from the filenames
    et_values = []  # To store et values extracted from the filenames
    print("Processing files:")    
    for file in file_list:
        print(file)    
    for i, file in enumerate(file_list):
        # Extract the mu value from the filename
        try:
            mu_str = re.search(r'mu(-?\d+(\.\d+)?)', file).group(1)
            mu = float(mu_str)
            mu_values.append(mu)
        except (AttributeError, ValueError):
            print("Filename format is incorrect, skipping file: {}".format(file))
            continue

        # Extract V value from the filename
        try:
            v_str = re.search(r'V\d+\.\d+', file).group(0)
            v_values.append(v_str)
        except (AttributeError, ValueError):
            print("No V value found in filename, skipping file: {}".format(file))
            continue
        
        # Extract et value from the filename
        try:
            et_str = re.search(r'et-?\d+\.\d+', file).group(0)
            et_values.append(et_str)
        except (AttributeError, ValueError):
            print("No et value found in filename, skipping file: {}".format(file))
            continue

        # Use Get_den_orb function to get density
        dens = Get_den_orb(file, norb, Nline)
        
        # Store the densities for each orbital in the appropriate row
        for j in range(norb):
            densities[i, j] = dens[j, 0]  # Take the first column (density)
        
        # Extract the average sign from the file
        avg_sign = extract_avg_sign(file)
        if avg_sign is not None:
            avg_signs.append(avg_sign)
    
    # Sort by mu values
    sorted_indices = np.argsort(mu_values)
    mu_values = np.array(mu_values)[sorted_indices]
    densities = densities[sorted_indices]
    avg_signs = np.array(avg_signs)[sorted_indices]  # Sort avg_signs according to mu
    v_values = np.array(v_values)[sorted_indices]  # Sort V values according to mu
    et_values = np.array(et_values)[sorted_indices]  # Sort et values according to mu
    
    print("mu values, densities, and avg signs:")
    for mu, density, sign, v, et in zip(mu_values, densities, avg_signs, v_values, et_values):
        print(f"mu = {mu}, Density = {density}, Avg sign = {sign}, V = {v}, et = {et}")

    return mu_values, densities, avg_signs, v_values, et_values

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
            print("Invalid file or file does not exist: {}".format(file))
            sys.exit(1)

    norb = 14  # Number of orbitals. Cu 0 Ox 1 Oy 2
    Nline = 419  # Number of lines to read. 419 for Ncell=4(2*2)

    # Extract title and filename strings from the first file in the list
    title_str, filename_str = parse_filename(file_list[0])

    # Initialize lists to hold the mu values, densities, and avg_signs
    mu_values, densities, avg_signs, v_values, et_values = analyze_dqmc_data(file_list, norb, Nline)
    
    # Calculate total density (sum of densities across orbitals)
    total_density = densities.sum(axis=1)
    total_avg_density = total_density/norb

    # Create a directory to store results
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Ask the user whether to group by 'V' or 'et'
    group_by = input("Group by (V/et)? ").strip().lower()

    if group_by == 'v':
        unique_values = np.unique(v_values)
        xlabel = 'V'
        ylabel = 'Avg Density'
        et_value = extract_v_or_et(file_list[0], "et")
        title_str = 'et'+str(et_value)+','+title_str
        filename_str = 'et'+str(et_value)+'_'+filename_str
        title_1T = f'Avg Density vs Chemical Potential μ (1T orbitals)\n ({title_str})'
        title_1H = f'Avg Density vs Chemical Potential μ (1H orbitals)\n ({title_str})'
    elif group_by == 'et':
        unique_values = np.unique(et_values)
        xlabel = 'et'
        ylabel = 'Avg Density'
        V_value = extract_v_or_et(file_list[0], "V")
        title_str = 'V'+str(V_value)+','+title_str
        filename_str = 'V'+str(V_value)+'_'+filename_str
        title_1T = f'Avg Density vs Chemical Potential μ (1T orbitals)\n ({title_str})'
        title_1H = f'Avg Density vs Chemical Potential μ (1H orbitals)\n ({title_str})'
    else:
        print("Invalid input, please choose 'V' or 'et'.")
        sys.exit(1)

    # Plot for 1T
    plt.figure(figsize=(8, 6))
    for value in unique_values:
        # Get indices for the current parameter value (V or et)
        if group_by == 'v':
            indices = np.where(v_values == value)[0]
        else:
            indices = np.where(et_values == value)[0]

        # Filter data for the current parameter value
        mu_value = mu_values[indices]
        densities_value = densities[indices]

        # Plot 1T density (Orb 1T_avg)
        plt.plot(mu_value, densities_value[:, 13], label=f'{value} 1T_avg', marker='o')

    plt.xlabel(r'$\mu$')
    plt.ylabel(ylabel)
    plt.title(title_1T)
    plt.legend()
    plt.grid(True)

    # Save plot for 1T
    output_filename_1T = f"./results/Avg_Density_vs_mu_1T_{filename_str}.pdf"
    plt.savefig(output_filename_1T, format='pdf')
    plt.close()

    # Plot for 1H
    plt.figure(figsize=(8, 6))
    for value in unique_values:
        # Get indices for the current parameter value (V or et)
        if group_by == 'v':
            indices = np.where(v_values == value)[0]
        else:
            indices = np.where(et_values == value)[0]

        # Filter data for the current parameter value
        mu_value = mu_values[indices]
        densities_value = densities[indices]
        total_density_value = total_density[indices]

        # Plot 1H density (Orb 1H_avg)
        plt.plot(mu_value, (total_density_value - densities_value[:, 13]) / (norb - 1), label=f'{value} 1H_avg', marker='o')

    plt.xlabel(r'$\mu$')
    plt.ylabel(ylabel)
    plt.title(title_1H)
    plt.legend()
    plt.grid(True)

    # Save plot for 1H
    output_filename_1H = f"./results/Avg_Density_vs_mu_1H_{filename_str}.pdf"
    plt.savefig(output_filename_1H, format='pdf')
    plt.close()


