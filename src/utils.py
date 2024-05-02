import numpy as np
import numpy.linalg as LA
from mdhandler import Handler as MDH
import structures
import collections.abc
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

def StaticOverEvolve(temp):
    Nat=10
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.Chain(Nat,R0)
    Chain2.SaveGeometry()
    Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,75])

    Chain1.add(Chain2)

    md = MDH(Chain1,False)
    md.RunMD(5000,temp,[0,9,10,19])
    ForcesMBD = md.GetForcesSOE("MBD")
    ForcesPW = md.GetForcesSOE("PW")
    ForcesShort = md.GetForcesSOE()


    with open('out/SOE_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(ForcesShort)):
            F_lower_MBD = 0
            F_lower_PW = 0
            F_lower_Short = 0
            for k in range(10):
                F_lower_MBD = F_lower_MBD + ForcesMBD[i][k][2]
                F_lower_PW = F_lower_PW + ForcesPW[i][k][2]
                F_lower_Short = F_lower_Short + ForcesShort[i][k][2]
            F_upper_MBD = 0
            F_upper_PW = 0
            F_upper_Short = 0
            for k in range(10,20):
                F_upper_MBD = F_upper_MBD + ForcesMBD[i][k][2]
                F_upper_PW = F_upper_PW + ForcesPW[i][k][2]
                F_upper_Short = F_upper_Short + ForcesShort[i][k][2]
            f.write(str(i)+ " " +str(F_upper_MBD-F_upper_Short)+ " "+ str(F_lower_MBD-F_lower_Short)+ " " +str(F_upper_PW-F_upper_Short)+ " "+ str(F_lower_PW-F_lower_Short)+ " " +str(F_upper_Short)+ " "+ str(F_lower_Short)+"\n")

def CorrelationOverEvolve(temp):
    Nat=10
    R0=2.4
    Chain1 = structures.Chain(Nat,R0)
    Chain1.SaveGeometry()
    Chain1.RunOptimize()
    Chain1.LoadGeometry()

    Chain2 = structures.Chain(Nat,R0)
    Chain2.SaveGeometry()
    Chain2.RunOptimize()
    Chain2.LoadGeometry()

    Chain2.MoveAll([0,0,20])
    Chain1.add(Chain2)

    Navg=100
    corrxx = np.zeros(10)
    corryy = np.zeros(10)
    corrzz = np.zeros(10)

    corrxy = np.zeros(10)
    corrxz = np.zeros(10)
    corryz = np.zeros(10)

    corrxx_abs = np.zeros(10)
    corryy_abs = np.zeros(10)
    corrzz_abs = np.zeros(10)

    corrxy_abs = np.zeros(10)
    corrxz_abs = np.zeros(10)
    corryz_abs = np.zeros(10)

    for k in range(Navg):
        print(str(k)+"/"+str(Navg))
        md = MDH(Chain1,False)
        md.RunMD(5000,temp,[0,9,10,19])
        md.LoadEvolution()
        evolution = md.evolution
        for j in range(10):
            v_lower_x = []
            v_lower_y = []
            v_lower_z = []
            v_lower_x_abs = []
            v_lower_y_abs = []
            v_lower_z_abs = []
            for i in range(len(evolution)):
                v_lower_x.append(evolution[i][j][3])
                v_lower_y.append(evolution[i][j][4])
                v_lower_z.append(evolution[i][j][5])
                v_lower_x_abs.append(np.abs(evolution[i][j][3]))
                v_lower_y_abs.append(np.abs(evolution[i][j][4]))
                v_lower_z_abs.append(np.abs(evolution[i][j][5]))
            v_lower_x=np.array(v_lower_x)
            v_lower_y=np.array(v_lower_y)
            v_lower_z=np.array(v_lower_z)
            v_lower_x_abs=np.array(v_lower_x_abs)
            v_lower_y_abs=np.array(v_lower_y_abs)
            v_lower_z_abs=np.array(v_lower_z_abs)

            v_upper_x = []
            v_upper_y = []
            v_upper_z = []
            v_upper_x_abs = []
            v_upper_y_abs = []
            v_upper_z_abs = []
            for i in range(len(evolution)):
                v_upper_x.append(evolution[i][j+10][3])
                v_upper_y.append(evolution[i][j+10][4])
                v_upper_z.append(evolution[i][j+10][5])
                v_upper_x_abs.append(np.abs(evolution[i][j+10][3]))
                v_upper_y_abs.append(np.abs(evolution[i][j+10][4]))
                v_upper_z_abs.append(np.abs(evolution[i][j+10][5]))
            v_upper_x=np.array(v_upper_x)
            v_upper_y=np.array(v_upper_y)
            v_upper_z=np.array(v_upper_z)
            v_upper_x_abs=np.array(v_upper_x_abs)
            v_upper_y_abs=np.array(v_upper_y_abs)
            v_upper_z_abs=np.array(v_upper_z_abs)

            v_lower = []
            v_lower.append(v_lower_x)
            v_lower.append(v_lower_y)
            v_lower.append(v_lower_z)
            v_lower.append(v_lower_x_abs)
            v_lower.append(v_lower_y_abs)
            v_lower.append(v_lower_z_abs)
            v_upper = []
            v_upper.append(v_upper_x)
            v_upper.append(v_upper_y)
            v_upper.append(v_upper_z)
            v_upper.append(v_upper_x_abs)
            v_upper.append(v_upper_y_abs)
            v_upper.append(v_upper_z_abs)

            corrxx[j]=corrxx[j]+np.corrcoef(v_lower,v_upper)[0,6]/Navg
            corryy[j]=corryy[j]+np.corrcoef(v_lower,v_upper)[1,7]/Navg
            corrzz[j]=corrzz[j]+np.corrcoef(v_lower,v_upper)[2,8]/Navg

            corrxy[j]=corrxy[j]+np.corrcoef(v_lower,v_upper)[0,7]/Navg
            corrxz[j]=corrxz[j]+np.corrcoef(v_lower,v_upper)[0,8]/Navg
            corryz[j]=corryz[j]+np.corrcoef(v_lower,v_upper)[1,8]/Navg

            corrxx_abs[j]=corrxx_abs[j]+np.corrcoef(v_lower,v_upper)[3,9]/Navg
            corryy_abs[j]=corryy_abs[j]+np.corrcoef(v_lower,v_upper)[4,10]/Navg
            corrzz_abs[j]=corrzz_abs[j]+np.corrcoef(v_lower,v_upper)[5,11]/Navg

            corrxy_abs[j]=corrxy_abs[j]+np.corrcoef(v_lower,v_upper)[3,10]/Navg
            corrxz_abs[j]=corrxz_abs[j]+np.corrcoef(v_lower,v_upper)[3,11]/Navg
            corryz_abs[j]=corryz_abs[j]+np.corrcoef(v_lower,v_upper)[4,11]/Navg

    with open('out/corr_'+str(temp)+'.txt', 'w') as f:
        for i in range(len(corrxx)):
            f.write(str(i)+ " " +str(corrxx[i])+" " +str(corryy[i])+" " +str(corrzz[i])+" " +str(corrxy[i])+" " +str(corrxz[i])+" " +str(corryz[i])+" " +str(corrxx_abs[i])+" " +str(corryy_abs[i])+" " +str(corrzz_abs[i])+" " +str(corrxy_abs[i])+" " +str(corrxz_abs[i])+" " +str(corryz_abs[i])+"\n")


def _write_file(name,content):
    if isinstance(content[0], collections.abc.Sized) == True:
        with open('out/'+name, 'w') as f:
            for i in range(len(content)):
                for j in range(len(content[i])):
                    f.write(str(content[i][j])+' ')
                f.write('\n')
    else:
        with open('out/'+name, 'w') as f:
            for i in range(len(content)):
                f.write(str(i)+' '+str(content[i])+' '+'\n')

def cart2sph(coord_cart):
    x = coord_cart[0]
    y = coord_cart[1]
    z = coord_cart[2]
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))+np.pi/2     # theta
    az = m.atan2(y,x)                           # phi
    return [r, elev, az]

def cart2sph_list(coord_cart):
    coord_spher = []
    for i in range(len(coord_cart)):
        aux = cart2sph(coord_cart[i])
        coord_spher.append(aux)
    return coord_spher

def _read_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by spaces and convert each element to a float
            row_data = [float(num) for num in line.split()]
            data.append(row_data)
    return data

def concatenate_data(data1, data2):
    # Create a mapping from index (as int) to data for easy lookup
    data1_dict = {int(row[0]): row[1:] for row in data1}
    data2_dict = {int(row[0]): row[1:] for row in data2}

    # Get the union of indices from both datasets
    all_indices = sorted(set(data1_dict.keys()) & set(data2_dict.keys()))

    concatenated_data = []
    for index in all_indices:
        # Start with the index as an integer
        row_data = [index]
        # Extend with data from the first file if it exists
        row_data.extend(data1_dict.get(index, []))
        # Extend with data from the second file if it exists
        row_data.extend(data2_dict.get(index, []))
        concatenated_data.append(row_data)

    return concatenated_data

def plot_histogram(data1, data2=None, data3=None, bins=10, data_range=None, label1=None, label2=None, label3=None, xlabel='Value'):
    """
    Plots a histogram of the data with the specified number of bins, range, and labels.

    Parameters:
    - data1: array-like, the first input data for the histogram
    - data2: array-like, the optional second input data for the histogram
    - bins: int, the number of bins for the histogram
    - data_range: tuple, the (min, max) range of the histogram
    - label1: str, the label for the first histogram
    - label2: str, the label for the second histogram
    - xlabel: str, the label for the x-axis
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate the increased font size (20% larger)
    default_font_size = plt.rcParams['font.size']
    larger_font_size = default_font_size * 2
    
    plt.hist(data1, bins=bins, range=data_range, density=True, label=label1, alpha=0.75, color='red', edgecolor='black', histtype='stepfilled')
    
    if data2 is not None:
        plt.hist(data2, bins=bins, range=data_range, density=True, label=label2, alpha=0.75, color='green', edgecolor='black', histtype='stepfilled')
    if data3 is not None:
        plt.hist(data3, bins=bins, range=data_range, density=True, label=label3, alpha=0.75, color='blue', edgecolor='black', histtype='stepfilled')


    plt.xlabel(xlabel, fontsize=larger_font_size)
    plt.ylabel('Density', fontsize=larger_font_size)
    plt.title('Histogram Comparison', fontsize=larger_font_size)
    
    if label1 or label2 or label3:
        plt.legend(fontsize=larger_font_size)
    
    # Increase tick label size
    plt.xticks(fontsize=larger_font_size)
    plt.yticks(fontsize=larger_font_size)
    
    plt.show()

def plot_heatmap(x_data, y_data, x_bins=10, y_bins=10, x_range=None, y_range=None, x_label=None, y_label=None, title=None):
    """
    Plots a heatmap from two data arrays.

    Parameters:
    - x_data: array-like, the input data for the x-axis
    - y_data: array-like, the input data for the y-axis
    - x_bins: int, the number of bins for the x-axis
    - y_bins: int, the number of bins for the y-axis
    - x_range: tuple, the (min, max) range of the x-axis
    - y_range: tuple, the (min, max) range of the y-axis
    - x_label: str, the label for the x-axis
    - y_label: str, the label for the y-axis
    - title: str, the title of the heatmap
    """
    cdict = {'red':   ((0.0, 0.2, 0.2),  # Grey (midpoint of 0.5 for R, G, and B)
                       (1.0, 1.0, 1.0)),  # Red
             'green': ((0.0, 0.2, 0.2),
                       (1.0, 0.0, 0.0)),  # No green at the end
             'blue':  ((0.0, 0.2, 0.2),
                       (1.0, 0.0, 0.0))}  # No blue at the end
    black_red_cmap = LinearSegmentedColormap('BlackRed', cdict)

    default_font_size = plt.rcParams['font.size']
    larger_font_size = default_font_size * 1.8

    plt.figure(figsize=(10, 8))
    heatmap, xedges, yedges, img = plt.hist2d(x_data, y_data, bins=[x_bins, y_bins], range=[x_range, y_range], cmap='hot', density=True)
    # Create the colorbar
    cbar = plt.colorbar(img, label='Density')
    # Set the font size for the colorbar labels
    cbar.ax.tick_params(labelsize=larger_font_size)  # Update the tick label font size
    
    # Set the font size for the colorbar title
    cbar.set_label('Density', fontsize=larger_font_size)
    plt.xlabel(x_label, fontsize=larger_font_size*1.2)
    plt.ylabel(y_label, fontsize=larger_font_size*1.2)
    plt.title(title, fontsize=larger_font_size)
    plt.tick_params(axis='both', which='major', labelsize=larger_font_size)
    plt.show()

def filter_samples_idx(data, idx, threshold,case="smaller"):
    # Check if the idx dimension of each sample is bigger than the threshold
    data = np.array(data)
    if case == "smaller":
        mask = data[:, idx] > threshold
        print("filtering data samples with idx="+str(idx)+" smaller than "+str(threshold))
    elif case == "bigger":
        mask = data[:, idx] < threshold
    else:
        print("case not implemented..")
        return None
    
    # Use the mask to filter out the samples that do not meet the condition
    filtered_data = data[mask]
    initial_samples_count = len(data)
    remaining_samples_count = len(filtered_data)
    filtered_samples_count = initial_samples_count - remaining_samples_count
    filtered_percentage = (filtered_samples_count / initial_samples_count) * 100
    print(f"Initial number of samples: {initial_samples_count}")
    print(f"Number of samples after filtering: {remaining_samples_count}")
    print(f"Number of samples filtered out: {filtered_samples_count}")
    print(f"Percentage of samples filtered out: {filtered_percentage:.2f}%")
    return filtered_data.tolist()

def filter_samples_complex(data):
    data = np.array(data)
    # Calculate the sum of the first and second dimensions for each sample
    sum_of_first_two_dims = -data[:, 6] * data[:, 5]/data[:, 4]
    
    # Check if the sum of the first and second dimensions is less than r0
    mask = sum_of_first_two_dims <= 24
    
    # Use the mask to filter out the samples that do not meet the condition
    filtered_data = data[mask]
    filtered_data = data[mask]
    initial_samples_count = len(data)
    remaining_samples_count = len(filtered_data)
    filtered_samples_count = initial_samples_count - remaining_samples_count
    filtered_percentage = (filtered_samples_count / initial_samples_count) * 100
    print(f"Initial number of samples: {initial_samples_count}")
    print(f"Number of samples after filtering: {remaining_samples_count}")
    print(f"Number of samples filtered out: {filtered_samples_count}")
    print(f"Percentage of samples filtered out: {filtered_percentage:.2f}%")
    
    return filtered_data.tolist()


def filter_equal_arrays_condfirst(a, b, c, threshold=0.1):
    # Create a boolean mask where True corresponds to elements of 'a' <= threshold
    mask = a > threshold
        
    # Apply the mask to both 'a' and 'b' to filter out the unwanted elements
    a_filtered = a[mask]
    b_filtered = b[mask]
    c_filtered = c[mask]
        
    return a_filtered, b_filtered, c_filtered

def fit_gaussian(data, initial_mean_guess,initial_std_guess):
    # Define the Gaussian function
    def gaussian(x, mean, amplitude, standard_deviation):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation ** 2))
    
    # Generate a histogram of the data to get bin centers and frequencies
    histogram, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial guesses for the parameters: mean, amplitude, standard deviation
    initial_guess = [initial_mean_guess, np.max(histogram), initial_std_guess]
    
    # Create weights for the histogram data
    # This creates a Gaussian-shaped weighting scheme centered around the initial mean guess
    weights = np.exp(-((bin_centers - initial_mean_guess) ** 2) / (2 * initial_std_guess ** 2))
    
    # Use curve_fit to fit the Gaussian function to the histogram data with weights
    popt, _ = curve_fit(gaussian, bin_centers, histogram, p0=initial_guess, sigma=1/weights)
    
    # The fitted parameters are in popt: mean, amplitude, and standard deviation
    fitted_mean, _, fitted_std = popt
    
    return fitted_mean, fitted_std