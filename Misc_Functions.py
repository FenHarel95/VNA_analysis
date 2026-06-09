import csv

import numpy as np
#from additional_tools.Susceptibility_Methods import *
import matplotlib.pyplot as plt
import os
from Susceptibility_Methods import *

def sort_re_im(values):
    return {
        "Re": values[::2],
        "Im": values[1::2],
    }

def avg_array(array, run=True):
    """Function calculates either the running or normal average for an array"""
    if run:
        cumsum = np.cumsum(array)
        running_avg = cumsum / np.arange(1, len(array) + 1)
        return running_avg[len(array)-1]
    else:
        return np.mean(array)

def create_complex_array(data):
    sorted_data = sort_re_im(np.array(data))
    re, im = sorted_data["Re"], sorted_data["Im"]
    if len(re) == len(im):
        return np.array(re) + 1j * np.array(im)
    else:
        return 0

def obtain_s_matrix(vna_channel, dictionary):
    # Initialize dictionaries to store results
    s_temp = {}
    # Store data from traces into dictionary
    for trace in dictionary.keys():
        data = getattr(vna_channel, trace).data_raw
        s_temp[dictionary[trace]] = create_complex_array(data)
    return s_temp

def ite_reflexion_p(s11, s22, s12, s21, deem=False, deem_phase=1):
    """Calculates the reflexion and p propagation constant of the shape=(j=fieldPoints, i=freqPoints) from of arrays
    sij of same shape"""
    n_fields = s11.shape[0]
    n_freqs  = s11.shape[1]

    reflexion = np.zeros((n_fields, n_freqs), dtype=np.complex128)
    p         = np.zeros((n_fields, n_freqs), dtype=np.complex128)

    for j in range(n_fields):
        for i in range(n_freqs):
            reflexion[j, i], p[j, i] = reflexion_p(
                s11[j, i], s22[j, i], s12[j, i], s21[j, i],
                deem, deem_phase
            )

    return reflexion, p

def ite_permitt_permeab(sample_l, ref, p, freq, epsilon):
    """Returns the effective permitivity (permitt), permeability (permea_1, permea_2) of the
    shape=(j=fieldPoints, i=freqPoints), from arrays ref, p of the same shape."""
    n_fields, n_freqs = ref.shape

    permitt = np.zeros((n_fields, n_freqs), dtype=np.complex128)
    permea_1 = np.zeros((n_fields, n_freqs), dtype=np.complex128)
    permea_2 = np.zeros((n_fields, n_freqs), dtype=np.complex128)

    for j in range(n_fields):
        for i in range(n_freqs):
            calc = permitt_permeab(
                sample_l,
                ref[j, i],
                p[j, i],
                freq[i],
                epsilon
            )

            permitt[j, i] = calc["permitt"]
            permea_1[j, i] = calc["first_eval_permeab"]
            permea_2[j, i] = calc["second_eval_permeab"]

    return {
        "permittivity": permitt,
        "first_eval_permeab": permea_1,
        "second_eval_permeab": permea_2
    }

def convert_dbm(value: str) -> str:
    if value.startswith('-'):
        value = 'm' + value[1:]  # Replace '-' with 'm'
    return value.replace('.', 'p')  # Replace '.' with 'p'

def delta_mij(single, mij, mij_ref):
    """Substract background from a spectra mij (np.array, eg. S21) depending on single choise
    If single= True, it substracts Mij_ref
    otherwise substracts background using SVD background removal
    """
    if single:
        mij_ref_I = mij_ref.imag
        mij_ref_R = mij_ref.real
    else:
        # Background components method
        U, S, Vt = np.linalg.svd(mij, full_matrices=False)
        mij_bg = np.outer(U[:, 0] * S[0], Vt[0, :])  # background = first component
        mij = mij - mij_bg
        # Median Background method(gives points with spikes)
        # Z_fieldmedians = medfilt2d(Z, kernel_size=(3, 7))
        # Z_fieldmedians = np.median(Z, axis=0)
        # Z = Z-Z_fieldmedians
        mij_ref_I = np.zeros(len(mij[0]), dtype=complex)
        mij_ref_R = np.zeros(len(mij[0]), dtype=complex)

    mij_R = mij.real - mij_ref_R
    mij_I = mij.imag - mij_ref_I

    return mij_R, mij_I

class ExportData:
    def __init__(self, def_quantity, def_units:str, filename:str, directory='.', check=True):
        def_meas = "_" + str(def_quantity).replace(".", "p") + def_units
        filename = filename + def_meas
        self.format_data = ".csv" #Format for exported data, if changed export_data needs to be modified
        self.format_figure = ".png" #Format for exported figure
        if check:
            self.file_name = self.check_file(filename, directory)
        else:
            self.file_name = filename
        self.directory = directory

        self.path_data = os.path.join(self.directory, self.file_name + self.format_data)
        self.path_figure = os.path.join(self.directory, self.file_name + self.format_figure)

    def check_file(self, name:str, dic):
        # Ensure the directory exists
        os.makedirs(dic, exist_ok=True)
        # Create the full file path
        file_path_1 = os.path.join(dic,  name + self.format_data)
        file_path_2 = os.path.join(dic, name + self.format_figure)

        new_filename = name
        # Check if the file already exists
        if os.path.exists(file_path_1) or os.path.exists(file_path_2) :
            base, ext = os.path.splitext(name)
            counter = 1
            # Modify the filename to create a unique one
            while os.path.exists(file_path_1) or os.path.exists(file_path_2):
                new_filename = f"{base}_{counter}{ext}"
                file_path_1 = os.path.join(dic, new_filename + self.format_data)
                file_path_2 = os.path.join(dic, new_filename + self.format_figure)
                counter += 1

        return new_filename

    def export_data(self, table, comments, header):
        with open(self.path_data, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write comments
            for line in comments:
                file.write(line + '\n')
            # Write the data
            for index, item in enumerate(header):
                if index == len(header) - 1:
                    file.write(f"{item}\n")
                else:
                    file.write(f"{item},")
            writer.writerows(table)

        print(f"Data exported to {self.path_data}")

    def export_data_here(self, table, comments, header):
        with open("data.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            # Write comments
            for line in comments:
                file.write(line + '\n')
            # Write the data
            for index, item in enumerate(header):
                if index == len(header) - 1:
                    file.write(f"{item}\n")
                else:
                    file.write(f"{item},")
            writer.writerows(table)

        print(f"Data exported to {self.path_data}")

    def plot_data(self, x_var, y_var_list, labels_list, x_axis_label:str , y_axis_label:str):
        colors = ['b', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:orange']
        plt.figure(figsize=(8, 6))

        for array, name, color in zip(y_var_list, labels_list, colors):
            plt.plot(x_var, array.real, marker='o', markerfacecolor='none', linestyle='-', color= color,
                     label=name)

        plt.grid(True)
        plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        # Save the plot as an image file
        plt.savefig(self.path_figure, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {self.path_figure}")

    def plot_data_here(self, x_var, y_var_list, labels_list, x_axis_label:str , y_axis_label:str):
        colors = ['b', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:orange']
        plt.figure(figsize=(8, 6))

        for array, name, color in zip(y_var_list, labels_list, colors):
            plt.plot(x_var, array.real, marker='o', markerfacecolor='none', linestyle='-', color= color,
                     label=name)

        plt.grid(True)
        plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        # Save the plot as an image file
        plt.savefig("plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {self.path_figure}")