"""
Exercise 3.19

Design of a nine-tap transversal minimum MSE equalizing filter

NOTE: I am not correcting values printed in the textbook. There may be missing
      minus signs.
"""

import numpy as np


# Desired output over the filter output support time (z)
desired_output = np.zeros((17, 1))
desired_output[8, 0] = 1

# Signal matrix (x) and its transpose
signal_matrix = np.array([
    [0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.10, 0.05, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.20, 0.10, 0.05, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00],
    [1.00, 0.20, 0.10, 0.05, 0.02, 0.00, 0.00, 0.00, 0.00],
    [0.10, 1.00, 0.20, 0.10, 0.05, 0.02, 0.00, 0.00, 0.00],
    [0.03, 0.10, 1.00, 0.20, 0.10, 0.05, 0.02, 0.00, 0.00],
    [0.02, 0.03, 0.10, 1.00, 0.20, 0.10, 0.05, 0.02, 0.00],
    [0.01, 0.02, 0.03, 0.10, 1.00, 0.20, 0.10, 0.05, 0.02],
    [0.00, 0.01, 0.02, 0.03, 0.10, 1.00, 0.20, 0.10, 0.05],
    [0.00, 0.00, 0.01, 0.02, 0.03, 0.10, 1.00, 0.20, 0.10],
    [0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.10, 1.00, 0.20],
    [0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.10, 1.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.10],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01]
])
signal_transpose = signal_matrix.T

# Rxx
signal_autocorrelation_matrix = np.matmul(signal_transpose, signal_matrix)

# Rxz
signal_output_crosscorr_matrix = np.matmul(signal_transpose, desired_output)

# Taps
filter_taps = np.matmul(
    np.linalg.inv(signal_autocorrelation_matrix),
    signal_output_crosscorr_matrix
)

# Print the taps
print("Filter Taps:")
for idx, tap in enumerate(filter_taps):
    print(f"Tap {idx}: {tap}")
print()

# Calculate the actual filter outputs
actual_outputs = np.array(
    [np.matmul(row, filter_taps) for row in signal_matrix]
)

# Print the outputs and the additional components of interest
print("Filter outputs:")
for cycle_idx, output in enumerate(actual_outputs):
    print(f"Cycle {cycle_idx}: {output}")
print()
isi_contributors = np.delete(actual_outputs, (8), 0)
print(f"Largest magnitude contributing to ISI: {np.max(isi_contributors)}")
print(f"Sum of magnitudes contributing to ISI: {np.sum(isi_contributors)}")
