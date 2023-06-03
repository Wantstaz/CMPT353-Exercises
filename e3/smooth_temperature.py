import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

filename = sys.argv[1]
cpu_data = pd.read_csv(filename)

# Convert timestamp to datetime (string to float) # Then sort
cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'])
cpu_data.sort_values('timestamp', inplace=True)

# LOESS Smoothing
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=0.08)

# Kalman Smoothing
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([0.3, 0.3, 0.3, 0.3]) ** 2
transition_covariance = np.diag([0.01, 0.01, 0.01, 0.01]) ** 2
transition = [[0.94, 0.5, 0.2, -0.001], [0.1, 0.4, 2.1, 0], 
              [0, 0, 0.94, 0], [0, 0, 0, 1]]

kf = KalmanFilter(
    initial_state_mean=initial_state,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
)
kalman_smoothed, _ = kf.smooth(kalman_data)


# Plot the original temperature data and the smoothed temperature
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')

# Set axis label and line label
plt.xlabel('Datetime')
plt.ylabel('Temp (°C)')
plt.legend(['data points', 'LOESS-smoothed line', 'Kalman-smoothed line'])
# plt.show() # maybe easier for testing
plt.savefig('cpu.svg') # for final submission

