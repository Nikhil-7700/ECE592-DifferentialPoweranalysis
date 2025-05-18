import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load the data from the CSV file
traces_df = pd.read_csv('traces.csv', header=None)

"""
# Plot the first power trace (first row of the data)
first_trace = traces_df.iloc[0]

# Plot the first power trace
plt.figure(figsize=(12, 8))
plt.plot(first_trace)
plt.title('Figure1: First Power Trace')
plt.xlabel('Sample Points')
plt.ylabel('Power Consumption')
plt.savefig("Figure1.png")
plt.show()
"""
traces_arr = traces_df.to_numpy(dtype=np.float32)

def read_text_file(filename):
  with open(filename, 'r') as f:
    hex_numbers = [line.strip() for line in f]

  array = np.zeros((10000, 16), dtype=np.uint8)
  for i, hex_number in enumerate(hex_numbers):
    for j in range(16):
      byte_value = int(hex_number[j * 2: (j + 1) * 2], 16)
      array[i, j] = byte_value
  return hex_numbers, array

filename = "input_plaintext.txt"
inputs, input_bytes = read_text_file(filename)

power_model = np.zeros((16,10000,256), dtype=np.uint8)
state_model = np.zeros((16,10000,256), dtype=np.uint8)

def hamming_weight(num):
  count = 0
  for i in range(8):
    if num & (1 << i):
      count += 1
  return count

def get_powerModel(byte_no):
    state_model = np.zeros((10000, 256), dtype=np.uint8)
    power_model = np.zeros((10000, 256), dtype=np.uint)
    for line in range(10000):
        for i in range(256):
          state_model[line][i] = input_bytes[line][byte_no] ^ i
          power_model[line][i] = hamming_weight(state_model[line][i])
    return power_model, state_model

for i in range(16):
    power_model[i], state_model[i] = get_powerModel(i)

correlation_results = np.zeros((16, 256, 400))

# Compute correlation for each of the 256 hypothetical values across 400 timestamps
def correlate_traces_pm(traces_arr, power_model, byte_no):
  for i in range(256):  # Iterate over the 256 columns in the hypothetical value
    for j in range(400):  # Iterate over the 400 timestamps
      # Get the column from the traces and the hypothetical values
      trace_samples = traces_arr[:, j]
      correlation_column = power_model[byte_no][:, i]

      # Compute the Pearson correlation
      correlation, _ = pearsonr(trace_samples, correlation_column)
      correlation_results[byte_no][i][j] = correlation

for i in range(16):
    correlate_traces_pm(traces_arr, power_model, i)

#for i in range(16):
    plt.figure(figsize=(16, 12))
    for j in range(256):
        plt.plot(correlation_results[i][j, :], label=f'Key Guess {i:02X}', alpha=0.6)

    plt.title('Correlation of 256 Key Guesses with 400 Timestamps')
    plt.xlabel('Timestamps')
    plt.ylabel('Correlation')
    plt.savefig("Correlation_Key Hypothesis_PowerTraces_byte" + str(i) + ".png")
    plt.show()

# TOP 2 KEY GUESSES
#max_abs_correlations = []
#top_two_key_guesses = []
#best_key_guess_1 = []
#best_key_guess_2 = []

for i in range(16):
    # Find the maximum absolute correlation values
    max_abs_correlations = np.max(np.abs(correlation_results[i]), axis=1)

    # Find the indices of the two key guesses with the highest maximum absolute correlation values
    top_four_key_guesses = np.argsort(max_abs_correlations)[-4:]

    # Extract the correlation traces for these two key guesses
    best_key_guess_1 = correlation_results[13][top_four_key_guesses[3], :]
    best_key_guess_2 = correlation_results[13][top_four_key_guesses[2], :]
    best_key_guess_3 = correlation_results[13][top_four_key_guesses[1], :]
    best_key_guess_4 = correlation_results[13][top_four_key_guesses[0], :]


    # Plot the correlation traces for the two best key guesses
    plt.figure(figsize=(12, 8))
    plt.plot(best_key_guess_1, label=f'Best Key Guess {top_four_key_guesses[3]:02X}', color='blue')
    plt.plot(best_key_guess_2, label=f'Second Best Key Guess {top_four_key_guesses[2]:02X}', color='red')
    plt.plot(best_key_guess_3, label=f'Third Best Key Guess {top_four_key_guesses[1]:02X}', color='green')
    plt.plot(best_key_guess_4, label=f'Fourth Best Key Guess {top_four_key_guesses[0]:02X}', color='yellow')
        
    plt.title('Figure 3: Correlation of Two Best Key Guesses with 400 Timestamps')
    plt.xlabel('Timestamps')
    plt.ylabel('Correlation')
    plt.legend()
    # Save the figure
    plt.savefig("TopKeyGuesses_Byte" + str(i) + ".png")
    plt.show()

# Problem 4
evolution_correlation_results = np.zeros((16, 256, 10000))

# Get the index (timestamp) of the maximum correlation for each key guess
max_timestamp_per_key_guess = []
for i in range(16):
    max_timestamp_per_key_guess.append(np.argmax(np.abs(correlation_results[i]), axis=1))

# Find the most frequent maximum timestamp (the leak point)
max_leak_timestamp = []
for i in range(16):
    max_leak_timestamp.append(np.argmax(np.max(np.abs(correlation_results[i]), axis=0)))
    
    print(f"Maximum leak point for byte {i} is at timestamp {max_leak_timestamp[i]}")

def evolution_correlation(traces_arr, power_model, byte_no):
    for j in range(256):
        for i in range(2, 10000):
            correlation_column = power_model[byte_no][:, j][:i]
            traces_col_evl = traces_arr[:i, max_leak_timestamp[byte_no]]
            correlation, _ = pearsonr(traces_col_evl, correlation_column)
            evolution_correlation_results[byte_no][j][i] = correlation

for i in range(16):
    evolution_correlation(traces_arr, power_model, i)

#for i in range(16):
    plt.figure(figsize=(16, 12))
    for j in range(256):
        plt.plot(evolution_correlation_results[i][j, :], label=f'Key Guess {i:02X}', alpha=0.6)

    plt.ylim(-0.25,0.25)
    plt.title('Correlation of 256 Key Guesses with 400 Timestamps')
    plt.xlabel('No. of measurements')
    plt.ylabel('Correlation')
    plt.savefig("EvolutionCorrelationGraph_Byte" + str(i) + ".png")
    plt.show()




















