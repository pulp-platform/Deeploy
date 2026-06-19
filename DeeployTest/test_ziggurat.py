# parse_and_check_gaussian.py

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

# Regex: float, int, int (e.g., 0.12345 2 123456)
pattern = re.compile(r'^\s*([-+]?\d*\.\d+|\d+)[ \t]+([-+]?\d+)[ \t]+([-+]?\d+)\s*$')

samples = []

with open('u_log.txt') as f:
    for line in f:
        m = pattern.match(line)
        if m:
            sample = float(m.group(1))
            # core_id = int(m.group(2))
            # seed = int(m.group(3))
            samples.append(sample)

samples = np.array(samples)
print(f"Parsed {len(samples)} samples.")

# Basic stats
print(f"Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")

# Normality test
stat, p = normaltest(samples)
print(f"Normality test p-value: {p:.4g}")
if p > 0.05:
    print("Samples look Gaussian (fail to reject normality).")
else:
    print("Samples do NOT look Gaussian (reject normality).")

# Optional: plot histogram
plt.hist(samples, bins=50, density=True)
plt.title("Histogram of Gaussian Samples")
plt.xlabel("Sample Value")
plt.ylabel("Density")
plt.savefig("ziggurat_histogram.png")