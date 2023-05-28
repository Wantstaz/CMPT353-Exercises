import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# Get the command line arguments (as strings)
filename1 = sys.argv[1]
filename2 = sys.argv[2]

### Plot 1: Distribution of Views
data1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
            names=['lang', 'page', 'views', 'bytes'])
sorted_data = data1.sort_values('views', ascending=False)

# Create the figure and subplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

# Plot the views against the x-coordinates
x = np.arange(len(sorted_data))
plt.plot(x, sorted_data['views'].values)
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')

### Plot 2: Hourly Views
data2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
                    names=['lang', 'page', 'views', 'bytes'])

# Create a DataFrame combining views from both days
combined_data = pd.DataFrame(columns=('views1', 'views2'))

plt.subplot(1, 2, 2)

# Scatterplot of views from day 1 and day 2
combined_data['views1'] = data1['views']
combined_data['views2'] = data2['views']
plt.scatter(combined_data['views1'].values, combined_data['views2'].values)
plt.title('Hourly Correlation')
plt.xlabel('Hour 1 views')
plt.ylabel('Hour 2 views')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

# plt.show()
plt.savefig('wikipedia.png')




