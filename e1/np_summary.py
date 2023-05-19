import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# TODO: Which city had the lowest total precipitation over the year? 
# Hints: sum across the rows (axis 1); 
# use argmin to determine which row has the lowest value. Print the row number.
print('Row with lowest total precipitation:')
print(np.argmin(totals.sum(axis=1)))

# TODO: Determine the average precipitation in these locations for each month. 
# That will be the total precipitation for each month (axis 0), 
# divided by the total observations for that months. Print the resulting array.
print('Average precipitation in each month:')
print((totals.sum(axis=0))/counts.sum(axis=0))

# TODO: Do the same for the cities: 
# give the average precipitation (daily precipitation averaged over the month) for each city by printing the array.
print('Average precipitation in each city:')
print((totals.sum(axis=1))/counts.sum(axis=1))

# TODO: Calculate the total precipitation for each quarter in each city (i.e. the totals for each station across three-month groups). 
# You can assume the number of columns will be divisible by 3. 
# Hint: use the reshape function to reshape to a 4n by 3 array, sum, and reshape back to n by 4.
print('Quarterly precipitation totals:')
rows = totals.shape[0]
print(totals.reshape(4*rows, 3).sum(axis=1).reshape(rows, 4))

