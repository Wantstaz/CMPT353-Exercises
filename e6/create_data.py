import time
import pandas as pd
import numpy as np
from implementations import all_implementations

def main():
    
    # Initialize an empty DataFrame to store the results
    columns = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
    data = pd.DataFrame(columns=columns)
    index = 0
    
    for sort in all_implementations:
        list = np.empty(200)

        for idx in range(200):
            random_array = np.random.randint(10000, size=3000)
            st = time.time()
            res = sort(random_array)
            en = time.time()
            list[idx] = en - st
        
        data[columns[index]] = list
        index += 1

    data.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()


