import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import ttest_ind

def main():
    
    data = pd.read_csv('data.csv')

    # print mean_time to rank the sorting implementations
    mean_time = data.mean()
    ranked_data = mean_time.sort_values(ascending=True)
    print(ranked_data)

    # use ANOVA to determine if the means of multiple samples are different
    anova = stats.f_oneway(data['qs1'], data['qs2'], data['qs3'], data['qs4'], data['qs5'],
                        data['merge1'], data['partition_sort'])
    print('The p-value of ANOVA test: ', anova.pvalue)
    
    # use Post Hoc Analysis to compare all pairs
    data_melt = pd.melt(data)
    posthoc = pairwise_tukeyhsd(data_melt['value'], data_melt['variable'], alpha=0.05)
    # print('The pairs which experiment could not conclude:')
    print(posthoc)

    
if __name__ == '__main__':
    main()