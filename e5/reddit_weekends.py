import sys
import numpy as np
import pandas as pd

from datetime import date
from scipy import stats
from matplotlib import pyplot as plt

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    reddit_counts = sys.argv[1]

    # ...
    
    # Read the compressed JSON file
    counts = pd.read_json(reddit_counts, lines=True)

    # Look only at values: 
    # (1) in 2012 and 2013 
    # (2) in the /r/canada subreddit.
    counts['day'] = counts['date'].dt.weekday
    counts['year'] = counts['date'].dt.year 
    data = counts[((counts['year'] == 2012) | (counts['year'] == 2013)) & (counts['subreddit'] == 'canada')]

    # Separate weekdays and weekends
    # weekdays = data.loc[data['date'].dt.weekday < 5]
    # weekends = data.loc[data['date'].dt.weekday >= 5]
    weekdays = data[data['date'].dt.weekday.isin(list(range(0, 5)))]
    weekends = data[data['date'].dt.weekday.isin(list(range(5, 7)))]
    # print(weekdays['comment_count'].mean())
    # print(weekends['comment_count'].mean())
    

    # Student's T-Test
    # init_t_statistic, init_p_value = stats.ttest_ind(weekdays['comment_count'],
    #                                                  weekends['comment_count'], equal_var=False)
    ttest_p_value = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count']).pvalue

    # Use stats.normaltest to see if the data is normally-distributed
    weekdays_normality = stats.normaltest(weekdays['comment_count']).pvalue
    weekends_normality = stats.normaltest(weekends['comment_count']).pvalue

    # Use stats.levene to see if the two data sets have equal variances
    init_levene = stats.levene(weekdays['comment_count'], weekends['comment_count']).pvalue

    # Fix 1: transforming data might save us.
    # transformed_weekdays = np.log(weekdays['comment_count'])
    # transformed_weekends = np.log(weekends['comment_count'])
    # transformed_weekdays = np.exp(weekdays['comment_count'])
    # transformed_weekends = np.exp(weekends['comment_count'])
    transformed_weekdays = np.sqrt(weekdays['comment_count'])
    transformed_weekends = np.sqrt(weekends['comment_count'])
    # transformed_weekdays = (weekdays['comment_count']) ** 2
    # transformed_weekends = (weekends['comment_count']) ** 2
    trans_weekday_normality = stats.normaltest(transformed_weekdays).pvalue
    trans_weekend_normality = stats.normaltest(transformed_weekends).pvalue

    trans_levene = stats.levene(transformed_weekdays, transformed_weekends).pvalue

    # Fix 2: the Central Limit Theorem might save us.
    weekdays_group = weekdays['date'].apply(lambda x: x.isocalendar()[:2])
    weekends_group = weekends['date'].apply(lambda x: x.isocalendar()[:2])
    grouped_weekdays = weekdays.groupby(weekdays_group).aggregate('mean')
    grouped_weekends = weekends.groupby(weekends_group).aggregate('mean')
    grouped_weekends_normality = stats.normaltest(grouped_weekends['comment_count']).pvalue 
    grouped_weekdays_normality = stats.normaltest(grouped_weekdays['comment_count']).pvalue
    
    weekly_levene = stats.levene(grouped_weekdays['comment_count'], grouped_weekends['comment_count']).pvalue
    weekly_p_value = stats.ttest_ind(grouped_weekdays['comment_count'], grouped_weekends['comment_count']).pvalue
    
    # Fix 3: a non-parametric test might save us.
    utest_p_value = stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count'], alternative='two-sided').pvalue
    
    # print(OUTPUT_TEMPLATE.format(
    #     initial_ttest_p=0,
    #     initial_weekday_normality_p=0,
    #     initial_weekend_normality_p=0,
    #     initial_levene_p=0,
    #     transformed_weekday_normality_p=0,
    #     transformed_weekend_normality_p=0,
    #     transformed_levene_p=0,
    #     weekly_weekday_normality_p=0,
    #     weekly_weekend_normality_p=0,
    #     weekly_levene_p=0,
    #     weekly_ttest_p=0,
    #     utest_p=0,
        
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = ttest_p_value,
        initial_weekday_normality_p = weekdays_normality,
        initial_weekend_normality_p = weekends_normality,
        initial_levene_p = init_levene,
        transformed_weekday_normality_p = trans_weekday_normality,
        transformed_weekend_normality_p = trans_weekend_normality,
        transformed_levene_p = trans_levene,  
        weekly_weekday_normality_p = grouped_weekdays_normality,
        weekly_weekend_normality_p = grouped_weekends_normality,
        weekly_levene_p = weekly_levene,
        weekly_ttest_p = weekly_p_value,
        utest_p = utest_p_value,
    ))


if __name__ == '__main__':
    main()
