import sys
import pandas as pd
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)

def main():
    searchdata_file = sys.argv[1]
    
    data = pd.read_json(searchdata_file, orient='records', lines=True)
    
    # ...
    
    # user data for stats.mannwhitneyu
    even_uid = data.loc[(data['uid'] % 2 == 0)] 
    odd_uid  = data.loc[(data['uid'] % 2 != 0)]
    # instructor data for stats.mannwhitneyu
    even_uid_instr = data.loc[(data['uid'] % 2 == 0) & data['is_instructor'] == True] 
    odd_uid_instr  = data.loc[(data['uid'] % 2 != 0) & data['is_instructor'] == True]
    
    
    # control group: searched & not_searched
    even_searched = data[(data['uid'] % 2 == 0) & (data['search_count'] > 0)]
    even_not_searched = data[(data['uid'] % 2 == 0) & (data['search_count'] == 0)]
    # treatment group: searched & not_searched
    odd_searched = data[(data['uid'] % 2 != 0) & (data['search_count'] > 0)]
    odd_not_searched = data[(data['uid'] % 2 != 0) & (data['search_count'] == 0)]
    # contingency table for user
    contingency_user  = [[len(even_searched), len(even_not_searched)], 
                         [len(odd_searched), len(odd_not_searched)]]
    
    
    # control group: instr searched & not_searched
    instr_even_searched = data[(data['uid'] % 2 == 0) & (data['search_count'] > 0) & data['is_instructor'] == True]
    instr_even_not_searched = data[(data['uid'] % 2 == 0) & (data['search_count'] == 0) & data['is_instructor'] == True]
    # treatment group: instru searched & not_searched
    instr_odd_searched = data[(data['uid'] % 2 != 0) & (data['search_count'] > 0) & data['is_instructor'] == True]
    instr_odd_not_searched = data[(data['uid'] % 2 != 0) & (data['search_count'] == 0) & data['is_instructor'] == True]
    # contingency table for instr
    contingency_instr = [[len(instr_even_searched), len(instr_even_not_searched)], 
                         [len(instr_odd_searched), len(instr_odd_not_searched)]]
    
    
    # Did more/less users use the search feature?
    more_users_p = p_value= stats.chi2_contingency(contingency_user)[1]
    # Did users search more/less?
    more_searches_p = stats.mannwhitneyu(odd_uid['search_count'], even_uid['search_count'], 
                                         alternative = 'two-sided').pvalue
    # Did more/less instructors use the search feature?
    more_instr_p = stats.chi2_contingency(contingency_instr)[1]
    # Did instructors search more/less?
    more_instr_searches_p = stats.mannwhitneyu(odd_uid_instr['search_count'], even_uid_instr['search_count'], 
                                         alternative = 'two-sided').pvalue
    
    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p = more_users_p,
        more_searches_p = more_searches_p,
        more_instr_p = more_instr_p,
        more_instr_searches_p = more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
