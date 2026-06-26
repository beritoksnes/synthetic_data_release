import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analyse_results import load_results_utility, plt_avg_accuracy
dirname = 'tests/utility/'
utility_record, utility_agg = load_results_utility(dirname)

labelVar = 'RiskMortality'
models = ['Raw','SanitiserNHSk10', 'BayesianNet', 'PrivBayesEps1.0']
fig = plt_avg_accuracy(utility_agg, models)