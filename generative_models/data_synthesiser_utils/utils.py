from string import ascii_lowercase
from itertools import combinations
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

import numpy as np
import pandas as pd

import json


def mutual_information(labels_x: pd.Series, labels_y: pd.DataFrame):
    """Mutual information of distributions in format of pd.Series or pd.DataFrame.

    Parameters
    ----------
    labels_x : pd.Series
    labels_y : pd.DataFrame
    """
    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.values), axis=1)

    return mutual_info_score(labels_x, labels_y)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a pd.DataFrame."""
    sorted_columns = sorted(dataset.columns)
    mi_df = pd.DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)


def infer_numerical_attributes_in_dataframe(dataframe):
    describe = dataframe.describe()
    # pd.DataFrame.describe() usually returns 8 rows.
    if describe.shape[0] == 8:
        return set(describe.columns)
    # pd.DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
    else:
        return set()


def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print('Constructed Bayesian network:')
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


def generate_random_string(length):
    return ''.join(np.random.choice(list(ascii_lowercase), size=length))


def bayes_worker(paras):
    child, V, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(V):
        for other_parents in combinations(V[split + 1:], num_parents - 1):
            parents = list(other_parents)
            parents.append(V[split])
            parents_pair_list.append((child, parents))
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.
    Parameters
    ----------
    num_tuples : int
        Number of tuples in sensitive dataset.
    Return
    --------
    int
        Sensitivity value.
    """
    if attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]]):
        a = np.log(num_tuples) / num_tuples
        b = (num_tuples - 1) / num_tuples
        b_inv = num_tuples / (num_tuples - 1)
        return a + b * np.log(b_inv)
    else:
        a = (2 / num_tuples) * np.log((num_tuples + 1) / 2)
        b = (1 - 1 / num_tuples) * np.log(1 + 2 / (num_tuples - 1))
        return a + b


def calculate_delta(num_attributes, sensitivity, epsilon):
    """Computing delta, which is a factor when applying differential privacy.
    More info is in PrivBayes Section 4.2 "A First-Cut Solution".
    Parameters
    ----------
    num_attributes : int
        Number of attributes in dataset.
    sensitivity : float
        Sensitivity of removing one tuple.
    epsilon : float
        Parameter of differential privacy.
    """
    return (num_attributes - 1) * sensitivity / epsilon


def exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples, num_attributes):
    """Applied in Exponential Mechanism to sample outcomes."""
    delta_array = []
    for (child, parents) in parents_pair_list:
        sensitivity = calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary)
        delta = calculate_delta(num_attributes, sensitivity, epsilon)
        delta_array.append(delta)

    mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
    mi_array = np.exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array


def privpgd_discretize_data(data, except_for, num_bins, json_filename='./data/ranges.json'):
    
    # Load JSON metadata
    with open(json_filename, 'r') as file:
        metadata = json.load(file)
    
    # Create a dictionary to easily retrieve metadata for each column
    column_metadata = {col['name']: col for col in metadata['columns']}
    
    # Determine columns to discretize by excluding the specified columns
    columns_to_discretize = [col for col in data.columns if col not in except_for]

    # Discretize each specified column into the specified number of equally spaced bins
    for column in columns_to_discretize:
        if column in data.columns:
            # Fetch min and max values from the metadata
            min_val = column_metadata[column]['min']
            max_val = column_metadata[column]['max']

            # Clip the column to the range specified by min and max
            clipped_series = data[column].clip(lower=min_val, upper=max_val)

            # Calculate bin edges
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            # Create equally spaced bins
            data[column] = pd.cut(clipped_series, bins=bin_edges, labels=False, include_lowest=True)
        else:
            print(f"Warning: Column '{column}' not found in the CSV file.")
    
    return data


def privpgd_revert_discretization(original_data, discretized_data, except_for, num_bins, json_filename='./data/ranges.json'):
    """
    Reverts the discretization of specified columns in a CSV file using the original dataset.

    :param original_data: Original dataframe with continuous data.
    :param discretized_data: Dataframe with discretized data.
    :param except_for: List of column names that were not discretized and should not be reverted.
    :param num_bins: Number of bins used in the discretization process.
    :param json_filename: File name of json file containing the covariates ranges.
    :return: pd.DataFrame with reverted continuous values.
    """

    # Load JSON metadata
    with open(json_filename, 'r') as file:
        metadata = json.load(file)
    
    # Create a dictionary to easily retrieve metadata for each column
    column_metadata = {col['name']: col for col in metadata['columns']}

    # Determine columns to revert by excluding the specified columns
    columns_to_revert = [col for col in discretized_data.columns if col not in except_for]

    # Revert discretization for each specified column
    for column in columns_to_revert:
        if column in discretized_data.columns:
            # Fetch min and max values from the metadata
            min_val = column_metadata[column]['min']
            max_val = column_metadata[column]['max']

            bin_width = (max_val - min_val) / num_bins

            # Calculate the center of each bin
            discretized_data[column] = discretized_data[column].apply(lambda x: min_val + (x + 0.5) * bin_width)
        else:
            print(f"Warning: Column '{column}' not found in the discretized file.")

    return discretized_data