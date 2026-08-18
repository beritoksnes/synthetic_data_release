"""
Procedures for running a privacy evaluation on a generative model
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils.constants import *

def get_accuracy(guesses, labels, targetPresence):
    idxIn = np.where(targetPresence == LABEL_IN)[0]
    idxOut = np.where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g,l in zip(guesses[idxIn], labels[idxIn])])/len(idxIn)
    pOut = sum([g == l for g,l in zip(guesses[idxOut], labels[idxOut])])/len(idxOut)
    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    targetIn = np.where(labels == LABEL_IN)[0]
    targetOut = np.where(labels == LABEL_OUT)[0]
    return sum(guesses[targetIn] == LABEL_IN)/len(targetIn), sum(guesses[targetOut] == LABEL_IN)/len(targetOut)


def get_probs_correct(pdf, targetPresence):
    idxIn = np.where(targetPresence == LABEL_IN)[0]
    idxOut = np.where(targetPresence == LABEL_OUT)[0]
    return np.mean(pdf[idxIn]), np.mean(pdf[idxOut])


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_ai_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut

def get_ai_odds(pCorrectS, pCorrectR):
    if pCorrectR ==0:
        odds = float('nan')
    else:
        odds = pCorrectS/pCorrectR
    return odds


def get_util_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_prob_removed(before, after):
    idxIn = np.where(before == LABEL_IN)[0]
    return 1.0 - sum(after[idxIn]/len(idxIn))


def standardize_before_AIA(data, metadata, scaler):
    columns = np.array([col["name"] for col in metadata["columns"]])
    types = np.array([col["type"] for col in metadata["columns"]])
    cat_cols = columns[np.isin(types, ["Categorical", "Ordinal"])]
    cont_cols = np.setdiff1d(data.columns,cat_cols)

    scaled_values = scaler.fit_transform(data[cont_cols])
    standSample = pd.DataFrame(data=scaled_values, columns=cont_cols)

    for col in cat_cols:
        standSample[col] = data[col].values

    standSample.index = data.index
    standSample.index.name = "ID"
    
    return standSample




