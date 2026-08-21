"""
Command-line interface for running utility evaluation
"""

import json

from os import mkdir, path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER

from sanitisation_techniques.sanitiser import SanitiserNHS
from generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes,
                                                Cvine,
                                                CvineSensitive,
                                                IMRV)
from generative_models.CTGAN import CTGAN
from generative_models.TVAE import TVAE
from generative_models.pate_gan import PATEGAN
from generative_models.PRIVPGD import PrivPGD
from predictive_models.predictive_model import RandForestClassTask, LogRegClassTask, LinRegTask

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='outputs/test', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    np.random.seed(SEED)
    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    ########################
    #### GAME INPUTS #######
    ########################
    # Train test split
    train_ind = np.random.choice([True, False], rawPop.shape[0], replace=True, p=[0.7,0.3])
    rawTrain = rawPop.iloc[train_ind]
    rawTest = rawPop.iloc[~train_ind]

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'IndependentHistogram':
                for params in paramsList:
                    gmList.append(IndependentHistogram(metadata, *params))
            elif gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            elif gm == 'CTGAN':
                for params in paramsList:
                    gmList.append(CTGAN(metadata, *params))
            elif gm == 'PATEGAN':
                for params in paramsList:
                    gmList.append(PATEGAN(metadata, *params))
            # Added:
            elif gm == 'TVAE':
                for params in paramsList:
                    gmList.append(TVAE(metadata, *params))
            elif gm == 'PrivPGD':
                for params in paramsList:
                    gmList.append(PrivPGD(metadata, *params))
            elif gm == 'Cvine':
                for params in paramsList:
                    gmList.append(Cvine(metadata, *params))
            elif gm == 'CvineSensitive':
                for params in paramsList:
                    gmList.append(CvineSensitive(metadata, *params))
            elif gm == 'IMRV':
                for params in paramsList:
                    gmList.append(IMRV(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')

    # Instatiate classifier for utility evaluation
    for taskName, paramsList in runconfig['utilityTasks'].items():
        if taskName == 'RandForestClass':
            for params in paramsList:
                um = RandForestClassTask(metadata, *params)
        elif taskName == 'LogRegClass':
            for params in paramsList:
                um = LogRegClassTask(metadata, *params)

    ##################################
    ######### EVALUATION #############
    ##################################
    keys = ["Raw"] + [gm.__name__ for gm in gmList]
    results = {k: {m: None for m in ['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR']} for k in keys}

    # Train on real test on real data
    um.train(rawTrain)
    rawMetrics = um.get_metrics(rawTest)
    results['Raw']['Accuracy'] = rawMetrics['Accuracy']
    results['Raw']['F1'] = rawMetrics['F1']
    results['Raw']['AUC-ROC'] = rawMetrics['AUC-ROC']
    results['Raw']['AUC-PR'] = rawMetrics['AUC-PR']

    # Iterate over generative models
    for gm in gmList:
        # Fit generative model and simulate synthetic data
        gm.fit(rawTrain)
        synData = gm.generate_samples(runconfig['sizeSynT'])
        # Train on synthetic test on real data
        um.train(synData)
        synMetrics = um.get_metrics(rawTest)
        results[gm.__name__]['Accuracy'] = synMetrics['Accuracy']
        results[gm.__name__]['F1'] = synMetrics['F1']
        results[gm.__name__]['AUC-ROC'] = synMetrics['AUC-ROC']
        results[gm.__name__]['AUC-PR'] = synMetrics['AUC-PR']


    ##################################
    ######### SAVE RESULTS ###########
    ##################################
    outfile = f"utility_metrics_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()