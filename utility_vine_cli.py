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

from generative_models.data_synthesiser import (Cvine,
                                                CvineSensitive,
                                                IMRV,
                                                CVCDA,
                                                CVCDASensitive)

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
    
    n, d = rawPop.shape
    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    np.random.seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    # Get sensitive features 
    sensitive = [sa for sa in runconfig['sensitiveAttributes'].keys()]

    # List of candidate vine copula models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'Cvine':
                for params in paramsList:
                    gmList.append(Cvine(metadata, *params))
            elif gm == 'CvineSensitive':
                for params in paramsList:
                    gmList.append(CvineSensitive(metadata, *params))
            elif gm == 'IMRV':
                for params in paramsList:
                    gmList.append(IMRV(metadata, *params))
            elif gm == 'CVCDA':
                for params in paramsList:
                    gmList.append(CVCDA(metadata, *params))
            elif gm == 'CVCDASensitive':
                for params in paramsList:
                    gmList.append(CVCDASensitive(metadata, *params))
            else:
                raise ValueError(f'Generative model {gm} is not an implemented vine copula model')

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
    results = {"Raw": {'Accuracy':np.zeros(runconfig["nIter"]), 'AUC-ROC':np.zeros(runconfig["nIter"])}}
    results.update({gm.__name__ : {trunc_lvl : {'Accuracy':np.zeros(runconfig["nIter"]), 
                                                'AUC-ROC':np.zeros(runconfig["nIter"])} for trunc_lvl in runconfig['truncLevels']} for gm in gmList})

    for nr in range(runconfig["nIter"]):
        # Draw reference data set
        idx = np.random.choice(range(n), runconfig["sizeRawT"], replace=False)
        ref = np.zeros(n, dtype=bool)
        ref[idx] = True
        rawTrain, rawTest = rawPop.iloc[ref], rawPop.iloc[~ref]

        # Train on real test on real data
        um.train(rawTrain)
        rawMetrics = um.get_metrics(rawTest)
        results['Raw']['Accuracy'][nr] = rawMetrics['Accuracy']
        results['Raw']['AUC-ROC'][nr] = rawMetrics['AUC-ROC']

        # Iterate over vine models
        for gm in gmList:
            # Fit vine model 
            gm.fit(rawTrain, sensitive=sensitive)
            # Consider all truncation levels
            for trunc_lvl in runconfig['truncLevels']:
                # Simulate data from fitted model with selected truncation level
                synData = gm.generate_samples(nsamples=runconfig['sizeSynT'], trunc_lvl=trunc_lvl)
                # Train on synthetic test on real data
                um.train(synData)
                synMetrics = um.get_metrics(rawTest)
                results[gm.__name__][trunc_lvl]['Accuracy'][nr] = synMetrics['Accuracy']
                results[gm.__name__][trunc_lvl]['AUC-ROC'][nr] = synMetrics['AUC-ROC']


    ##################################
    ######### SAVE RESULTS ###########
    ##################################
    outfile = f"utility_metrics_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()