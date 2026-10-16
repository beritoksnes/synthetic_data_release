"""
Command-line interface for running privacy evaluation of vine copula models under an attribute inference adversary
"""

import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from os import mkdir, path
from argparse import ArgumentParser

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from generative_models.data_synthesiser import (Cvine,
                                                CvineSensitive,
                                                IMRV,
                                                CVCDA,
                                                CVCDASensitive)

from attack_models.reconstruction import LinRegAttack, RandForestAttack
from utils.evaluation_framework import standardize_before_AIA

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
    dname = args.datapath.split('/')[-1]
    n, d = rawPop.shape
    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    # Set seed for reproducibility
    np.random.seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    
    # Get sensitive features 
    sensitive = [sa for sa in runconfig['sensitiveAttributes'].keys()]
    
    # Standardize rawPop
    scaler = StandardScaler()
    standRawPop = standardize_before_AIA(rawPop, metadata, scaler)

    # Pick targets
    targetIDs = np.random.choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()
    
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])
    
    targets = rawPop.loc[targetIDs, :]
    standTargets = standRawPop.loc[targetIDs, :]

    # Drop targets from population (will be added in turn later)
    rawPopDropTargets = rawPop.drop(targetIDs)

    # List of candidate vine copula-based generative models to evaluate
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

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsRegCoeff = pd.DataFrame(columns=['genModel', 'truncLvl', 'sa', 'tid', 'gameID', 'synDataID', 'coefficient', 'value'])

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a reference dataset from the raw data without target observations
        rIdx = np.random.choice(list(rawPopDropTargets.index), 
                                size=runconfig['sizeRawT'], 
                                replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        ###############
        ## ATTACKS ####
        ###############
        attacks = {}
        for sa, atype in runconfig['sensitiveAttributes'].items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)
        
        # Iterate over all generative models
        for gm in gmList:
            LOGGER.info(f'Start: Evaluation for model {gm.__name__}...')
            gm.fit(rawTout, sensitive=sensitive)
            
            # Consider all truncation levels
            for trunc_lvl in runconfig['truncLevels']:
                LOGGER.info(f'Evaluate truncation level {trunc_lvl}')
                
                # Add target observation to reference data set one at a time
                for tid in targetIDs:
                    LOGGER.info(f'Target: {tid}')
                    target = targets.loc[[tid]]
                    rawTin = pd.concat([rawTout, target], ignore_index=True)
                    rawTin['Y'] = rawTin['Y'].astype(str)
                    
                    # Fit generateive model and sample several synthetic data sets with given truncation level
                    gm.fit(rawTin, sensitive=sensitive)
                    synTwithTarget = [gm.generate_samples(nsamples=runconfig['sizeSynT'], trunc_lvl=trunc_lvl) for _ in range(runconfig['nSynT'])]

                    # Attack training for each sensitive attribute
                    for sa, Attack in attacks.items():
                        j_star = np.where(sa==rawTin.columns)[0][0]
                        coeff_names = rawTin.columns[:j_star].tolist() + rawTin.columns[j_star+1:].tolist()

                        for synID, syn in enumerate(synTwithTarget):
                            standSyn = standardize_before_AIA(syn, metadata, scaler)
                            assert not standSyn.isna().any().any()
                            Attack.train(standSyn)
                            
                            # Store regression coefficients for MAB computation
                            tmp = pd.DataFrame({
                                'genModel':[gm.__name__]*(d-1),  # vine model
                                'truncLvl':[trunc_lvl]*(d-1),    # truncation level
                                'sa':[sa]*(d-1),                 # sensitive attribute
                                'tid':[tid]*(d-1),               # target observation 
                                'gameID':[nr]*(d-1),             # game iteration
                                'synDataID':[synID]*(d-1),       # synthetic data set ID 
                                'coefficient':coeff_names,       # coefficient names
                                'value':Attack.coefficients[1:]  # store regression coefficients except for intercept
                                })
                            
                            resultsRegCoeff = pd.concat([resultsRegCoeff, tmp], axis=0)
                            
                del synTwithTarget

    # Save results
    LOGGER.info(f"Write results")
    resultsRegCoeff.to_csv(f"{args.outdir}/reg_coeff.csv", index=False)

if __name__ == "__main__":
    main()