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
                                                CVCDA)

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
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
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

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])
    
    targets = rawPop.loc[targetIDs, :]
    standTargets = standRawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

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
            else:
                raise ValueError(f'Generative model {gm} is not an implemented vine copula model')

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {sa: {f'{gm.__name__}_trunc_{t}': {} for gm in gmList for t in runconfig['truncLevels']} for sa in runconfig['sensitiveAttributes']} for tid in targetIDs}
    # Add entry for raw
    for tid in targetIDs:
        for sa in runconfig['sensitiveAttributes']:
            resultsTargetPrivacy[tid][sa]['Raw'] = {}

    # For checking regression coefficients
    column_names = [f'b{i}' for i in range(rawPop.shape[1])] + ["sa", "tid", "genModel"]
    regCoeff = []

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = np.random.choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
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

        for sa, Attack in attacks.items():
            for tid in targetIDs:
                resultsTargetPrivacy[tid][sa]['Raw'][nr] = {
                    'AttackerGuess': [],
                    'ProbCorrect': [],
                    'MSE': [],
                    'TargetPresence': [LABEL_OUT for _ in range(runconfig['nSynT'])]
                }

                
        #### Assess advantage raw
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target], ignore_index=True)

            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = targetSecret
                pCorrect = 1
                mse = 0

                resultsTargetPrivacy[tid][sa]['Raw'][nr]['AttackerGuess'].append(guess)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['ProbCorrect'].append(pCorrect)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['MSE'].append(mse)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['TargetPresence'].append(LABEL_IN)
        
        ##### Assess advantage Syn
        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            
            # Fit model
            GenModel.fit(rawTout, sensitive=sensitive)
            
            # Consider all truncation levels
            for trunc_lvl in runconfig['truncLevels']:
                LOGGER.info(f'Evaluate {GenModel.__name__} with truncation level {trunc_lvl}')
                synTwithoutTarget = [GenModel.generate_samples(nsamples=runconfig['sizeSynT'], trunc_lvl=trunc_lvl) for _ in range(runconfig['nSynT'])]

                for sa, Attack in attacks.items():
                    for tid in targetIDs:
                        resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr] = {
                            'AttackerGuess': [],
                            'ProbCorrect': [],
                            'MSE': [],
                            'TargetPresence': [LABEL_OUT for _ in range(runconfig['nSynT'])]
                        }

                    for syn in synTwithoutTarget:
                        standSyn = standardize_before_AIA(syn, metadata, scaler)
                        assert not standSyn.isna().any().any()
                        Attack.train(standSyn)

                        for tid in targetIDs:
                            target = standTargets.loc[[tid]]
                            targetAux = target.loc[[tid], Attack.knownAttributes]
                            targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                            guess = Attack.attack(targetAux)
                            pCorrect = int(targetSecret - runconfig["alpha"] < guess < targetSecret + runconfig["alpha"] )
                            mse = (guess - targetSecret)**2

                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['AttackerGuess'].append(guess)
                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['ProbCorrect'].append(pCorrect)

                del synTwithoutTarget

                for tid in targetIDs:
                    LOGGER.info(f'Target: {tid}')
                    target = targets.loc[[tid]]
                    rawTin = pd.concat([rawTout, target], ignore_index=True)
                    rawTin['Y'] = rawTin['Y'].astype(str) # for classification, check if it works for regression ...
                    GenModel.fit(rawTin, sensitive=sensitive)

                    synTwithTarget = [GenModel.generate_samples(nsamples=runconfig['sizeSynT'], trunc_lvl=trunc_lvl) for _ in range(runconfig['nSynT'])]

                    for sa, Attack in attacks.items():
                        targetAux = target.loc[[tid], Attack.knownAttributes]
                        targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                        for syn in synTwithTarget:
                            standSyn = standardize_before_AIA(syn, metadata, scaler)
                            assert not standSyn.isna().any().any()

                            Attack.train(standSyn)

                            # For checking regression coefficients
                            new_row = np.append(Attack.coefficients, [sa, tid, f"{GenModel.__name__}_trunc_{trunc_lvl}"])
                            regCoeff.append(new_row)
                            
                            guess = Attack.attack(targetAux)
                            pCorrect = int( targetSecret - runconfig["alpha"] < guess < targetSecret + runconfig["alpha"] )
                            mse = (guess - targetSecret)**2

                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['AttackerGuess'].append(guess)
                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['ProbCorrect'].append(pCorrect)
                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['MSE'].append(mse)
                            resultsTargetPrivacy[tid][sa][f"{GenModel.__name__}_trunc_{trunc_lvl}"][nr]['TargetPresence'].append(LABEL_IN)
                del synTwithTarget


    # Save results
    outfile = f"VineResultsMLEAI_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    # For saving regression coefficients
    regCoeff = pd.DataFrame(regCoeff, columns=column_names)
    regCoeff.to_csv(f"{args.outdir}/reg_coeff_vine_{attacks.keys()}.csv", index=False)

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)

if __name__ == "__main__":
    main()