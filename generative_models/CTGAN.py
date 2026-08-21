# Based on https://github.com/spring-epfl/synthetic_data_release/blob/master/generative_models/ctgan.py
# Modified based on https://github.com/ElisabethGriesbauer/synthetic_data_release/blob/c09269a89d00d5305361a7147443f25111778960/generative_models/ctganSDV.py
# Original author: Elisabeth Griesbauer, 2024
# Modified by: Berit Omli Øksnes, 2026

# Warning to handle: The 'SingleTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers.


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pandas import DataFrame

from utils.logging import LOGGER

from generative_models.generative_model import GenerativeModel

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata



class CTGAN(GenerativeModel):
    """A conditional generative adversarial network for tabular data"""
    def __init__(self, 
                 metadata=None,
                 epochs=300,
                 batch_size=500):

        self.metadata = metadata
        self.enforce_min_max_values = True
        self.enforce_rounding = True
        self.locales = ['en_US']
        self.epochs = epochs
        self.verbose = False
        self.enable_gpu = True
        self.multiprocess = False
        self.batch_size = batch_size
        
        self.datatype = DataFrame
        self.infer_ranges = True
        self.trained = False

        self.__name__ = f'CTGAN({self.epochs}, {self.batch_size})'

    def fit(self, data):
        """Train a generative adversarial network on tabular data.
        Input data is assumed to be of shape (n_samples, n_features)
        See https://github.com/DAI-Lab/SDGym for details"""
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        self.metadata = metadata

        self.synthesiser = CTGANSynthesizer(metadata=self.metadata,
                                            enforce_min_max_values=self.enforce_min_max_values, 
                                            enforce_rounding=self.enforce_rounding, 
                                            locales=self.locales,
                                            epochs=self.epochs, 
                                            verbose=self.verbose, 
                                            enable_gpu=self.verbose,
                                            batch_size=self.batch_size)
        

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        self.synthesiser.fit(data)

        LOGGER.debug(f'Finished fitting')
        self.trained = True

    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_data = self.synthesiser.sample(nsamples)

        return synthetic_data