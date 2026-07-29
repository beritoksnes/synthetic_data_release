# Based on https://github.com/ElisabethGriesbauer/synthetic_data_release/blob/master/generative_models/tvae.py
# Original author: Elisabeth Griesbauer, 2024
# Modified by: Berit Omli Øksnes, 2026

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import LOGGER

import pandas as pd

from generative_models.generative_model import GenerativeModel
from sdv.single_table.ctgan import TVAESynthesizer
from sdv.metadata import SingleTableMetadata


class TVAE(GenerativeModel):
    def __init__(self,
                 metadata=None,
                 epochs=1500,
                 batch_size=400,
                 embedding_dim=3,
                 compress_dims=(128, 128),
                 decompress_dims=(128, 128),
                 l2scale=1e-5,
                 loss_factor=2,
                 verbose=True,
                 multiprocess=False,
                 cuda=True):
        
        self.metadata = metadata
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.loss_factor = loss_factor
        self.verbose = verbose
        self.multiprocess = bool(multiprocess)
        self.cuda = cuda

        self.datatype = pd.DataFrame
        self.infer_ranges = True
        self.trained = False
        self.__name__ = 'TVAE'

    def fit(self, data, *args):
        """Train a tabular variational autoencoder."""
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        self.metadata = metadata

        self.synthesiser = TVAESynthesizer(
            metadata=metadata,
            cuda = self.cuda,
            epochs=self.epochs,
            batch_size=self.batch_size,
            compress_dims = self.compress_dims,
            decompress_dims = self.decompress_dims,
            embedding_dim = self.embedding_dim,
            l2scale = self.l2scale,
            loss_factor = self.loss_factor,
            verbose = self.verbose)

        self.synthesiser._model_kwargs = self.synthesiser._model_kwargs | {'verbose': self.verbose}

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        self.synthesiser.fit(data)
        LOGGER.debug(f'Finished fitting')
        self.trained = True

        return self

    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_data = self.synthesiser.sample(num_rows=nsamples, output_file_path=None)

        return synthetic_data


    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
        
    def transform(self, X):
        return self.synthesiser.sample(num_rows=len(X), output_file_path=None)
    
if __name__ == "__main__":
    test = TVAE()
    print(test.__name__)