"""
PrivPGD generative model for the `synthetic_data_release` (Groundhog) framework.
 
Wraps Donhauser, Abad, Hulkund & Yang (ICML 2024), "Privacy-preserving data
release leveraging optimal transport and particle gradient descent".
Reference implementation: https://github.com/jaabmar/private-pgd
"""

# Author: Berit Omli Øksens, 2026
 
import logging
import os
import sys
from math import ceil, floor
 
import numpy as np
from pandas import DataFrame
 
from generative_models.generative_model import GenerativeModel
from utils.constants import CATEGORICAL, ORDINAL, INTEGER, FLOAT
 
LOGGER = logging.getLogger(__name__)

PRIVPGD_SRC = os.environ.get(
    'PRIVPGD_SRC',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'private-pgd', 'src'),
)
 
_PRIVPGD = {}

def _import_privpgd(src=None):
    """Import the private-pgd modules lazily and cache them."""
    if _PRIVPGD:
        return _PRIVPGD
    src = src or PRIVPGD_SRC
    if not os.path.isdir(src):
        raise ImportError(
            f'Could not find the private-pgd source at {src}. Clone it with\n'
            f'    git clone https://github.com/jaabmar/private-pgd.git\n'
            f'next to this repository, or set the PRIVPGD_SRC environment '
            f'variable to its "src" directory.')
    if src not in sys.path:
        sys.path.insert(0, src)
    from inference.dataset import Dataset
    from inference.domain import Domain
    from inference.privpgd.inference import AdvancedSlicedInference
    from mechanisms.kway import KWay
    from mechanisms.utils_mechanisms import generate_all_kway_workload
    _PRIVPGD.update(Dataset=Dataset, Domain=Domain,
                    AdvancedSlicedInference=AdvancedSlicedInference,
                    KWay=KWay,
                    generate_all_kway_workload=generate_all_kway_workload)
    return _PRIVPGD



class PrivPGD(GenerativeModel):
    def __init__(self,
                 metadata,
                 epsilon,
                 delta = 1.0,
                 nbins = 32,
                 degree = 2,
                 n_particles = None,
                 iters = 1000,
                 lr = 0.1,
                 batch_size = 5,
                 p_mask = 80,
                 num_projections = 10,
                 scheduler_step = 50,
                 scheduler_gamma = 0.75,
                 iters_proj = 300, 
                 num_projections_proj = 50,
                 bounded = True,
                 dequantize = True,
                 seed = None):
        """_summary_

        Args:
            metadata (dict): Attribute metadata describing the data domain.
            epsilon (float): Privacy budget.
            delta (float): DP delta. Defaults to 1.0.
            nbins (int): Bins per continuous attribute. Defaults to 32.
            degree (int): Marginal degree. Defaults to 2 (= all pairwise marginals).
            n_particles (int): Particles. Defaults to None (training set size).
            iters (int): Particle gradient descent iterations. Defaults to 1000.
            lr (float): Tuning parameter. Defaults to 0.1.
            batch_size (int): Tuning parameter. Defaults to 5.
            p_mask (int): Tuning parameter. Defaults to 80.
            num_projections (int): Random projections in the projection step. Defaults to 10.
            scheduler_step (int): Tuning parameter. Defaults to 50.
            scheduler_gamma (float): Tuning parameter. Defaults to 0.75.
            iters_proj (int): Iterations of the marginal projection step. Defaults to 300.
            num_projections_proj (int): Tuning parameter. Defaults to 50.
            bounded (bool): _description_. Defaults to True.
            dequantize (bool): _description_. Defaults to True.
            seed (int): Random seed. Defaults to None.
        """
        self.metadata = metadata
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.nbins = int(nbins)
        self.degree = int(degree)
        self.n_particles = n_particles
        self.bounded = bool(bounded)
        self.dequantize = bool(dequantize)
        self.seed = seed

        self.hp = {
            'iters': int(iters),
            'lr': float(lr),
            'batch_size': int(batch_size),
            'p_mask': int(p_mask),
            'num_projections': int(num_projections),
            'scheduler_step': scheduler_step,
            'scheduler_gamma': scheduler_gamma,
            'iters_proj': int(iters_proj),
            'num_projections_proj': int(num_projections_proj),
        }

        self.encoders = self._build_encoders(metadata)
        self.attrs = [a['name'] for a in metadata['columns'] if a['name'] in self.encoders]

        self.synth_df = None 
        self.synt_weights = None
        self.loss = None

        self.datatype = DataFrame
        self.multiprocess = False
        self.infer_ranges = False
        self.trained = False
 
        self.__name__ = f'PrivPGD({self.epsilon}, {self.delta})'

    def _build_encoders(self, metadata):
        encoders = {}
        for col in metadata['columns']:
            name, type = col['name'], col['type']
            if type in ('Categorical', 'Ordinal'):
                i2s = list(col['i2s'])
                encoders[name] = {
                    'kind': 'cat',
                    'i2s': i2s,
                    's2i': {v: i for i, v in enumerate(i2s)},
                    'size': len(i2s)
                }
            elif type in ('Integer', 'Float'):
                low, high = float(col['min']), float(col['max'])
                if high <= low:
                    high = low + 1e-9
                if type == 'Integer' and (high - low + 1) <= self.nbins:
                    encoders[name] = {
                        'kind': 'int_id',
                        'lo': int(round(low)),
                        'size': int(round(high-low)+1)
                    }
                else:
                    encoders[name] = {
                        'kind':'num',
                        'lo':low,
                        'hi':high,
                        'edges':np.linspace(low,high,self.nbins+1),
                        'size':self.nbins,
                        'is_int':type=='Integer'
                    }
        return encoders

    def _encode(self, data):
        out = {}
        for a in self.attrs:
            enc = self.encoders[a]
            col = data[a]
            if enc['kind'] == 'cat':
                codes = col.map(enc['s2i'])
                if codes.isna().any():
                    unseen = sorted(set(col[codes.isna()].unique()))[:5]
                    raise ValueError(
                        f'PrivPGD: attribute {a} contains values absent from metadata["i2s"], e.g. {unseen}. '
                        f'Fix the metadata rather than inferring the domain from data (inferring would break the DP guarantee).')
                out[a] = codes.astype(np.int64).values
            elif enc['kind'] == 'int_id':
                out[a] = np.clip(np.round(col.values.astype(np.float64))
                                 - enc['lo'], 0, enc['size'] - 1
                                 ).astype(np.int64)
            else:
                pos = ((col.values.astype(np.float64) - enc['lo'])
                       / (enc['hi'] - enc['lo'])) * enc['size']
                out[a] = np.clip(np.floor(pos), 0,
                                 enc['size'] - 1).astype(np.int64)
        return DataFrame(out, columns=self.attrs)
 
    def _decode(self, codes, rng):
        out = {}
        for a in self.attrs:
            enc = self.encoders[a]
            c = np.asarray(codes[a], dtype=np.int64)
            if enc['kind'] == 'cat':
                i2s = enc['i2s']
                out[a] = np.array([i2s[i] for i in c], dtype=object)
            elif enc['kind'] == 'int_id':
                out[a] = (c + enc['lo']).astype(np.int64)
            else:
                lo_e, hi_e = enc['edges'][c], enc['edges'][c + 1]
                if enc['is_int']:
                    a_i = np.ceil(lo_e).astype(np.int64)
                    b_i = np.floor(np.nextafter(hi_e, lo_e)).astype(np.int64)
                    b_i = np.maximum(b_i, a_i)
                    out[a] = (rng.randint(a_i, b_i + 1) if self.dequantize
                              else np.round((lo_e + hi_e) / 2))
                    out[a] = np.asarray(out[a], dtype=np.int64)
                else:
                    out[a] = (lo_e + rng.random_sample(c.size)
                              * (hi_e - lo_e)) if self.dequantize \
                        else (lo_e + hi_e) / 2
        return DataFrame(out, columns=self.attrs)

    def fit(self, data):
        """Fit PrivPGD to the input dataset and cache the particle model.
 
        :param data: DataFrame: Training set
        """
        assert isinstance(data, self.datatype), \
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
 
        if self.trained:
            self.trained = False
            self.synth_df, self.synth_weights, self.loss = None, None, None
 
        pkg = _import_privpgd(PRIVPGD_SRC)
        rng = np.random.RandomState(self.seed)
        if self.seed is not None:
            np.random.seed(self.seed)
            try:
                import torch
                torch.manual_seed(self.seed)
            except ImportError:
                pass
 
        nrows = len(data)
 
        LOGGER.debug(f'Start fitting PrivPGD to data of shape {data.shape} (eps={self.epsilon}, delta={self.delta})')
 
        encoded = self._encode(data)
        domain = pkg['Domain'](self.attrs,
                               [self.encoders[a]['size'] for a in self.attrs],
                               d=len(self.attrs))
        dataset = pkg['Dataset'](encoded, domain)
 
        workload = pkg['generate_all_kway_workload'](data=dataset,
                                                     degree=self.degree)
        hp = dict(self.hp)
        hp['n_particles'] = int(self.n_particles or nrows)
 
        engine = pkg['AdvancedSlicedInference'](domain=domain, hp=hp)
        mechanism = pkg['KWay'](epsilon=self.epsilon, delta=self.delta,
                                degree=self.degree, bounded=self.bounded)
 
        synth, self.loss = mechanism.run(data=dataset, workload=workload,
                                         engine=engine)
        
        self.synth_df = self._decode(synth.df, rng)
        w = np.asarray(synth.weights, dtype=np.float64)
        w = np.clip(w, 0, None)
        self.synth_weights = w / w.sum()
 
        LOGGER.debug(f'Finished fitting PrivPGD (loss={self.loss})')
        self.trained = True

    def generate_samples(self, nsamples):
        """Sample a synthetic dataset of size nsamples from the fitted model."""
        assert self.trained, 'Model must be fitted to some data first'
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(self.synth_df), 
                         size=int(nsamples), 
                         replace=True,
                         p=self.synth_weights)
        return self.synth_df.iloc[idx].reset_index(drop=True)
 