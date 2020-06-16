import argparse
import logging
from pathlib import Path
import pickle
from rdkit import Chem
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.params import TrainParams
from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import disable_rdkit_log

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj


def test_fetch_dict(model):
    return {'edges': model.edges_argmax,
            'nodes': model.nodes_argmax}


def test_feed_dict(model, batch_dim):
    embeddings = model.sample_z(batch_dim)
    feed_dict = {
         model.embeddings: embeddings,
         model.training: False
    }
    return feed_dict


def is_valid_smiles(s):
    is_valid = s != '' and '*' not in s and '.' not in s
    return is_valid


class MolganSampleGenerator:

    def __init__(self, params):
        self._data = SparseMolecularDataset()
        self._params = params
        self._model = None

    def _create_model(self):
        self._data.load(self._params.train_data)

        # model
        model = GraphGANModel(self._data.vertexes,
                              self._data.bond_num_types,
                              self._data.atom_num_types,
                              self._params.z_dim,
                              decoder_units=(128, 256, 512),
                              discriminator_units=((64, 32), 128, (128,)),  # [0] = graph-conv, [1] = graph aggregation, [2] = fully-connected
                              decoder=decoder_adj,
                              discriminator=encoder_rgcn,
                              soft_gumbel_softmax=False,
                              hard_gumbel_softmax=False,
                              batch_discriminator=False)
        return model

    def generate(self, number_samples):
        samples = tqdm(self.iter_generate(number_samples),
                       desc='Sampling from model',
                       total=number_samples)
        return list(samples)

    def iter_generate(self, number_samples):
        g = tf.Graph()
        with g.as_default():
            return map(self._to_smiles, self._iter_generate(number_samples, g))

    def _iter_generate(self, number_samples, graph):
        if self._model is None:
            self._model = self._create_model()

            tf.train.create_global_step()
            self._init_op = tf.global_variables_initializer()

            # trainer
            self._trainer = Trainer(self._model, None,
                              save_every=1,
                              directory=str(self._params.model_base_dir))
            graph.finalize()

        with tf.Session() as session:
            session.run(self._init_op)

            self._trainer.session = session
            self._trainer.load()

            fetch_dict = test_fetch_dict(self._model)

            n_epochs = number_samples // self._params.batch_dim

            for _ in range(n_epochs):
                feed_dict = test_feed_dict(self._model, self._params.batch_dim)
                output = session.run(fetch_dict, feed_dict=feed_dict)
                n = np.argmax(output['nodes'], axis=-1)
                e = np.argmax(output['edges'], axis=-1)
                for n_, e_ in zip(n, e):
                    yield n_, e_

            n_remain = number_samples - n_epochs * self._params.batch_dim
            if n_remain > 0:
                feed_dict = test_feed_dict(self._model, self._params.batch_dim)
                output = session.run(fetch_dict, feed_dict=feed_dict)
                n = np.argmax(output['nodes'], axis=-1)[:n_remain]
                e = np.argmax(output['edges'], axis=-1)[:n_remain]
                for n_, e_ in zip(n, e):
                    yield n_, e_

    def _to_smiles(self, graph):
        nodes, edges = graph
        mol = self._data.matrices2mol(nodes, edges, strict=True)

        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            if is_valid_smiles(smi):
                return smi

        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--number_samples', type=int, default=10000)

    args = parser.parse_args()

    PARAMS = TrainParams(
        batch_dim = 100,
        learning_rate = None,
        la = 1,
        dropout = 0,
        n_critic = None,
        metric = None,
        n_samples = None,
        z_dim = 32,
        epochs = None,
        save_every = None,
        model_base_dir = Path(args.model_dir),
        train_data = 'data/qm9-mysplits-data.pkl'
    )

    tf.logging.set_verbosity(logging.DEBUG)
    logger = logging.getLogger('guacamol')
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    generator = MolganSampleGenerator(PARAMS)

    with disable_rdkit_log(), open(args.output_file, "w") as fout:
        for smi in generator.iter_generate(args.number_samples):
            fout.write(smi)
            fout.write("\n")


if __name__ == '__main__':
    main()
