from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np
import logging
import tensorflow as tf
from tqdm import trange

from utils.params import TrainParams
from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import disable_rdkit_log

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer


PARAMS = TrainParams(
batch_dim = 100,
learning_rate = 1e-3,
la = 1,
dropout = 0,
n_critic = 5,
metric = 'validity,sas',
n_samples = 5000,
z_dim = 32,
epochs = 100,
save_every = 5,
model_base_dir = Path('GraphGAN/norl'),
train_data = 'data/qm9-mysplits-data.pkl'
)

if not PARAMS.checkpoint_dir.exists():
    PARAMS.checkpoint_dir.mkdir(parents=True)

data = SparseMolecularDataset()
data.load(PARAMS.train_data)

tf.logging.set_verbosity(logging.DEBUG)
logger = logging.getLogger('molgan')
logger.setLevel(logging.DEBUG)


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


# model
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      PARAMS.z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((64, 32), 128, (128,)),  # [0] = graph-conv, [1] = graph aggregation, [2] = fully-connected
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)

# session
tf.train.create_global_step()
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    # trainer
    trainer = Trainer(model, None,
                      save_every=1,
                      directory=str(PARAMS.checkpoint_dir))
    trainer.session = session

    session.run(init_op)
    session.graph.finalize()

    trainer.load()

    outputs = []
    fetch_dict = test_fetch_dict(model)
    with disable_rdkit_log():
        for _ in trange(PARAMS.epochs):
            feed_dict = test_feed_dict(model, PARAMS.batch_dim)
            output = session.run(fetch_dict, feed_dict=feed_dict)
            n, e = np.argmax(output['nodes'], axis=-1), np.argmax(output['edges'], axis=-1)
            for n_, e_ in zip(n, e):
                outputs.append(data.matrices2mol(n_, e_, strict=True))

    with open('predictions.pkl', 'wb') as fout:
        pickle.dump(outputs, fout)
