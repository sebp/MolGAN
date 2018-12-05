from collections import namedtuple
from pathlib import Path
import numpy as np
import logging
import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import MolecularMetrics, samples, all_scores
from utils.utils import disable_rdkit_log

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer

class TrainParams(namedtuple('TrainParams',
    ('batch_dim', 'learning_rate', 'la', 'dropout', 'n_critic', 'metric',
     'n_samples', 'z_dim', 'epochs', 'save_every', 'model_base_dir', 'train_data'))):

    @property
    def checkpoint_dir(self):
        return self.model_base_dir / 'lam{}'.format(self.la)

    @property
    def log_file(self):
        return self.checkpoint_dir / 'molgan.log'


PARAMS = TrainParams(
batch_dim = 128,
learning_rate = 1e-3,
la = 1,
dropout = 0,
n_critic = 5,
metric = 'validity,sas',
n_samples = 5000,
z_dim = 32,
epochs = 300,
save_every = 5,
model_base_dir = Path('GraphGAN'),
train_data = 'data/qm9-mysplits-data.pkl'
)

if not PARAMS.checkpoint_dir.exists():
    PARAMS.checkpoint_dir.mkdir(parents=True)

data = SparseMolecularDataset()
data.load(PARAMS.train_data)

steps = (len(data) // PARAMS.batch_dim)

logger = logging.getLogger('molgan')
fh = logging.FileHandler(PARAMS.log_file)
fh.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    if i % PARAMS.n_critic == 0:
        train_step = [optimizer.train_step_G]
        if False:
            train_step.append(optimizer.train_step_V)
            summary_op = optimizer.summary_op_RL
        else:
            summary_op = optimizer.summary_op_G

        train_op = [optimizer.step_G, summary_op] + train_step
    else:
        train_op = [optimizer.step_D, optimizer.summary_op_D, optimizer.train_step_D]

    return train_op


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    global session
    mols, _, _, a, x, _, _, _, _ = data.next_train_batch(PARAMS.batch_dim)
    embeddings = model.sample_z(PARAMS.batch_dim)

    if PARAMS.la < 1:

        if i % PARAMS.n_critic == 0:
            # rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            with disable_rdkit_log():
                mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            # rewardF = reward(mols)
            # logger.debug("step %d: average reward on real=%.4f, on generated=%.4f",
            #     i, np.mean(rewardR), np.mean(rewardF))

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
            #             model.rewardR: rewardR,
            #             model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: PARAMS.dropout,
                         optimizer.la: PARAMS.la if epoch > 0 else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: PARAMS.dropout,
                         optimizer.la: PARAMS.la if epoch > 0 else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: PARAMS.dropout,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            #'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    global session
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    #rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    with disable_rdkit_log():
        mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    #rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 #model.rewardR: rewardR,
                 #model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            #'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    global session
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    #rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    with disable_rdkit_log():
        mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    #rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 #model.rewardR: rewardR,
                 #model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def reward(mols):
    rr = 1.
    with disable_rdkit_log():
        for m in ('logp,sas,qed,unique' if PARAMS.metric == 'all' else PARAMS.metric).split(','):
            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    global session
    mols = samples(data, model, session, model.sample_z(PARAMS.n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    global session
    mols = samples(data, model, session, model.sample_z(PARAMS.n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


# model
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      PARAMS.z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((64, 32), 128, (128,)),  # [0] = graph-conv, [1] = graph aggregation, [2] = fully-connected
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=True,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)

# optimizer
optimizer = GraphGANOptimizer(model, learning_rate=PARAMS.learning_rate, feature_matching=False)

# session
tf.train.create_global_step()
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    # trainer
    trainer = Trainer(model, optimizer,
                      save_every=PARAMS.save_every,
                      directory=str(PARAMS.checkpoint_dir))

    logger.info('Parameters: %r', np.sum([np.prod(e.shape.as_list()) for e in tf.trainable_variables()]))

    session.run(init_op)
    session.graph.finalize()
    trainer.train(session=session,
                  batch_dim=PARAMS.batch_dim,
                  epochs=PARAMS.epochs,
                  steps=steps,
                  train_fetch_dict=train_fetch_dict,
                  train_feed_dict=train_feed_dict,
                  eval_fetch_dict=eval_fetch_dict,
                  eval_feed_dict=eval_feed_dict,
                  test_fetch_dict=test_fetch_dict,
                  test_feed_dict=test_feed_dict,
                  _eval_update=_eval_update,
                  _test_update=_test_update)
