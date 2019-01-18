from collections import namedtuple


class TrainParams(namedtuple('TrainParams',
    ('batch_dim', 'learning_rate', 'la', 'dropout', 'n_critic', 'metric',
     'n_samples', 'z_dim', 'epochs', 'save_every', 'model_base_dir', 'train_data'))):

    @property
    def checkpoint_dir(self):
        return self.model_base_dir / 'lam{}'.format(self.la)

    @property
    def log_file(self):
        return self.checkpoint_dir / 'molgan.log'
