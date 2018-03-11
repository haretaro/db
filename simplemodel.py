import chainer
import chainer.links as L

class SimpleModel(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_out):
        super(SimpleModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.lstm = L.LSTM(n_units, n_units)
            self.out = L.Linear(n_units, n_out)
