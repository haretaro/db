import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import loader
import numpy as np

UNK = 0
EOS = 1

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs

class SimpleModel(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_out):
        super(SimpleModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.encoder = L.NStepLSTM(1, n_units, n_units, dropout=1)
            self.out = L.Linear(n_units, n_out)

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, xs, t, hx=None, cs=None):
        exs = sequence_embed(self.embed, xs)
        hx, cs, os = self.encoder(hx, cs, exs)
        y = self.out(hx[0])
        loss = F.mean_squared_error(y, t)
        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)

    corpora = loader.get_corpora()
    word2index = {'<EOS>':EOS, '<UNK>':UNK}
    for corpus in corpora:
        for utterance, label in zip(*corpus):
            for word in utterance.split(' '):
                if word not in word2index.keys():
                    word2index[word] = len(word2index)
    index2word = {k: word2index[k] for k in word2index}
    n_vocab = len(word2index)

    xs, ys = loader.preprocess(corpora, word2index)
    test_xs, test_ys = loader.preprocess(loader.get_test_corpora(), word2index)

    model = SimpleModel(n_vocab, 500, 3)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    batchsize = 50
    n_epoch = 20
    idx = 0
    epoch = 0

    def evaluate(model, xs, ts):
        evaluator = model.copy()
        loss = model(xs, ys)
        return loss.data

    for i in range(len(xs) * n_epoch // batchsize):
        prev_idx = idx
        idx = (i * batchsize) % len(xs)
        loss = model(xs[idx: idx+batchsize], ys[idx: idx+batchsize])
        if idx < prev_idx:
            epoch += 1
            test_error = evaluate(model, test_xs, test_ys)
            print('epoch %d: %f, %f' % (epoch, loss.data, test_error))
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

if __name__ == '__main__':
    main()
