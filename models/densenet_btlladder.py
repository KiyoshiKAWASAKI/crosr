import chainer
import chainer.functions as F
import chainer.links as L
import pdb
from chainer import reporter
from chainer.functions.evaluation import accuracy

class BNReLUConvDropConcat(chainer.Chain):

    def __init__(self, in_ch, out_ch, dropout_ratio):
        super(BNReLUConvDropConcat, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = self.conv(F.relu(self.bn(x)))
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        return F.concat((x, h))


class DenseBlock(chainer.ChainList):

    def __init__(self, in_ch, growth_rate, n_layer, dropout_ratio):
        super(DenseBlock, self).__init__()
        for i in range(n_layer):
            self.add_link(BNReLUConvDropConcat(
                in_ch + i * growth_rate, growth_rate, dropout_ratio))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class Transition(chainer.Chain):

    def __init__(self, in_ch, dropout_ratio):
        super(Transition, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, in_ch, 1, 1, 0, initialW=w)
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = F.relu(self.bn(x))
        if self.dropout_ratio > 0:
            h = F.dropout(self.conv(h), ratio=self.dropout_ratio)
        h = F.average_pooling_2d(h, 2)
        return h


class BNReLUAPoolFC(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        super(BNReLUAPoolFC, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.fc = L.Linear(in_ch, out_ch)

    def __call__(self, x):
        h = F.relu(self.bn(x))
        h = F.average_pooling_2d(h, h.shape[2:])
        return self.fc(h)


class DenseNetBtlLadder(chainer.ChainList):
    def __init__(
            self, n_layer=32, growth_rate=24, n_class=10, dropout_ratio=0,
            in_ch=16, n_block=3, lateral_dim=32, rloss_weight=1.0):
        super(DenseNetBtlLadder, self).__init__()
        self.rloss_weight = rloss_weight
        w = chainer.initializers.HeNormal()
        self.laterals = []
        self.postprocess = []
        _in_ch = in_ch
        with self.init_scope():
            self.add_link(L.Convolution2D(None, in_ch, 3, 1, 1, initialW=w))
            for i in range(n_block):
                _in_ch_prev = _in_ch
                _in_ch = in_ch + n_layer * growth_rate * i
                _out_ch = in_ch + n_layer * growth_rate * (i + 1)
                self.add_link(DenseBlock(
                    _in_ch, growth_rate, n_layer, dropout_ratio))

                #lateral conections
                btl1 = L.Convolution2D(_out_ch, lateral_dim, 3, pad=1)
                btl1.to_gpu(0)
                btl2 = L.Convolution2D(lateral_dim, _out_ch, 3, pad=1)
                btl2.to_gpu(0)
                if i == 0:
                    deconv = L.Convolution2D(_out_ch, _in_ch, 3, pad=1, stride=1)
                else:
                    deconv = L.Deconvolution2D(_out_ch, _in_ch, 2, pad=0, stride=2)
                deconv.to_gpu(0)
                self.laterals.append((btl1, btl2, deconv))

                if i < n_block - 1:
                    _in_ch = in_ch + n_layer * growth_rate * (i + 1)
                    trans = Transition(_in_ch, dropout_ratio)
                    self.add_link(trans)

            _in_ch = in_ch + n_layer * growth_rate * n_block
            self.add_link(BNReLUAPoolFC(_in_ch, n_class))
            #pdb.set_trace()
            self.postprocess.append(L.Convolution2D(in_ch, 3, 3, pad=1, stride=1))
            self.postprocess[0].to_gpu(0)

    def predict(self, x):
        xr = F.resize_images(x, [32, 32])
        xs = []
        for f in self:
            if isinstance(f, Transition) or isinstance(f, BNReLUAPoolFC):
                xs.append(x)
            x = f(x)

        g = None
        #pdb.set_trace()
        xs.reverse()
        for i, x2 in enumerate(xs):
            btl1, btl2, deconv = self.laterals[-(i + 1)]
            #pdb.set_trace()
            z = F.relu(btl1(x2))
            h = F.relu(btl2(z))
            if g is not None:
                g = F.relu(deconv(F.add(h + g)))
            else:
                g = F.relu(deconv(F.add(h)))
        g = self.postprocess[0](g)

        return x, g
    def predict_z(self, x):
        xr = F.resize_images(x, [32, 32])
        xs = []
        for f in self:
            if isinstance(f, Transition) or isinstance(f, BNReLUAPoolFC):
                xs.append(x)
            x = f(x)

        g = None
        #pdb.set_trace()
        xs.reverse()
        zs = []
        for i, x2 in enumerate(xs):
            btl1, btl2, deconv = self.laterals[-(i + 1)]
            #pdb.set_trace()
            z = F.relu(btl1(x2))
            h = F.relu(btl2(z))
            zs.append(z)
            if g is not None:
                g = F.relu(deconv(F.add(h + g)))
            else:
                g = F.relu(deconv(F.add(h)))
        g = self.postprocess[0](g)

        return x, g, zs

    def forward(self, x, y):
        res = self.predict(x)
        c = res[0]
        recon = res[1]
        #pdb.set_trace()
        #xr = F.resize_images(x, [32, 32])
        xr = x
        closs = F.softmax_cross_entropy(c, y)
        rloss = F.mean_squared_error(recon, xr)

        reporter.report(
        {
            'closs': closs,
            'rloss': rloss,
            'accuracy': accuracy.accuracy(c, y),
        }, self)

        return closs + rloss * self.rloss_weight


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = DenseNet(10)
    y = model(x)
    from chainer import computational_graph
    cg = computational_graph.build_computational_graph([y])
    with open('densenet.dot', 'w') as fp:
        fp.write(cg.dump())
