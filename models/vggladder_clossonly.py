import chainer
import chainer.functions as F
import chainer.links as L
import pdb
from chainer import reporter
from chainer.functions.evaluation import accuracy
rloss_unit = 1.0 / 28 / 28 / 3

class VGGLadderClossOnly(chainer.Chain):

    def __init__(self, n_class=11, rloss_weight = 1.0):
        super(VGGLadderClossOnly, self).__init__()
        self.rloss_weight = rloss_weight
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024)
            self.fc5 = L.Linear(1024, 1024)
            self.fc6 = L.Linear(1024, n_class)

            self.deconv3 = L.Deconvolution2D(256, 128, 2, pad=0, stride=2)
            self.deconv2 = L.Deconvolution2D(128, 64, 2, pad=0, stride=2)
            self.deconv1 = L.Deconvolution2D(64, 3, 2, pad=0, stride=2)

    def predict(self, x):
        h1 = F.resize_images(x, [32, 32])
        h1 = F.relu(self.bn1_1(self.conv1_1(h1)))
        h1 = F.relu(self.bn1_2(self.conv1_2(h1)))
        h1 = F.max_pooling_2d(h1, 2, 2)
        h1 = F.dropout(h1, ratio=0.25)

        h2 = F.relu(self.bn2_1(self.conv2_1(h1)))
        h2 = F.relu(self.bn2_2(self.conv2_2(h2)))
        h2 = F.max_pooling_2d(h2, 2, 2)
        h2 = F.dropout(h2, ratio=0.25)

        h3 = F.relu(self.bn3_1(self.conv3_1(h2)))
        h3 = F.relu(self.bn3_2(self.conv3_2(h3)))
        h3 = F.relu(self.bn3_3(self.conv3_3(h3)))
        h3 = F.relu(self.bn3_4(self.conv3_4(h3)))
        h3 = F.max_pooling_2d(h3, 2, 2)
        h3 = F.dropout(h3, ratio=0.25)

        h4 = F.dropout(F.relu(self.fc4(h3)), ratio=0.5)
        h4 = F.dropout(F.relu(self.fc5(h4)), ratio=0.5)
        h4 = self.fc6(h4)

        g2 = F.relu(self.deconv3(h3))
        g1 = F.relu(self.deconv2(F.add(h2 + g2)))
        g0 = F.relu(self.deconv1(F.add(h1 + g1)))

        return h4, g0

    def forward(self, x, y):
        res = self.predict(x)
        c = res[0]
        recon = res[1]
        #pdb.set_trace()
        xr = F.resize_images(x, [32, 32])
        closs = F.softmax_cross_entropy(c, y)
        rloss = F.mean_squared_error(recon, xr)

        reporter.report(
        {
            'closs': closs,
            'rloss': rloss,
            'accuracy': accuracy.accuracy(c, y),
        }, self)

        return closs
