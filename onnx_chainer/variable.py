import chainer
import chainer.functions as F
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


@property
@as_funcnode('Shape')
def shape(self):
    return self.xp.asarray(self.array.shape, dtype=np.float32)


def __mod__(self, other):
    return self.array % other


def __rmod__(self, other):
    return other % self.array


def __rmul__(self, other):
    return other * self.array


def __int__(self):
    assert len(self.array.shape) == 0
    return int(self.array)


def __eq__(self, other):
    print(other)
    print(self.array)
    return self.array == other


def __ne__(self, other):
    return self.array != other


def __gt__(self, other):
    return self.array > other


def __lt__(self, other):
    return self.array < other


org_reshape = chainer.functions.reshape


@as_funcnode('Reshape')
def dynamic_reshape(x, shape):
    return x.array.reshape(shape.array.astype(np.int64))


def reshape(x, shape):
    if any([isinstance(s, chainer.Variable) for s in shape]):
        shape_list = []
        for s in shape:
            if isinstance(s, chainer.Variable):
                shape_list.append(s)
            else:
                ss = chainer.Variable(np.array(s, dtype=np.float32))
                global_xx.append(ss)
                shape_list.append(ss)
        shape = F.stack(shape_list)
        return dynamic_reshape(x, shape)
    return org_reshape(x, shape)


global_xx = []


chainer.Variable.shape = shape
chainer.Variable.__mod__ = __mod__
chainer.Variable.__rmod__ = __rmod__
chainer.Variable.__rmul__ = __rmul__
chainer.Variable.__int__ = __int__
chainer.Variable.__eq__ = __eq__
chainer.Variable.__ne__ = __ne__
chainer.Variable.__lt__ = __lt__
chainer.Variable.__gt__ = __gt__
chainer.functions.reshape = reshape
