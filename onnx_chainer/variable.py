import chainer
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


def convert_Shape(func, opset_version, input_names, output_names, context, parameters):
    return onnx_helper.make_node(
        'Shape', input_names, output_names)


@property
def shape(self):
    # return self.xp.asarray(super(BatchMarkedVariable, self).shape)
    return ShapeVariable(self.array.shape, 0)

_reshape = chainer.Variable.reshape

def reshape(*shape):
    self = shape[0]
    shape = list(shape[1:])
    print('reshape', shape)
    for i, s in enumerate(shape):
        if isinstance(s, ShapeItemVariable):
            print('danger!!')
            shape[i] = int(s)
    return _reshape(self, tuple(shape))

chainer.Variable.shape = shape
chainer.Variable.reshape = reshape


class ShapeVariable(object):

    def __init__(self, shape, batch_idx=None):
        print('shape_variable')
        self.shape = shape
        self.batch_idx = batch_idx

    def __getitem__(self, i):
        print('get_item', i, self.batch_idx)
        if self.batch_idx is not None and i == self.batch_idx:
            print('batched!!')
            return ShapeItemVariable(self.shape[i])
        return self.shape[i]

    def __iter__(self):
        print('iter')
        for s in self.shape:
            yield s

    def __eq__(self, other):
        return self.shape == other

    def __lt__(self, other):
        return self.shape < other

    def __gt__(self, other):
        return self.shape > other

    def __mod__(self, other):
        return self.shape % other

    def __truediv__(self, other):
        return self.shape / other

    def __mul__(self, other):
        return self.shape * other

    def __imul__(self, other):
        return self.shape * other

    def __len__(self):
        return len(self.shape)


class ShapeItemVariable(object):
    def __init__(self, val=0):
        self._val = int(val)
    def __add__(self, val):
        if isinstance(val, Integer):
            return Integer(self._val + val._val)
        return self._val + val
    def __iadd__(self, val):
        self._val += val
        return self
    def __mul__(self, val):
        print(';;;;;')
        return self._val * val
    def __imul__(self, val):
        print(';;;;;')
        self._val *= val
        return self._val
    def __str__(self):
        return str(self._val)
    def __repr__(self):
        return 'Integer(%s)' % self._val
    def __eq__(self, other):
        return self._val == other
    def __int__(self):
        return self._val

    def __lt__(self, other):
        return self._val < other

    def __gt__(self, other):
        return self._val > other