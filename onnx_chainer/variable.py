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

_reshape = chainer.functions.reshape

def reshape(x, shape):
    print('reshape', shape)
    if isinstance(shape, tuple):
        shape = list(shape)
    if isinstance(shape, list):
        for i, s in enumerate(shape):
            if isinstance(s, ShapeItemVariable):
                print('danger!!')
                shape[i] = int(s)
    return _reshape(x, shape)

chainer.Variable.shape = shape
chainer.functions.reshape = reshape


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

    # NOTE(disktnk): numbers.Integral is better, but select simple one

    def __init__(self, val=0):
        assert isinstance(val, int)
        self._val = int(val)

    def __int__(self):
        return self._val

    def __lt__(self, other):
        return self._val < other

    def __rmul__(self, other):
        return self._val * other
