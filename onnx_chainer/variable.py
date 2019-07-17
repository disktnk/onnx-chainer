import chainer
import chainer.functions as F
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


@property
@as_funcnode('Shape')
def shape(self):
    return self.xp.asarray(self.array.shape, dtype=np.int64)


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
                ss = chainer.Variable(np.array(s))
                to_keep_global_variables.append(ss)
                shape_list.append(ss)
        shape = F.stack(shape_list)
        return dynamic_reshape(x, shape)
    return org_reshape(x, shape)


@as_funcnode('GetItem', [(1, 'slices')])
def get_item(x, slices):
    from chainer import utils
    return utils.force_array(x.array[slices]),


to_keep_global_variables = []


chainer.Variable.shape = shape
chainer.functions.reshape = reshape


class ShapeVariable(chainer.Variable):

    @staticmethod
    def create(var):
        target = ShapeVariable()
        target.__dict__ = var.__dict__
        target._node = var._node
        return target

    def __getitem__(self, slices):
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = slices,
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = slices,
        return get_item(self, slices)


class ShapeItemVariable(chainer.Variable):

    @staticmethod
    def create(var):
        if len(var.array.shape) != 0:
            raise ValueError('item variable must be scalar: {}'.format(var))
        target = ShapeItemVariable()
        target.__dict__ = var.__dict__
        target._node = var._node
        return target

    def __int__(self):
        return int(self.array)

    def __mod__(self, other):
        return self.array % other

    def __truediv__(self, other):
        return self.array / other

    def __rmul__(self, other):
        return other * self.array
