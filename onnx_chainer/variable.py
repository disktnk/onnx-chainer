import chainer
import chainer.functions as F
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


_to_keep_global_variables = []
_input_shape = ()


def enable_shape_variable(input_shape):
    global _input_shape
    _input_shape = input_shape
    chainer.Variable.shape = _shape
    chainer.functions.reshape = _reshape


def _cast_shape_variable(outputs, in_args, in_kwargs):
    assert len(outputs) == 1
    return ShapeVariable.create(outputs[0])


def _cast_shape_item_variable(outputs, in_args, in_kwargs):
    assert len(outputs) == 1
    if isinstance(in_args[0], ShapeVariable) and len(in_args[1]) == 1:
        if isinstance(in_args[1][0], int):
            return ShapeItemVariable.create(outputs[0])
        if isinstance(in_args[1][0], slice):
            return ShapeVariable.create(outputs[0])
    return outputs[0]


@property
@as_funcnode('Shape', post_converter=_cast_shape_variable)
def _shape(self):
    return self.xp.asarray(self.array.shape, dtype=np.int64)


org_reshape = chainer.functions.reshape


@as_funcnode('Reshape')
def dynamic_reshape(x, shape):
    # to enable this function node, shape must be variable type.
    return x.array.reshape(shape.array.astype(np.int64))


def _reshape(x, shape):
    if any([isinstance(s, chainer.Variable) for s in shape]):
        shape_list = []
        for s in shape:
            if isinstance(s, chainer.Variable):
                shape_list.append(s)
            else:
                ss = chainer.Variable(np.array(s))
                _to_keep_global_variables.append(ss)
                shape_list.append(ss)
        shape = F.stack(shape_list)
        return dynamic_reshape(x, shape)
    return org_reshape(x, shape)


@as_funcnode(
    'GetItem', rename_attributes=[(1, 'slices')],
    post_converter=_cast_shape_item_variable)
def _get_item(x, slices):
    from chainer import utils
    return utils.force_array(x.array[slices]),


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
        return _get_item(self, slices)


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
