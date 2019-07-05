import chainer
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


def convert_Shape(func, opset_version, input_names, output_names, context, parameters):
    return onnx_helper.make_node(
        'Shape', input_names, output_names)


class BatchMarkedVariable(chaienr.Variable):

    def __init__(self, array, batch_idx, name=None):
        self.batch_idx = batch_idx
        super(BatchMarkedVariable, self).__init__(arra, name)


class ShapeVariable(chainer.Parameter):

    def __init__(self, var):
        print('5555', var)
        self.var = var
        super(ShapeVariable, self).__init__(var.array)

    @property
    def array(self):
        return self.var.array

    @property
    def shape(self):
        return self.var.array.shape

    @as_funcnode('GetItem')
    def __getitem__(self, i):
        v = self.var[i]
        print('c', v, len(v.shape))
        if len(v.shape) > 0:
            return v.array
        print('ggggetitem')
        # return ShapeItemVariable(self.var[i])
        return self.var.array[i]

    def __eq__(self, other):
        return False
        # return self.var.array == other

    def __lt__(self, other):
        return self.var.array < other

    def __gt__(self, other):
        return self.var.array > other

    def __mod__(self, other):
        return self.var.array % other

    def __truediv__(self, other):
        return self.var.array / other

    def __mul__(self, other):
        return self.var.array * other

    def __imul__(self, other):
        return self.var.array * other


class ShapeItemVariable(chainer.Parameter):

    def __init__(self, var):
        print('IIIIII', var)
        self.var = var
        self.i = int(var.array)
        super(ShapeItemVariable, self).__init__(var.array)

    def __int__(self):
        # return int(self.var.array)
        return self.i

    def __eq__(self, other):
        return self.var.array == other

    def __lt__(self, other):
        return self.var.array < other

    def __gt__(self, other):
        return self.var.array > other

    def __mod__(self, other):
        return self.var.array % other

    def __truediv__(self, other):
        return self.var.array / other

    def __mul__(self, other):
        return self.var.array * other

    def __imul__(self, other):
        print('imul')
        return self.var.array * other
