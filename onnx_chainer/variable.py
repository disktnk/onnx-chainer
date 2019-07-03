import chainer
import numpy as np

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


def convert_Shape(func, opset_version, input_names, output_names, context, parameters):
    return onnx_helper.make_node(
        'Shape', input_names, output_names)


class ShapeVariable(chainer.Variable):

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

    def __getitem__(self, i):
        v = self.var[i]
        print('c', v, len(v.shape))
        if len(v.shape) > 0:
            return ShapeVariable(v)
        print('ggggetitem')
        return ShapeItemVariable(self.var[i])

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
        return self.var.array * other


class ShapeItemVariable(chainer.Variable):

    def __init__(self, var):
        print('IIIIII', var)
        self.var = var
        super(ShapeItemVariable, self).__init__(var.array)

    def __int__(self):
        return int(self.var.array)

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
        return self.var.array * other
