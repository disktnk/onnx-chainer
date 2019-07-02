import chainer

from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode


def convert_Shape(func, opset_version, input_names, output_names, context, parameters):
    return onnx_helper.make_node(
        'Shape', input_names, output_names)


class VVariable(chainer.Variable):

    def __init__(self, data=None, **kwargs):
        super(VVariable, self).__init__(data, **kwargs)

    @classmethod
    @as_funcnode('Identity')
    def init(cls, var, **kwargs):
        return VVariable(var.data, **kwargs)

    @property
    @as_funcnode('Shape')
    def shape(self):
        return self.__class__(self.xp.asarray(self.array.shape))

    def __getitem__(self, i):
        return self.array[i]
