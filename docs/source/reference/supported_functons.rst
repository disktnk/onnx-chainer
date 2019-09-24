.. _supported_functions:

Supported Functions
===================

ONNX-Chainer converts each ``chainer.FunctionNode`` to ONNX operator, **not ``chainer.functions.*``**, means that output ONNX graph is not match with model definition. For example, ``chainer.functions.bias`` will be broken down to ``BroadcastTo`` by Chainer and converted to ONNX ``Expand`` operator by ONNX-Chainer. Another example, ``y = x[0, ...]`` is called ``chainer.functions.get_item`` internally then converted to ``Slice`` and ``Squeeze`` operator. Supported ``FunctionNode`` are listed on ``mapping.py``.

Function list
-------------


