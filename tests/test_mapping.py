import chainer
import chainer.functions as F

from onnx_chainer import export
from onnx_chainer.testing import input_generator


# This list is used for documentation, only check a function can be exported
# or not. A key of mapping is category, and values are consisted from
# (function_name, input_shapes, kwargs, description).
_function_mapping = {
    'activation': [
        ('clipped_relu', (2, 2)),
        ('relu6', (2, 2)),
        # ('crelu', (2, 2)),
        ('elu', (2, 2)),
        ('hard_sigmoid', (2, 2)),
        ('leaky_relu', (2, 2)),
        ('log_softmax', (2, 2)),
        ('maxout', (2, 2), {'pool_size': 2}),
        ('prelu', [(2, 2), (2,)]),
        ('relu', (2, 2)),
        # ('rrelu', (2, 2)),
        ('selu', (2, 2)),
        ('sigmoid', (2, 2)),
        ('softmax', (2, 2)),
        ('softplus', (2, 2)),
        # ('swish', [(2, 2), (2,)],),
        ('tanh', (2, 2)),
    ],
    'array': [
        ('as_strided', (4,), {'shape': (3, 2), 'strides': (1, 1), 'storage_offset': 0}),
        ('broadcast',),
        ('broadcast_to',),
        ('cast',),
        ('concat',),
        ('copy',),
        ('depth2space',),
        ('diagonal',),
        ('dstack',),
        ('expand_dims',),
        ('flatten',),
        ('flip',),
        ('fliplr',),
        ('flipud',),
        ('get_item',),
        ('hstack',),
        ('im2col',),
        ('moveaxis',),
        ('pad',),
        ('pad_sequence',),
        ('permutate',),
        ('repeat',),
        ('reshape',),
        ('resize_images',),
        ('rollaxis',),
        ('scatter_add',),
        ('select_item',),
        ('separate',),
        ('space2depth',),
        ('spatial_transformer_grid',),
        ('spatial_transformer_sampler',),
        ('split_axis',),
        ('squeeze',),
        ('stack',),
        ('swapaxes',),
        ('tile',),
        ('transpose',),
        ('transpose_sequence',),
        ('vstack',),
        ('where',),
    ],
    'connection': [
        ('bilinear',),
        ('convolution_2d',),
        ('convolution_1d',),
        ('convolution_3d',),
        ('convolution_nd',),
        ('deconvolution_2d',),
        ('deconvolution_1d',),
        ('deconvolution_3d',),
        ('deconvolution_nd',),
        ('deformable_convolution_2d_sampler',),
        ('depthwise_convolution_2d',),
        ('dilated_convolution_2d',),
        ('embed_id',),
        ('linear',),
        ('local_convolution_2d',),
        ('shift',),
    ],
    'loss': [
        ('softmax_cross_entropy',),
    ],
    'math': [
        ('arctanh',),
        ('average',),
        ('absolute',),
        ('add',),
        ('batch_l2_norm_squared',),
        ('bias',),
        ('ceil',),
        ('clip',),
        ('cumprod',),
        ('cumsum',),
        ('batch_det',),
        ('det',),
        ('digamma',),
        ('einsum',),
        ('erf',),
        ('erfc',),
        ('erfcinv',),
        ('erfcx',),
        ('erfinv',),
        ('exp',),
        ('log',),
        ('log10',),
        ('log2',),
        ('expm1',),
        ('fft',),
        ('ifft',),
        ('fix',),
        ('floor',),
        ('fmod',),
        ('cosh',),
        ('sinh',),
        ('identity',),
        ('batch_inv',),
        ('inv',),
        ('lgamma',),
        ('linear_interpolate',),
        ('log_ndtr',),
        ('log1p',),
        ('logsumexp',),
        ('batch_matmul',),
        ('matmul',),
        ('maximum',),
        ('minimum',),
        ('argmax',),
        ('argmin',),
        ('max',),
        ('min',),
        ('ndtr',),
        ('ndtri',),
        ('polygamma',),
        ('prod',),
        ('scale',),
        ('sign',),
        ('sparse_matmul',),
        ('rsqrt',),
        ('sqrt',),
        ('square',),
        ('sum',),
        ('sum_to',),
        ('tensordot',),
        ('arccos',),
        ('arcsin',),
        ('arctan',),
        ('arctan2',),
        ('cos',),
        ('sin',),
        ('tan',),
        ('average',),
    ],
    'noise': [
        ('dropout',)
    ],
    'normalizaton': [
        ('batch_normalization',),
        ('fixed_batch_normalization',),
        ('batch_renormalization',),
        ('fixed_batch_renormalization',),
        ('decorrelated_batch_normalization',),
        ('fixed_decorrelated_batch_normalization',),
        ('group_normalization',),
        ('normalize',),
        ('layer_normalization',),
        ('local_response_normalization',),
    ],
    'pooling': [
        ('average_pooling_2d',),
        ('average_pooling_1d',),
        ('average_pooling_3d',),
        ('average_pooling_nd',),
        ('max_pooling_1d',),
        ('max_pooling_2d',),
        ('max_pooling_3d',),
        ('max_pooling_nd',),
        ('roi_average_align_2d',),
        ('roi_average_pooling_2d',),
        ('roi_max_align_2d',),
        ('roi_max_pooling_2d',),
        ('roi_pooling_2d',),
        ('spatial_pyramid_pooling_2d',),
        ('unpooling_2d',),
        ('unpooling_1d',),
        ('unpooling_3d',),
        ('unpooling_nd',),
        ('upsampling_2d',),
    ],
}


def test_supported_functions():
    class _Documentation(object):
        def __init__(self, func_name, input_shapes, description):
            self.func_name = func_name
            self.description = description

    for category, supported_func in _function_mapping.items():
        for info in supported_func:
            if len(info) < 2:
                continue
            kwargs = info[2] if len(info) > 2 else {}
            description = info[3] if len(info) > 3 else None
            doc = _Documentation(info[0], info[1], description)

            target_func = getattr(F, doc.func_name, None)
            assert target_func is not None, '{} is not found'.format(
                doc.func_name)
            model = chainer.Sequential(
                lambda *x: target_func(*x, **kwargs))
            if isinstance(info[1], list):
                xs = tuple(input_generator.increasing(*s) for s in info[1])
            else:
                xs = input_generator.increasing(*info[1])
            export(model, xs)
