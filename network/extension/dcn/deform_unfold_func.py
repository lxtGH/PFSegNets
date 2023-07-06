import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from network import _C


class DeformUnfoldFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                deformable_groups=1,
                im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset)

        output = input.new_empty(
            DeformUnfoldFunction._output_size(input, ctx.kernel_size, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0)]  # columns

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            _C.deform_unfold_forward(
                input, 
                offset, 
                output, 
                ctx.bufs_[0],
                ctx.kernel_size[1], 
                ctx.kernel_size[0], 
                ctx.stride[1], 
                ctx.stride[0],
                ctx.padding[1], 
                ctx.padding[0], 
                ctx.dilation[1],
                ctx.dilation[0], 
                ctx.deformable_groups,
                cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors

        grad_input = grad_offset = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_unfold_backward_input(
                    input, 
                    offset, 
                    grad_output, 
                    grad_input,
                    grad_offset, 
                    ctx.bufs_[0], 
                    ctx.kernel_size[1],
                    ctx.kernel_size[0], 
                    ctx.stride[1], 
                    ctx.stride[0],
                    ctx.padding[1], 
                    ctx.padding[0], 
                    ctx.dilation[1],
                    ctx.dilation[0], 
                    ctx.deformable_groups,
                    cur_im2col_step)

        return (grad_input, grad_offset, None, None, None, None, None, None,
                None)

    @staticmethod
    def _output_size(input, kernel_size, padding, dilation, stride):
        channels = input.size(1)
        img_output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (kernel_size[d] - 1) + 1
            stride_ = stride[d]
            img_output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, img_output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, img_output_size))))
        output_size = (input.size(0), channels * kernel_size[0] * kernel_size[1], img_output_size[2] * img_output_size[3])
        return output_size


deform_unfold = DeformUnfoldFunction.apply
