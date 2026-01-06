from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from ._ext import (
    IFLUT_cforward, IFLUT_cbackward
)


class IFLUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                img: torch.Tensor,
                lut: torch.Tensor) -> torch.Tensor:

        img = img.contiguous()
        lut = lut.contiguous()

        assert img.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [6], \
            "only support 5D lookup table without batch dimension (5D tensor)"

        output = img.new_zeros((img.size(0), lut.size(1), img.size(2), img.size(3)))
        IFLUT_cforward(img, lut, output)

        ctx.save_for_backward(img, lut)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.contiguous()

        img, lut = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)

        IFLUT_cbackward(grad_output, img, lut, grad_img, grad_lut)

        return grad_img, grad_lut


def IFLUT_transform(
    img: torch.Tensor,
    lut: torch.Tensor) -> torch.Tensor:
    r"""Spatil and Temporal 4D Lookup Table Transform (IFLUT-Transform).

    Args:
        img (torch.Tensor): input image of shape (b, 5, h, w).
        lut (torch.Tensor): output values of the 4D LUT, shape (b, 4, d, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 4, h, w).
    """
    return IFLUTTransformFunction.apply(img, lut)