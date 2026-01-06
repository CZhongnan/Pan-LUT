#include <torch/extension.h>

/* CUDA Forward Declarations */

void SD90LUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void SD90LUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void SD90LUT_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    SD90LUTTransformForwardCUDAKernelLauncher(input, lut, output);
}


void SD90LUT_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    SD90LUTTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


// void SD90LUT_transform_cpu_forward(
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor output);


// void SD90LUT_transform_cpu_backward(
//     const torch::Tensor &grad_output,
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor grad_inp,
//     torch::Tensor grad_lut);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void SD90LUT_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    CHECK_INPUT(input);
    CHECK_INPUT(lut);
    CHECK_INPUT(output);

    SD90LUT_transform_cuda_forward(input, lut, output);
}


void SD90LUT_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(lut);
    CHECK_INPUT(grad_inp);
    CHECK_INPUT(grad_lut);

    SD90LUT_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SD90LUT_cforward", &SD90LUT_transform_forward, "SD90LUT-Transform forward");
  m.def("SD90LUT_cbackward", &SD90LUT_transform_backward, "SD90LUT-Transform backward");
}

