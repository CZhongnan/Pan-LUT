#include <torch/extension.h>

/* CUDA Forward Declarations */

void IF4DLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void IF4DLUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void IF4DLUT_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    IF4DLUTTransformForwardCUDAKernelLauncher(input, lut, output);
}


void IF4DLUT_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    IF4DLUTTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


// void IF4DLUT_transform_cpu_forward(
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor output);


// void IF4DLUT_transform_cpu_backward(
//     const torch::Tensor &grad_output,
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor grad_inp,
//     torch::Tensor grad_lut);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void IF4DLUT_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    CHECK_INPUT(input);
    CHECK_INPUT(lut);
    CHECK_INPUT(output);

    IF4DLUT_transform_cuda_forward(input, lut, output);
}


void IF4DLUT_transform_backward(
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

    IF4DLUT_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("IF4DLUT_cforward", &IF4DLUT_transform_forward, "IF4DLUT-Transform forward");
  m.def("IF4DLUT_cbackward", &IF4DLUT_transform_backward, "IF4DLUT-Transform backward");
}

