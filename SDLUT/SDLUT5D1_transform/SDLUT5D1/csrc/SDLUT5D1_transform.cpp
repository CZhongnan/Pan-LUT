#include <torch/extension.h>

/* CUDA Forward Declarations */

void SDLUT5D1TransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void SDLUT5D1TransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void SDLUT5D1_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    SDLUT5D1TransformForwardCUDAKernelLauncher(input, lut, output);
}


void SDLUT5D1_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    SDLUT5D1TransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


// void SDLUT5D1_transform_cpu_forward(
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor output);


// void SDLUT5D1_transform_cpu_backward(
//     const torch::Tensor &grad_output,
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor grad_inp,
//     torch::Tensor grad_lut);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void SDLUT5D1_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    CHECK_INPUT(input);
    CHECK_INPUT(lut);
    CHECK_INPUT(output);

    SDLUT5D1_transform_cuda_forward(input, lut, output);
}


void SDLUT5D1_transform_backward(
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

    SDLUT5D1_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SDLUT5D1_cforward", &SDLUT5D1_transform_forward, "SDLUT5D1-Transform forward");
  m.def("SDLUT5D1_cbackward", &SDLUT5D1_transform_backward, "SDLUT5D1-Transform backward");
}

