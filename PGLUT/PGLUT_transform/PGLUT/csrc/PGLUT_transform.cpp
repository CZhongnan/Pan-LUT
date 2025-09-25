#include <torch/extension.h>

/* CUDA Forward Declarations */

void PGLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void PGLUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void PGLUT_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    PGLUTTransformForwardCUDAKernelLauncher(input, lut, output);
}


void PGLUT_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    PGLUTTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


// void PGLUT_transform_cpu_forward(
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor output);


// void PGLUT_transform_cpu_backward(
//     const torch::Tensor &grad_output,
//     const torch::Tensor &input,
//     const torch::Tensor &lut,
//     torch::Tensor grad_inp,
//     torch::Tensor grad_lut);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void PGLUT_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    CHECK_INPUT(input);
    CHECK_INPUT(lut);
    CHECK_INPUT(output);

    PGLUT_transform_cuda_forward(input, lut, output);
}


void PGLUT_transform_backward(
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

    PGLUT_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("PGLUT_cforward", &PGLUT_transform_forward, "PGLUT-Transform forward");
  m.def("PGLUT_cbackward", &PGLUT_transform_backward, "PGLUT-Transform backward");
}

