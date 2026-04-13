#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                                \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return min(optimal_block_num, max_block_num);
}


/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline __device__ constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}


template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void iflut_transform_4d_cuda_forward_kernel(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ data_col) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);
    const int stride_lut_2 = stride_lut * stride_lut;

    CUDA_1D_KERNEL_LOOP(index, n) {

        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index +  height * width];
        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id00 = (rid    ) + stride_lut * (gid    );
        const int id10 = (rid + 1) + stride_lut * (gid    );
        const int id01 = (rid    ) + stride_lut * (gid + 1);
        const int id11 = (rid + 1) + stride_lut * (gid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        
        const scalar_t w00 = (1 - rd) * (1 - gd);
        const scalar_t w10 = (    rd) * (1 - gd);
        const scalar_t w01 = (1 - rd) * (    gd);
        const scalar_t w11 = (rd) * (gd);

        /* Execute the interpolation */
        // printf("num_channels: %d\n", num_channels);
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + height * width * i] =
                w00 * data_lut[id00 + stride_lut_2 * i] + w10 * data_lut[id10 + stride_lut_2 * i] +
                w01 * data_lut[id01 + stride_lut_2 * i] + w11 * data_lut[id11 + stride_lut_2 * i];
        }
    }
}



template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void iflut_transform_4d_cuda_backward_kernel(
        const int n,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ grad_inp,
        scalar_t* __restrict__ grad_lut) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);
    const int stride_lut_2 = stride_lut * stride_lut;

    CUDA_1D_KERNEL_LOOP(index, n) {

        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index +  height * width];
        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        /* utility varagbles for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id00 = (rid    ) + stride_lut * (gid    );
        const int id10 = (rid + 1) + stride_lut * (gid    );
        const int id01 = (rid    ) + stride_lut * (gid + 1);
        const int id11 = (rid + 1) + stride_lut * (gid + 1);
        
        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        
        const scalar_t w00 = (1 - rd) * (1 - gd);
        const scalar_t w10 = (    rd) * (1 - gd);
        const scalar_t w01 = (1 - rd) * (    gd);
        const scalar_t w11 = (rd) * (gd);


        /* derivatives: w to rd */
        const scalar_t w00_rd = - (1 - gd);
        const scalar_t w10_rd =   (1 - gd);
        const scalar_t w01_rd = - (    gd);
        const scalar_t w11_rd =   (    gd);

        /* derivatives: w to gd */
        const scalar_t w00_gd = - (1 - rd);
        const scalar_t w10_gd = - (rd);
        const scalar_t w01_gd =   (1 - rd);
        const scalar_t w11_gd =   (    rd);


        /* derivatives: w to bd */

        for (int i = 0; i < num_channels; ++i) {
            scalar_t grad_o_ = grad_output[index +  height * width * i];

            /* compute gradient of lut */
            atomicAdd(grad_lut + id00 + stride_lut_2 * i, grad_o_ * w00);
            atomicAdd(grad_lut + id10 + stride_lut_2 * i, grad_o_ * w10);
            atomicAdd(grad_lut + id01 + stride_lut_2 * i, grad_o_ * w01);
            atomicAdd(grad_lut + id11 + stride_lut_2 * i, grad_o_ * w11);
            

            /* compute gradient of vertices */
            scalar_t grad_d = 0;
            const scalar_t lut00 = data_lut[id00 + stride_lut_2 * i];
            const scalar_t lut10 = data_lut[id10 + stride_lut_2 * i];
            const scalar_t lut01 = data_lut[id01 + stride_lut_2 * i];
            const scalar_t lut11 = data_lut[id11 + stride_lut_2 * i];
            grad_d = grad_o_ *
                (w00_rd * lut00 + w10_rd * lut10 + w01_rd * lut01 + w11_rd * lut11 );
            // r
            atomicAdd(grad_inp + index, grad_d * 1 / size_bin);

            grad_d = grad_o_ *
                (w00_gd * lut00 + w10_gd * lut10 + w01_gd * lut01 + w11_gd * lut11);
            // g
            atomicAdd(grad_inp + index + height * width, grad_d * 1 / size_bin);

        }
    }
}


void IFLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output) {


    c10::cuda::CUDAGuard device_guard(input.device());

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    // printf("num_channels: %d\n", batch_size);
    int height     = input.size(2);
    int width      = input.size(3);
    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);
    // printf("num_channels: %d\n", num_channels);
    int num_kernels =  height * width;
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "iflut_transform_cuda_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                iflut_transform_4d_cuda_forward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_inp, data_lut,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}



void IFLUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut) {

    c10::cuda::CUDAGuard device_guard(grad_output.device());

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    // printf(num_channels);
    int stride_lut   = lut.size(2);

    int num_kernels =  height * width;
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "iflut_transform_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();

                iflut_transform_4d_cuda_backward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_inp, data_lut, 
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}