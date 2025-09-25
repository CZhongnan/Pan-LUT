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


/* binary search on a sorted array to find and clamp the lower bound */
template <typename scalar_t>
inline __device__ int32_t lower_bound(
        const scalar_t *data_ss,
        int32_t start,
        int32_t end,
        scalar_t val) {

    const int32_t ori_start = start;
    const int32_t upper_bound = end - start - 2;
    while (start < end) {
        int64_t mid = start + ((end - start) >> 1);
        if (!(data_ss[mid] >= val)) {
            start = mid + 1;
        }
        else {
            end = mid;
        }
    }
    return clamp(start - ori_start - 1, 0, upper_bound);
}


void sdlut5D1_transform_sanity_check(
    const torch::Tensor input, const torch::Tensor lut, torch::Tensor output) {

    TORCH_CHECK((input.ndimension() == 4),
                "4D input tensor (b, c, h, w) expected, but got: ",
                input.ndimension());
    TORCH_CHECK((input.size(1) == 5),
                "5-channel img expected, but got: ",
                input.size(1));
    TORCH_CHECK((lut.ndimension() == (input.size(1) + 2)),
                (input.size(1) + 2),
                "D lut tensor (b, m, d[, d, [d, [d]]]) expected, but got: ",
                lut.ndimension());
    TORCH_CHECK((input.size(0) == lut.size(0)),
                "input and lut should have identical batch size, but got: ",
                "input (", input.size(0), "), lut (", lut.size(0),")");
}


template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void sdlut5D1_transform_4d_cuda_forward_kernel(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ data_col) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);

    CUDA_1D_KERNEL_LOOP(index, n) {
        int x = index % (width);
        float y = (index / (width)) % (height);
        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index +  
            (x == 0 ? 0 : -1)];
        const scalar_t b = data_inp[index + 
            (x >= width - 1 ? 0 : 1)];
        const scalar_t c = data_inp[index + 
            (y < 1 ? 0 : -width)];
        const scalar_t p = data_inp[index + 
            (y >= height - 1 ? 0 : width)];
        
        // const scalar_t r = data_inp[index];
        // const scalar_t g = data_inp[index +  height * width];
        // const scalar_t b = data_inp[index +  height * width * 2];
        // const scalar_t c = data_inp[index +  height * width * 3];
        
        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid = clamp((int32_t)floor(c * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t pid = clamp((int32_t)floor(p * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility varsdbles for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id00000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id10000 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id01000 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id11000 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id00100 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id10100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id01100 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id11100 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id00010 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id10010 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id01010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id11010 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id00110 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id10110 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id01110 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id11110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id00001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id10001 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id01001 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id11001 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id00101 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id10101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id01101 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id11101 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id00011 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id10011 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id01011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id11011 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id00111 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id10111 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id01111 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id11111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;
        const scalar_t cd = (c - size_bin * cid) / size_bin;
        const scalar_t pd = (p - size_bin * pid) / size_bin;

        const scalar_t w00000 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10000 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w01000 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w00100 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w00010 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w00001 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w11000 = (    rd) * (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10100 = (    rd) * (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w10010 = (    rd) * (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w10001 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w01100 = (1 - rd) * (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w01010 = (1 - rd) * (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w01001 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w00110 = (1 - rd) * (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w00101 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w00011 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w11100 = (    rd) * (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w11010 = (    rd) * (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w11001 = (    rd) * (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w10110 = (    rd) * (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w10101 = (    rd) * (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w10011 = (    rd) * (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w01110 = (1 - rd) * (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w01101 = (1 - rd) * (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w01011 = (1 - rd) * (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w00111 = (1 - rd) * (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11110 = (    rd) * (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w11101 = (    rd) * (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w11011 = (    rd) * (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w10111 = (    rd) * (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w01111 = (1 - rd) * (    gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11111 = (    rd) * (    gd) * (    bd) * (    cd) * (    pd);

        /* Execute the interpolation */
        // printf("num_channels: %d\n", num_channels);
        for (int i = 0; i < num_channels; ++i) {
            data_col[index +  height * width * i] =
                w00000 * data_lut[id00000] + w10000 * data_lut[id10000] +
                w01000 * data_lut[id01000] + w11000 * data_lut[id11000] +
                w00100 * data_lut[id00100] + w10100 * data_lut[id10100] +
                w01100 * data_lut[id01100] + w11100 * data_lut[id11100] +
                w00010 * data_lut[id00010] + w10010 * data_lut[id10010] +
                w01010 * data_lut[id01010] + w11010 * data_lut[id11010] +
                w00110 * data_lut[id00110] + w10110 * data_lut[id10110] +
                w01110 * data_lut[id01110] + w11110 * data_lut[id11110] +
                w00001 * data_lut[id00001] + w10001 * data_lut[id10001] +
                w01001 * data_lut[id01001] + w11001 * data_lut[id11001] +
                w00101 * data_lut[id00101] + w10101 * data_lut[id10101] +
                w01101 * data_lut[id01101] + w11101 * data_lut[id11101] +
                w00011 * data_lut[id00011] + w10011 * data_lut[id10011] +
                w01011 * data_lut[id01011] + w11011 * data_lut[id11011] +
                w00111 * data_lut[id00111] + w10111 * data_lut[id10111] +
                w01111 * data_lut[id01111] + w11111 * data_lut[id11111];
        }
    }
}



template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void sdlut5D1_transform_4d_cuda_backward_kernel(
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

    CUDA_1D_KERNEL_LOOP(index, n) {

        /* retrieve rgbc value of the pixel */
        int x = index % (width);
        float y = (index / (width)) % (height);
        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index +  
            (x == 0 ? 0 : -1)];
        const scalar_t b = data_inp[index + 
            (x >= width - 1 ? 0 : 1)];
        const scalar_t c = data_inp[index + 
            (y < 1 ? 0 : -width)];
        const scalar_t p = data_inp[index + 
            (y >= height - 1 ? 0 : width)];
        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid = clamp((int32_t)floor(c * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t pid = clamp((int32_t)floor(p * (stride_lut - 1)), 0, stride_lut - 2);
        /* utility varaables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id00000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id10000 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id01000 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id11000 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id00100 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id10100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id01100 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id11100 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid    );
        const int id00010 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id10010 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id01010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id11010 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id00110 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id10110 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id01110 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id11110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid    );
        const int id00001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id10001 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id01001 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id11001 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id00101 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id10101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id01101 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id11101 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    ) + stride_lut_4 * (pid + 1);
        const int id00011 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id10011 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id01011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id11011 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id00111 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id10111 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id01111 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);
        const int id11111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1) + stride_lut_4 * (pid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;
        const scalar_t cd = (c - size_bin * cid) / size_bin;
        const scalar_t pd = (p - size_bin * pid) / size_bin;

        const scalar_t w00000 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10000 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w01000 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w00100 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w00010 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w00001 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w11000 = (    rd) * (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10100 = (    rd) * (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w10010 = (    rd) * (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w10001 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w01100 = (1 - rd) * (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w01010 = (1 - rd) * (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w01001 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w00110 = (1 - rd) * (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w00101 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w00011 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w11100 = (    rd) * (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w11010 = (    rd) * (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w11001 = (    rd) * (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w10110 = (    rd) * (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w10101 = (    rd) * (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w10011 = (    rd) * (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w01110 = (1 - rd) * (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w01101 = (1 - rd) * (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w01011 = (1 - rd) * (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w00111 = (1 - rd) * (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11110 = (    rd) * (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w11101 = (    rd) * (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w11011 = (    rd) * (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w10111 = (    rd) * (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w01111 = (1 - rd) * (    gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11111 = (    rd) * (    gd) * (    bd) * (    cd) * (    pd);

                /* derivatives: w to rd */
        const scalar_t w00000_rd = - (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10000_rd =   (1 - gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w01000_rd = - (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w11000_rd =   (    gd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w00100_rd = - (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w10100_rd =   (1 - gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w01100_rd = - (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w11100_rd =   (    gd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w00010_rd = - (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w10010_rd =   (1 - gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w01010_rd = - (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w11010_rd =   (    gd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w00110_rd = - (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w10110_rd =   (1 - gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w01110_rd = - (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w11110_rd =   (    gd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w00001_rd = - (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w10001_rd =   (1 - gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w01001_rd = - (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w11001_rd =   (    gd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w00101_rd = - (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w10101_rd =   (1 - gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w01101_rd = - (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w11101_rd =   (    gd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w00011_rd = - (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w10011_rd =   (1 - gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w01011_rd = - (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w11011_rd =   (    gd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w00111_rd = - (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w10111_rd =   (1 - gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w01111_rd = - (    gd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11111_rd =   (    gd) * (    bd) * (    cd) * (    pd);

        /* derivatives: w to gd */
        const scalar_t w00000_gd = - (1 - rd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w10000_gd = - (    rd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w01000_gd =   (1 - rd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w11000_gd =   (    rd) * (1 - bd) * (1 - cd) * (1 - pd);
        const scalar_t w00100_gd = - (1 - rd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w10100_gd = - (    rd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w01100_gd =   (1 - rd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w11100_gd =   (    rd) * (    bd) * (1 - cd) * (1 - pd);
        const scalar_t w00010_gd = - (1 - rd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w10010_gd = - (    rd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w01010_gd =   (1 - rd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w11010_gd =   (    rd) * (1 - bd) * (    cd) * (1 - pd);
        const scalar_t w00110_gd = - (1 - rd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w10110_gd = - (    rd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w01110_gd =   (1 - rd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w11110_gd =   (    rd) * (    bd) * (    cd) * (1 - pd);
        const scalar_t w00001_gd = - (1 - rd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w10001_gd = - (    rd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w01001_gd =   (1 - rd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w11001_gd =   (    rd) * (1 - bd) * (1 - cd) * (    pd);
        const scalar_t w00101_gd = - (1 - rd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w10101_gd = - (    rd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w01101_gd =   (1 - rd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w11101_gd =   (    rd) * (    bd) * (1 - cd) * (    pd);
        const scalar_t w00011_gd = - (1 - rd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w10011_gd = - (    rd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w01011_gd =   (1 - rd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w11011_gd =   (    rd) * (1 - bd) * (    cd) * (    pd);
        const scalar_t w00111_gd = - (1 - rd) * (    bd) * (    cd) * (    pd);
        const scalar_t w10111_gd = - (    rd) * (    bd) * (    cd) * (    pd);
        const scalar_t w01111_gd =   (1 - rd) * (    bd) * (    cd) * (    pd);
        const scalar_t w11111_gd =   (    rd) * (    bd) * (    cd) * (    pd);

        /* derivatives: w to bd */
        const scalar_t w00000_bd = - (1 - rd) * (1 - gd) * (1 - cd) * (1 - pd);
        const scalar_t w10000_bd = - (    rd) * (1 - gd) * (1 - cd) * (1 - pd);
        const scalar_t w01000_bd = - (1 - rd) * (    gd) * (1 - cd) * (1 - pd);
        const scalar_t w11000_bd = - (    rd) * (    gd) * (1 - cd) * (1 - pd);
        const scalar_t w00100_bd =   (1 - rd) * (1 - gd) * (1 - cd) * (1 - pd);
        const scalar_t w10100_bd =   (    rd) * (1 - gd) * (1 - cd) * (1 - pd);
        const scalar_t w01100_bd =   (1 - rd) * (    gd) * (1 - cd) * (1 - pd);
        const scalar_t w11100_bd =   (    rd) * (    gd) * (1 - cd) * (1 - pd);
        const scalar_t w00010_bd = - (1 - rd) * (1 - gd) * (    cd) * (1 - pd);
        const scalar_t w10010_bd = - (    rd) * (1 - gd) * (    cd) * (1 - pd);
        const scalar_t w01010_bd = - (1 - rd) * (    gd) * (    cd) * (1 - pd);
        const scalar_t w11010_bd = - (    rd) * (    gd) * (    cd) * (1 - pd);
        const scalar_t w00110_bd =   (1 - rd) * (1 - gd) * (    cd) * (1 - pd);
        const scalar_t w10110_bd =   (    rd) * (1 - gd) * (    cd) * (1 - pd);
        const scalar_t w01110_bd =   (1 - rd) * (    gd) * (    cd) * (1 - pd);
        const scalar_t w11110_bd =   (    rd) * (    gd) * (    cd) * (1 - pd);
        const scalar_t w00001_bd = - (1 - rd) * (1 - gd) * (1 - cd) * (    pd);
        const scalar_t w10001_bd = - (    rd) * (1 - gd) * (1 - cd) * (    pd);
        const scalar_t w01001_bd = - (1 - rd) * (    gd) * (1 - cd) * (    pd);
        const scalar_t w11001_bd = - (    rd) * (    gd) * (1 - cd) * (    pd);
        const scalar_t w00101_bd =   (1 - rd) * (1 - gd) * (1 - cd) * (    pd);
        const scalar_t w10101_bd =   (    rd) * (1 - gd) * (1 - cd) * (    pd);
        const scalar_t w01101_bd =   (1 - rd) * (    gd) * (1 - cd) * (    pd);
        const scalar_t w11101_bd =   (    rd) * (    gd) * (1 - cd) * (    pd);
        const scalar_t w00011_bd = - (1 - rd) * (1 - gd) * (    cd) * (    pd);
        const scalar_t w10011_bd = - (    rd) * (1 - gd) * (    cd) * (    pd);
        const scalar_t w01011_bd = - (1 - rd) * (    gd) * (    cd) * (    pd);
        const scalar_t w11011_bd = - (    rd) * (    gd) * (    cd) * (    pd);
        const scalar_t w00111_bd =   (1 - rd) * (1 - gd) * (    cd) * (    pd);
        const scalar_t w10111_bd =   (    rd) * (1 - gd) * (    cd) * (    pd);
        const scalar_t w01111_bd =   (1 - rd) * (    gd) * (    cd) * (    pd);
        const scalar_t w11111_bd =   (    rd) * (    gd) * (    cd) * (    pd);

        /* derivatives: w to cd */
        const scalar_t w00000_cd = - (1 - rd) * (1 - gd) * (1 - bd) * (1 - pd);
        const scalar_t w10000_cd = - (    rd) * (1 - gd) * (1 - bd) * (1 - pd);
        const scalar_t w01000_cd = - (1 - rd) * (    gd) * (1 - bd) * (1 - pd);
        const scalar_t w11000_cd = - (    rd) * (    gd) * (1 - bd) * (1 - pd);
        const scalar_t w00100_cd = - (1 - rd) * (1 - gd) * (    bd) * (1 - pd);
        const scalar_t w10100_cd = - (    rd) * (1 - gd) * (    bd) * (1 - pd);
        const scalar_t w01100_cd = - (1 - rd) * (    gd) * (    bd) * (1 - pd);
        const scalar_t w11100_cd = - (    rd) * (    gd) * (    bd) * (1 - pd);
        const scalar_t w00010_cd =   (1 - rd) * (1 - gd) * (1 - bd) * (1 - pd);
        const scalar_t w10010_cd =   (    rd) * (1 - gd) * (1 - bd) * (1 - pd);
        const scalar_t w01010_cd =   (1 - rd) * (    gd) * (1 - bd) * (1 - pd);
        const scalar_t w11010_cd =   (    rd) * (    gd) * (1 - bd) * (1 - pd);
        const scalar_t w00110_cd =   (1 - rd) * (1 - gd) * (    bd) * (1 - pd);
        const scalar_t w10110_cd =   (    rd) * (1 - gd) * (    bd) * (1 - pd);
        const scalar_t w01110_cd =   (1 - rd) * (    gd) * (    bd) * (1 - pd);
        const scalar_t w11110_cd =   (    rd) * (    gd) * (    bd) * (1 - pd);
        const scalar_t w00001_cd = - (1 - rd) * (1 - gd) * (1 - bd) * (    pd);
        const scalar_t w10001_cd = - (    rd) * (1 - gd) * (1 - bd) * (    pd);
        const scalar_t w01001_cd = - (1 - rd) * (    gd) * (1 - bd) * (    pd);
        const scalar_t w11001_cd = - (    rd) * (    gd) * (1 - bd) * (    pd);
        const scalar_t w00101_cd = - (1 - rd) * (1 - gd) * (    bd) * (    pd);
        const scalar_t w10101_cd = - (    rd) * (1 - gd) * (    bd) * (    pd);
        const scalar_t w01101_cd = - (1 - rd) * (    gd) * (    bd) * (    pd);
        const scalar_t w11101_cd = - (    rd) * (    gd) * (    bd) * (    pd);
        const scalar_t w00011_cd =   (1 - rd) * (1 - gd) * (1 - bd) * (    pd);
        const scalar_t w10011_cd =   (    rd) * (1 - gd) * (1 - bd) * (    pd);
        const scalar_t w01011_cd =   (1 - rd) * (    gd) * (1 - bd) * (    pd);
        const scalar_t w11011_cd =   (    rd) * (    gd) * (1 - bd) * (    pd);
        const scalar_t w00111_cd =   (1 - rd) * (1 - gd) * (    bd) * (    pd);
        const scalar_t w10111_cd =   (    rd) * (1 - gd) * (    bd) * (    pd);
        const scalar_t w01111_cd =   (1 - rd) * (    gd) * (    bd) * (    pd);
        const scalar_t w11111_cd =   (    rd) * (    gd) * (    bd) * (    pd);

        /* derivatives: w to pd */
        const scalar_t w00000_pd = - (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w10000_pd = - (    rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w01000_pd = - (1 - rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w11000_pd = - (    rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w00100_pd = - (1 - rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w10100_pd = - (    rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w01100_pd = - (1 - rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w11100_pd = - (    rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w00010_pd = - (1 - rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w10010_pd = - (    rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w01010_pd = - (1 - rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w11010_pd = - (    rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w00110_pd = - (1 - rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w10110_pd = - (    rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w01110_pd = - (1 - rd) * (    gd) * (    bd) * (    cd);
        const scalar_t w11110_pd = - (    rd) * (    gd) * (    bd) * (    cd);
        const scalar_t w00001_pd =   (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w10001_pd =   (    rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w01001_pd =   (1 - rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w11001_pd =   (    rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w00101_pd =   (1 - rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w10101_pd =   (    rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w01101_pd =   (1 - rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w11101_pd =   (    rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w00011_pd =   (1 - rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w10011_pd =   (    rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w01011_pd =   (1 - rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w11011_pd =   (    rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w00111_pd =   (1 - rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w10111_pd =   (    rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w01111_pd =   (1 - rd) * (    gd) * (    bd) * (    cd);
        const scalar_t w11111_pd =   (    rd) * (    gd) * (    bd) * (    cd);
        for (int i = 0; i < num_channels; ++i) {
            scalar_t grad_o_ = grad_output[index +  height * width * i];

            /* compute gradient of lut */
            atomicAdd(grad_lut + id00000, grad_o_ * w00000);
            atomicAdd(grad_lut + id10000, grad_o_ * w10000);
            atomicAdd(grad_lut + id01000, grad_o_ * w01000);
            atomicAdd(grad_lut + id11000, grad_o_ * w11000);
            atomicAdd(grad_lut + id00100, grad_o_ * w00100);
            atomicAdd(grad_lut + id10100, grad_o_ * w10100);
            atomicAdd(grad_lut + id01100, grad_o_ * w01100);
            atomicAdd(grad_lut + id11100, grad_o_ * w11100);
            atomicAdd(grad_lut + id00010, grad_o_ * w00010);
            atomicAdd(grad_lut + id10010, grad_o_ * w10010);
            atomicAdd(grad_lut + id01010, grad_o_ * w01010);
            atomicAdd(grad_lut + id11010, grad_o_ * w11010);
            atomicAdd(grad_lut + id00110, grad_o_ * w00110);
            atomicAdd(grad_lut + id10110, grad_o_ * w10110);
            atomicAdd(grad_lut + id01110, grad_o_ * w01110);
            atomicAdd(grad_lut + id11110, grad_o_ * w11110);
            atomicAdd(grad_lut + id00001, grad_o_ * w00001);
            atomicAdd(grad_lut + id10001, grad_o_ * w10001);
            atomicAdd(grad_lut + id01001, grad_o_ * w01001);
            atomicAdd(grad_lut + id11001, grad_o_ * w11001);
            atomicAdd(grad_lut + id00101, grad_o_ * w00101);
            atomicAdd(grad_lut + id10101, grad_o_ * w10101);
            atomicAdd(grad_lut + id01101, grad_o_ * w01101);
            atomicAdd(grad_lut + id11101, grad_o_ * w11101);
            atomicAdd(grad_lut + id00011, grad_o_ * w00011);
            atomicAdd(grad_lut + id10011, grad_o_ * w10011);
            atomicAdd(grad_lut + id01011, grad_o_ * w01011);
            atomicAdd(grad_lut + id11011, grad_o_ * w11011);
            atomicAdd(grad_lut + id00111, grad_o_ * w00111);
            atomicAdd(grad_lut + id10111, grad_o_ * w10111);
            atomicAdd(grad_lut + id01111, grad_o_ * w01111);
            atomicAdd(grad_lut + id11111, grad_o_ * w11111);
            /* compute gradient of vertices */
            scalar_t grad_d = 0;
            const scalar_t lut00000 = data_lut[id00000];
            const scalar_t lut10000 = data_lut[id10000];
            const scalar_t lut01000 = data_lut[id01000];
            const scalar_t lut11000 = data_lut[id11000];
            const scalar_t lut00100 = data_lut[id00100];
            const scalar_t lut10100 = data_lut[id10100];
            const scalar_t lut01100 = data_lut[id01100];
            const scalar_t lut11100 = data_lut[id11100];
            const scalar_t lut00010 = data_lut[id00010];
            const scalar_t lut10010 = data_lut[id10010];
            const scalar_t lut01010 = data_lut[id01010];
            const scalar_t lut11010 = data_lut[id11010];
            const scalar_t lut00110 = data_lut[id00110];
            const scalar_t lut10110 = data_lut[id10110];
            const scalar_t lut01110 = data_lut[id01110];
            const scalar_t lut11110 = data_lut[id11110];
            const scalar_t lut00001 = data_lut[id00001];
            const scalar_t lut10001 = data_lut[id10001];
            const scalar_t lut01001 = data_lut[id01001];
            const scalar_t lut11001 = data_lut[id11001];
            const scalar_t lut00101 = data_lut[id00101];
            const scalar_t lut10101 = data_lut[id10101];
            const scalar_t lut01101 = data_lut[id01101];
            const scalar_t lut11101 = data_lut[id11101];
            const scalar_t lut00011 = data_lut[id00011];
            const scalar_t lut10011 = data_lut[id10011];
            const scalar_t lut01011 = data_lut[id01011];
            const scalar_t lut11011 = data_lut[id11011];
            const scalar_t lut00111 = data_lut[id00111];
            const scalar_t lut10111 = data_lut[id10111];
            const scalar_t lut01111 = data_lut[id01111];
            const scalar_t lut11111 = data_lut[id11111];
            grad_d = grad_o_ *
                (w00000_rd * lut00000 + w10000_rd * lut10000 + w01000_rd * lut01000 + w11000_rd * lut11000 +
                 w00100_rd * lut00100 + w10100_rd * lut10100 + w01100_rd * lut01100 + w11100_rd * lut11100 +
                 w00010_rd * lut00010 + w10010_rd * lut10010 + w01010_rd * lut01010 + w11010_rd * lut11010 +
                 w00110_rd * lut00110 + w10110_rd * lut10110 + w01110_rd * lut01110 + w11110_rd * lut11110 +
                 w00001_rd * lut00001 + w10001_rd * lut10001 + w01001_rd * lut01001 + w11001_rd * lut11001 +
                 w00101_rd * lut00101 + w10101_rd * lut10101 + w01101_rd * lut01101 + w11101_rd * lut11101 +
                 w00011_rd * lut00011 + w10011_rd * lut10011 + w01011_rd * lut01011 + w11011_rd * lut11011 +
                 w00111_rd * lut00111 + w10111_rd * lut10111 + w01111_rd * lut01111 + w11111_rd * lut11111);
            // r
            atomicAdd(grad_inp + index, grad_d * 1 / size_bin);

            grad_d = grad_o_ *
                (w00000_gd * lut00000 + w10000_gd * lut10000 + w01000_gd * lut01000 + w11000_gd * lut11000 +
                 w00100_gd * lut00100 + w10100_gd * lut10100 + w01100_gd * lut01100 + w11100_gd * lut11100 +
                 w00010_gd * lut00010 + w10010_gd * lut10010 + w01010_gd * lut01010 + w11010_gd * lut11010 +
                 w00110_gd * lut00110 + w10110_gd * lut10110 + w01110_gd * lut01110 + w11110_gd * lut11110 +
                 w00001_gd * lut00001 + w10001_gd * lut10001 + w01001_gd * lut01001 + w11001_gd * lut11001 +
                 w00101_gd * lut00101 + w10101_gd * lut10101 + w01101_gd * lut01101 + w11101_gd * lut11101 +
                 w00011_gd * lut00011 + w10011_gd * lut10011 + w01011_gd * lut01011 + w11011_gd * lut11011 +
                 w00111_gd * lut00111 + w10111_gd * lut10111 + w01111_gd * lut01111 + w11111_gd * lut11111);
            // g
            atomicAdd(grad_inp + index +  (x == 0 ? 0 : -1), grad_d * 1 / size_bin);

            grad_d = grad_o_ *
                (w00000_bd * lut00000 + w10000_bd * lut10000 + w01000_bd * lut01000 + w11000_bd * lut11000 +
                 w00100_bd * lut00100 + w10100_bd * lut10100 + w01100_bd * lut01100 + w11100_bd * lut11100 +
                 w00010_bd * lut00010 + w10010_bd * lut10010 + w01010_bd * lut01010 + w11010_bd * lut11010 +
                 w00110_bd * lut00110 + w10110_bd * lut10110 + w01110_bd * lut01110 + w11110_bd * lut11110 +
                 w00001_bd * lut00001 + w10001_bd * lut10001 + w01001_bd * lut01001 + w11001_bd * lut11001 +
                 w00101_bd * lut00101 + w10101_bd * lut10101 + w01101_bd * lut01101 + w11101_bd * lut11101 +
                 w00011_bd * lut00011 + w10011_bd * lut10011 + w01011_bd * lut01011 + w11011_bd * lut11011 +
                 w00111_bd * lut00111 + w10111_bd * lut10111 + w01111_bd * lut01111 + w11111_bd * lut11111);
            // b
            atomicAdd(grad_inp + index + (x >= width - 1 ? 0 : 1), grad_d * 1 / size_bin);

            grad_d = grad_o_ *
                (w00000_cd * lut00000 + w10000_cd * lut10000 + w01000_cd * lut01000 + w11000_cd * lut11000 +
                 w00100_cd * lut00100 + w10100_cd * lut10100 + w01100_cd * lut01100 + w11100_cd * lut11100 +
                 w00010_cd * lut00010 + w10010_cd * lut10010 + w01010_cd * lut01010 + w11010_cd * lut11010 +
                 w00110_cd * lut00110 + w10110_cd * lut10110 + w01110_cd * lut01110 + w11110_cd * lut11110 +
                 w00001_cd * lut00001 + w10001_cd * lut10001 + w01001_cd * lut01001 + w11001_cd * lut11001 +
                 w00101_cd * lut00101 + w10101_cd * lut10101 + w01101_cd * lut01101 + w11101_cd * lut11101 +
                 w00011_cd * lut00011 + w10011_cd * lut10011 + w01011_cd * lut01011 + w11011_cd * lut11011 +
                 w00111_cd * lut00111 + w10111_cd * lut10111 + w01111_cd * lut01111 + w11111_cd * lut11111);
            // c
            atomicAdd(grad_inp + index + 
                (y < 1 ? 0 : -width), grad_d * 1 / size_bin);

    grad_d = grad_o_ *
                (w00000_pd * lut00000 + w10000_pd * lut10000 + w01000_pd * lut01000 + w11000_pd * lut11000 +
                 w00100_pd * lut00100 + w10100_pd * lut10100 + w01100_pd * lut01100 + w11100_pd * lut11100 +
                 w00010_pd * lut00010 + w10010_pd * lut10010 + w01010_pd * lut01010 + w11010_pd * lut11010 +
                 w00110_pd * lut00110 + w10110_pd * lut10110 + w01110_pd * lut01110 + w11110_pd * lut11110 +
                 w00001_pd * lut00001 + w10001_pd * lut10001 + w01001_pd * lut01001 + w11001_pd * lut11001 +
                 w00101_pd * lut00101 + w10101_pd * lut10101 + w01101_pd * lut01101 + w11101_pd * lut11101 +
                 w00011_pd * lut00011 + w10011_pd * lut10011 + w01011_pd * lut01011 + w11011_pd * lut11011 +
                 w00111_pd * lut00111 + w10111_pd * lut10111 + w01111_pd * lut01111 + w11111_pd * lut11111);
            // p
            atomicAdd(grad_inp + index +  (y >= height - 1 ? 0 : width), grad_d * 1 / size_bin);
        }
    }
}


void SDLUT5D1TransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output) {

    // sdlut5D1_transform_sanity_check(input, lut, output);

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
            input.scalar_type(), "sdlut5D1_transform_cuda_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                sdlut5D1_transform_4d_cuda_forward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_inp, data_lut,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}



void SDLUT5D1TransformBackwardCUDAKernelLauncher(
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
            input.scalar_type(), "sdlut5D1_transform_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();

                sdlut5D1_transform_4d_cuda_backward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_inp, data_lut, 
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}