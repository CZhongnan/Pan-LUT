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


void sdlut_transform_sanity_check(
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
__global__ void sdlut_transform_4d_cuda_forward_kernel(
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
        const scalar_t r0 = data_inp[index];
        const scalar_t g0 = data_inp[index + (x >= width - 1 ? -1 : 1)];
        const scalar_t b0 = data_inp[index + (y >= height - 1 ? -width : width)];
        const scalar_t c0 = data_inp[index + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r1 = data_inp[index + height * width];
        const scalar_t g1 = data_inp[index + height * width + (x >= width - 1 ? -1 : 1)];
        const scalar_t b1 = data_inp[index + height * width + (y >= height - 1 ? -width : width)];
        const scalar_t c1 = data_inp[index + height * width + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r2 = data_inp[index + height * width * 2];
        const scalar_t g2 = data_inp[index + height * width * 2 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b2 = data_inp[index + height * width * 2 + (y >= height - 1 ? -width : width)];
        const scalar_t c2 = data_inp[index + height * width * 2 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r3 = data_inp[index + height * width * 3];
        const scalar_t g3 = data_inp[index + height * width * 3 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b3 = data_inp[index + height * width * 3 + (y >= height - 1 ? -width : width)];
        const scalar_t c3 = data_inp[index + height * width * 3 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r4 = data_inp[index + height * width * 4];
        const scalar_t g4 = data_inp[index + height * width * 4 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b4 = data_inp[index + height * width * 4 + (y >= height - 1 ? -width : width)];
        const scalar_t c4 = data_inp[index + height * width * 4 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        // const scalar_t r = data_inp[index];
        // const scalar_t g = data_inp[index +  height * width];
        // const scalar_t b = data_inp[index +  height * width * 2];
        // const scalar_t c = data_inp[index +  height * width * 3];
        
        /* retrieve index of the interpolation verticess */
        const int32_t rid0 = clamp((int32_t)floor(r0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid0 = clamp((int32_t)floor(g0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid0 = clamp((int32_t)floor(b0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid0 = clamp((int32_t)floor(c0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid1 = clamp((int32_t)floor(r1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid1 = clamp((int32_t)floor(g1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid1 = clamp((int32_t)floor(b1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid1 = clamp((int32_t)floor(c1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid2 = clamp((int32_t)floor(r2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid2 = clamp((int32_t)floor(g2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid2 = clamp((int32_t)floor(b2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid2 = clamp((int32_t)floor(c2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid3 = clamp((int32_t)floor(r3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid3 = clamp((int32_t)floor(g3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid3 = clamp((int32_t)floor(b3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid3 = clamp((int32_t)floor(c3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid4 = clamp((int32_t)floor(r4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid4 = clamp((int32_t)floor(g4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid4 = clamp((int32_t)floor(b4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid4 = clamp((int32_t)floor(c4 * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility varsdbles for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int _0id0000 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id1000 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id0100 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id0010 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id0001 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id1100 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id1010 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id1001 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id0110 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id0101 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id0011 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id1110 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id1101 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id1011 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id0111 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id1111 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);

        const int _1id0000 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id1000 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id0100 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id0010 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id0001 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id1100 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id1010 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1001 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id0110 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id0101 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id0011 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1110 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1101 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id1011 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);
        const int _1id0111 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);
        const int _1id1111 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);

        const int _2id0000 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id1000 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id0100 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id0010 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id0001 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id1100 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id1010 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1001 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id0110 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id0101 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id0011 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1110 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1101 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id1011 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        const int _2id0111 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        const int _2id1111 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        
        const int _3id0000 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id1000 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id0100 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id0010 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id0001 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id1100 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id1010 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1001 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id0110 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id0101 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id0011 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1110 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1101 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id1011 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        const int _3id0111 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        const int _3id1111 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        
        const int _4id0000 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id1000 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id0100 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id0010 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id0001 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id1100 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id1010 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1001 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id0110 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id0101 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id0011 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1110 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1101 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id1011 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        const int _4id0111 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        const int _4id1111 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        

        
        

        
        
        

        /* compute interpolation weights */
        const scalar_t rd0 = (r0 - size_bin * rid0) / size_bin;
        const scalar_t gd0 = (g0 - size_bin * gid0) / size_bin;
        const scalar_t bd0 = (b0 - size_bin * bid0) / size_bin;
        const scalar_t cd0 = (c0 - size_bin * cid0) / size_bin;
        const scalar_t rd1 = (r1 - size_bin * rid1) / size_bin;
        const scalar_t gd1 = (g1 - size_bin * gid1) / size_bin;
        const scalar_t bd1 = (b1 - size_bin * bid1) / size_bin;
        const scalar_t cd1 = (c1 - size_bin * cid1) / size_bin;
        const scalar_t rd2 = (r2 - size_bin * rid2) / size_bin;
        const scalar_t gd2 = (g2 - size_bin * gid2) / size_bin;
        const scalar_t bd2 = (b2 - size_bin * bid2) / size_bin;
        const scalar_t cd2 = (c2 - size_bin * cid2) / size_bin;
        const scalar_t rd3 = (r3 - size_bin * rid3) / size_bin;
        const scalar_t gd3 = (g3 - size_bin * gid3) / size_bin;
        const scalar_t bd3 = (b3 - size_bin * bid3) / size_bin;
        const scalar_t cd3 = (c3 - size_bin * cid3) / size_bin;
        const scalar_t rd4 = (r4 - size_bin * rid4) / size_bin;
        const scalar_t gd4 = (g4 - size_bin * gid4) / size_bin;
        const scalar_t bd4 = (b4 - size_bin * bid4) / size_bin;
        const scalar_t cd4 = (c4 - size_bin * cid4) / size_bin;

        const scalar_t _0w0000 = (1 - rd0) * (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w1000 = (    rd0) * (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w0100 = (1 - rd0) * (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w0010 = (1 - rd0) * (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w0001 = (1 - rd0) * (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w1100 = (    rd0) * (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w1010 = (    rd0) * (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w1001 = (    rd0) * (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w0110 = (1 - rd0) * (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w0101 = (1 - rd0) * (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w0011 = (1 - rd0) * (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t _0w1110 = (    rd0) * (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w1101 = (    rd0) * (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w1011 = (    rd0) * (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t _0w0111 = (1 - rd0) * (    gd0) * (    bd0) * (    cd0);
        const scalar_t _0w1111 = (    rd0) * (    gd0) * (    bd0) * (    cd0);

        const scalar_t _1w0000 = (1 - rd1) * (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w1000 = (    rd1) * (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w0100 = (1 - rd1) * (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w0010 = (1 - rd1) * (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w0001 = (1 - rd1) * (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w1100 = (    rd1) * (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w1010 = (    rd1) * (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w1001 = (    rd1) * (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w0110 = (1 - rd1) * (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w0101 = (1 - rd1) * (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w0011 = (1 - rd1) * (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t _1w1110 = (    rd1) * (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w1101 = (    rd1) * (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w1011 = (    rd1) * (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t _1w0111 = (1 - rd1) * (    gd1) * (    bd1) * (    cd1);
        const scalar_t _1w1111 = (    rd1) * (    gd1) * (    bd1) * (    cd1);

        const scalar_t _2w0000 = (1 - rd2) * (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w1000 = (    rd2) * (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w0100 = (1 - rd2) * (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w0010 = (1 - rd2) * (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w0001 = (1 - rd2) * (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w1100 = (    rd2) * (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w1010 = (    rd2) * (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w1001 = (    rd2) * (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w0110 = (1 - rd2) * (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w0101 = (1 - rd2) * (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w0011 = (1 - rd2) * (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t _2w1110 = (    rd2) * (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w1101 = (    rd2) * (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w1011 = (    rd2) * (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t _2w0111 = (1 - rd2) * (    gd2) * (    bd2) * (    cd2);
        const scalar_t _2w1111 = (    rd2) * (    gd2) * (    bd2) * (    cd2);

        const scalar_t _3w0000 = (1 - rd3) * (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w1000 = (    rd3) * (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w0100 = (1 - rd3) * (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w0010 = (1 - rd3) * (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w0001 = (1 - rd3) * (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w1100 = (    rd3) * (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w1010 = (    rd3) * (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w1001 = (    rd3) * (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w0110 = (1 - rd3) * (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w0101 = (1 - rd3) * (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w0011 = (1 - rd3) * (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t _3w1110 = (    rd3) * (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w1101 = (    rd3) * (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w1011 = (    rd3) * (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t _3w0111 = (1 - rd3) * (    gd3) * (    bd3) * (    cd3);
        const scalar_t _3w1111 = (    rd3) * (    gd3) * (    bd3) * (    cd3);

        const scalar_t _4w0000 = (1 - rd4) * (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w1000 = (    rd4) * (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w0100 = (1 - rd4) * (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w0010 = (1 - rd4) * (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w0001 = (1 - rd4) * (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w1100 = (    rd4) * (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w1010 = (    rd4) * (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w1001 = (    rd4) * (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w0110 = (1 - rd4) * (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w0101 = (1 - rd4) * (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w0011 = (1 - rd4) * (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t _4w1110 = (    rd4) * (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w1101 = (    rd4) * (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w1011 = (    rd4) * (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t _4w0111 = (1 - rd4) * (    gd4) * (    bd4) * (    cd4);
        const scalar_t _4w1111 = (    rd4) * (    gd4) * (    bd4) * (    cd4);
        
        
        
        
        
        data_col[index] =
                _0w0000 * data_lut[_0id0000] + _0w1000 * data_lut[_0id1000] +
                _0w0100 * data_lut[_0id0100] + _0w0010 * data_lut[_0id0010] +
                _0w0001 * data_lut[_0id0001] + _0w1100 * data_lut[_0id1100] +
                _0w1010 * data_lut[_0id1010] + _0w1001 * data_lut[_0id1001] +
                _0w0110 * data_lut[_0id0110] + _0w0101 * data_lut[_0id0101] +
                _0w0011 * data_lut[_0id0011] + _0w1110 * data_lut[_0id1110] +
                _0w1101 * data_lut[_0id1101] + _0w1011 * data_lut[_0id1011] +
                _0w0111 * data_lut[_0id0111] + _0w1111 * data_lut[_0id1111];
        data_col[index + height * width * 1] =
                _1w0000 * data_lut[_1id0000 + stride_lut_4 * 1] + _1w1000 * data_lut[_1id1000 + stride_lut_4 * 1] +
                _1w0100 * data_lut[_1id0100 + stride_lut_4 * 1] + _1w0010 * data_lut[_1id0010 + stride_lut_4 * 1] +
                _1w0001 * data_lut[_1id0001 + stride_lut_4 * 1] + _1w1100 * data_lut[_1id1100 + stride_lut_4 * 1] +
                _1w1010 * data_lut[_1id1010 + stride_lut_4 * 1] + _1w1001 * data_lut[_1id1001 + stride_lut_4 * 1] +
                _1w0110 * data_lut[_1id0110 + stride_lut_4 * 1] + _1w0101 * data_lut[_1id0101 + stride_lut_4 * 1] +
                _1w0011 * data_lut[_1id0011 + stride_lut_4 * 1] + _1w1110 * data_lut[_1id1110 + stride_lut_4 * 1] +
                _1w1101 * data_lut[_1id1101 + stride_lut_4 * 1] + _1w1011 * data_lut[_1id1011 + stride_lut_4 * 1] +
                _1w0111 * data_lut[_1id0111 + stride_lut_4 * 1] + _1w1111 * data_lut[_1id1111 + stride_lut_4 * 1];
        data_col[index + height * width * 2] = 
                _2w0000 * data_lut[_2id0000 + stride_lut_4 * 2] + _2w1000 * data_lut[_2id1000 + stride_lut_4 * 2] +
                _2w0100 * data_lut[_2id0100 + stride_lut_4 * 2] + _2w0010 * data_lut[_2id0010 + stride_lut_4 * 2] +
                _2w0001 * data_lut[_2id0001 + stride_lut_4 * 2] + _2w1100 * data_lut[_2id1100 + stride_lut_4 * 2] +
                _2w1010 * data_lut[_2id1010 + stride_lut_4 * 2] + _2w1001 * data_lut[_2id1001 + stride_lut_4 * 2] +
                _2w0110 * data_lut[_2id0110 + stride_lut_4 * 2] + _2w0101 * data_lut[_2id0101 + stride_lut_4 * 2] +
                _2w0011 * data_lut[_2id0011 + stride_lut_4 * 2] + _2w1110 * data_lut[_2id1110 + stride_lut_4 * 2] +
                _2w1101 * data_lut[_2id1101 + stride_lut_4 * 2] + _2w1011 * data_lut[_2id1011 + stride_lut_4 * 2] +
                _2w0111 * data_lut[_2id0111 + stride_lut_4 * 2] + _2w1111 * data_lut[_2id1111 + stride_lut_4 * 2];
        data_col[index + height * width * 3] = 
                _3w0000 * data_lut[_3id0000 + stride_lut_4 * 3] + _3w1000 * data_lut[_3id1000 + stride_lut_4 * 3] +
                _3w0100 * data_lut[_3id0100 + stride_lut_4 * 3] + _3w0010 * data_lut[_3id0010 + stride_lut_4 * 3] +
                _3w0001 * data_lut[_3id0001 + stride_lut_4 * 3] + _3w1100 * data_lut[_3id1100 + stride_lut_4 * 3] +
                _3w1010 * data_lut[_3id1010 + stride_lut_4 * 3] + _3w1001 * data_lut[_3id1001 + stride_lut_4 * 3] +
                _3w0110 * data_lut[_3id0110 + stride_lut_4 * 3] + _3w0101 * data_lut[_3id0101 + stride_lut_4 * 3] +
                _3w0011 * data_lut[_3id0011 + stride_lut_4 * 3] + _3w1110 * data_lut[_3id1110 + stride_lut_4 * 3] +   
                _3w1101 * data_lut[_3id1101 + stride_lut_4 * 3] + _3w1011 * data_lut[_3id1011 + stride_lut_4 * 3] +
                _3w0111 * data_lut[_3id0111 + stride_lut_4 * 3] + _3w1111 * data_lut[_3id1111 + stride_lut_4 * 3];
        data_col[index + height * width * 4] = 
                _4w0000 * data_lut[_4id0000 + stride_lut_4 * 4] + _4w1000 * data_lut[_4id1000 + stride_lut_4 * 4] +
                _4w0100 * data_lut[_4id0100 + stride_lut_4 * 4] + _4w0010 * data_lut[_4id0010 + stride_lut_4 * 4] +
                _4w0001 * data_lut[_4id0001 + stride_lut_4 * 4] + _4w1100 * data_lut[_4id1100 + stride_lut_4 * 4] +   
                _4w1010 * data_lut[_4id1010 + stride_lut_4 * 4] + _4w1001 * data_lut[_4id1001 + stride_lut_4 * 4] +
                _4w0110 * data_lut[_4id0110 + stride_lut_4 * 4] + _4w0101 * data_lut[_4id0101 + stride_lut_4 * 4] +
                _4w0011 * data_lut[_4id0011 + stride_lut_4 * 4] + _4w1110 * data_lut[_4id1110 + stride_lut_4 * 4] +
                _4w1101 * data_lut[_4id1101 + stride_lut_4 * 4] + _4w1011 * data_lut[_4id1011 + stride_lut_4 * 4] +
                _4w0111 * data_lut[_4id0111 + stride_lut_4 * 4] + _4w1111 * data_lut[_4id1111 + stride_lut_4 * 4];

        /* Execute the interpolation */
        // printf("num_channels: %d\n", num_channels);
        // for (int i = 0; i < num_channels; ++i) {
        //     data_col[index + height * width * i] =
        //         w0000 * data_lut[id0000 + stride_lut_4 * i] + w1000 * data_lut[id1000 + stride_lut_4 * i] +
        //         w0100 * data_lut[id0100 + stride_lut_4 * i] + w0010 * data_lut[id0010 + stride_lut_4 * i] +
        //         w0001 * data_lut[id0001 + stride_lut_4 * i] + w1100 * data_lut[id1100 + stride_lut_4 * i] +
        //         w1010 * data_lut[id1010 + stride_lut_4 * i] + w1001 * data_lut[id1001 + stride_lut_4 * i] +
        //         w0110 * data_lut[id0110 + stride_lut_4 * i] + w0101 * data_lut[id0101 + stride_lut_4 * i] +
        //         w0011 * data_lut[id0011 + stride_lut_4 * i] + w1110 * data_lut[id1110 + stride_lut_4 * i] +
        //         w1101 * data_lut[id1101 + stride_lut_4 * i] + w1011 * data_lut[id1011 + stride_lut_4 * i] +
        //         w0111 * data_lut[id0111 + stride_lut_4 * i] + w1111 * data_lut[id1111 + stride_lut_4 * i];
        }
    }




template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void sdlut_transform_4d_cuda_backward_kernel(
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
        const scalar_t r0 = data_inp[index];
        const scalar_t g0 = data_inp[index + (x >= width - 1 ? -1 : 1)];
        const scalar_t b0 = data_inp[index + (y >= height - 1 ? -width : width)];
        const scalar_t c0 = data_inp[index + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r1 = data_inp[index + height * width];
        const scalar_t g1 = data_inp[index + height * width + (x >= width - 1 ? -1 : 1)];
        const scalar_t b1 = data_inp[index + height * width + (y >= height - 1 ? -width : width)];
        const scalar_t c1 = data_inp[index + height * width + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r2 = data_inp[index + height * width * 2];
        const scalar_t g2 = data_inp[index + height * width * 2 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b2 = data_inp[index + height * width * 2 + (y >= height - 1 ? -width : width)];
        const scalar_t c2 = data_inp[index + height * width * 2 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r3 = data_inp[index + height * width * 3];
        const scalar_t g3 = data_inp[index + height * width * 3 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b3 = data_inp[index + height * width * 3 + (y >= height - 1 ? -width : width)];
        const scalar_t c3 = data_inp[index + height * width * 3 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        const scalar_t r4 = data_inp[index + height * width * 4];
        const scalar_t g4 = data_inp[index + height * width * 4 + (x >= width - 1 ? -1 : 1)];
        const scalar_t b4 = data_inp[index + height * width * 4 + (y >= height - 1 ? -width : width)];
        const scalar_t c4 = data_inp[index + height * width * 4 + 
            (y >= height - 1 ? -width : width) + 
            (x >= width - 1 ? -1 : 1)];
        // const scalar_t r = data_inp[index];
        // const scalar_t g = data_inp[index +  height * width];
        // const scalar_t b = data_inp[index +  height * width * 2];
        // const scalar_t c = data_inp[index +  height * width * 3];
        
        /* retrieve index of the interpolation verticess */
        const int32_t rid0 = clamp((int32_t)floor(r0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid0 = clamp((int32_t)floor(g0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid0 = clamp((int32_t)floor(b0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid0 = clamp((int32_t)floor(c0 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid1 = clamp((int32_t)floor(r1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid1 = clamp((int32_t)floor(g1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid1 = clamp((int32_t)floor(b1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid1 = clamp((int32_t)floor(c1 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid2 = clamp((int32_t)floor(r2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid2 = clamp((int32_t)floor(g2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid2 = clamp((int32_t)floor(b2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid2 = clamp((int32_t)floor(c2 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid3 = clamp((int32_t)floor(r3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid3 = clamp((int32_t)floor(g3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid3 = clamp((int32_t)floor(b3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid3 = clamp((int32_t)floor(c3 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t rid4 = clamp((int32_t)floor(r4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid4 = clamp((int32_t)floor(g4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid4 = clamp((int32_t)floor(b4 * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid4 = clamp((int32_t)floor(c4 * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility varsdbles for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;
        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int _0id0000 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id1000 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id0100 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id0010 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id0001 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id1100 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0    );
        const int _0id1010 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id1001 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id0110 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id0101 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id0011 = (rid0    ) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id1110 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0    );
        const int _0id1101 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0    ) + stride_lut_3 * (cid0 + 1);
        const int _0id1011 = (rid0 + 1) + stride_lut * (gid0    ) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id0111 = (rid0    ) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);
        const int _0id1111 = (rid0 + 1) + stride_lut * (gid0 + 1) + stride_lut_2 * (bid0 + 1) + stride_lut_3 * (cid0 + 1);

        const int _1id0000 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id1000 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id0100 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id0010 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id0001 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id1100 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1    );
        const int _1id1010 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1001 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id0110 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id0101 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id0011 = (rid1  ) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1110 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1    );
        const int _1id1101 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1    ) + stride_lut_3 * (cid1 + 1);
        const int _1id1011 = (rid1 + 1) + stride_lut * (gid1    ) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);
        const int _1id0111 = (rid1  ) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);
        const int _1id1111 = (rid1 + 1) + stride_lut * (gid1 + 1) + stride_lut_2 * (bid1 + 1) + stride_lut_3 * (cid1 + 1);

        const int _2id0000 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id1000 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id0100 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id0010 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id0001 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id1100 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2    );
        const int _2id1010 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1001 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id0110 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id0101 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id0011 = (rid2  ) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1110 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2    );
        const int _2id1101 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2    ) + stride_lut_3 * (cid2 + 1);
        const int _2id1011 = (rid2 + 1) + stride_lut * (gid2    ) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        const int _2id0111 = (rid2  ) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        const int _2id1111 = (rid2 + 1) + stride_lut * (gid2 + 1) + stride_lut_2 * (bid2 + 1) + stride_lut_3 * (cid2 + 1);
        
        const int _3id0000 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id1000 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id0100 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id0010 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id0001 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id1100 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3    );
        const int _3id1010 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1001 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id0110 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id0101 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id0011 = (rid3  ) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1110 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3    );
        const int _3id1101 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3    ) + stride_lut_3 * (cid3 + 1);
        const int _3id1011 = (rid3 + 1) + stride_lut * (gid3    ) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        const int _3id0111 = (rid3  ) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        const int _3id1111 = (rid3 + 1) + stride_lut * (gid3 + 1) + stride_lut_2 * (bid3 + 1) + stride_lut_3 * (cid3 + 1);
        
        const int _4id0000 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id1000 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id0100 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id0010 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id0001 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id1100 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4    );
        const int _4id1010 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1001 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id0110 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id0101 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id0011 = (rid4  ) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1110 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4    );
        const int _4id1101 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4    ) + stride_lut_3 * (cid4 + 1);
        const int _4id1011 = (rid4 + 1) + stride_lut * (gid4    ) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        const int _4id0111 = (rid4  ) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        const int _4id1111 = (rid4 + 1) + stride_lut * (gid4 + 1) + stride_lut_2 * (bid4 + 1) + stride_lut_3 * (cid4 + 1);
        

        
        

        
        
        

        /* compute interpolation weights */
        const scalar_t rd0 = (r0 - size_bin * rid0) / size_bin;
        const scalar_t gd0 = (g0 - size_bin * gid0) / size_bin;
        const scalar_t bd0 = (b0 - size_bin * bid0) / size_bin;
        const scalar_t cd0 = (c0 - size_bin * cid0) / size_bin;
        const scalar_t rd1 = (r1 - size_bin * rid1) / size_bin;
        const scalar_t gd1 = (g1 - size_bin * gid1) / size_bin;
        const scalar_t bd1 = (b1 - size_bin * bid1) / size_bin;
        const scalar_t cd1 = (c1 - size_bin * cid1) / size_bin;
        const scalar_t rd2 = (r2 - size_bin * rid2) / size_bin;
        const scalar_t gd2 = (g2 - size_bin * gid2) / size_bin;
        const scalar_t bd2 = (b2 - size_bin * bid2) / size_bin;
        const scalar_t cd2 = (c2 - size_bin * cid2) / size_bin;
        const scalar_t rd3 = (r3 - size_bin * rid3) / size_bin;
        const scalar_t gd3 = (g3 - size_bin * gid3) / size_bin;
        const scalar_t bd3 = (b3 - size_bin * bid3) / size_bin;
        const scalar_t cd3 = (c3 - size_bin * cid3) / size_bin;
        const scalar_t rd4 = (r4 - size_bin * rid4) / size_bin;
        const scalar_t gd4 = (g4 - size_bin * gid4) / size_bin;
        const scalar_t bd4 = (b4 - size_bin * bid4) / size_bin;
        const scalar_t cd4 = (c4 - size_bin * cid4) / size_bin;

        const scalar_t _0w0000 = (1 - rd0) * (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w1000 = (    rd0) * (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w0100 = (1 - rd0) * (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w0010 = (1 - rd0) * (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w0001 = (1 - rd0) * (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w1100 = (    rd0) * (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t _0w1010 = (    rd0) * (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w1001 = (    rd0) * (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w0110 = (1 - rd0) * (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w0101 = (1 - rd0) * (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w0011 = (1 - rd0) * (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t _0w1110 = (    rd0) * (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t _0w1101 = (    rd0) * (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t _0w1011 = (    rd0) * (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t _0w0111 = (1 - rd0) * (    gd0) * (    bd0) * (    cd0);
        const scalar_t _0w1111 = (    rd0) * (    gd0) * (    bd0) * (    cd0);

        const scalar_t _1w0000 = (1 - rd1) * (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w1000 = (    rd1) * (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w0100 = (1 - rd1) * (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w0010 = (1 - rd1) * (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w0001 = (1 - rd1) * (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w1100 = (    rd1) * (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t _1w1010 = (    rd1) * (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w1001 = (    rd1) * (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w0110 = (1 - rd1) * (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w0101 = (1 - rd1) * (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w0011 = (1 - rd1) * (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t _1w1110 = (    rd1) * (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t _1w1101 = (    rd1) * (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t _1w1011 = (    rd1) * (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t _1w0111 = (1 - rd1) * (    gd1) * (    bd1) * (    cd1);
        const scalar_t _1w1111 = (    rd1) * (    gd1) * (    bd1) * (    cd1);

        const scalar_t _2w0000 = (1 - rd2) * (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w1000 = (    rd2) * (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w0100 = (1 - rd2) * (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w0010 = (1 - rd2) * (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w0001 = (1 - rd2) * (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w1100 = (    rd2) * (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t _2w1010 = (    rd2) * (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w1001 = (    rd2) * (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w0110 = (1 - rd2) * (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w0101 = (1 - rd2) * (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w0011 = (1 - rd2) * (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t _2w1110 = (    rd2) * (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t _2w1101 = (    rd2) * (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t _2w1011 = (    rd2) * (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t _2w0111 = (1 - rd2) * (    gd2) * (    bd2) * (    cd2);
        const scalar_t _2w1111 = (    rd2) * (    gd2) * (    bd2) * (    cd2);

        const scalar_t _3w0000 = (1 - rd3) * (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w1000 = (    rd3) * (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w0100 = (1 - rd3) * (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w0010 = (1 - rd3) * (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w0001 = (1 - rd3) * (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w1100 = (    rd3) * (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t _3w1010 = (    rd3) * (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w1001 = (    rd3) * (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w0110 = (1 - rd3) * (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w0101 = (1 - rd3) * (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w0011 = (1 - rd3) * (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t _3w1110 = (    rd3) * (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t _3w1101 = (    rd3) * (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t _3w1011 = (    rd3) * (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t _3w0111 = (1 - rd3) * (    gd3) * (    bd3) * (    cd3);
        const scalar_t _3w1111 = (    rd3) * (    gd3) * (    bd3) * (    cd3);

        const scalar_t _4w0000 = (1 - rd4) * (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w1000 = (    rd4) * (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w0100 = (1 - rd4) * (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w0010 = (1 - rd4) * (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w0001 = (1 - rd4) * (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w1100 = (    rd4) * (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t _4w1010 = (    rd4) * (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w1001 = (    rd4) * (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w0110 = (1 - rd4) * (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w0101 = (1 - rd4) * (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w0011 = (1 - rd4) * (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t _4w1110 = (    rd4) * (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t _4w1101 = (    rd4) * (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t _4w1011 = (    rd4) * (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t _4w0111 = (1 - rd4) * (    gd4) * (    bd4) * (    cd4);
        const scalar_t _4w1111 = (    rd4) * (    gd4) * (    bd4) * (    cd4);

        /* derivatives: w to rd */
        const scalar_t w0000_rd0 = - (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w1000_rd0 =   (1 - gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w0100_rd0 = - (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w0010_rd0 = - (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t w0001_rd0 = - (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t w1100_rd0 =   (    gd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w1010_rd0 =   (1 - gd0) * (    bd0) * (1 - cd0);
        const scalar_t w1001_rd0 =   (1 - gd0) * (1 - bd0) * (    cd0);
        const scalar_t w0110_rd0 = - (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t w0101_rd0 = - (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t w0011_rd0 = - (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t w1110_rd0 =   (    gd0) * (    bd0) * (1 - cd0);
        const scalar_t w1101_rd0 =   (    gd0) * (1 - bd0) * (    cd0);
        const scalar_t w1011_rd0 =   (1 - gd0) * (    bd0) * (    cd0);
        const scalar_t w0111_rd0 = - (    gd0) * (    bd0) * (    cd0);
        const scalar_t w1111_rd0 =   (    gd0) * (    bd0) * (    cd0);

        const scalar_t w0000_gd0 = - (1 - rd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w1000_gd0 =   (1 - rd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w0100_gd0 = - (    rd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w0010_gd0 = - (1 - rd0) * (    bd0) * (1 - cd0);
        const scalar_t w0001_gd0 = - (1 - rd0) * (1 - bd0) * (    cd0);
        const scalar_t w1100_gd0 =   (    rd0) * (1 - bd0) * (1 - cd0);
        const scalar_t w1010_gd0 =   (1 - rd0) * (    bd0) * (1 - cd0);
        const scalar_t w1001_gd0 =   (1 - rd0) * (1 - bd0) * (    cd0);
        const scalar_t w0110_gd0 = - (    rd0) * (    bd0) * (1 - cd0);
        const scalar_t w0101_gd0 = - (    rd0) * (1 - bd0) * (    cd0);
        const scalar_t w0011_gd0 = - (1 - rd0) * (    bd0) * (    cd0);
        const scalar_t w1110_gd0 =   (    rd0) * (    bd0) * (1 - cd0);
        const scalar_t w1101_gd0 =   (    rd0) * (1 - bd0) * (    cd0);
        const scalar_t w1011_gd0 =   (1 - rd0) * (    bd0) * (    cd0);
        const scalar_t w0111_gd0 = - (    rd0) * (    bd0) * (    cd0);
        const scalar_t w1111_gd0 =   (    rd0) * (    bd0) * (    cd0);

        // bd0系列
        const scalar_t w0000_bd0 = - (1 - rd0) * (1 - gd0) * (1 - cd0);
        const scalar_t w1000_bd0 =   (1 - rd0) * (1 - gd0) * (1 - cd0);
        const scalar_t w0100_bd0 = - (    rd0) * (1 - gd0) * (1 - cd0);
        const scalar_t w0010_bd0 = - (1 - rd0) * (    gd0) * (1 - cd0);
        const scalar_t w0001_bd0 = - (1 - rd0) * (1 - gd0) * (    cd0);
        const scalar_t w1100_bd0 =   (    rd0) * (1 - gd0) * (1 - cd0);
        const scalar_t w1010_bd0 =   (1 - rd0) * (    gd0) * (1 - cd0);
        const scalar_t w1001_bd0 =   (1 - rd0) * (1 - gd0) * (    cd0);
        const scalar_t w0110_bd0 = - (    rd0) * (    gd0) * (1 - cd0);
        const scalar_t w0101_bd0 = - (    rd0) * (1 - gd0) * (    cd0);
        const scalar_t w0011_bd0 = - (1 - rd0) * (    gd0) * (    cd0);
        const scalar_t w1110_bd0 =   (    rd0) * (    gd0) * (1 - cd0);
        const scalar_t w1101_bd0 =   (    rd0) * (1 - gd0) * (    cd0);
        const scalar_t w1011_bd0 =   (1 - rd0) * (    gd0) * (    cd0);
        const scalar_t w0111_bd0 = - (    rd0) * (    gd0) * (    cd0);
        const scalar_t w1111_bd0 =   (    rd0) * (    gd0) * (    cd0);

        // cd0系列
        const scalar_t w0000_cd0 = - (1 - rd0) * (1 - gd0) * (1 - bd0);
        const scalar_t w1000_cd0 =   (1 - rd0) * (1 - gd0) * (1 - bd0);
        const scalar_t w0100_cd0 = - (    rd0) * (1 - gd0) * (1 - bd0);
        const scalar_t w0010_cd0 = - (1 - rd0) * (    gd0) * (1 - bd0);
        const scalar_t w0001_cd0 = - (1 - rd0) * (1 - gd0) * (    bd0);
        const scalar_t w1100_cd0 =   (    rd0) * (1 - gd0) * (1 - bd0);
        const scalar_t w1010_cd0 =   (1 - rd0) * (    gd0) * (1 - bd0);
        const scalar_t w1001_cd0 =   (1 - rd0) * (1 - gd0) * (    bd0);
        const scalar_t w0110_cd0 = - (    rd0) * (    gd0) * (1 - bd0);
        const scalar_t w0101_cd0 = - (    rd0) * (1 - gd0) * (    bd0);
        const scalar_t w0011_cd0 = - (1 - rd0) * (    gd0) * (    bd0);
        const scalar_t w1110_cd0 =   (    rd0) * (    gd0) * (1 - bd0);
        const scalar_t w1101_cd0 =   (    rd0) * (1 - gd0) * (    bd0);
        const scalar_t w1011_cd0 =   (1 - rd0) * (    gd0) * (    bd0);
        const scalar_t w0111_cd0 = - (    rd0) * (    gd0) * (    bd0);
        const scalar_t w1111_cd0 =   (    rd0) * (    gd0) * (    bd0);

        // rd1系列
        const scalar_t w0000_rd1 = - (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w1000_rd1 =   (1 - gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w0100_rd1 = - (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w0010_rd1 = - (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t w0001_rd1 = - (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t w1100_rd1 =   (    gd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w1010_rd1 =   (1 - gd1) * (    bd1) * (1 - cd1);
        const scalar_t w1001_rd1 =   (1 - gd1) * (1 - bd1) * (    cd1);
        const scalar_t w0110_rd1 = - (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t w0101_rd1 = - (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t w0011_rd1 = - (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t w1110_rd1 =   (    gd1) * (    bd1) * (1 - cd1);
        const scalar_t w1101_rd1 =   (    gd1) * (1 - bd1) * (    cd1);
        const scalar_t w1011_rd1 =   (1 - gd1) * (    bd1) * (    cd1);
        const scalar_t w0111_rd1 = - (    gd1) * (    bd1) * (    cd1);
        const scalar_t w1111_rd1 =   (    gd1) * (    bd1) * (    cd1);

        // gd1系列
        const scalar_t w0000_gd1 = - (1 - rd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w1000_gd1 =   (1 - rd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w0100_gd1 = - (    rd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w0010_gd1 = - (1 - rd1) * (    bd1) * (1 - cd1);
        const scalar_t w0001_gd1 = - (1 - rd1) * (1 - bd1) * (    cd1);
        const scalar_t w1100_gd1 =   (    rd1) * (1 - bd1) * (1 - cd1);
        const scalar_t w1010_gd1 =   (1 - rd1) * (    bd1) * (1 - cd1);
        const scalar_t w1001_gd1 =   (1 - rd1) * (1 - bd1) * (    cd1);
        const scalar_t w0110_gd1 = - (    rd1) * (    bd1) * (1 - cd1);
        const scalar_t w0101_gd1 = - (    rd1) * (1 - bd1) * (    cd1);
        const scalar_t w0011_gd1 = - (1 - rd1) * (    bd1) * (    cd1);
        const scalar_t w1110_gd1 =   (    rd1) * (    bd1) * (1 - cd1);
        const scalar_t w1101_gd1 =   (    rd1) * (1 - bd1) * (    cd1);
        const scalar_t w1011_gd1 =   (1 - rd1) * (    bd1) * (    cd1);
        const scalar_t w0111_gd1 = - (    rd1) * (    bd1) * (    cd1);
        const scalar_t w1111_gd1 =   (    rd1) * (    bd1) * (    cd1);

        // bd1系列
        const scalar_t w0000_bd1 = - (1 - rd1) * (1 - gd1) * (1 - cd1);
        const scalar_t w1000_bd1 =   (1 - rd1) * (1 - gd1) * (1 - cd1);
        const scalar_t w0100_bd1 = - (    rd1) * (1 - gd1) * (1 - cd1);
        const scalar_t w0010_bd1 = - (1 - rd1) * (    gd1) * (1 - cd1);
        const scalar_t w0001_bd1 = - (1 - rd1) * (1 - gd1) * (    cd1);
        const scalar_t w1100_bd1 =   (    rd1) * (1 - gd1) * (1 - cd1);
        const scalar_t w1010_bd1 =   (1 - rd1) * (    gd1) * (1 - cd1);
        const scalar_t w1001_bd1 =   (1 - rd1) * (1 - gd1) * (    cd1);
        const scalar_t w0110_bd1 = - (    rd1) * (    gd1) * (1 - cd1);
        const scalar_t w0101_bd1 = - (    rd1) * (1 - gd1) * (    cd1);
        const scalar_t w0011_bd1 = - (1 - rd1) * (    gd1) * (    cd1);
        const scalar_t w1110_bd1 =   (    rd1) * (    gd1) * (1 - cd1);
        const scalar_t w1101_bd1 =   (    rd1) * (1 - gd1) * (    cd1);
        const scalar_t w1011_bd1 =   (1 - rd1) * (    gd1) * (    cd1);
        const scalar_t w0111_bd1 = - (    rd1) * (    gd1) * (    cd1);
        const scalar_t w1111_bd1 =   (    rd1) * (    gd1) * (    cd1);

        // cd1系列
        const scalar_t w0000_cd1 = - (1 - rd1) * (1 - gd1) * (1 - bd1);
        const scalar_t w1000_cd1 =   (1 - rd1) * (1 - gd1) * (1 - bd1);
        const scalar_t w0100_cd1 = - (    rd1) * (1 - gd1) * (1 - bd1);
        const scalar_t w0010_cd1 = - (1 - rd1) * (    gd1) * (1 - bd1);
        const scalar_t w0001_cd1 = - (1 - rd1) * (1 - gd1) * (    bd1);
        const scalar_t w1100_cd1 =   (    rd1) * (1 - gd1) * (1 - bd1);
        const scalar_t w1010_cd1 =   (1 - rd1) * (    gd1) * (1 - bd1);
        const scalar_t w1001_cd1 =   (1 - rd1) * (1 - gd1) * (    bd1);
        const scalar_t w0110_cd1 = - (    rd1) * (    gd1) * (1 - bd1);
        const scalar_t w0101_cd1 = - (    rd1) * (1 - gd1) * (    bd1);
        const scalar_t w0011_cd1 = - (1 - rd1) * (    gd1) * (    bd1);
        const scalar_t w1110_cd1 =   (    rd1) * (    gd1) * (1 - bd1);
        const scalar_t w1101_cd1 =   (    rd1) * (1 - gd1) * (    bd1);
        const scalar_t w1011_cd1 =   (1 - rd1) * (    gd1) * (    bd1);
        const scalar_t w0111_cd1 = - (    rd1) * (    gd1) * (    bd1);
        const scalar_t w1111_cd1 =   (    rd1) * (    gd1) * (    bd1);

        // rd2系列
        const scalar_t w0000_rd2 = - (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w1000_rd2 =   (1 - gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w0100_rd2 = - (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w0010_rd2 = - (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t w0001_rd2 = - (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t w1100_rd2 =   (    gd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w1010_rd2 =   (1 - gd2) * (    bd2) * (1 - cd2);
        const scalar_t w1001_rd2 =   (1 - gd2) * (1 - bd2) * (    cd2);
        const scalar_t w0110_rd2 = - (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t w0101_rd2 = - (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t w0011_rd2 = - (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t w1110_rd2 =   (    gd2) * (    bd2) * (1 - cd2);
        const scalar_t w1101_rd2 =   (    gd2) * (1 - bd2) * (    cd2);
        const scalar_t w1011_rd2 =   (1 - gd2) * (    bd2) * (    cd2);
        const scalar_t w0111_rd2 = - (    gd2) * (    bd2) * (    cd2);
        const scalar_t w1111_rd2 =   (    gd2) * (    bd2) * (    cd2);

        // gd2系列
        const scalar_t w0000_gd2 = - (1 - rd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w1000_gd2 =   (1 - rd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w0100_gd2 = - (    rd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w0010_gd2 = - (1 - rd2) * (    bd2) * (1 - cd2);
        const scalar_t w0001_gd2 = - (1 - rd2) * (1 - bd2) * (    cd2);
        const scalar_t w1100_gd2 =   (    rd2) * (1 - bd2) * (1 - cd2);
        const scalar_t w1010_gd2 =   (1 - rd2) * (    bd2) * (1 - cd2);
        const scalar_t w1001_gd2 =   (1 - rd2) * (1 - bd2) * (    cd2);
        const scalar_t w0110_gd2 = - (    rd2) * (    bd2) * (1 - cd2);
        const scalar_t w0101_gd2 = - (    rd2) * (1 - bd2) * (    cd2);
        const scalar_t w0011_gd2 = - (1 - rd2) * (    bd2) * (    cd2);
        const scalar_t w1110_gd2 =   (    rd2) * (    bd2) * (1 - cd2);
        const scalar_t w1101_gd2 =   (    rd2) * (1 - bd2) * (    cd2);
        const scalar_t w1011_gd2 =   (1 - rd2) * (    bd2) * (    cd2);
        const scalar_t w0111_gd2 = - (    rd2) * (    bd2) * (    cd2);
        const scalar_t w1111_gd2 =   (    rd2) * (    bd2) * (    cd2);

        // bd2系列
        const scalar_t w0000_bd2 = - (1 - rd2) * (1 - gd2) * (1 - cd2);
        const scalar_t w1000_bd2 =   (1 - rd2) * (1 - gd2) * (1 - cd2);
        const scalar_t w0100_bd2 = - (    rd2) * (1 - gd2) * (1 - cd2);
        const scalar_t w0010_bd2 = - (1 - rd2) * (    gd2) * (1 - cd2);
        const scalar_t w0001_bd2 = - (1 - rd2) * (1 - gd2) * (    cd2);
        const scalar_t w1100_bd2 =   (    rd2) * (1 - gd2) * (1 - cd2);
        const scalar_t w1010_bd2 =   (1 - rd2) * (    gd2) * (1 - cd2);
        const scalar_t w1001_bd2 =   (1 - rd2) * (1 - gd2) * (    cd2);
        const scalar_t w0110_bd2 = - (    rd2) * (    gd2) * (1 - cd2);
        const scalar_t w0101_bd2 = - (    rd2) * (1 - gd2) * (    cd2);
        const scalar_t w0011_bd2 = - (1 - rd2) * (    gd2) * (    cd2);
        const scalar_t w1110_bd2 =   (    rd2) * (    gd2) * (1 - cd2);
        const scalar_t w1101_bd2 =   (    rd2) * (1 - gd2) * (    cd2);
        const scalar_t w1011_bd2 =   (1 - rd2) * (    gd2) * (    cd2);
        const scalar_t w0111_bd2 = - (    rd2) * (    gd2) * (    cd2);
        const scalar_t w1111_bd2 =   (    rd2) * (    gd2) * (    cd2);

        // cd2系列
        const scalar_t w0000_cd2 = - (1 - rd2) * (1 - gd2) * (1 - bd2);
        const scalar_t w1000_cd2 =   (1 - rd2) * (1 - gd2) * (1 - bd2);
        const scalar_t w0100_cd2 = - (    rd2) * (1 - gd2) * (1 - bd2);
        const scalar_t w0010_cd2 = - (1 - rd2) * (    gd2) * (1 - bd2);
        const scalar_t w0001_cd2 = - (1 - rd2) * (1 - gd2) * (    bd2);
        const scalar_t w1100_cd2 =   (    rd2) * (1 - gd2) * (1 - bd2);
        const scalar_t w1010_cd2 =   (1 - rd2) * (    gd2) * (1 - bd2);
        const scalar_t w1001_cd2 =   (1 - rd2) * (1 - gd2) * (    bd2);
        const scalar_t w0110_cd2 = - (    rd2) * (    gd2) * (1 - bd2);
        const scalar_t w0101_cd2 = - (    rd2) * (1 - gd2) * (    bd2);
        const scalar_t w0011_cd2 = - (1 - rd2) * (    gd2) * (    bd2);
        const scalar_t w1110_cd2 =   (    rd2) * (    gd2) * (1 - bd2);
        const scalar_t w1101_cd2 =   (    rd2) * (1 - gd2) * (    bd2);
        const scalar_t w1011_cd2 =   (1 - rd2) * (    gd2) * (    bd2);
        const scalar_t w0111_cd2 = - (    rd2) * (    gd2) * (    bd2);
        const scalar_t w1111_cd2 =   (    rd2) * (    gd2) * (    bd2);

        // rd3系列
        const scalar_t w0000_rd3 = - (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w1000_rd3 =   (1 - gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w0100_rd3 = - (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w0010_rd3 = - (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t w0001_rd3 = - (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t w1100_rd3 =   (    gd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w1010_rd3 =   (1 - gd3) * (    bd3) * (1 - cd3);
        const scalar_t w1001_rd3 =   (1 - gd3) * (1 - bd3) * (    cd3);
        const scalar_t w0110_rd3 = - (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t w0101_rd3 = - (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t w0011_rd3 = - (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t w1110_rd3 =   (    gd3) * (    bd3) * (1 - cd3);
        const scalar_t w1101_rd3 =   (    gd3) * (1 - bd3) * (    cd3);
        const scalar_t w1011_rd3 =   (1 - gd3) * (    bd3) * (    cd3);
        const scalar_t w0111_rd3 = - (    gd3) * (    bd3) * (    cd3);
        const scalar_t w1111_rd3 =   (    gd3) * (    bd3) * (    cd3);

        // gd3系列
        const scalar_t w0000_gd3 = - (1 - rd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w1000_gd3 =   (1 - rd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w0100_gd3 = - (    rd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w0010_gd3 = - (1 - rd3) * (    bd3) * (1 - cd3);
        const scalar_t w0001_gd3 = - (1 - rd3) * (1 - bd3) * (    cd3);
        const scalar_t w1100_gd3 =   (    rd3) * (1 - bd3) * (1 - cd3);
        const scalar_t w1010_gd3 =   (1 - rd3) * (    bd3) * (1 - cd3);
        const scalar_t w1001_gd3 =   (1 - rd3) * (1 - bd3) * (    cd3);
        const scalar_t w0110_gd3 = - (    rd3) * (    bd3) * (1 - cd3);
        const scalar_t w0101_gd3 = - (    rd3) * (1 - bd3) * (    cd3);
        const scalar_t w0011_gd3 = - (1 - rd3) * (    bd3) * (    cd3);
        const scalar_t w1110_gd3 =   (    rd3) * (    bd3) * (1 - cd3);
        const scalar_t w1101_gd3 =   (    rd3) * (1 - bd3) * (    cd3);
        const scalar_t w1011_gd3 =   (1 - rd3) * (    bd3) * (    cd3);
        const scalar_t w0111_gd3 = - (    rd3) * (    bd3) * (    cd3);
        const scalar_t w1111_gd3 =   (    rd3) * (    bd3) * (    cd3);

        // bd3系列
        const scalar_t w0000_bd3 = - (1 - rd3) * (1 - gd3) * (1 - cd3);
        const scalar_t w1000_bd3 =   (1 - rd3) * (1 - gd3) * (1 - cd3);
        const scalar_t w0100_bd3 = - (    rd3) * (1 - gd3) * (1 - cd3);
        const scalar_t w0010_bd3 = - (1 - rd3) * (    gd3) * (1 - cd3);
        const scalar_t w0001_bd3 = - (1 - rd3) * (1 - gd3) * (    cd3);
        const scalar_t w1100_bd3 =   (    rd3) * (1 - gd3) * (1 - cd3);
        const scalar_t w1010_bd3 =   (1 - rd3) * (    gd3) * (1 - cd3);
        const scalar_t w1001_bd3 =   (1 - rd3) * (1 - gd3) * (    cd3);
        const scalar_t w0110_bd3 = - (    rd3) * (    gd3) * (1 - cd3);
        const scalar_t w0101_bd3 = - (    rd3) * (1 - gd3) * (    cd3);
        const scalar_t w0011_bd3 = - (1 - rd3) * (    gd3) * (    cd3);
        const scalar_t w1110_bd3 =   (    rd3) * (    gd3) * (1 - cd3);
        const scalar_t w1101_bd3 =   (    rd3) * (1 - gd3) * (    cd3);
        const scalar_t w1011_bd3 =   (1 - rd3) * (    gd3) * (    cd3);
        const scalar_t w0111_bd3 = - (    rd3) * (    gd3) * (    cd3);
        const scalar_t w1111_bd3 =   (    rd3) * (    gd3) * (    cd3);

        // cd3系列
        const scalar_t w0000_cd3 = - (1 - rd3) * (1 - gd3) * (1 - bd3);
        const scalar_t w1000_cd3 =   (1 - rd3) * (1 - gd3) * (1 - bd3);
        const scalar_t w0100_cd3 = - (    rd3) * (1 - gd3) * (1 - bd3);
        const scalar_t w0010_cd3 = - (1 - rd3) * (    gd3) * (1 - bd3);
        const scalar_t w0001_cd3 = - (1 - rd3) * (1 - gd3) * (    bd3);
        const scalar_t w1100_cd3 =   (    rd3) * (1 - gd3) * (1 - bd3);
        const scalar_t w1010_cd3 =   (1 - rd3) * (    gd3) * (1 - bd3);
        const scalar_t w1001_cd3 =   (1 - rd3) * (1 - gd3) * (    bd3);
        const scalar_t w0110_cd3 = - (    rd3) * (    gd3) * (1 - bd3);
        const scalar_t w0101_cd3 = - (    rd3) * (1 - gd3) * (    bd3);
        const scalar_t w0011_cd3 = - (1 - rd3) * (    gd3) * (    bd3);
        const scalar_t w1110_cd3 =   (    rd3) * (    gd3) * (1 - bd3);
        const scalar_t w1101_cd3 =   (    rd3) * (1 - gd3) * (    bd3);
        const scalar_t w1011_cd3 =   (1 - rd3) * (    gd3) * (    bd3);
        const scalar_t w0111_cd3 = - (    rd3) * (    gd3) * (    bd3);
        const scalar_t w1111_cd3 =   (    rd3) * (    gd3) * (    bd3);

        const scalar_t w0000_rd4 = - (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w1000_rd4 =   (1 - gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w0100_rd4 = - (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w0010_rd4 = - (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t w0001_rd4 = - (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t w1100_rd4 =   (    gd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w1010_rd4 =   (1 - gd4) * (    bd4) * (1 - cd4);
        const scalar_t w1001_rd4 =   (1 - gd4) * (1 - bd4) * (    cd4);
        const scalar_t w0110_rd4 = - (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t w0101_rd4 = - (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t w0011_rd4 = - (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t w1110_rd4 =   (    gd4) * (    bd4) * (1 - cd4);
        const scalar_t w1101_rd4 =   (    gd4) * (1 - bd4) * (    cd4);
        const scalar_t w1011_rd4 =   (1 - gd4) * (    bd4) * (    cd4);
        const scalar_t w0111_rd4 = - (    gd4) * (    bd4) * (    cd4);
        const scalar_t w1111_rd4 =   (    gd4) * (    bd4) * (    cd4);

        const scalar_t w0000_gd4 = - (1 - rd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w1000_gd4 =   (1 - rd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w0100_gd4 = - (    rd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w0010_gd4 = - (1 - rd4) * (    bd4) * (1 - cd4);
        const scalar_t w0001_gd4 = - (1 - rd4) * (1 - bd4) * (    cd4);
        const scalar_t w1100_gd4 =   (    rd4) * (1 - bd4) * (1 - cd4);
        const scalar_t w1010_gd4 =   (1 - rd4) * (    bd4) * (1 - cd4);
        const scalar_t w1001_gd4 =   (1 - rd4) * (1 - bd4) * (    cd4);
        const scalar_t w0110_gd4 = - (    rd4) * (    bd4) * (1 - cd4);
        const scalar_t w0101_gd4 = - (    rd4) * (1 - bd4) * (    cd4);
        const scalar_t w0011_gd4 = - (1 - rd4) * (    bd4) * (    cd4);
        const scalar_t w1110_gd4 =   (    rd4) * (    bd4) * (1 - cd4);
        const scalar_t w1101_gd4 =   (    rd4) * (1 - bd4) * (    cd4);
        const scalar_t w1011_gd4 =   (1 - rd4) * (    bd4) * (    cd4);
        const scalar_t w0111_gd4 = - (    rd4) * (    bd4) * (    cd4);
        const scalar_t w1111_gd4 =   (    rd4) * (    bd4) * (    cd4);

        // bd4系列
        const scalar_t w0000_bd4 = - (1 - rd4) * (1 - gd4) * (1 - cd4);
        const scalar_t w1000_bd4 =   (1 - rd4) * (1 - gd4) * (1 - cd4);
        const scalar_t w0100_bd4 = - (    rd4) * (1 - gd4) * (1 - cd4);
        const scalar_t w0010_bd4 = - (1 - rd4) * (    gd4) * (1 - cd4);
        const scalar_t w0001_bd4 = - (1 - rd4) * (1 - gd4) * (    cd4);
        const scalar_t w1100_bd4 =   (    rd4) * (1 - gd4) * (1 - cd4);
        const scalar_t w1010_bd4 =   (1 - rd4) * (    gd4) * (1 - cd4);
        const scalar_t w1001_bd4 =   (1 - rd4) * (1 - gd4) * (    cd4);
        const scalar_t w0110_bd4 = - (    rd4) * (    gd4) * (1 - cd4);
        const scalar_t w0101_bd4 = - (    rd4) * (1 - gd4) * (    cd4);
        const scalar_t w0011_bd4 = - (1 - rd4) * (    gd4) * (    cd4);
        const scalar_t w1110_bd4 =   (    rd4) * (    gd4) * (1 - cd4);
        const scalar_t w1101_bd4 =   (    rd4) * (1 - gd4) * (    cd4);
        const scalar_t w1011_bd4 =   (1 - rd4) * (    gd4) * (    cd4);
        const scalar_t w0111_bd4 = - (    rd4) * (    gd4) * (    cd4);
        const scalar_t w1111_bd4 =   (    rd4) * (    gd4) * (    cd4);

        // cd4系列
        const scalar_t w0000_cd4 = - (1 - rd4) * (1 - gd4) * (1 - bd4);
        const scalar_t w1000_cd4 =   (1 - rd4) * (1 - gd4) * (1 - bd4);
        const scalar_t w0100_cd4 = - (    rd4) * (1 - gd4) * (1 - bd4);
        const scalar_t w0010_cd4 = - (1 - rd4) * (    gd4) * (1 - bd4);
        const scalar_t w0001_cd4 = - (1 - rd4) * (1 - gd4) * (    bd4);
        const scalar_t w1100_cd4 =   (    rd4) * (1 - gd4) * (1 - bd4);
        const scalar_t w1010_cd4 =   (1 - rd4) * (    gd4) * (1 - bd4);
        const scalar_t w1001_cd4 =   (1 - rd4) * (1 - gd4) * (    bd4);
        const scalar_t w0110_cd4 = - (    rd4) * (    gd4) * (1 - bd4);
        const scalar_t w0101_cd4 = - (    rd4) * (1 - gd4) * (    bd4);
        const scalar_t w0011_cd4 = - (1 - rd4) * (    gd4) * (    bd4);
        const scalar_t w1110_cd4 =   (    rd4) * (    gd4) * (1 - bd4);
        const scalar_t w1101_cd4 =   (    rd4) * (1 - gd4) * (    bd4);
        const scalar_t w1011_cd4 =   (1 - rd4) * (    gd4) * (    bd4);
        const scalar_t w0111_cd4 = - (    rd4) * (    gd4) * (    bd4);
        const scalar_t w1111_cd4 =   (    rd4) * (    gd4) * (    bd4);
        
            scalar_t grad_o_0 = grad_output[index];
            scalar_t grad_o_1 = grad_output[index + height * width * 1];
            scalar_t grad_o_2 = grad_output[index + height * width * 2];
            scalar_t grad_o_3 = grad_output[index + height * width * 3];
            scalar_t grad_o_4 = grad_output[index + height * width * 4];

            /* compute gradient of lut */
            atomicAdd(grad_lut + _0id0000, grad_o_0 * _0w0000);
            atomicAdd(grad_lut + _0id1000, grad_o_0 * _0w1000);
            atomicAdd(grad_lut + _0id0100, grad_o_0 * _0w0100);
            atomicAdd(grad_lut + _0id0010, grad_o_0 * _0w0010);
            atomicAdd(grad_lut + _0id0001, grad_o_0 * _0w0001);
            atomicAdd(grad_lut + _0id1100, grad_o_0 * _0w1100);
            atomicAdd(grad_lut + _0id1010, grad_o_0 * _0w1010);
            atomicAdd(grad_lut + _0id1001, grad_o_0 * _0w1001);
            atomicAdd(grad_lut + _0id0110, grad_o_0 * _0w0110);
            atomicAdd(grad_lut + _0id0101, grad_o_0 * _0w0101);
            atomicAdd(grad_lut + _0id0011, grad_o_0 * _0w0011);
            atomicAdd(grad_lut + _0id1110, grad_o_0 * _0w1110);
            atomicAdd(grad_lut + _0id1101, grad_o_0 * _0w1101);
            atomicAdd(grad_lut + _0id1011, grad_o_0 * _0w1011);
            atomicAdd(grad_lut + _0id0111, grad_o_0 * _0w0111);
            atomicAdd(grad_lut + _0id1111, grad_o_0 * _0w1111);
            
            atomicAdd(grad_lut + _1id0000 + stride_lut_4 * 1, grad_o_1 * _1w0000);
            atomicAdd(grad_lut + _1id1000 + stride_lut_4 * 1, grad_o_1 * _1w1000);
            atomicAdd(grad_lut + _1id0100 + stride_lut_4 * 1, grad_o_1 * _1w0100);
            atomicAdd(grad_lut + _1id0010 + stride_lut_4 * 1, grad_o_1 * _1w0010);
            atomicAdd(grad_lut + _1id0001 + stride_lut_4 * 1, grad_o_1 * _1w0001);
            atomicAdd(grad_lut + _1id1100 + stride_lut_4 * 1, grad_o_1 * _1w1100);
            atomicAdd(grad_lut + _1id1010 + stride_lut_4 * 1, grad_o_1 * _1w1010);
            atomicAdd(grad_lut + _1id1001 + stride_lut_4 * 1, grad_o_1 * _1w1001);
            atomicAdd(grad_lut + _1id0110 + stride_lut_4 * 1, grad_o_1 * _1w0110);
            atomicAdd(grad_lut + _1id0101 + stride_lut_4 * 1, grad_o_1 * _1w0101);
            atomicAdd(grad_lut + _1id0011 + stride_lut_4 * 1, grad_o_1 * _1w0011);
            atomicAdd(grad_lut + _1id1110 + stride_lut_4 * 1, grad_o_1 * _1w1110);
            atomicAdd(grad_lut + _1id1101 + stride_lut_4 * 1, grad_o_1 * _1w1101);
            atomicAdd(grad_lut + _1id1011 + stride_lut_4 * 1, grad_o_1 * _1w1011);
            atomicAdd(grad_lut + _1id0111 + stride_lut_4 * 1, grad_o_1 * _1w0111);
            atomicAdd(grad_lut + _1id1111 + stride_lut_4 * 1, grad_o_1 * _1w1111);
            
            atomicAdd(grad_lut + _2id0000 + stride_lut_4 * 2, grad_o_2 * _2w0000);
            atomicAdd(grad_lut + _2id1000 + stride_lut_4 * 2, grad_o_2 * _2w1000);
            atomicAdd(grad_lut + _2id0100 + stride_lut_4 * 2, grad_o_2 * _2w0100);
            atomicAdd(grad_lut + _2id0010 + stride_lut_4 * 2, grad_o_2 * _2w0010);
            atomicAdd(grad_lut + _2id0001 + stride_lut_4 * 2, grad_o_2 * _2w0001);
            atomicAdd(grad_lut + _2id1100 + stride_lut_4 * 2, grad_o_2 * _2w1100);
            atomicAdd(grad_lut + _2id1010 + stride_lut_4 * 2, grad_o_2 * _2w1010);
            atomicAdd(grad_lut + _2id1001 + stride_lut_4 * 2, grad_o_2 * _2w1001);
            atomicAdd(grad_lut + _2id0110 + stride_lut_4 * 2, grad_o_2 * _2w0110);
            atomicAdd(grad_lut + _2id0101 + stride_lut_4 * 2, grad_o_2 * _2w0101);
            atomicAdd(grad_lut + _2id0011 + stride_lut_4 * 2, grad_o_2 * _2w0011);
            atomicAdd(grad_lut + _2id1110 + stride_lut_4 * 2, grad_o_2 * _2w1110);
            atomicAdd(grad_lut + _2id1101 + stride_lut_4 * 2, grad_o_2 * _2w1101);
            atomicAdd(grad_lut + _2id1011 + stride_lut_4 * 2, grad_o_2 * _2w1011);
            atomicAdd(grad_lut + _2id0111 + stride_lut_4 * 2, grad_o_2 * _2w0111);
            atomicAdd(grad_lut + _2id1111 + stride_lut_4 * 2, grad_o_2 * _2w1111);

            atomicAdd(grad_lut + _3id0000 + stride_lut_4 * 3, grad_o_3 * _3w0000);
            atomicAdd(grad_lut + _3id1000 + stride_lut_4 * 3, grad_o_3 * _3w1000);
            atomicAdd(grad_lut + _3id0100 + stride_lut_4 * 3, grad_o_3 * _3w0100);
            atomicAdd(grad_lut + _3id0010 + stride_lut_4 * 3, grad_o_3 * _3w0010);
            atomicAdd(grad_lut + _3id0001 + stride_lut_4 * 3, grad_o_3 * _3w0001);
            atomicAdd(grad_lut + _3id1100 + stride_lut_4 * 3, grad_o_3 * _3w1100);
            atomicAdd(grad_lut + _3id1010 + stride_lut_4 * 3, grad_o_3 * _3w1010);
            atomicAdd(grad_lut + _3id1001 + stride_lut_4 * 3, grad_o_3 * _3w1001);   
            atomicAdd(grad_lut + _3id0110 + stride_lut_4 * 3, grad_o_3 * _3w0110);
            atomicAdd(grad_lut + _3id0101 + stride_lut_4 * 3, grad_o_3 * _3w0101);
            atomicAdd(grad_lut + _3id0011 + stride_lut_4 * 3, grad_o_3 * _3w0011);
            atomicAdd(grad_lut + _3id1110 + stride_lut_4 * 3, grad_o_3 * _3w1110);
            atomicAdd(grad_lut + _3id1101 + stride_lut_4 * 3, grad_o_3 * _3w1101);
            atomicAdd(grad_lut + _3id1011 + stride_lut_4 * 3, grad_o_3 * _3w1011);
            atomicAdd(grad_lut + _3id0111 + stride_lut_4 * 3, grad_o_3 * _3w0111);
            atomicAdd(grad_lut + _3id1111 + stride_lut_4 * 3, grad_o_3 * _3w1111);

            atomicAdd(grad_lut + _4id0000 + stride_lut_4 * 4, grad_o_4 * _4w0000);
            atomicAdd(grad_lut + _4id1000 + stride_lut_4 * 4, grad_o_4 * _4w1000);
            atomicAdd(grad_lut + _4id0100 + stride_lut_4 * 4, grad_o_4 * _4w0100);
            atomicAdd(grad_lut + _4id0010 + stride_lut_4 * 4, grad_o_4 * _4w0010);
            atomicAdd(grad_lut + _4id0001 + stride_lut_4 * 4, grad_o_4 * _4w0001);
            atomicAdd(grad_lut + _4id1100 + stride_lut_4 * 4, grad_o_4 * _4w1100);
            atomicAdd(grad_lut + _4id1010 + stride_lut_4 * 4, grad_o_4 * _4w1010);    
            atomicAdd(grad_lut + _4id1001 + stride_lut_4 * 4, grad_o_4 * _4w1001);
            atomicAdd(grad_lut + _4id0110 + stride_lut_4 * 4, grad_o_4 * _4w0110);
            atomicAdd(grad_lut + _4id0101 + stride_lut_4 * 4, grad_o_4 * _4w0101);
            atomicAdd(grad_lut + _4id0011 + stride_lut_4 * 4, grad_o_4 * _4w0011);
            atomicAdd(grad_lut + _4id1110 + stride_lut_4 * 4, grad_o_4 * _4w1110);
            atomicAdd(grad_lut + _4id1101 + stride_lut_4 * 4, grad_o_4 * _4w1101);
            atomicAdd(grad_lut + _4id1011 + stride_lut_4 * 4, grad_o_4 * _4w1011);
            atomicAdd(grad_lut + _4id0111 + stride_lut_4 * 4, grad_o_4 * _4w0111);
            atomicAdd(grad_lut + _4id1111 + stride_lut_4 * 4, grad_o_4 * _4w1111);

            

            /* compute gradient of vertices */
            scalar_t grad_d0 = 0;
            const scalar_t _0lut0000 = data_lut[_0id0000];
            const scalar_t _0lut1000 = data_lut[_0id1000];
            const scalar_t _0lut0100 = data_lut[_0id0100];
            const scalar_t _0lut0010 = data_lut[_0id0010];
            const scalar_t _0lut0001 = data_lut[_0id0001];
            const scalar_t _0lut1100 = data_lut[_0id1100];
            const scalar_t _0lut1010 = data_lut[_0id1010];
            const scalar_t _0lut1001 = data_lut[_0id1001];
            const scalar_t _0lut0110 = data_lut[_0id0110];
            const scalar_t _0lut0101 = data_lut[_0id0101];
            const scalar_t _0lut0011 = data_lut[_0id0011];
            const scalar_t _0lut1110 = data_lut[_0id1110];
            const scalar_t _0lut1101 = data_lut[_0id1101];
            const scalar_t _0lut1011 = data_lut[_0id1011];
            const scalar_t _0lut0111 = data_lut[_0id0111];
            const scalar_t _0lut1111 = data_lut[_0id1111];
            grad_d0 = grad_o_0 *
                (w0000_rd0 * _0lut0000 + w1000_rd0 * _0lut1000 + w0100_rd0 * _0lut0100 + w0010_rd0 * _0lut0010 +
                w0001_rd0 * _0lut0001 + w1100_rd0 * _0lut1100 + w1010_rd0 * _0lut1010 + w1001_rd0 * _0lut1001 +
                w0110_rd0 * _0lut0110 + w0101_rd0 * _0lut0101 + w0011_rd0 * _0lut0011 + w1110_rd0 * _0lut1110 +
                w1101_rd0 * _0lut1101 + w1011_rd0 * _0lut1011 + w0111_rd0 * _0lut0111 + w1111_rd0 * _0lut1111);
            // r
            atomicAdd(grad_inp + index, grad_d0 * 1 / size_bin);

            grad_d0 = grad_o_0 *
                (w0000_gd0 * _0lut0000 + w1000_gd0 * _0lut1000 + w0100_gd0 * _0lut0100 + w0010_gd0 * _0lut0010 +
                w0001_gd0 * _0lut0001 + w1100_gd0 * _0lut1100 + w1010_gd0 * _0lut1010 + w1001_gd0 * _0lut1001 +
                w0110_gd0 * _0lut0110 + w0101_gd0 * _0lut0101 + w0011_gd0 * _0lut0011 + w1110_gd0 * _0lut1110 +
                w1101_gd0 * _0lut1101 + w1011_gd0 * _0lut1011 + w0111_gd0 * _0lut0111 + w1111_gd0 * _0lut1111);
            // g
            atomicAdd(grad_inp + index + (x >= width - 1 ? -1 : 1), grad_d0 * 1 / size_bin);

            grad_d0 = grad_o_0 *
                (w0000_bd0 * _0lut0000 + w1000_bd0 * _0lut1000 + w0100_bd0 * _0lut0100 + w0010_bd0 * _0lut0010 +
                w0001_bd0 * _0lut0001 + w1100_bd0 * _0lut1100 + w1010_bd0 * _0lut1010 + w1001_bd0 * _0lut1001 +
                w0110_bd0 * _0lut0110 + w0101_bd0 * _0lut0101 + w0011_bd0 * _0lut0011 + w1110_bd0 * _0lut1110 +
                w1101_bd0 * _0lut1101 + w1011_bd0 * _0lut1011 + w0111_bd0 * _0lut0111 + w1111_bd0 * _0lut1111);
            // b
            atomicAdd(grad_inp + index + (y >= height - 1 ? -width : width), grad_d0 * 1 / size_bin);

            grad_d0 = grad_o_0 *
                (w0000_cd0 * _0lut0000 + w1000_cd0 * _0lut1000 + w0100_cd0 * _0lut0100 + w0010_cd0 * _0lut0010 +
                w0001_cd0 * _0lut0001 + w1100_cd0 * _0lut1100 + w1010_cd0 * _0lut1010 + w1001_cd0 * _0lut1001 +
                w0110_cd0 * _0lut0110 + w0101_cd0 * _0lut0101 + w0011_cd0 * _0lut0011 + w1110_cd0 * _0lut1110 +
                w1101_cd0 * _0lut1101 + w1011_cd0 * _0lut1011 + w0111_cd0 * _0lut0111 + w1111_cd0 * _0lut1111);
            // c
            atomicAdd(grad_inp + index + 
    (y >= height - 1 ? -width : width) + 
    (x >= width - 1 ? -1 : 1), grad_d0 * 1 / size_bin);

    scalar_t grad_d1 = 0;
            const scalar_t _1lut0000 = data_lut[_1id0000 + stride_lut_4 * 1];
            const scalar_t _1lut1000 = data_lut[_1id1000 + stride_lut_4 * 1];
            const scalar_t _1lut0100 = data_lut[_1id0100 + stride_lut_4 * 1];
            const scalar_t _1lut0010 = data_lut[_1id0010 + stride_lut_4 * 1];
            const scalar_t _1lut0001 = data_lut[_1id0001 + stride_lut_4 * 1];
            const scalar_t _1lut1100 = data_lut[_1id1100 + stride_lut_4 * 1];
            const scalar_t _1lut1010 = data_lut[_1id1010 + stride_lut_4 * 1];
            const scalar_t _1lut1001 = data_lut[_1id1001 + stride_lut_4 * 1];
            const scalar_t _1lut0110 = data_lut[_1id0110 + stride_lut_4 * 1];
            const scalar_t _1lut0101 = data_lut[_1id0101 + stride_lut_4 * 1];
            const scalar_t _1lut0011 = data_lut[_1id0011 + stride_lut_4 * 1];
            const scalar_t _1lut1110 = data_lut[_1id1110 + stride_lut_4 * 1];
            const scalar_t _1lut1101 = data_lut[_1id1101 + stride_lut_4 * 1];
            const scalar_t _1lut1011 = data_lut[_1id1011 + stride_lut_4 * 1];
            const scalar_t _1lut0111 = data_lut[_1id0111 + stride_lut_4 * 1];
            const scalar_t _1lut1111 = data_lut[_1id1111 + stride_lut_4 * 1];
            grad_d1 = grad_o_1 *
                (w0000_rd1 * _1lut0000 + w1000_rd1 * _1lut1000 + w0100_rd1 * _1lut0100 + w0010_rd1 * _1lut0010 +
                w0001_rd1 * _1lut0001 + w1100_rd1 * _1lut1100 + w1010_rd1 * _1lut1010 + w1001_rd1 * _1lut1001 +
                w0110_rd1 * _1lut0110 + w0101_rd1 * _1lut0101 + w0011_rd1 * _1lut0011 + w1110_rd1 * _1lut1110 +
                w1101_rd1 * _1lut1101 + w1011_rd1 * _1lut1011 + w0111_rd1 * _1lut0111 + w1111_rd1 * _1lut1111);
            // r
            atomicAdd(grad_inp + index + height * width * 1, grad_d1 * 1 / size_bin);

            grad_d1 = grad_o_1 *
                (w0000_gd1 * _1lut0000 + w1000_gd1 * _1lut1000 + w0100_gd1 * _1lut0100 + w0010_gd1 * _1lut0010 +
                w0001_gd1 * _1lut0001 + w1100_gd1 * _1lut1100 + w1010_gd1 * _1lut1010 + w1001_gd1 * _1lut1001 +
                w0110_gd1 * _1lut0110 + w0101_gd1 * _1lut0101 + w0011_gd1 * _1lut0011 + w1110_gd1 * _1lut1110 +
                w1101_gd1 * _1lut1101 + w1011_gd1 * _1lut1011 + w0111_gd1 * _1lut0111 + w1111_gd1 * _1lut1111);
            // g
            atomicAdd(grad_inp + index + height * width * 1 + (x >= width - 1 ? -1 : 1), grad_d1 * 1 / size_bin);

            grad_d1 = grad_o_1 *
                (w0000_bd1 * _1lut0000 + w1000_bd1 * _1lut1000 + w0100_bd1 * _1lut0100 + w0010_bd1 * _1lut0010 +
                w0001_bd1 * _1lut0001 + w1100_bd1 * _1lut1100 + w1010_bd1 * _1lut1010 + w1001_bd1 * _1lut1001 +
                w0110_bd1 * _1lut0110 + w0101_bd1 * _1lut0101 + w0011_bd1 * _1lut0011 + w1110_bd1 * _1lut1110 +
                w1101_bd1 * _1lut1101 + w1011_bd1 * _1lut1011 + w0111_bd1 * _1lut0111 + w1111_bd1 * _1lut1111);
            // b
            atomicAdd(grad_inp + index + height * width * 1 + (y >= height - 1 ? -width : width), grad_d1 * 1 / size_bin);

            grad_d1 = grad_o_1 *
                (w0000_cd1 * _1lut0000 + w1000_cd1 * _1lut1000 + w0100_cd1 * _1lut0100 + w0010_cd1 * _1lut0010 +
                w0001_cd1 * _1lut0001 + w1100_cd1 * _1lut1100 + w1010_cd1 * _1lut1010 + w1001_cd1 * _1lut1001 +
                w0110_cd1 * _1lut0110 + w0101_cd1 * _1lut0101 + w0011_cd1 * _1lut0011 + w1110_cd1 * _1lut1110 +
                w1101_cd1 * _1lut1101 + w1011_cd1 * _1lut1011 + w0111_cd1 * _1lut0111 + w1111_cd1 * _1lut1111);
            // c
            atomicAdd(grad_inp + index + height * width * 1 + 
    (y >= height - 1 ? -width : width) + 
    (x >= width - 1 ? -1 : 1), grad_d1 * 1 / size_bin);

    scalar_t grad_d2 = 0;
            const scalar_t _2lut0000 = data_lut[_2id0000 + stride_lut_4 * 2];
            const scalar_t _2lut1000 = data_lut[_2id1000 + stride_lut_4 * 2];
            const scalar_t _2lut0100 = data_lut[_2id0100 + stride_lut_4 * 2];
            const scalar_t _2lut0010 = data_lut[_2id0010 + stride_lut_4 * 2];
            const scalar_t _2lut0001 = data_lut[_2id0001 + stride_lut_4 * 2];
            const scalar_t _2lut1100 = data_lut[_2id1100 + stride_lut_4 * 2];
            const scalar_t _2lut1010 = data_lut[_2id1010 + stride_lut_4 * 2];
            const scalar_t _2lut1001 = data_lut[_2id1001 + stride_lut_4 * 2];
            const scalar_t _2lut0110 = data_lut[_2id0110 + stride_lut_4 * 2];
            const scalar_t _2lut0101 = data_lut[_2id0101 + stride_lut_4 * 2];
            const scalar_t _2lut0011 = data_lut[_2id0011 + stride_lut_4 * 2];
            const scalar_t _2lut1110 = data_lut[_2id1110 + stride_lut_4 * 2];
            const scalar_t _2lut1101 = data_lut[_2id1101 + stride_lut_4 * 2];
            const scalar_t _2lut1011 = data_lut[_2id1011 + stride_lut_4 * 2];
            const scalar_t _2lut0111 = data_lut[_2id0111 + stride_lut_4 * 2];
            const scalar_t _2lut1111 = data_lut[_2id1111 + stride_lut_4 * 2];
            grad_d2 = grad_o_2 *
                (w0000_rd2 * _2lut0000 + w1000_rd2 * _2lut1000 + w0100_rd2 * _2lut0100 + w0010_rd2 * _2lut0010 +
                w0001_rd2 * _2lut0001 + w1100_rd2 * _2lut1100 + w1010_rd2 * _2lut1010 + w1001_rd2 * _2lut1001 +
                w0110_rd2 * _2lut0110 + w0101_rd2 * _2lut0101 + w0011_rd2 * _2lut0011 + w1110_rd2 * _2lut1110 +
                w1101_rd2 * _2lut1101 + w1011_rd2 * _2lut1011 + w0111_rd2 * _2lut0111 + w1111_rd2 * _2lut1111);
            // r
            atomicAdd(grad_inp + index + height * width * 2, grad_d2 * 1 / size_bin);

            grad_d2 = grad_o_2 *
                (w0000_gd2 * _2lut0000 + w1000_gd2 * _2lut1000 + w0100_gd2 * _2lut0100 + w0010_gd2 * _2lut0010 +
                w0001_gd2 * _2lut0001 + w1100_gd2 * _2lut1100 + w1010_gd2 * _2lut1010 + w1001_gd2 * _2lut1001 +
                w0110_gd2 * _2lut0110 + w0101_gd2 * _2lut0101 + w0011_gd2 * _2lut0011 + w1110_gd2 * _2lut1110 +
                w1101_gd2 * _2lut1101 + w1011_gd2 * _2lut1011 + w0111_gd2 * _2lut0111 + w1111_gd2 * _2lut1111);
            // g
            atomicAdd(grad_inp + index + height * width * 2 + (x >= width - 1 ? -1 : 1), grad_d2 * 1 / size_bin);

            grad_d2 = grad_o_2 *
                (w0000_bd2 * _2lut0000 + w1000_bd2 * _2lut1000 + w0100_bd2 * _2lut0100 + w0010_bd2 * _2lut0010 +
                w0001_bd2 * _2lut0001 + w1100_bd2 * _2lut1100 + w1010_bd2 * _2lut1010 + w1001_bd2 * _2lut1001 +
                w0110_bd2 * _2lut0110 + w0101_bd2 * _2lut0101 + w0011_bd2 * _2lut0011 + w1110_bd2 * _2lut1110 +
                w1101_bd2 * _2lut1101 + w1011_bd2 * _2lut1011 + w0111_bd2 * _2lut0111 + w1111_bd2 * _2lut1111);
            // b
            atomicAdd(grad_inp + index + height * width * 2 + (y >= height - 1 ? -width : width), grad_d2 * 1 / size_bin);

            grad_d2 = grad_o_2 *
                (w0000_cd2 * _2lut0000 + w1000_cd2 * _2lut1000 + w0100_cd2 * _2lut0100 + w0010_cd2 * _2lut0010 +
                w0001_cd2 * _2lut0001 + w1100_cd2 * _2lut1100 + w1010_cd2 * _2lut1010 + w1001_cd2 * _2lut1001 +
                w0110_cd2 * _2lut0110 + w0101_cd2 * _2lut0101 + w0011_cd2 * _2lut0011 + w1110_cd2 * _2lut1110 +
                w1101_cd2 * _2lut1101 + w1011_cd2 * _2lut1011 + w0111_cd2 * _2lut0111 + w1111_cd2 * _2lut1111);
            // c
            atomicAdd(grad_inp + index + height * width * 2 + 
    (y >= height - 1 ? -width : width) + 
    (x >= width - 1 ? -1 : 1), grad_d2 * 1 / size_bin);

    scalar_t grad_d3 = 0;
            const scalar_t _3lut0000 = data_lut[_3id0000 + stride_lut_4 * 3];
            const scalar_t _3lut1000 = data_lut[_3id1000 + stride_lut_4 * 3];
            const scalar_t _3lut0100 = data_lut[_3id0100 + stride_lut_4 * 3];
            const scalar_t _3lut0010 = data_lut[_3id0010 + stride_lut_4 * 3];
            const scalar_t _3lut0001 = data_lut[_3id0001 + stride_lut_4 * 3];
            const scalar_t _3lut1100 = data_lut[_3id1100 + stride_lut_4 * 3];
            const scalar_t _3lut1010 = data_lut[_3id1010 + stride_lut_4 * 3];
            const scalar_t _3lut1001 = data_lut[_3id1001 + stride_lut_4 * 3];
            const scalar_t _3lut0110 = data_lut[_3id0110 + stride_lut_4 * 3];
            const scalar_t _3lut0101 = data_lut[_3id0101 + stride_lut_4 * 3];
            const scalar_t _3lut0011 = data_lut[_3id0011 + stride_lut_4 * 3];
            const scalar_t _3lut1110 = data_lut[_3id1110 + stride_lut_4 * 3];
            const scalar_t _3lut1101 = data_lut[_3id1101 + stride_lut_4 * 3];
            const scalar_t _3lut1011 = data_lut[_3id1011 + stride_lut_4 * 3];
            const scalar_t _3lut0111 = data_lut[_3id0111 + stride_lut_4 * 3];
            const scalar_t _3lut1111 = data_lut[_3id1111 + stride_lut_4 * 3];
            grad_d3 = grad_o_3 *
                (w0000_rd3 * _3lut0000 + w1000_rd3 * _3lut1000 + w0100_rd3 * _3lut0100 + w0010_rd3 * _3lut0010 +
                w0001_rd3 * _3lut0001 + w1100_rd3 * _3lut1100 + w1010_rd3 * _3lut1010 + w1001_rd3 * _3lut1001 +
                w0110_rd3 * _3lut0110 + w0101_rd3 * _3lut0101 + w0011_rd3 * _3lut0011 + w1110_rd3 * _3lut1110 +
                w1101_rd3 * _3lut1101 + w1011_rd3 * _3lut1011 + w0111_rd3 * _3lut0111 + w1111_rd3 * _3lut1111);
            // r
            atomicAdd(grad_inp + index + height * width * 3, grad_d3 * 1 / size_bin);

            grad_d3 = grad_o_3 *
                (w0000_gd3 * _3lut0000 + w1000_gd3 * _3lut1000 + w0100_gd3 * _3lut0100 + w0010_gd3 * _3lut0010 +
                w0001_gd3 * _3lut0001 + w1100_gd3 * _3lut1100 + w1010_gd3 * _3lut1010 + w1001_gd3 * _3lut1001 +
                w0110_gd3 * _3lut0110 + w0101_gd3 * _3lut0101 + w0011_gd3 * _3lut0011 + w1110_gd3 * _3lut1110 +
                w1101_gd3 * _3lut1101 + w1011_gd3 * _3lut1011 + w0111_gd3 * _3lut0111 + w1111_gd3 * _3lut1111);
            // g
            atomicAdd(grad_inp + index + height * width * 3 + (x >= width - 1 ? -1 : 1), grad_d3 * 1 / size_bin);

            grad_d3 = grad_o_3 *
                (w0000_bd3 * _3lut0000 + w1000_bd3 * _3lut1000 + w0100_bd3 * _3lut0100 + w0010_bd3 * _3lut0010 +
                w0001_bd3 * _3lut0001 + w1100_bd3 * _3lut1100 + w1010_bd3 * _3lut1010 + w1001_bd3 * _3lut1001 +
                w0110_bd3 * _3lut0110 + w0101_bd3 * _3lut0101 + w0011_bd3 * _3lut0011 + w1110_bd3 * _3lut1110 +
                w1101_bd3 * _3lut1101 + w1011_bd3 * _3lut1011 + w0111_bd3 * _3lut0111 + w1111_bd3 * _3lut1111);
            // b
            atomicAdd(grad_inp + index + height * width * 3 + (y >= height - 1 ? -width : width), grad_d3 * 1 / size_bin);

            grad_d3 = grad_o_3 *
                (w0000_cd3 * _3lut0000 + w1000_cd3 * _3lut1000 + w0100_cd3 * _3lut0100 + w0010_cd3 * _3lut0010 +
                w0001_cd3 * _3lut0001 + w1100_cd3 * _3lut1100 + w1010_cd3 * _3lut1010 + w1001_cd3 * _3lut1001 +
                w0110_cd3 * _3lut0110 + w0101_cd3 * _3lut0101 + w0011_cd3 * _3lut0011 + w1110_cd3 * _3lut1110 +
                w1101_cd3 * _3lut1101 + w1011_cd3 * _3lut1011 + w0111_cd3 * _3lut0111 + w1111_cd3 * _3lut1111);
            // c
            atomicAdd(grad_inp + index + height * width * 3 + 
    (y >= height - 1 ? -width : width) + 
    (x >= width - 1 ? -1 : 1), grad_d3  * 1 / size_bin);

    scalar_t grad_d4 = 0;
            const scalar_t _4lut0000 = data_lut[_4id0000 + stride_lut_4 * 4];
            const scalar_t _4lut1000 = data_lut[_4id1000 + stride_lut_4 * 4];
            const scalar_t _4lut0100 = data_lut[_4id0100 + stride_lut_4 * 4];
            const scalar_t _4lut0010 = data_lut[_4id0010 + stride_lut_4 * 4];
            const scalar_t _4lut0001 = data_lut[_4id0001 + stride_lut_4 * 4];
            const scalar_t _4lut1100 = data_lut[_4id1100 + stride_lut_4 * 4];
            const scalar_t _4lut1010 = data_lut[_4id1010 + stride_lut_4 * 4];
            const scalar_t _4lut1001 = data_lut[_4id1001 + stride_lut_4 * 4];
            const scalar_t _4lut0110 = data_lut[_4id0110 + stride_lut_4 * 4];
            const scalar_t _4lut0101 = data_lut[_4id0101 + stride_lut_4 * 4];
            const scalar_t _4lut0011 = data_lut[_4id0011 + stride_lut_4 * 4];
            const scalar_t _4lut1110 = data_lut[_4id1110 + stride_lut_4 * 4];
            const scalar_t _4lut1101 = data_lut[_4id1101 + stride_lut_4 * 4];
            const scalar_t _4lut1011 = data_lut[_4id1011 + stride_lut_4 * 4];
            const scalar_t _4lut0111 = data_lut[_4id0111 + stride_lut_4 * 4];
            const scalar_t _4lut1111 = data_lut[_4id1111 + stride_lut_4 * 4];
            grad_d4 = grad_o_4 *
                (w0000_rd4 * _4lut0000 + w1000_rd4 * _4lut1000 + w0100_rd4 * _4lut0100 + w0010_rd4 * _4lut0010 +
                w0001_rd4 * _4lut0001 + w1100_rd4 * _4lut1100 + w1010_rd4 * _4lut1010 + w1001_rd4 * _4lut1001 +
                w0110_rd4 * _4lut0110 + w0101_rd4 * _4lut0101 + w0011_rd4 * _4lut0011 + w1110_rd4 * _4lut1110 +
                w1101_rd4 * _4lut1101 + w1011_rd4 * _4lut1011 + w0111_rd4 * _4lut0111 + w1111_rd4 * _4lut1111);
            // r
            atomicAdd(grad_inp + index + height * width * 4, grad_d4 * 1 / size_bin);

            grad_d4 = grad_o_4 *
                (w0000_gd4 * _4lut0000 + w1000_gd4 * _4lut1000 + w0100_gd4 * _4lut0100 + w0010_gd4 * _4lut0010 +
                w0001_gd4 * _4lut0001 + w1100_gd4 * _4lut1100 + w1010_gd4 * _4lut1010 + w1001_gd4 * _4lut1001 +
                w0110_gd4 * _4lut0110 + w0101_gd4 * _4lut0101 + w0011_gd4 * _4lut0011 + w1110_gd4 * _4lut1110 +
                w1101_gd4 * _4lut1101 + w1011_gd4 * _4lut1011 + w0111_gd4 * _4lut0111 + w1111_gd4 * _4lut1111);
            // g
            atomicAdd(grad_inp + index + height * width * 4 + (x >= width - 1 ? -1 : 1), grad_d4 * 1 / size_bin);

            grad_d4 = grad_o_4 *
                (w0000_bd4 * _4lut0000 + w1000_bd4 * _4lut1000 + w0100_bd4 * _4lut0100 + w0010_bd4 * _4lut0010 +
                w0001_bd4 * _4lut0001 + w1100_bd4 * _4lut1100 + w1010_bd4 * _4lut1010 + w1001_bd4 * _4lut1001 +
                w0110_bd4 * _4lut0110 + w0101_bd4 * _4lut0101 + w0011_bd4 * _4lut0011 + w1110_bd4 * _4lut1110 +
                w1101_bd4 * _4lut1101 + w1011_bd4 * _4lut1011 + w0111_bd4 * _4lut0111 + w1111_bd4 * _4lut1111);
            // b
            atomicAdd(grad_inp + index + height * width * 4 + (y >= height - 1 ? -width : width), grad_d4 * 1 / size_bin);

            grad_d4 = grad_o_4 *
                (w0000_cd4 * _4lut0000 + w1000_cd4 * _4lut1000 + w0100_cd4 * _4lut0100 + w0010_cd4 * _4lut0010 +
                w0001_cd4 * _4lut0001 + w1100_cd4 * _4lut1100 + w1010_cd4 * _4lut1010 + w1001_cd4 * _4lut1001 +
                w0110_cd4 * _4lut0110 + w0101_cd4 * _4lut0101 + w0011_cd4 * _4lut0011 + w1110_cd4 * _4lut1110 +
                w1101_cd4 * _4lut1101 + w1011_cd4 * _4lut1011 + w0111_cd4 * _4lut0111 + w1111_cd4 * _4lut1111);
            // c
            atomicAdd(grad_inp + index + height * width * 4 + 
    (y >= height - 1 ? -width : width) + 
    (x >= width - 1 ? -1 : 1), grad_d4 * 1 / size_bin);
        
    }
}


void SDLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output) {

    // sdlut_transform_sanity_check(input, lut, output);

    c10::cuda::CUDAGuard device_guard(input.device());

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(1);
    // printf("num_channels: %d\n", batch_size);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(0);
    int stride_lut   = lut.size(1);
    // printf("num_channels: %d\n", num_channels);
    int num_kernels =  height * width;
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "sdlut_transform_cuda_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                sdlut_transform_4d_cuda_forward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_inp, data_lut,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}



void SDLUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut) {

    c10::cuda::CUDAGuard device_guard(grad_output.device());

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(0);
    // printf(num_channels);
    int stride_lut   = lut.size(1);

    int num_kernels =  height * width;
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "sdlut_transform_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();

                sdlut_transform_4d_cuda_backward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_inp, data_lut, 
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}