#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

template <typename T>
__global__ void ca_forward_kernel(const T *t, const T *f, T *weight, int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int z = blockIdx.z;

    if (x < width && y < height && z < height+width-1) {
        for (int batch = 0; batch < num; ++batch) {
            for (int plane = 0; plane < chn; ++plane) {
                T _t = t[(batch * chn + plane) * sp + y * width + x];

                if (z < width) {
                    int i = z;
                    T _f = f[(batch * chn + plane) * sp + y * width + i];
                    weight[(batch * len + i) * sp + y*width + x] += _t*_f;
                }
                else {
                    int i = z - width;
                    int j = i<y ? i : i+1;

                    T _f = f[(batch * chn + plane) * sp + j*width + x];
                    weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
                }
            }
        }
    }
}

template <typename T>
__global__ void ca_backward_kernel_t(const T *dw, const T *t, const T *f, T *dt,
                                     int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int plane = blockIdx.z;

    if (x < width && y < height && plane < chn) {
        for (int batch = 0; batch < num; ++batch) {
            for (int i = 0; i < width; ++i) {
                T _dw = dw[(batch * len + i) * sp + y*width + x];
                T _f = f[(batch * chn + plane) * sp + y*width + i];
                dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
            }
            for (int i = 0; i < height; ++i)  {
                if (i == y) continue;
                int j = i<y ? i : i-1;

                T _dw = dw[(batch * len + width + j) * sp + y*width + x];
                T _f = f[(batch * chn + plane) * sp + i*width + x];
                dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
            }
        }
    }
}

template <typename T>
__global__ void ca_backward_kernel_f(const T *dw, const T *t, const T *f, T *df,
                                     int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int plane = blockIdx.z;

    if (x < width && y < height && plane < chn) {
        for (int batch = 0; batch < num; ++batch) {
            for (int i = 0; i < width; ++i) {
                T _dw = dw[(batch * len + x) * sp + y*width + i];
                T _t = t[(batch * chn + plane) * sp + y*width + i];
                df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
            }
            for (int i = 0; i < height; ++i) {
                if (i == y) continue;
                int j = i>y ? y : y-1;

                T _dw = dw[(batch * len + width + j) * sp + i*width + x];
                T _t = t[(batch * chn + plane) * sp + i*width + x];
                df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
            }
        }
    }
}

template <typename T>
__global__ void ca_map_forward_kernel(const T *weight, const T *g, T *out, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
      for (int batch = 0; batch < num; ++batch) {
          for (int i = 0; i < width; ++i) {
              T _g = g[(batch * chn + plane) * sp + y*width + i];
              T _w = weight[(batch * len + i) * sp + y*width + x];
              out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
          }
          for (int i = 0; i < height; ++i) {
              if (i == y) continue;

              int j = i<y ? i : i-1;

              T _g = g[(batch * chn + plane) * sp + i*width + x];
              T _w = weight[(batch * len + width + j) * sp + y*width + x];
              out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
          }
      }
  }
}

template <typename T>
__global__ void ca_map_backward_kernel_w(const T *dout, const T *weight, const T *g, T *dw, int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int z = blockIdx.z;

    if (x < width && y < height && z < height+width-1) {
        for (int batch = 0; batch < num; ++batch) {
            for (int plane = 0; plane < chn; ++plane) {
                T _dout = dout[(batch * chn + plane) * sp + y*width + x];

                if (z < width) {
                    int i = z;
                    T _g = g[(batch * chn + plane) * sp + y*width + i];
                    dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
                }
                else {
                    int i = z - width;
                    int j = i<y ? i : i+1;

                    T _g = g[(batch * chn + plane) * sp + j*width + x];
                    dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
                }
            }
        }
    }
}

template <typename T>
__global__ void ca_map_backward_kernel_g(const T *dout, const T *weight, const T *g, T *dg, int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int plane = blockIdx.z;

    if (x < width && y < height && plane < chn) {
        for (int batch = 0; batch < num; ++batch) {
            for (int i = 0; i < width; ++i) {
                T _dout = dout[(batch * chn + plane) * sp + y*width + i];
                T _w = weight[(batch * len + x) * sp + y*width + i];
                dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
            }
            for (int i = 0; i < height; ++i) {
                if (i == y) continue;
                int j = i>y ? y : y-1;

                T _dout = dout[(batch * chn + plane) * sp + i*width + x];
                T _w = weight[(batch * len + width + j) * sp + i*width + x];
                dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
            }
        }
    }
}


namespace segmentron {
/*
 * Implementations
 */
at::Tensor ca_forward_cuda(const at::Tensor& t, const at::Tensor& f) {
    AT_ASSERTM(t.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(f.type().is_cuda(), "input must be a CUDA tensor");

    auto n = t.size(0);
    auto c = t.size(1);
    auto h = t.size(2);
    auto w = t.size(3);

    at::Tensor weight = at::zeros({n, h + w - 1, h, w}, t.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Run kernel
    dim3 threads(32, 32);
    int d1 = (w + threads.x - 1) / threads.x;
    int d2 = (h + threads.y - 1) / threads.y;
    int d3 = h + w;
    dim3 blocks(d1, d2, d3);

    AT_DISPATCH_FLOATING_TYPES(t.type(), "ca_forward", [&] {
        ca_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            t.contiguous().data<scalar_t>(),
            f.contiguous().data<scalar_t>(),
            weight.contiguous().data<scalar_t>(),
            n, c, h, w);
    });
    THCudaCheck(cudaGetLastError());
    return weight;
}

std::tuple<at::Tensor, at::Tensor> ca_backward_cuda(const at::Tensor& dw, const at::Tensor& t, const at::Tensor& f) {
    AT_ASSERTM(dw.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(t.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(f.type().is_cuda(), "input must be a CUDA tensor");

    auto n = t.size(0);
    auto c = t.size(1);
    auto h = t.size(2);
    auto w = t.size(3);

    at::Tensor dt = at::zeros_like(t);
    at::Tensor df = at::zeros_like(f);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Run kernel
    dim3 threads(32, 32);
    int d1 = (w + threads.x - 1) / threads.x;
    int d2 = (h + threads.y - 1) / threads.y;
    int d3 = c;
    dim3 blocks(d1, d2, d3);

    AT_DISPATCH_FLOATING_TYPES(t.type(), "ca_backward_kernel_t", [&] {
        ca_backward_kernel_t<scalar_t><<<blocks, threads, 0, stream>>> (
            dw.contiguous().data<scalar_t>(),
            t.contiguous().data<scalar_t>(),
            f.contiguous().data<scalar_t>(),
            dt.contiguous().data<scalar_t>(),
            n, c, h, w);
    });

    AT_DISPATCH_FLOATING_TYPES(f.type(), "ca_backward_kernel_f", [&] {
        ca_backward_kernel_f<scalar_t><<<blocks, threads, 0, stream>>> (
            dw.contiguous().data<scalar_t>(),
            t.contiguous().data<scalar_t>(),
            f.contiguous().data<scalar_t>(),
            df.contiguous().data<scalar_t>(),
            n, c, h, w);
    });
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(dt, df);
}

at::Tensor ca_map_forward_cuda(const at::Tensor& weight, const at::Tensor& g) {
    AT_ASSERTM(weight.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(g.type().is_cuda(), "input must be a CUDA tensor");

    auto n = g.size(0);
    auto c = g.size(1);
    auto h = g.size(2);
    auto w = g.size(3);

    at::Tensor out = at::zeros_like(g);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Run kernel
    dim3 threads(32, 32);
    int d1 = (w + threads.x - 1) / threads.x;
    int d2 = (h + threads.y - 1) / threads.y;
    int d3 = c;
    dim3 blocks(d1, d2, d3);

    AT_DISPATCH_FLOATING_TYPES(g.type(), "ca_map_forward", [&] {
        ca_map_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            weight.contiguous().data<scalar_t>(),
            g.contiguous().data<scalar_t>(),
            out.contiguous().data<scalar_t>(),
            n, c, h, w);
    });
    THCudaCheck(cudaGetLastError());
    return out;
}

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cuda(const at::Tensor& dout, const at::Tensor& weight, const at::Tensor& g) {
    AT_ASSERTM(dout.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(g.type().is_cuda(), "input must be a CUDA tensor");

    auto n = dout.size(0);
    auto c = dout.size(1);
    auto h = dout.size(2);
    auto w = dout.size(3);

    at::Tensor dw = at::zeros_like(weight);
    at::Tensor dg = at::zeros_like(g);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Run kernel
    dim3 threads(32, 32);
    int d1 = (w + threads.x - 1) / threads.x;
    int d2 = (h + threads.y - 1) / threads.y;
    int d3 = h + w;
    dim3 blocks(d1, d2, d3);

    AT_DISPATCH_FLOATING_TYPES(weight.type(), "ca_map_backward_kernel_w", [&] {
        ca_map_backward_kernel_w<scalar_t><<<blocks, threads, 0, stream>>> (
            dout.contiguous().data<scalar_t>(),
            weight.contiguous().data<scalar_t>(),
            g.contiguous().data<scalar_t>(),
            dw.contiguous().data<scalar_t>(),
            n, c, h, w);
    });

    AT_DISPATCH_FLOATING_TYPES(g.type(), "ca_map_backward_kernel_g", [&] {
        ca_map_backward_kernel_g<scalar_t><<<blocks, threads, 0, stream>>> (
            dout.contiguous().data<scalar_t>(),
            weight.contiguous().data<scalar_t>(),
            g.contiguous().data<scalar_t>(),
            dg.contiguous().data<scalar_t>(),
            n, c, h, w);
    });
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(dw, dg);
}

} // namespace segmentron