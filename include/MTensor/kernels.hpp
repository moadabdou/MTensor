#include <config/kernels_export.hpp>
#include <cstdint>


KERNEL_API void fused_adam_kernel_avx512(
    float *param,
    float *grad,
    float *m,
    float *v,
    int64_t N,
    float beta1,
    float beta2,
    float *beta1_t,
    float *beta2_t,
    float eps,
    float lr
);

KERNEL_API void fused_adam_w_kernel_avx512(
    float *param,
    float *grad,
    float *m,
    float *v,
    int64_t N,
    float beta1,
    float beta2,
    float *beta1_t,
    float *beta2_t,
    float eps,
    float lr,
    float weight_decay
);

KERNEL_API void fused_sgd_m_kernel_avx512(
    float *param,
    float *grad,
    float *v,
    int64_t N,
    float u,
    float lr,
    float weight_decay
);

KERNEL_API void fused_sgd_kernel_avx512(
    float *param,
    float *grad,
    int64_t N,
    float lr,
    float weight_decay
);
