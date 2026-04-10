#ifndef OPS_H
#define OPS_H 
#include <vector>
#include <string>
#include <tuple>
#include <stdint.h>
#include <memory>
#include <dnnl.hpp>
#include <config/mtensor_export.hpp>
#include <immintrin.h> // For AVX512

namespace mt{

class TensorImpl;

using sliceList = std::vector<std::pair<int64_t, int64_t>>;
constexpr int EOD = -99999999;

namespace ops{

///////////////////////////////////////////
//         COMMON  HELPERS
///////////////////////////////////////////

void accumulate(
    dnnl::memory& src,
    const std::shared_ptr<TensorImpl>& dst,
    const dnnl::engine engine,
    const dnnl::stream stream_engine
);

void accumulate_mem(
    dnnl::memory& src,
    dnnl::memory& dst,
    dnnl::engine engine,
    dnnl::stream stream_engine
);


std::pair<float, int> horizontal_max_with_index_512(__m512 max_vals_vec, __m512i max_indices_vec);
std::shared_ptr<float> reduce_max_last_dim_avx512(
    const float* data_ptr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t m_dim_out,
    std::vector<std::vector<int64_t>>& max_indices_vec
);

std::pair<float, int> horizontal_min_with_index_512(__m512 min_vals_vec, __m512i min_indices_vec);
std::shared_ptr<float> reduce_min_last_dim_avx512(
    const float* data_ptr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t m_dim_out,
    std::vector<std::vector<int64_t>>& min_indices_vec
);


///////////////////////////////////////////
//             CUSTOM OPS
///////////////////////////////////////////

std::shared_ptr<float> custom_eltwise_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::eltwise_forward::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
);

std::shared_ptr<float> custom_binary_op(
    const std::shared_ptr<TensorImpl>& in_tensor_0, 
    const std::shared_ptr<TensorImpl>& in_tensor_1, 
    const dnnl::binary::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_0_md,
    const dnnl::memory::desc& src_1_md,
    const dnnl::memory::desc& dst_md
);

std::shared_ptr<float> custom_reduction_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::reduction::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
);


std::shared_ptr<float> custom_pooling_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    dnnl::algorithm pool_algorithm,  
    const dnnl::memory::dims& dst_dims,  
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream,
    dnnl::pooling_forward::primitive_desc& pool_fwd_pd,
    bool need_work_space,
    std::unique_ptr<dnnl::memory>& workspace_mem
);

inline dnnl::memory prepare_memory_for_primitive(
    float* user_data_ptr,
    const std::vector<int64_t>& user_shape,
    const std::vector<int64_t>& user_strides,
    const dnnl::memory::desc& expected_md,
    const dnnl::engine& engine,
    dnnl::stream& engine_stream
);

std::shared_ptr<float> custom_conv_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    const std::shared_ptr<TensorImpl>& weights,
    const std::shared_ptr<TensorImpl>& bias,
    const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& dst_strides,    
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream,
    dnnl::convolution_forward::primitive_desc& fwd_conv_pd
);

void conv_backward(
    const std::shared_ptr<TensorImpl>& x,
    const std::shared_ptr<TensorImpl>& w,
    const std::shared_ptr<TensorImpl>& b,
    const std::shared_ptr<TensorImpl>& diff_loss_out,
    const dnnl::convolution_forward::primitive_desc& fwd_pd,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pad_l,
    const std::vector<int64_t>& pad_r,
    const dnnl::engine& engine,
    dnnl::stream& engine_stream
);


std::shared_ptr<float> custom_deconv_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    const std::shared_ptr<TensorImpl>& weights,
    const std::shared_ptr<TensorImpl>& bias,
    const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& dst_strides,    
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream,
    dnnl::deconvolution_forward::primitive_desc& fwd_deconv_pd
);


void deconv_backward(
    dnnl::engine& engine,
    dnnl::stream& stream,
    const std::shared_ptr<TensorImpl>& x,
    const std::shared_ptr<TensorImpl>& w,
    const std::shared_ptr<TensorImpl>& b, // Can be nullptr
    const std::shared_ptr<TensorImpl>& diff_loss_out,
    const dnnl::deconvolution_forward::primitive_desc& fwd_pd_hint,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r
);



//Operations abstract class
class MTENSOR_API  Operation{
public :

    virtual std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) = 0;
    virtual void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) = 0;

    std::string name() const{
        return m_name;
    }

    //in case of operation that makes a node in grad graph we set the operands
    void set_operands(const std::vector<std::shared_ptr<TensorImpl>>& operands){
        m_operands = operands;
    }

    std::vector<std::shared_ptr<TensorImpl>> operands(){
        return m_operands;
    }

protected:
    std::vector<std::shared_ptr<TensorImpl>> m_operands;
    std::string m_name;
};


//view operations 

class MTENSOR_API View: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    View(const std::vector<int64_t>& shape, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_shape;
    static int64_t count;
    int64_t m_out_numel;
};

class MTENSOR_API Expand: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Expand(const std::vector<int64_t>& shape, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_expanded_dims;
    static int64_t count;
};

class MTENSOR_API Transpose: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Transpose(int64_t dim0, int64_t dim1, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim0;
    int64_t m_dim1;
    static int64_t count;
};

class MTENSOR_API Permute: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Permute(const std::vector<int64_t>& dims_permute , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_dims_permute;
    static int64_t count;
};


class MTENSOR_API Squeeze: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Squeeze(const int64_t& dim , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    static int64_t count;
};

class MTENSOR_API Unsqueeze: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Unsqueeze(const int64_t& dim , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    static int64_t count;
};

class MTENSOR_API Narrow: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Narrow(const int64_t& dim, const int64_t& start, const int64_t& length , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    int64_t m_start;
    int64_t m_length;
    static int64_t count;
};

class MTENSOR_API Slice: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Slice(const sliceList& dim , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    sliceList m_slice_list;
    static int64_t count;
};



//memory operations  

class MTENSOR_API Clone: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Clone(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Contiguous: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Contiguous(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


// joining operations 

class MTENSOR_API Cat: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Cat(const int64_t& dim , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    static int64_t count;
};

class MTENSOR_API Stack: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Stack(const int64_t& dim , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    static int64_t count;
};

class MTENSOR_API Embedding: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Embedding(const std::vector<int64_t>& indices , bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_indices;
    static int64_t count;
};

//eltwise operations 

class MTENSOR_API Exp: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Exp(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Tanh: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Tanh(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Abs: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Abs(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Clip: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Clip(float alpha, float beta, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    float m_alpha, m_beta;
};

class MTENSOR_API Log: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Log(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

// beta * x ^ alpha

class MTENSOR_API Pow: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Pow(float m_alpha, float m_beta = 1.0f, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    float m_alpha, m_beta;
};

class MTENSOR_API Linear: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Linear(float m_alpha, float m_beta, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    float m_alpha, m_beta;
};


class MTENSOR_API Relu: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Relu(float alpha, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    float m_alpha;
};

class MTENSOR_API Sigmoid: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Sigmoid(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Sqrt: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Sqrt(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


//binary ops 

class MTENSOR_API Add: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Add(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


class MTENSOR_API Sub: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Sub(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


class MTENSOR_API Mul: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Mul(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Div: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Div(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Eq: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Eq(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Ne: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Ne(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Ge: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Ge(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Gt: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Gt(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Le: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Le(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Lt: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Lt(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Max: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Max(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};

class MTENSOR_API Min: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Min(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


//reduction 

class MTENSOR_API Mean: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Mean(int64_t dim, float eps = 0.0f, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    int64_t m_dim;
    float m_eps;
    static int64_t count;
};

class MTENSOR_API Sum: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Sum(const std::vector<int64_t>& dims, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_dims;
    int64_t max_allowed_dim;
    static int64_t count;
};

class MTENSOR_API Mul_reduction: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Mul_reduction(const std::vector<int64_t>& dims, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_dims;
    int64_t max_allowed_dim;
    static int64_t count;
};

class MTENSOR_API Max_reduction: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Max_reduction(int64_t dim, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
    std::vector<std::vector<int64_t>> indices() const;
private:
    int64_t m_dim;
    std::vector<std::vector<int64_t>> m_max_indices;
    static int64_t count;
};

class MTENSOR_API Min_reduction: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Min_reduction(int64_t dim, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
    std::vector<std::vector<int64_t>> indices() const;
private:
    int64_t m_dim;
    std::vector<std::vector<int64_t>> m_min_indices;
    static int64_t count;
};


class MTENSOR_API Norm_lp_sum: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Norm_lp_sum(const std::vector<int64_t>& dims, float eps = 0.0f, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_dims;
    int64_t max_allowed_dim;
    float m_eps;
    static int64_t count;
};


class MTENSOR_API Norm_lp_power_p_sum: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Norm_lp_power_p_sum(const std::vector<int64_t>& dims, float p = 2.0f,float eps = 0.0f, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    std::vector<int64_t> m_dims;
    int64_t max_allowed_dim;
    float m_eps;
    float m_p;
    static int64_t count;
};


//linear  

class MTENSOR_API Matmul: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Matmul(bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
};


//softmax 


class MTENSOR_API Softmax: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Softmax(int dim, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::shared_ptr<TensorImpl> m_dst_tensor; //a pointer on the output
    int m_dim;
};

class MTENSOR_API SoftmaxLog: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    SoftmaxLog(int dim, bool inc_counter = false);
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    int m_dim;
    std::shared_ptr<TensorImpl> m_dst_tensor;
};


// pooling 


class MTENSOR_API MaxPooling1d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    MaxPooling1d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    std::unique_ptr<dnnl::memory> m_workspace_mem = nullptr;
};

class MTENSOR_API MaxPooling2d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    MaxPooling2d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    std::unique_ptr<dnnl::memory> m_workspace_mem = nullptr;
};

class MTENSOR_API MaxPooling3d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    MaxPooling3d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    std::unique_ptr<dnnl::memory> m_workspace_mem = nullptr;
};

class MTENSOR_API AvgPooling1d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    AvgPooling1d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool include_padding = false,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    bool m_include_padding;
};

class MTENSOR_API AvgPooling2d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    AvgPooling2d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool include_padding = false,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    bool m_include_padding;
};

class MTENSOR_API AvgPooling3d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    AvgPooling3d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool include_padding = false,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_kernel;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::pooling_forward::primitive_desc m_pool_fwd_pd;
    bool m_include_padding;
};


//convolution 


class MTENSOR_API Conv1d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Conv1d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::convolution_forward::primitive_desc m_conv_fwd_pd;
};

class MTENSOR_API Conv2d: public Operation {
public: 
    //inc_counter is true for operations that are inserted in grad graph
    Conv2d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::convolution_forward::primitive_desc m_conv_fwd_pd;
};


class MTENSOR_API Conv3d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Conv3d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::convolution_forward::primitive_desc m_conv_fwd_pd;
};


//deconvolution 

class MTENSOR_API Deconv1d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Deconv1d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::deconvolution_forward::primitive_desc m_fwd_deconv_pd;
};

class MTENSOR_API Deconv2d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Deconv2d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::deconvolution_forward::primitive_desc m_fwd_deconv_pd;
};


class MTENSOR_API Deconv3d: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    Deconv3d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_padding_l;
    std::vector<int64_t> m_padding_r;
    dnnl::deconvolution_forward::primitive_desc m_fwd_deconv_pd;
};

//normalization 


class MTENSOR_API BatchNormalization: public Operation {
public:
    BatchNormalization():m_training(true){}
    //inc_counter is true for operations that are inserted in grad graph
    BatchNormalization(
        bool training,
        const std::shared_ptr<TensorImpl>& running_mean,
        const std::shared_ptr<TensorImpl>& running_variance,
        float m_momentum = 0.1,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    bool m_training;
    float m_momentum = 0.1;
    dnnl::memory m_mean;
    dnnl::memory m_variance;
    std::shared_ptr<TensorImpl> m_running_mean;
    std::shared_ptr<TensorImpl> m_running_variance;
    dnnl::batch_normalization_forward::primitive_desc m_fwd_bnorm_pd;
};


class MTENSOR_API LayerNormalization: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    LayerNormalization(
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
    friend class GroupNormalization;
private:
    static int64_t count;
    dnnl::memory m_mean;
    dnnl::memory m_variance;
    dnnl::layer_normalization_forward::primitive_desc m_fwd_lnorm_pd;
};

class MTENSOR_API RMSNormalization: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    RMSNormalization(
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
    friend class GroupNormalization;
private:
    static int64_t count;
    dnnl::memory m_mean;
    dnnl::memory m_variance;
    dnnl::layer_normalization_forward::primitive_desc m_fwd_rmsnorm_pd;
};

class MTENSOR_API GroupNormalization: public Operation {
public:
    //inc_counter is true for operations that are inserted in grad graph
    GroupNormalization(
        int64_t groups,
        bool inc_counter = false
    );
    std::shared_ptr<TensorImpl> forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) override;
    void backward(const std::shared_ptr<TensorImpl>& diff_loss_out) override;
private:
    static int64_t count;
    int64_t m_groups;
    dnnl::memory m_mean ;
    dnnl::memory m_variance ;
    dnnl::group_normalization_forward::primitive_desc m_fwd_gnorm_pd;
};




} //ops

} //mt 



#endif  //OPS_H