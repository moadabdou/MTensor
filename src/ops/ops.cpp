#include <unordered_map>
#include <stdexcept>
#include <limits>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/utils/tensor_iterator.hpp>
#include <MTensor/ops.hpp>

namespace mt{

namespace ops{


void accumulate(
    dnnl::memory& src,
    const std::shared_ptr<TensorImpl>& dst,
    dnnl::engine engine,
    dnnl::stream stream_engine
){

    auto dst_md = dnnl::memory::desc(
            dst->shape() , dnnl::memory::data_type::f32, dst->stride());

    auto dst_mem = dnnl::memory(dst_md, engine, dst->data_ptr().get() + dst->data_offset() );

    auto binary_pd = dnnl::binary::primitive_desc(engine, dnnl::algorithm::binary_add,
            src.get_desc() , dst_md, dst_md);

    auto binary_prim = dnnl::binary(binary_pd);

    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src});
    binary_args.insert({DNNL_ARG_SRC_1, dst_mem});
    binary_args.insert({DNNL_ARG_DST, dst_mem});

    binary_prim.execute(stream_engine, binary_args);

    stream_engine.wait();

}


void accumulate_mem(
    dnnl::memory& src,
    dnnl::memory& dst,
    dnnl::engine engine,
    dnnl::stream stream_engine
){


    auto binary_pd = dnnl::binary::primitive_desc(engine, dnnl::algorithm::binary_add,
            src.get_desc() , dst.get_desc(), dst.get_desc());

    auto binary_prim = dnnl::binary(binary_pd);

    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src});
    binary_args.insert({DNNL_ARG_SRC_1, dst});
    binary_args.insert({DNNL_ARG_DST, dst});

    binary_prim.execute(stream_engine, binary_args);

    stream_engine.wait();

}

std::pair<float, int> horizontal_max_with_index_512(__m512 max_vals_vec, __m512i max_indices_vec) {

    __m256 half_vals = _mm512_extractf32x8_ps(max_vals_vec, 1);
    __m256i half_indices = _mm512_extracti32x8_epi32(max_indices_vec, 1);
    __m256 max_vals_vec_256 = _mm512_castps512_ps256(max_vals_vec);
    __m256i max_indices_vec_256 = _mm512_castsi512_si256(max_indices_vec);
    __m256 cmp_mask8 = _mm256_cmp_ps(max_vals_vec_256, half_vals, _CMP_LT_OQ);
    __m256 max_vals_256 = _mm256_blendv_ps(max_vals_vec_256, half_vals, cmp_mask8);
    __m256i cmp_mask8i = _mm256_castps_si256(cmp_mask8);
    __m256i max_indices_256 = _mm256_blendv_epi8(max_indices_vec_256, half_indices, cmp_mask8i);

    __m128 half_vals_128 = _mm256_extractf128_ps(max_vals_256, 1);
    __m128i half_indices_128 = _mm256_extracti128_si256(max_indices_256, 1);
    __mmask8 mask4 = _mm_cmp_ps_mask(_mm256_castps256_ps128(max_vals_256), half_vals_128, _CMP_LT_OQ);
    __m128 max_vals_128 = _mm_mask_blend_ps(mask4, _mm256_castps256_ps128(max_vals_256), half_vals_128);
    __m128i max_indices_128 = _mm_mask_blend_epi32(mask4, _mm256_castsi256_si128(max_indices_256), half_indices_128);

    half_vals_128 = _mm_movehl_ps(max_vals_128, max_vals_128);
    half_indices_128 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(max_indices_128), _mm_castsi128_ps(max_indices_128)));
    __mmask8 mask2 = _mm_cmp_ps_mask(max_vals_128, half_vals_128, _CMP_LT_OQ);
    max_vals_128 = _mm_mask_blend_ps(mask2, max_vals_128, half_vals_128);
    max_indices_128 = _mm_mask_blend_epi32(mask2, max_indices_128, half_indices_128);

    half_vals_128 = _mm_shuffle_ps(max_vals_128, max_vals_128, _MM_SHUFFLE(1, 1, 1, 1));
    half_indices_128 = _mm_shuffle_epi32(max_indices_128, _MM_SHUFFLE(1, 1, 1, 1));
    __mmask8 mask1 = _mm_cmp_ss_mask(max_vals_128, half_vals_128, _CMP_LT_OQ);
    max_vals_128 = _mm_mask_blend_ps(mask1, max_vals_128, half_vals_128);
    max_indices_128 = _mm_mask_blend_epi32(mask1, max_indices_128, half_indices_128);
    
    return { _mm_cvtss_f32(max_vals_128), _mm_cvtsi128_si32(max_indices_128) };
}

std::shared_ptr<float> reduce_max_last_dim_avx512(
    const float* data_ptr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t m_dim_out,
    std::vector<std::vector<int64_t>>& max_indices_vec
) {
    const int64_t last_dim_size = shape.back();
    std::vector<int64_t> outer_shape(shape.begin(), shape.end() - 1);
    std::vector<int64_t> outer_strides(strides.begin(), strides.end() - 1);

    int64_t output_size = 1;
    for (int64_t dim : outer_shape) output_size *= dim;

    auto output_ptr = std::shared_ptr<float>(new float[output_size], std::default_delete<float[]>());
    max_indices_vec.resize(output_size);

    mt::utils::TensorIterator it(outer_shape, outer_strides);

    auto kernel = [&](const std::vector<int64_t>& outer_coords, int64_t logical_idx) {
        const int64_t slice_start_offset = it.get_flat_index_from_coords(outer_coords);
        const float* slice_ptr = data_ptr + slice_start_offset;

        float max_val = -std::numeric_limits<float>::infinity();
        int64_t max_idx = -1;

        const int vec_width = 16; 
        int64_t vec_end = (last_dim_size / vec_width) * vec_width;

        if (vec_end > 0) {
            __m512 max_vals_vec = _mm512_loadu_ps(slice_ptr);
            __m512i max_indices_vec_i = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

            for (int64_t i = vec_width; i < vec_end; i += vec_width) {
                __m512 next_vals_vec = _mm512_loadu_ps(slice_ptr + i);
                __m512i next_indices_vec_i = _mm512_add_epi32(max_indices_vec_i, _mm512_set1_epi32(vec_width));
                
                __mmask16 mask = _mm512_cmp_ps_mask(next_vals_vec, max_vals_vec, _CMP_GT_OQ);

                max_vals_vec = _mm512_mask_blend_ps(mask, max_vals_vec, next_vals_vec);
                max_indices_vec_i = _mm512_mask_blend_epi32(mask, max_indices_vec_i, next_indices_vec_i);
            }
            
            auto result = horizontal_max_with_index_512(max_vals_vec, max_indices_vec_i);
            max_val = result.first;
            max_idx = result.second;
        }

        for (int64_t i = vec_end; i < last_dim_size; ++i) {
            if (slice_ptr[i] > max_val) {
                max_val = slice_ptr[i];
                max_idx = i;
            }
        }

        output_ptr.get()[logical_idx] = max_val;
        std::vector<int64_t> out_indices = outer_coords;
        out_indices.resize(outer_coords.size() + 1);
        out_indices.back() = out_indices[m_dim_out];
        out_indices[m_dim_out] = max_idx;
        max_indices_vec[logical_idx] = out_indices;
    };

    it.parallel_for_each(kernel);

    return output_ptr;
}


std::pair<float, int> horizontal_min_with_index_512(__m512 min_vals_vec, __m512i min_indices_vec) {

    __m256 half_vals = _mm512_extractf32x8_ps(min_vals_vec, 1);
    __m256i half_indices = _mm512_extracti32x8_epi32(min_indices_vec, 1);
    __m256 min_vals_vec_256 = _mm512_castps512_ps256(min_vals_vec);
    __m256i min_indices_vec_256 = _mm512_castsi512_si256(min_indices_vec);
    __m256 cmp_mask8 = _mm256_cmp_ps(min_vals_vec_256, half_vals, _CMP_GT_OQ);
    __m256 min_vals_256 = _mm256_blendv_ps(min_vals_vec_256, half_vals, cmp_mask8);
    __m256i cmp_mask8i = _mm256_castps_si256(cmp_mask8);
    __m256i min_indices_256 = _mm256_blendv_epi8(min_indices_vec_256, half_indices, cmp_mask8i);

    __m128 half_vals_128 = _mm256_extractf128_ps(min_vals_256, 1);
    __m128i half_indices_128 = _mm256_extracti128_si256(min_indices_256, 1);
    __mmask8 mask4 = _mm_cmp_ps_mask(_mm256_castps256_ps128(min_vals_256), half_vals_128, _CMP_GT_OQ);
    __m128 min_vals_128 = _mm_mask_blend_ps(mask4, _mm256_castps256_ps128(min_vals_256), half_vals_128);
    __m128i min_indices_128 = _mm_mask_blend_epi32(mask4, _mm256_castsi256_si128(min_indices_256), half_indices_128);

    half_vals_128 = _mm_movehl_ps(min_vals_128, min_vals_128);
    half_indices_128 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(min_indices_128), _mm_castsi128_ps(min_indices_128)));
    __mmask8 mask2 = _mm_cmp_ps_mask(min_vals_128, half_vals_128, _CMP_GT_OQ);
    min_vals_128 = _mm_mask_blend_ps(mask2, min_vals_128, half_vals_128);
    min_indices_128 = _mm_mask_blend_epi32(mask2, min_indices_128, half_indices_128);

    half_vals_128 = _mm_shuffle_ps(min_vals_128, min_vals_128, _MM_SHUFFLE(1, 1, 1, 1));
    half_indices_128 = _mm_shuffle_epi32(min_indices_128, _MM_SHUFFLE(1, 1, 1, 1));
    __mmask8 mask1 = _mm_cmp_ss_mask(min_vals_128, half_vals_128, _CMP_GT_OQ);
    min_vals_128 = _mm_mask_blend_ps(mask1, min_vals_128, half_vals_128);
    min_indices_128 = _mm_mask_blend_epi32(mask1, min_indices_128, half_indices_128);
    
    return { _mm_cvtss_f32(min_vals_128), _mm_cvtsi128_si32(min_indices_128) };
}

std::shared_ptr<float> reduce_min_last_dim_avx512(
    const float* data_ptr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t m_dim_out,
    std::vector<std::vector<int64_t>>& min_indices_vec
) {

    const int64_t last_dim_size = shape.back();
    std::vector<int64_t> outer_shape(shape.begin(), shape.end() - 1);
    std::vector<int64_t> outer_strides(strides.begin(), strides.end() - 1);

    int64_t output_size = 1;
    for (int64_t dim : outer_shape) output_size *= dim;

    auto output_ptr = std::shared_ptr<float>(new float[output_size], std::default_delete<float[]>());
    min_indices_vec.resize(output_size);

    mt::utils::TensorIterator it(outer_shape, outer_strides);

    auto kernel = [&](const std::vector<int64_t>& outer_coords, int64_t logical_idx) {
        const int64_t slice_start_offset = it.get_flat_index_from_coords(outer_coords);
        const float* slice_ptr = data_ptr + slice_start_offset;

        float min_val = std::numeric_limits<float>::infinity();
        int64_t min_idx = -1;

        const int vec_width = 16; 
        int64_t vec_end = (last_dim_size / vec_width) * vec_width;

        if (vec_end > 0) {
            __m512 min_vals_vec = _mm512_loadu_ps(slice_ptr);
            __m512i min_indices_vec_i = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

            for (int64_t i = vec_width; i < vec_end; i += vec_width) {
                __m512 next_vals_vec = _mm512_loadu_ps(slice_ptr + i);
                __m512i next_indices_vec_i = _mm512_add_epi32(min_indices_vec_i, _mm512_set1_epi32(vec_width));
                
                __mmask16 mask = _mm512_cmp_ps_mask(next_vals_vec, min_vals_vec, _CMP_LT_OQ);

                min_vals_vec = _mm512_mask_blend_ps(mask, min_vals_vec, next_vals_vec);
                min_indices_vec_i = _mm512_mask_blend_epi32(mask, min_indices_vec_i, next_indices_vec_i);
            }
            
            auto result = horizontal_min_with_index_512(min_vals_vec, min_indices_vec_i);
            min_val = result.first;
            min_idx = result.second;
        }

        for (int64_t i = vec_end; i < last_dim_size; ++i) {
            if (slice_ptr[i] < min_val) {
                min_val = slice_ptr[i];
                min_idx = i;
            }
        }

        output_ptr.get()[logical_idx] = min_val;
        std::vector<int64_t> out_indices = outer_coords;
        out_indices.resize(outer_coords.size() + 1);
        out_indices.back() = out_indices[m_dim_out];
        out_indices[m_dim_out] = min_idx;
        min_indices_vec[logical_idx] = out_indices;
    };

    it.parallel_for_each(kernel);

    return output_ptr;
}


std::shared_ptr<float> custom_eltwise_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::eltwise_forward::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
){
    
    dnnl::stream strm(engine);

    std::shared_ptr<float> dst_mp(new float[in_tensor->numel()], std::default_delete<float[]>());

    dnnl::memory src_m(src_md, engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    dnnl::memory dst_m(dst_md, engine, dst_mp.get());

    dnnl::eltwise_forward eltwise_op(primitive_desc);
    
    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    eltwise_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


//input tensor are of the same shape 
std::shared_ptr<float> custom_binary_op(
    const std::shared_ptr<TensorImpl>& in_tensor_0, 
    const std::shared_ptr<TensorImpl>& in_tensor_1, 
    const dnnl::binary::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_0_md,
    const dnnl::memory::desc& src_1_md,
    const dnnl::memory::desc& dst_md
){

    dnnl::stream strm(engine);

    //prepare memories 

    const auto& dst_mp = std::shared_ptr<float>(new float[in_tensor_0->numel()], std::default_delete<float[]>());

    const auto& in_tensor_0_mem = dnnl::memory(src_0_md, engine, in_tensor_0->data_ptr().get() + in_tensor_0->data_offset());
    const auto& in_tensor_1_mem = dnnl::memory(src_1_md, engine, in_tensor_1->data_ptr().get() + in_tensor_1->data_offset());
    const auto& dst_mem = dnnl::memory(dst_md, engine, dst_mp.get());


    //binary execution
    dnnl::binary binary_op(primitive_desc);

    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC_0, in_tensor_0_mem});
    op_args.insert({DNNL_ARG_SRC_1, in_tensor_1_mem});
    op_args.insert({DNNL_ARG_DST, dst_mem});

    binary_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


std::shared_ptr<float> custom_reduction_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::reduction::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
){
    
    dnnl::stream strm(engine);

    std::shared_ptr<float> dst_mp(new float[ dst_md.get_dims()[0] * dst_md.get_strides()[0] ], std::default_delete<float[]>());

    dnnl::memory src_m(src_md, engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    dnnl::memory dst_m(dst_md, engine, dst_mp.get());

    dnnl::reduction reduction_op(primitive_desc);
    
    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    reduction_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


//pooling  

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
){

    
    auto src_md = dnnl::memory::desc(in_tensor->shape() , dnnl::memory::data_type::f32, in_tensor->stride());
    // Let oneDNN determine the best output dimensions and format
    auto dst_md = dnnl::memory::desc(dst_dims , dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    pool_fwd_pd = dnnl::pooling_forward::primitive_desc(
        engine,
        need_work_space ? dnnl::prop_kind::forward_training  : dnnl::prop_kind::forward_inference, 
        pool_algorithm,
        src_md,
        dst_md,
        strides,
        kernel,
        {0,0,0},
        padding_l, 
        padding_r
    );

    const auto& dst_shape = pool_fwd_pd.dst_desc().get_dims();
    const auto& dst_stride = row_major_stride(dst_shape);

    std::shared_ptr<float> dst_data(new float[dst_shape[0] * dst_stride[0]], std::default_delete<float[]>());

    auto src_m = dnnl::memory(src_md , engine , in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto dst_m = dnnl::memory(pool_fwd_pd.dst_desc(), engine, dst_data.get());

    auto pooling = dnnl::pooling_forward(pool_fwd_pd);

    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    if (need_work_space){
        workspace_mem = std::make_unique<dnnl::memory>(pool_fwd_pd.workspace_desc(), engine);
        op_args.insert({DNNL_ARG_WORKSPACE, *workspace_mem});
    }

    pooling.execute(stream, op_args);
    stream.wait();

    return dst_data;
}

inline dnnl::memory prepare_memory_for_primitive(
    float* user_data_ptr,
    const std::vector<int64_t>& user_shape,
    const std::vector<int64_t>& user_strides,
    const dnnl::memory::desc& expected_md,
    const dnnl::engine& engine,
    dnnl::stream& engine_stream)
{
    // Create user memory descriptor
    dnnl::memory::desc user_md(user_shape, dnnl::memory::data_type::f32, user_strides);

    // Wrap user memory
    dnnl::memory user_mem(user_md, engine, user_data_ptr);

    // Check if reorder is needed
    if (user_md != expected_md) {
        dnnl::memory reordered_mem(expected_md, engine);
        dnnl::reorder(user_mem, reordered_mem).execute(engine_stream, user_mem, reordered_mem);
        return reordered_mem;
    }

    return user_mem;
}


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
){

    const auto& src_dims = in_tensor->shape();
    const auto& src_stride = in_tensor->stride();
    const auto& weights_dims = weights->shape();
    const auto& weights_stride = weights->stride();


    auto user_src_mem = dnnl::memory(
            {src_dims, dnnl::memory::data_type::f32, src_stride},
            engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto user_weights_mem = dnnl::memory(
            {weights_dims , dnnl::memory::data_type::f32, weights_stride},
            engine, weights->data_ptr().get() + weights->data_offset());
    
    const auto dst_numel = dst_strides[0] * dst_dims[0];


    std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());
    
    auto user_dst_mem = dnnl::memory(
            {dst_dims, dnnl::memory::data_type::f32, dst_strides},
            engine, dst_data.get());

    auto conv_src_md = dnnl::memory::desc(
            src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv_weights_md = dnnl::memory::desc(
            weights_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv_dst_md = dnnl::memory::desc(
            dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    
    
    
    auto user_bias_md = bias
            ? dnnl::memory::desc( bias->shape(), dnnl::memory::data_type::f32, bias->stride())
            : dnnl::memory::desc();
    auto user_bias_mem = bias ? dnnl::memory(user_bias_md, engine, bias->data_ptr().get() + bias->data_offset()) : dnnl::memory(user_bias_md, engine);

    fwd_conv_pd = dnnl::convolution_forward::primitive_desc(engine,
            dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides , padding_l, padding_r);
        
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    if (fwd_conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = dnnl::memory(fwd_conv_pd.src_desc(), engine);
        dnnl::reorder(user_src_mem, conv_src_mem)
                .execute(stream, user_src_mem, conv_src_mem);
    }

    if (fwd_conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = dnnl::memory(fwd_conv_pd.weights_desc(), engine);
        dnnl::reorder(user_weights_mem, conv_weights_mem)
                .execute(stream, user_weights_mem, conv_weights_mem);
    }

    if (fwd_conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = dnnl::memory(fwd_conv_pd.dst_desc(), engine);
    }

    auto conv_prim = dnnl::convolution_forward(fwd_conv_pd);

    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    conv_prim.execute(stream, conv_args);

    if (fwd_conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        dnnl::reorder(conv_dst_mem, user_dst_mem)
                .execute(stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;
    
    stream.wait();
    
    return dst_data;
}



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
    dnnl::stream& engine_stream)
{
    const auto& diff_dst_user_md = dnnl::memory::desc(
        diff_loss_out->shape(), dnnl::memory::data_type::f32, diff_loss_out->stride());

    // -------- Backward Data (dx) --------
    if (x->requires_grad())
    {
        auto diff_src_any_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto w_any_md = dnnl::memory::desc(w->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

        auto bwd_data_pd = dnnl::convolution_backward_data::primitive_desc(
            engine,
            dnnl::algorithm::convolution_direct,
            diff_src_any_md,
            w_any_md,
            diff_dst_user_md,
            strides, pad_l, pad_r, fwd_pd);

        dnnl::memory diff_dst_mem = prepare_memory_for_primitive(
            diff_loss_out->data_ptr().get() + diff_loss_out->data_offset(),
            diff_loss_out->shape(),
            diff_loss_out->stride(),
            bwd_data_pd.diff_dst_desc(),
            engine, engine_stream);

        dnnl::memory w_mem = prepare_memory_for_primitive(
            w->data_ptr().get() + w->data_offset(),
            w->shape(),
            w->stride(),
            bwd_data_pd.weights_desc(),
            engine, engine_stream);

        dnnl::memory diff_src_mem;
        std::shared_ptr<float> dx_storage;
        if (x->get_grad()) {
            diff_src_mem = dnnl::memory(bwd_data_pd.diff_src_desc(), engine);
        } else {
            dx_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
            diff_src_mem = dnnl::memory(bwd_data_pd.diff_src_desc(), engine, dx_storage.get());
        }

        dnnl::convolution_backward_data conv_bwd_data(bwd_data_pd);
        conv_bwd_data.execute(engine_stream, {
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_WEIGHTS, w_mem},
            {DNNL_ARG_DIFF_SRC, diff_src_mem}
        });
        engine_stream.wait();

        if (x->get_grad()) {
            accumulate(diff_src_mem, x->get_grad(), engine, engine_stream);
        } else {
            x->set_grad(std::make_shared<TensorImpl>(
                dx_storage, 0, bwd_data_pd.diff_src_desc().get_dims(),
                nullptr, false, true, bwd_data_pd.diff_src_desc().get_strides()));
        }
    }

    // -------- Backward Weights (dw, db) --------
    if (w->requires_grad() || (b && b->requires_grad()))
    {
        auto x_any_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto diff_w_any_md = dnnl::memory::desc(w->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto diff_b_any_md = b ? dnnl::memory::desc(b->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any)
                               : dnnl::memory::desc();

        auto bwd_wt_pd = dnnl::convolution_backward_weights::primitive_desc(
            engine,
            dnnl::algorithm::convolution_direct,
            x_any_md,
            diff_w_any_md,
            diff_b_any_md,
            diff_dst_user_md,
            strides, pad_l, pad_r, fwd_pd);

        dnnl::memory x_mem = prepare_memory_for_primitive(
            x->data_ptr().get() + x->data_offset(),
            x->shape(),
            x->stride(),
            bwd_wt_pd.src_desc(),
            engine, engine_stream);

        dnnl::memory diff_dst_mem = prepare_memory_for_primitive(
            diff_loss_out->data_ptr().get() + diff_loss_out->data_offset(),
            diff_loss_out->shape(),
            diff_loss_out->stride(),
            bwd_wt_pd.diff_dst_desc(),
            engine, engine_stream);

        dnnl::memory dw_mem;
        std::shared_ptr<float> dw_storage;
        if (w->get_grad()) {
            dw_mem = dnnl::memory(bwd_wt_pd.diff_weights_desc(), engine);
        } else {
            dw_storage = std::shared_ptr<float>(new float[w->numel()], std::default_delete<float[]>());
            dw_mem = dnnl::memory(bwd_wt_pd.diff_weights_desc(), engine, dw_storage.get());
        }

        dnnl::memory db_mem;
        std::shared_ptr<float> db_storage;
        if (b) {
            if (b->get_grad()) {
                db_mem = dnnl::memory(bwd_wt_pd.diff_bias_desc(), engine);
            } else {
                db_storage = std::shared_ptr<float>(new float[b->numel()], std::default_delete<float[]>());
                db_mem = dnnl::memory(bwd_wt_pd.diff_bias_desc(), engine, db_storage.get());
            }
        }

        dnnl::convolution_backward_weights conv_bwd_wt(bwd_wt_pd);
        conv_bwd_wt.execute(engine_stream, {
            {DNNL_ARG_SRC, x_mem},
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_DIFF_WEIGHTS, dw_mem},
            {DNNL_ARG_DIFF_BIAS, db_mem}
        });
        engine_stream.wait();

        if (w->get_grad()) {
            accumulate(dw_mem, w->get_grad(), engine, engine_stream);
        } else {
            w->set_grad(std::make_shared<TensorImpl>(
                dw_storage, 0, bwd_wt_pd.diff_weights_desc().get_dims(),
                nullptr, false, true, bwd_wt_pd.diff_weights_desc().get_strides()));
        }

        if (b) {
            if (b->get_grad()) {
                accumulate(db_mem, b->get_grad(), engine, engine_stream);
            } else {
                b->set_grad(std::make_shared<TensorImpl>(
                    db_storage, 0, bwd_wt_pd.diff_bias_desc().get_dims(),
                    nullptr, false, true, bwd_wt_pd.diff_bias_desc().get_strides()));
            }
        }
    }
}


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
){
    const auto& src_dims = in_tensor->shape();
    const auto& src_stride = in_tensor->stride();
    const auto& weights_dims = weights->shape();
    const auto& weights_stride = weights->stride();


    auto user_src_mem = dnnl::memory(
            {src_dims, dnnl::memory::data_type::f32, src_stride},
            engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto user_weights_mem = dnnl::memory(
            {weights_dims , dnnl::memory::data_type::f32, weights_stride},
            engine, weights->data_ptr().get() + weights->data_offset());
    
    const auto dst_numel = dst_strides[0] * dst_dims[0];


    std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());
    
    auto user_dst_mem = dnnl::memory(
            {dst_dims, dnnl::memory::data_type::f32, dst_strides},
            engine, dst_data.get());

    auto deconv_src_md = dnnl::memory::desc(
            src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto deconv_weights_md = dnnl::memory::desc(
            weights_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto deconv_dst_md = dnnl::memory::desc(
            dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    
    
    
    auto user_bias_md = bias
            ? dnnl::memory::desc( bias->shape(), dnnl::memory::data_type::f32, bias->stride())
            : dnnl::memory::desc();
    auto user_bias_mem = bias ? dnnl::memory(user_bias_md, engine, bias->data_ptr().get() + bias->data_offset()) : dnnl::memory(user_bias_md, engine);

    fwd_deconv_pd = dnnl::deconvolution_forward::primitive_desc(engine,
            dnnl::prop_kind::forward_training, dnnl::algorithm::deconvolution_direct,
            deconv_src_md, deconv_weights_md, user_bias_md, deconv_dst_md,
            strides , padding_l, padding_r);
        
    auto deconv_src_mem = user_src_mem;
    auto deconv_weights_mem = user_weights_mem;
    auto deconv_dst_mem = user_dst_mem;

    if (fwd_deconv_pd.src_desc() != user_src_mem.get_desc()) {
        deconv_src_mem = dnnl::memory(fwd_deconv_pd.src_desc(), engine);
        dnnl::reorder(user_src_mem, deconv_src_mem)
                .execute(stream, user_src_mem, deconv_src_mem);
    }

    if (fwd_deconv_pd.weights_desc() != user_weights_mem.get_desc()) {
        deconv_weights_mem = dnnl::memory(fwd_deconv_pd.weights_desc(), engine);
        dnnl::reorder(user_weights_mem, deconv_weights_mem)
                .execute(stream, user_weights_mem, deconv_weights_mem);
    }

    if (fwd_deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        deconv_dst_mem = dnnl::memory(fwd_deconv_pd.dst_desc(), engine);
    }

    auto conv_prim = dnnl::deconvolution_forward(fwd_deconv_pd);

    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, deconv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, deconv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, deconv_dst_mem});

    conv_prim.execute(stream, conv_args);

    if (fwd_deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        dnnl::reorder(deconv_dst_mem, user_dst_mem)
                .execute(stream, deconv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = deconv_dst_mem;
    
    stream.wait();
    
    return dst_data;
}


dnnl::memory create_user_memory(
    const std::shared_ptr<TensorImpl>& tensor,
    dnnl::engine& engine)
{
    auto md = dnnl::memory::desc(tensor->shape(), dnnl::memory::data_type::f32, tensor->stride());
    return dnnl::memory(md, engine, tensor->data_ptr().get() + tensor->data_offset());
}

// Reorders data to the primitive's preferred format if necessary.
dnnl::memory reorder_to_primitive_if_needed(
    dnnl::memory& user_mem,
    const dnnl::memory::desc& prim_md,
    dnnl::engine& engine,
    dnnl::stream& stream)
{
    if (user_mem.get_desc() == prim_md) {
        return user_mem;
    }

    auto prim_mem = dnnl::memory(prim_md, engine);
    dnnl::reorder(user_mem, prim_mem).execute(stream, user_mem, prim_mem);
    return prim_mem;
}

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
    const dnnl::memory::dims& padding_r)
{
    auto user_x_mem = create_user_memory(x, engine);
    auto user_w_mem = create_user_memory(w, engine);
    auto user_diff_dst_mem = create_user_memory(diff_loss_out, engine);

    if (x->requires_grad())
    {
        auto diff_src_md_any = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto w_md_any = dnnl::memory::desc(w->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto diff_dst_md_any = dnnl::memory::desc(diff_loss_out->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

        auto bwd_data_pd = dnnl::deconvolution_backward_data::primitive_desc(
            engine, dnnl::algorithm::deconvolution_direct,
            diff_src_md_any, w_md_any, diff_dst_md_any,
            strides, padding_l, padding_r, fwd_pd_hint);

        auto prim_w_mem = reorder_to_primitive_if_needed(user_w_mem, bwd_data_pd.weights_desc(), engine, stream);
        auto prim_diff_dst_mem = reorder_to_primitive_if_needed(user_diff_dst_mem, bwd_data_pd.diff_dst_desc(), engine, stream);
        auto prim_diff_src_mem = dnnl::memory(bwd_data_pd.diff_src_desc(), engine);

        std::unordered_map<int, dnnl::memory> bwd_data_args;
        bwd_data_args[DNNL_ARG_DIFF_DST] = prim_diff_dst_mem;
        bwd_data_args[DNNL_ARG_WEIGHTS] = prim_w_mem;
        bwd_data_args[DNNL_ARG_DIFF_SRC] = prim_diff_src_mem;
        dnnl::deconvolution_backward_data(bwd_data_pd).execute(stream, bwd_data_args);

        auto user_diff_src_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, row_major_stride(x->shape()));
        std::shared_ptr<float> data_storage;
        dnnl::memory final_diff_src_mem;
        
        if (x->get_grad()) {
            final_diff_src_mem = dnnl::memory(user_diff_src_md, engine);
        } else {
            data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
            final_diff_src_mem = dnnl::memory(user_diff_src_md, engine, data_storage.get());
        }

        dnnl::reorder(prim_diff_src_mem, final_diff_src_mem).execute(stream, prim_diff_src_mem, final_diff_src_mem);
        stream.wait();

        if (x->get_grad()) {
            accumulate(final_diff_src_mem, x->get_grad(), engine, stream);
        } else {
            x->set_grad(std::make_shared<TensorImpl>(data_storage, 0, user_diff_src_md.get_dims(), nullptr, false, true, user_diff_src_md.get_strides()));
        }
    }

    if (w->requires_grad() || (b && b->requires_grad()))
    {
        auto x_md_any = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto diff_w_md_any = dnnl::memory::desc(w->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        auto diff_b_md_any = b ? dnnl::memory::desc(b->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any) : dnnl::memory::desc();
        auto diff_dst_md_any = dnnl::memory::desc(diff_loss_out->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
        
        auto bwd_weights_pd = dnnl::deconvolution_backward_weights::primitive_desc(
            engine, dnnl::algorithm::deconvolution_direct,
            x_md_any, diff_w_md_any, diff_b_md_any, diff_dst_md_any,
            strides, padding_l, padding_r, fwd_pd_hint);

        auto prim_x_mem = reorder_to_primitive_if_needed(user_x_mem, bwd_weights_pd.src_desc(), engine, stream);
        auto prim_diff_dst_mem = reorder_to_primitive_if_needed(user_diff_dst_mem, bwd_weights_pd.diff_dst_desc(), engine, stream);
        auto prim_diff_w_mem = dnnl::memory(bwd_weights_pd.diff_weights_desc(), engine);
        auto prim_diff_b_mem = b ? dnnl::memory(bwd_weights_pd.diff_bias_desc(), engine) : dnnl::memory();

        std::unordered_map<int, dnnl::memory> bwd_weights_args;
        bwd_weights_args[DNNL_ARG_SRC] = prim_x_mem;
        bwd_weights_args[DNNL_ARG_DIFF_DST] = prim_diff_dst_mem;
        bwd_weights_args[DNNL_ARG_DIFF_WEIGHTS] = prim_diff_w_mem;
        if (b) {
            bwd_weights_args[DNNL_ARG_DIFF_BIAS] = prim_diff_b_mem;
        }
        dnnl::deconvolution_backward_weights(bwd_weights_pd).execute(stream, bwd_weights_args);
        stream.wait();

        if (w->requires_grad()) {
            auto user_diff_w_md = dnnl::memory::desc(w->shape(), dnnl::memory::data_type::f32, row_major_stride(w->shape()));
            std::shared_ptr<float> w_grad_storage;
            dnnl::memory final_diff_w_mem;
            
            if (w->get_grad()) {
                final_diff_w_mem = dnnl::memory(user_diff_w_md, engine);
            } else {
                w_grad_storage = std::shared_ptr<float>(new float[w->numel()], std::default_delete<float[]>());
                final_diff_w_mem = dnnl::memory(user_diff_w_md, engine, w_grad_storage.get());
            }

            dnnl::reorder(prim_diff_w_mem, final_diff_w_mem).execute(stream, prim_diff_w_mem, final_diff_w_mem);
            stream.wait();

            if (w->get_grad()) {
                accumulate(final_diff_w_mem, w->get_grad(), engine, stream);
            } else {
                w->set_grad(std::make_shared<TensorImpl>(w_grad_storage, 0, user_diff_w_md.get_dims(), nullptr, false, true, user_diff_w_md.get_strides()));
            }
        }

        if (b && b->requires_grad()) {
            auto user_diff_b_md = dnnl::memory::desc(b->shape(), dnnl::memory::data_type::f32, row_major_stride(b->shape()));
            std::shared_ptr<float> b_grad_storage;
            dnnl::memory final_diff_b_mem;

            if (b->get_grad()) {
                final_diff_b_mem = dnnl::memory(user_diff_b_md, engine);
            } else {
                b_grad_storage = std::shared_ptr<float>(new float[b->numel()], std::default_delete<float[]>());
                final_diff_b_mem = dnnl::memory(user_diff_b_md, engine, b_grad_storage.get());
            }

            dnnl::reorder(prim_diff_b_mem, final_diff_b_mem).execute(stream, prim_diff_b_mem, final_diff_b_mem);
            stream.wait();

            if (b->get_grad()) {
                accumulate(final_diff_b_mem, b->get_grad(), engine, stream);
            } else {
                b->set_grad(std::make_shared<TensorImpl>(b_grad_storage, 0, user_diff_b_md.get_dims(), nullptr, false, true, user_diff_b_md.get_strides()));
            }
        }
    }
}
    
}//ops

} //mt 


