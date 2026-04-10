#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>


void update_running_stat(float* r, dnnl::memory& current, float momentum) {
    size_t size = current.get_desc().get_dims()[0];
    float* c = static_cast<float*>(current.get_data_handle());
    for (size_t i = 0; i < size; ++i) {
        r[i] = (1.0f - momentum) * r[i] + momentum * c[i];
    }
}

namespace mt {
namespace ops{

    int64_t BatchNormalization::count = 0;

    BatchNormalization::BatchNormalization(
        bool training,
        const std::shared_ptr<TensorImpl>& running_mean,
        const std::shared_ptr<TensorImpl>& running_variance,
        float momentum,
        bool inc_counter):
        m_momentum(momentum),
        m_training(training)
    {

        if (
            !running_mean ||
            !running_variance->is_contiguous() ||
             running_mean->shape().size() != 1 ||
            !running_variance ||
            !running_variance->is_contiguous()  ||
             running_variance->shape().size() != 1 ||
             running_mean->shape()[0]  != running_variance->shape()[0]
         ){
            throw std::invalid_argument("error BatchNormalization() : must provide two 1D tensors for running_mean and running_variance (must have same dim0 value)");
        }

        m_running_mean = running_mean;
        m_running_variance = running_variance;

        if(inc_counter){
            m_name = "BatchNormalization"+std::to_string(count);
            count++;
        }

    } 
 
    std::shared_ptr<TensorImpl> BatchNormalization::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        try {
            const auto& in_tensor = operands[0];
            const auto& src_shape = in_tensor->shape();
            const auto& scale = operands[1];
            const auto& shift = operands[2];
            const auto& scale_shift_shape = scale->shape();
            const auto dst_stride =  row_major_stride(src_shape);

            if (
                src_shape.size() < 2 || 
                src_shape.size() > 4 ||
                scale_shift_shape.size() != 1 ||
                scale_shift_shape != shift->shape()
            ){
                throw std::invalid_argument(" in_tensor must be 2D,3D or 4D while scale & shift must be 1D contiguous tensors with same shape ");
            }

            if (
                ! scale->is_contiguous() || 
                ! shift->is_contiguous()
            ){
                throw std::invalid_argument(" scale & shift must be contiguous ");
            }

            if (
                scale_shift_shape[0] !=  m_running_mean->shape()[0]||
                src_shape[1] != scale_shift_shape[0] 
            ){
                throw std::invalid_argument(" in_tensor and scale_shift and running_mean_variance must have same channels number (ie, in_tensor.shape[1] = scale_shift.shape[0] .. ) ! ");
            }

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            
            std::shared_ptr<float> dst_data(new float[src_shape[0] * dst_stride[0]], std::default_delete<float[]>());

            /**************/
            // TODO : eliminate code redundancy
            /*************/

            auto src_md = dnnl::memory::desc(
                        src_shape, dnnl::memory::data_type::f32,  in_tensor->stride());
            auto dst_md = dnnl::memory::desc(
                    src_shape, dnnl::memory::data_type::f32, dst_stride);
            auto scaleshift_md = dnnl::memory::desc(
                    scale_shift_shape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
            
            
            auto src_mem = dnnl::memory(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
            auto dst_mem = dnnl::memory(dst_md, eng, dst_data.get());
            auto scale_mem = dnnl::memory(scaleshift_md, eng, scale->data_ptr().get() + scale->data_offset());
            auto shift_mem = dnnl::memory(scaleshift_md, eng, shift->data_ptr().get() + shift->data_offset());

            std::shared_ptr<BatchNormalization> grad_fn = nullptr;
            bool requires_grad = false;

            if (m_training){

                // Create primitive descriptor.
                m_fwd_bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_training, src_md, dst_md, 1.e-10f,
                        dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

                

                m_mean = dnnl::memory(m_fwd_bnorm_pd.mean_desc(), eng);
                m_variance = dnnl::memory(m_fwd_bnorm_pd.variance_desc(), eng);

                
                // Create the primitive.
                auto bnorm_prim = dnnl::batch_normalization_forward(m_fwd_bnorm_pd);

                // Primitive arguments. Set up in-place execution by assigning src as DST.
                std::unordered_map<int, dnnl::memory> bnorm_args;
                bnorm_args.insert({DNNL_ARG_SRC, src_mem});
                bnorm_args.insert({DNNL_ARG_MEAN, m_mean});
                bnorm_args.insert({DNNL_ARG_VARIANCE, m_variance});
                bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
                bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
                bnorm_args.insert({DNNL_ARG_DST, dst_mem});
                // Primitive execution: batch normalization with ReLU.
                bnorm_prim.execute(strm, bnorm_args);
                
                // Wait for the computation to finalize.
                strm.wait();

                if ( in_tensor->requires_grad() || shift->requires_grad() || scale->requires_grad()){
                    requires_grad = true; 
                    grad_fn = std::make_shared<BatchNormalization>(
                        true,
                        m_running_mean,
                        m_running_variance,
                        m_momentum,
                        true
                    );   
                    
                    grad_fn->set_operands({in_tensor, scale, shift}); 
                    grad_fn->m_mean = m_mean;
                    grad_fn->m_variance = m_variance;
                    grad_fn->m_fwd_bnorm_pd = m_fwd_bnorm_pd;

                    update_running_stat(m_running_mean->data_ptr().get() + m_running_mean->data_offset(), grad_fn->m_mean, m_momentum);
                    update_running_stat(m_running_variance->data_ptr().get() + m_running_variance->data_offset(), grad_fn->m_variance, m_momentum);

                }

        
            }else {

                auto src_md_any = dnnl::memory::desc(
                        src_shape, dnnl::memory::data_type::f32,  dnnl::memory::format_tag::any);
                auto dst_md_any = dnnl::memory::desc(
                        src_shape, dnnl::memory::data_type::f32,  dnnl::memory::format_tag::any);


                // Create primitive descriptor.
                auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_inference, src_md, dst_md, 1.e-10f,
                        dnnl::normalization_flags::use_scale | 
                        dnnl::normalization_flags::use_shift | 
                        dnnl::normalization_flags::use_global_stats 
                );

                auto inf_src_mem = src_mem;
                auto inf_dst_mem = dst_mem;
                
                if (src_md != bnorm_pd.src_desc()) {
                    dnnl::memory reordered_src(bnorm_pd.src_desc(), eng);
                    dnnl::reorder(src_mem, reordered_src).execute(strm, src_mem, reordered_src);
                    strm.wait();
                    inf_src_mem = reordered_src;
                }

                if (dst_md != bnorm_pd.dst_desc()) {
                    dnnl::memory reordered_dst(bnorm_pd.dst_desc(), eng);
                    dnnl::reorder(dst_mem, reordered_dst).execute(strm, dst_mem, reordered_dst);
                    strm.wait();
                    inf_dst_mem = reordered_dst;
                }


                dnnl::memory mean_mem = dnnl::memory(bnorm_pd.mean_desc(), eng, m_running_mean->data_ptr().get() + m_running_mean->data_offset());
                dnnl::memory variance_mem = dnnl::memory(bnorm_pd.variance_desc(), eng, m_running_variance->data_ptr().get() + m_running_variance->data_offset());
   
                // Create the primitive.
                auto bnorm_prim = dnnl::batch_normalization_forward(bnorm_pd);

                // Primitive arguments. Set up in-place execution by assigning src as DST.
                std::unordered_map<int, dnnl::memory> bnorm_args;
                bnorm_args.insert({DNNL_ARG_SRC, inf_src_mem});
                bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
                bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
                bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
                bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
                bnorm_args.insert({DNNL_ARG_DST, inf_dst_mem});
                // Primitive execution: batch normalization with ReLU.
                bnorm_prim.execute(strm, bnorm_args);

                // Wait for the computation to finalize.
                strm.wait();

                if (dst_md != bnorm_pd.dst_desc()) {
                    dnnl::reorder(inf_dst_mem,  dst_mem).execute(strm, inf_src_mem, dst_mem);
                    strm.wait();
                }

            }

            return std::make_shared<TensorImpl>(dst_data , 0 , src_shape , grad_fn , requires_grad, true , dst_stride);

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: BatchNormalization() was not possible for in_tensor: ") + e.what());

        }
    }  

    void BatchNormalization::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        const auto& x = m_operands[0];
        const auto& scale = m_operands[1];
        const auto& shift = m_operands[2];

        if (x->requires_grad() || scale->requires_grad() || shift->requires_grad()){

            auto src_md        = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, x->stride());
            auto scale_md      = dnnl::memory::desc(scale->shape() , dnnl::memory::data_type::f32, scale->stride());
            auto shift_md      = dnnl::memory::desc(shift->shape() , dnnl::memory::data_type::f32, shift->stride());
            auto diff_dst_md   = dnnl::memory::desc(diff_loss_out->shape() , dnnl::memory::data_type::f32, diff_loss_out->stride());
            auto diff_src_md   = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, row_major_stride(x->shape()));
            auto diff_scale_md = dnnl::memory::desc(scale->shape() , dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
            auto diff_shift_md = dnnl::memory::desc(shift->shape() , dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
            
            auto bwd_pd = dnnl::batch_normalization_backward::primitive_desc(
                engine,
                dnnl::prop_kind::backward,
                diff_src_md,
                diff_dst_md,
                src_md,
                1.e-10f,
                dnnl::normalization_flags::use_scale  |
                dnnl::normalization_flags::use_shift,
                m_fwd_bnorm_pd
            );

            dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
            dnnl::memory src_mem(src_md, engine, x->data_ptr().get() + x->data_offset());
            dnnl::memory scale_mem(scale_md, engine, scale->data_ptr().get() + scale->data_offset());

            dnnl::memory diff_src_mem;
            std::shared_ptr<float> src_data_storage; 
            dnnl::memory diff_scale_mem;
            std::shared_ptr<float> scale_data_storage; 
            dnnl::memory diff_shift_mem;
            std::shared_ptr<float> shift_data_storage; 

            if (x->get_grad()){
                diff_src_mem = dnnl::memory(diff_src_md, engine); 
            }else{
                src_data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
                diff_src_mem = dnnl::memory(diff_src_md, engine, src_data_storage.get()); 
            }

            if (scale->get_grad()){
                diff_scale_mem = dnnl::memory(diff_scale_md, engine); 
            }else{
                scale_data_storage = std::shared_ptr<float>(new float[scale->numel()], std::default_delete<float[]>());
                diff_scale_mem = dnnl::memory(diff_scale_md, engine, scale_data_storage.get()); 
            }

            if (shift->get_grad()){
                diff_shift_mem = dnnl::memory(diff_shift_md, engine); 
            }else{
                shift_data_storage = std::shared_ptr<float>(new float[shift->numel()], std::default_delete<float[]>());
                diff_shift_mem = dnnl::memory(diff_shift_md, engine, shift_data_storage.get()); 
            }
            
            auto bn_bwd = dnnl::batch_normalization_backward(bwd_pd);
            bn_bwd.execute(engine_stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_MEAN, m_mean},
                {DNNL_ARG_VARIANCE, m_variance},
                {DNNL_ARG_SCALE, scale_mem},
                {DNNL_ARG_DIFF_DST, diff_dst_mem},

                {DNNL_ARG_DIFF_SRC, diff_src_mem},
                {DNNL_ARG_DIFF_SCALE, diff_scale_mem},
                {DNNL_ARG_DIFF_SHIFT, diff_shift_mem}
            });

            engine_stream.wait();

            
            {

            if (x->get_grad()){

                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream
                );


            }else {
                x->set_grad(std::make_shared<TensorImpl>(src_data_storage, 0 , diff_src_md.get_dims(), nullptr , false, true , diff_src_md.get_strides()));
            }

            }

            
            {

            if (scale->get_grad()){
                accumulate(
                    diff_scale_mem,
                    scale->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                scale->set_grad(std::make_shared<TensorImpl>(scale_data_storage, 0 , diff_scale_md.get_dims(), nullptr , false, true , diff_scale_md.get_strides()));
            }
            }

            
            {
            if (shift->get_grad()){
                accumulate(
                    diff_shift_mem,
                    shift->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                shift->set_grad(std::make_shared<TensorImpl>(shift_data_storage, 0 , diff_shift_md.get_dims(), nullptr , false, true , diff_shift_md.get_strides()));
            }
            }

        }
    }



}//ops
}//mt