add_library(MTensor SHARED)

target_sources(MTensor PRIVATE

  #tensorImpl
  ${CMAKE_CURRENT_SOURCE_DIR}/core/tensorImpl.cpp

  #tensor 
  ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor.cpp


  ##############################################
  #                   UTILS                    #
  ##############################################

  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/tensor_print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/broadcast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/tensor_iterator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/general.cpp



  ##############################################
  #           TENSOR      OPERATIONS           #
  ##############################################

  ${CMAKE_CURRENT_SOURCE_DIR}/ops/ops.cpp

  #view_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/view.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/permute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/squeeze.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/unsqueeze.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/narrow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/expand.cpp

  #memory_ops 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/memory/clone.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/memory/contiguous.cpp

  #join operations
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/joining/cat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/joining/stack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/joining/embedding.cpp

  #eltwise_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/exp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/tanh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/linear.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/abs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/clip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/pow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/sigmoid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/sqrt.cpp

  #binary_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/sub.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/mul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/div.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/eq.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/ne.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/ge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/gt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/le.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/lt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/max.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/min.cpp

  #reduction_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/mean.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/max_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/min_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/sum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/mul_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/norm_lp_sum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/norm_lp_power_p_sum.cpp

  #linear 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/linear/matmul.cpp

  #softmax 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/softmax/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/softmax/log_softmax.cpp

  #pooling 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_3d.cpp

  #conv
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_3d.cpp

  #normalization
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/batch_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/layer_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/rms_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/group_normalization.cpp

  
  #################################################
  #                     GRAPHS                    #
  #################################################


  #grad_graph 
  ${CMAKE_CURRENT_SOURCE_DIR}/graph/grad_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graph/export.cpp



  ##############################################
  #                  AUTOGRAD                  #
  ##############################################
  ${CMAKE_CURRENT_SOURCE_DIR}/autograd/engine.cpp


  ##############################################
  #                Kernels                     #
  ##############################################
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fused_adam_avx512.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fused_sgd_avx512.cpp

  
  ##############################################
  #                Neural Networks             #
  ##############################################

  ${CMAKE_CURRENT_SOURCE_DIR}/nn/core/module.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/core/sequential.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/init/init.cpp

  #layers 
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/linear.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/dropout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/embedding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/flatten.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/unflatten.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/transpose_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/pooling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/layers/normalization.cpp

  #activations

  ${CMAKE_CURRENT_SOURCE_DIR}/nn/activations/relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/activations/sigmoid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/activations/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/activations/tanh.cpp

  #optimizers
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/optimizers/optimizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/optimizers/sgd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/optimizers/adam.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/optimizers/adamw.cpp

  #losses  
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/losses/l1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/losses/mse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/losses/kl_divergence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/losses/cross_entropy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nn/losses/cross_entropy_with_logits.cpp


  ##############################################
  #                DATA HANDELING              #
  ##############################################

  #images 
  ${CMAKE_CURRENT_SOURCE_DIR}/data/image/image.cpp

  #mnist
  ${CMAKE_CURRENT_SOURCE_DIR}/data/mnist/mnist.cpp

  #dataset
  ${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/dataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/mnist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/image_folder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data/loader/loader.cpp

  ##############################################
  #                 TRAINER                    #
  ############################################## 

  ${CMAKE_CURRENT_SOURCE_DIR}/nn/train/export_import.cpp



) 


set_target_properties(MTensor PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  
  # RPATH settings for Linux shared library loading
  BUILD_RPATH_USE_ORIGIN TRUE
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)


target_compile_definitions(
  MTensor
  PRIVATE MTENSOR_EXPORTS
  PRIVATE mt_use_img
  PUBLIC  cimg_use_png
  PUBLIC  cimg_display=0
)


target_include_directories(MTensor PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(MTensor PUBLIC  DNNL::dnnl PNG::PNG ZLIB::ZLIB) 
#public cause of the ops.hpp(which will be used in the target that will link to Mtensor) needs dnnl.hpp.
#same for png, cimg needs it and i want to make it available to use by default

target_link_libraries(MTensor PRIVATE OpenMP::OpenMP_CXX)