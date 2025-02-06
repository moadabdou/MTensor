#include "./pooling.hpp"
#include "../../tensor/tensor.hpp"
#include "../../scalar/function/function.hpp"

#include <vector>
#include <stdexcept>
MaxPooling2d::MaxPooling2d(int _pool_size): pool_size(_pool_size), Module() {}

Tensor MaxPooling2d::forward(Tensor x){

    if (x.Shape_v().size() !=  4){
        throw std::runtime_error("Error :  MaxPooling2d()  only 4D Tensors are allowed ");
    }

    std::vector<int> newShape = x.Shape_v();
    newShape[2] /= pool_size;
    newShape[3] /= pool_size;

    Tensor newTensor(newShape);

    newTensor._per_element_index([x, this](const std::vector<int>&  position, Scalar& a){
        for(int i = 0; i < pool_size;  i++){
            for(int j = 0; j  <  pool_size ; j++){
                a = Function::Max(a ,  x.d_access({ position[0], position[1], pool_size*position[2] + i , pool_size * position[3] + j }));
            }
        }
    });

    return newTensor;
}