#include "./Functional.hpp"
#include "../../tensor/tensor.hpp"
#include "../../scalar/function/function.hpp"
#include "../../scalar/scalar.hpp"
#include "Functional.hpp"
#include <stdexcept>
#include <iostream>

Tensor Functional::ReLU(const Tensor &tensor){
    return tensor._per_element_op_unair([](const Scalar& a){
        return Function::Max(a, Scalar(0));
    });
}

Tensor Functional::Leaky_ReLU(const Tensor &tensor, double alfa ){
    return tensor._per_element_op_unair([alfa](const Scalar& a){
        return a.Item()  >  0 ?  a : a * Scalar(alfa);
    });
}

Tensor Functional::Segmoid(const Tensor &tensor){
    return tensor._per_element_op_unair([](const Scalar& a){
        return Scalar(1)/(Scalar(1) + Function::Exp(-a));
    });
}

Tensor Functional::Softmax(const Tensor &tensor, int dim ){
    return tensor.Exp() /  tensor.Exp().Sum(dim);
}



Tensor Functional::MSE(const Tensor &_X,const Tensor &_Y){
    Tensor X = _X, Y = _Y; //this to  make them non constant, because - does a broadcasting
    return (X-Y).Pow(2).Reshape({X.Shape_v()[0], -1}).Sum(1).Mean();
}



Tensor Functional::FullCrossEntropy(Tensor X, Tensor Y){
    return X._per_element_op(Y, [](const Scalar& p, const Scalar&y ){
        return -(y*Function::Log(p) + (Scalar(1) - y)*Function::Log(Scalar(1) - p));
    }).Reshape({X.Shape_v()[0], -1}).Sum(1).Mean();
}

Tensor Functional::CrossEntropy(Tensor X, Tensor Y){
    return X._per_element_op(Y, [](const Scalar& p, const Scalar&y ){
        return -(y*Function::Log(p));
    }).Reshape({X.Shape_v()[0], -1}).Sum(1).Mean();
}

