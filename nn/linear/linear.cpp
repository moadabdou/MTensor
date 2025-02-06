#include "./linear.hpp"
#include <cmath>
#include <iostream>
Linear::Linear(int in, int out, bool Bias ):is_bias(Bias){
    // Xavier/Glorot uniform initialization
    double r = std::sqrt(6.0 / (in + out));
    wieght = setParameter(Tensor::Rand({in,out}, -r, r));
    if (Bias){
        bias = setParameter(Tensor::Zeros({out}));
    }
}

Tensor Linear::forward(Tensor x){
    x = x.Matmul(wieght);
    return is_bias ? x + bias : x ;
}
