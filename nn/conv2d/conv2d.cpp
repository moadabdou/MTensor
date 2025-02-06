#include "./conv2d.hpp"
#include <cmath>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size ,int _stride ,int _padding , bool Bias )
:is_bias(Bias), padding(_padding), stride(_stride){
    //Kaiming/He initialization 
    double mean = 0;
    double stddev = std::sqrt(2.0 / in_channels );
    wieght = setParameter(Tensor::Randn({out_channels, in_channels, kernel_size, kernel_size},stddev, mean));
    if (Bias){
        bias = setParameter(Tensor::Zeros({out_channels}));
    }else {
        bias = Tensor::Zeros({out_channels});
    }
}

Tensor Conv2d::forward(Tensor x){
    return x.Conv2d(wieght, bias, stride, padding);
}