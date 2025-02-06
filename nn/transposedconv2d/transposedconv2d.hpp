#ifndef TRANSPOSEDCONV2D_HPP
#define TRANSPOSEDCONV2D_HPP

#include "../module.hpp"
#include "../../tensor/tensor.hpp"


class TransposedConv2d: public Module{ 
public:
    Tensor wieght;
    Tensor bias;
    int padding;
    int stride;
    bool is_bias;
    TransposedConv2d(){};
    TransposedConv2d(int in_channels, int out_channels, int kernel_size ,int stride=2,int padding=0, bool Bias= true);
    Tensor forward(Tensor x) override;
};

#endif // TRANSPOSEDCONV2D_HPP