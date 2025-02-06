#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "../module.hpp"
#include "../../tensor/tensor.hpp"


class Conv2d: public Module{ 
public:
    Tensor wieght;
    Tensor bias;
    int padding;
    int stride;
    bool is_bias;
    Conv2d(){};
    Conv2d(int in_channels, int out_channels, int kernel_size ,int stride=1,int padding=0, bool Bias= true);
    Tensor forward(Tensor x) override;
};

#endif // CONV2D_HPP