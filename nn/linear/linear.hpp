#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "../module.hpp"
#include "../../tensor/tensor.hpp"


class Linear: public Module{ 
public:
    Tensor wieght;
    Tensor bias;
    bool is_bias;
    Linear(){};
    Linear(int in, int out, bool Bias = true);
    Tensor forward(Tensor x) override;
};

#endif // LINEAR_HPP