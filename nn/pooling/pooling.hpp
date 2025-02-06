#ifndef POOLING_HPP
#define POOLING_HPP

#include "../module.hpp"

class MaxPooling2d: public Module{
    int pool_size ;
public:
    MaxPooling2d(int _pool_size =  2);
    Tensor forward(Tensor x) override;
};


#endif //POOLING_HPP