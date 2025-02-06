#include "./module.hpp"
#include "../Tensor/tensor.hpp"
#include <typeinfo>
#include <stdexcept>
#include "module.hpp"
#include <iostream>

Module::Module(){}

Tensor Module::operator()(Tensor& x){
    return forward(x);
}


Tensor Module::forward(Tensor x){
    std::wcerr<< "warning: default  forward()  triggered, consider to override forward \n";
    return x; //defualt behavior
}

void  Module::_parameters( std::vector<Tensor>& collecter ) const{
    for (auto &&p : Paramerters){
        collecter.push_back(p);
    }
    for (auto &&module : subModules){
        module._parameters(collecter);
    }
}

std::vector<Tensor> Module::parameters() const{
    std::vector<Tensor> collecter;
    _parameters(collecter);
    return collecter;
}


Tensor Module::setParameter(const Tensor& param){
    Paramerters.push_back(param);
    return param;
}
