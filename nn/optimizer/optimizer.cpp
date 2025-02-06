#include "optimizer.hpp"
#include "../../tensor/tensor.hpp"
#include "../../scalar/scalar.hpp"
#include <cmath>
#include <iostream>

//general
Optimizer::Optimizer(const std::vector<Tensor>& _params, double _lr):parameters(_params), lr(_lr) {}

void Optimizer::Zero_grad(){

    for (auto&& p : parameters){
        p.zeroGrad();
    }
    
}

long long Optimizer::totalParameters() const{
    long long res = 0;
    for (auto&& p : parameters){
        res += p.numSize();
    }
    return res;
}

//SGD

optimizers::SGD::SGD(const std::vector<Tensor> &_params, double _lr): Optimizer(_params, _lr){}

void optimizers::SGD::Step(){
    for (auto&& p :  parameters){
        p._per_element([this](Scalar& ps){
            ps.Item() -= lr * ps.Grad();
        });
    }
}

optimizers::Adam::Adam(const std::vector<Tensor> &_params, double _lr): Optimizer(_params, _lr){
    for (auto&& p :  parameters){
        m.push_back(Tensor({p.Shape_v()}));
        v.push_back(Tensor({p.Shape_v()}));
    }
}

void optimizers::Adam::Step(){
    t++;
    for (int i = 0; i <  parameters.size() ;  i++){
        parameters[i]._per_element_index([i,this](const std::vector<int>& _position, Scalar& ps){

            double& p_m =  m[i].d_access(_position).Item(); 
            double& p_v = v[i].d_access(_position).Item(); 
            double  grad = ps.Grad();
            p_m =  beta1*p_m + (1 - beta1)*  grad ;
            p_v =  beta2*p_v + (1 - beta2)*  grad * grad; 
 
            
            double p_m_hat = p_m / (1 - std::pow(beta1 , t));
            double p_v_hat = p_v / (1 - std::pow(beta2 , t));

            ps.Item() -= lr * p_m_hat / (std::sqrt(p_v_hat) + epsilon);
        });
    }
}
