#include "./scalar.hpp"
#include "./operation/operation.hpp"
#include <stdexcept>
#include <iostream>
Scalar::Scalar(){
    wrapped_value = std::make_shared<ScalarValue>((ScalarValue){
        .value = 0, 
        .operation = std::make_shared<Initilization>()
    });
}


Scalar::Scalar(const double value, std::shared_ptr<Operation> operation){
    if (!operation){
       operation = std::make_shared<Initilization>() ;
    }
    wrapped_value = std::make_shared<ScalarValue>((ScalarValue){
        .value = value , 
        .operation = operation
    });
}

Scalar::Scalar(const std::shared_ptr<ScalarValue> scalarValue){
    wrapped_value = scalarValue;
}

double& Scalar::Item() const {
    return wrapped_value->value;
}

double Scalar::Grad() const{
    return wrapped_value->grad;
}

void Scalar::zeroGrad(){
    wrapped_value->grad = 0;
}

void Scalar::zeroGrad_r(){
    zeroGrad();
    for (auto&& child : wrapped_value->children){
        Scalar(child).zeroGrad_r();
    }
}

std::shared_ptr<ScalarValue> Scalar::Value() const{
    return wrapped_value;
}

void Scalar::_backward(){
    //std::cout << "back called\n" ;
    wrapped_value->operation->grad(*this);
    if ((!dynamic_cast<Initilization*>( wrapped_value->operation.get()))) {
        wrapped_value->grad = 0;
    }
    for(auto&& child :  wrapped_value->children){
        if (!dynamic_cast<Initilization*>( child->operation.get())){
            Scalar(child)._backward();
        }
    }
    
}

void Scalar::backward(){
    wrapped_value->grad = 1; 
    _backward();
}

Scalar Scalar::operator+(const Scalar& other ) const{
    return Addition().eval( (std::vector<Scalar>){*this, other});
}
Scalar& Scalar::operator+=(const Scalar& other ){
    return *this = Addition().eval( (std::vector<Scalar>){*this, other});
}

Scalar Scalar::operator*(const Scalar &other) const{
    return Multiplication().eval((std::vector<Scalar>){*this, other});
}
Scalar& Scalar::operator*=(const Scalar &other){
    return *this = Multiplication().eval((std::vector<Scalar>){*this, other});
}

Scalar Scalar::operator-(const Scalar &other) const{
    return Substraction().eval((std::vector<Scalar>){*this, other});
}
Scalar& Scalar::operator-=(const Scalar &other){
    return *this = Substraction().eval((std::vector<Scalar>){*this, other});
}

Scalar Scalar::operator/(const Scalar &other) const{
    return Division().eval((std::vector<Scalar>){*this, other});
}
Scalar& Scalar::operator/=(const Scalar &other){
    return *this = Division().eval((std::vector<Scalar>){*this, other});
}

Scalar Scalar::operator-() const{
    return Scalar(-1) * (*this);
}
