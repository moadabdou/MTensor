#include "./function.hpp"
#include "../scalar.hpp"
#include <cmath>

Scalar  Function::Exp(const Scalar &scalar){
    Scalar newScalar( std::exp(scalar.Item()) , std::make_shared<Exponential>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {scalar.Value()});
    return newScalar;
}

void Exponential::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad +=  std::exp(value.children[0]->value) *  scalar.Value()->grad ;
}


Scalar  Function::Log(const Scalar &scalar){
    Scalar newScalar( std::log(scalar.Item()) , std::make_shared<Logarithm>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {scalar.Value()});
    return newScalar;
}

void Logarithm::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    if (value.children[0]->value > 0){  
        value.children[0]->grad +=  1/value.children[0]->value *  scalar.Value()->grad ;
    }else {
        value.children[0]->grad +=  std::nan("") ;
    }
}


Scalar  Function::Pow(const Scalar &scalar, double p){
    Scalar newScalar( std::pow(scalar.Item(), p) , std::make_shared<Power>(Power(p)));
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {scalar.Value()});
    return newScalar;
}


void Power::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad +=  p * std::pow(value.children[0]->value, p - 1) *  scalar.Value()->grad ;
}


Scalar  Function::Sqrt(const Scalar &scalar){
    Scalar newScalar( std::sqrt(scalar.Item()) , std::make_shared<SquarRoot>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {scalar.Value()});
    return newScalar;
}

void SquarRoot::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad +=  (1 / (2*std::sqrt(value.children[0]->value))) *  scalar.Value()->grad ;
}


Scalar Function::Max(const Scalar &a,const Scalar &b){
    return a.Item() > b.Item() ? a  :  b;
}

Scalar Function::Min(const Scalar &a,const Scalar &b){
    return a.Item() > b.Item() ? b  :  a;
}

