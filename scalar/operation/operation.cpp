#include "./operation.hpp"
#include "../scalar.hpp"


Scalar Operation::eval(const std::vector<Scalar>& operands){
    return Scalar();
};

void Operation::grad(const Scalar& scalar){
};


Scalar Initilization::eval(const std::vector<Scalar> &operands){
    return Scalar();
}

void Initilization::grad(const Scalar &scalar){
}

Scalar Addition::eval(const std::vector<Scalar>& operands){
    Scalar newScalar( operands[0].Item() + operands[1].Item(), std::make_shared<Addition>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {operands[0].Value(), operands[1].Value()});
    return newScalar;
}

void Addition::grad(const Scalar& scalar){
    ScalarValue& value = *scalar.Value();
    for (auto&& child : value.children){
        child->grad += 1 * scalar.Value()->grad;
    }
};


Scalar Multiplication::eval(const std::vector<Scalar> &operands){
    Scalar newScalar( operands[0].Item() * operands[1].Item(), std::make_shared<Multiplication>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {operands[0].Value(), operands[1].Value()});
    return newScalar;
}

void Multiplication::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad += value.children[1]->value * scalar.Value()->grad;
    value.children[1]->grad += value.children[0]->value * scalar.Value()->grad ;
}

Scalar Substraction::eval(const std::vector<Scalar> &operands) {
    Scalar newScalar( operands[0].Item() - operands[1].Item(), std::make_shared<Substraction>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {operands[0].Value(), operands[1].Value()});
    return newScalar;
}

void Substraction::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad += 1 * scalar.Value()->grad;
    value.children[1]->grad += -1 * scalar.Value()->grad ;
}
Scalar Division::eval(const std::vector<Scalar> &operands){
    Scalar newScalar( operands[0].Item() / operands[1].Item(), std::make_shared<Division>());
    ScalarValue&  value = *newScalar.Value();
    value.children.insert(value.children.end() , {operands[0].Value(), operands[1].Value()});
    return newScalar;
}

void Division::grad(const Scalar &scalar){
    ScalarValue& value = *scalar.Value();
    value.children[0]->grad +=   scalar.Value()->grad / value.children[1]->value  ;
    value.children[1]->grad += -value.children[0]->value / ( value.children[1]->value * value.children[1]->value ) * scalar.Value()->grad ;
}
