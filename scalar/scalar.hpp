#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <vector>
#include <memory>

class Operation;


struct ScalarValue{
    double  value;
    double  grad = 0;
    std::vector<std::shared_ptr<ScalarValue>> children;
    std::shared_ptr<Operation> operation;
};


// scalar  Type  :  smallest  unit
class Scalar{
    std::shared_ptr<ScalarValue> wrapped_value;
public:
    Scalar();
    Scalar(const double value, const std::shared_ptr<Operation> operation = nullptr);
    Scalar(const std::shared_ptr<ScalarValue> scalarValue);
    double& Item() const;
    double Grad() const;
    void  zeroGrad();
    void  zeroGrad_r();
    std::shared_ptr<ScalarValue> Value() const;
    void _backward();
    void backward();
    Scalar operator+ (const Scalar& other ) const;
    Scalar& operator+= (const Scalar& other ) ;
    Scalar operator* (const Scalar& other ) const;
    Scalar& operator*= (const Scalar& other ) ;
    Scalar operator- (const Scalar& other ) const;
    Scalar& operator-= (const Scalar& other ) ;
    Scalar operator/ (const Scalar& other ) const;
    Scalar& operator/= (const Scalar& other ) ;
    Scalar operator-() const;
};





#endif // SCALAR_HPP
