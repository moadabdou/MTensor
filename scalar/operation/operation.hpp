#ifndef OPERATION_HPP
#define OPERATION_HPP

#include <vector>

struct ScalarValue;
class Scalar;


class Operation{
public:
    virtual Scalar eval(const std::vector<Scalar>& operands);
    virtual void grad(const Scalar& scalar);
};

class Initilization: public Operation{
public:
    Scalar eval(const std::vector<Scalar>& operands) override;
    void grad(const Scalar& scalar) override;
};

//basic  operations 
class Addition :  public Operation{
public: 
    Scalar eval(const std::vector<Scalar>& operands) override;
    void grad(const Scalar& scalar) override;
};

class Multiplication : public Operation{
public: 
    Scalar eval(const std::vector<Scalar>& operands) override;
    void grad(const Scalar& scalar) override;  
};

class Substraction : public Operation{ 
public: 
    Scalar eval(const std::vector<Scalar>& operands) override;
    void grad(const Scalar& scalar) override;    
};

class Division: public Operation{
public: 
    Scalar eval(const std::vector<Scalar>& operands) override;
    void grad(const Scalar& scalar) override;    
};

#endif // OPERATION_HPP
