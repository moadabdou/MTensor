#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include "../operation/operation.hpp"

class Scalar;

//Math functions 

//exp
class Exponential:public Operation{
public: 
    void grad(const Scalar& scalar) override;    
};
namespace Function{
    Scalar Exp(const Scalar &scalar);
}

//ln 
class Logarithm:public Operation{
public: 
    void grad(const Scalar& scalar) override;    
};
namespace Function{
    Scalar Log(const Scalar &scalar);
}

//pow 

class Power:public Operation{
public: 
    double p;
    Power (double _p): p(_p) {}
    void grad(const Scalar& scalar) override;    
};
namespace Function{
    Scalar Pow(const Scalar &scalar, double  p);
}

//sqrt 
class SquarRoot:public Operation{
public: 
    void grad(const Scalar& scalar) override;    
};
namespace Function{
    Scalar Sqrt(const Scalar &scalar);
}

//max , Min

namespace Function{
    Scalar Max(const Scalar &a,const  Scalar &b);
    Scalar Min(const Scalar &a,const  Scalar &b);
}

#endif // FUNCTIONS_HPP
