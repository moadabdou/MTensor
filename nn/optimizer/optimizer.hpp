#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP
#include <vector>

class Tensor;

//general 
class Optimizer {
public:
    std::vector<Tensor> parameters;
    double lr;

    Optimizer(const std::vector<Tensor>& _params, double _lr);
    void Zero_grad();
    virtual void Step(){};
    long long totalParameters() const;
};

namespace optimizers{

    class SGD: public Optimizer{
    public:
        SGD( const std::vector<Tensor>& _params, double _lr );
        void Step() override;
    };

    class Adam: public Optimizer{
        std::vector<Tensor> m;
        std::vector<Tensor> v;
    public:
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        int  t = 0;
        Adam( const std::vector<Tensor>& _params, double _lr );
        void Step() override;
    };

}


#endif //OPTIMIZER_HPP