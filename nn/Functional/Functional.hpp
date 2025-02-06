#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

class Tensor;
namespace Functional{
    //activation
    Tensor ReLU(const Tensor& tensor);
    Tensor Leaky_ReLU(const Tensor &tensor, double alfa = 0.01);
    Tensor Segmoid(const Tensor& tensor);
    Tensor Softmax(const Tensor& tensor, int dim = -1);

    //loss
    Tensor MSE(const Tensor& X,const Tensor& Y);
    Tensor CrossEntropy(Tensor X,Tensor Y);
    Tensor FullCrossEntropy(Tensor X, Tensor Y);
}

#endif // FUNCTIONAL_HPP