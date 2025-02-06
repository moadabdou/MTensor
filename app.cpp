#include "./tensor/tensor.hpp"
#include "./nn/linear/linear.hpp"
#include "./nn/conv2d/conv2d.hpp"
#include "./nn/transposedconv2d/transposedconv2d.hpp"
#include "./nn/module.hpp"
#include "./nn/Functional/Functional.hpp"
#include "./nn/optimizer/optimizer.hpp"
#include <iostream>
#include <vector>
using namespace std;

class testModule:  public Module{
public:
    Linear lin1;
    Linear lin2;
    Conv2d conv1;
    TransposedConv2d tconv1;
    testModule(){
        lin1 = setSubModule(Linear(16*2,4));
        lin2 = setSubModule(Linear(4,1));
        conv1 = setSubModule(Conv2d(1,2,2, 2));
        tconv1 = setSubModule(TransposedConv2d(1,1,2));
    }

    Tensor forward(Tensor x) override{
        x = tconv1(x);
        x = Functional::ReLU(x); 
        x = conv1(x); //
        x = Functional::ReLU(x); 
        x = x.Reshape({x.Shape_v()[0], 16*2});
        x = lin1(x);
        x = Functional::Segmoid(x);
        x = lin2(x);
        x = Functional::Segmoid(x);
        return x;
    }
};

int main(){
    Tensor X = Tensor::Randn({4,1,4,4} , 5, 20);
    Tensor Y = Tensor::Ones({4,1});
    testModule M;
    vector<Tensor> params = M.parameters();
    optimizers::SGD optimizer(params , 1);
    cout << optimizer.totalParameters();
    cout << M(X);
    for (int i = 0;  i <  1000 ;  i++){
        Tensor loss = Functional::MSE(M(X), Y);
        optimizer.Zero_grad();
        loss.backward();
        optimizer.Step(); 
    }
    cout << M(X);

    return 0;
}