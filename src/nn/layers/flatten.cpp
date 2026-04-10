#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    FlattenImpl::FlattenImpl(){}

    Tensor FlattenImpl::forward(Tensor input){
        int64_t N = input.shape()[0];
        return input.reshape({ N , input.numel()/N});
    }

    MTENSOR_API std::shared_ptr<Module> Flatten(){
        return std::make_shared<FlattenImpl>();
    }

}//nn
}//mt
