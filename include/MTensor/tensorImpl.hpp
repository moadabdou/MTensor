#ifndef MTENSOR_H
#define MTENSOR_H
#include <vector>
#include <functional>
#include <memory>
#include <config/mtensor_export.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/utils/tensor_print.hpp>
#include <MTensor/utils/braodcast.hpp>
#include <MTensor/utils/print_vector.hpp>

namespace mt{


std::vector<int64_t> row_major_stride( const std::vector<int64_t>& shape);

class MTENSOR_API TensorImpl{

public:

    TensorImpl(float* data, const std::vector<int64_t>& shape, bool requires_grad = false);
    TensorImpl(const std::vector<int64_t>& shape, bool requires_grad = false);
    TensorImpl(std::function<float(int64_t)> func, const std::vector<int64_t>& shape, bool requires_grad = false);
        
    //to initlize from another tensor, this will be used primerly for view operations
    TensorImpl(
        const std::shared_ptr<float>& data,  //of another tensorImpl
        const int64_t& data_offset,
        const std::vector<int64_t>& shape,
        const std::shared_ptr<ops::Operation>& grad_fn, 
        bool requires_grad,
        bool is_contiguous,
        const std::vector<int64_t>& stride = {}
    );

    TensorImpl(const TensorImpl& other) = delete;
    TensorImpl operator=(const TensorImpl& other) = delete;

    static std::shared_ptr<TensorImpl> zeros(const std::vector<int64_t>& shape, bool requires_grad = false);
    static std::shared_ptr<TensorImpl> ones(const std::vector<int64_t>& shape, bool requires_grad = false);
    static std::shared_ptr<TensorImpl> randn(const std::vector<int64_t>& shape, float mean = 0, float stddev = 1, int64_t seed = -1, bool requires_grad = false);
    static std::shared_ptr<TensorImpl> rand(const std::vector<int64_t>& shape,float lower_bound = 0, float upper_bound = 1, int64_t seed = -1, bool requires_grad = false);

    std::shared_ptr<TensorImpl> detach() const;

    std::vector<int64_t> shape() const;
    std::vector<int64_t> stride() const;
    
    int64_t data_offset() const;
    int64_t numel() const;
    std::shared_ptr<float> data_ptr() const;

    bool requires_grad() const;
    void set_requires_grad( bool value );
    // std::shared_ptr<TensorImpl> grad() const;
    std::shared_ptr<ops::Operation> grad_fn() const;

    bool is_contiguous() const;
    bool is_leaf() const;

    std::string get_name() const ;
    void set_name(std::string name = "");

    const std::shared_ptr<TensorImpl>& get_grad() const;
    void set_grad( const std::shared_ptr<TensorImpl>& grad);
    void set_grad_fn( const std::shared_ptr<ops::Operation>& grad_fn);

    void set_data_ptr(std::shared_ptr<float> m_data);

    static bool is_valid_shape(const std::vector<int64_t>& shape);

    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<TensorImpl>& tensor_impl){
            if (!tensor_impl) {
                os << "TensorImpl(nullptr)";
                return os;
            }
            const auto& shape = tensor_impl->shape();
            const auto& stride = tensor_impl->stride();
            const float* data = tensor_impl->data_ptr().get();
            const int64_t data_offset = tensor_impl->data_offset();
            int64_t numel = tensor_impl->numel();
            os << "Tensor(";
            utils::print_tensor(data,data_offset, shape, stride, numel);
            
            if (tensor_impl->is_leaf()){
                if (tensor_impl->requires_grad()){
                    std::cout << ", requires_grad = True";
                }
            }else {
                std::cout << ", grad_fn = "<< tensor_impl->grad_fn()->name();
            }
            
            std::cout << ")"<< std::endl;
            return os;
    }

private:


    void init_tensor_desc(bool allocate = true);

    std::string m_name;

    bool m_requires_grad = false;
    std::shared_ptr<TensorImpl> m_grad = nullptr;
    std::shared_ptr<ops::Operation> m_grad_fn;

    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_stride;

    int64_t m_data_offset = 0;
    int64_t m_numel = 0;
    std::shared_ptr<float> m_data;

    bool m_is_contiguous = true;

};

} //mt


#endif //MTENSOR_H