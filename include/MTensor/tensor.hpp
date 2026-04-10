#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <config/mtensor_export.hpp>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/graph.hpp>

namespace mt {

    using Shape = std::vector<int64_t>;
    using Dim   = int64_t;
    struct values_indices;

    class  MTENSOR_API Tensor{

        public: 

            Tensor(){}
            Tensor(const std::shared_ptr<TensorImpl>& tensor_impl);
            Tensor(float scalar);
            Tensor(const json& input);

            //******************************************/
            //                Tensor
            //******************************************/

            //_________________  info  ____________________

            Shape shape() const;
            Shape stride() const;
            int64_t data_offset() const;
            int64_t numel() const;
            float* data_ptr() const;
            bool is_contiguous() const;
            bool is_leaf() const;
            bool is_scalar() const;
            std::string get_name() const;
            float item() const;
            void set_name(std::string name);
            std::shared_ptr<TensorImpl> tensor_impl() const;

            friend std::ostream& operator<<(std::ostream& os,const mt::Tensor& tensor){
                os << tensor.m_tensor_impl;
                return os;
            }

            //_________________  initializers  ____________________
    
            static Tensor randn(const Shape& shape, float mean = 0.0f ,float stddev = 1.0f, int64_t seed = -1, bool requires_grad = false);
            static Tensor rand(const Shape& shape, float lbound = 0.0f ,float ubound = 1.0f, int64_t seed = -1,  bool requires_grad = false);
            static Tensor fill(const Shape& shape, float value, bool requires_grad = false);
            static Tensor ones(const Shape& shape, bool requires_grad = false);
            static Tensor ones_like(const Tensor& tensor, bool requires_grad = false);
            static Tensor zeros_like(const Tensor& tensor, bool requires_grad = false);
            static Tensor zeros(const Shape& shape, bool requires_grad = false);
            static Tensor arange(float start , float end, float step = 1.0f, bool requires_grad = false);
            static Tensor empty(const Shape& shape);

            //******************************************/
            //                operations
            //******************************************/

            //________________ view _______________

            Tensor expand(const Shape& shape) const;
            Tensor narrow(Dim dim, Dim start, Dim end, Dim length) const;
            Tensor permute(const Shape& permute) const;
            Tensor slice(const sliceList& slice_list) const;
            Tensor operator[](const sliceList& slice_list) const;
            Tensor squeeze(Dim dim) const;
            Tensor unsqueeze(Dim dim) const;
            Tensor transpose(Dim dim_0, Dim dim_1) const;
            Tensor view(const Shape& shape) const;
            Tensor reshape(const Shape& shape) const;
            Tensor flat() const;

            //________________ arithmetic _______________
            Tensor operator+(const Tensor& other) const;
            Tensor operator-(const Tensor& other) const;
            Tensor operator/(const Tensor& other) const;
            Tensor operator*(const Tensor& other) const;
            Tensor operator+(const float& scalar) const;
            Tensor operator-(const float& scalar) const;
            Tensor operator/(const float& scalar) const;
            Tensor operator*(const float& scalar) const;
            friend Tensor operator+(const float& scalar, const Tensor& tensor){
                return tensor + scalar;
            }
            friend Tensor operator-(const float& scalar, const Tensor& tensor){
                return tensor.linear(-1.f, scalar);
            }
            friend Tensor operator/(const float& scalar, const Tensor& tensor){
                return tensor.pow(-1.0f, scalar);
            }
            friend Tensor operator*(const float& scalar, const Tensor& tensor){
                return tensor * scalar;
            }

            Tensor operator+=(const Tensor& other);
            Tensor operator-=(const Tensor& other);
            Tensor operator/=(const Tensor& other);
            Tensor operator*=(const Tensor& other);

            Tensor max(const Tensor& other) const;
            Tensor min(const Tensor& other) const;

            //________________ logic _______________
            Tensor operator<(const Tensor& other) const;
            Tensor operator<=(const Tensor& other) const;
            Tensor operator>(const Tensor& other) const;
            Tensor operator>=(const Tensor& other) const;
            Tensor operator==(const Tensor& other) const;
            Tensor operator!=(const Tensor& other) const;

            //________________ eltwise ____________
            Tensor abs() const;
            Tensor clip(float lbound, float ubound) const;
            Tensor exp() const;
            Tensor linear(float alpha, float beta = 0.0f) const;
            Tensor log() const;
            Tensor pow(float exposant, float scale = 1.0f) const;
            Tensor relu(float alpha = 0.0f) const;
            Tensor simgoid() const;
            Tensor sqrt() const;
            Tensor tanh() const;


            //________________ joining ____________
            static Tensor cat(const std::vector<Tensor>& tensors, Dim dim);
            static Tensor stack(const std::vector<Tensor>& tensors, Dim dim);

            //________________ linear _____________

            Tensor matmul(const Tensor& other) const;
            
            //________________ memory _____________ 
            Tensor clone() const;
            Tensor contiguous() const;

            //________________ reduction ___________ 
            values_indices max(Dim dim) const;
            values_indices min(Dim dim) const;
            Tensor mean(Dim dim = EOD, float eps=1e-10f) const;
            Tensor mul(Shape dims) const;
            Tensor norm_lp_power_p_sum(Shape dims, float p, float eps=1e-10f) const;
            Tensor norm_lp_sum(Shape dims, float p, float eps=1e-10f) const;
            Tensor sum(Shape dims = {}) const;

            //_______________ softmax ____________

            Tensor softmax(Dim dim) const;
            Tensor logsoftmax(Dim dim) const;

            
            //******************************************/
            //                autograd
            //******************************************/

            Tensor grad() const;
            ops::Operation* grad_fn() const; //todo: make a wrapper for Operation
            Tensor detach() const;
            void backward(const Tensor& grad_out, bool retain_graph = false) const;
            void backward(bool retain_graph = false) const;
            bool requires_grad() const;
            void make_requires_grad();
            graph::GradGraph build_graph() const;
                       
        private:
            std::shared_ptr<TensorImpl> m_tensor_impl;

    };
 
    struct values_indices{
        Tensor values;
        std::vector<Shape> indices;
    };

}//mt


//to print vectors easly istead if using utils::...

#ifdef mt_print_vector
std::ostream& operator<<(std::ostream& os, mt::Shape dims){
    mt::utils::print_vector(dims);
    return os;
}
#endif

#endif //TENSOR_HPP