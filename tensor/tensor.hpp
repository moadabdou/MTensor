#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <functional>
#include <memory>
#include "../scalar/scalar.hpp"
#include "../scalar/function/function.hpp"

class Scalar;

// Function declarations for helper methods
void indice_increament(std::vector<int>& c, const std::vector<int>& r);
void indice_increament_steps(std::vector<int>& c, const std::vector<int>& r, const std::vector<int>& steps);
void transpose_indices(std::vector<int>& indices, int dim1, int dim2);

// Tensor class definition
class Tensor {
private:
    long long size;
    std::vector<int> shape;
    std::vector<int> v_shape;
    std::vector<int> acc_indices;
    std::shared_ptr<Scalar[]> data;
    Scalar* own_data;

    std::vector<int> writing_pos;

    void setShape(const std::vector<int>& _shape);
    void _print(std::ostream& os, int dim, std::vector<int>& positions) const;
    void broadCast(Tensor& other, int delimiter = 0);
    Tensor(const std::vector<int>& _shape, const std::vector<int>& _acc_indices, 
           const std::shared_ptr<Scalar[]>& _data, Scalar* const _own_data);

public:
    Scalar& v_access(const std::vector<int>& params) const;
    Scalar& d_access(const std::vector<int>& params) const;
    
    Tensor _per_element_op(Tensor& other, std::function<Scalar(const Scalar&, const Scalar&)> op);
    Tensor _per_element_op_unair(std::function<Scalar(const Scalar&)> op) const;
    void _per_element(std::function<void(Scalar &)> apply) const;
    void _per_element_index(std::function<void(const std::vector<int>& _position, Scalar &)> apply) const;

    Tensor();
    Tensor(const std::vector<int>& _shape);

    void pointsToValue(const Tensor& other) const;
    Tensor Reshape(const std::vector<int>& _shape);
    Tensor Transpose(int dim1, int dim2);
    Scalar* Data() const;
    Tensor Size() const;
    int numSize() const;
    Tensor clone() const;

    // Gradient functions
    Tensor Grad() const;
    void zeroGrad() const;
    void zeroGrad_r() const;
    void backward() const;

    // Math functions
    Tensor Exp() const;
    Tensor Sqrt() const;
    Tensor Pow(double p) const;
    Tensor Log() const;

    // Access subTensors
    template <typename... Args, typename = std::enable_if_t<(std::is_integral_v<Args>&&...)>>
    Tensor operator()(Args... args) {
        std::vector<int> params =  {args...}; 
        return (*this)(params);
    }

    Tensor operator()(const std::vector<int>& params) const;

    // Arithmetic operations
    Tensor& operator=(const std::vector<double>& elements);
    Tensor operator+(Tensor& other);
    Tensor operator+(Tensor&& other);
    Tensor& operator+=(Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator-(Tensor&& other);
    Tensor& operator-=(Tensor& other);
    Tensor operator-();
    Tensor operator*(Tensor& other);
    Tensor operator*(Tensor&& other);
    Tensor& operator*=(Tensor& other);
    Tensor operator/(Tensor& other);
    Tensor operator/(Tensor&& other);
    Tensor& operator/=(Tensor& other);
    friend std::ostream& operator << (std::ostream& os,const Tensor& tensor) ;

    Tensor& operator,(const double& val);
    Tensor& operator,(const char& _char);

    Tensor Sum(int dim = -1) const;
    Tensor Mean(int dim = -1) const;
    
    static Tensor Fill(const std::vector<int>& shape, double value);
    static Tensor Randn(const std::vector<int>& shape, double stddev = 1 , double mean = 0, int seed = -1);
    static Tensor Rand(const std::vector<int>& shape, double start = 0 , double end = 1,int seed = -1);
    static Tensor Ones(const std::vector<int>& shape);
    static Tensor Zeros(const std::vector<int>& shape);
    
    
    // Matrix multiplication
    Tensor Matmul(Tensor& other);
    Tensor Matmul(Tensor&& other);

    Tensor Upscale2d(int  k_h, int k_w , int stride ) const;
    Tensor TransposeConv2d( const Tensor& Kernel, const Tensor& Bias, int stride = 2, int padding = 0) const;
    
    // Padding2d
    Tensor Padding2d(int padding) const;

    // Crop2d
    Tensor Crop2d(int padding) const;

    // Conv2d
    Tensor Conv2d(const Tensor& Kernel, const Tensor& Bias, int stride = 1, int padding = 0) const;

    // Shape getter
    Tensor Shape() const;
    std::vector<int> Shape_v() const;
    //window
    Tensor Win(std::vector<std::pair<int,int>> shape_map) const;

    void clamping(double lower, double upper);
};

#endif // TENSOR_HPP
