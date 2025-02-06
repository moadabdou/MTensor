#include "./tensor.hpp"
#include <random>
#include <map>
#include <stdexcept>
#include <iostream>
#include "tensor.hpp"

void  indice_increament (std::vector<int>& c,const std::vector<int>& r ){
    for (int i = c.size() - 1; i  > -1 ; i--){
        c[i] += 1;
        if(c[i] >=  r[i]) c[i] = 0;
        else break;
    }
} 

void  indice_increament_steps (std::vector<int>& c,const std::vector<int>& r , const std::vector<int>& steps){
    for (int i = c.size() - 1; i  > -1 ; i--){
        c[i] += steps[i];
        if(c[i] >=  r[i]) c[i] = 0;
        else break;
    }
} 

void transpose_indices(std::vector<int>& indices, int dim1, int dim2){
    int tmp =  indices[dim1];
    indices[dim1] =  indices[dim2];
    indices[dim2] = tmp;
}

// Tensor of  Scalar

Tensor::Tensor(
    const std::vector<int>& _shape, 
    const std::vector<int>& _acc_indices,
    const std::shared_ptr<Scalar[]>& _data,  
    Scalar* const _own_data):  data(_data), own_data(_own_data) {//initialization  on a  portion of  existed Tensor
    setShape(_shape);
    if (_acc_indices.size()) acc_indices = _acc_indices; //override the calculated indices 
}


void Tensor::_print(std::ostream& os, int dim, std::vector<int>& positions) const{
    if  (dim ==  shape.size() - 1 && positions[dim-1] != 0){
        for (int i = 0;  i <  dim ; i++) os << " ";
    }
    os << "[";
    if (dim ==  shape.size() - 1){
        for (int i = 0; i  <  shape[dim] ; i++){
            positions[dim] =  i;
            long long offset = 0;
            for(int i = 0; i < shape.size() ; i++){
                offset += positions[i] * acc_indices[i];
            }
            os << (own_data + offset)->Item()  << (i == shape[dim] - 1 ? "": ", ");
        }
        os << "]"<< (dim == 0|| positions[dim-1] == shape[dim-1]-1 ? "": ",")<<"\n";
    }else {
        for (int i = 0; i  <  shape[dim] ; i++){
            positions[dim] =  i;
            _print(os,dim + 1,positions);
        }
        os << "]"<< ( dim == 0|| positions[dim-1] == shape[dim-1]-1 ? "": ",");
    }
}

void Tensor::broadCast(Tensor& other, int delimiter){
    std::vector<int>& bigShape = shape.size() > other.shape.size() ? shape : other.shape; 
    std::vector<int>& smallShape = shape.size() > other.shape.size() ? other.shape : shape; 

    v_shape = std::vector<int>(bigShape.size());
    other.v_shape = std::vector<int>(bigShape.size());
    int i;
    for (i = 0; i <  bigShape.size() - smallShape.size()  ; i++){
        other.v_shape[i] = v_shape[i] = bigShape[i];
    }

    int j;
    for(j = i ;  j <  bigShape.size() - delimiter; j ++){
        if (bigShape[j] == 1){
            v_shape[j] = smallShape[j-i];
            other.v_shape[j] = smallShape[j-i];
        }else if(smallShape[j-i] == 1 || smallShape[j-i] == bigShape[j] ){
            v_shape[j] = bigShape[j];
            other.v_shape[j] = bigShape[j];
        }else {
            throw std::runtime_error("error :  invalid broadcasting \n");
        }
    }

    int indice_corr = shape.size() == bigShape.size() ? 0 : i;
    int o_indice_corr = indice_corr == i ?  0 : i;
    for (int k = j; k  < bigShape.size() ;  k++){
        v_shape[k] = shape[k - indice_corr];
        other.v_shape[k] = other.shape[k- o_indice_corr];
    }
}

Scalar& Tensor::v_access(const std::vector<int>& params) const{
    if (params.size()  !=  v_shape.size()){
        throw std::runtime_error("error: when using v_access() consider to  give exact location of the Scalar \n");  
    }
    int zero_dims_ignore = v_shape.size() - shape.size();
    long long offset = 0;
    for (int i = zero_dims_ignore ; i <  v_shape.size() ;  i++){
        offset += acc_indices[i - zero_dims_ignore ] * ( params[i] % shape[i - zero_dims_ignore] );
    }
    return *(own_data + offset);
}

Scalar& Tensor::d_access(const std::vector<int>& params) const{
    if (params.size()  !=  shape.size()){
        throw std::runtime_error("error: when using d_access() consider to  give exact location of the Scalar \n");  
    }
    long long offset = 0;
    for (int i = 0 ; i <  shape.size() ;  i++){
        if (params[i] >=  shape[i]) throw std::runtime_error("error: d_access() : out_of_range \n");; 
        offset += acc_indices[i] * params[i];
    }
    return *(own_data + offset);
}


Tensor Tensor::_per_element_op(Tensor& other, std::function<Scalar(const Scalar&,const Scalar&)> op) {
    broadCast(other);
    Tensor newTensor(v_shape);
    std::vector<int> current(v_shape.size());
    for(int i = 0 ; i <  newTensor.size ;  i ++){
        newTensor.d_access(current) = op(v_access(current) ,other.v_access(current));
        indice_increament(current, v_shape);
    }
    return newTensor;
}

Tensor Tensor::_per_element_op_unair(std::function<Scalar(const Scalar&)> op) const{
    Tensor newTensor(shape);
    std::vector<int> current(shape.size());
    for(int i = 0 ; i <  newTensor.size  ;  i ++){
        newTensor.d_access(current) = op(d_access(current));
        indice_increament(current, shape);
    }
    return newTensor;
}

void Tensor::_per_element(std::function<void(Scalar&)> apply) const {
    std::vector<int> current(shape.size());
    for(int i = 0 ; i <  size  ;  i ++){
        apply(d_access(current));
        indice_increament(current, shape);
    }
}

void Tensor::_per_element_index(std::function<void(const std::vector<int> &_position, Scalar &)> apply) const{
    std::vector<int> current(shape.size());
    for(int i = 0 ; i <  size  ;  i ++){
        apply(current, d_access(current));
        indice_increament(current, shape);
    }
}

void  Tensor::setShape(const std::vector<int>& _shape){
    writing_pos.assign(_shape.size(), 0);  //setting w_pos to 0,0,.. 
    shape = _shape;  
    size = 1; 
    for (auto&& el: shape) size *= el;
    acc_indices = std::vector<int>(shape.size());
    long long acc;
    for (int i = 0 ; i < shape.size() ; i++){
        acc = 1;
        for (int j = i+1 ; j <  shape.size() ;  j++){
            acc *= shape[j];
        }
        acc_indices[i] = acc;
    }
}

Tensor::Tensor(): shape({0}), size(0) {};
Tensor::Tensor(const std::vector<int>& _shape){
    setShape(_shape);
    data = std::shared_ptr<Scalar[]>(new Scalar[size]); //auto  delete, no worries 
    own_data= data.get();
}

//operations

Tensor Tensor::Shape() const{
    Tensor T_shape({(int)shape.size()});
    T_shape = std::vector<double>(shape.begin(), shape.end());//to  convert  int to  double
    return T_shape;
}

std::vector<int> Tensor::Shape_v() const{
    return shape;
}
void Tensor::clamping(double lower, double upper){
    _per_element([&upper,&lower](const Scalar& a){
        if (a.Item() > upper){
            a.Item() = upper;
        }else if (a.Item() < lower){
            a.Item() = lower;
        }
    });
}

void Tensor::pointsToValue(const Tensor &other) const{
    if (shape.size() !=  other.shape.size() || shape !=  other.shape){
        throw  std::runtime_error("error:  pointsToValue() , both tensors have to be with same shape ");
    }
    std::vector<int> current(shape.size());
    for(int i = 0 ; i <  size ;  i ++){
        d_access(current) = other.d_access(current);
        indice_increament(current, shape);
    }
}

Tensor Tensor::Reshape(const std::vector<int>& __shape){
    std::vector<int> _shape =__shape;
    long long _size = 1;
    int minus1 = -1;
    for (int i = 0; i  < _shape.size() ;  i++) {
        if (_shape[i] == -1){
            if (minus1 == -1){
                minus1 = i;
            }else {
                throw std::runtime_error("error : reshape() deplicated -1, is  ambiguous ");
            }
        }else {   
            _size *= _shape[i];
        }
    }
    
    if (minus1 != -1){
        // size =  LKJH.. 
        // _size  = KJH 
        // L  = size / _size 
        _shape[minus1] = size / _size;
        _size = size;
    }

    if ( _size !=  size){
        throw std::runtime_error("error : reshape() invalide _shape ");
    }
    return Tensor(
        _shape, std::vector<int>() , data, own_data
    );
}

bool  is_valid_range(std::pair<int,int>& range, int max){
    /*
    -1 ,-1 <=> [0:max] 
    -1 , n <=> [0:n]  n <= max
    n , -1 <=> [n:max]  n < max 
    m , n <=> [m:n]  m < n 

    second <= max && first > -1 && second > first

    */
    if (range.first  == -1) range.first = 0;
    if (range.second  == -1) range.second = max;

    if (  range.first < range.second &&  range.second  <= max && range.second  > -1){
        return true;
    }

    return  false;
}

Tensor Tensor::Win(std::vector<std::pair<int,int>> shape_map) const {
    if ( shape_map.size() != shape.size() ){
        throw std::runtime_error("error : Win() invalide args, shape_map != shape");
    }
    
    long long _move_own = 0;
    std::vector<int> _new_shape(shape.size());

    for (int i = 0;  i < shape.size()  ; i++){
        if (! is_valid_range(shape_map[i], shape[i])){
            throw std::runtime_error("error : Win() invalide args, invalide range found !");
        }

        _new_shape[i] = shape_map[i].second - shape_map[i].first;
        _move_own += shape_map[i].first * acc_indices[i]; 
    }

    Tensor newTensor(_new_shape , acc_indices , data, own_data + _move_own);
    return newTensor;
}

Tensor Tensor::Transpose(int dim1 ,  int dim2){
    
    if ( dim1 < 0 || dim2 < 0 || dim1 >= shape.size() || dim2 >= shape.size() ){
        throw std::runtime_error("error : Transpose() invalide args ");
    }
    Tensor newTensor(shape, acc_indices , data, own_data);
    transpose_indices(newTensor.shape,dim1, dim2);
    transpose_indices(newTensor.acc_indices,dim1, dim2);
    return newTensor;
}

Scalar* Tensor::Data() const{
    return own_data;
}

Tensor Tensor::Size() const{
    Tensor _size({1});
    _size = {(double)size};
    return _size;
}
int Tensor::numSize() const{
    return size;
}

Tensor Tensor::clone() const {
    Tensor newTensor(shape);
    std::vector<int> current(shape.size()); 
    for(int i = 0 ; i <  size ;  i ++){
        newTensor.d_access(current).Item() = d_access(current).Item();
        indice_increament(current, shape);
    }
    return newTensor;
}


//Gradient
Tensor Tensor::Grad() const {
    Tensor newTensor(shape);
    std::vector<int> current(shape.size()); 
    for(int i = 0 ; i <  size ;  i ++){
        newTensor.d_access(current).Item() = d_access(current).Grad();
        indice_increament(current, shape);
    }
    return newTensor;
}

void Tensor::zeroGrad() const {
    _per_element([](Scalar& elm){
        elm.zeroGrad();
    });
}

void Tensor::zeroGrad_r() const {
    _per_element([](Scalar& elm){
        elm.zeroGrad_r();
    });
}

void Tensor::backward() const {
    _per_element([](Scalar& elm){
        elm.backward();
    });
}
//Math Functions 
Tensor Tensor::Exp() const{
    return _per_element_op_unair([](const Scalar& a){return Function::Exp(a);});
} 
Tensor Tensor::Sqrt() const{
    return _per_element_op_unair([](const Scalar& a){return Function::Sqrt(a);});
} 
Tensor Tensor::Pow(double p) const{
    return _per_element_op_unair([&p](const Scalar& a){return Function::Pow(a, p);});
} 
Tensor Tensor::Log() const{
    return _per_element_op_unair([](const Scalar& a){return Function::Log(a);});
} 


//access subTensors  


Tensor Tensor::operator() (const std::vector<int>& params) const {
    if (params.size()  >  shape.size()){
        throw std::runtime_error("error: cant  access the Tensor using more indices than the shape \n");  
    }
    std::vector<int> _shape = {1};
    std::vector<int> _acc_indices = {1};
    if (params.size() < shape.size()){
        _shape.assign(shape.begin() + params.size(), shape.end());
        _acc_indices.assign(acc_indices.begin() + params.size(), acc_indices.end());
    } 
    long long offset = 0; 
    for (int i = 0; i <  params.size() ;  i++){
        if (params[i] >= shape[i]){
            throw std::runtime_error("error: cant  access the Tensor , out_of_range\n");
        }
        offset += params[i] * acc_indices[i];
    }
    return Tensor(
        _shape,_acc_indices, data, own_data + offset
    );
}

//artithmic operations

Tensor& Tensor::operator = (const std::vector<double>& elemets){
    if (shape.size() != 1){
        throw std::runtime_error("error: trying to  assign values to non-1d tensor \n");
    }
    int _stop = std::min<double>(shape[0], elemets.size()); 
    for (int i= 0; i <  _stop ;  i++){
        (own_data+i)->Item() =  elemets[i];
    }
    return *this;
} 

// Addition
Tensor Tensor::operator+(Tensor& other) {
    return _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a + b; });
}
Tensor Tensor::operator+(Tensor&& other) {
    return *this + other;
}
Tensor& Tensor::operator+=(Tensor& other) {
    return *this = _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a + b; });
}

// Subtraction
Tensor Tensor::operator-(Tensor& other) {
    return _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a - b; });
}
Tensor Tensor::operator-(Tensor&& other) {
    return *this - other;
}
Tensor& Tensor::operator-=(Tensor& other) {
    return *this = _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a - b; });
}
Tensor Tensor::operator-() {
    return _per_element_op_unair([](const Scalar& a) { return -a; });
}

// Multiplication
Tensor Tensor::operator*(Tensor& other) {
    return _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a * b; });
}
Tensor Tensor::operator*(Tensor&& other) {
    return *this * other;
}

Tensor& Tensor::operator*=(Tensor& other) {
    return *this = _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a * b; });
}

// Division
Tensor Tensor::operator/(Tensor& other) {
    return _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a / b; });
}
Tensor Tensor::operator/(Tensor&& other) {
    return *this / other;
}
Tensor& Tensor::operator/=(Tensor& other) {
    return *this = _per_element_op(other, [](const Scalar& a, const Scalar& b) { return a / b; });
}

Tensor &Tensor::operator,(const double &val){
    d_access(writing_pos).Item() =  val;
    indice_increament(writing_pos, shape);
    return *this;
}

Tensor &Tensor::operator,(const char &_char){
    writing_pos.assign(shape.size(), 0);
    return *this;
}

// Matrix Multiplication
Tensor Tensor::Matmul(Tensor& other) {
    broadCast(other, 2);
    if (v_shape.back() != *(other.v_shape.end() - 2)) {
        throw std::runtime_error("error : Matmul(), invalid dims for the matrix multiplication");
    }
    
    std::vector<int> newShape;
    newShape.assign(v_shape.begin(), v_shape.end() - 2);
    newShape.insert(newShape.end(), {*(v_shape.end() - 2), other.v_shape.back()});
    Tensor newTensor({newShape});

    std::vector<int> c_current(newShape.size()), a_current, b_current;
    int r = v_shape.back(); 
    for (int i = 0; i < newTensor.size; i++) {
        a_current = b_current = c_current;
        for (int j = 0; j < r; j++) {
            a_current.back() = j;
            *(b_current.end() - 2) = j;
            newTensor.d_access(c_current) += v_access(a_current) * other.v_access(b_current);
        }
        indice_increament(c_current, newTensor.shape);
    }
    return newTensor;
}
Tensor Tensor::Matmul(Tensor&& other) {
    return this->Matmul(other);
}


//Padding2d

Tensor Tensor::Padding2d(int  padding ) const{
    if (shape.size() != 4){
        throw std::runtime_error(" error : Padding2d()  is limited to 4D objects ");
    }
    Tensor tensor({ shape[0],shape[1], shape[2]+ 2*padding , shape[3]+ 2*padding });
    std::vector<int> current(4); 
    for(int i = 0; i <  size ;  i++){
        tensor.d_access({ current[0],current[1], current[2]+ padding , current[3]+ padding  }) = d_access(current); // making elements  from tensor  pointing to the same point data as *this
        indice_increament(current, shape);
    }
    return tensor;
}

//crop2d

Tensor Tensor::Crop2d(int  padding ) const{
    if (shape.size() != 4){
        throw std::runtime_error(" error : Crop2d()  is limited to 4D objects ");
    }
    Tensor tensor({ shape[0],shape[1], shape[2] - 2*padding , shape[3] -  2*padding });
    std::vector<int> current(4); 
    for(int i = 0; i <  size ;  i++){
        tensor.d_access(current) = d_access({ current[0],current[1], current[2]+ padding , current[3]+ padding  }); // making elements  from tensor  pointing to the same point data as *this
        indice_increament(current, tensor.shape);
    }
    return tensor;
}

//Conv2d

Tensor Tensor::Conv2d( const Tensor& Kernel, const Tensor& Bias, int stride , int padding) const {

    if ( shape.size() != 4  || Kernel.shape.size() != 4 || Bias.shape.size() != 1){
        throw std::runtime_error(" error : Conv2d()  both input and  kernel have  to be 4D dims and Bias 1D or null-Tensor, X(B,C,H,W), K(F,C,KH,KW)");
    }
    if ( shape[1] !=  Kernel.shape[1] || Kernel.shape[0] !=  Bias.shape[0] ){
        throw std::runtime_error("error : both Kernel and Input dont  have same channel number, and Bias size must  the same as  Filter Number or null-Tensor");
    }
    // X(B,C,H,W) Cnv2d K(F,C,KH,KW) = (B, F, Hout, Wout)
    // Hout = (H +2P - KH)/S + 1 , Wout = (W +2P - KW)/S + 1
    std::vector<int> outputShape =   {shape[0], Kernel.shape[0], 
                                ( shape[2] + 2*padding - Kernel.shape[2]) / stride + 1 ,
                                ( shape[3] + 2*padding - Kernel.shape[3]) / stride + 1};
    std::vector<int> sp_shape = { shape[1] , Kernel.shape[2] ,  Kernel.shape[3]};
    int sp_size = shape[1] * Kernel.shape[2] * Kernel.shape[3] ;
    std::vector<int> outCurrent(4);
    Tensor output(outputShape);
    Tensor X_padding = Padding2d(padding); 
    for (int i= 0; i  < output.size ;  i++){
        std::vector<int> sp_current(3);
        for (int j = 0; j  < sp_size ; j++){
            output.d_access(outCurrent) +=  
            X_padding.d_access({ outCurrent[0], sp_current[0],  stride * outCurrent[2] + sp_current[1],  stride * outCurrent[3] + sp_current[2]})
            * Kernel.d_access({ outCurrent[1], sp_current[0],sp_current[1], sp_current[2] })
            + Bias.d_access({outCurrent[1]});
            indice_increament(sp_current, sp_shape);
        }
        indice_increament(outCurrent, outputShape);
    }

    return output;
}

//Upscale2d  

Tensor Tensor::Upscale2d(int  k_h, int k_w , int stride ) const{
    if (shape.size() != 4 ){
        throw std::runtime_error(" error : Upscale2d()  is limited to 4D objects ");
    }
     if (k_h <= 0 || k_w <= 0 ){
        throw std::runtime_error(" error : Upscale2d() k_h<=0 or k_w<=0 is not allowed ");
    }
    //stride * (w - 1)  + 1 + 2*k_w - 2 = stride * (w - 1) - 1 * 2k_w
    Tensor tensor({ shape[0],shape[1], stride * (shape[2]-1) - 1 + 2*k_h , stride * (shape[3]-1) - 1 + 2*k_w});
    std::vector<int> current(4); 
    for(int i = 0; i <  size ;  i++){
        tensor.d_access({ current[0],current[1], stride * current[2]+ k_h - 1, stride * current[3]+ k_w - 1 }) = d_access(current); // making elements  from tensor  pointing to the same point data as *this
        indice_increament(current, shape);
    }
    return tensor;
}

//TransposeConv2d

Tensor Tensor::TransposeConv2d( const Tensor& Kernel, const Tensor& Bias, int stride , int padding) const {
    if ( shape.size() != 4  || Kernel.shape.size() != 4 || Bias.shape.size() != 1){
        throw std::runtime_error(" error : TransposeConv2d()  both input and  kernel have  to be 4D dims and Bias 1D, X(B,C,H,W), K(F,C,KH,KW)");
    }

    if ( shape[1] !=  Kernel.shape[1] || Kernel.shape[0] !=  Bias.shape[0] ){
        throw std::runtime_error("error : both Kernel and Input dont  have same channel number, and Bias size must  the same as  Filter Number ");
    }

    Tensor output = Upscale2d(Kernel.shape[2], Kernel.shape[3], stride).Conv2d(Kernel, Bias);
    if (padding){
        output = output.Crop2d(padding);
    }
    return output;
}


//Sum 
Tensor Tensor::Sum(int dim) const{
    if ( dim >= (int)shape.size()){
        throw std::runtime_error("error: Sum()  invalid dim arg");
    }

    if (dim == -1){
        Tensor res({1});
        _per_element([&res](Scalar& elm){
            (*res.Data()) +=  elm; // only  one  element,  no problem with using the  pointer
        });
        return res;
    }

    std::vector<int> newShape = shape;
    newShape.erase(newShape.begin() + dim);
    Tensor newTensor(newShape);
    int loop_size = 1;
    std::vector<int> current(newShape.size()),  origine_current;

    for (int i = 0;  i < newTensor.size ; i++){
        origine_current =  current;
        origine_current.insert( origine_current.begin() + dim , 0 );
        
        for (int j = 0;  j <  shape[dim]  ; j++){
            origine_current[dim] = j;
            newTensor.d_access(current) += d_access(origine_current);
        }

        indice_increament(current, newShape);
    }
    
    newShape.insert( newShape.begin() + dim , 1 );
    return newTensor.Reshape(newShape);
}

//Mean 
Tensor Tensor::Mean(int dim) const{
    if ( dim >= (int)shape.size()){
        throw std::runtime_error("error: Sum()  invalid dim arg");
    }

    if (dim == -1){
        return Sum()/(this->Size());
    }
    return Sum(dim) / this->Shape()(dim);
}

//initialiazers 

Tensor Tensor::Randn(const std::vector<int>& shape, double stddev , double mean , int seed ){
    Tensor tensor(shape);
    std::random_device rd; //seed
    std::mt19937 gen(seed == -1 ? rd() : seed); //  setting the  generator 
    std::normal_distribution<> dist(mean, stddev);// setting the distribution 
    for  (int i = 0 ; i < tensor.size ; i++){
        (tensor.Data() + i)->Item() = dist(gen);
    }

    return tensor;
}

Tensor Tensor::Rand(const std::vector<int>& shape, double start , double end ,int seed){
    Tensor tensor(shape);
    std::random_device rd; //seed
    std::mt19937 gen(seed == -1 ? rd() : seed);  
    std::uniform_real_distribution<> dist(start, end);// setting the distribution 
    for  (int i = 0 ; i < tensor.size ; i++){
        (tensor.Data() + i)->Item() = dist(gen);
    }

    return tensor;
}


Tensor Tensor::Fill(const std::vector<int>& shape, double value){
    Tensor tensor(shape); 
    for  (int i = 0 ; i < tensor.size ; i++){
        (tensor.Data() + i)->Item() = value;
    }

    return tensor;
}

Tensor Tensor::Ones(const std::vector<int>& shape){
    return Tensor::Fill(shape , 1);
}

Tensor Tensor::Zeros(const std::vector<int>& shape){
    return Tensor::Fill(shape , 0);
}


std::ostream& operator << (std::ostream& os,const Tensor& tensor){
    os << "Tensor(\n";
    std::vector<int> starting_pos(tensor.shape.size());
    tensor._print(os,0, starting_pos);
    os << ")";
    return os;
}