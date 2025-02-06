#ifndef MODULE_HPP
#define MODULE_HPP

#include <vector>
#include <stdexcept>
#include <memory>
class Tensor;
/*
Module ? 
  > Parameters (vector<Tensor>) : public
     | Linear : 0 => Weights, 1 => Bias
  > SubModels  (vector<Modules>) : public
  > Tensor forward:virtual , ()  operators :  public
  > vector<Tensor> parameters : function , does a recursive collection of parameters 
*/

class Module{
    void _parameters( std::vector<Tensor>& collecter ) const;
    std::vector<Module> subModules;
public : 
    Module();
    std::vector<Tensor> Paramerters;

    Tensor operator()(Tensor& x);

    virtual Tensor forward(Tensor x);

    std::vector<Tensor> parameters() const;

    template<typename T>
    T& setSubModule(T &&module){
        if (!std::is_base_of<Module,T>::value){
            throw std::runtime_error("error : setSubModule() can't Add non-Module elements to subModuls");
        }
        subModules.push_back(module);
        return module;
    }

    Tensor setParameter(const Tensor& param);
    /*
    any  thing pushed to the subModules will lose its special properties,
    there is no problem with that when it comes to the use of subModules, 
    thus this  function  return the original Module
    */

};


#endif // MODULE_HPP