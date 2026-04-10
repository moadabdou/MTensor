#include <MTensor/data.hpp>
#include <MTensor/tensor.hpp>
#include <omp.h>

namespace mt{

    class TensorImpl;
    class Tensor;

namespace data{


    // /************************************************/
    // //               image & tensor                 //
    // /************************************************/

    MTENSOR_API Tensor image_to_tensor(const cimg_library::CImg<float>& img){

        int IW = img.width(), IH = img.height(), IC = img.spectrum();

        auto tensor = Tensor::empty({1, IC, IH, IW});

        auto data_ptr = tensor.data_ptr();

        #pragma omp parallel for
        for (int c = 0; c < IC; ++c) {
            for (int y = 0; y < IH; ++y) {
                for (int x = 0; x < IW; ++x) {
                    float pixel = img(x, y, 0, c);
                    data_ptr[c * IH * IW + y * IW + x] = pixel ;
                }
            }
        }

        return tensor;
    }

    MTENSOR_API cimg_library::CImg<float> tensor_to_image(const Tensor& tensor){

        const auto& shape = tensor.shape();

        if (shape.size() != 4 || shape[0] != 1 || shape[1] > 4){
            throw std::invalid_argument(" tensot_to_image(): input tensor must be 4D tensor with N=1 (single element in the batch) and C <= 4");
        }

        const auto& stride = tensor.stride();
        auto data_ptr = tensor.data_ptr() + tensor.data_offset();


        int64_t IC = shape[1], IH = shape[2], IW = shape[3];

        auto img = cimg_library::CImg<float>( IW, IH, 1, IC, 0);

        #pragma omp parallel for
        for (int c = 0; c < IC; ++c) {
            for (int y = 0; y < IH; ++y) {
                for (int x = 0; x < IW; ++x) {
                    img(x, y, 0, c) = data_ptr[c * stride[1] + y * stride[2] + x * stride[3]];
                }
            }
        }

        return img;
    }

}//data

} //mt