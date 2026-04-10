#include <MTensor/utils/tensor_iterator.hpp>
namespace mt {
namespace utils {

    TensorIterator::TensorIterator(std::vector<int64_t> shape, std::vector<int64_t> strides)
        : shape_(std::move(shape)), strides_(std::move(strides)) {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("Shape and strides must have the same number of dimensions.");
        }
        total_elements_ = 1;
        for (int64_t dim_size : shape_) {
            if (dim_size < 0) throw std::invalid_argument("Dimension size cannot be negative.");
            total_elements_ *= dim_size;
        }
    }

    int64_t  TensorIterator::get_flat_index_from_coords(const std::vector<int64_t>& coords) const {
        int64_t flat_index = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            flat_index += coords[i] * strides_[i];
        }
        return flat_index;
    }

    std::vector<int64_t> TensorIterator::get_coords_from_iteration_num(int64_t iteration_num) const {
        std::vector<int64_t> coords(shape_.size(), 0);
        int64_t current_index = iteration_num;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (shape_[i] > 0) {
                coords[i] = current_index % shape_[i];
                current_index /= shape_[i];
            }
        }
        return coords;
    }

    void TensorIterator::increment_coords_by_one(std::vector<int64_t>& coords) const {
        for (int i = shape_.size() - 1; i >= 0; --i) {
            coords[i]++;
            if (coords[i] < shape_[i]) return;
            coords[i] = 0;
        }
    }

}//utils
}//mt
