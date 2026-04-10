#include <vector>
#include <stdexcept>
#include <cstdint>
#include <omp.h>

namespace mt {
namespace utils{

class TensorIterator {
public:
    TensorIterator(std::vector<int64_t> shape, std::vector<int64_t> strides);

    int64_t get_total_elements() const { return total_elements_; }

    int64_t get_flat_index_from_coords(const std::vector<int64_t>& coords) const;

    template<typename F>
    void parallel_for_each(const F& func) const {
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            int64_t items_per_thread = total_elements_ / num_threads;
            int64_t start_k = thread_id * items_per_thread;
            int64_t end_k = (thread_id == num_threads - 1) ? total_elements_ : start_k + items_per_thread;

            if (start_k < end_k) {
                std::vector<int64_t> coords = get_coords_from_iteration_num(start_k);
                for (int64_t k = start_k; k < end_k; ++k) {
                    func(coords, k); // Pass coordinates and logical index
                    if (k < end_k - 1) {
                        increment_coords_by_one(coords);
                    }
                }
            }
        }
    }

private:
    std::vector<int64_t> get_coords_from_iteration_num(int64_t iteration_num) const;

    void increment_coords_by_one(std::vector<int64_t>& coords) const;

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t total_elements_;
};

}//utils

}// mt
