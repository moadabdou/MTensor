#include <iostream>
#include <string>
#include <memory>
#include <algorithm>
#define mt_print_vector
#include <MTensor/tensor.hpp>
#include <MTensor/nn.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/data.hpp>

using namespace mt;

/******************************************************/
/*              LeNet-5 Model Definition              */
/******************************************************/

class LeNet5 : public nn::Module {
private:
    nn::m net = module(nn::Sequential({
        nn::Conv2d(1, 6, {5, 5}, true, {1, 1}, {0, 0}, {0, 0}),
        nn::MaxPooling2d({2, 2}, {2, 2}),
        nn::Relu(),
        nn::Conv2d(6, 16, {5, 5}, true, {1, 1}, {0, 0}, {0, 0}),
        nn::MaxPooling2d({2, 2}, {2, 2}),
        nn::Relu(),
        nn::Flatten(),
        nn::Linear(256, 120),
        nn::Relu(),
        nn::Linear(120, 84),
        nn::Relu(),
        nn::Linear(84, 10)
    }));

public:
    Tensor forward(Tensor input) override {
        return net->forward(input);
    }
};


/******************************************************/
/*                  Training Function                 */
/******************************************************/

void train(LeNet5& model, int epochs, int batch_size, float learning_rate) {

    std::cout << "Generating dummy training data..." << std::endl;

    int64_t num_samples = 256;
    auto dummy_images = Tensor::randn({num_samples, 1, 28, 28});
    auto dummy_labels = Tensor::rand({num_samples, 10}, 0.0f, 1.0f);

    std::cout << "Training samples: " << num_samples << std::endl;
    std::cout << "Images shape: " << dummy_images.shape() << std::endl;

    nn::CrossEntropyLoss loss_fn;
    auto params = model.paramters();
    nn::optimizer::Adam optimizer(params, learning_rate);

    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;

    int64_t steps_per_epoch = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int64_t step = 0;

        while (step * batch_size < num_samples) {
            int64_t start = step * batch_size;
            int64_t end = std::min(start + batch_size, num_samples);

            auto inputs = dummy_images.slice({{start, end}}).contiguous();
            auto targets = dummy_labels.slice({{start, end}}).contiguous();

            auto outputs = model(inputs);
            auto loss_tensor = loss_fn(outputs, targets);
            float loss = loss_tensor.item();

            loss_tensor.backward();
            optimizer.step();
            optimizer.zero_grad();

            epoch_loss += loss;

            if (step % 10 == 0) {
                float avg_loss = epoch_loss / (step + 1);
                std::cout << "  Epoch [" << (epoch + 1) << "/" << epochs
                          << "] Step [" << step << "/" << steps_per_epoch
                          << "] Loss: " << avg_loss << std::endl;
            }

            step++;
        }

        float avg_epoch_loss = epoch_loss / steps_per_epoch;
        std::cout << "===== Epoch [" << (epoch + 1) << "/" << epochs
                  << "] Avg Loss: " << avg_epoch_loss << " =====" << std::endl;
    }
}


/******************************************************/
/*                        Main                        */
/******************************************************/

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "     LeNet-5 Training with MTensor      " << std::endl;
    std::cout << "========================================\n" << std::endl;

    int epochs = 200;
    int batch_size = 128;
    float learning_rate = 0.001f;

    if (argc > 1) epochs = std::atoi(argv[1]);
    if (argc > 2) batch_size = std::atoi(argv[2]);
    if (argc > 3) learning_rate = std::atof(argv[3]);

    std::cout << "Initializing LeNet-5 model..." << std::endl;
    LeNet5 model;

    auto params = model.paramters();
    std::cout << "Model has " << params.size() << " parameter tensors" << std::endl;

    train(model, epochs, batch_size, learning_rate);

    std::cout << "\nTraining completed!" << std::endl;
    return 0;
}
