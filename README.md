
<img src="./assets/logo.png" width="45" align="left">

## 🧠 What is MTensor?

**MTensor** is a CPU-based deep learning library written in C++, inspired by PyTorch's design. It supports dynamic graphs, training, autograd, and a wide range of tensor and neural network operations.

> ⚠️ **Beta Notice**: MTensor 2.0 is tested on **Windows** (MSVC 17) and **Linux** (GCC 11+/Clang 14+). macOS support is coming soon.

---

## 🚀 Technologies Used

* **oneDNN** – High-performance tensor primitives
* **OpenMP** – Threaded parallelism
* **AVX512** – Custom fused operations for optimizers, reductions
* **CImg + libpng** – Image handling support
* **nlohmann/json** – Exporting graphs and manual tensor initialization
* **vcpkg** – Dependency manager

---

## 🧱 Features

### 🔹 Core Tensor API

* Internal `TensorImpl` with smart pointers
* Tensor methods: `shape()`, `stride()`, `numel()`, `randn()`, `ones()`, `arange()`, `fill()`, ...
* Torch-style formatted printing
* Broadcasting support (binary ops, matmul)

### 🔹 Tensor Operations

* **Arithmetic & Logic**: `+`, `-`, `*`, `/`, `<`, `>`, `==`, `min`, `max`
* **Elementwise**: `abs`, `clip`, `exp`, `log`, `pow`, `relu`, `sigmoid`, ...
* **Reduction**: `sum`, `mean`, `min`, `max`, `mul`, `norm_lp_*`, ...
* **Normalization**: `batchnorm`, `groupnorm`, `layernorm`, `rmsnorm`
* **Convolution**: `conv1d/2d/3d`, `transposedConv1d/2d/3d`
* **Pooling**: `avgpool`, `maxpool` (1d, 2d, 3d)
* **Memory & Views**: `clone`, `reshape`, `permute`, `transpose`, `squeeze`, `contiguous`
* **Matrix ops**: `matmul`, `A @ B + C`
* **Other**: `embedding`, `softmax`, `logsoftmax`, `cat()`, `stack()`

### 🔹 Autograd & Graphing

* Dynamic computation graph with `backward()`
* Tensor gradient helpers: `requires_grad`, `detach`, `grad`, ...
* Visualize autograd graph as an interactive HTML (`build_graph()`)

---

## 🧠 Neural Networks

* **Layers**: `Linear`, `Conv`, `TransposeConv`, `Embedding`, `Dropout`, `Flatten`, `Unflatten`, `Sequential`
* **Activations**: `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`
* **Losses**: `MSELoss`, `L1Loss`, `CrossEntropy`, `KLDiv`, `BCEWithLogits`
* **Optimizers** (AVX512-fused): `SGD`, `MomentumSGD`, `Adam`, `AdamW`
* **Initializers**: Kaiming and Xavier (Uniform/Normal)

---

## 🗂️ Data Handling

* **Datasets**:

  * `MNISTDataset` (reads MNIST binary format)
  * `ImageFolderDataset` (uses CImg for loading images)
* **Dataloader**: For training batches
* **Image I/O**: `image_to_tensor()`, `tensor_to_image()` (via CImg)

---

## ⚙️ Installation

> 🧪 MTensor 2.0 is tested on **Windows** (MSVC 17) and **Linux** (GCC 11+/Clang 14+). All dependencies are managed with `vcpkg`.

---

### Prerequisites

#### Windows
- Visual Studio 2022 (MSVC 17)
- CMake 3.31.6+

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install -y build-essential ninja-build cmake libomp-dev
```

#### Linux (Fedora/RHEL)
```bash
sudo dnf install -y gcc-c++ ninja-build cmake libomp-devel
```

#### Linux (Arch Linux / CachyOS)
```bash
sudo pacman -S gcc ninja cmake openmp
```

---

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/your-username/MTensor.git
cd MTensor
```

> ✅ `vcpkg` is included as a submodule under `external/vcpkg`.

---

### 2. Set Up vcpkg

If needed:

```bash
git submodule update --init --recursive
```

Bootstrap vcpkg:

```bash
cd external/vcpkg
./bootstrap-vcpkg.bat     # Windows
./bootstrap-vcpkg.sh      # Linux
```

Install dependencies:

```bash
./vcpkg install
```

> 📦 Dependencies are defined in `vcpkg.json`.

> 🔧 On Linux, vcpkg will automatically use the `x64-linux` triplet.

---

### 3. Configure the Project

#### Windows (Visual Studio 2022)

Use the default CMake preset:

```bash
cmake --preset windows-vs2022
```

#### Linux (Ninja)

```bash
cmake --preset linux-ninja
```

> 🛠️ To change generator or toolchain, edit `CMakePresets.json`.

---

### 4. Build the Project

#### Windows

Build in **release** mode:

```bash
cmake --build --preset release
```

Or in **debug** mode:

```bash
cmake --build --preset debug
```

#### Linux

Build in **release** mode:

```bash
cmake --build --preset linux-release
```

Or in **debug** mode:

```bash
cmake --build --preset linux-debug
```

> 🧪 The build produces a shared library (`MTensor.dll` on Windows, `libMTensor.so` on Linux) in the `build/` directory.

---

## 📌 Roadmap

* ✅ Tensor & autograd engine
* ✅ Neural network layers and optimizers
* ✅ Dataset/dataloader system
* ✅ Linux support
* 🔜 External project integration
* 🔜 macOS support
* 🔜 Python bindings
* 🔜 Training UI (browser-based)
