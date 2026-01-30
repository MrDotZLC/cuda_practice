# flash-attention-minimal
A minimal re-implementation of Flash Attention with CUDA and libtorch c++, while learning flash-attention by debuging C++ program.

This project forked from [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) and used [libtorch](https://docs.pytorch.org/cppdocs/installing.html) instead of PyTorch.

The official [implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner
(like myself), so this repo tries to be small and educational.

## Usage
### Prerequisite
* PyTorch (with CUDA)
* libtorch (C++, )
* `Ninja` for loading in C++
* CMake

### Debug Config
* .vscode/launch.json -> configurations.program
* CMakeLists.txt -> target_compile_options