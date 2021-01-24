# ZK statement for machine learning


# HowTo
* Under directory `ZEN/zk-ml-baseline/`, run `mkdir test-data` and `cargo run --bin gen_data` to generate the mock inputs for baseline and microbenchmark purposes only. For optimization level 3, we load real quantization parameters generated from `ZEN/numpyInferenceEngine/XXNet/`. For example, cd to `ZEN/numpyInferenceEngine/LeNet_CIFAR10` and run `python3.8 LeNet_end_to_end_quant.py --model LeNet_Small`. The generated quantized parameters are located at `ZEN/numpyInferenceEngine/LeNet_CIFAR10/LeNet_CIFAR_pretrained/`. For easily reproducing the results, we have saved a copy of parameters for all combinations of model and dataset in director `ZEN/zk-ml/pretrained_model/`


# Commitments

REMINDER: before benchmarking, please use the corresponding `num_window` parameter. *For different input image size, the commitment setting is different*. Refer to pedersen_commit.rs and please ensure you use the correct `num_window` parameter for corresponding dataset.
* Pedersen input size = window_size * num_window / 8
* pub const PERDERSON_WINDOW_SIZE: usize = 25; // this is for 28X28X1 MNIST input
* pub const PERDERSON_WINDOW_SIZE: usize = 100; // this is for 32X32X3 CIFAR u8 input
* pub const PERDERSON_WINDOW_SIZE: usize = 100; // this is for 46X56X1 FACE u8 input

# Optmization level 3 for all combinations of models and datasets
* `cargo run --bin shallownet_optimized_pedersen_mnist --release 2>/dev/null`
* `cargo run --bin lenet_small_optimized_pedersen_cifar --release 2>/dev/null`
* `cargo run --bin lenet_medium_optimized_pedersen_cifar --release 2>/dev/null`
* `cargo run --bin lenet_small_optimized_pedersen_face --release 2>/dev/null`
* `cargo run --bin lenet_medium_optimized_pedersen_face --release 2>/dev/null`
* `cargo run --bin lenet_large_optimized_pedersen_face --release 2>/dev/null`








