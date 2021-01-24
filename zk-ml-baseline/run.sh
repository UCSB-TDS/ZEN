curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get update
sudo apt-get install cargo
mkdir test-data
cargo run --bin gen_data --release 

#microbenchmarks

cargo run --bin microbench_lenet_medium_cifar_naive --release  > lenet_medium_cifar_microbench_naive.log
cargo run --bin microbench_lenet_medium_cifar_op1 --release  > lenet_medium_cifar_microbench_op1.log
cargo run --bin microbench_lenet_medium_cifar_op2 --release  > lenet_medium_cifar_microbench_op2.log

cargo run --bin microbench_conv_layered_optimization_by_kernel_size --release > conv_layered_optimization.log
cargo run --bin microbench_fc_layered_optimization --release > fc_layered_optimization.log

#baseline
# cargo run --bin shallownet_naive_mnist --release > shallownet_mnist_naive.log
cargo run --bin lenet_small_naive_pedersen_cifar --release > lenet_small_cifar_naive.log
cargo run --bin lenet_medium_naive_pedersen_cifar --release > lenet_medium_cifar_naive.log
cargo run --bin lenet_small_naive_pedersen_face --release > lenet_small_face_naive.log
cargo run --bin lenet_medium_naive_pedersen_face --release > lenet_medium_face_naive.log
cargo run --bin lenet_large_naive_pedersen_face --release  > lenet_large_face_naive.log





