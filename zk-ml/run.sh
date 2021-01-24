curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get update
sudo apt-get install cargo



#optimization level3
#cargo run --bin shallownet_optimized_pedersen_mnist --release > shallownet_mnist_optimized.log
cargo run --bin lenet_small_optimized_pedersen_cifar --release > lenet_small_cifar_optimized_wire.log
cargo run --bin lenet_medium_optimized_pedersen_cifar --release > lenet_medium_cifar_optimized_wire.log
cargo run --bin lenet_small_optimized_pedersen_face --release > lenet_small_face_optimized_wire.log
cargo run --bin lenet_medium_optimized_pedersen_face --release > lenet_medium_face_optimized.log
cargo run --bin lenet_large_optimized_pedersen_face --release > lenet_large_face_optimized.log




