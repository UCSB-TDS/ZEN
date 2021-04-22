# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# sudo apt-get update
# sudo apt-get install cargo
# mkdir test-data
# cargo run --bin gen_data --release 



#baseline
# cargo run --bin shallownet_naive_mnist --release > shallownet_mnist_naive_commitment.log
cargo run --bin lenet_small_naive_pedersen_cifar --release > lenet_small_cifar_naive_commitment.log
cargo run --bin lenet_medium_naive_pedersen_cifar --release > lenet_medium_cifar_naive_commitment.log
cargo run --bin lenet_small_naive_pedersen_face --release > lenet_small_face_naive_commitment.log
cargo run --bin lenet_medium_naive_pedersen_face --release > lenet_medium_face_naive_commitment.log
cargo run --bin lenet_large_naive_pedersen_face --release  > lenet_large_face_naive_commitment.log





