curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get update
sudo apt-get install cargo



cargo run --example shallownet_mnist --release 
cargo run --example lenet_small_cifar --release 
cargo run --example lenet_medium_cifar --release
cargo run --example lenet_small_face --release
cargo run --example lenet_medium_face --release
cargo run --example lenet_large_face --release 
