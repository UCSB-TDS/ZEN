build_test(){
  exit_code=0
      rustup override set 1.47.0
      cargo build --verbose --manifest-path "./zk-ml-private-model/Cargo.toml" || exit_code=1 
      cargo build --verbose --manifest-path "./zk-ml-private-model-baseline/Cargo.toml" || exit_code=1 
      cargo build --verbose --manifest-path "./zk-ml-accuracy/Cargo.toml" || exit_code=1 
      rustup override set 1.51.0
      cargo build --verbose --manifest-path "./zen-arkworks/Cargo.toml" || exit_code=1
  exit $exit_code
}

find . -name 'Cargo.toml' | sort -u | build_test