use rand::RngCore;
use std::fs::OpenOptions;
use std::io::Write;

fn gen_lenet_small_cifar_i8() {
    let mut rng = rand::thread_rng();
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv1_i8.data")
        .unwrap();

    for _ in 0..(6 * 3 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv2_i8.data")
        .unwrap();

    for _ in 0..(16 * 6 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv3_i8.data")
        .unwrap();

    for _ in 0..(120 * 16 * 4 * 4) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc1_i8.data")
        .unwrap();

    for _ in 0..(84 * 120) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc1_cifar_i8.data")
        .unwrap();

    for _ in 0..(84 * 120 * 2 * 2) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc2_i8.data")
        .unwrap();

    for _ in 0..(10 * 84) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }
}

fn gen_dataset() {
    let mut rng = rand::thread_rng();

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/mnist.data")
        .unwrap();

    for _ in 0..784 {
        let tmp = ((rng.next_u32() & 0xff) as u8) % 100 + 10u8; //ensure the randomly generated image are all positive
        f.write_all(&[tmp]).unwrap();
    }

    //generate CIFAR random input dataset
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/cifar.data")
        .unwrap();

    for _ in 0..(32 * 32 * 3) {
        let tmp = ((rng.next_u32() & 0xff) as u8) % 100 + 10u8; //ensure the randomly generated image are all positive
        f.write_all(&[tmp]).unwrap();
    }

    //generate CIFAR random input dataset
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/cifar_i8.data")
        .unwrap();

    for _ in 0..(32 * 32 * 3) {
        let tmp = ((rng.next_u32() & 0xff) as u8) % 100 + 10u8; //ensure the randomly generated image are all positive
        f.write_all(&[tmp]).unwrap();
    }

    //generate face recognition input dataset
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/face_recognition_92_112_3.data")
        .unwrap();

    for _ in 0..(46 * 56 * 3) {
        let tmp = ((rng.next_u32() & 0xff) as u8) % 100 + 10u8; //ensure the randomly generated image are all positive
        f.write_all(&[tmp]).unwrap();
    }

    //generate face recognition input dataset
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/face_recognition_92_112_3_i8.data")
        .unwrap();

    for _ in 0..(46 * 56 * 3) {
        let tmp = ((rng.next_u32() & 0xff) as u8) % 100 + 10u8; //ensure the randomly generated image are all positive
        f.write_all(&[tmp]).unwrap();
    }
}

fn gen_lenet_small_u8() {
    let mut rng = rand::thread_rng();

    //generate lenet small parameters
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv1.data")
        .unwrap();

    for _ in 0..(6 * 1 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    //generate lenet small parameters
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv1_cifar.data")
        .unwrap();

    for _ in 0..(6 * 3 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv2.data")
        .unwrap();

    for _ in 0..(16 * 6 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_conv3.data")
        .unwrap();

    for _ in 0..(120 * 16 * 4 * 4) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc1.data")
        .unwrap();

    for _ in 0..(84 * 120) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc1_face.data")
        .unwrap();

    for _ in 0..(84 * 120 * 5 * 8) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc1_cifar.data")
        .unwrap();

    for _ in 0..(84 * 120 * 2 * 2) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc2_cifar.data")
        .unwrap();

    for _ in 0..(10 * 84) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_small_fc2_face.data")
        .unwrap();

    for _ in 0..(40 * 84) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }
}

fn gen_lenet_medium_u8() {
    //generate lenet medium parameters
    let mut rng = rand::thread_rng();

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_conv1.data")
        .unwrap();

    for _ in 0..(32 * 1 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_conv1_cifar.data")
        .unwrap();

    for _ in 0..(32 * 3 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_conv2.data")
        .unwrap();

    for _ in 0..(64 * 32 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_conv3.data")
        .unwrap();

    for _ in 0..(256 * 64 * 4 * 4) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_fc1.data")
        .unwrap();

    for _ in 0..(256 * 128) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_fc1_cifar.data")
        .unwrap();

    for _ in 0..(256 * 128 * 2 * 2) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_fc1_face.data")
        .unwrap();

    for _ in 0..(256 * 128 * 5 * 8) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_fc2.data")
        .unwrap();

    for _ in 0..(10 * 128) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_medium_fc2_face.data")
        .unwrap();

    for _ in 0..(40 * 128) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }
}

fn gen_lenet_large_u8() {
    //generate lenet large parameters
    let mut rng = rand::thread_rng();

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_conv1.data")
        .unwrap();

    for _ in 0..(64 * 1 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_conv2.data")
        .unwrap();

    for _ in 0..(128 * 64 * 5 * 5) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_conv3.data")
        .unwrap();

    for _ in 0..(512 * 128 * 4 * 4) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_fc1.data")
        .unwrap();

    for _ in 0..(512 * 256) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_fc1_face.data")
        .unwrap();

    for _ in 0..(256 * 512 * 5 * 8) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/lenet_large_fc2_face.data")
        .unwrap();

    for _ in 0..(40 * 256) {
        let tmp = (rng.next_u32() & 0xff) as u8;
        f.write_all(&[tmp]).unwrap();
    }
}

fn gen_shallownet_u8() {
    let mut rng = rand::thread_rng();

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/l1mat.data")
        .unwrap();

    for _ in 0..784 {
        for _ in 0..128 {
            let tmp = (rng.next_u32() & 0xff) as u8;
            f.write_all(&[tmp]).unwrap();
        }
    }

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test-data/l2mat.data")
        .unwrap();

    for _ in 0..128 {
        for _ in 0..10 {
            let tmp = (rng.next_u32() & 0xff) as u8;
            f.write_all(&[tmp]).unwrap();
        }
    }
}

fn main() {
    gen_lenet_small_cifar_i8();
    gen_lenet_small_u8();
    gen_lenet_medium_u8();
    gen_lenet_large_u8();
    gen_dataset();
    gen_shallownet_u8();
}
