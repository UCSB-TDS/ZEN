use crate::*;
use std::fs::File;
use std::io::Read;
use std::io::{BufRead, BufReader};

pub fn read_vector1d(filename: String, len: usize) -> Vec<u8> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<u8> = vec![0u8; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{} {:?}",filename, res);
    res
}

pub fn read_vector1d_i8(filename: String, len: usize) -> Vec<i8> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<i8> = vec![0i8; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<i8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<i8>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    println!("{} {:?}", filename, res);
    res
}

pub fn read_vector1d_f32(filename: String, len: usize) -> Vec<f32> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<f32> = vec![0.0f32; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;

    for line in buffered.lines() {
        //println!("{:?}", line.unwrap().split(" ").collect::<Vec<&str>>());
        let raw_vec: Vec<f32> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{:?}", res);
    res
}

pub fn read_vector2d(filename: String, rows: usize, cols: usize) -> Vec<Vec<u8>> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<u8>> = vec![vec![0u8; cols]; rows];
    let buffered = BufReader::new(f);

    let mut counter = 0usize;

    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        if counter < rows * cols {
            res[counter / cols][counter % cols] = raw_vec[0]; //flattened before writing to the file. each line only contains one number
        }
        counter += 1;
    }

    res
}

pub fn read_vector4d(
    filename: String,
    in_channel: usize,
    out_channel: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<Vec<Vec<u8>>>> =
        vec![vec![vec![vec![0u8; cols]; rows]; out_channel]; in_channel];
    let mut tmp: Vec<u8> = vec![0u8; cols * rows * out_channel * in_channel];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        tmp[counter] = raw_vec[0];
        counter += 1;
    }

    let mut counter = 0;
    for i in 0..in_channel {
        for j in 0..out_channel {
            for k in 0..rows {
                for m in 0..cols {
                    res[i][j][k][m] = tmp[counter];
                    counter += 1;
                }
            }
        }
    }
    //println!("{} {:?}\n\n", filename, res);
    res
}

pub fn read_inputs() -> (X, L1Mat, L2Mat) {
    // read x
    let mut f = File::open("test-data/mnist.data").unwrap();
    let mut x: X = vec![0i8; L];
    let mut tmp = [0u8; 1];
    for i in 0..L {
        f.read_exact(&mut tmp).unwrap();
        x[i] = tmp[0] as i8;
    }
    // read l1_mat
    let mut f = File::open("test-data/l1mat.data").unwrap();

    let mut l1_mat = vec![vec![0i8; L]; M];
    for i in 0..M {
        for j in 0..L {
            f.read_exact(&mut tmp).unwrap();
            l1_mat[i][j] = tmp[0] as i8;
        }
    }

    // read l2_mat
    let mut f = File::open("test-data/l2mat.data").unwrap();
    let mut l2_mat = vec![vec![0i8; M]; N];
    for i in 0..N {
        for j in 0..M {
            f.read_exact(&mut tmp).unwrap();
            l2_mat[i][j] = tmp[0] as i8;
        }
    }
    (x, l1_mat, l2_mat)
}

pub fn read_shallownet_inputs_u8() -> (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    // read x
    let mut f = File::open("test-data/mnist.data").unwrap();
    let mut x = vec![0u8; L];
    let mut tmp = [0u8; 1];
    for i in 0..L {
        f.read_exact(&mut tmp).unwrap();
        x[i] = tmp[0] as u8;
    }
    // read l1_mat
    let mut f = File::open("test-data/l1mat.data").unwrap();

    let mut l1_mat = vec![vec![0u8; L]; M];
    for i in 0..M {
        for j in 0..L {
            f.read_exact(&mut tmp).unwrap();
            l1_mat[i][j] = tmp[0] as u8;
        }
    }

    // read l2_mat
    let mut f = File::open("test-data/l2mat.data").unwrap();
    let mut l2_mat = vec![vec![0u8; M]; N];
    for i in 0..N {
        for j in 0..M {
            f.read_exact(&mut tmp).unwrap();
            l2_mat[i][j] = tmp[0] as u8;
        }
    }

    (x, l1_mat, l2_mat)
}

//corresponding to cifar-10 + LeNet-5-small section in the paper
pub fn read_lenet_small_inputs_i8_cifar() -> (
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<i8>>,
    Vec<Vec<i8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_small_conv1_i8.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0i8; 5]; 5]; 3]; 6];
    for i in 0..6 {
        for j in 0..3 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv2_i8.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0i8; 5]; 5]; 6]; 16];
    for i in 0..16 {
        for j in 0..6 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv3_i8.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0i8; 4]; 4]; 16]; 120];
    for i in 0..120 {
        for j in 0..16 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_fc1_cifar_i8.data").unwrap();
    let mut fc1_weights = vec![vec![0i8; 120 * 2 * 2]; 84];
    for i in 0..84 {
        for j in 0..(120 * 2 * 2) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as i8;
        }
    }

    let mut f = File::open("test-data/lenet_small_fc2_i8.data").unwrap();
    let mut fc2_weights = vec![vec![0i8; 84]; 10];
    for i in 0..10 {
        for j in 0..84 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as i8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to cifar-10 + LeNet-5-small section in the paper
pub fn read_lenet_small_inputs_u8_cifar() -> (
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_small_conv1_cifar.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0u8; 5]; 5]; 3]; 6];
    for i in 0..6 {
        for j in 0..3 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0u8; 5]; 5]; 6]; 16];
    for i in 0..16 {
        for j in 0..6 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0u8; 4]; 4]; 16]; 120];
    for i in 0..120 {
        for j in 0..16 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_fc1_cifar.data").unwrap();
    let mut fc1_weights = vec![vec![0u8; 120 * 2 * 2]; 84];
    for i in 0..84 {
        for j in 0..(120 * 2 * 2) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as u8;
        }
    }

    let mut f = File::open("test-data/lenet_small_fc2_cifar.data").unwrap();
    let mut fc2_weights = vec![vec![0u8; 84]; 10];
    for i in 0..10 {
        for j in 0..84 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as u8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to cifar-100 + LeNet-5-medium section in the paper

pub fn read_lenet_medium_inputs_u8_cifar() -> (
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_medium_conv1_cifar.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0u8; 5]; 5]; 3]; 32];
    for i in 0..32 {
        for j in 0..3 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0u8; 5]; 5]; 32]; 64];
    for i in 0..64 {
        for j in 0..32 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0u8; 4]; 4]; 64]; 256];
    for i in 0..256 {
        for j in 0..64 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc1_cifar.data").unwrap();
    let mut fc1_weights = vec![vec![0u8; 256 * 2 * 2]; 128];
    for i in 0..128 {
        for j in 0..(256 * 2 * 2) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as u8;
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc2.data").unwrap();
    let mut fc2_weights = vec![vec![0u8; 128]; 10];
    for i in 0..10 {
        for j in 0..128 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as u8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to cifar-10 + LeNet-5-small section in the paper
pub fn read_lenet_small_inputs_i8_face() -> (
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<i8>>,
    Vec<Vec<i8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_small_conv1_i8.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0i8; 5]; 5]; 1]; 6];
    for i in 0..6 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv2_i8.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0i8; 5]; 5]; 6]; 16];
    for i in 0..16 {
        for j in 0..6 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv3_i8.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0i8; 4]; 4]; 16]; 120];
    for i in 0..120 {
        for j in 0..16 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0i8; 120 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 84];
    for i in 0..84 {
        for j in 0..(120 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as i8;
        }
    }

    let mut f = File::open("test-data/lenet_small_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0i8; 84]; 40];
    for i in 0..40 {
        for j in 0..84 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as i8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

pub fn read_lenet_medium_inputs_u8_face() -> (
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_medium_conv1.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0u8; 5]; 5]; 1]; 32];
    for i in 0..32 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0u8; 5]; 5]; 32]; 64];
    for i in 0..64 {
        for j in 0..32 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0u8; 4]; 4]; 64]; 256];
    for i in 0..256 {
        for j in 0..64 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0u8; 256 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 128];
    for i in 0..128 {
        for j in 0..(256 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as u8;
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0u8; 128]; 40];
    for i in 0..40 {
        for j in 0..128 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as u8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

pub fn read_lenet_medium_inputs_i8_face() -> (
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<i8>>,
    Vec<Vec<i8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_medium_conv1.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0i8; 5]; 5]; 1]; 32];
    for i in 0..32 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0i8; 5]; 5]; 32]; 64];
    for i in 0..64 {
        for j in 0..32 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0i8; 4]; 4]; 64]; 256];
    for i in 0..256 {
        for j in 0..64 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0i8; 256 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 128];
    for i in 0..128 {
        for j in 0..(256 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as i8;
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0i8; 128]; 40];
    for i in 0..40 {
        for j in 0..128 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as i8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

pub fn read_lenet_medium_inputs_i8_cifar() -> (
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<i8>>,
    Vec<Vec<i8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_medium_conv1_cifar.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0i8; 5]; 5]; 3]; 32];
    for i in 0..32 {
        for j in 0..3 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0i8; 5]; 5]; 32]; 64];
    for i in 0..64 {
        for j in 0..32 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0i8; 4]; 4]; 64]; 256];
    for i in 0..256 {
        for j in 0..64 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc1_cifar.data").unwrap();
    let mut fc1_weights = vec![vec![0i8; 256 * 2 * 2]; 128];
    for i in 0..128 {
        for j in 0..(256 * 2 * 2) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as i8;
        }
    }

    let mut f = File::open("test-data/lenet_medium_fc2.data").unwrap();
    let mut fc2_weights = vec![vec![0i8; 128]; 10];
    for i in 0..10 {
        for j in 0..128 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as i8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to LFW + LeNet-5-large section in the paper
pub fn read_lenet_large_inputs_u8_face() -> (
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_large_conv1.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0u8; 5]; 5]; 1]; 64];
    for i in 0..64 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0u8; 5]; 5]; 64]; 128];
    for i in 0..128 {
        for j in 0..64 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0u8; 4]; 4]; 128]; 512];
    for i in 0..512 {
        for j in 0..128 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0u8; 512 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 256];
    for i in 0..256 {
        for j in 0..(512 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as u8;
        }
    }

    let mut f = File::open("test-data/lenet_large_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0u8; 256]; 40];
    for i in 0..40 {
        for j in 0..256 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as u8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to LFW + LeNet-5-large section in the paper
pub fn read_lenet_large_inputs_i8_face() -> (
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<Vec<Vec<i8>>>>,
    Vec<Vec<i8>>,
    Vec<Vec<i8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_large_conv1.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0i8; 5]; 5]; 1]; 64];
    for i in 0..64 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0i8; 5]; 5]; 64]; 128];
    for i in 0..128 {
        for j in 0..64 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0i8; 4]; 4]; 128]; 512];
    for i in 0..512 {
        for j in 0..128 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as i8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_large_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0i8; 512 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 256];
    for i in 0..256 {
        for j in 0..(512 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as i8;
        }
    }

    let mut f = File::open("test-data/lenet_large_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0i8; 256]; 40];
    for i in 0..40 {
        for j in 0..256 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as i8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

//corresponding to cifar-10 + LeNet-5-small section in the paper
pub fn read_lenet_small_inputs_u8_face() -> (
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
) {
    let mut tmp = [0u8; 1];

    let mut f = File::open("test-data/lenet_small_conv1.data").unwrap();
    let mut conv1_weights = vec![vec![vec![vec![0u8; 5]; 5]; 1]; 6];
    for i in 0..6 {
        for j in 0..1 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv1_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv2.data").unwrap();
    let mut conv2_weights = vec![vec![vec![vec![0u8; 5]; 5]; 6]; 16];
    for i in 0..16 {
        for j in 0..6 {
            for h in 0..5 {
                for w in 0..5 {
                    f.read_exact(&mut tmp).unwrap();
                    conv2_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_conv3.data").unwrap();
    let mut conv3_weights = vec![vec![vec![vec![0u8; 4]; 4]; 16]; 120];
    for i in 0..120 {
        for j in 0..16 {
            for h in 0..4 {
                for w in 0..4 {
                    f.read_exact(&mut tmp).unwrap();
                    conv3_weights[i][j][h][w] = tmp[0] as u8;
                }
            }
        }
    }

    let mut f = File::open("test-data/lenet_small_fc1_face.data").unwrap();
    let mut fc1_weights = vec![vec![0u8; 120 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1]; 84];
    for i in 0..84 {
        for j in 0..(120 * FACE_HEIGHT_FC1 * FACE_WIDTH_FC1) {
            f.read_exact(&mut tmp).unwrap();
            fc1_weights[i][j] = tmp[0] as u8;
        }
    }

    let mut f = File::open("test-data/lenet_small_fc2_face.data").unwrap();
    let mut fc2_weights = vec![vec![0u8; 84]; 40];
    for i in 0..40 {
        for j in 0..84 {
            f.read_exact(&mut tmp).unwrap();
            fc2_weights[i][j] = tmp[0] as u8;
        }
    }

    (
        conv1_weights,
        conv2_weights,
        conv3_weights,
        fc1_weights,
        fc2_weights,
    )
}

pub fn read_mnist_input_u8() -> Vec<Vec<Vec<Vec<u8>>>> {
    // read x
    let mut f = File::open("test-data/mnist.data").unwrap();
    let mut x = vec![vec![vec![vec![0u8; 28]; 28]; 1]; 1]; //we only have one image in one batch
    let mut tmp = [0u8; 1];
    for i in 0..28 {
        for j in 0..28 {
            f.read_exact(&mut tmp).unwrap();
            x[0][0][i][j] = tmp[0] as u8;
        }
    }

    x
}

pub fn read_cifar_input_u8() -> Vec<Vec<Vec<Vec<u8>>>> {
    // read x
    let mut f = File::open("test-data/cifar.data").unwrap();
    let mut x = vec![vec![vec![vec![0u8; 32]; 32]; 3]; 1]; //we only have one image in one batch
    let mut tmp = [0u8; 1];
    for i in 0..28 {
        for j in 0..28 {
            for k in 0..3 {
                f.read_exact(&mut tmp).unwrap();
                x[0][k][i][j] = tmp[0] as u8;
            }
        }
    }

    x
}

pub fn read_cifar_input_i8() -> Vec<Vec<Vec<Vec<i8>>>> {
    // read x
    let mut f = File::open("test-data/cifar.data").unwrap();
    let mut x = vec![vec![vec![vec![0i8; 32]; 32]; 3]; 1]; //we only have one image in one batch
    let mut tmp = [0u8; 1];
    for i in 0..28 {
        for j in 0..28 {
            for k in 0..3 {
                f.read_exact(&mut tmp).unwrap();
                x[0][k][i][j] = tmp[0] as i8;
            }
        }
    }

    x
}

pub fn read_face_recognition_92_112_input_u8() -> Vec<Vec<Vec<Vec<u8>>>> {
    // read x
    let mut f = File::open("test-data/face_recognition_92_112_3.data").unwrap();
    let mut x = vec![vec![vec![vec![0u8; FACE_WIDTH]; FACE_HEIGHT]; 1]; 1]; //we only have one image in one batch
    let mut tmp = [0u8; 1];
    for i in 0..FACE_HEIGHT {
        for j in 0..FACE_WIDTH {
            for k in 0..1 {
                f.read_exact(&mut tmp).unwrap();
                x[0][k][i][j] = tmp[0] as u8;
            }
        }
    }

    x
}

pub fn read_face_recognition_92_112_input_i8() -> Vec<Vec<Vec<Vec<i8>>>> {
    // read x
    let mut f = File::open("test-data/face_recognition_92_112_3.data").unwrap();
    let mut x = vec![vec![vec![vec![0i8; FACE_WIDTH]; FACE_HEIGHT]; 1]; 1]; //we only have one image in one batch
    let mut tmp = [0u8; 1];
    for i in 0..FACE_HEIGHT {
        for j in 0..FACE_WIDTH {
            for k in 0..1 {
                f.read_exact(&mut tmp).unwrap();
                x[0][k][i][j] = tmp[0] as i8;
            }
        }
    }

    x
}
