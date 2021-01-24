use crate::*;
use crypto_primitives::commitment::blake2s::Commitment;
use crypto_primitives::CommitmentScheme;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub const DEFAULT_ZERO_POINT: u8 = 10;

#[allow(non_snake_case)]
pub(crate) fn relu_u8(input: &mut [u8], zero_point: u8) -> Vec<bool> {
    let mut cmp_res: Vec<bool> = Vec::new();
    for e in input {
        if *e < zero_point {
            *e = zero_point;
            cmp_res.push(false);
        } else {
            cmp_res.push(true);
        }
    }
    cmp_res
}

#[allow(non_snake_case)]
pub(crate) fn relu2d_u8(input: &mut Vec<Vec<u8>>, zero_point: u8) {
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            if input[i][j] < zero_point {
                input[i][j] = zero_point;
            }
        }
    }
}

pub(crate) fn relu4d_u8(
    input: &mut Vec<Vec<Vec<Vec<u8>>>>,
    zero_point: u8,
) -> Vec<Vec<Vec<Vec<bool>>>> {
    let mut cmp_res: Vec<Vec<Vec<Vec<bool>>>> =
        vec![
            vec![vec![vec![false; input[0][0][0].len()]; input[0][0].len()]; input[0].len()];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for l in 0..input[i][j][k].len() {
                    if input[i][j][k][l] < zero_point {
                        input[i][j][k][l] = zero_point;
                        cmp_res[i][j][k][l] = false;
                    } else {
                        cmp_res[i][j][k][l] = true;
                    }
                }
            }
        }
    }

    cmp_res
}

pub(crate) fn scalar_with_remainder_u8(a: &[u8], b: &[u8], a_0: u8, b_0: u8, y_0: u64) -> u64 {
    if a.len() != b.len() {
        panic!("incorrect dim {} {}", a.len(), b.len());
    }
    let mut tmp1: u64 = 0;
    let mut tmp2: u64 = 0;
    let mut tmp3: u64 = 0;
    let mut tmp4: u64 = 0;
    for i in 0..a.len() {
        tmp1 += a[i] as u64 * b[i] as u64;
        tmp2 += a[i] as u64 * b_0 as u64;
        tmp3 += a_0 as u64 * b[i] as u64;
        tmp4 += a_0 as u64 * b_0 as u64;
    }

    let res = (tmp1 + tmp4 + y_0) - (tmp2 + tmp3);

    res
}

//return the remainder of divided by 2^24
pub(crate) fn vec_mat_mul_with_remainder_u8(
    vec: &[u8],
    mat: &[&[u8]],
    res: &mut [u8],
    vec_0: u8,
    mat_0: u8,
    res_0: u8,
    multiplier: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    //record info loss during u64/u32 to u8 for later recovery
    //println!("q1 before shift {:?}", mat[0].clone());
    let mut remainder = vec![0u32; res.len()];
    let mut div_res = vec![0u32; res.len()];
    for i in 0..mat.len() {
        let m = (multiplier[i] * 2u64.pow(22) as f32) as u64;
        let res_converted = (res_0 as u64 * 2u64.pow(22)) / m;
        let scalar_tmp =
            m * scalar_with_remainder_u8(vec.clone(), mat[i], vec_0, mat_0, res_converted);
        remainder[i] = (scalar_tmp % 2u64.pow(22)) as u32;
        div_res[i] = (scalar_tmp / 2u64.pow(22 + 8)) as u32;
        res[i] = (scalar_tmp / 2u64.pow(22)) as u8;
    }

    //println!("res {:?}", res);
    (remainder, div_res)
}

fn conv_kernel_scala_with_remainder_u8(
    x: &Vec<Vec<Vec<u8>>>,
    kernel: &Vec<Vec<Vec<u8>>>,
    h_index: usize,
    w_index: usize,

    x_zeropoint: u8,
    kernel_zeropoint: u8,
    y_0: u64,
) -> u64 {
    let num_channels = kernel.len();
    let kernel_size = kernel[0].len();
    let mut tmp1: u64 = 0;
    let mut tmp2: u64 = 0;
    let mut tmp3: u64 = 0;
    let mut tmp4: u64 = 0;
    //println!("multiplier : {}\n y_converted : {}", m, y_converted);
    for i in 0..num_channels {
        //iterate through all channels

        for j in h_index..(h_index + kernel_size) {
            // println!("data {:?}", &x[i][j][w_index..w_index+kernel_size]);
            // println!("kernel {:?}", &kernel[i][j][0..kernel_size]);
            for k in w_index..(w_index + kernel_size) {
                //println!("i,j,k {} {} {}",i, j - h_index, k - w_index);
                tmp1 += x[i][j][k] as u64 * kernel[i][j - h_index][k - w_index] as u64;

                tmp2 += x[i][j][k] as u64 * kernel_zeropoint as u64;
                tmp3 += kernel[i][j - h_index][k - w_index] as u64 * x_zeropoint as u64;

                tmp4 += x_zeropoint as u64 * kernel_zeropoint as u64;
            }
        }
    }
    //println!("conv output {}  {} ", tmp1 *m +  tmp4*m, tmp2*m + tmp3*m,);
    //assert_eq!(tmp1, tmp2);

    //println!("tmp14 {}\ntmp23{}", tmp1+ tmp4, tmp2+ tmp3);
    let res = (tmp1 + tmp4 + y_0) - (tmp2 + tmp3);

    res
}

pub(crate) fn vec_conv_with_remainder_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel: &Vec<Vec<Vec<Vec<u8>>>>,
    res: &mut Vec<Vec<Vec<Vec<u8>>>>,
    vec_0: u8,
    kernel_0: u8,
    res_0: u8,
    multiplier: &[f32],
) -> (Vec<Vec<Vec<Vec<u32>>>>, Vec<Vec<Vec<Vec<u32>>>>) {
    let num_kernels = kernel.len();
    let kernel_size = kernel[0][0].len();
    let batch_size = vec.len();
    let input_height = vec[0][0].len();
    let input_width = vec[0][0][0].len();
    //println!("kernel {:?}", kernel.clone());
    //record info loss during u64/u32 to u8 for later recovery
    let mut remainder =
        vec![vec![vec![vec![0u32; res[0][0][0].len()]; res[0][0].len()]; res[0].len()]; res.len()];
    let mut div =
        vec![vec![vec![vec![0u32; res[0][0][0].len()]; res[0][0].len()]; res[0].len()]; res.len()];
    for n in 0..batch_size {
        for h in 0..(input_height - kernel_size + 1) {
            for w in 0..(input_width - kernel_size + 1) {
                for k in 0..num_kernels {
                    //println!("{} {} {} {}",n, k, h, w);
                    let m = (multiplier[k] * 2u64.pow(22) as f32) as u64;
                    let res_converted = (res_0 as u64 * 2u64.pow(22)) / m;
                    let tmp = m * conv_kernel_scala_with_remainder_u8(
                        &vec[n],
                        &kernel[k],
                        h,
                        w,
                        vec_0,
                        kernel_0,
                        res_converted,
                    );

                    res[n][k][h][w] = (tmp / 2u64.pow(22)) as u8;

                    remainder[n][k][h][w] = (tmp % 2u64.pow(22)) as u32;
                    div[n][k][h][w] = (tmp / 2u64.pow(30)) as u32;
                }
            }
        }
    }
    //println!("kernel shape ({},{},{},{})", K,C,kernel_size,kernel_size);
    (remainder, div)
}

pub(crate) fn avg_pool_with_remainder_helper_u8(
    input: &Vec<Vec<u8>>,
    h_start: usize,
    w_start: usize,
    kernel_size: usize,
) -> (u8, u8) {
    let mut res: u32 = 0;

    for i in h_start..(h_start + kernel_size) {
        for j in w_start..(w_start + kernel_size) {
            res += input[i][j] as u32;
        }
    }

    (
        (res / (kernel_size as u32 * kernel_size as u32)) as u8,
        (res % (kernel_size as u32 * kernel_size as u32)) as u8,
    )
}

pub(crate) fn avg_pool_helper_u8(
    input: &Vec<Vec<u8>>,
    h_start: usize,
    w_start: usize,
    kernel_size: usize,
) -> u8 {
    let mut res: u32 = 0;

    for i in h_start..(h_start + kernel_size) {
        for j in w_start..(w_start + kernel_size) {
            res += input[i][j] as u32;
        }
    }

    (res / (kernel_size as u32 * kernel_size as u32)) as u8
}
pub(crate) fn avg_pool_scala_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel_size: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = vec.len();
    let num_channels = vec[0].len(); //num of channels
    let input_height = vec[0][0].len(); // height of image
    let input_width = vec[0][0][0].len(); // width of image
    let mut output = vec![
        vec![
            vec![vec![0u8; input_width / kernel_size]; input_height / kernel_size];
            num_channels
        ];
        batch_size
    ];
    for n in 0..batch_size {
        for c in 0..num_channels {
            for h in 0..(input_height / kernel_size) {
                for w in 0..(input_width / kernel_size) {
                    output[n][c][h][w] = avg_pool_helper_u8(
                        &vec[n][c],
                        kernel_size * h,
                        kernel_size * w,
                        kernel_size,
                    );
                }
            }
        }
    }
    output
}

pub(crate) fn avg_pool_with_remainder_scala_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel_size: usize,
) -> (Vec<Vec<Vec<Vec<u8>>>>, Vec<Vec<Vec<Vec<u8>>>>) {
    let batch_size = vec.len();
    let num_channels = vec[0].len(); //num of channels
    let input_height = vec[0][0].len(); // height of image
    let input_width = vec[0][0][0].len(); // width of image
    let mut output =
        vec![vec![vec![vec![0u8; input_width / 2]; input_height / 2]; num_channels]; batch_size];
    let mut remainder =
        vec![vec![vec![vec![0u8; input_width]; input_height]; num_channels]; batch_size];

    for n in 0..batch_size {
        for c in 0..num_channels {
            for h in 0..(input_height / kernel_size) {
                for w in 0..(input_width / kernel_size) {
                    let (res, remained) = avg_pool_with_remainder_helper_u8(
                        &vec[n][c],
                        kernel_size * h,
                        kernel_size * w,
                        kernel_size,
                    );
                    output[n][c][h][w] = res;
                    remainder[n][c][h][w] = remained;
                }
            }
        }
    }
    (output, remainder)
}

pub fn lenet_circuit_forward_u8(
    x: Vec<Vec<Vec<Vec<u8>>>>,
    conv1_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv2_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    fc1_weight: Vec<Vec<u8>>,
    fc2_weight: Vec<Vec<u8>>,
    x_0: u8,
    conv1_output_0: u8,
    conv2_output_0: u8,
    conv3_output_0: u8,
    fc1_output_0: u8,
    fc2_output_0: u8,
    conv1_weights_0: u8,
    conv2_weights_0: u8,
    conv3_weights_0: u8,
    fc1_weights_0: u8,
    fc2_weights_0: u8,
    multiplier_conv1: Vec<f32>,
    multiplier_conv2: Vec<f32>,
    multiplier_conv3: Vec<f32>,
    multiplier_fc1: Vec<f32>,
    multiplier_fc2: Vec<f32>,
) -> Vec<Vec<u8>> {
    println!("lenet vallina forward");
    //layer 1
    let mut conv1_output = vec![vec![vec![vec![0u8; x[0][0][0].len() - conv1_kernel[0][0][0].len() + 1];  // w - kernel_size + 1
                                        x[0][0].len() - conv1_kernel[0][0].len() + 1]; // h - kernel_size + 1
                                        conv1_kernel.len()]; //number of conv kernels
                                        x.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &x,
        &conv1_kernel,
        &mut conv1_output,
        x_0,
        conv1_weights_0,
        conv1_output_0,
        &multiplier_conv1,
    );

    //layer 1

    relu4d_u8(&mut conv1_output, conv1_output_0);

    let avg_pool1_output = avg_pool_scala_u8(&conv1_output, 2);
    //println!("{} {} {} ", avg_pool1_output[0].len() , avg_pool1_output[0][0].len() , avg_pool1_output[0][0][0].len());

    //layer 2

    let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - conv2_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - conv2_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                                        conv2_kernel.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &avg_pool1_output,
        &conv2_kernel,
        &mut conv2_output,
        conv1_output_0,
        conv2_weights_0,
        conv2_output_0,
        &multiplier_conv2,
    );
    relu4d_u8(&mut conv2_output, conv2_output_0);

    let avg_pool2_output = avg_pool_scala_u8(&conv2_output, 2);
    //println!("{} {} {} ", avg_pool2_output[0].len() , avg_pool2_output[0][0].len() , avg_pool2_output[0][0][0].len());

    //layer 3
    let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - conv3_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool2_output[0][0].len() - conv3_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                                        conv3_kernel.len()]; //number of conv kernels
                                                                        avg_pool2_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &avg_pool2_output,
        &conv3_kernel,
        &mut conv3_output,
        conv2_output_0,
        conv3_weights_0,
        conv3_output_0,
        &multiplier_conv3,
    );

    relu4d_u8(&mut conv3_output, conv3_output_0);
    //println!("{} {} {} ", conv3_output[0].len() , conv3_output[0][0].len() , conv3_output[0][0][0].len());

    //at the end of layer 3 we have to transform conv3_output to different shape to fit in FC layer.
    // previous shape is [batch size, xxx, 1, 1]. we  want to reshape it to [batch size, xxx]
    let mut transformed_conv3_output =
        vec![
            vec![
                0u8;
                conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
            ];
            conv3_output.len()
        ];
    for i in 0..conv3_output.len() {
        let mut counter = 0;
        for j in 0..conv3_output[0].len() {
            for p in 0..conv3_output[0][0].len() {
                for q in 0..conv3_output[0][0][0].len() {
                    transformed_conv3_output[i][counter] = conv3_output[i][j][p][q];
                    counter += 1;
                }
            }
        }
    }
    //println!("flattened conv3 output shape {} {}", transformed_conv3_output.len(), transformed_conv3_output[0].len());
    #[cfg(debug_assertion)]
    println!(
        " FC layer input len : {}, FC layer weight len {}",
        transformed_conv3_output[0].len(),
        fc1_weight[0].len()
    );
    //layer 4
    let mut fc1_output = vec![vec![0u8; fc1_weight.len()];  // channels
                                                transformed_conv3_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_conv3_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &transformed_conv3_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv3_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1,
        );
    }
    relu2d_u8(&mut fc1_output, fc1_output_0);

    //layer 5
    let mut fc2_output = vec![vec![0u8; fc2_weight.len()]; // channels
                                                    fc1_output.len()]; //batch size
    let fc2_weight_ref: Vec<&[u8]> = fc2_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..fc1_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &fc1_output[i],
            fc2_weight_ref[..].as_ref(),
            &mut fc2_output[i],
            fc1_output_0,
            fc2_weights_0,
            fc2_output_0,
            &multiplier_fc2,
        );
    }

    fc2_output
}

pub fn full_circuit_forward_u8(
    x: Vec<u8>,
    l1_mat: Vec<Vec<u8>>,
    l2_mat: Vec<Vec<u8>>,
    x_0: u8,
    y_0: u8,
    z_0: u8,
    l1_mat_0: u8,
    l2_mat_0: u8,
    multiplier_l1: Vec<f32>,
    multiplier_l2: Vec<f32>,
) -> Vec<u8> {
    let mut y = vec![0u8; M];
    let l1_mat_ref: Vec<&[u8]> = l1_mat.iter().map(|x| x.as_ref()).collect();
    vec_mat_mul_with_remainder_u8(
        &x,
        l1_mat_ref[..].as_ref(),
        &mut y,
        x_0,
        l1_mat_0,
        y_0,
        &multiplier_l1,
    );
    //println!("x_0 {}, l1_mat_0 {}, l1_output_0 {}", x_0, l1_mat_0, y_0);
    //println!("l1 output {:?}\n", y);

    relu_u8(&mut y, y_0);
    // println!("relu output {:?}\n", y);
    let mut z = vec![0u8; N];
    let l2_mat_ref: Vec<&[u8]> = l2_mat.iter().map(|x| x.as_ref()).collect();
    vec_mat_mul_with_remainder_u8(
        &y,
        l2_mat_ref[..].as_ref(),
        &mut z,
        y_0,
        l2_mat_0,
        z_0,
        &multiplier_l2,
    );
    // println!("l2 output {:?}\n", z);

    z
}

pub fn vec_mat_mul_cos_helper(vec: &[u8], mat: &[u8]) -> u64 {
    let mut res = 0u64;
    for i in 0..mat.len() {
        res += vec[i] as u64 * mat[i] as u64;
    }
    res
}

pub fn cosine_similarity(vec1: Vec<u8>, vec2: Vec<u8>, threshold: u32) -> bool {
    let norm_1 = vec_mat_mul_cos_helper(&vec1, &vec1);
    let norm_2 = vec_mat_mul_cos_helper(&vec2, &vec2);
    let numerator = vec_mat_mul_cos_helper(&vec1, &vec2);

    let res: bool =
        (10000 * numerator * numerator) > (threshold as u64) * (threshold as u64) * norm_1 * norm_2;

    res
}

pub fn argmax_u8(input: Vec<u8>) -> usize {
    let mut res = 0usize;
    let mut tmp_max = 0u8;
    for i in 0..input.len() {
        if input[i] > tmp_max {
            tmp_max = input[i];
            res = i;
        }
    }
    res
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_x(data: &[i8], seed: &[u8; 32]) -> (Commit, Open) {
    // input
    let input = compress_x(data);

    commit_u8(&input, seed)
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_z(data: &[i8], seed: &[u8; 32]) -> (Commit, Open) {
    let input = data.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

    commit_u8(&input, seed)
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_u8(data: &[u8], seed: &[u8; 32]) -> (Commit, Open) {
    // blake2s do not take parameters
    let parameters = ();

    // openning
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut open = [0u8; 32];
    rng.fill(&mut open);

    // commit
    (Commitment::commit(&parameters, &data, &open).unwrap(), open)
}

/// compress x
/// requirement: -128 <= x[i] < 128
pub fn compress_x(data: &[i8]) -> Vec<u8> {
    data.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>()
}

/// compress x
/// requirement: 0 <= x[i] < 256
pub fn compress_x_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|x| *x as u8).collect::<Vec<u8>>()
}
