use crate::read_inputs::*;
use crate::vanilla::*;
use r1cs_core::*;

#[test]
fn lenet_medium() {
    println!("LeNet optimized medium on CIFAR dataset");

    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/X_q.txt".to_string(),
        1000,
        3,
        32,
        32,
    );

    // let real_output: Vec<Vec<u8>> = read_vector2d(
    //     "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_output_q.txt".to_string(),
    //     1000,
    //     10,
    // );
    let classification_result: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_classification.txt".to_string(),
        1000,
    );

    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_q.txt".to_string(),
        32,
        3,
        5,
        5,
    );
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_q.txt".to_string(),
        64,
        32,
        5,
        5,
    );
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_q.txt".to_string(),
        256,
        64,
        4,
        4,
    );
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_q.txt".to_string(),
        128,
        1024,
    );
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_q.txt".to_string(),
        10,
        128,
    );

    let x_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/X_z.txt".to_string(),
        1,
    );
    let conv1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_output_z.txt".to_string(),
        1,
    );
    let conv2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_output_z.txt".to_string(),
        1,
    );
    let conv3_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_output_z.txt".to_string(),
        1,
    );
    let fc1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_output_z.txt".to_string(),
        1,
    );
    let fc2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_output_z.txt".to_string(),
        1,
    );

    let conv1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_z.txt".to_string(),
        1,
    );
    let conv2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_z.txt".to_string(),
        1,
    );
    let conv3_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_z.txt".to_string(),
        1,
    );
    let fc1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_z.txt".to_string(),
        1,
    );
    let fc2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_z.txt".to_string(),
        1,
    );

    let multipler_conv1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv1_weight_s.txt".to_string(),
        32,
    );
    let multipler_conv2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv2_weight_s.txt".to_string(),
        64,
    );
    let multipler_conv3: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_conv3_weight_s.txt".to_string(),
        256,
    );

    let multipler_fc1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear1_weight_s.txt".to_string(),
        128,
    );
    let multipler_fc2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Medium_linear2_weight_s.txt".to_string(),
        10,
    );

    println!("finish reading parameters");

    let z: Vec<Vec<u8>> = lenet_circuit_forward_u8(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        x_0[0],
        conv1_output_0[0],
        conv2_output_0[0],
        conv3_output_0[0],
        fc1_output_0[0],
        fc2_output_0[0], // which is also lenet output(z) zero point
        conv1_weights_0[0],
        conv2_weights_0[0],
        conv3_weights_0[0],
        fc1_weights_0[0],
        fc2_weights_0[0],
        multipler_conv1.clone(),
        multipler_conv2.clone(),
        multipler_conv3.clone(),
        multipler_fc1.clone(),
        multipler_fc2.clone(),
    );

    println!("finish forwarding");

    for i in 0..z.len() {
        let prediction = argmax_u8(z[i].clone());
        // the rust forward and python forward result vector shall be exactly the same.
        assert_eq!(
            prediction.clone(),
            classification_result[i].clone() as usize
        );
    }
}
