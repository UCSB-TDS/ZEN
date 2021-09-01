use std::time::Instant;
use pedersen_example::*;
use ark_serialize::CanonicalDeserialize;
use ark_ff::UniformRand;
use ark_groth16::*;
use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_bls12_381::Bls12_381;
use pedersen_example::full_circuit::convert_2d_vector_into_1d;

fn main() {
    let mut rng = rand::thread_rng();

    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_ORL_pretrained/X_q.txt".to_string(),
        1,
        1,
        56,
        46,
    ); // only read one image
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv1_weight_q.txt".to_string(),
        6,
        1,
        5,
        5,
    );
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv2_weight_q.txt".to_string(),
        16,
        6,
        5,
        5,
    );
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv3_weight_q.txt".to_string(),
        120,
        16,
        4,
        4,
    );
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear1_weight_q.txt".to_string(),
        84,
        120 * 5 * 8,
    );
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear2_weight_q.txt".to_string(),
        40,
        84,
    );

    let x_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/X_z.txt".to_string(),
        1,
    );
    let conv1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv1_output_z.txt".to_string(),
        1,
    );
    let conv2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv2_output_z.txt".to_string(),
        1,
    );
    let conv3_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv3_output_z.txt".to_string(),
        1,
    );
    let fc1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear1_output_z.txt".to_string(),
        1,
    );
    let fc2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear2_output_z.txt".to_string(),
        1,
    );

    let conv1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv1_weight_z.txt".to_string(),
        1,
    );
    let conv2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv2_weight_z.txt".to_string(),
        1,
    );
    let conv3_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv3_weight_z.txt".to_string(),
        1,
    );
    let fc1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear1_weight_z.txt".to_string(),
        1,
    );
    let fc2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear2_weight_z.txt".to_string(),
        1,
    );

    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv1_weight_s.txt".to_string(),
        6,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv2_weight_s.txt".to_string(),
        16,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_conv3_weight_s.txt".to_string(),
        120,
    );

    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear1_weight_s.txt".to_string(),
        84,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_Small_linear2_weight_s.txt".to_string(),
        40,
    );

    let person_feature_vector = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/person_feature_vector.txt".to_string(),
        40,
    );

    println!("finish reading parameters");

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    //actually we only process one image per forward.
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
        multiplier_conv1.clone(),
        multiplier_conv2.clone(),
        multiplier_conv3.clone(),
        multiplier_fc1.clone(),
        multiplier_fc2.clone(),
    );
    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();
    //println!("x outside {:?}", x.clone());
    println!("z outside {:?}", flattened_z1d.clone());

    let begin = Instant::now();
    let param = pedersen_setup(&[0u8; 32]);
    let x_open = Randomness::<JubJub>(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    let z_open = Randomness::<JubJub>(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);

    let conv1_open = Randomness(Fr::rand(&mut rng));
    let conv1_weights_1d = convert_4d_vector_into_1d(conv1_w.clone());
    let conv1_com_vec = pedersen_commit_long_vector(&conv1_weights_1d, &param, &conv1_open);
    let conv2_open = Randomness(Fr::rand(&mut rng));
    let conv2_weights_1d = convert_4d_vector_into_1d(conv2_w.clone());
    let conv2_com_vec = pedersen_commit_long_vector(&conv2_weights_1d, &param, &conv2_open);
    let conv3_open = Randomness(Fr::rand(&mut rng));
    let conv3_weights_1d = convert_4d_vector_into_1d(conv3_w.clone());
    let conv3_com_vec = pedersen_commit_long_vector(&conv3_weights_1d, &param, &conv3_open);

    let fc1_open = Randomness(Fr::rand(&mut rng));
    let fc1_weights_1d = convert_2d_vector_into_1d(fc1_w.clone());
    let fc1_com_vec = pedersen_commit_long_vector(&fc1_weights_1d, &param, &fc1_open);
    let fc2_open = Randomness(Fr::rand(&mut rng));
    let fc2_weights_1d = convert_2d_vector_into_1d(fc2_w.clone());
    let fc2_com_vec = pedersen_commit_long_vector(&fc2_weights_1d, &param, &fc2_open);

    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let is_the_same_person: bool =
        cosine_similarity(z[0].clone(), person_feature_vector.clone(), 50);
    println!("is the same person ? {}", is_the_same_person);

    let full_circuit = LeNetCircuitU8OptimizedLv3PedersenRecognition {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv1_weights: conv1_w.clone(),
        conv1_open: conv1_open.clone(),
        conv1_com_vec: conv1_com_vec.clone(),
        conv2_weights: conv2_w.clone(),
        conv2_open: conv2_open.clone(),
        conv2_com_vec: conv2_com_vec.clone(),
        conv3_weights: conv3_w.clone(),
        conv3_open: conv3_open.clone(),
        conv3_com_vec: conv3_com_vec.clone(),
        fc1_weights: fc1_w.clone(),
        fc1_open: fc1_open.clone(),
        fc1_com_vec: fc1_com_vec.clone(),
        fc2_weights: fc2_w.clone(),
        fc2_open: fc2_open.clone(),
        fc2_com_vec: fc2_com_vec.clone(),

        //zero points for quantization.
        x_0: x_0[0],
        conv1_output_0: conv1_output_0[0],
        conv2_output_0: conv2_output_0[0],
        conv3_output_0: conv3_output_0[0],
        fc1_output_0: fc1_output_0[0],
        fc2_output_0: fc2_output_0[0], // which is also lenet output(z) zero point

        conv1_weights_0: conv1_weights_0[0],
        conv2_weights_0: conv2_weights_0[0],
        conv3_weights_0: conv3_weights_0[0],
        fc1_weights_0: fc1_weights_0[0],
        fc2_weights_0: fc2_weights_0[0],

        //multiplier for quantization
        multiplier_conv1: multiplier_conv1.clone(),
        multiplier_conv2: multiplier_conv2.clone(),
        multiplier_conv3: multiplier_conv3.clone(),
        multiplier_fc1: multiplier_fc1.clone(),
        multiplier_fc2: multiplier_fc2.clone(),

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        person_feature_vector: person_feature_vector.clone(),
        threshold: 50,
        result: is_the_same_person,
    };



    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();
    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();

    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    //verifier
    let mut conv1_inputs = Vec::new();
    for i in 0..conv1_com_vec.len() {
        conv1_inputs.push(conv1_com_vec[i].x);
        conv1_inputs.push(conv1_com_vec[i].y);
    }
    let mut conv2_inputs = Vec::new();
    for i in 0..conv2_com_vec.len() {
        conv2_inputs.push(conv2_com_vec[i].x);
        conv2_inputs.push(conv2_com_vec[i].y);
    }
    let mut conv3_inputs = Vec::new();
    for i in 0..conv3_com_vec.len() {
        conv3_inputs.push(conv3_com_vec[i].x);
        conv3_inputs.push(conv3_com_vec[i].y);
    }
    let mut fc1_inputs = Vec::new();
    for i in 0..fc1_com_vec.len() {
        fc1_inputs.push(fc1_com_vec[i].x);
        fc1_inputs.push(fc1_com_vec[i].y);
    }
    let mut fc2_inputs = Vec::new();
    for i in 0..fc2_com_vec.len() {
        fc2_inputs.push(fc2_com_vec[i].x);
        fc2_inputs.push(fc2_com_vec[i].y);
    }

    let other_commit_inputs = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();
    let inputs = [
        other_commit_inputs[..].as_ref(),
        conv1_inputs.as_ref(),
        conv2_inputs.as_ref(),
        conv3_inputs.as_ref(),
        fc1_inputs.as_ref(),
        fc2_inputs.as_ref(),
    ]
    .concat();
    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
