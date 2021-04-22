use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_accuracy::lenet_circuit::*;
use zk_ml_accuracy::pedersen_commit::*;
use zk_ml_accuracy::read_inputs::*;
use zk_ml_accuracy::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();

    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_ORL_pretrained/X_q.txt".to_string(),
        100,
        1,
        56,
        46,
    );

    //this is a dummy true label result file. we did not save it from python inference results.
    // just use it to calculate the total number of constraints in ZEN_accuracy scheme
    let true_labels: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_ORL_pretrained/LeNet_ORL_accuracy.txt".to_string(),
        100,
    );
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

    let mut num_of_correct_prediction = 0u64;
    let mut accuracy_results = Vec::new();
    for i in 0..x.len() {
        let is_the_same_person: bool =
            cosine_similarity(z[i].clone(), person_feature_vector.clone(), 50);
        let mut is_the_same_person_numeric = 0u8;
        if is_the_same_person {
            is_the_same_person_numeric = 1u8;
        }
        if is_the_same_person_numeric == true_labels[i].clone() {
            accuracy_results.push(1u8);
            num_of_correct_prediction += 1;
        } else {
            accuracy_results.push(0u8);
        }
    }

    let begin = Instant::now();
    let param = setup(&[0u8; 32]);

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
    let mut accuracy_open = Vec::new();
    let mut accuracy_com = Vec::new();
    let mut accuracy_input: Vec<Vec<u8>> = Vec::new();

    //assume each machine process batch_size images out of the total testing public dataset.
    //but we do not implement parallel infer 100 images. we only calculate the constraint number of inference on one image plus the accuracy commitment accumulation check

    let batch_size = 1;
    for i in (0..x.len()).step_by(batch_size) {
        let tmp_accuracy_data = &accuracy_results[i..i + batch_size];
        //println!("accuracy slice {:?}", tmp_accuracy_data);
        accuracy_input.push(tmp_accuracy_data.iter().cloned().collect());
        let tmp_accuracy_open = Randomness(Fr::rand(&mut rng));
        let tmp_accuracy_com = pedersen_commit(&tmp_accuracy_data, &param, &tmp_accuracy_open);
        accuracy_com.push(tmp_accuracy_com);
        accuracy_open.push(tmp_accuracy_open);
    }

    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let x_current_batch: Vec<Vec<Vec<Vec<u8>>>> = (&x[0..batch_size]).iter().cloned().collect();
    let true_labels_batch: Vec<u8> = (&true_labels[0..batch_size]).iter().cloned().collect();

    let full_circuit = LeNetCircuitU8OptimizedLv3PedersenRecognition {
        params: param.clone(),
        x: x_current_batch.clone(),
        person_feature_vector: person_feature_vector.clone(),
        true_labels: true_labels_batch.clone(),
        threshold: 50u8,
        accuracy_result: accuracy_input[0].clone(),
        accuracy_open: accuracy_open[0].clone(),
        accuracy_com: accuracy_com[0].clone(),

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
    };

    //aggregate multiple previous inference circuit output
    //(for simplicity, we directly use the commitment of accuracy results to check whether the number of correct prediction is correct)
    let accuracy_sumcheck_circuit = PedersenComAccuracyCircuit {
        param: param.clone(),
        input: accuracy_input.clone(),
        open: accuracy_open.clone(),
        commit: accuracy_com.clone(),
        num_of_correct_prediction: num_of_correct_prediction,
    };

    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let param_acc = generate_random_parameters::<algebra::Bls12_381, _, _>(
        accuracy_sumcheck_circuit.clone(),
        &mut rng,
    )
    .unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    let mut buf_acc = vec![];
    param_acc.serialize(&mut buf_acc).unwrap();
    println!("crs size: {}", buf.len() + buf_acc.len());

    let pvk = prepare_verifying_key(&param.vk);
    let pvk_acc = prepare_verifying_key(&param_acc.vk);

    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let proof_acc = create_random_proof(accuracy_sumcheck_circuit, &param_acc, &mut rng).unwrap();

    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    //verifier
    let x_input = convert_4d_vector_into_fq(x_current_batch.clone());
    let true_label_inputs: Vec<Fq> = convert_1d_vector_into_fq(true_labels_batch.clone());
    let person_feature_input = convert_1d_vector_into_fq(person_feature_vector.clone());
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
    let mut acc_com_inputs = Vec::new();

    acc_com_inputs.push(accuracy_com[0].x);
    acc_com_inputs.push(accuracy_com[0].y);

    let inputs = [
        x_input[..].as_ref(),
        true_label_inputs.as_ref(),
        person_feature_input.as_ref(),
        conv1_inputs.as_ref(),
        conv2_inputs.as_ref(),
        conv3_inputs.as_ref(),
        fc1_inputs.as_ref(),
        fc2_inputs.as_ref(),
        acc_com_inputs.as_ref(),
    ]
    .concat();

    //prepare commitment aggregated accuracy circuit inputs
    let mut inputs_acc: Vec<Fq> = Vec::new();
    // number of correct predictions is public input for verification
    let num_of_correct_prediction_fq: Fq = num_of_correct_prediction.into();
    inputs_acc.push(num_of_correct_prediction_fq);
    for i in 0..accuracy_com.len() {
        //commitment of accuracy vector obtained from each batch inference.
        inputs_acc.push(accuracy_com[i].x);
        inputs_acc.push(accuracy_com[i].y);
    }

    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    assert!(verify_proof(&pvk_acc, &proof_acc, &inputs_acc[..]).unwrap());

    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
