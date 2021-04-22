use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_accuracy::full_circuit::*;
use zk_ml_accuracy::pedersen_commit::*;
use zk_ml_accuracy::read_inputs::*;
use zk_ml_accuracy::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();
    let x: Vec<Vec<u8>> =
        read_vector2d("pretrained_model/shallownet/X_q.txt".to_string(), 100, 784); // only read one image
    let true_labels: Vec<u8> = read_vector1d("pretrained_model/shallownet/Y.txt".to_string(), 100); //read 100 image inference results
    let l1_mat: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/shallownet/l1_weight_q.txt".to_string(),
        128,
        784,
    );
    let l2_mat: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/shallownet/l2_weight_q.txt".to_string(),
        10,
        128,
    );
    let x_0: Vec<u8> = read_vector1d("pretrained_model/shallownet/X_z.txt".to_string(), 1);
    let l1_output_0: Vec<u8> =
        read_vector1d("pretrained_model/shallownet/l1_output_z.txt".to_string(), 1);
    let l2_output_0: Vec<u8> =
        read_vector1d("pretrained_model/shallownet/l2_output_z.txt".to_string(), 1);
    let l1_mat_0: Vec<u8> =
        read_vector1d("pretrained_model/shallownet/l1_weight_z.txt".to_string(), 1);
    let l2_mat_0: Vec<u8> =
        read_vector1d("pretrained_model/shallownet/l2_weight_z.txt".to_string(), 1);

    let l1_mat_multiplier: Vec<f32> = read_vector1d_f32(
        "pretrained_model/shallownet/l1_weight_s.txt".to_string(),
        128,
    );
    let l2_mat_multiplier: Vec<f32> = read_vector1d_f32(
        "pretrained_model/shallownet/l2_weight_s.txt".to_string(),
        10,
    );

    let mut num_of_correct_prediction = 0u64;
    let mut accuracy_results = Vec::new();
    for i in 0..x.len() {
        let z: Vec<u8> = full_circuit_forward_u8(
            x[i].clone(),
            l1_mat.clone(),
            l2_mat.clone(),
            x_0[0],
            l1_output_0[0],
            l2_output_0[0],
            l1_mat_0[0],
            l2_mat_0[0],
            l1_mat_multiplier.clone(),
            l2_mat_multiplier.clone(),
        );
        let argmax_res = argmax_u8(z) as u8;
        if argmax_res == true_labels[i] {
            //inference accuracy
            accuracy_results.push(1u8);
            num_of_correct_prediction += 1;
        } else {
            accuracy_results.push(0u8);
        }
    }
    //println!("prediction accuracy : {:?}", accuracy_results.clone());

    let begin = Instant::now();
    let param = setup(&[0; 32]);

    let l1_open = Randomness(Fr::rand(&mut rng));
    let l1_mat_1d = convert_2d_vector_into_1d(l1_mat.clone());
    let l1_com_vec = pedersen_commit_long_vector(&l1_mat_1d, &param, &l1_open);
    let l2_open = Randomness(Fr::rand(&mut rng));
    let l2_mat_1d = convert_2d_vector_into_1d(l2_mat.clone());
    let l2_com_vec = pedersen_commit_long_vector(&l2_mat_1d, &param, &l2_open);

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

    //for current machine, we only process one batch
    let x_current_batch: Vec<Vec<u8>> = (&x[0..batch_size]).iter().cloned().collect();
    let true_labels_batch: Vec<u8> = (&true_labels[0..batch_size]).iter().cloned().collect();

    //println!("true label batch {:?}", true_labels_batch);
    let full_circuit = FullCircuitClassificationAccuracy {
        params: param.clone(),

        l1: l1_mat,
        l1_open: l1_open.clone(),
        l1_com_vec: l1_com_vec.clone(),
        l2: l2_mat,
        l2_open: l2_open.clone(),
        l2_com_vec: l2_com_vec.clone(),

        //we only process one batch on one machine
        x: x_current_batch.clone(),
        true_labels: true_labels_batch.clone(),
        accuracy_result: accuracy_input[0].clone(),
        accuracy_open: accuracy_open[0].clone(),
        accuracy_com: accuracy_com[0].clone(),

        x_0: x_0[0],
        y_0: l1_output_0[0],
        z_0: l2_output_0[0],
        l1_mat_0: l1_mat_0[0],
        l2_mat_0: l2_mat_0[0],
        multiplier_l1: l1_mat_multiplier.clone(),
        multiplier_l2: l2_mat_multiplier.clone(),
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

    // verifier
    let mut inputs: Vec<Fq> = Vec::new();
    for i in 0..l1_com_vec.len() {
        inputs.push(l1_com_vec[i].x);
        inputs.push(l1_com_vec[i].y);
    }
    for i in 0..l2_com_vec.len() {
        inputs.push(l2_com_vec[i].x);
        inputs.push(l2_com_vec[i].y);
    }
    inputs.push(accuracy_com[0].x);
    inputs.push(accuracy_com[0].y);

    let x_inputs: Vec<Fq> = convert_2d_vector_into_fq(x_current_batch.clone());
    let true_label_inputs: Vec<Fq> = convert_1d_vector_into_fq(true_labels_batch.clone());
    let inputs: Vec<Fq> = [
        inputs[..].as_ref(),
        x_inputs.as_ref(),
        true_label_inputs.as_ref(),
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
