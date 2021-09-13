use std::time::Instant;
use zen_accuracy_arkworks::*;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use ark_ff::UniformRand;
use ark_groth16::*;
use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_bls12_381::Bls12_381;
use ark_std::test_rng;
use ark_sponge::{ CryptographicSponge, FieldBasedCryptographicSponge, poseidon::PoseidonSponge};
use crate::full_circuit::*;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    let mut rng = test_rng();
    //let (x, l1_mat, l2_mat): (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) = read_shallownet_inputs_u8();
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

    //println!("zero points x_0 {}  l1_out_0 {} l2_out_0 {}, l1_mat_0 {}, l2_mat_0 {}", x_0[0], l1_output_0[0], l2_output_0[0], l1_mat_0[0], l2_mat_0[0]);

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


    let  parameter : SPNGParam = poseidon_parameters_for_test_s();
    let mut l1_sponge = PoseidonSponge::< >::new(&parameter);
    let mut l2_sponge = PoseidonSponge::< >::new(&parameter);
    let mut acc_sponge = PoseidonSponge::< >::new(&parameter);

    let mut accuracy_squeeze = Vec::new();
    let mut accuracy_input: Vec<Vec<u8>> = Vec::new();
    let batch_size = 1;
    for i in (0..x.len()).step_by(batch_size) {
        let tmp_accuracy_data = &accuracy_results[i..i + batch_size];
        //println!("accuracy slice {:?}", tmp_accuracy_data);
        accuracy_input.push(tmp_accuracy_data.iter().cloned().collect());
        acc_sponge.absorb(&tmp_accuracy_data);
        let tmp_acc_squeeze : SPNGOutput = acc_sponge.squeeze_native_field_elements(tmp_accuracy_data.clone().len() / 32 + 1);
        accuracy_squeeze.push(tmp_acc_squeeze);
    }


    //for current machine, we only process one batch
    let x_current_batch: Vec<Vec<u8>> = (&x[0..batch_size]).iter().cloned().collect();
    let true_labels_batch: Vec<u8> = (&true_labels[0..batch_size]).iter().cloned().collect();


    let l1_mat_1d = convert_2d_vector_into_1d(l1_mat.clone());
    let l2_mat_1d = convert_2d_vector_into_1d(l2_mat.clone());
    l1_sponge.absorb(&l1_mat_1d);
    let l1_squeeze : SPNGOutput=l1_sponge.squeeze_native_field_elements(l1_mat_1d.clone().len() / 32 + 1);

    l2_sponge.absorb(&l2_mat_1d);
    let l2_squeeze : SPNGOutput=l2_sponge.squeeze_native_field_elements(l2_mat_1d.clone().len() / 32 + 1);
    // println!("x_squeeze len {}", x_squeeze.len());
    // println!("l1_squeeze len {}", l1_squeeze.len());
    // println!("l2_squeeze len {}", l2_squeeze.len());
    // println!("z_squeeze len {}", z_squeeze.len());





    let full_circuit = FullCircuitClassificationAccuracy {
        params: parameter.clone(),
        l1: l1_mat,
        l1_squeeze: l1_squeeze.clone(),
        l2: l2_mat,
        l2_squeeze: l2_squeeze.clone(),

        x: x_current_batch.clone(),
        true_labels: true_labels_batch.clone(),
        accuracy_result: accuracy_input[0].clone(),
        accuracy_squeeze: accuracy_squeeze[0].clone(),

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

    let mut acc_sponge2 = PoseidonSponge::< >::new(&parameter);

    acc_sponge2.absorb(&accuracy_results);
    let accuracy_squeeze2 : SPNGOutput = acc_sponge2.squeeze_native_field_elements(accuracy_results.clone().len() / 32 + 1);
    

    let accuracy_sumcheck_circuit = SPNGAccuracyCircuit{
        param: parameter.clone(),   
        input: accuracy_results.clone(),
        output: accuracy_squeeze2.clone(),
        num_of_correct_prediction: num_of_correct_prediction
    };




    println!("start generating random parameters");
    let begin = Instant::now();

    //pre-computed parameters
    let param =
        generate_random_parameters::<Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let param_acc = generate_random_parameters::<Bls12_381, _, _>(
        accuracy_sumcheck_circuit.clone(),
        &mut rng,
    ).unwrap();
    let end = Instant::now();
    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    let mut buf_acc = vec![];
    param_acc.serialize(&mut buf_acc).unwrap();
    println!("crs size: {}", buf.len() + buf_acc.len() );

    let pvk = prepare_verifying_key(&param.vk);
    let pvk_acc = prepare_verifying_key(&param_acc.vk);

    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let proof_acc = create_random_proof(accuracy_sumcheck_circuit, &param_acc, &mut rng).unwrap();

    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));
    let x_inputs: Vec<Fq> = convert_2d_vector_into_fq(x_current_batch.clone());
    let true_label_inputs: Vec<Fq> = convert_1d_vector_into_fq(true_labels_batch.clone());

    let inputs = [
        l1_squeeze,
        l2_squeeze,
        accuracy_squeeze[0].clone(),
        x_inputs,
        true_label_inputs
    ]
    .concat();

    //prepare commitment aggregated accuracy circuit inputs
    // number of correct predictions is public input for verification

    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..].as_ref()).unwrap());
    assert!(verify_proof(&pvk_acc, &proof_acc, &accuracy_squeeze2).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
