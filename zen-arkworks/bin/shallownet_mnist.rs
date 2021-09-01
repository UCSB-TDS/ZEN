use std::time::Instant;
use pedersen_example::*;
use ark_serialize::CanonicalDeserialize;
use ark_ff::UniformRand;
use ark_groth16::*;
use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_bls12_381::Bls12_381;

fn main() {
    let mut rng = rand::thread_rng();
    //let (x, l1_mat, l2_mat): (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) = read_shallownet_inputs_u8();
    let x: Vec<u8> = read_vector1d("pretrained_model/shallownet/X_q.txt".to_string(), 784); // only read one image
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

    let z: Vec<u8> = full_circuit_forward_u8(
        x.clone(),
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

    let begin = Instant::now();
    let param = pedersen_setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&x, &param, &x_open);
    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&z, &param, &z_open);

    let l1_open = Randomness(Fr::rand(&mut rng));
    let l1_mat_1d = convert_2d_vector_into_1d(l1_mat.clone());
    let l1_com_vec = pedersen_commit_long_vector(&l1_mat_1d, &param, &l1_open);
    let l2_open = Randomness(Fr::rand(&mut rng));
    let l2_mat_1d = convert_2d_vector_into_1d(l2_mat.clone());
    let l2_com_vec = pedersen_commit_long_vector(&l2_mat_1d, &param, &l2_open);

    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));
    let classification_res = argmax_u8(z.clone());

    let full_circuit = FullCircuitOpLv3PedersenClassification {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,
        l1: l1_mat,
        l1_open: l1_open.clone(),
        l1_com_vec: l1_com_vec.clone(),
        l2: l2_mat,
        l2_open: l2_open.clone(),
        l2_com_vec: l2_com_vec.clone(),
        z: z.clone(),
        z_com: z_com.clone(),
        z_open,
        argmax_res: classification_res,

        x_0: x_0[0],
        y_0: l1_output_0[0],
        z_0: l2_output_0[0],
        l1_mat_0: l1_mat_0[0],
        l2_mat_0: l2_mat_0[0],
        multiplier_l1: l1_mat_multiplier.clone(),
        multiplier_l2: l2_mat_multiplier.clone(),
    };

    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();
    println!("setup time {:?}", end.duration_since(begin));

    // let mut buf = vec![];
    // param.serialize(&mut buf).unwrap();
    // println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    // verifier
    let mut l1_inputs = Vec::new();
    for i in 0..l1_com_vec.len() {
        l1_inputs.push(l1_com_vec[i].x);
        l1_inputs.push(l1_com_vec[i].y);
    }
    let mut l2_inputs = Vec::new();
    for i in 0..l2_com_vec.len() {
        l2_inputs.push(l2_com_vec[i].x);
        l2_inputs.push(l2_com_vec[i].y);
    }
    let other_commit_inputs = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();

    let inputs = [
        other_commit_inputs[..].as_ref(),
        l1_inputs.as_ref(),
        l2_inputs.as_ref(),
    ]
    .concat();
    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
