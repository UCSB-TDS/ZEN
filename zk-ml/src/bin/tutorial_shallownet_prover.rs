use algebra::ed_on_bls12_381::*;
use algebra::CanonicalDeserialize;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use zk_ml::full_circuit::*;
use zk_ml::pedersen_commit::*;
use zk_ml::vanilla::*;
use zk_ml::read_inputs::*;

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

    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&x, &param, &x_open);
    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&z, &param, &z_open);

    let mut buf_x_com = vec![];
    x_com.serialize(&mut buf_x_com).unwrap();
    let mut f = File::create("x_com.data").expect("Unable to create file");
    f.write_all(&buf_x_com).expect("Unable to write");

    let mut buf_z_com = vec![];
    z_com.serialize(&mut buf_z_com).unwrap();
    let mut f = File::create("x_com.data").expect("Unable to create file");
    f.write_all(&buf_z_com).expect("Unable to write");

    let classification_res = argmax_u8(z.clone());

    let full_circuit = FullCircuitOpLv3PedersenClassification {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,
        l1: l1_mat,
        l2: l2_mat,
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


    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    let mut f = File::create("crs.data").expect("Unable to create file");
    f.write_all(&buf).expect("Unable to write");


    let pvk = prepare_verifying_key(&param.vk);

    // prover
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let mut proof_buf = vec![];
    proof.serialize(&mut proof_buf).unwrap();
    let mut f = File::create("proof.data").expect("Unable to create file");
    f.write_all((&proof_buf)).expect("Unable to write data");

    // verifier
    // let inputs = [x_com.x, x_com.y, z_com.x, z_com.y];

    // println!("{:?}", inputs);


}
