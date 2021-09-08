use std::time::Instant;
use pedersen_example::*;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use ark_ff::UniformRand;
use ark_groth16::*;
use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_bls12_381::Bls12_381;
use crate::full_circuit::convert_2d_vector_into_1d;
use ark_std::test_rng;
use ark_sponge::{ CryptographicSponge, FieldBasedCryptographicSponge, poseidon::PoseidonSponge};

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    let mut rng = test_rng();
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


    let  parameter : SPNGParam = poseidon_parameters_for_test_s();
    let mut x_sponge = PoseidonSponge::< >::new(&parameter);
    let mut l1_sponge = PoseidonSponge::< >::new(&parameter);
    let mut l2_sponge = PoseidonSponge::< >::new(&parameter);
    let mut z_sponge = PoseidonSponge::< >::new(&parameter);

    x_sponge.absorb(&x);
    let x_squeeze : SPNGOutput=x_sponge.squeeze_native_field_elements(x.clone().len() / 32 + 1);
    z_sponge.absorb(&z);
    let z_squeeze : SPNGOutput=z_sponge.squeeze_native_field_elements(z.clone().len() / 32 + 1);


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

    let full_circuit = FullCircuitOpLv3Poseidon {
        params: parameter.clone(),
        x: x.clone(),
        x_squeeze: x_squeeze.clone(),
        l1: l1_mat,
        l1_squeeze: l1_squeeze.clone(),
        l2: l2_mat,
        l2_squeeze: l2_squeeze.clone(),
        z: z.clone(),
        z_squeeze: z_squeeze.clone(),

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



    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));


    let inputs = [
        x_squeeze,
        l1_squeeze,
        l2_squeeze,
        z_squeeze,
    ]
    .concat();
    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..].as_ref()).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
