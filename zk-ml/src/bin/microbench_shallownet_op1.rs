use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use r1cs_core::*;

use zk_ml::full_circuit::*;
use zk_ml::pedersen_commit::*;
use zk_ml::read_inputs::read_shallownet_inputs_u8;
use zk_ml::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();
    let (x, l1_mat, l2_mat): (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) = read_shallownet_inputs_u8();

    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&x, &param, &x_open);

    // println!("x: {:?}\n", x);
    // println!("l1_mat: {:?}\n", l1_mat);
    // println!("l2_mat: {:?}\n", l2_mat);

    let z: Vec<u8> = full_circuit_forward_u8(
        x.clone(),
        l1_mat.clone(),
        l2_mat.clone(),
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        vec![0.0001; l1_mat.len()],
        vec![0.0001; l2_mat.len()],
    );
    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&z, &param, &z_open);

    let full_circuit = FullCircuitOpLv1Pedersen {
        params: param,
        x: x,
        x_com: x_com.clone(),
        x_open: x_open,
        l1: l1_mat.clone(),
        l2: l2_mat.clone(),
        z: z,
        z_com: z_com.clone(),
        z_open: z_open,

        x_0: DEFAULT_ZERO_POINT,
        y_0: DEFAULT_ZERO_POINT,
        z_0: DEFAULT_ZERO_POINT,
        l1_mat_0: DEFAULT_ZERO_POINT,
        l2_mat_0: DEFAULT_ZERO_POINT,
        multiplier_l1: vec![0.0001; l1_mat.len()],
        multiplier_l2: vec![0.0001; l2_mat.len()],
    };

    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    full_circuit
        .clone()
        .generate_constraints(sanity_cs.clone())
        .unwrap();
}
