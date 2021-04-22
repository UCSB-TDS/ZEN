use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use zk_ml_private_model_baseline::full_circuit::*;
use zk_ml_private_model_baseline::pedersen_commit::*;
use zk_ml_private_model_baseline::read_inputs::read_inputs;
use zk_ml_private_model_baseline::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();
    let (x, l1_mat, l2_mat) = read_inputs();
    // println!("x: {:?}\n", x);
    // println!("l1_mat: {:?}\n", l1_mat);
    // println!("l2_mat: {:?}\n", l2_mat);

    let z = forward(x.clone(), l1_mat.clone(), l2_mat.clone());

    let x_u8 = x.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

    let z_u8 = z.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&x_u8, &param, &x_open);

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&z_u8, &param, &z_open);

    let l1_open = Randomness(Fr::rand(&mut rng));
    let l1_mat_1d = convert_2d_vector_into_1d(l1_mat.clone());
    let l1_mat_1d_u8 = l1_mat_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let l1_com_vec = pedersen_commit_long_vector(&l1_mat_1d_u8, &param, &l1_open);
    let l2_open = Randomness(Fr::rand(&mut rng));
    let l2_mat_1d = convert_2d_vector_into_1d(l2_mat.clone());
    let l2_mat_1d_u8 = l2_mat_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();
    let l2_com_vec = pedersen_commit_long_vector(&l2_mat_1d_u8, &param, &l2_open);

    let full_circuit = FullCircuitPedersen {
        x: x,
        x_com: x_com.clone(),
        x_open,
        params: param,
        l1: l1_mat,
        l1_open: l1_open.clone(),
        l1_com_vec: l1_com_vec.clone(),
        l2: l2_mat,
        l2_open: l2_open.clone(),
        l2_com_vec: l2_com_vec.clone(),
        z: z,
        z_com: z_com.clone(),
        z_open,
    };

    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    // full_circuit
    //     .clone()
    //     .generate_constraints(sanity_cs.clone())
    //     .unwrap();

    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
}
