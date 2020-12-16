use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use r1cs_core::*;
use zk_ml::full_circuit::*;
use zk_ml::pedersen_commit::*;
use zk_ml::read_inputs::read_inputs;
use zk_ml::vanilla::*;

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

    let full_circuit = FullCircuitPedersen {
        x: x,
        x_com: x_com.clone(),
        x_open,
        params: param,
        l1: l1_mat,
        l2: l2_mat,
        z: z,
        z_com: z_com.clone(),
        z_open,
    };

    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    full_circuit
        .clone()
        .generate_constraints(sanity_cs.clone())
        .unwrap();
}
