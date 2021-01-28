use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_baseline::lenet_circuit::*;
use zk_ml_baseline::pedersen_commit::*;
use zk_ml_baseline::read_inputs::*;
use zk_ml_baseline::vanilla::*;
fn main() {
    let mut rng = rand::thread_rng();
    let (conv1_w, conv2_w, conv3_w, fc1_w, fc2_w): (
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<i8>>,
        Vec<Vec<i8>>,
    ) = read_lenet_small_inputs_i8_cifar();
    let x: Vec<Vec<Vec<Vec<i8>>>> = read_cifar_input_i8();
    println!("LeNet small baseline on cifar dataset");

    let z: Vec<Vec<i8>> = lenet_circuit_forward(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
    );

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<i8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<i8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<i8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_x1d_u8 = flattened_x1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let flattened_z1d: Vec<i8> = z.clone().into_iter().flatten().collect();

    let flattened_z1d_u8 = flattened_z1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d_u8, &param, &x_open);

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d_u8, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let full_circuit = LeNetCircuitNaivePedersen {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv1_weights: conv1_w.clone(),
        conv2_weights: conv2_w.clone(),
        conv3_weights: conv3_w.clone(),
        fc1_weights: fc1_w.clone(),
        fc2_weights: fc2_w.clone(),

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
    };

    //microbenchmark only for getting the number of constraints.
    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    full_circuit
        .clone()
        .generate_constraints(sanity_cs.clone())
        .unwrap();
}
