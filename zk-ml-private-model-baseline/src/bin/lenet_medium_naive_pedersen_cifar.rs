use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_private_model_baseline::lenet_circuit::*;
use zk_ml_private_model_baseline::pedersen_commit::*;
use zk_ml_private_model_baseline::read_inputs::*;
use zk_ml_private_model_baseline::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();
    let (conv1_w, conv2_w, conv3_w, fc1_w, fc2_w): (
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<Vec<Vec<i8>>>>,
        Vec<Vec<i8>>,
        Vec<Vec<i8>>,
    ) = read_lenet_medium_inputs_i8_cifar();
    let x: Vec<Vec<Vec<Vec<i8>>>> = read_cifar_input_i8();
    println!("LeNet medium baseline on cifar dataset");

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

    let conv1_open = Randomness(Fr::rand(&mut rng));
    let conv1_weights_1d = convert_4d_vector_into_1d(conv1_w.clone());
    let conv1_weights_1d = conv1_weights_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let conv1_com_vec = pedersen_commit_long_vector(&conv1_weights_1d, &param, &conv1_open);
    let conv2_open = Randomness(Fr::rand(&mut rng));
    let conv2_weights_1d = convert_4d_vector_into_1d(conv2_w.clone());
    let conv2_weights_1d = conv2_weights_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let conv2_com_vec = pedersen_commit_long_vector(&conv2_weights_1d, &param, &conv2_open);
    let conv3_open = Randomness(Fr::rand(&mut rng));
    let conv3_weights_1d = convert_4d_vector_into_1d(conv3_w.clone());
    let conv3_weights_1d = conv3_weights_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let conv3_com_vec = pedersen_commit_long_vector(&conv3_weights_1d, &param, &conv3_open);

    let fc1_open = Randomness(Fr::rand(&mut rng));
    let fc1_weights_1d = convert_2d_vector_into_1d(fc1_w.clone());
    let fc1_weights_1d = fc1_weights_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let fc1_com_vec = pedersen_commit_long_vector(&fc1_weights_1d, &param, &fc1_open);
    let fc2_open = Randomness(Fr::rand(&mut rng));
    let fc2_weights_1d = convert_2d_vector_into_1d(fc2_w.clone());
    let fc2_weights_1d = fc2_weights_1d
        .iter()
        .map(|x| (*x as i8) as u8)
        .collect::<Vec<u8>>();

    let fc2_com_vec = pedersen_commit_long_vector(&fc2_weights_1d, &param, &fc2_open);

    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let full_circuit = LeNetCircuitNaivePedersen {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

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

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
    };

    //microbenchmark only for getting the number of constraints.
    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    // full_circuit
    //     .clone()
    //     .generate_constraints(sanity_cs.clone())
    //     .unwrap();
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
}
