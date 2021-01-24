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
        Vec<Vec<Vec<Vec<u8>>>>,
        Vec<Vec<Vec<Vec<u8>>>>,
        Vec<Vec<Vec<Vec<u8>>>>,
        Vec<Vec<u8>>,
        Vec<Vec<u8>>,
    ) = read_lenet_medium_inputs_u8_cifar();
    let x: Vec<Vec<Vec<Vec<u8>>>> = read_cifar_input_u8();
    println!("LeNet small optimized level 1  on CIFAR dataset");
    //we use random numbers to do the microbenchmark

    let z: Vec<Vec<u8>> = lenet_circuit_forward_u8(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        DEFAULT_ZERO_POINT,
        vec![0.0001; conv1_w.len()],
        vec![0.0001; conv2_w.len()],
        vec![0.0001; conv3_w.len()],
        vec![0.0001; fc1_w.len()],
        vec![0.0001; fc2_w.len()],
    );

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();

    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let full_circuit = LeNetCircuitU8OptimizedLv1Pedersen {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv1_weights: conv1_w.clone(),
        conv2_weights: conv2_w.clone(),
        conv3_weights: conv3_w.clone(),
        fc1_weights: fc1_w.clone(),
        fc2_weights: fc2_w.clone(),

        // I use random numbers for check circuit verification process correctness
        //zero points for quantization.
        x_0: 10u8,
        conv1_output_0: 10u8,
        conv2_output_0: 10u8,
        conv3_output_0: 10u8,
        fc1_output_0: 10u8,
        fc2_output_0: 10u8, // which is also lenet output(z) zero point

        conv1_weights_0: 10u8,
        conv2_weights_0: 10u8,
        conv3_weights_0: 10u8,
        fc1_weights_0: 10u8,
        fc2_weights_0: 10u8,

        //multiplier for quantization
        multiplier_conv1: vec![0.1; conv1_w.len()],
        multiplier_conv2: vec![0.1; conv2_w.len()],
        multiplier_conv3: vec![0.1; conv3_w.len()],
        multiplier_fc1: vec![0.0001; fc1_w.len()],
        multiplier_fc2: vec![0.0001; fc2_w.len()],

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
