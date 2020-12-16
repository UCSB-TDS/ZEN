use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml::full_circuit::*;
use zk_ml::pedersen_commit::*;
use zk_ml::read_inputs::*;
use zk_ml::vanilla::*;
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

    //println!("{:?}", l1_mat_multiplier.clone());
    // println!("x: {:?}\n", x);
    // println!("l1_mat: {:?}\n", l1_mat);
    // println!("l2_mat: {:?}\n", l2_mat);
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
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&x, &param, &x_open);
    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&z, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));

    let full_circuit = FullCircuitOpLv3Pedersen {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,
        l1: l1_mat,
        l2: l2_mat,
        z,
        z_com: z_com.clone(),
        z_open,

        x_0: x_0[0],
        y_0: l1_output_0[0],
        z_0: l2_output_0[0],
        l1_mat_0: l1_mat_0[0],
        l2_mat_0: l2_mat_0[0],
        multiplier_l1: l1_mat_multiplier.clone(),
        multiplier_l2: l2_mat_multiplier.clone(),
    };

    // println!("{:?}", full_circuit);
    // sanity checks
    {
        let sanity_cs = ConstraintSystem::<Fq>::new_ref();
        full_circuit
            .clone()
            .generate_constraints(sanity_cs.clone())
            .unwrap();
        let res = sanity_cs.is_satisfied().unwrap();
        println!("are the constraints satisfied?: {}\n", res);

        if !res {
            println!(
                "{:?} {} {:#?}",
                sanity_cs.constraint_names(),
                sanity_cs.num_constraints(),
                sanity_cs.which_is_unsatisfied().unwrap()
            );
        }
    }
    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();
    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    // verifier
    let inputs = [x_com.x, x_com.y, z_com.x, z_com.y];

    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
