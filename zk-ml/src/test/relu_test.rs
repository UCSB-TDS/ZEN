use crate::relu_circuit::ReLUCircuit;
use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::One;
use groth16::*;
use r1cs_core::*;
use rand::RngCore;

#[test]
#[allow(non_snake_case)]
fn test_ZK_ReLU() {
    let mut rng = rand::thread_rng();
    let mut y_in = vec![0i8; M];
    for e in y_in.iter_mut() {
        let tmp = ((rng.next_u32() & 0xFF) as i8) as i8;
        *e = tmp;
    }
    println!("y_in: {:?}\n", y_in);
    let mut y_out = y_in.clone();
    crate::vanilla::relu(&mut y_out);
    println!("y_out: {:?}\n", y_out);

    let circuit = ReLUCircuit { y_in, y_out };

    // sanity checks
    {
        let sanity_cs = ConstraintSystem::<Fq>::new_ref();
        circuit
            .clone()
            .generate_constraints(sanity_cs.clone())
            .unwrap();
        let res = sanity_cs.is_satisfied().unwrap();
        println!("are the constraints satisfied?: {}\n", res);
    }

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(circuit.clone(), &mut rng).unwrap();
    let pvk = prepare_verifying_key(&param.vk);

    // prover
    let proof = create_random_proof(circuit, &param, &mut rng).unwrap();

    // verifier
    let len = pvk.gamma_abc_g1.len() - 1;
    let inputs = vec![Fq::one(); len];
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap())
}
