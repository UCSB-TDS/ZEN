use crate::commit_circuit::MLCommitCircuit;
use crate::vanilla::*;
use crate::*;
use algebra::ed_on_bls12_381::*;
use groth16::*;
use r1cs_core::*;
use rand::RngCore;

#[test]
fn test_commit_x() {
    let mut rng = rand::thread_rng();
    let mut x = vec![0i8; 784];
    for e in x.iter_mut() {
        let tmp = ((rng.next_u32() & 0xFF) as i8) as i8;
        *e = tmp;
    }
    let (commit, open) = commit_x(&x, &[0u8; 32]);

    let circuit = MLCommitCircuit {
        is_x: true,
        data: x,
        open,
        com: commit,
    };

    println!("circuit complete");
    // sanity checks
    {
        let sanity_cs = ConstraintSystem::<Fq>::new_ref();
        circuit
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

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(circuit.clone(), &mut rng).unwrap();

    println!("parameter generated");
    let pvk = prepare_verifying_key(&param.vk);

    println!("pvk generated");

    // prover
    let proof = create_random_proof(circuit, &param, &mut rng).unwrap();

    println!("proof generated");

    // verifier
    let mut inputs: Vec<Fq> = vec![];
    for e in commit.iter() {
        let mut f = *e;
        for _ in 0..8 {
            inputs.push((f & 0b1).into());
            f = f >> 1;
        }
    }
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap())
}

#[test]
fn test_commit_z() {
    let mut rng = rand::thread_rng();
    let mut z = vec![0i8; 10];
    for e in z.iter_mut() {
        let tmp = rng.next_u32() as i8;
        *e = tmp;
    }
    let (commit, open) = commit_z(&z, &[0u8; 32]);

    let circuit = MLCommitCircuit {
        is_x: false,
        data: z,
        open,
        com: commit,
    };

    println!("circuit complete");
    // sanity checks
    {
        let sanity_cs = ConstraintSystem::<Fq>::new_ref();
        circuit
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

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(circuit.clone(), &mut rng).unwrap();

    println!("parameter generated");
    let pvk = prepare_verifying_key(&param.vk);

    println!("pvk generated");

    // prover
    let proof = create_random_proof(circuit, &param, &mut rng).unwrap();

    println!("proof generated");

    // verifier
    let mut inputs: Vec<Fq> = vec![];
    for e in commit.iter() {
        let mut f = *e;
        for _ in 0..8 {
            inputs.push((f & 0b1).into());
            f = f >> 1;
        }
    }
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap())
}
