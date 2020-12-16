use crate::mul_circuit::FCCircuit;
use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::One;
use groth16::*;
use r1cs_core::*;
use rand::RngCore;

#[test]
fn test_fc() {
    let mut rng = rand::thread_rng();
    let mut x = vec![0i8; L];
    for e in x.iter_mut() {
        let tmp = ((rng.next_u32() & 0xFF) as i8) as i8;
        *e = tmp;
    }
    // println!("x: {:?}\n", x);
    let mut l1 = vec![vec![0i8; L]; M];
    for i in 0..M {
        for j in 0..L {
            let tmp = ((rng.next_u32() & 0xFF) as i8) as i8;
            l1[i][j] = tmp;
        }
    }
    // println!("y: {:?}\n", y);

    let mut y = vec![0i8; M];
    let l1_ref: Vec<&[i8]> = l1.iter().map(|x| x.as_ref()).collect();
    crate::vanilla::vec_mat_mul(&x, &l1_ref, &mut y);
    // println!("z: {:?}\n", z);

    let circuit = FCCircuit { x, l1_mat: l1, y };

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
    let pvk = prepare_verifying_key(&param.vk);

    // prover
    let proof = create_random_proof(circuit, &param, &mut rng).unwrap();

    // verifier
    let len = pvk.gamma_abc_g1.len() - 1;
    let inputs = vec![Fq::one(); len];
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap())

    // assert!(false)
}
