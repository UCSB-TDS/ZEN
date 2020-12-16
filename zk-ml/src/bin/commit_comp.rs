use algebra::ed_on_bls12_381::Fq;
use algebra::ed_on_bls12_381::*;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use r1cs_core::ConstraintSynthesizer;
use r1cs_core::*;
use zk_ml::pedersen_commit::*;

fn main() {
    let mut rng = rand::thread_rng();
    let len = 230400;
    let param = setup(&[0u8; 32]);
    let input = vec![0u8; len];
    let open = Randomness::<JubJub>(Fr::rand(&mut rng));
    let commit = pedersen_commit(&input, &param, &open);

    let circuit = PedersenComCircuit {
        param,
        input,
        open,
        commit,
    };
    // sanity checks
    {
        let sanity_cs = ConstraintSystem::<Fq>::new_ref();
        circuit.generate_constraints(sanity_cs.clone()).unwrap();
        let res = sanity_cs.is_satisfied().unwrap();
        println!("are the constraints satisfied?: {}\n", res);
        println!(
            "number of constraint {} for data size: {}\n",
            sanity_cs.num_constraints(),
            len
        );
        if !res {
            println!(
                "{:?} {} {:#?}",
                sanity_cs.constraint_names(),
                sanity_cs.num_constraints(),
                sanity_cs.which_is_unsatisfied().unwrap()
            );
        }
    }
}
