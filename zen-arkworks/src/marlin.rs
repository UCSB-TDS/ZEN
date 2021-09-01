use crate::pedersen::*;
use crate::r1cs::*;
use ark_bls12_381::Bls12_381;
use ark_crypto_primitives::commitment::pedersen::Randomness;
use ark_ed_on_bls12_381::*;
use ark_ff::UniformRand;
use ark_marlin::ahp::AHPForR1CS;
use ark_marlin::*;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_poly_commit::PCUniversalParams;
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_relations::r1cs::ConstraintSystem;
use blake2::Blake2s;

pub fn marlin_test() {
    let mut rng = rand::thread_rng();
    let len = 256;
    let param = pedersen_setup(&[0u8; 32]);
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
    let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    circuit
        .clone()
        .generate_constraints(sanity_cs.clone())
        .unwrap();
    // let res = sanity_cs.is_satisfied().unwrap();

    let no_cs = sanity_cs.num_constraints();
    let no_var = sanity_cs.num_witness_variables();
    let no_non_zero = sanity_cs.num_instance_variables();

    println!("inputs {} {} {}", no_cs, no_var, no_non_zero);

    let index = AHPForR1CS::index(circuit.clone()).unwrap();
    println!("index {}", index.max_degree());

    type MultiPC = MarlinKZG10<Bls12_381, DensePolynomial<Fq>>;
    type MarlinInst = Marlin<Fq, MultiPC, Blake2s>;

    let srs = MarlinInst::universal_setup(no_cs, no_var * 2, 0, &mut rng).unwrap();
    println!("srs generated");
    println!(
        "srs max degree: {}, index degree {}",
        srs.max_degree(),
        index.max_degree()
    );

    let (pk, vk) = MarlinInst::index(&srs, circuit.clone()).unwrap();
    println!("keys generated");

    let proof = MarlinInst::prove(&pk, circuit, &mut rng).unwrap();
    println!("proof generated");

    assert!(MarlinInst::verify(&vk, &[commit.x, commit.y], &proof, &mut rng).unwrap());
}
