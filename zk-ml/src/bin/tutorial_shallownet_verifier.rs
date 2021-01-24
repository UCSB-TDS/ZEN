use algebra::ed_on_bls12_381::*;
use algebra::CanonicalDeserialize;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use zk_ml::full_circuit::*;
use zk_ml::pedersen_commit::*;
use zk_ml::vanilla::*;
use zk_ml::read_inputs::*;
use algebra_core::ProjectiveCurve;

fn main() {


    let mut f2 = File::open("crs.data").unwrap();
    let param: Parameters<algebra::Bls12_381> = Parameters::deserialize(&mut f2).unwrap();
    let pvk = prepare_verifying_key(&param.vk);

    let mut f2 = File::open("proof.data").unwrap();
    let proof: Proof<algebra::Bls12_381> = Proof::deserialize(&mut f2).unwrap();

    let mut f2 = File::open("x_com.data").unwrap();
    let x_com: PedersenCommitment = PedersenCommitment::deserialize(&mut f2).unwrap();

    let mut f2 = File::open("z_com.data").unwrap();
    let z_com: PedersenCommitment = PedersenCommitment::deserialize(&mut f2).unwrap();

    let inputs = [x_com.x, x_com.y, z_com.x, z_com.y];

    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
}