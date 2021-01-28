use algebra::CanonicalDeserialize;
use groth16::*;
use std::fs::File;
use zk_ml::pedersen_commit::*;

fn main() {
    // in fact we should use secure network transmission. here for simplicity, we write and read CRS and proof from disk.
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
