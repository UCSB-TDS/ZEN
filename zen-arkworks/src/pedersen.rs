use ark_crypto_primitives::commitment::pedersen;
use ark_crypto_primitives::commitment::pedersen::Commitment;
use ark_crypto_primitives::commitment::pedersen::Randomness;
use ark_crypto_primitives::CommitmentScheme;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::cmp::*;

//=======================
// curves: JubJub and BLS
//=======================
#[cfg(feature="bls12_381")]
pub type JubJub = ark_ed_on_bls12_381::EdwardsProjective;
#[cfg(feature="bls12_377")]
pub type JubJub = ark_ed_on_bls12_377::EdwardsProjective;
#[cfg(feature="bn254c")]
pub type JubJub = ark_ed_on_bn254::EdwardsProjective;
//=======================
// pedersen hash and related defintions
// the hash function is defined over the JubJub curve
// this parameter allows us to commit to 256 * 4 = 1024 bits
//=======================
pub const PERDERSON_WINDOW_SIZE: usize = 100;
pub const PERDERSON_WINDOW_NUM: usize = 256;

#[derive(Clone)]
pub struct Window;
impl pedersen::Window for Window {
    const WINDOW_SIZE: usize = PERDERSON_WINDOW_SIZE;
    const NUM_WINDOWS: usize = PERDERSON_WINDOW_NUM;
}

// alias for pedersen commitment scheme
pub type PedersenComScheme = Commitment<JubJub, Window>;
pub type PedersenCommitment = <PedersenComScheme as CommitmentScheme>::Output;
pub type PedersenParam = <PedersenComScheme as CommitmentScheme>::Parameters;
pub type PedersenRandomness = Randomness<JubJub>;

#[allow(dead_code)]
pub fn pedersen_setup(seed: &[u8; 32]) -> PedersenParam {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    PedersenComScheme::setup(&mut rng).unwrap()
}

#[allow(dead_code)]
pub fn pedersen_commit(
    x: &[u8],
    param: &PedersenParam,
    r: &PedersenRandomness,
) -> PedersenCommitment {
    PedersenComScheme::commit(param, &x, r).unwrap()
}



pub fn pedersen_commit_long_vector(
    x: &[u8],
    param: &PedersenParam,
    r: &PedersenRandomness,
) -> Vec<PedersenCommitment> {
    let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment
    let num_of_commit_needed = x.len() / len_per_commit + 1;
    let mut commit_res = Vec::new();
    for i in 0..num_of_commit_needed {
        let mut tmp = Vec::new();
        for j in i * len_per_commit..min((i + 1) * len_per_commit, x.len()) {
            tmp.push(x[j]);
        }
        commit_res.push(PedersenComScheme::commit(param, &tmp, r).unwrap());
    }
    commit_res
}
