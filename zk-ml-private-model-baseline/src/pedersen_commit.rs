use algebra::ed_on_bls12_381;
use algebra::ed_on_bls12_381::EdwardsParameters;
use algebra::ed_on_bls12_381::Fq;
use crypto_primitives::commitment::pedersen;
use crypto_primitives::commitment::pedersen::constraints::CommGadget;
use crypto_primitives::commitment::pedersen::Commitment;
use crypto_primitives::commitment::pedersen::Randomness;
use crypto_primitives::CommitmentGadget;
use crypto_primitives::CommitmentScheme;
use r1cs_core::ConstraintSynthesizer;
use r1cs_core::ConstraintSystemRef;
use r1cs_core::SynthesisError;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::EdwardsVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::groups::curves::twisted_edwards::AffineVar;
use r1cs_std::uint8::UInt8;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::cmp::*;

//=======================
// curves: JubJub and BLS
//=======================
pub type JubJub = ed_on_bls12_381::EdwardsProjective;
pub type JubJubFq = ed_on_bls12_381::Fq;
pub type JubJubFr = ed_on_bls12_381::Fr;

//=======================
// pedersen hash and related defintions
// the hash function is defined over the JubJub curve
//=======================

//TODO under different input size. choose different window size
pub const PERDERSON_WINDOW_SIZE: usize = 100; // this is for 28*28*1 MNIST input
                                              //pub const PERDERSON_WINDOW_SIZE: usize = 100; // this is for 32*32*3 CIFAR u8 input
                                              //pub const PERDERSON_WINDOW_SIZE: usize = 100; // this is for 46*56*1 u8 FACE input

//this parameter is consistent
pub const PERDERSON_WINDOW_NUM: usize = 256;

#[derive(Clone)]
pub struct Window;
impl pedersen::Window for Window {
    const WINDOW_SIZE: usize = PERDERSON_WINDOW_SIZE;
    const NUM_WINDOWS: usize = PERDERSON_WINDOW_NUM;
}

// alias
pub type PedersenComScheme = Commitment<JubJub, Window>;
pub type PedersenCommitment = <PedersenComScheme as CommitmentScheme>::Output;
pub type PedersenParam = <PedersenComScheme as CommitmentScheme>::Parameters;
pub type PedersenRandomness = Randomness<JubJub>;
// r1cs alias
pub type PedersenComSchemeVar = CommGadget<JubJub, EdwardsVar, Window>;
pub type PedersenParamVar =
    <PedersenComSchemeVar as CommitmentGadget<PedersenComScheme, Fq>>::ParametersVar;
pub type PedersenRandomnessVar =
    <PedersenComSchemeVar as CommitmentGadget<PedersenComScheme, Fq>>::RandomnessVar;
pub type PedersenCommitmentVar = AffineVar<EdwardsParameters, FpVar<Fq>>;

pub fn setup(seed: &[u8; 32]) -> PedersenParam {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    PedersenComScheme::setup(&mut rng).unwrap()
}

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
// ZK proved statements:
//  commit(data, open) = commitment
pub struct PedersenComCircuit {
    pub param: PedersenParam,
    pub input: Vec<u8>,
    pub open: PedersenRandomness,
    pub commit: PedersenCommitment,
}
// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for PedersenComCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertions)]
        println!("is setup mode?: {}", cs.is_in_setup_mode());
        let _cs_no = cs.num_constraints();
        // step 1. Allocate Parameters for perdersen commitment
        let param_var =
            PedersenParamVar::new_witness(r1cs_core::ns!(cs, "gadget_parameters"), || {
                Ok(&self.param)
            })
            .unwrap();
        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for parameters: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 2. Allocate inputs
        let mut input_var = vec![];
        for input_byte in self.input.iter() {
            input_var.push(UInt8::new_witness(cs.clone(), || Ok(*input_byte)).unwrap());
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for account: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // step 3. Allocate the opening
        let open_var =
            PedersenRandomnessVar::new_witness(r1cs_core::ns!(cs, "gadget_randomness"), || {
                Ok(&self.open)
            })
            .unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for opening: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // step 4. Allocate the output
        let result_var = PedersenComSchemeVar::commit(&param_var, &input_var, &open_var).unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for commitment: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // circuit to compare the commited value with supplied value

        let commitment_var2 =
            PedersenCommitmentVar::new_input(r1cs_core::ns!(cs, "gadget_commitment"), || {
                Ok(self.commit)
            })
            .unwrap();
        result_var.enforce_equal(&commitment_var2).unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for comparison: {}", _cs_no);

        #[cfg(debug_assertions)]
        println!("total cs for Commitment: {}", cs.num_constraints());
        Ok(())
    }
}
