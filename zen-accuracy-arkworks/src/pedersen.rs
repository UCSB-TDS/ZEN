use ark_crypto_primitives::commitment::pedersen;
use ark_crypto_primitives::commitment::pedersen::Commitment;
use ark_crypto_primitives::commitment::pedersen::Randomness;
use ark_crypto_primitives::CommitmentScheme;
use ark_std::rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::cmp::*;

//=======================
// curves: JubJub and BLS
//=======================
pub type JubJub = ark_ed_on_bls12_381::EdwardsProjective;

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

//aggregate multiple inference accuracy result vector(1 for correct prediction. 0 for wrong prediction)
//sum them up and check of the number of correct prediction is correct.
#[derive(Clone)]
pub struct PedersenComAccuracyCircuit {
    pub param: PedersenParam,
    pub input: Vec<Vec<u8>>,
    pub open: Vec<PedersenRandomness>,
    pub commit: Vec<PedersenCommitment>,
    pub num_of_correct_prediction: u64,
}
// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for PedersenComAccuracyCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let accuracy_fq: Fq = self.num_of_correct_prediction.into();
        let accuracy_var =
            FpVar::<Fq>::new_input(r1cs_core::ns!(cs, "accuracy"), || Ok(accuracy_fq)).unwrap();

        let mut sum_of_correct_prediction =
            FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "accuracy"), || Ok(Fq::zero())).unwrap();

        for i in 0..self.input.len() {
            //assume we have multiple machine to run accuracy inference in parallel, and finally we need to sum the prediction correctness results together.

            let prediction_correctness_tmp = generate_fqvar(cs.clone(), self.input[i].clone());
            for j in 0..prediction_correctness_tmp.len() {
                sum_of_correct_prediction += prediction_correctness_tmp[j].clone();
            }

            // step 1. Allocate Parameters for perdersen commitment
            let param_var =
                PedersenParamVar::new_witness(r1cs_core::ns!(cs, "gadget_parameters"), || {
                    Ok(&self.param)
                })
                .unwrap();

            // step 2. Allocate inputs
            let mut input_var = vec![];
            for input_byte in self.input[i].iter() {
                input_var.push(UInt8::new_witness(cs.clone(), || Ok(*input_byte)).unwrap());
            }

            // step 3. Allocate the opening
            let open_var =
                PedersenRandomnessVar::new_witness(r1cs_core::ns!(cs, "gadget_randomness"), || {
                    Ok(&self.open[i])
                })
                .unwrap();

            // step 4. Allocate the output
            let result_var =
                PedersenComSchemeVar::commit(&param_var, &input_var, &open_var).unwrap();

            // circuit to compare the commited value with supplied value

            let commitment_var2 =
                PedersenCommitmentVar::new_input(r1cs_core::ns!(cs, "gadget_commitment"), || {
                    Ok(self.commit[i])
                })
                .unwrap();
            result_var.enforce_equal(&commitment_var2).unwrap();
        }

        //check whether the number of correct predictions is correct

        accuracy_var
            .enforce_equal(&sum_of_correct_prediction)
            .unwrap();
        println!(
            "total cs for AccuracyCommitment Circuit: {}",
            cs.num_constraints()
        );
        Ok(())
    }
}
