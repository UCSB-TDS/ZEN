use crate::pedersen::*;
use ark_crypto_primitives::commitment::pedersen::constraints::CommGadget;
use ark_crypto_primitives::commitment::pedersen::Randomness;
use ark_crypto_primitives::CommitmentGadget;
use ark_ed_on_bls12_381::{constraints::FqVar, Fq, Fr};

use ark_ff::UniformRand;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::groups::curves::twisted_edwards::AffineVar;
use ark_r1cs_std::uint8::UInt8;
use ark_relations::r1cs::ConstraintSystem;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{println, vec, vec::Vec};
// alias for R1CS gadgets of pedersen commitment scheme
pub type PedersenComSchemeVar = CommGadget<JubJub, EdwardsVar, Window>;
pub type PedersenParamVar =
    <PedersenComSchemeVar as CommitmentGadget<PedersenComScheme, Fq>>::ParametersVar;
pub type PedersenRandomnessVar =
    <PedersenComSchemeVar as CommitmentGadget<PedersenComScheme, Fq>>::RandomnessVar;
pub type PedersenCommitmentVar = AffineVar<EdwardsParameters, FpVar<Fq>>;
use crate::*;
use crate::JubJub;

// ZK proved statements:
//  commit(data, open) = commitment
#[derive(Clone)]
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
            PedersenParamVar::new_input(ark_relations::ns!(cs, "gadget_parameters"), || {
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
            PedersenRandomnessVar::new_witness(ark_relations::ns!(cs, "gadget_randomness"), || {
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
            PedersenCommitmentVar::new_input(ark_relations::ns!(cs, "gadget_commitment"), || {
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


fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
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
            FpVar::<Fq>::new_input(ark_relations::ns!(cs, "accuracy"), || Ok(accuracy_fq)).unwrap();
        let zero_fq : Fq = 0u64.into();
        let mut sum_of_correct_prediction =
            FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "accuracy"), || Ok(zero_fq)).unwrap();

        for i in 0..self.input.len() {
            //assume we have multiple machine to run accuracy inference in parallel, and finally we need to sum the prediction correctness results together.

            let prediction_correctness_tmp = generate_fqvar(cs.clone(), self.input[i].clone());
            for j in 0..prediction_correctness_tmp.len() {
                sum_of_correct_prediction += prediction_correctness_tmp[j].clone();
            }

            // step 1. Allocate Parameters for perdersen commitment
            let param_var =
                PedersenParamVar::new_witness(ark_relations::ns!(cs, "gadget_parameters"), || {
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
                PedersenRandomnessVar::new_witness(ark_relations::ns!(cs, "gadget_randomness"), || {
                    Ok(&self.open[i])
                })
                .unwrap();

            // step 4. Allocate the output
            let result_var =
                PedersenComSchemeVar::commit(&param_var, &input_var, &open_var).unwrap();

            // circuit to compare the commited value with supplied value

            let commitment_var2 =
                PedersenCommitmentVar::new_input(ark_relations::ns!(cs, "gadget_commitment"), || {
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
