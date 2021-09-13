use crate::*;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::boolean::Boolean;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{println, vec, vec::Vec};
use ark_ed_on_bls12_381::{constraints::FqVar, Fq, Fr};
use ark_ff::*;


#[derive(Debug, Clone)]
pub(crate) struct ReLUCircuitOp3 {
    pub(crate) y_in: Vec<FqVar>,
    pub(crate) y_out: Vec<FqVar>,
    pub(crate) y_zeropoint: u8,
    pub(crate) cmp_res: Vec<bool>,
}

impl ConstraintSynthesizer<Fq> for ReLUCircuitOp3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        //println!("ReLU zero point {}", self.y_zeropoint);
        let zero: Fq = self.y_zeropoint.into();
        let zero_var = FqVar::Constant(zero);
        //zero point is constant wire in the circuit

        for i in 0..self.y_in.len() {
            let cmp =
                Boolean::new_witness(ark_relations::ns!(cs, "relu"), || {
                    Ok(self.cmp_res[i])
                })
                .unwrap();
            self.y_out[i]
                .enforce_equal(&cmp.select(&self.y_in[i], &zero_var).unwrap())
                .unwrap();

        }
        Ok(())
    }
}
