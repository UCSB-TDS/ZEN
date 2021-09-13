use crate::*;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::boolean::Boolean;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{println, vec, vec::Vec};
use ark_ed_on_bls12_381::{constraints::FqVar, Fq, Fr};
use ark_ff::*;
use core::cmp::Ordering;
use ark_r1cs_std::fields::fp::FpVar;


//used in mnist and cifar10 classification problem
#[derive(Debug, Clone)]
pub struct ArgmaxCircuitU8 {
    pub input: Vec<u8>,
    pub argmax_res: usize,
}

impl ConstraintSynthesizer<Fq> for ArgmaxCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _cir_number = cs.num_constraints();
        let argmax_fq: Fq = self.input[self.argmax_res].into();
        let argmax_var =
            FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "argmax var"), || Ok(argmax_fq)).unwrap();

        for i in 0..self.input.len() {
            let tmp_fq: Fq = self.input[i].into();
            let tmp = FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "argmax tmp var"), || Ok(tmp_fq))
                .unwrap();
            //the argmax result should be larger or equal than all the input values.
            argmax_var
                .enforce_cmp(&tmp, Ordering::Greater, true)
                .unwrap();
        }

        println!(
            "Number of constraints for ArgmaxCircuitU8 Circuit {}, Accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        Ok(())
    }
}


//used in mnist and cifar10 classification problem
#[derive(Debug, Clone)]
pub struct ArgmaxCircuit {
    pub input: Vec<FqVar>,
    pub argmax_res: FqVar,
}

impl ConstraintSynthesizer<Fq> for ArgmaxCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        for j in 0..self.input.len() {
            let tmp = self.input[j].clone();
            //the argmax result should be larger or equal than all the input values.
            self.argmax_res
                .enforce_cmp(&tmp, Ordering::Greater, true)
                .unwrap();
        }

        Ok(())
    }
}
