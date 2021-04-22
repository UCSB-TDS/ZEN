use crate::*;
use algebra::ed_on_bls12_381::*;
use core::cmp::Ordering;
use r1cs_core::*;
use r1cs_std::ed_on_bls12_381::FqVar;

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
