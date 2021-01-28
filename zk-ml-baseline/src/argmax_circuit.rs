use crate::*;
use algebra::ed_on_bls12_381::*;
use core::cmp::Ordering;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::fields::fp::FpVar;

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
            FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "argmax var"), || Ok(argmax_fq)).unwrap();

        for i in 0..self.input.len() {
            let tmp_fq: Fq = self.input[i].into();
            let tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "argmax tmp var"), || Ok(tmp_fq))
                .unwrap();
            //the argmax result should be larger or equal than all the input values.
            argmax_var
                .enforce_cmp(&tmp, Ordering::Greater, true)
                .unwrap();
        }

        println!(
            "Number of constraints for ArgmaxCircuitU8 Circuit {}, Accumulated constaints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        Ok(())
    }
}
