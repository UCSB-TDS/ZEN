use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use core::cmp::Ordering;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::boolean::Boolean;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::R1CSVar;

use std::ops::*;

//used in ORL face recognition problem
#[derive(Debug, Clone)]
pub struct CosineSimilarityCircuitU8 {
    pub vec1: Vec<FqVar>,
    pub vec2: Vec<FqVar>,
    pub threshold: u8,
    pub result: bool,
}

impl ConstraintSynthesizer<Fq> for CosineSimilarityCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _cir_number = cs.num_constraints();

        let res = Boolean::<Fq>::constant(self.result);

        let norm1 = scala_cs_helper_var(cs.clone(), &self.vec1.clone(), &self.vec1.clone());
        let norm2 = scala_cs_helper_var(cs.clone(), &self.vec2.clone(), &self.vec2.clone());
        let numerator = scala_cs_helper_var(cs.clone(), &self.vec1.clone(), &self.vec2.clone());
        let ten_thousand: Fq = (10000u64).into();
        let ten_thousand_const = FpVar::<Fq>::Constant(ten_thousand);

        let threshold_fq: Fq = self.threshold.into();
        let threshold_const = FpVar::<Fq>::Constant(threshold_fq);
        let left = ten_thousand_const * numerator.clone() * numerator.clone();
        let right = threshold_const.clone() * threshold_const.clone() * norm2 * norm1;

        if res.value().unwrap_or_default() == true {
            left.enforce_cmp(&right, Ordering::Greater, false).unwrap();
        } else {
            left.enforce_cmp(&right, Ordering::Less, true).unwrap();
        }
        println!(
            "Number of constraints for CosineSimilarity Circuit {}, Accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        Ok(())
    }
}

fn scala_cs_helper_var(cs: ConstraintSystemRef<Fq>, vec1: &[FqVar], vec2: &[FqVar]) -> FqVar {
    let _no_cs = cs.num_constraints();
    if vec1.len() != vec2.len() {
        panic!("scala mul: length not equal");
    }
    let mut res =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();

    for i in 0..vec1.len() {
        res += vec1[i].clone().mul(&vec2[i]);
    }
    res
}
