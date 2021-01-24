use crate::*;
use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::boolean::Boolean;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
// statement:
//  if y_in[i] < 0, y_out[i] = 0;
//  else y_out[i] = y_in[i]

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
                Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                    Ok(self.cmp_res[i])
                })
                .unwrap();
            self.y_out[i]
                .enforce_equal(&cmp.select(&self.y_in[i], &zero_var).unwrap())
                .unwrap();

            //this is the old way without using `select` API.

            // if cmp.value().unwrap_or_default() == true {
            //     //assert_eq!(e.clone().to_bits_le().unwrap().value().unwrap(), self.y_out[i].to_bits_le().unwrap().value().unwrap());
            //     e.enforce_equal(&self.y_out[i]).unwrap();
            // } else {
            //     //assert_eq!(zero_var.clone().to_bits_le().unwrap().value().unwrap()[0..7], self.y_out[i].to_bits_le().unwrap().value().unwrap()[0..7]);
            //     self.y_out[i].enforce_equal(&zero_var).unwrap();
            // }
        }
        Ok(())
    }
}
