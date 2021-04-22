use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use core::cmp::Ordering;
use num_traits::Pow;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::boolean::Boolean;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::R1CSVar;
use r1cs_std::ToBitsGadget;
use std::ops::*;
// statement:
//  if y_in[i] < 0, y_out[i] = 0;
//  else y_out[i] = y_in[i]
#[derive(Debug, Clone)]
pub(crate) struct ReLUCircuit {
    pub(crate) y_in: Y,
    pub(crate) y_out: Y,
}

impl ConstraintSynthesizer<Fq> for ReLUCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        for (i, e) in self.y_in.iter().enumerate() {
            // cast y_in[i] as a gadget

            let mut cmp =
                Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                    Ok(false)
                })
                .unwrap();
            let zero_var = FqVar::Constant(Fq::zero());
            let mut in_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, format!("input 0 gadget",)), || {
                    Ok(Fq::zero())
                })
                .unwrap();
            if (self.y_in[i] > 0) {
                cmp =
                    Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                        Ok(true)
                    })
                    .unwrap();
                let tmp_in: Fq = (*e as u32).into();
                in_var = FpVar::<Fq>::new_witness(
                    r1cs_core::ns!(cs, format!("input 0 gadget",)),
                    || Ok(tmp_in),
                )
                .unwrap();
            }
            // cast y_out[i] as a gadget
            let tmp: Fq = (self.y_out[i] as u32).into();
            let out_var = FpVar::<Fq>::new_witness(
                r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                || Ok(tmp),
            )
            .unwrap();

            // enforce y_in[i] == y_out[i]
            out_var
                .enforce_equal(&cmp.select(&in_var, &zero_var).unwrap())
                .unwrap();
        }
        Ok(())
    }
}

// statement:
//  if y_in[i] < 0, y_out[i] = 0;
//  else y_out[i] = y_in[i]
#[derive(Debug, Clone)]
pub(crate) struct ReLUCircuitU8 {
    pub(crate) y_in: Vec<u8>,
    pub(crate) y_out: Vec<u8>,
    pub(crate) y_zeropoint: u8,
}

impl ConstraintSynthesizer<Fq> for ReLUCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let zero: Fq = self.y_zeropoint.into();
        let zero_var = FqVar::Constant(zero);
        //zero point is constant wire in the circuit
        for (i, e) in self.y_in.iter().enumerate() {
            let mut cmp;

            let tmp_zero: Fq = (self.y_zeropoint as u32).into();
            let zero_var = FqVar::Constant(tmp_zero);
            // cast y_out[i] as a gadget
            let tmp: Fq = (self.y_out[i] as u32).into();
            let out_var = FpVar::<Fq>::new_witness(
                r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                || Ok(tmp),
            )
            .unwrap();

            let tmp_in: Fq = (*e as u32).into();
            let in_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, format!("input 0 gadget",)), || {
                    Ok(tmp_in)
                })
                .unwrap();
            if (self.y_in[i] > self.y_zeropoint) {
                cmp =
                    Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                        Ok(true)
                    })
                    .unwrap();
            } else {
                cmp =
                    Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                        Ok(false)
                    })
                    .unwrap();
            }

            // enforce y_in[i] == y_out[i]
            out_var
                .enforce_equal(&cmp.select(&in_var, &zero_var).unwrap())
                .unwrap();
        }
        Ok(())
    }
}

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
