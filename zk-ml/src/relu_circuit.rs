use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
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
            let in_var = match *e >= 0 {
                // do nothing
                true => {
                    let tmp: Fq = (*e as u32).into();
                    FpVar::<Fq>::new_witness(
                        r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                        || Ok(tmp),
                    )
                    .unwrap()
                }
                // zero gadget
                false => {
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, format!("input 0 gadget",)), || {
                        Ok(Fq::zero())
                    })
                    .unwrap()
                }
            };
            // cast y_out[i] as a gadget
            let tmp: Fq = (self.y_out[i] as u32).into();
            let out_var = FpVar::<Fq>::new_witness(
                r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                || Ok(tmp),
            )
            .unwrap();

            // enforce y_in[i] == y_out[i]
            in_var.enforce_equal(&out_var).unwrap();
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
        for (i, e) in self.y_in.iter().enumerate() {
            // cast y_in[i] as a gadget
            let in_var = match *e >= self.y_zeropoint {
                // do nothing
                true => {
                    let tmp: Fq = (*e as u32).into();
                    FpVar::<Fq>::new_witness(
                        r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                        || Ok(tmp),
                    )
                    .unwrap()
                }
                // zero gadget
                false => zero_var.clone(),
            };
            // cast y_out[i] as a gadget
            let tmp: Fq = (self.y_out[i] as u32).into();
            let out_var = FpVar::<Fq>::new_witness(
                r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                || Ok(tmp),
            )
            .unwrap();
            //println!("y_in {:?}\n y_out {:?}\n", in_var.value().unwrap(), out_var.value().unwrap());
            // enforce y_in[i] == y_out[i]
            in_var.enforce_equal(&out_var).unwrap();
        }
        Ok(())
    }
}
