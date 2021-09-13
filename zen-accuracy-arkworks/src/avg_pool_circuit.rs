use crate::*;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::boolean::*;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{println, vec, vec::Vec};
use ark_ed_on_bls12_381::{constraints::FqVar, Fq};
use ark_ff::*;


//stranded encoding for avg pool layer
#[derive(Debug, Clone)]
pub struct AvgPoolCircuitLv3 {
    pub x: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
    pub remainder: Vec<Vec<Vec<Vec<u8>>>>,
    // we do not need the quantization parameters to calculate the avg pool output
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuitLv3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "AvgPoolCircuitU8BitDecomposeOptimized is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let num_images = self.x.len();
        let num_channels = self.x[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        // let delta_bits_length = 20;
        // let delta_fq = (2u128.pow(delta_bits_length)).into();
        // let delta = FpVar::<Fq>::Constant(delta_fq);

        //let k_simd: usize = (254u32 / delta_bits_length) as usize;
        //let k_simd: usize = 1;
        let kernel_size_fq: Fq = (self.kernel_size as u32).into();
        let kernel_size_const = FpVar::<Fq>::Constant(kernel_size_fq);
        for n in 0..num_images {
            for h in 0..(input_height / self.kernel_size) {
                for w in 0..(input_width / self.kernel_size) {
                    for c in 0..num_channels {
                        let tmp = sum_helper_fq(
                            cs.clone(),
                            self.x[n][c].clone(),
                            self.kernel_size * h,
                            self.kernel_size * w,
                            self.kernel_size,
                        );

                        let yy_var = self.y[n][c][h][w].clone();

                        let remainder: Fq = (self.remainder[n][c][h][w] as u64).into();
                        let remainder_var = FpVar::<Fq>::new_witness(
                            ark_relations::ns!(cs, "remainder gadget"),
                            || Ok(remainder),
                        )
                        .unwrap();

                        let output_var =
                            yy_var * kernel_size_const.clone() * kernel_size_const.clone()
                                + remainder_var;

                        tmp.enforce_equal(&output_var).unwrap();
                    }
                }
            }
        }

        Ok(())
    }
}

fn sum_helper_fq(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<FqVar>>,
    h_index: usize,
    w_index: usize,
    kernel_size: usize,
) -> FqVar {
    let mut tmp =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "avg pool sum helper gadget"), || {
            Ok(Fq::zero())
        })
        .unwrap();
    for i in h_index..(h_index + kernel_size) {
        for j in w_index..(w_index + kernel_size) {
            tmp += x[i][j].clone();
        }
    }

    tmp
}
