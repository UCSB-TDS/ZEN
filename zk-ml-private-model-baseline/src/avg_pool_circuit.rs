use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use core::cmp::Ordering;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::boolean::Boolean;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::*;
use std::cmp;
use std::ops::*;

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
                    for c in (0..num_channels) {
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
                            r1cs_core::ns!(cs, "remainder gadget"),
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
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "avg pool sum helper gadget"), || {
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

#[derive(Debug, Clone)]
pub struct AvgPoolCircuitU8BitDecomposeOptimized {
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
    pub remainder: Vec<Vec<Vec<Vec<u8>>>>,
    // we do not need the quantization parameters to calculate the avg pool output
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuitU8BitDecomposeOptimized {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "AvgPoolCircuitU8BitDecomposeOptimized is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let batch_size = self.x.len();
        let num_channels = self.x[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        for n in 0..batch_size {
            for c in 0..num_channels {
                for h in 0..(input_height / self.kernel_size) {
                    for w in 0..(input_width / self.kernel_size) {
                        // self.y[n][c][x][y] = np.mean(self.x[n][c][kernel_size*x:kernel_size*(x+1)][kernel_size*y:kernel_size*(y+1)])

                        let tmp = sum_helper_u8(
                            cs.clone(),
                            self.x[n][c].clone(),
                            self.kernel_size * h,
                            self.kernel_size * w,
                            self.kernel_size,
                        );

                        let yy: Fq = (self.y[n][c][h][w] as u32
                            * self.kernel_size as u32
                            * self.kernel_size as u32
                            + self.remainder[n][c][h][w] as u32)
                            .into();
                        let yy_var = FpVar::<Fq>::new_witness(
                            r1cs_core::ns!(cs, "avg pool output gadget"),
                            || Ok(yy),
                        )
                        .unwrap();
                        //assert_eq!(yy_var.to_bits_le().unwrap().value().unwrap(), tmp.to_bits_le().unwrap().value().unwrap());
                        yy_var.enforce_equal(&tmp).unwrap();
                    }
                }
            }
        }
        Ok(())
    }
}

fn sum_helper_u8(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<u8>>,
    h_index: usize,
    w_index: usize,
    kernel_size: usize,
) -> FqVar {
    //we don't need to multiply the multiplier. we can obtain the mean value directly on u8 type(accumulated using u32 to avoid overflow)
    let mut tmp =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "avg pool gadget"), || Ok(Fq::zero())).unwrap();

    for i in h_index..(h_index + kernel_size) {
        for j in w_index..(w_index + kernel_size) {
            let aa: Fq = x[i][j].into();
            let a_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
            tmp += a_var;
        }
    }

    tmp
}

#[derive(Debug, Clone)]
pub struct AvgPoolCircuit {
    pub x: Vec<Vec<Vec<Vec<i8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<i8>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "AvgPoolCircuitU8BitDecomposeOptimized is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let t = Boolean::<Fq>::constant(true);
        let f = Boolean::<Fq>::constant(false);

        let batch_size = self.x.len();
        let num_channels = self.x[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        for n in 0..batch_size {
            for c in 0..num_channels {
                for h in 0..(input_height / self.kernel_size) {
                    for w in 0..(input_width / self.kernel_size) {
                        // self.y[n][c][x][y] = np.mean(self.x[n][c][kernel_size*x:kernel_size*(x+1)][kernel_size*y:kernel_size*(y+1)])

                        let (tmp, sign) = sum_helper(
                            cs.clone(),
                            self.x[n][c].clone(),
                            self.kernel_size * h,
                            self.kernel_size * w,
                            self.kernel_size,
                        );

                        let zz: Fq = (if self.y[n][c][h][w] < 0 {
                            sign.enforce_equal(&f).unwrap();
                            -self.y[n][c][h][w]
                        } else {
                            sign.enforce_equal(&t).unwrap();
                            self.y[n][c][h][w]
                        } as u32)
                            .into();

                        let zz_var = FpVar::<Fq>::new_witness(
                            r1cs_core::ns!(cs, "avg pool output gadget"),
                            || Ok(zz),
                        )
                        .unwrap();
                        //assert_eq!(yy_var.to_bits_le().unwrap().value().unwrap(), tmp.to_bits_le().unwrap().value().unwrap());

                        zz_var.enforce_equal(&tmp).unwrap();
                    }
                }
            }
        }
        Ok(())
    }
}

fn sum_helper(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<i8>>,
    h_index: usize,
    w_index: usize,
    kernel_size: usize,
) -> (FqVar, Boolean<Fq>) {
    //we don't need to multiply the multiplier. we can obtain the mean value directly on u8 type(accumulated using u32 to avoid overflow)
    // tmp for position sum in i8
    let mut tmp1 = 0i32;
    // tmp for negative sum in i8
    let mut tmp2 = 0i32;
    // tmp for position sum in circuit
    let mut pos =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "pos gadget"), || Ok(Fq::zero())).unwrap();
    // tmp for position sum in circuit
    let mut neg =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "neg gadget"), || Ok(Fq::zero())).unwrap();
    let t = Boolean::constant(true);
    let f = Boolean::constant(false);

    for i in h_index..(h_index + kernel_size) {
        for j in w_index..(w_index + kernel_size) {
            if x[i][j] >= 0 {
                let aa: Fq = (x[i][j] as u32).into();
                let a_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "pos gadget"), || Ok(aa)).unwrap();
                pos += a_var;
                tmp1 += x[i][j] as i32;
            } else {
                let aa: Fq = (-x[i][j] as u32).into();
                let a_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "neg gadget"), || Ok(aa)).unwrap();
                neg += a_var;
                tmp2 -= x[i][j] as i32;
            }
        }
    }

    let res = if tmp1 >= tmp2 {
        pos.enforce_cmp(&neg, Ordering::Greater, true).unwrap();
        let remainder: Fq = (((tmp1 - tmp2) % (kernel_size * kernel_size) as i32) as u32).into();
        let remainder_var =
            FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder))
                .unwrap();
        (pos.sub(&neg).sub(remainder_var), t)
    } else {
        neg.enforce_cmp(&pos, Ordering::Greater, false).unwrap();
        let remainder: Fq = (((tmp2 - tmp1) % (kernel_size * kernel_size) as i32) as u32).into();
        let remainder_var =
            FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder))
                .unwrap();
        (neg.sub(pos).sub(remainder_var), f)
    };

    res
}

#[derive(Debug, Clone)]
pub struct AvgPoolCircuitU8 {
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
    // we do not need the quantization parameters to calculate the avg pool output
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "AvgPoolCircuitU8BitDecomposeOptimized is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let batch_size = self.x.len();
        let num_channels = self.x[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        for n in 0..batch_size {
            for c in 0..num_channels {
                for h in 0..(input_height / self.kernel_size) {
                    for w in 0..(input_width / self.kernel_size) {
                        // self.y[n][c][x][y] = np.mean(self.x[n][c][kernel_size*x:kernel_size*(x+1)][kernel_size*y:kernel_size*(y+1)])

                        let tmp = sum_helper_u8(
                            cs.clone(),
                            self.x[n][c].clone(),
                            self.kernel_size * h,
                            self.kernel_size * w,
                            self.kernel_size,
                        );
                        let mut tmp_bits = tmp.to_bits_le().unwrap();

                        //use bit decomposition to implement AVG division. only supports for 2*2 and 4*4 8*8 kernel size
                        tmp_bits.drain(0..self.kernel_size);
                        tmp_bits.drain(8..);
                        let mut shift_res = FpVar::<Fq>::new_witness(
                            r1cs_core::ns!(cs, "shift result gadget"),
                            || Ok(Fq::zero()),
                        )
                        .unwrap();
                        let a = 2u8;
                        let b = 1u8;
                        let double: Fq = a.into();
                        let double_var = FpVar::Constant(double);
                        let one: Fq = b.into();
                        let one_var = FpVar::<Fq>::Constant(one);
                        let zero_var = FpVar::<Fq>::Constant(Fq::zero());
                        for (_i, bit) in tmp_bits.iter().rev().enumerate() {
                            //This is the correct way to pack bits back to FpVar
                            shift_res = shift_res
                                .mul(&double_var)
                                .add(&bit.select(&one_var, &zero_var).unwrap());
                        }
                        let yy: Fq = (self.y[n][c][h][w] as u32).into();
                        let yy_var = FpVar::<Fq>::new_witness(
                            r1cs_core::ns!(cs, "avg pool output gadget"),
                            || Ok(yy),
                        )
                        .unwrap();
                        //assert_eq!(yy_var.to_bits_le().unwrap().value().unwrap(), tmp.to_bits_le().unwrap().value().unwrap());
                        yy_var.enforce_equal(&shift_res).unwrap();
                    }
                }
            }
        }
        Ok(())
    }
}
