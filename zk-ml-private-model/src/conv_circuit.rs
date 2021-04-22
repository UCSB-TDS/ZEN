use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::biginteger::*;
use algebra_core::Zero;
use num_traits::Pow;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::ToBitsGadget;

use std::ops::*;

#[derive(Debug, Clone)]
pub struct ConvCircuitOp3 {
    pub x: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height, Width]
    pub conv_kernel: Vec<Vec<Vec<Vec<FqVar>>>>, //[Num Kernel, Num Channel, kernel_size, kernel_size]
    pub y: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Kernel, Height - kernel_size + 1, Width - kernel_size + 1]

    //these two variables are used to restore the real y
    pub remainder: Vec<Vec<Vec<Vec<u32>>>>,
    pub div: Vec<Vec<Vec<Vec<u32>>>>,

    //zero points for quantization
    pub x_0: u8,
    pub conv_kernel_0: u8,
    pub y_0: Vec<u64>,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for ConvCircuitOp3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "ConvCircuitOp3OldSIMD is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let batch_size = self.x.len();
        let num_channels = self.conv_kernel[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();
        let num_kernels = self.conv_kernel.len();
        let kernel_size = self.conv_kernel[0][0].len();

        let two_power_8: Fq = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
        let m_exp_fq: Fq = (2u64.pow(M_EXP)).into();
        let m_exp_constant = FpVar::<Fq>::Constant(m_exp_fq);
        //println!("input size {} {} {} {}", batch_size, num_channels, input_height, input_width);
        //println!("kernel size {} {} {} {}", num_kernels, num_channels, kernel_size, kernel_size);
        if num_channels * kernel_size * kernel_size < SIMD_BOTTLENECK {
            //we do not use SIMD
            println!("we don't use old SIMD");
            for k in 0..num_kernels {
                //println!("k {} multiplier {}", k, ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128));
                let multiplier: Fq = ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier);
                for n in 0..batch_size {
                    for h in 0..(input_height - kernel_size + 1) {
                        for w in 0..(input_width - kernel_size + 1) {
                            let tmp = conv_kernel_helper_fq(
                                cs.clone(),
                                self.x[n].clone(),
                                self.conv_kernel[k].clone(),
                                h,
                                w,
                                self.x_0,
                                self.conv_kernel_0,
                                self.y_0[k],
                            );

                            let tmp = tmp * multiplier_var.clone();

                            let yy_var = self.y[n][k][h][w].clone();
                            let div: Fq = (self.div[n][k][h][w] as u64).into();
                            let div_var =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || {
                                    Ok(div)
                                })
                                .unwrap();
                            let remainder: Fq = (self.remainder[n][k][h][w] as u64).into();
                            let remainder_var = FpVar::<Fq>::new_witness(
                                r1cs_core::ns!(cs, "remainder gadget"),
                                || Ok(remainder),
                            )
                            .unwrap();
                            let output_var = (yy_var + div_var * two_power_8_constant.clone())
                                * m_exp_constant.clone()
                                + remainder_var;

                            output_var.enforce_equal(&tmp).unwrap();
                        }
                    }
                }
            }
        } else {
            //println!("x_0 {}, conv_kernel_0 {}", self.x_0, self.conv_kernel_0);
            let assembled_x_0: u128 = (self.x_0 as u128)
                + (self.x_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);

            let assembled_conv_kernel_0: u128 = (self.conv_kernel_0 as u128)
                + (self.conv_kernel_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.conv_kernel_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.conv_kernel_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);
            let zero_var = FpVar::<Fq>::Constant(Fq::zero());
            let mut assembled_x = vec![zero_var.clone(); num_channels * kernel_size * kernel_size];
            let mut assembled_conv_kernel =
                vec![zero_var.clone(); num_channels * kernel_size * kernel_size];

            //the length of y is possible not to be divided by 4;

            let fours = ((input_width - kernel_size + 1) / 4) * 4;
            //let fours = 0;
            let bit_shift_1: Fq = (2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)).into();
            let bit_shift_2: Fq = (2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)).into();
            let bit_shift_3: Fq = (2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4)).into();
            let bit_shift_1_const = FpVar::Constant(bit_shift_1);
            let bit_shift_2_const = FpVar::Constant(bit_shift_2);
            let bit_shift_3_const = FpVar::Constant(bit_shift_3);

            if fours >= 4 {
                //if fours is larger than 4, we can SIMD among multiple channels
                println!("OLD SIMD conv among multiple kernel positions on image");
                for k in 0..num_kernels {
                    //println!("k {} multiplier {}", k, ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128));
                    let multiplier: Fq =
                        ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128).into();
                    let multiplier_var = FpVar::<Fq>::Constant(multiplier);
                    for n in 0..batch_size {
                        for h in 0..(input_height - kernel_size + 1) {
                            for w in (0..fours).step_by(4) {
                                let mut counter = 0;
                                //assemble the vectors
                                //println!("n{} k{} h{} w{}", n, k, h, w);
                                for c in 0..num_channels {
                                    for hh in h..(h + kernel_size) {
                                        for ww in w..(w + kernel_size) {
                                            assembled_x[counter] = self.x[n][c][hh][ww].clone()
                                                + self.x[n][c][hh][ww + 1].clone()
                                                    * bit_shift_1_const.clone()
                                                + self.x[n][c][hh][ww + 2].clone()
                                                    * bit_shift_2_const.clone()
                                                + self.x[n][c][hh][ww + 3].clone()
                                                    * bit_shift_3_const.clone();
                                            assembled_conv_kernel[counter] = self.conv_kernel[k][c]
                                                [hh - h][ww - w]
                                                .clone()
                                                + self.conv_kernel[k][c][hh - h][ww - w].clone()
                                                    * bit_shift_1_const.clone()
                                                + self.conv_kernel[k][c][hh - h][ww - w].clone()
                                                    * bit_shift_2_const.clone()
                                                + self.conv_kernel[k][c][hh - h][ww - w].clone()
                                                    * bit_shift_3_const.clone();

                                            counter += 1;
                                        }
                                    }
                                }

                                //because it is too large to be held in Rust u128, we use BigInteger256 provided by LibZEXE
                                let t1 = BigInteger256::from(self.y_0[k]);
                                let mut t2 = BigInteger256::from(self.y_0[k]);
                                let mut t3 = BigInteger256::from(self.y_0[k]);
                                let mut t4 = BigInteger256::from(self.y_0[k]);

                                //println!("before t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);
                                t2.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                t3.muln((16 + SIMD_4VEC_EXTRA_BITS) * 6);
                                t4.muln((16 + SIMD_4VEC_EXTRA_BITS) * 8);
                                //println!("after t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);

                                let t1_fq: Fq = t1.into();
                                let t2_fq: Fq = t2.into();
                                let t3_fq: Fq = t3.into();
                                let t4_fq: Fq = t4.into();

                                let mut garbage_filler =
                                    BigInteger256::from(2u64.pow(14 + SIMD_4VEC_EXTRA_BITS));
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler1: Fq = garbage_filler.clone().into();
                                garbage_filler.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                let filler2: Fq = garbage_filler.clone().into();
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler3: Fq = garbage_filler.clone().into();
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler4: Fq = garbage_filler.clone().into();
                                garbage_filler.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                let filler5: Fq = garbage_filler.clone().into();

                                /*
                                illustration of 4 vector SIMD output
                                | | G*H  | filler5  | E*F | filler4 | filler3 | filler2 | C*D | filler1 | A*B |
                                */

                                let assembled_y_0_fq: Fq = t1_fq
                                    + t2_fq
                                    + t3_fq
                                    + t4_fq
                                    + filler1
                                    + filler2
                                    + filler3
                                    + filler4
                                    + filler5;

                                let assembled_y_0 = FpVar::Constant(assembled_y_0_fq);
                                //println!("assembled_y_0: {:?}", assembled_y_0.to_bits_le().unwrap().value().unwrap());
                                let tmp = conv_kernel_helper_u8_simd(
                                    cs.clone(),
                                    assembled_x.clone(),
                                    assembled_conv_kernel.clone(),
                                    assembled_x_0,
                                    assembled_conv_kernel_0,
                                    assembled_y_0,
                                );
                                //println!("tmp {:?}", tmp.to_bits_le().unwrap().value().unwrap());

                                let tmp_bits = tmp.to_bits_le().unwrap();

                                let mut simd_extract_1 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget1"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_2 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget2"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_3 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget3"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_4 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget4"),
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
                                for (i, bit) in tmp_bits.iter().rev().enumerate() {
                                    //This is the correct way to pack bits back to FpVar
                                    //only 255 bits

                                    if i >= (255 - (16 + SIMD_4VEC_EXTRA_BITS) as usize) && i < 255
                                    {
                                        simd_extract_1 = simd_extract_1
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 3) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 2) as usize)
                                    {
                                        simd_extract_2 = simd_extract_2
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 7) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 6) as usize)
                                    {
                                        simd_extract_3 = simd_extract_3
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 9) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 8) as usize)
                                    {
                                        simd_extract_4 = simd_extract_4
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    }
                                }
                                //println!("SIMD extract 2 : {:?}", simd_extract_2.to_bits_le().unwrap().value().unwrap());
                                simd_extract_1 = multiplier_var.clone() * simd_extract_1;
                                simd_extract_2 = multiplier_var.clone() * simd_extract_2;
                                simd_extract_3 = multiplier_var.clone() * simd_extract_3;
                                simd_extract_4 = multiplier_var.clone() * simd_extract_4;

                                //println!("conv layer left {} == right {}", u32_res_debugging ,(self.y[n][k][h][w] - self.y_0) as u32 * 2u32.pow(24u32) + self.remainder[n][k][h][w]);
                                let div1: Fq = (self.div[n][k][h][w] as u64).into();
                                let div1_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div1 gadget"),
                                    || Ok(div1),
                                )
                                .unwrap();
                                let remainder1: Fq = (self.remainder[n][k][h][w] as u64).into();
                                let remainder1_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder1 gadget"),
                                    || Ok(remainder1),
                                )
                                .unwrap();
                                let yy1_var = (self.y[n][k][h][w].clone()
                                    + div1_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder1_var;

                                let div2: Fq = (self.div[n][k][h][w + 1] as u64).into();
                                let div2_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div2 gadget"),
                                    || Ok(div2),
                                )
                                .unwrap();
                                let remainder2: Fq = (self.remainder[n][k][h][w + 1] as u64).into();
                                let remainder2_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder2 gadget"),
                                    || Ok(remainder2),
                                )
                                .unwrap();
                                let yy2_var = (self.y[n][k][h][w + 1].clone()
                                    + div2_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder2_var;

                                let div3: Fq = (self.div[n][k][h][w + 2] as u64).into();
                                let div3_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div3 gadget"),
                                    || Ok(div3),
                                )
                                .unwrap();
                                let remainder3: Fq = (self.remainder[n][k][h][w + 2] as u64).into();
                                let remainder3_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder3 gadget"),
                                    || Ok(remainder3),
                                )
                                .unwrap();
                                let yy3_var = (self.y[n][k][h][w + 2].clone()
                                    + div3_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder3_var;

                                let div4: Fq = (self.div[n][k][h][w + 3] as u64).into();
                                let div4_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div4 gadget"),
                                    || Ok(div4),
                                )
                                .unwrap();
                                let remainder4: Fq = (self.remainder[n][k][h][w + 3] as u64).into();
                                let remainder4_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder4 gadget"),
                                    || Ok(remainder4),
                                )
                                .unwrap();
                                let yy4_var = (self.y[n][k][h][w + 3].clone()
                                    + div4_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder4_var;

                                yy1_var.enforce_equal(&simd_extract_1).unwrap();
                                yy2_var.enforce_equal(&simd_extract_2).unwrap();
                                yy3_var.enforce_equal(&simd_extract_3).unwrap();
                                yy4_var.enforce_equal(&simd_extract_4).unwrap();
                            }

                            for w in fours..(input_width - kernel_size + 1) {
                                let tmp = conv_kernel_helper_fq(
                                    cs.clone(),
                                    self.x[n].clone(),
                                    self.conv_kernel[k].clone(),
                                    h,
                                    w,
                                    self.x_0,
                                    self.conv_kernel_0,
                                    self.y_0[k],
                                );

                                let tmp = tmp * multiplier_var.clone();

                                let yy_var = self.y[n][k][h][w].clone();
                                let div: Fq = (self.div[n][k][h][w] as u64).into();
                                let div_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div gadget"),
                                    || Ok(div),
                                )
                                .unwrap();
                                let remainder: Fq = (self.remainder[n][k][h][w] as u64).into();
                                let remainder_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder gadget"),
                                    || Ok(remainder),
                                )
                                .unwrap();
                                let output_var = (yy_var + div_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder_var;

                                output_var.enforce_equal(&tmp).unwrap();
                            }
                        }
                    }
                }
            } else {
                println!(
                    "OLD SIMD conv among multiple kernel channels because now input is too small."
                );
                //if fours is 0, then we need to SIMD among multiple kernels.
                let k_fours = (num_kernels / 4) * 4;
                for n in 0..batch_size {
                    for h in 0..(input_height - kernel_size + 1) {
                        for w in 0..(input_width - kernel_size + 1) {
                            for k in (0..k_fours).step_by(4) {
                                let multiplier1: Fq =
                                    ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128).into();
                                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);

                                let multiplier2: Fq =
                                    ((self.multiplier[k + 1] * (2.pow(M_EXP)) as f32) as u128)
                                        .into();
                                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);

                                let multiplier3: Fq =
                                    ((self.multiplier[k + 2] * (2.pow(M_EXP)) as f32) as u128)
                                        .into();
                                let multiplier_var3 = FpVar::<Fq>::Constant(multiplier3);

                                let multiplier4: Fq =
                                    ((self.multiplier[k + 3] * (2.pow(M_EXP)) as f32) as u128)
                                        .into();
                                let multiplier_var4 = FpVar::<Fq>::Constant(multiplier4);

                                let mut counter = 0;
                                //assemble the vectors
                                for c in 0..num_channels {
                                    for hh in h..(h + kernel_size) {
                                        for ww in w..(w + kernel_size) {
                                            assembled_x[counter] = self.x[n][c][hh][ww].clone()
                                                + self.x[n][c][hh][ww].clone()
                                                    * bit_shift_1_const.clone()
                                                + self.x[n][c][hh][ww].clone()
                                                    * bit_shift_2_const.clone()
                                                + self.x[n][c][hh][ww].clone()
                                                    * bit_shift_3_const.clone();
                                            //we are SIMD 4 conv kernel vectors
                                            assembled_conv_kernel[counter] =
                                                self.conv_kernel[k][c][hh - h][ww - w].clone()
                                                    + self.conv_kernel[k + 1][c][hh - h][ww - w]
                                                        .clone()
                                                        * bit_shift_1_const.clone()
                                                    + self.conv_kernel[k + 2][c][hh - h][ww - w]
                                                        .clone()
                                                        * bit_shift_2_const.clone()
                                                    + self.conv_kernel[k + 3][c][hh - h][ww - w]
                                                        .clone()
                                                        * bit_shift_3_const.clone();
                                            counter += 1;
                                        }
                                    }
                                }

                                let t1 = BigInteger256::from(self.y_0[k]);
                                let mut t2 = BigInteger256::from(self.y_0[k + 1]);
                                let mut t3 = BigInteger256::from(self.y_0[k + 2]);
                                let mut t4 = BigInteger256::from(self.y_0[k + 3]);

                                //println!("before t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);
                                t2.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                t3.muln((16 + SIMD_4VEC_EXTRA_BITS) * 6);
                                t4.muln((16 + SIMD_4VEC_EXTRA_BITS) * 8);
                                //println!("after t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);

                                let t1_fq: Fq = t1.into();
                                let t2_fq: Fq = t2.into();
                                let t3_fq: Fq = t3.into();
                                let t4_fq: Fq = t4.into();

                                let mut garbage_filler =
                                    BigInteger256::from(2u64.pow(14 + SIMD_4VEC_EXTRA_BITS));
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler1: Fq = garbage_filler.clone().into();
                                garbage_filler.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                let filler2: Fq = garbage_filler.clone().into();
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler3: Fq = garbage_filler.clone().into();
                                garbage_filler.muln(16 + SIMD_4VEC_EXTRA_BITS);
                                let filler4: Fq = garbage_filler.clone().into();
                                garbage_filler.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                                let filler5: Fq = garbage_filler.clone().into();

                                /*
                                illustration of 4 vector SIMD output

                                | | G*H  | filler5  | E*F | filler4 | filler3 | filler2 | C*D | filler1 | A*B |

                                */
                                let assembled_y_0_fq: Fq = t1_fq
                                    + t2_fq
                                    + t3_fq
                                    + t4_fq
                                    + filler1
                                    + filler2
                                    + filler3
                                    + filler4
                                    + filler5;

                                let assembled_y_0 = FpVar::Constant(assembled_y_0_fq);

                                let tmp = conv_kernel_helper_u8_simd(
                                    cs.clone(),
                                    assembled_x.clone(),
                                    assembled_conv_kernel.clone(),
                                    assembled_x_0,
                                    assembled_conv_kernel_0,
                                    assembled_y_0,
                                );

                                let tmp_bits = tmp.to_bits_le().unwrap();

                                let mut simd_extract_1 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget1"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_2 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget2"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_3 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget3"),
                                    || Ok(Fq::zero()),
                                )
                                .unwrap();
                                let mut simd_extract_4 = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "shift result gadget4"),
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
                                for (i, bit) in tmp_bits.iter().rev().enumerate() {
                                    //This is the correct way to pack bits back to FpVar
                                    //only 255 bits
                                    if i >= (255 - (16 + SIMD_4VEC_EXTRA_BITS) as usize) && i < 255
                                    {
                                        simd_extract_1 = simd_extract_1
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 3) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 2) as usize)
                                    {
                                        simd_extract_2 = simd_extract_2
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 7) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 6) as usize)
                                    {
                                        simd_extract_3 = simd_extract_3
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    } else if i
                                        >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 9) as usize)
                                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 8) as usize)
                                    {
                                        simd_extract_4 = simd_extract_4
                                            .mul(&double_var)
                                            .add(&bit.select(&one_var, &zero_var).unwrap());
                                    }
                                }

                                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                                simd_extract_2 = multiplier_var2.clone() * simd_extract_2;
                                simd_extract_3 = multiplier_var3.clone() * simd_extract_3;
                                simd_extract_4 = multiplier_var4.clone() * simd_extract_4;

                                let div1: Fq = (self.div[n][k][h][w] as u64).into();
                                let div1_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div1 gadget"),
                                    || Ok(div1),
                                )
                                .unwrap();
                                let remainder1: Fq = (self.remainder[n][k][h][w] as u64).into();
                                let remainder1_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder1 gadget"),
                                    || Ok(remainder1),
                                )
                                .unwrap();
                                let yy1_var = (self.y[n][k][h][w].clone()
                                    + div1_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder1_var;

                                let div2: Fq = (self.div[n][k + 1][h][w] as u64).into();
                                let div2_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div2 gadget"),
                                    || Ok(div2),
                                )
                                .unwrap();
                                let remainder2: Fq = (self.remainder[n][k + 1][h][w] as u64).into();
                                let remainder2_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder2 gadget"),
                                    || Ok(remainder2),
                                )
                                .unwrap();
                                let yy2_var = (self.y[n][k + 1][h][w].clone()
                                    + div2_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder2_var;

                                let div3: Fq = (self.div[n][k + 2][h][w] as u64).into();
                                let div3_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div3 gadget"),
                                    || Ok(div3),
                                )
                                .unwrap();
                                let remainder3: Fq = (self.remainder[n][k + 2][h][w] as u64).into();
                                let remainder3_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder3 gadget"),
                                    || Ok(remainder3),
                                )
                                .unwrap();
                                let yy3_var = (self.y[n][k + 2][h][w].clone()
                                    + div3_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder3_var;

                                let div4: Fq = (self.div[n][k + 3][h][w] as u64).into();
                                let div4_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div4 gadget"),
                                    || Ok(div4),
                                )
                                .unwrap();
                                let remainder4: Fq = (self.remainder[n][k + 3][h][w] as u64).into();
                                let remainder4_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder4 gadget"),
                                    || Ok(remainder4),
                                )
                                .unwrap();
                                let yy4_var = (self.y[n][k + 3][h][w].clone()
                                    + div4_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder4_var;

                                //println!("left {:?}\n\nright{:?}\n\n\n\n", simd_extract_2.to_bits_le().unwrap().value().unwrap(), yy_var2.to_bits_le().unwrap().value().unwrap());
                                //assert_eq!(yy_var2.to_bits_le().unwrap().value().unwrap(), simd_extract_2.to_bits_le().unwrap().value().unwrap());

                                yy1_var.enforce_equal(&simd_extract_1).unwrap();
                                yy2_var.enforce_equal(&simd_extract_2).unwrap();
                                yy3_var.enforce_equal(&simd_extract_3).unwrap();
                                yy4_var.enforce_equal(&simd_extract_4).unwrap();
                            }

                            for k in k_fours..num_kernels {
                                //deal with the rest which can not be batched for SIMD processing.

                                let multiplier: Fq =
                                    ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128).into();
                                let multiplier_var = FpVar::<Fq>::Constant(multiplier);

                                let tmp = conv_kernel_helper_fq(
                                    cs.clone(),
                                    self.x[n].clone(),
                                    self.conv_kernel[k].clone(),
                                    h,
                                    w,
                                    self.x_0,
                                    self.conv_kernel_0,
                                    self.y_0[k],
                                );

                                let tmp = tmp * multiplier_var.clone();

                                let yy_var = self.y[n][k][h][w].clone();
                                let div: Fq = (self.div[n][k][h][w] as u64).into();
                                let div_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div gadget"),
                                    || Ok(div),
                                )
                                .unwrap();
                                let remainder: Fq = (self.remainder[n][k][h][w] as u64).into();
                                let remainder_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder gadget"),
                                    || Ok(remainder),
                                )
                                .unwrap();
                                let output_var = (yy_var + div_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder_var;

                                output_var.enforce_equal(&tmp).unwrap();
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

fn conv_kernel_helper_u8_simd(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<FqVar>,
    kernel: Vec<FqVar>,
    x_zeropoint: u128,
    kernel_zeropoint: u128,

    y_zeropoint_converted: FqVar,
) -> FqVar {
    let _no_cs = cs.num_constraints();

    let length = kernel.len();
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    let kernel0: Fq = kernel_zeropoint.into();
    let input0: Fq = x_zeropoint.into();
    let kernel0_const = FpVar::Constant(kernel0);
    let input0_const = FpVar::Constant(input0);
    for i in 0..length {
        //x_zeropoint, kernel_zeropoints and y_zeropoints are all Constant wires because they are independent of input image
        //let w_fq: Fq = kernel[i].into();
        //let w_const = FpVar::Constant(w_fq);
        //let w_const = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "w "), || Ok(w_fq)).unwrap();
        let w_const = kernel[i].clone();
        tmp1 += x[i].clone() * w_const.clone();

        tmp2 += x[i].clone() * kernel0_const.clone();
        tmp3 += w_const.clone() * input0_const.clone();

        tmp4 += input0_const.clone() * kernel0_const.clone();
    }
    // println!("tmp1 + tmp4 {:?}", (tmp1.clone() + tmp4.clone()).to_bits_le().unwrap().value().unwrap());
    // println!("y_zero {:?}", y_zeropoint_converted.to_bits_le().unwrap().value().unwrap());
    // println!("tmp2 + tmp3 {:?}", (tmp2.clone() + tmp3.clone()).to_bits_le().unwrap().value().unwrap());
    let res = (tmp1 + tmp4 + y_zeropoint_converted) - (tmp2 + tmp3);

    res
}

fn conv_kernel_helper_fq(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<Vec<FqVar>>>,
    kernel: Vec<Vec<Vec<FqVar>>>,
    h_index: usize,
    w_index: usize,

    x_zeropoint: u8,
    kernel_zeropoint: u8,

    y_0_converted: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();

    let num_channels = kernel.len();
    let kernel_size = kernel[0].len();
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    let y_zeropoint_fq: Fq = y_0_converted.into();
    let y_zeropoint_var = FpVar::<Fq>::Constant(y_zeropoint_fq);

    let kernel0: Fq = kernel_zeropoint.into();
    let input0: Fq = x_zeropoint.into();
    let kernel0_const = FpVar::Constant(kernel0);
    let input0_const = FpVar::Constant(input0);
    for i in 0..num_channels {
        //iterate through all channels
        for j in h_index..(h_index + kernel_size) {
            for k in w_index..(w_index + kernel_size) {
                //let w: Fq = kernel[i][j - h_index][k - w_index].into();
                //let w_const = FpVar::Constant(w);
                //x_zeropoint, kernel_zeropoints and y_zeropoints are all Constant wires because they are independent of input image
                //let w_const = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "w tmp"), || Ok(w)).unwrap();
                let w_const = kernel[i][j - h_index][k - w_index].clone();
                tmp1 += x[i][j][k].clone().mul(w_const.clone());
                tmp2 += x[i][j][k].clone().mul(kernel0_const.clone());
                tmp3 += w_const.clone().mul(input0_const.clone());

                tmp4 += input0_const.clone().mul(kernel0_const.clone());
            }
        }
    }

    let res = (tmp1 + tmp4 + y_zeropoint_var) - (tmp2 + tmp3);

    res
}
