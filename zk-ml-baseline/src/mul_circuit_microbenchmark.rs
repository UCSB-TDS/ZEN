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
use std::cmp;
use std::ops::*;

//this class is only for microbenchmarking the number of constraints. correctness is not guaranteed.

#[derive(Debug, Clone)]
pub struct FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
    pub x: Vec<u8>,
    pub l1_mat: Vec<Vec<u8>>,
    pub y: Vec<u8>,

    //these two variables are used to restore the real y
    pub remainder: Vec<u32>,
    pub div: Vec<u32>,

    //zero points for quantization
    pub x_0: u8,
    pub l1_mat_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,

    //this class is for microbenchmark the constraint reduction among different batch size
    pub batch_size: usize,
}

impl ConstraintSynthesizer<Fq> for FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FC SIMD Microbenchmark for batch size {}", self.batch_size);

        let mut assembled_vec_x = vec![0u128; self.x.len()];

        if self.batch_size == 3 {
            println!("Use 3 vector SIMD in current FC layer");

            for i in 0..self.x.len() {
                assembled_vec_x[i] = (self.x[i] as u128)
                    + (self.x[i] as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                    + (self.x[i] as u128) * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);
            }
            let assembled_x_0: u128 = (self.x_0 as u128)
                + (self.x_0 as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);

            let assembled_l1_0: u128 = (self.l1_mat_0 as u128)
                + (self.l1_mat_0 as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);

            //the length of y is possible not to be divided by 3;
            let threes = (self.y.len() / 3) * 3;

            for i in (0..threes).step_by(3) {
                //every four vectors we encode them together to do SIMD vector dot product to save constraints.
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp
                let mut assembled_vec_l1 = vec![0u128; self.x.len()];
                for j in 0..self.x.len() {
                    assembled_vec_l1[j] = (self.l1_mat[i][j] as u128)
                        + (self.l1_mat[i + 1][j] as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                        + (self.l1_mat[i + 2][j] as u128)
                            * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);
                }

                let m1 = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(M_EXP)) as f32) as u64;
                //y_0 is a constant, the assmbled y_0 should also be a constant

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;

                let t1 = BigInteger256::from(y_0_1);
                let mut t2 = BigInteger256::from(y_0_2);
                let mut t3 = BigInteger256::from(y_0_3);

                //println!("before t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);
                t2.muln((16 + SIMD_3VEC_EXTRA_BITS) * 2);
                t3.muln((16 + SIMD_3VEC_EXTRA_BITS) * 6);
                //println!("after t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);

                let t1_fq: Fq = t1.into();
                let t2_fq: Fq = t2.into();
                let t3_fq: Fq = t3.into();

                let mut garbage_filler = BigInteger256::from(2u64.pow(14 + SIMD_3VEC_EXTRA_BITS));
                garbage_filler.muln(16 + SIMD_3VEC_EXTRA_BITS);
                let filler1: Fq = garbage_filler.clone().into();
                garbage_filler.muln((16 + SIMD_3VEC_EXTRA_BITS) * 2);
                let filler2: Fq = garbage_filler.clone().into();
                garbage_filler.muln(16 + SIMD_3VEC_EXTRA_BITS);
                let filler3: Fq = garbage_filler.clone().into();
                garbage_filler.muln(16 + SIMD_3VEC_EXTRA_BITS);
                let filler4: Fq = garbage_filler.clone().into();

                /*
                illustration of 3 vector SIMD output

                |  | E*F | filler4 | filler3 | filler2 | C*D | filler1 | A*B |

                */
                let assembled_y_0_fq: Fq =
                    t1_fq + t2_fq + t3_fq + filler1 + filler2 + filler3 + filler4;

                let assembled_y_0 = FpVar::Constant(assembled_y_0_fq);

                let tmp = scala_cs_helper_u8_simd(
                    cs.clone(),
                    &assembled_vec_x,
                    &assembled_vec_l1,
                    assembled_x_0,
                    assembled_l1_0,
                    assembled_y_0,
                );

                let tmp_bits = tmp.to_bits_le().unwrap();

                let mut simd_extract_1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget1"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget2"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_3 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget3"), || {
                        Ok(Fq::zero())
                    })
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
                    if i >= (255 - (16 + SIMD_3VEC_EXTRA_BITS) as usize) && i < 255 {
                        simd_extract_1 = simd_extract_1
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 3) as usize)
                        && i < (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 2) as usize)
                    {
                        simd_extract_2 = simd_extract_2
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 7) as usize)
                        && i < (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 6) as usize)
                    {
                        simd_extract_3 = simd_extract_3
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    }
                }

                let multiplier1: Fq = ((self.multiplier[i] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);
                let multiplier2: Fq =
                    ((self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);
                let multiplier3: Fq =
                    ((self.multiplier[i + 2] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var3 = FpVar::<Fq>::Constant(multiplier3);

                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                simd_extract_2 = multiplier_var2.clone() * simd_extract_2;
                simd_extract_3 = multiplier_var3.clone() * simd_extract_3;

                let yy1: Fq = (((self.y[i]) as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy2: Fq = (((self.y[i + 1]) as u64 + (self.div[i + 1] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 1] as u64)
                    .into();
                let yy3: Fq = (((self.y[i + 2]) as u64 + (self.div[i + 2] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 2] as u64)
                    .into();

                let yy_var1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget1"), || Ok(yy1))
                        .unwrap();
                let yy_var2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget2"), || Ok(yy2))
                        .unwrap();
                let yy_var3 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget3"), || Ok(yy3))
                        .unwrap();

                simd_extract_1.enforce_equal(&yy_var1).unwrap();
                simd_extract_2.enforce_equal(&yy_var2).unwrap();
                simd_extract_3.enforce_equal(&yy_var3).unwrap();
            }

            for i in threes..self.y.len() {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp

                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_remainder_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        y_0_converted,
                    );

                let yy: Fq = ((self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget"), || Ok(yy))
                        .unwrap();

                tmp.enforce_equal(&yy_var).unwrap();
            }
        } else if self.batch_size == 4 {
            println!("Use 4 vector SIMD in current FC layer");
            //we are safe to use 4 vector SIMD now
            for i in 0..self.x.len() {
                assembled_vec_x[i] = (self.x[i] as u128)
                    + (self.x[i] as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                    + (self.x[i] as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                    + (self.x[i] as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);
            }
            let assembled_x_0: u128 = (self.x_0 as u128)
                + (self.x_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);

            let assembled_l1_0: u128 = (self.l1_mat_0 as u128)
                + (self.l1_mat_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);
            //the length of y is possible not to be divided by 4;
            let fours = (self.y.len() / 4) * 4;

            for i in (0..fours).step_by(4) {
                //every four vectors we encode them together to do SIMD vector dot product to save constraints.

                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp
                let mut assembled_vec_l1 = vec![0u128; self.x.len()];
                for j in 0..self.x.len() {
                    assembled_vec_l1[j] = (self.l1_mat[i][j] as u128)
                        + (self.l1_mat[i + 1][j] as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                        + (self.l1_mat[i + 2][j] as u128)
                            * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                        + (self.l1_mat[i + 3][j] as u128)
                            * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);
                }
                let m1 = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(M_EXP)) as f32) as u64;
                let m4 = (self.multiplier[i + 3] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;
                let y_0_4: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m4;

                let t1 = BigInteger256::from(y_0_1);
                let mut t2 = BigInteger256::from(y_0_2);
                let mut t3 = BigInteger256::from(y_0_3);
                let mut t4 = BigInteger256::from(y_0_4);

                //because it is too large to be held in Rust u128, we use BigInteger256 provided by LibZEXE
                t2.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                t3.muln((16 + SIMD_4VEC_EXTRA_BITS) * 6);
                t4.muln((16 + SIMD_4VEC_EXTRA_BITS) * 8);

                let t1_fq: Fq = t1.into();
                let t2_fq: Fq = t2.into();
                let t3_fq: Fq = t3.into();
                let t4_fq: Fq = t4.into();

                let mut garbage_filler = BigInteger256::from(2u64.pow(14 + SIMD_4VEC_EXTRA_BITS));
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
                let assembled_y_0_fq: Fq =
                    t1_fq + t2_fq + t3_fq + t4_fq + filler1 + filler2 + filler3 + filler4 + filler5;

                let assembled_y_0 = FpVar::Constant(assembled_y_0_fq);

                let tmp = scala_cs_helper_u8_simd(
                    cs.clone(),
                    &assembled_vec_x,
                    &assembled_vec_l1,
                    assembled_x_0,
                    assembled_l1_0,
                    assembled_y_0,
                );

                let tmp_bits = tmp.to_bits_le().unwrap();

                let mut simd_extract_1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget1"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget2"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_3 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget3"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_4 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget4"), || {
                        Ok(Fq::zero())
                    })
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
                    if i >= (255 - (16 + SIMD_4VEC_EXTRA_BITS) as usize) && i < 255 {
                        simd_extract_1 = simd_extract_1
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 3) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 2) as usize)
                    {
                        simd_extract_2 = simd_extract_2
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 7) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 6) as usize)
                    {
                        simd_extract_3 = simd_extract_3
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 9) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 8) as usize)
                    {
                        simd_extract_4 = simd_extract_4
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    }
                }

                #[cfg(debug_assertion)]
                println!(
                    "number of constrants after extracting results from 4vec SIMD {}",
                    cs.num_constraints()
                );
                let multiplier1: Fq = m1.into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);

                let multiplier2: Fq = m2.into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);

                let multiplier3: Fq = m3.into();
                let multiplier_var3 = FpVar::<Fq>::Constant(multiplier3);

                let multiplier4: Fq = m4.into();
                let multiplier_var4 = FpVar::<Fq>::Constant(multiplier4);

                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                simd_extract_2 = multiplier_var2.clone() * (simd_extract_2);

                simd_extract_3 = multiplier_var3.clone() * (simd_extract_3);
                simd_extract_4 = multiplier_var4.clone() * simd_extract_4;

                let yy1: Fq = ((self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy2: Fq = ((self.y[i + 1] as u64 + (self.div[i + 1] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 1] as u64)
                    .into();
                let yy3: Fq = ((self.y[i + 2] as u64 + (self.div[i + 2] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 2] as u64)
                    .into();
                let yy4: Fq = ((self.y[i + 3] as u64 + (self.div[i + 3] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 3] as u64)
                    .into();

                let yy_var1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget1"), || Ok(yy1))
                        .unwrap();
                let yy_var2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget2"), || Ok(yy2))
                        .unwrap();
                let yy_var3 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget3"), || Ok(yy3))
                        .unwrap();
                let yy_var4 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget4"), || Ok(yy4))
                        .unwrap();

                simd_extract_1.enforce_equal(&yy_var1).unwrap();
                simd_extract_2.enforce_equal(&yy_var2).unwrap();
                simd_extract_3.enforce_equal(&yy_var3).unwrap();
                simd_extract_4.enforce_equal(&yy_var4).unwrap();
            }

            for i in fours..self.y.len() {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp
                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_remainder_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        y_0_converted,
                    );

                let yy: Fq = ((self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget"), || Ok(yy))
                        .unwrap();

                tmp.enforce_equal(&yy_var).unwrap();
            }
        } else if self.batch_size == 2 {
            #[cfg(debug_assertion)]
            println!("Use 2 vector SIMD in current FC layer");

            for i in 0..self.x.len() {
                assembled_vec_x[i] = (self.x[i] as u128)
                    + (self.x[i] as u128) * 2u128.pow(16 + SIMD_2VEC_EXTRA_BITS);
            }
            let assembled_x_0: u128 =
                (self.x_0 as u128) + (self.x_0 as u128) * 2u128.pow(16 + SIMD_2VEC_EXTRA_BITS);

            let assembled_l1_0: u128 = (self.l1_mat_0 as u128)
                + (self.l1_mat_0 as u128) * 2u128.pow(16 + SIMD_2VEC_EXTRA_BITS);
            //the length of y is possible not to be divided by 2;
            let twos = (self.y.len() / 2) * 2;

            for i in (0..twos).step_by(2) {
                //every four vectors we encode them together to do SIMD vector dot product to save constraints.
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp
                let mut assembled_vec_l1 = vec![0u128; self.x.len()];
                for j in 0..self.x.len() {
                    assembled_vec_l1[j] = (self.l1_mat[i][j] as u128)
                        + (self.l1_mat[i + 1][j] as u128) * 2u128.pow(16 + SIMD_2VEC_EXTRA_BITS);
                }

                let m1 = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;

                let t1 = BigInteger256::from(y_0_1);
                let mut t2 = BigInteger256::from(y_0_2);

                t2.muln((16 + SIMD_3VEC_EXTRA_BITS) * 2);

                let t1_fq: Fq = t1.into();
                let t2_fq: Fq = t2.into();

                //SIMD_2VEC_EXTRA_BITS + 14 > 64. So we minus 20 first and add 20 later. BigInteger256 does not support from u128.
                let mut garbage_filler =
                    BigInteger256::from(2u64.pow(14 + SIMD_2VEC_EXTRA_BITS - 20));
                garbage_filler.muln(16 + SIMD_2VEC_EXTRA_BITS + 20);
                let filler1: Fq = garbage_filler.clone().into();
                garbage_filler.muln((16 + SIMD_2VEC_EXTRA_BITS) * 2);
                let filler2: Fq = garbage_filler.clone().into();

                /*
                illustration of 2 vector SIMD output

                | | filler2 | C*D | filler1 | A*B |

                */
                let assembled_y_0_fq: Fq = t1_fq + t2_fq + filler1 + filler2;

                let assembled_y_0 = FpVar::Constant(assembled_y_0_fq);

                let tmp = scala_cs_helper_u8_simd(
                    cs.clone(),
                    &assembled_vec_x,
                    &assembled_vec_l1,
                    assembled_x_0,
                    assembled_l1_0,
                    assembled_y_0,
                );

                let tmp_bits = tmp.to_bits_le().unwrap();

                let mut simd_extract_1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget1"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget2"), || {
                        Ok(Fq::zero())
                    })
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
                    if i >= (255 - (16 + SIMD_3VEC_EXTRA_BITS) as usize) && i < 255 {
                        simd_extract_1 = simd_extract_1
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    } else if i >= (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 3) as usize)
                        && i < (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 2) as usize)
                    {
                        simd_extract_2 = simd_extract_2
                            .mul(&double_var)
                            .add(&bit.select(&one_var, &zero_var).unwrap());
                    }
                }

                let multiplier1: Fq = ((self.multiplier[i] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);
                let multiplier2: Fq =
                    ((self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u128).into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);

                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                simd_extract_2 = multiplier_var2.clone() * simd_extract_2;

                let yy1: Fq = (((self.y[i]) as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy2: Fq = (((self.y[i + 1]) as u64 + (self.div[i + 1] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i + 1] as u64)
                    .into();

                let yy_var1 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget1"), || Ok(yy1))
                        .unwrap();
                let yy_var2 =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget2"), || Ok(yy2))
                        .unwrap();

                simd_extract_1.enforce_equal(&yy_var1).unwrap();
                simd_extract_2.enforce_equal(&yy_var2).unwrap();
            }

            for i in twos..self.y.len() {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp

                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_remainder_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        y_0_converted,
                    );

                let yy: Fq = ((self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
                    * 2u64.pow(22)
                    + self.remainder[i] as u64)
                    .into();
                let yy_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget"), || Ok(yy))
                        .unwrap();

                tmp.enforce_equal(&yy_var).unwrap();
            }
        }

        Ok(())
    }
}

fn scala_cs_helper_u8_simd_partition(
    cs: ConstraintSystemRef<Fq>,
    input: &[u128],
    weight: &[u128],
    input_zeropoint: u128,
    weight_zeropoint: u128,
    start_index: usize,
    partition_len: usize,
    y_zeropoint_converted: FqVar,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }
    //println!("a {:?} \n b {:?}", a, b);
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    for i in start_index..cmp::min(input.len(), start_index + partition_len) {
        let aa: Fq = input[i].into();
        let bb: Fq = weight[i].into();
        let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
        //parameters of each layer in the model is fixed after training, so they are Constant wires in the circuit.
        let b_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(bb)).unwrap();
        let a0: Fq = input_zeropoint.into();
        let b0: Fq = weight_zeropoint.into();
        let a0_var = FpVar::Constant(a0);
        let b0_var = FpVar::Constant(b0);
        tmp1 += a_var.clone().mul(b_var.clone());
        tmp2 += a_var.mul(b0_var.clone());
        tmp3 += b_var.mul(a0_var.clone());
        tmp4 += a0_var.mul(b0_var);
    }

    // println!("y_assembled {:?}\n\n", y_zeropoint_converted.clone().to_bits_le().unwrap().value().unwrap());
    let res = (tmp1 + tmp4 + y_zeropoint_converted) - (tmp2 + tmp3);
    //println!(" true res {:?}\n\n", res.to_bits_le().unwrap().value().unwrap());
    res
}

// build constraint system for scalar multiplications
// each coefficient is assembled by multiple u8 derived from the idea of SIMD
fn scala_cs_helper_u8_simd(
    cs: ConstraintSystemRef<Fq>,
    a: &[u128],
    b: &[u128],
    a_zeropoint: u128,
    b_zeropoint: u128,
    y_zeropoint_converted: FqVar,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if a.len() != b.len() {
        panic!("scala mul: length not equal");
    }
    let partition_size = 5000;
    let mut res = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "vec dot product gadget"), || {
        Ok(Fq::zero())
    })
    .unwrap();
    let mut num_partitions = 0u32;
    for i in (0..a.len()).step_by(partition_size) {
        res += scala_cs_helper_u8_simd_partition(
            cs.clone(),
            a,
            b,
            a_zeropoint,
            b_zeropoint,
            i,
            partition_size,
            y_zeropoint_converted.clone(),
        );
        num_partitions += 1;
    }

    let num_partitions_fq: Fq = (num_partitions - 1).into();
    let num_partitions_var =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "num of partitions gadget"), || {
            Ok(num_partitions_fq)
        })
        .unwrap();

    //we have added y_zeropoint for N-1 more times just to ensure every SIMD partition vec dot poduct is positive. we need to minus them to ensure correctness.
    res -= y_zeropoint_converted * num_partitions_var;

    res
}

fn mul_cs_helper_u8(cs: ConstraintSystemRef<Fq>, a: u8, c: u8) -> FqVar {
    let aa: Fq = a.into();
    let cc: Fq = c.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let c_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "c gadget"), || Ok(cc)).unwrap();
    a_var.mul(c_var)
}

fn constant_mul_cs_helper_u8(cs: ConstraintSystemRef<Fq>, a: u8, constant: u8) -> FqVar {
    let aa: Fq = a.into();
    let cc: Fq = constant.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let c_var = FpVar::Constant(cc);
    a_var.mul(c_var)
}

fn constant_mul_constant_cs_helper_u8(_cs: ConstraintSystemRef<Fq>, c1: u8, c2: u8) -> FqVar {
    let aa: Fq = c1.into();
    let cc: Fq = c2.into();
    let a_var = FpVar::Constant(aa);
    let c_var = FpVar::Constant(cc);
    a_var.mul(c_var)
}
// build constraint system for scalar multiplications
// each coefficient is an u8
// return the circuit representation of (u32, sign); grow the CS accordingly
fn scala_cs_helper_u8(
    cs: ConstraintSystemRef<Fq>,
    input: &[u8],
    weight: &[u8],
    input_zeropoint: u8,
    weight_zeropoint: u8,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }
    //println!("a {:?} \n b {:?}", a, b);
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    for i in 0..input.len() {
        tmp1 += mul_cs_helper_u8(cs.clone(), input[i], weight[i]);
        tmp2 += constant_mul_cs_helper_u8(cs.clone(), input[i], weight_zeropoint);
        tmp3 += constant_mul_cs_helper_u8(cs.clone(), weight[i], input_zeropoint);
        tmp4 += constant_mul_constant_cs_helper_u8(cs.clone(), input_zeropoint, weight_zeropoint);
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = tmp1 + tmp4 - tmp2 - tmp3;
    #[cfg(debug_assertion)]
    println!(
        "number of constraints for scalar {}",
        cs.num_constraints() - _no_cs
    );

    res
}

fn scala_cs_helper_remainder_u8(
    cs: ConstraintSystemRef<Fq>,
    input: &[u8],
    weight: &[u8],
    input_zeropoint: u8,
    weight_zeropoint: u8,
    y_zeropoint_converted: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }
    //println!("a {:?} \n b {:?}", a, b);
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    //zero points of input, weight and y for quantization are all fixed after training, so they are Constant wires.
    let y_zeropoint_fq: Fq = y_zeropoint_converted.into();
    let y_zeropoint_var = FpVar::<Fq>::Constant(y_zeropoint_fq);

    //println!("multiplier : {}", (multiplier * (2u64.pow(22)) as f32) as u32);
    //println!("y_0 {}, y_converted : {}", y_zeropoint, (y_zeropoint as u64 * 2u64.pow(22)));

    for i in 0..input.len() {
        tmp1 += mul_cs_helper_u8(cs.clone(), input[i], weight[i]);
        tmp2 += constant_mul_cs_helper_u8(cs.clone(), input[i], weight_zeropoint);
        tmp3 += constant_mul_cs_helper_u8(cs.clone(), weight[i], input_zeropoint);
        tmp4 += constant_mul_constant_cs_helper_u8(cs.clone(), input_zeropoint, weight_zeropoint);
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = (tmp1 + tmp4 + y_zeropoint_var) - (tmp2 + tmp3);
    #[cfg(debug_assertion)]
    println!(
        "number of constrants for scalar {}",
        cs.num_constraints() - _no_cs
    );

    res
}
