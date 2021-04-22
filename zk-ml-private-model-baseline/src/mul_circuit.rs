use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::biginteger::*;
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
use std::cmp;
use std::ops::*;

fn scala_cs_helper_fq(
    cs: ConstraintSystemRef<Fq>,
    input: &[FqVar], //witness
    weight: &[u8],   //constant
    input_zeropoint: u8,
    weight_zeropoint: u8,
    y_zeropoint: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }

    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    //zero points of input, weight and y for quantization are all fixed after training, so they are Constant wires.
    let y0: Fq = y_zeropoint.into();
    let y0_const = FpVar::<Fq>::Constant(y0);
    let w0: Fq = weight_zeropoint.into();
    let input0: Fq = input_zeropoint.into();
    let w0_const = FpVar::Constant(w0);
    let input0_const = FpVar::Constant(input0);
    //println!("input0 {:?}\n\n\n", input[0].clone().to_bits_le().unwrap().value().unwrap());

    for i in 0..input.len() {
        let w: Fq = weight[i].into();
        let w_const = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "w tmp"), || Ok(w)).unwrap();
        tmp1 += input[i].clone().mul(w_const.clone());
        tmp2 += input[i].clone().mul(w0_const.clone());
        tmp3 += w_const.clone().mul(input0_const.clone());
        tmp4 += w0_const.clone().mul(input0_const.clone());
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}\n\n\n\n", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = (tmp1.clone() + tmp4.clone() + y0_const) - (tmp2 + tmp3);
    //println!("{:?}\n\n\n", (tmp1.clone() + tmp4.clone()).to_bits_le().unwrap().value().unwrap());
    res
}

#[derive(Debug, Clone)]
pub struct FCCircuitOp3 {
    //x and y are already encoded and mapped to FqVar for use.
    pub x: Vec<FqVar>,
    pub l1_mat: Vec<Vec<u8>>,
    pub y: Vec<FqVar>, //it is already restored.

    //these two variables are used to restore the real y. this happens outside the circuit
    pub remainder: Vec<u32>,
    pub div: Vec<u32>,

    //zero points for quantization
    pub x_0: u8,
    pub l1_mat_0: u8,
    pub y_0: Vec<u64>,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FCCircuitOp3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let two_power_8: Fq = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
        let m_exp_fq: Fq = (2u64.pow(M_EXP)).into();
        let m_exp_constant = FpVar::<Fq>::Constant(m_exp_fq);
        let zero_var = FpVar::<Fq>::Constant(Fq::zero());

        let mut assembled_vec_x = vec![zero_var.clone(); self.x.len()];
        //only implemented 3 and 4 vector SIMD processing
        if self.x.len() < 2u32.pow(SIMD_BOTTLENECK as u32) as usize {
            //we do not use simd because vector length is tooooo short, and we can not benefit from it.
            for i in 0..self.y.len() {
                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_fq(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                tmp.enforce_equal(&yy1_var).unwrap();
            }
        } else if self.x.len() > 2u32.pow(SIMD_4VEC_EXTRA_BITS) as usize {
            println!("Use 3 vector SIMD in current FC layer");
            //we are not safe to use 4 vector SIMD because the vector length is too large which may cause overflow.
            let bit_shift_1: Fq = (2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)).into();
            let bit_shift_2: Fq = (2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3)).into();
            let bit_shift_1_const = FpVar::Constant(bit_shift_1);
            let bit_shift_2_const = FpVar::Constant(bit_shift_2);
            //we are safe to use 4 vector SIMD now
            for i in 0..self.x.len() {
                assembled_vec_x[i] = self.x[i].clone()
                    + self.x[i].clone() * bit_shift_1_const.clone()
                    + self.x[i].clone() * bit_shift_2_const.clone();
            }
            let assembled_x_0: u128 = (self.x_0 as u128)
                + (self.x_0 as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);

            let assembled_l1_0: u128 = (self.l1_mat_0 as u128)
                + (self.l1_mat_0 as u128) * 2u128.pow(16 + SIMD_3VEC_EXTRA_BITS)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_3VEC_EXTRA_BITS) * 3);
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

                let t1 = BigInteger256::from(self.y_0[i]);
                let mut t2 = BigInteger256::from(self.y_0[i + 1]);
                let mut t3 = BigInteger256::from(self.y_0[i + 2]);

                //because it is too large to be held in Rust u128, we use BigInteger256 provided by LibZEXE
                t2.muln((16 + SIMD_3VEC_EXTRA_BITS) * 2);
                t3.muln((16 + SIMD_3VEC_EXTRA_BITS) * 6);

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

                let tmp = scala_cs_helper_u8_simd_fq(
                    cs.clone(),
                    assembled_vec_x.clone(),
                    assembled_vec_l1.clone(),
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
                let m1 = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(M_EXP)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(M_EXP)) as f32) as u64;

                let multiplier1: Fq = m1.into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);

                let multiplier2: Fq = m2.into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);

                let multiplier3: Fq = m3.into();
                let multiplier_var3 = FpVar::<Fq>::Constant(multiplier3);

                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;

                simd_extract_2 = multiplier_var2.clone() * simd_extract_2;

                simd_extract_3 = multiplier_var3.clone() * simd_extract_3;

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                let div2: Fq = (self.div[i + 1] as u64).into();
                let div2_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div2 gadget"), || Ok(div2))
                        .unwrap();
                let remainder2: Fq = (self.remainder[i + 1] as u64).into();
                let remainder2_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder2 gadget"), || {
                        Ok(remainder2)
                    })
                    .unwrap();
                let yy2_var = (self.y[i + 1].clone() + div2_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder2_var;

                let div3: Fq = (self.div[i + 2] as u64).into();
                let div3_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div3 gadget"), || Ok(div3))
                        .unwrap();
                let remainder3: Fq = (self.remainder[i + 2] as u64).into();
                let remainder3_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder3 gadget"), || {
                        Ok(remainder3)
                    })
                    .unwrap();
                let yy3_var = (self.y[i + 2].clone() + div3_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder3_var;

                //assert_eq!(yy1_var.to_bits_le().unwrap().value().unwrap(), simd_extract_1.to_bits_le().unwrap().value().unwrap());
                //assert_eq!(yy2_var.to_bits_le().unwrap().value().unwrap(), simd_extract_2.to_bits_le().unwrap().value().unwrap());
                //assert_eq!(yy3_var.to_bits_le().unwrap().value().unwrap(), simd_extract_3.to_bits_le().unwrap().value().unwrap());

                simd_extract_1.enforce_equal(&yy1_var).unwrap();
                simd_extract_2.enforce_equal(&yy2_var).unwrap();
                simd_extract_3.enforce_equal(&yy3_var).unwrap();
            }

            for i in threes..self.y.len() {
                //process the rest

                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_fq(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                tmp.enforce_equal(&yy1_var).unwrap();
            }
        } else {
            println!("Use 4 vector SIMD in current FC layer");
            let bit_shift_1: Fq = (2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)).into();
            let bit_shift_2: Fq = (2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)).into();
            let bit_shift_3: Fq = (2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4)).into();
            let bit_shift_1_const = FpVar::Constant(bit_shift_1);
            let bit_shift_2_const = FpVar::Constant(bit_shift_2);
            let bit_shift_3_const = FpVar::Constant(bit_shift_3);
            //we are safe to use 4 vector SIMD now
            for i in 0..self.x.len() {
                assembled_vec_x[i] = self.x[i].clone()
                    + self.x[i].clone() * bit_shift_1_const.clone()
                    + self.x[i].clone() * bit_shift_2_const.clone()
                    + self.x[i].clone() * bit_shift_3_const.clone();
            }
            let assembled_x_0: u128 = (self.x_0 as u128)
                + (self.x_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.x_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);

            let assembled_l1_0: u128 = (self.l1_mat_0 as u128)
                + (self.l1_mat_0 as u128) * 2u128.pow(16 + SIMD_4VEC_EXTRA_BITS)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 3)
                + (self.l1_mat_0 as u128) * 2u128.pow((16 + SIMD_4VEC_EXTRA_BITS) * 4);
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

                let t1 = BigInteger256::from(self.y_0[i]);
                let mut t2 = BigInteger256::from(self.y_0[i + 1]);
                let mut t3 = BigInteger256::from(self.y_0[i + 2]);
                let mut t4 = BigInteger256::from(self.y_0[i + 3]);

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

                let tmp = scala_cs_helper_u8_simd_fq(
                    cs.clone(),
                    assembled_vec_x.clone(),
                    assembled_vec_l1.clone(),
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

                //multipliers for quantization is fixed parameters after training the model.

                let multiplier1: Fq = m1.into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);

                let multiplier2: Fq = m2.into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);

                let multiplier3: Fq = m3.into();
                let multiplier_var3 = FpVar::<Fq>::Constant(multiplier3);

                let multiplier4: Fq = m4.into();
                let multiplier_var4 = FpVar::<Fq>::Constant(multiplier4);

                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                //println!("product1 {:?}\n\n", simd_extract_1.clone().to_bits_le().unwrap().value().unwrap());

                simd_extract_2 = multiplier_var2.clone() * simd_extract_2;
                //println!("product2 {:?}\n\n", simd_extract_2.clone().to_bits_le().unwrap().value().unwrap());

                simd_extract_3 = multiplier_var3.clone() * simd_extract_3;

                simd_extract_4 = multiplier_var4.clone() * simd_extract_4;

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                let div2: Fq = (self.div[i + 1] as u64).into();
                let div2_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div2 gadget"), || Ok(div2))
                        .unwrap();
                let remainder2: Fq = (self.remainder[i + 1] as u64).into();
                let remainder2_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder2 gadget"), || {
                        Ok(remainder2)
                    })
                    .unwrap();
                let yy2_var = (self.y[i + 1].clone() + div2_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder2_var;

                let div3: Fq = (self.div[i + 2] as u64).into();
                let div3_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div3 gadget"), || Ok(div3))
                        .unwrap();
                let remainder3: Fq = (self.remainder[i + 2] as u64).into();
                let remainder3_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder3 gadget"), || {
                        Ok(remainder3)
                    })
                    .unwrap();
                let yy3_var = (self.y[i + 2].clone() + div3_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder3_var;

                let div4: Fq = (self.div[i + 3] as u64).into();
                let div4_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div4 gadget"), || Ok(div4))
                        .unwrap();
                let remainder4: Fq = (self.remainder[i + 3] as u64).into();
                let remainder4_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder4 gadget"), || {
                        Ok(remainder4)
                    })
                    .unwrap();
                let yy4_var = (self.y[i + 3].clone() + div4_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder4_var;

                simd_extract_1.enforce_equal(&yy1_var).unwrap();
                simd_extract_2.enforce_equal(&yy2_var).unwrap();
                simd_extract_3.enforce_equal(&yy3_var).unwrap();
                simd_extract_4.enforce_equal(&yy4_var).unwrap();
            }

            for i in fours..self.y.len() {
                //process the rest

                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_fq(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                tmp.enforce_equal(&yy1_var).unwrap();
            }
        }

        Ok(())
    }
}

// statement:
//  y = x * l1_mat
#[derive(Debug, Clone)]
pub struct FCCircuit {
    pub x: X,
    pub l1_mat: L1Mat,
    pub y: Y,
}

impl ConstraintSynthesizer<Fq> for FCCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("is setup mode: {}", cs.is_in_setup_mode());

        // constants
        let t = Boolean::<Fq>::constant(true);
        let f = Boolean::<Fq>::constant(false);

        for i in 0..self.y.len() {
            // compute <x, l1[i]>, store the result in tmp and sign
            let (tmp, _sign) = scala_cs_helper_i8(cs.clone(), &self.x, &self.l1_mat[i]);

            // zz = |y[i]|; also checks the sign is correct
            let zz: Fq = (if self.y[i] < 0 {
                //TODO just for baseline calculation. sometimes AssignmentMissing error occurs during baseline testing, so i manually set the sign variable.
                let sign = Boolean::<Fq>::constant(false);
                sign.enforce_equal(&f).unwrap();
                -self.y[i]
            } else {
                let sign = Boolean::<Fq>::constant(true);
                sign.enforce_equal(&t).unwrap();
                self.y[i]
            } as u32)
                .into();

            let zz_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z gadget"), || Ok(zz)).unwrap();

            // zz == tmp
            tmp.enforce_equal(&zz_var).unwrap();
        }
        Ok(())
    }
}

// build constraint system for i8 multiplications
// we represent i8 as a combination of (u32, sign)
// and carry out the multiplication accordingly
// it returns the variable for (u32, sign); and mutates the constraint system
fn mul_cs_helper_i8(cs: ConstraintSystemRef<Fq>, a: i8, b: i8) -> (FqVar, Boolean<Fq>) {
    let sign = if (a >= 0 && b >= 0) || (a <= 0 && b <= 0) {
        Boolean::constant(true)
    } else {
        Boolean::constant(false)
    };

    let aa: Fq = (if a < 0 { -a } else { a } as u32).into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();

    let bb: Fq = (if b < 0 { -b } else { b } as u32).into();
    let b_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "b gadget"), || Ok(bb)).unwrap();

    (a_var.mul(&b_var), sign)
}

// build constraint system for scalar multiplications
// each coefficient is an i8
// return the circuit representation of (u32, sign); grow the CS accordingly
fn scala_cs_helper_i8(cs: ConstraintSystemRef<Fq>, a: &[i8], b: &[i8]) -> (FqVar, Boolean<Fq>) {
    let _no_cs = cs.num_constraints();
    if a.len() != b.len() {
        panic!("scala mul: length not equal");
    }
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

    for i in 0..a.len() {
        let (c, sign) = mul_cs_helper_i8(cs.clone(), a[i], b[i]);
        if sign.value().unwrap() {
            pos = pos.add(&c);
            tmp1 += (a[i] * b[i]) as i32;
        } else {
            neg = neg.add(&c);
            tmp2 += (a[i] * b[i]) as i32;
        }
    }
    tmp2 = -tmp2;
    //println!("pos {:?} neg {:?}", pos.value().unwrap(), neg.value().unwrap());
    // merge the pos and neg results
    let res = if tmp1 >= tmp2 {
        pos.enforce_cmp(&neg, Ordering::Greater, true).unwrap();
        (pos.sub(&neg), t)
    } else {
        neg.enforce_cmp(&pos, Ordering::Greater, false).unwrap();
        (neg.sub(pos), f)
    };

    #[cfg(debug_assertion)]
    println!(
        "number of constrants for scalar {}",
        cs.num_constraints() - _no_cs
    );

    res
}

#[derive(Debug, Clone)]
pub struct FCCircuitU8 {
    pub x: Vec<u8>,
    pub l1_mat: Vec<Vec<u8>>,
    pub y: Vec<u8>,

    //zero points for quantization
    pub x_0: u8,
    pub l1_mat_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FCCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FCCircuitU8 is setup mode: {}", cs.is_in_setup_mode());

        for i in 0..self.y.len() {
            // compute multiplier * <x, l1[i]>, store the result in tmp
            let multiplier: Fq = ((self.multiplier[i] * (2.pow(M_EXP)) as f32) as u32).into();
            let multiplier_var = FpVar::<Fq>::Constant(multiplier);
            let tmp = multiplier_var.clone()
                * scala_cs_helper_u8(
                    cs.clone(),
                    &self.x,
                    &self.l1_mat[i],
                    self.x_0,
                    self.l1_mat_0,
                );
            let mut tmp_bits = tmp.to_bits_le().unwrap();
            //println!("tmp value :{:?} \n bits before drain: {:?}\n\n", tmp.value().unwrap(), tmp_bits.value());

            //because the bits decomposed are little endian. we need to drop the first 24 bits to achieve right-shift.
            tmp_bits.drain(0..22);
            tmp_bits.drain(8..);

            //println!("bits after drain: {:?}\n\n", tmp_bits.value());

            let mut shift_res =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "shift result gadget"), || {
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
            for (_i, bit) in tmp_bits.iter().rev().enumerate() {
                //This is the correct way to pack bits back to FpVar
                shift_res = shift_res
                    .mul(&double_var)
                    .add(&bit.select(&one_var, &zero_var).unwrap());
            }

            let yy: Fq = (self.y[i] - self.y_0).into();
            let yy_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget"), || Ok(yy))
                    .unwrap();

            //println!("res:{:?}\nyy :{:?}\n", &shift_res.to_bits_le().unwrap().value().unwrap()[..16], &yy_var.to_bits_le().unwrap().value().unwrap()[..16]);

            shift_res.enforce_equal(&yy_var).unwrap();
        }

        Ok(())
    }
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

#[derive(Debug, Clone)]
pub struct FCCircuitU8BitDecomposeOptimized {
    pub x: Vec<u8>,
    pub l1_mat: Vec<Vec<u8>>,
    pub y: Vec<u8>,

    //these two variables are used to restore the real y
    pub remainder: Vec<u32>,
    pub div: Vec<u32>,

    // we need enforce quality between:
    // (y - y_0) as u32 * div * 2^24 as u32 + remainder = [\sum(x - x_0)(l1_mat - l1_mat_0)] as u32 * (multiplier * 2^24 as f32) as u32

    //zero points for quantization
    pub x_0: u8,
    pub l1_mat_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FCCircuitU8BitDecomposeOptimized {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FCCircuitU8 is setup mode: {}", cs.is_in_setup_mode());

        for i in 0..self.y.len() {
            // compute multiplier * <x, l1[i]>(dot product), store the result in tmp

            let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

            let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;
            //println!("y_0_converted {}", y_0_converted);

            //multipliers for quantization is fixed parameters after training the model.
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
            // let multiplier: Fq = ((self.multiplier[i] * (2.pow(M_EXP)) as f32) as u32).into();
            // let multiplier_var =
            //     FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "multiplier gadget"), || Ok(multiplier))
            //         .unwrap();
            // let tmp = tmp * multiplier_var.clone();

            let yy: Fq = (self.y[i] as u64).into();
            let yy_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "yy gadget"), || Ok(yy)).unwrap();
            let div: Fq = (self.div[i] as u64).into();
            let div_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || Ok(div)).unwrap();
            let remainder: Fq = (self.remainder[i] as u64).into();
            let remainder_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder))
                    .unwrap();
            let two_power_8: Fq = (2u64.pow(8)).into();
            let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
            let two_power_22: Fq = (2u64.pow(22)).into();
            let two_power_22_constant = FpVar::<Fq>::Constant(two_power_22);

            let output_var =
                (yy_var + div_var * two_power_8_constant) * two_power_22_constant + remainder_var;

            tmp.enforce_equal(&output_var).unwrap();

            // println!("yy {}", (self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
            //                     * 2u64.pow(22)
            //                     + self.remainder[i] as u64);

            //println!("left {} == right {}", u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(M_EXP) + self.remainder[i]);
            //println!("{} {}", (self.y[i] - self.y_0) as u32, self.remainder[i]);
            //assert_eq!(u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(M_EXP) + self.remainder[i]);
            //println!("left {:?}\nright{:?}\n\n\n\n", tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
            //assert_eq!(tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
        }

        Ok(())
    }
}

fn scala_cs_helper_u8_simd_fq(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<FqVar>,
    kernel: Vec<u128>,
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
        let w_fq: Fq = kernel[i].into();
        let w_const = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "w "), || Ok(w_fq)).unwrap();

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
