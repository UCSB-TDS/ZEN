use crate::*;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::bits::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{println, vec, vec::Vec};
use ark_ed_on_bls12_381::{constraints::FqVar, Fq, Fr};
use ark_ff::*;


#[derive(Debug, Clone)]
pub struct FCCircuitOp3 {
     //x and y are already encoded and mapped to FqVar for use.
     pub x: Vec<FqVar>,
     pub l1_mat: Vec<Vec<FqVar>>,
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

// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for FCCircuitOp3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertions)]
        println!("is setup mode?: {}", cs.is_in_setup_mode());
        
        let two_power_8: Fq = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
        let m_exp_fq: Fq = (2u64.pow(M_EXP)).into();
        let m_exp_constant = FpVar::<Fq>::Constant(m_exp_fq);
        let zero_fq : Fq = 0u64.into();
        let zero_var = FpVar::<Fq>::Constant(Fq::zero());


        let mut assembled_vec_x = vec![zero_var.clone(); self.x.len()];
        //only implemented 3 and 4 vector SIMD processing
        if self.x.len() < 2u32.pow(SIMD_BOTTLENECK as u32) as usize {
            //we do not use simd because vector length is tooooo short, and we can not benefit from it.
            for i in 0..self.y.len() {
                let m = (self.multiplier[i] * (2u32.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder1 gadget"), || {
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
                let mut assembled_vec_l1 = vec![zero_var.clone(); self.x.len()];
                for j in 0..self.x.len() {
                    assembled_vec_l1[j] = self.l1_mat[i][j].clone()
                        + self.l1_mat[i + 1][j].clone() * bit_shift_1_const.clone()
                        + self.l1_mat[i + 2][j].clone() * bit_shift_2_const.clone();
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

                let tmp = scala_cs_helper_u8_simd(
                    cs.clone(),
                    assembled_vec_x.clone(),
                    assembled_vec_l1.clone(),
                    assembled_x_0,
                    assembled_l1_0,
                    assembled_y_0,
                );

                let tmp_bits = tmp.to_bits_le().unwrap();

                let mut simd_extract_1 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget1"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_2 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget2"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_3 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget3"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();

                let a = 2u8;
                let b = 1u8;
                let double: Fq = a.into();
                let double_var = FpVar::Constant(double);
                let one: Fq = b.into();
                let one_var = FpVar::<Fq>::Constant(one);
                let zero_var = FpVar::<Fq>::Constant(zero_fq);
                for (i, bit) in tmp_bits.iter().rev().enumerate() {
                    //This is the correct way to pack bits back to FpVar
                    if i >= (255 - (16 + SIMD_3VEC_EXTRA_BITS) as usize) && i < 255 {
                        simd_extract_1 = simd_extract_1
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
                    } else if i >= (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 3) as usize)
                        && i < (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 2) as usize)
                    {
                        simd_extract_2 = simd_extract_2
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
                    } else if i >= (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 7) as usize)
                        && i < (255 - ((16 + SIMD_3VEC_EXTRA_BITS) * 6) as usize)
                    {
                        simd_extract_3 = simd_extract_3
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
                    }
                }
                let m1 = (self.multiplier[i] * (2u32.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2u32.pow(M_EXP)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2u32.pow(M_EXP)) as f32) as u64;

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
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                let div2: Fq = (self.div[i + 1] as u64).into();
                let div2_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div2 gadget"), || Ok(div2))
                        .unwrap();
                let remainder2: Fq = (self.remainder[i + 1] as u64).into();
                let remainder2_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder2 gadget"), || {
                        Ok(remainder2)
                    })
                    .unwrap();
                let yy2_var = (self.y[i + 1].clone() + div2_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder2_var;

                let div3: Fq = (self.div[i + 2] as u64).into();
                let div3_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div3 gadget"), || Ok(div3))
                        .unwrap();
                let remainder3: Fq = (self.remainder[i + 2] as u64).into();
                let remainder3_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder3 gadget"), || {
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

                let m = (self.multiplier[i] * (2u32.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder1 gadget"), || {
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
                let mut assembled_vec_l1 = vec![zero_var.clone(); self.x.len()];
                for j in 0..self.x.len() {
                    assembled_vec_l1[j] = self.l1_mat[i][j].clone()
                        + self.l1_mat[i + 1][j].clone() * bit_shift_1_const.clone()
                        + self.l1_mat[i + 2][j].clone() * bit_shift_2_const.clone()
                        + self.l1_mat[i + 3][j].clone() * bit_shift_3_const.clone();
                }
                let m1 = (self.multiplier[i] * (2u32.pow(M_EXP)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2u32.pow(M_EXP)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2u32.pow(M_EXP)) as f32) as u64;
                let m4 = (self.multiplier[i + 3] * (2u32.pow(M_EXP)) as f32) as u64;

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

                let tmp = scala_cs_helper_u8_simd(
                    cs.clone(),
                    assembled_vec_x.clone(),
                    assembled_vec_l1.clone(),
                    assembled_x_0,
                    assembled_l1_0,
                    assembled_y_0,
                );

                let tmp_bits = tmp.to_bits_le().unwrap();

                let mut simd_extract_1 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget1"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_2 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget2"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_3 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget3"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();
                let mut simd_extract_4 =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "shift result gadget4"), || {
                        Ok(Fq::zero())
                    })
                    .unwrap();

                let a = 2u8;
                let b = 1u8;
                let double: Fq = a.into();
                let double_var = FpVar::Constant(double);
                let one: Fq = b.into();
                let one_var = FpVar::<Fq>::Constant(one);
                let zero_var = FpVar::<Fq>::Constant(zero_fq);
                for (i, bit) in tmp_bits.iter().rev().enumerate() {
                    //This is the correct way to pack bits back to FpVar
                    if i >= (255 - (16 + SIMD_4VEC_EXTRA_BITS) as usize) && i < 255 {
                        simd_extract_1 = simd_extract_1
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 3) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 2) as usize)
                    {
                        simd_extract_2 = simd_extract_2
                            *double_var.clone() +
                            bit.select(&one_var, &zero_var).unwrap();
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 7) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 6) as usize)
                    {
                        simd_extract_3 = simd_extract_3
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
                    } else if i >= (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 9) as usize)
                        && i < (255 - ((16 + SIMD_4VEC_EXTRA_BITS) * 8) as usize)
                    {
                        simd_extract_4 = simd_extract_4
                            *double_var.clone()
                            + bit.select(&one_var, &zero_var).unwrap();
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
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder1 gadget"), || {
                        Ok(remainder1)
                    })
                    .unwrap();
                let yy1_var = (self.y[i].clone() + div1_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder1_var;

                let div2: Fq = (self.div[i + 1] as u64).into();
                let div2_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div2 gadget"), || Ok(div2))
                        .unwrap();
                let remainder2: Fq = (self.remainder[i + 1] as u64).into();
                let remainder2_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder2 gadget"), || {
                        Ok(remainder2)
                    })
                    .unwrap();
                let yy2_var = (self.y[i + 1].clone() + div2_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder2_var;

                let div3: Fq = (self.div[i + 2] as u64).into();
                let div3_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div3 gadget"), || Ok(div3))
                        .unwrap();
                let remainder3: Fq = (self.remainder[i + 2] as u64).into();
                let remainder3_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder3 gadget"), || {
                        Ok(remainder3)
                    })
                    .unwrap();
                let yy3_var = (self.y[i + 2].clone() + div3_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder3_var;

                let div4: Fq = (self.div[i + 3] as u64).into();
                let div4_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div4 gadget"), || Ok(div4))
                        .unwrap();
                let remainder4: Fq = (self.remainder[i + 3] as u64).into();
                let remainder4_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder4 gadget"), || {
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

                let m = (self.multiplier[i] * (2u32.pow(M_EXP)) as f32) as u64;

                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        self.y_0[i],
                    );

                let div1: Fq = (self.div[i] as u64).into();
                let div1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "div1 gadget"), || Ok(div1))
                        .unwrap();
                let remainder1: Fq = (self.remainder[i] as u64).into();
                let remainder1_var =
                    FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "remainder1 gadget"), || {
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





//when model is public and image is secret, model parameters should use new_input() instead of new_witness()
fn scala_cs_helper_u8_simd(
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
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    let kernel0: Fq = kernel_zeropoint.into();
    let input0: Fq = x_zeropoint.into();
    let kernel0_const = FpVar::Constant(kernel0);
    let input0_const = FpVar::Constant(input0);
    for i in 0..length {
        //x_zeropoint, kernel_zeropoints and y_zeropoints are all Constant wires because they are independent of input image
        let w_const = kernel[i].clone();
        //let w_fq: Fq = kernel[i].into();
        //let w_const = FpVar::new_input(ark_relations::ns!(cs, "w "), || Ok(w_fq)).unwrap();

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

//when model is public and image is secret, model parameters should use new_input() instead of new_witness()
fn scala_cs_helper_u8(
    cs: ConstraintSystemRef<Fq>,
    input: &[FqVar],  //witness
    weight: &[FqVar], //constant
    input_zeropoint: u8,
    weight_zeropoint: u8,
    y_zeropoint: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }
    
    let mut tmp1 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    //zero points of input, weight and y for quantization are all fixed after training, so they are Constant wires.
    let y0: Fq = y_zeropoint.into();
    let y0_const = FpVar::Constant(y0);
    let w0: Fq = weight_zeropoint.into();
    let input0: Fq = input_zeropoint.into();
    let w0_const = FpVar::Constant(w0);
    let input0_const = FpVar::Constant(input0);
    //println!("input0 {:?}\n\n\n", input[0].clone().to_bits_le().unwrap().value().unwrap());
    //let mut v = Vec::new();
    for i in 0..input.len() {
        let w_const = weight[i].clone();
        //let w_const = FpVar::new_input(ark_relations::ns!(cs, "w tmp"), || Ok(w)).unwrap();
        //v.push(input[i].clone() * w_const.clone());
        tmp1 += input[i].clone() * w_const.clone();
        tmp2 += input[i].clone() * w0_const.clone();
        tmp3 += w_const.clone() * input0_const.clone();
        tmp4 += w0_const.clone() * input0_const.clone();
    }
    //let tmp1 : FpVar<Fq>= v.iter().sum();
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}\n\n\n\n", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = (tmp1.clone() + tmp4.clone() + y0_const) - (tmp2 + tmp3);
    //println!("{:?}\n\n\n", (tmp1.clone() + tmp4.clone()).to_bits_le().unwrap().value().unwrap());
    res
}
