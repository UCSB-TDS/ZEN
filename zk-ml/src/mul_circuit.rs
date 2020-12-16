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
            let (tmp, _sign) = scala_cs_helper(cs.clone(), &self.x, &self.l1_mat[i]);

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
fn mul_cs_helper(cs: ConstraintSystemRef<Fq>, a: i8, b: i8) -> (FqVar, Boolean<Fq>) {
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
fn scala_cs_helper(cs: ConstraintSystemRef<Fq>, a: &[i8], b: &[i8]) -> (FqVar, Boolean<Fq>) {
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
        let (c, sign) = mul_cs_helper(cs.clone(), a[i], b[i]);
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
            let multiplier: Fq = ((self.multiplier[i] * (2.pow(22u32)) as f32) as u32).into();
            let multiplier_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "multiplier gadget"), || {
                    Ok(multiplier)
                })
                .unwrap();
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

            tmp_bits.drain(0..24);
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

            // (2^24 * multiplier * sum((x_i - x_0) * (weight_i - weight_0))) / 2^24 == (y[i] - y_0)
            //println!("res:{:?}\nyy :{:?}\n", &shift_res.to_bits_le().unwrap().value().unwrap()[..16], &yy_var.to_bits_le().unwrap().value().unwrap()[..16]);

            shift_res.enforce_equal(&yy_var).unwrap();
        }

        Ok(())
    }
}

// build constraint system for u8 multiplications
// we represent u8 as a combination of u8
// and carry out the multiplication accordingly
// it returns the variable for u8; and mutates the constraint system
fn mul_cs_helper_u8(cs: ConstraintSystemRef<Fq>, a: u8, b: u8) -> FqVar {
    let aa: Fq = a.into();
    let bb: Fq = b.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let b_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "b gadget"), || Ok(bb)).unwrap();
    a_var.mul(&b_var)
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
    a: &[u8],
    b: &[u8],
    a_zeropoint: u8,
    b_zeropoint: u8,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if a.len() != b.len() {
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

    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());

    for i in 0..a.len() {
        tmp1 += mul_cs_helper_u8(cs.clone(), a[i], b[i]);

        tmp2 += constant_mul_cs_helper_u8(cs.clone(), a[i], b_zeropoint);
        tmp3 += constant_mul_cs_helper_u8(cs.clone(), b[i], a_zeropoint);

        tmp4 += constant_mul_constant_cs_helper_u8(cs.clone(), a_zeropoint, b_zeropoint);
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = tmp1 + tmp4 - tmp2 - tmp3;
    #[cfg(debug_assertion)]
    println!(
        "number of constrants for scalar {}",
        cs.num_constraints() - _no_cs
    );

    res
}

fn scala_cs_helper_remainder_u8(
    cs: ConstraintSystemRef<Fq>,
    a: &[u8],
    b: &[u8],
    a_zeropoint: u8,
    b_zeropoint: u8,
    y_zeropoint_converted: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if a.len() != b.len() {
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

    let y_zeropoint_fq: Fq = y_zeropoint_converted.into();
    let y_zeropoint_var =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "y_0 gadget"), || Ok(y_zeropoint_fq)).unwrap();

    //println!("multiplier : {}", (multiplier * (2u64.pow(22)) as f32) as u32);
    //println!("y_0 {}, y_converted : {}", y_zeropoint, (y_zeropoint as u64 * 2u64.pow(22)));

    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());

    for i in 0..a.len() {
        tmp1 += mul_cs_helper_u8(cs.clone(), a[i], b[i]);

        tmp2 += constant_mul_cs_helper_u8(cs.clone(), a[i], b_zeropoint);
        tmp3 += constant_mul_cs_helper_u8(cs.clone(), b[i], a_zeropoint);

        tmp4 += constant_mul_constant_cs_helper_u8(cs.clone(), a_zeropoint, b_zeropoint);
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = (tmp1 + tmp4 + y_zeropoint_var) - (tmp2 + tmp3);
    //println!("counter 1/2/3/4 are {} {} {} {}", counter1, counter2, counter3, counter4);
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

            let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

            let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;
            //println!("y_0_converted {}", y_0_converted);
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
            // let multiplier: Fq = ((self.multiplier[i] * (2.pow(22u32)) as f32) as u32).into();
            // let multiplier_var =
            //     FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "multiplier gadget"), || Ok(multiplier))
            //         .unwrap();
            // let tmp = tmp * multiplier_var.clone();

            let yy: Fq = ((self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8))) * 2u64.pow(22)
                + self.remainder[i] as u64)
                .into();
            let yy_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "l1 output gadget"), || Ok(yy))
                    .unwrap();

            // println!("yy {}", (self.y[i] as u64 + (self.div[i] as u64 * 2u64.pow(8)))
            //                     * 2u64.pow(22)
            //                     + self.remainder[i] as u64);

            //println!("left {} == right {}", u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(22u32) + self.remainder[i]);
            //println!("{} {}", (self.y[i] - self.y_0) as u32, self.remainder[i]);
            //assert_eq!(u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(22u32) + self.remainder[i]);
            //println!("left {:?}\nright{:?}\n\n\n\n", tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
            //assert_eq!(tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
            tmp.enforce_equal(&yy_var).unwrap();
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FCCircuitU8BitDecomposeOptimizedSIMD {
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

impl ConstraintSynthesizer<Fq> for FCCircuitU8BitDecomposeOptimizedSIMD {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "FCCircuitU8BitDecomposeOptimizedSIMD is setup mode: {}",
            cs.is_in_setup_mode()
        );
        let mut assembled_vec_x = vec![0u128; self.x.len()];

        if self.x.len() > 2u32.pow(SIMD_4VEC_EXTRA_BITS) as usize {
            //we can not use 4 vector SIMD, use 3 vector SIMD instead

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

                let m1 = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u64;
                //y_0 is a constant, the assmbled y_0 should also be a constant

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;

                let  t1 = BigInteger256::from(y_0_1);
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

                // #[cfg(debug_assertion)]
                // println!(
                //     "number of constrants before bit decomposition {}",
                //     cs.num_constraints()
                // );

                let tmp_bits = tmp.to_bits_le().unwrap();

                // #[cfg(debug_assertion)]
                // println!(
                //     "number of constrants after bit decomposition {}",
                //     cs.num_constraints()
                // );

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

                // println!(
                //     "number of constrants before extracting results from 3vec SIMD {}",
                //     cs.num_constraints()
                // );
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
                //#[cfg(debug_assertion)]
                // println!(
                //     "number of constrants after extracting results from 3vec SIMD {}",
                //     cs.num_constraints()
                // );
                //println!("before multiplier {:?}\n", &simd_extract_2.to_bits_le().unwrap().value().unwrap()[..8]);
                let multiplier1: Fq = ((self.multiplier[i] * (2.pow(22u32)) as f32) as u128).into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);
                let multiplier2: Fq =
                    ((self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u128).into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);
                let multiplier3: Fq =
                    ((self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u128).into();
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

                // (2^24 * multiplier * sum((x_i - x_0) * (weight_i - weight_0))) == (y[i] - y_0) * 2^24 + remainder
                simd_extract_1.enforce_equal(&yy_var1).unwrap();
                simd_extract_2.enforce_equal(&yy_var2).unwrap();
                simd_extract_3.enforce_equal(&yy_var3).unwrap();

                // assert_eq!( simd_extract_1.to_bits_le().unwrap().value().unwrap()[0..30], yy_var1.to_bits_le().unwrap().value().unwrap()[0..30]);
                // assert_eq!( simd_extract_2.to_bits_le().unwrap().value().unwrap()[0..30], yy_var2.to_bits_le().unwrap().value().unwrap()[0..30]);
                //assert_eq!( simd_extract_3.to_bits_le().unwrap().value().unwrap()[0..30], yy_var3.to_bits_le().unwrap().value().unwrap()[0..30]);
            }

            for i in threes..self.y.len() {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp

                let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

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

                // (2^24 * multiplier * sum((x_i - x_0) * (weight_i - weight_0))) == (y[i] - y_0) * 2^24 + remainder
                tmp.enforce_equal(&yy_var).unwrap();
            }
        } else {
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
                let m1 = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u64;
                let m4 = (self.multiplier[i + 3] * (2.pow(22u32)) as f32) as u64;

                // println!("y_converted {} {} {} {}", (self.y_0 as u64 * 2u64.pow(22)) / m1, (self.y_0 as u64 * 2u64.pow(22)) / m2,
                //                                     (self.y_0 as u64 * 2u64.pow(22)) / m3, (self.y_0 as u64 * 2u64.pow(22)) / m4);

                //y_0 is a constant, the assmbled y_0 should also be a constant

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;
                let y_0_4: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m4;

                let  t1 = BigInteger256::from(y_0_1);
                let mut t2 = BigInteger256::from(y_0_2);
                let mut t3 = BigInteger256::from(y_0_3);
                let mut t4 = BigInteger256::from(y_0_4);

                //println!("before t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);
                //because it is too large to be held in Rust u128, we use BigInteger256 provided by LibZEXE
                t2.muln((16 + SIMD_4VEC_EXTRA_BITS) * 2);
                t3.muln((16 + SIMD_4VEC_EXTRA_BITS) * 6);
                t4.muln((16 + SIMD_4VEC_EXTRA_BITS) * 8);
                //println!("after t1/2/3/4 {:?} {:?} {:?} {:?}\n\n", t1, t2, t3, t4);

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

                #[cfg(debug_assertion)]
                println!(
                    "number of constrants before bit decomposition {}",
                    cs.num_constraints()
                );
                let tmp_bits = tmp.to_bits_le().unwrap();
                #[cfg(debug_assertion)]
                println!(
                    "number of constrants after bit decomposition {}",
                    cs.num_constraints()
                );

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
                //println!("tmp1 {:?}\n\n m1{:?}\n", simd_extract_1.clone().to_bits_le().unwrap().value().unwrap(), multiplier_var1.clone().to_bits_le().unwrap().value().unwrap());
                simd_extract_1 = multiplier_var1.clone() * simd_extract_1;
                //println!("product1 {:?}\n\n", simd_extract_1.clone().to_bits_le().unwrap().value().unwrap());

                //println!("tmp2 {:?}\n\n m2{:?}\n", simd_extract_2.clone().to_bits_le().unwrap().value().unwrap(), multiplier_var2.clone().to_bits_le().unwrap().value().unwrap());
                simd_extract_2 = multiplier_var2.clone() * (simd_extract_2);
                //println!("product2 {:?}\n\n", simd_extract_2.clone().to_bits_le().unwrap().value().unwrap());

                //println!("tmp3 {:?}\n\n m3{:?}\n", simd_extract_3.clone().to_bits_le().unwrap().value().unwrap(), multiplier_var3.clone().to_bits_le().unwrap().value().unwrap());

                simd_extract_3 = multiplier_var3.clone() * (simd_extract_3);

                //println!("tmp4 {:?}\n\n m4{:?}\n", simd_extract_4.clone().to_bits_le().unwrap().value().unwrap(), multiplier_var4.clone().to_bits_le().unwrap().value().unwrap());

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

                // println!("yy2 {}", (self.y[i+1] as u64 + (self.div[i+1] as u64 * 2u64.pow(8)))
                // * 2u64.pow(22)
                // + self.remainder[i+1] as u64);

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

                // (2^24 * multiplier * sum((x_i - x_0) * (weight_i - weight_0))) == (y[i] - y_0) * 2^24 + remainder

                simd_extract_1.enforce_equal(&yy_var1).unwrap();
                simd_extract_2.enforce_equal(&yy_var2).unwrap();
                simd_extract_3.enforce_equal(&yy_var3).unwrap();
                simd_extract_4.enforce_equal(&yy_var4).unwrap();

                // assert_eq!( simd_extract_1.to_bits_le().unwrap().value().unwrap()[0..30], yy_var1.to_bits_le().unwrap().value().unwrap()[0..30]);
                // assert_eq!( simd_extract_2.to_bits_le().unwrap().value().unwrap()[0..30], yy_var2.to_bits_le().unwrap().value().unwrap()[0..30]);
                // assert_eq!( simd_extract_3.to_bits_le().unwrap().value().unwrap()[0..30], yy_var3.to_bits_le().unwrap().value().unwrap()[0..30]);
                // assert_eq!( simd_extract_4.to_bits_le().unwrap().value().unwrap()[0..30], yy_var4.to_bits_le().unwrap().value().unwrap()[0..30]);
            }

            for i in fours..self.y.len() {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp
                let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

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

                // (2^24 * multiplier * sum((x_i - x_0) * (weight_i - weight_0))) == (y[i] - y_0) * 2^24 + remainder

                //println!("left {} == right {}", u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(22u32) + self.remainder[i]);
                //println!("{} {}", (self.y[i] - self.y_0) as u32, self.remainder[i]);
                //assert_eq!(u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(22u32) + self.remainder[i]);
                //println!("left {:?}\nright{:?}", tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
                tmp.enforce_equal(&yy_var).unwrap();
            }
        }

        Ok(())
    }
}



fn scala_cs_helper_u8_simd_partition(
    cs: ConstraintSystemRef<Fq>,
    a: &[u128],
    b: &[u128],
    a_zeropoint: u128,
    b_zeropoint: u128,
    start_index: usize,
    partition_len: usize,
    y_zeropoint_converted: FqVar,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if a.len() != b.len() {
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

    for i in start_index..cmp::min(a.len(), start_index + partition_len) {
        let aa: Fq = a[i].into();
        let bb: Fq = b[i].into();
        let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
        let b_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "b gadget"), || Ok(bb)).unwrap();
        let a0: Fq = a_zeropoint.into();
        let b0: Fq = b_zeropoint.into();
        let a0_var = FpVar::Constant(a0);
        let b0_var = FpVar::Constant(b0);
        tmp1 += a_var.clone().mul(b_var.clone());
        tmp2 += a_var.mul(b0_var.clone());
        tmp3 += b_var.mul(a0_var.clone());
        tmp4 += a0_var.mul(b0_var);
    }

    // println!("tmp1 {:?}\n\n", tmp1.clone().to_bits_le().unwrap().value().unwrap());
    // println!("tmp2 {:?}\n\n", tmp2.clone().to_bits_le().unwrap().value().unwrap());
    // println!("tmp3 {:?}\n\n", tmp3.clone().to_bits_le().unwrap().value().unwrap());
    // println!("tmp4 {:?}\n\n", tmp4.clone().to_bits_le().unwrap().value().unwrap());

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

                let m1 = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u64;
                //y_0 is a constant, the assmbled y_0 should also be a constant

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;

                let  t1 = BigInteger256::from(y_0_1);
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

                let multiplier1: Fq = ((self.multiplier[i] * (2.pow(22u32)) as f32) as u128).into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);
                let multiplier2: Fq =
                    ((self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u128).into();
                let multiplier_var2 = FpVar::<Fq>::Constant(multiplier2);
                let multiplier3: Fq =
                    ((self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u128).into();
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

                let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

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
                let m1 = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u64;
                let m3 = (self.multiplier[i + 2] * (2.pow(22u32)) as f32) as u64;
                let m4 = (self.multiplier[i + 3] * (2.pow(22u32)) as f32) as u64;

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;
                let y_0_3: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m3;
                let y_0_4: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m4;

                let  t1 = BigInteger256::from(y_0_1);
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
                let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

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

                let m1 = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;
                let m2 = (self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u64;

                let y_0_1: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m1;
                let y_0_2: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m2;

                let  t1 = BigInteger256::from(y_0_1);
                let mut t2 = BigInteger256::from(y_0_2);

                t2.muln((16 + SIMD_3VEC_EXTRA_BITS) * 2);

                let t1_fq: Fq = t1.into();
                let t2_fq: Fq = t2.into();

                let mut garbage_filler = BigInteger256::from(2u64.pow(14 + SIMD_2VEC_EXTRA_BITS));
                garbage_filler.muln(16 + SIMD_2VEC_EXTRA_BITS);
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

                let multiplier1: Fq = ((self.multiplier[i] * (2.pow(22u32)) as f32) as u128).into();
                let multiplier_var1 = FpVar::<Fq>::Constant(multiplier1);
                let multiplier2: Fq =
                    ((self.multiplier[i + 1] * (2.pow(22u32)) as f32) as u128).into();
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

                let m = (self.multiplier[i] * (2.pow(22u32)) as f32) as u64;

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
