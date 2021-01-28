use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::biginteger::*;
use algebra_core::Zero;
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




//The circuits below are not in use.



#[derive(Debug, Clone)]
pub struct ConvCircuitOp3Depreciated {
    pub x: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height, Width]
    pub conv_kernel: Vec<Vec<Vec<Vec<u8>>>>, //[Num Kernel, Num Channel, kernel_size, kernel_size]
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

impl ConstraintSynthesizer<Fq> for ConvCircuitOp3Depreciated {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "ConvCircuitU8BitDecomposeOptimizationSIMD is setup mode: {}",
            cs.is_in_setup_mode()
        );
        let delta_bits_length = fast_math::log2(
            (self.conv_kernel[0].len()
                * self.conv_kernel[0][0].len()
                * self.conv_kernel[0][0][0].len()) as f32,
        ) as u32
            + 16u32
            + M_EXP;

        let num_images = self.x.len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        let num_kernels = self.conv_kernel.len();
        let kernel_size = self.conv_kernel[0][0].len();

        let delta_fq = (2u128.pow(delta_bits_length)).into();
        let delta = FpVar::<Fq>::Constant(delta_fq);
        let two_power_8: Fq = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
        let m_exp_fq: Fq = (2u64.pow(M_EXP)).into();
        let m_exp_constant = FpVar::<Fq>::Constant(m_exp_fq);

        let k_simd: usize = (254u32 / delta_bits_length) as usize;
        //let k_simd: usize = 1;

        if input_width - kernel_size + 1 >= k_simd {
            println!("NEW SIMD conv among multiple kernel positions on image");
            for k in 0..num_kernels {
                for n in 0..num_images {
                    for h in 0..(input_height - kernel_size + 1) {
                        for w in (0..input_width - kernel_size + 1).step_by(k_simd) {
                            let simd_batch_size =
                                cmp::min(k_simd, input_width - kernel_size - w + 1);

                            let mut m_list: Vec<FqVar> = Vec::new();

                            let mut dot_product_list: Vec<FqVar> = Vec::new();

                            let mut output_var_list: Vec<FqVar> = Vec::new();

                            for index in 0..simd_batch_size {
                                //multiplier for quantization

                                //SIMD among multipler image locations. they use the same multiplier
                                let m: Fq =
                                    ((self.multiplier[k] * (2.pow(M_EXP)) as f32) as u128).into();
                                m_list.push(FpVar::<Fq>::Constant(m));

                                let dot_product_tmp = conv_kernel_helper_fq(
                                    cs.clone(),
                                    self.x[n].clone(),
                                    self.conv_kernel[k].clone(),
                                    h,
                                    w + index,
                                    self.x_0,
                                    self.conv_kernel_0,
                                    self.y_0[k],
                                );
                                dot_product_list.push(dot_product_tmp);

                                let yy_var = self.y[n][k][h][w + index].clone();
                                let div: Fq = (self.div[n][k][h][w + index] as u64).into();
                                let div_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div gadget"),
                                    || Ok(div),
                                )
                                .unwrap();
                                let remainder: Fq =
                                    (self.remainder[n][k][h][w + index] as u64).into();
                                let remainder_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder gadget"),
                                    || Ok(remainder),
                                )
                                .unwrap();
                                let output_var = (yy_var + div_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder_var;

                                output_var_list.push(output_var);
                            }

                            //stranded encoded equality check two equation at the same time

                            let mut left =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || {
                                    Ok(Fq::zero())
                                })
                                .unwrap();
                            let mut right =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || {
                                    Ok(Fq::zero())
                                })
                                .unwrap();

                            //stranded encoding
                            for (k, e) in output_var_list.iter().rev().enumerate() {
                                left = left * delta.clone();
                                left = left + e;

                                right = right * delta.clone();
                                right = right
                                    + m_list[simd_batch_size - k - 1].clone()
                                        * dot_product_list[simd_batch_size - k - 1].clone();
                            }

                            left.enforce_equal(&right).unwrap();
                        }
                    }
                }
            }
        } else {
            println!("conv SIMD among multiple kernel channels due to too small input size");
            for n in 0..num_images {
                for h in 0..(input_height - kernel_size + 1) {
                    for w in 0..(input_width - kernel_size + 1) {
                        for k in (0..num_kernels).step_by(k_simd) {
                            let simd_batch_size = cmp::min(k_simd, num_kernels - k);

                            let mut m_list: Vec<FqVar> = Vec::new();

                            let mut dot_product_list: Vec<FqVar> = Vec::new();

                            let mut output_var_list: Vec<FqVar> = Vec::new();

                            for index in 0..simd_batch_size {
                                //multiplier for quantization

                                //SIMD among multi kernels. they use different multipliers
                                let m: Fq = ((self.multiplier[k + index] * (2.pow(M_EXP)) as f32)
                                    as u128)
                                    .into();
                                m_list.push(FpVar::<Fq>::Constant(m));

                                let dot_product_tmp = conv_kernel_helper_fq(
                                    cs.clone(),
                                    self.x[n].clone(),
                                    self.conv_kernel[k + index].clone(),
                                    h,
                                    w,
                                    self.x_0,
                                    self.conv_kernel_0,
                                    self.y_0[k + index],
                                );
                                dot_product_list.push(dot_product_tmp);

                                let yy_var = self.y[n][k + index][h][w].clone();
                                let div: Fq = (self.div[n][k + index][h][w] as u64).into();
                                let div_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "div gadget"),
                                    || Ok(div),
                                )
                                .unwrap();
                                let remainder: Fq =
                                    (self.remainder[n][k + index][h][w] as u64).into();
                                let remainder_var = FpVar::<Fq>::new_witness(
                                    r1cs_core::ns!(cs, "remainder gadget"),
                                    || Ok(remainder),
                                )
                                .unwrap();
                                let output_var = (yy_var + div_var * two_power_8_constant.clone())
                                    * m_exp_constant.clone()
                                    + remainder_var;

                                output_var_list.push(output_var);
                            }

                            //stranded encoded equality check two equation at the same time

                            let mut left =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || {
                                    Ok(Fq::zero())
                                })
                                .unwrap();
                            let mut right =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || {
                                    Ok(Fq::zero())
                                })
                                .unwrap();

                            //stranded encoding
                            for (k, e) in output_var_list.iter().rev().enumerate() {
                                left = left * delta.clone();
                                left = left + e;

                                right = right * delta.clone();
                                right = right
                                    + m_list[simd_batch_size - k - 1].clone()
                                        * dot_product_list[simd_batch_size - k - 1].clone();
                            }

                            left.enforce_equal(&right).unwrap();
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

fn conv_kernel_helper_fq(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<Vec<FqVar>>>,
    kernel: Vec<Vec<Vec<u8>>>,
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
                let w: Fq = kernel[i][j - h_index][k - w_index].into();
                //let w_const = FpVar::Constant(w);
                //x_zeropoint, kernel_zeropoints and y_zeropoints are all Constant wires because they are independent of input image
                let w_const =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "w tmp"), || Ok(w)).unwrap();
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





#[derive(Debug, Clone)]
pub struct FCCircuitOp3Depreciated {
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

impl ConstraintSynthesizer<Fq> for FCCircuitOp3Depreciated {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        //log(delta) = 16 + log2(vec_length) + log(M)
        let delta_bits_length = fast_math::log2(self.l1_mat[0].len() as f32) as u32 + 16u32 + M_EXP;

        let delta_fq = (2u128.pow(delta_bits_length)).into();
        let delta = FpVar::<Fq>::Constant(delta_fq);
        let two_power_8: Fq = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
        let m_exp_fq: Fq = (2u64.pow(M_EXP)).into();
        let m_exp_constant = FpVar::<Fq>::Constant(m_exp_fq);

        let k_simd: usize = (254u32 / delta_bits_length) as usize;
        //let k_simd: usize = 1;

        //the length of y is possible not to be divided by 3;

        for i in (0..self.y.len()).step_by(k_simd) {
            //multipliers for quantization is fixed parameters after training the model.
            let mut m_list: Vec<FqVar> = Vec::new();

            let mut dot_product_list: Vec<FqVar> = Vec::new();

            let mut output_var_list: Vec<FqVar> = Vec::new();

            //the final batch could be less than k_simd;
            let batch_size = cmp::min(k_simd, self.y.len() - i);

            for k in 0..batch_size {
                //multiplier for quantization
                let m: Fq = ((self.multiplier[i + k] * (2.pow(M_EXP)) as f32) as u128).into();
                m_list.push(FpVar::<Fq>::Constant(m));

                //vector dot product
                let dot_product_tmp = scala_cs_helper_u8(
                    cs.clone(),
                    &self.x,
                    &self.l1_mat[i + k],
                    self.x_0,
                    self.l1_mat_0,
                    self.y_0[i + k],
                );
                dot_product_list.push(dot_product_tmp);

                //obtain the output for verification

                let yy_var = self.y[i + k].clone();
                let div: Fq = (self.div[i + k] as u64).into();
                let div_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || Ok(div)).unwrap();
                let remainder: Fq = (self.remainder[i + k] as u64).into();
                let remainder_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || {
                        Ok(remainder)
                    })
                    .unwrap();
                let output_var = (yy_var + div_var * two_power_8_constant.clone())
                    * m_exp_constant.clone()
                    + remainder_var;

                output_var_list.push(output_var);
            }

            //stranded encoded equality check two equation at the same time

            let mut left =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || Ok(Fq::zero()))
                    .unwrap();
            let mut right =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || Ok(Fq::zero()))
                    .unwrap();

            //stranded encoding
            for (k, e) in output_var_list.iter().rev().enumerate() {
                left = left * delta.clone();
                left = left + e;

                right = right * delta.clone();
                right = right
                    + m_list[batch_size - k - 1].clone()
                        * dot_product_list[batch_size - k - 1].clone();
            }

            left.enforce_equal(&right).unwrap();
        }

        Ok(())
    }
}



fn scala_cs_helper_u8(
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
        //let w_const = FpVar::Constant(w);
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



//stranded encoding for avg pool layer
#[derive(Debug, Clone)]
pub struct AvgPoolCircuitLv3Depreciated {
    pub x: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<FqVar>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
    pub remainder: Vec<Vec<Vec<Vec<u8>>>>,
    // we do not need the quantization parameters to calculate the avg pool output
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuitLv3Depreciated {
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

        //for avg pool, it is simply suming some u8 together so we do not need that much bits to encode each result.
        let delta_bits_length = 20;  
        let delta_fq = (2u128.pow(delta_bits_length)).into();
        let delta = FpVar::<Fq>::Constant(delta_fq);

        let k_simd : usize= (254u32 / delta_bits_length) as usize;

        let kernel_size_fq : Fq = (self.kernel_size as u32).into();
        let kernel_size_const = FpVar::<Fq>::Constant(kernel_size_fq);


        if(input_width / self.kernel_size < 6 && num_channels > k_simd){
            //simd among multi channels
            for n in 0..num_images {
                    for h in 0..(input_height / self.kernel_size) {
                        for w in 0..(input_width / self.kernel_size){
                            for c in (0..num_channels).step_by(k_simd) {
                                let simd_batch_size = cmp::min(k_simd, num_channels - c);
                                let mut sum_list : Vec<FqVar> = Vec::new();
                    
                                let mut output_var_list : Vec<FqVar> = Vec::new();
        
                                for index in 0..simd_batch_size {
                                    let tmp = sum_helper_fq(
                                        cs.clone(),
                                        self.x[n][c + index].clone(),
                                        self.kernel_size * h,
                                        self.kernel_size * w ,
                                        self.kernel_size,
                                    );
                                    sum_list.push(tmp);
        
                                    let yy_var = self.y[n][c + index][h][w].clone();
        
                                    let remainder: Fq = (self.remainder[n][c + index][h][w] as u64).into();
                                    let remainder_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder)).unwrap();
        
                                    let output_var = yy_var * kernel_size_const.clone() * kernel_size_const.clone() + remainder_var;
        
                                    output_var_list.push(output_var);
                                }
        
                                //stranded encoded equality check two equation at the same time
            
                                let mut left = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"),  || Ok(Fq::zero())).unwrap();
                                let mut right = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"),  || Ok(Fq::zero())).unwrap();
        
                                //stranded encoding
                                for (k, e)in output_var_list.iter().rev().enumerate() {
                                    left = left * delta.clone();
                                    left = left + e; 
        
                                    right = right * delta.clone();
                                    right = right + sum_list[simd_batch_size - k - 1].clone();
                                }
        
                                left.enforce_equal(&right).unwrap();
                            }
                        }
                    }
                }
        }else{
            //simd among multi input locations
            for n in 0..num_images {
                for c in 0..num_channels {
                    for h in 0..(input_height / self.kernel_size) {
                        for w in (0..(input_width / self.kernel_size)).step_by(k_simd) {
                            // self.y[n][c][x][y] = np.mean(self.x[n][c][kernel_size*x:kernel_size*(x+1)][kernel_size*y:kernel_size*(y+1)])
                            let simd_batch_size = cmp::min(k_simd, input_width / self.kernel_size - w);
                            
                            let mut sum_list : Vec<FqVar> = Vec::new();
                    
                            let mut output_var_list : Vec<FqVar> = Vec::new();
    
                            for index in 0..simd_batch_size {
                                let tmp = sum_helper_fq(
                                    cs.clone(),
                                    self.x[n][c].clone(),
                                    self.kernel_size * h,
                                    self.kernel_size * (w + index),
                                    self.kernel_size,
                                );
                                sum_list.push(tmp);
    
                                let yy_var = self.y[n][c][h][w + index].clone();
    
                                let remainder: Fq = (self.remainder[n][c][h][w + index] as u64).into();
                                let remainder_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder)).unwrap();
    
                                let output_var = yy_var * kernel_size_const.clone() * kernel_size_const.clone() + remainder_var;
    
                                output_var_list.push(output_var);
                            }
    
                            //stranded encoded equality check two equation at the same time
        
                            let mut left = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"),  || Ok(Fq::zero())).unwrap();
                            let mut right = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"),  || Ok(Fq::zero())).unwrap();
    
                            //stranded encoding
                            for (k, e)in output_var_list.iter().rev().enumerate() {
                                left = left * delta.clone();
                                left = left + e; 
    
                                right = right * delta.clone();
                                right = right + sum_list[simd_batch_size - k - 1].clone();
                            }
    
                            left.enforce_equal(&right).unwrap();
                        }
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
    let mut tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "avg pool sum helper gadget"), || Ok(Fq::zero())).unwrap();
    for i in h_index..(h_index + kernel_size) {
        for j in w_index..(w_index + kernel_size) {
            tmp +=  x[i][j].clone();
        }
    }

    tmp
}
