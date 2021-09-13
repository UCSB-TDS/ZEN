
use crate::argmax_circuit::*;
use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::cosine_circuit::*;
use crate::fc_circuit::*;
use crate::relu_circuit::*;
use crate::vanilla::*;
use crate::*;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use std::cmp::*;
use ark_ff::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_ed_on_bls12_381::{constraints::FqVar, Fq};
use ark_r1cs_std::boolean::Boolean;
use ark_sponge::poseidon::PoseidonParameters;
use ark_r1cs_std::eq::EqGadget;




pub fn convert_2d_vector_into_1d(vec: Vec<Vec<u8>>) -> Vec<u8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        res.extend(&vec[i]);
    }
    res
}

pub fn convert_4d_vector_into_1d(vec: Vec<Vec<Vec<Vec<u8>>>>) -> Vec<u8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            for k in 0..vec[0][0].len() {
                res.extend(&vec[i][j][k]);
            }
        }
    }
    res
}

pub fn convert_2d_vector_into_fq(vec: Vec<Vec<u8>>) -> Vec<Fq> {
    let mut res = vec![Fq::zero(); vec[0].len() * vec.len()];
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            let tmp: Fq = vec[i][j].into();
            res[i * vec[0].len() + j] = tmp;
        }
    }
    res
}

pub fn convert_4d_vector_into_fq(vec: Vec<Vec<Vec<Vec<u8>>>>) -> Vec<Fq> {
    let mut res = vec![Fq::zero(); vec[0][0][0].len() * vec[0][0].len() * vec[0].len() * vec.len()];
    let mut counter = 0;
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            for k in 0..vec[0][0].len() {
                for m in 0..vec[0][0][0].len() {
                    let tmp: Fq = vec[i][j][k][m].into();
                    res[counter] = tmp;
                    counter += 1;
                }
            }
        }
    }
    res
}

fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
}

fn generate_fqvar4d(
    cs: ConstraintSystemRef<Fq>,
    input: Vec<Vec<Vec<Vec<u8>>>>,
) -> Vec<Vec<Vec<Vec<FqVar>>>> {
    let mut res: Vec<Vec<Vec<Vec<FqVar>>>> =
        vec![
            vec![
                vec![
                    vec![FpVar::<Fq>::Constant(Fq::zero()); input[0][0][0].len()];
                    input[0][0].len()
                ];
                input[0].len()
            ];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for l in 0..input[i][j][k].len() {
                    let fq: Fq = input[i][j][k][l].into();
                    let tmp =
                        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
                    res[i][j][k][l] = tmp;
                }
            }
        }
    }
    res
}

fn generate_fqvar_witness2D(cs: ConstraintSystemRef<Fq>, input: Vec<Vec<u8>>) -> Vec<Vec<FqVar>> {
    let zero_var = FpVar::<Fq>::Constant(Fq::zero());
    let mut res: Vec<Vec<FqVar>> = vec![vec![zero_var; input[0].len()]; input.len()];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            let fq: Fq = input[i][j].into();
            let tmp = FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
            res[i][j] = tmp;
        }
    }
    res
}

fn generate_fqvar_witness4D(
    cs: ConstraintSystemRef<Fq>,
    input: Vec<Vec<Vec<Vec<u8>>>>,
) -> Vec<Vec<Vec<Vec<FqVar>>>> {
    let zero_var = FpVar::<Fq>::Constant(Fq::zero());
    let mut res: Vec<Vec<Vec<Vec<FqVar>>>> =
        vec![
            vec![vec![vec![zero_var; input[0][0][0].len()]; input[0][0].len()]; input[0].len()];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for m in 0..input[i][j][k].len() {
                    let fq: Fq = input[i][j][k][m].into();
                    let tmp =
                        FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
                    res[i][j][k][m] = tmp;
                }
            }
        }
    }
    res
}





fn generate_fqvar_input4D(
    cs: ConstraintSystemRef<Fq>,
    input: Vec<Vec<Vec<Vec<u8>>>>,
) -> Vec<Vec<Vec<Vec<FqVar>>>> {
    let zero_var = FpVar::<Fq>::Constant(Fq::zero());
    let mut res: Vec<Vec<Vec<Vec<FqVar>>>> =
        vec![
            vec![vec![vec![zero_var; input[0][0][0].len()]; input[0][0].len()]; input[0].len()];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for m in 0..input[i][j][k].len() {
                    let fq: Fq = input[i][j][k][m].into();
                    let tmp = FpVar::<Fq>::new_input(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
                    res[i][j][k][m] = tmp;
                }
            }
        }
    }
    res
}


fn generate_fqvar_input(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_input(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
}
#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv3PoseidonClassificationAccuracy {
    pub params: SPNGParam,

    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_squeeze: SPNGOutput,

    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv1_squeeze: SPNGOutput,

    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_squeeze: SPNGOutput,

    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_squeeze: SPNGOutput,

    pub fc1_weights: Vec<Vec<u8>>,
    pub fc1_squeeze: SPNGOutput,

    pub fc2_weights: Vec<Vec<u8>>,
    pub fc2_squeeze: SPNGOutput,

    pub z: Vec<Vec<u8>>,
    pub z_squeeze: SPNGOutput,

    //zero points for quantization.
    pub x_0: u8,
    pub conv1_output_0: u8,
    pub conv2_output_0: u8,
    pub conv3_output_0: u8,
    pub fc1_output_0: u8,
    pub fc2_output_0: u8, // which is also lenet output(z) zero point

    pub conv1_weights_0: u8,
    pub conv2_weights_0: u8,
    pub conv3_weights_0: u8,
    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,

    //multiplier for quantization
    pub multiplier_conv1: Vec<f32>,
    pub multiplier_conv2: Vec<f32>,
    pub multiplier_conv3: Vec<f32>,
    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    //we do not need multiplier in relu and AvgPool layer

    pub true_labels: Vec<u8>,
    pub accuracy_result: Vec<u8>, //1 stands for prediction is correct, 0 is wrong prediction.
    pub accuracy_squeeze: Vec<Fq>
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv3PoseidonClassificationAccuracy {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "LeNetCircuitU8OptimizedLv3PedersenClassification is setup mode: {}",
            cs.is_in_setup_mode()
        );


        let mut _cir_number = cs.num_constraints();
        let correct_prediction = Boolean::<Fq>::Constant(true);
        let wrong_prediction = Boolean::<Fq>::Constant(false);


        let conv1_mat_1d = convert_4d_vector_into_1d(self.conv1_weights.clone());
        let conv1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv1_mat_1d.clone(),
            output: self.conv1_squeeze.clone()
        };
        conv1_com_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for conv1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv2_mat_1d = convert_4d_vector_into_1d(self.conv2_weights.clone());
        let conv2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv2_mat_1d.clone(),
            output: self.conv2_squeeze.clone()
        };
        conv2_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for conv2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv3_mat_1d = convert_4d_vector_into_1d(self.conv3_weights.clone());
        let conv3_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv3_mat_1d.clone(),
            output: self.conv3_squeeze.clone()
        };

        conv3_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for conv3 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let l1_mat_1d = convert_2d_vector_into_1d(self.fc1_weights.clone());
        let l1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l1_mat_1d.clone(),
            output: self.fc1_squeeze.clone()
        };
        l1_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for fc1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let l2_mat_1d = convert_2d_vector_into_1d(self.fc2_weights.clone());
        let l2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l2_mat_1d.clone(),
            output: self.fc2_squeeze.clone()
        };
        l2_com_circuit.generate_constraints(cs.clone())?;
        

        println!(
            "Number of constraints for fc2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let x_input = generate_fqvar_input4D(cs.clone(), self.x.clone());
        let true_label_input = generate_fqvar_input(cs.clone(), self.true_labels.clone());

          //layer 1
        //conv1
        let mut conv1_output = vec![vec![vec![vec![0u8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv1_weights.len()]; //number of conv kernels
                                            self.x.len()]; //input (image) batch size
        let (remainder_conv1, div_conv1) = vec_conv_with_remainder_u8(
            &self.x,
            &self.conv1_weights,
            &mut conv1_output,
            self.x_0,
            self.conv1_weights_0,
            self.conv1_output_0,
            &self.multiplier_conv1,
        );

        let conv1_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv1_output.clone());
        let conv1_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv1_weights.clone());
        // conv1_output_0 and multiplier_conv1 are both constants.
        let mut conv1_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv1.len() {
            let m = (self.multiplier_conv1[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv1_output_zeropoint_converted
                .push((self.conv1_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD for reducing constraints
        let conv1_circuit = ConvCircuitOp3 {
            x: x_input.clone(),
            conv_kernel: conv1_weight_fqvar_input.clone(),
            y: conv1_output_fqvar.clone(),
            remainder: remainder_conv1.clone(),
            div: div_conv1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: conv1_output_zeropoint_converted,

            multiplier: self.multiplier_conv1,
        };
        conv1_circuit.generate_constraints(cs.clone())?;

        //relu1
        let relu1_cmp_res = relu4d_u8(&mut conv1_output, self.conv1_output_0);
        let relu1_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv1_output.clone());

        let relu1_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu1_cmp_res.into_iter().flatten().collect();
        let relu1_cmp_res_2d: Vec<Vec<bool>> = relu1_cmp_res_3d.into_iter().flatten().collect();
        let relu1_cmp_res_1d: Vec<bool> = relu1_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu1_input3d: Vec<Vec<Vec<FqVar>>> =
            conv1_output_fqvar.into_iter().flatten().collect();
        let flattened_relu1_input2d: Vec<Vec<FqVar>> =
            flattened_relu1_input3d.into_iter().flatten().collect();
        let flattened_relu1_input1d: Vec<FqVar> =
            flattened_relu1_input2d.into_iter().flatten().collect();

        let flattened_relu1_output3d: Vec<Vec<Vec<FqVar>>> =
            relu1_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu1_output2d: Vec<Vec<FqVar>> =
            flattened_relu1_output3d.into_iter().flatten().collect();
        let flattened_relu1_output1d: Vec<FqVar> =
            flattened_relu1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu1_input1d.clone(),
            y_out: flattened_relu1_output1d.clone(),
            y_zeropoint: self.conv1_output_0,
            cmp_res: relu1_cmp_res_1d.clone(),
        };

        relu1_circuit.generate_constraints(cs.clone())?;

        //avg_pool1

        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_output_fqvar = generate_fqvar_witness4D(cs.clone(), avg_pool1_output.clone());
        let avg_pool1_circuit = AvgPoolCircuitLv3 {
            x: relu1_output_fqvar.clone(),
            y: avg_pool1_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;

        //layer 2 :
        //Conv2 -> relu -> AvgPool
        let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                        self.conv2_weights.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size

        let (remainder_conv2, div_conv2) = vec_conv_with_remainder_u8(
            &avg_pool1_output,
            &self.conv2_weights,
            &mut conv2_output,
            self.conv1_output_0,
            self.conv2_weights_0,
            self.conv2_output_0,
            &self.multiplier_conv2,
        );
        //println!("{:?}", self.conv2_weights.clone());
        let conv2_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv2_output.clone());
        let conv2_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv2_weights.clone());

        // y_0 and multiplier_l1 are both constants.
        let mut conv2_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv2.len() {
            let m = (self.multiplier_conv2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv2_output_zeropoint_converted
                .push((self.conv2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }
        // println!("conv2_output_zeropoint_converted {:?}", conv2_output_zeropoint_converted.clone());
        // println!("conv2 multiplier {:?}", self.multiplier_conv2.clone());
        //use SIMD to reduce the number of constraints
        let conv2_circuit = ConvCircuitOp3 {
            x: avg_pool1_output_fqvar.clone(),
            conv_kernel: conv2_weight_fqvar_input.clone(),
            y: conv2_output_fqvar.clone(),
            remainder: remainder_conv2.clone(),
            div: div_conv2.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: conv2_output_zeropoint_converted,

            multiplier: self.multiplier_conv2,
        };
        conv2_circuit.generate_constraints(cs.clone())?;

        //relu2 layer

        let relu2_cmp_res = relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let relu2_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv2_output.clone());
        let relu2_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu2_cmp_res.into_iter().flatten().collect();
        let relu2_cmp_res_2d: Vec<Vec<bool>> = relu2_cmp_res_3d.into_iter().flatten().collect();
        let relu2_cmp_res_1d: Vec<bool> = relu2_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu2_input3d: Vec<Vec<Vec<FqVar>>> =
            conv2_output_fqvar.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<FqVar>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<FqVar> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_relu2_output3d: Vec<Vec<Vec<FqVar>>> =
            relu2_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu2_output2d: Vec<Vec<FqVar>> =
            flattened_relu2_output3d.into_iter().flatten().collect();
        let flattened_relu2_output1d: Vec<FqVar> =
            flattened_relu2_output2d.into_iter().flatten().collect();

        let relu2_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu2_input1d.clone(),
            y_out: flattened_relu2_output1d.clone(),
            y_zeropoint: self.conv2_output_0,
            cmp_res: relu2_cmp_res_1d.clone(),
        };
        relu2_circuit.generate_constraints(cs.clone())?;

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_output_fqvar = generate_fqvar_witness4D(cs.clone(), avg_pool2_output.clone());
        let avg_pool2_circuit = AvgPoolCircuitLv3 {
            x: relu2_output_fqvar.clone(),
            y: avg_pool2_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;

        //layer 3 :
        //Conv3 -> relu -> reshape output for following FC layer
        let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                            avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                            self.conv3_weights.len()]; //number of conv kernels
                                                                            avg_pool2_output.len()]; //input (image) batch size
                                                                                                     //conv3 layer
        let (remainder_conv3, div_conv3) = vec_conv_with_remainder_u8(
            &avg_pool2_output,
            &self.conv3_weights,
            &mut conv3_output,
            self.conv2_output_0,
            self.conv3_weights_0,
            self.conv3_output_0,
            &self.multiplier_conv3,
        );

        let conv3_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv3_output.clone());
        let conv3_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv3_weights.clone());

        // y_0 and multiplier_l1 are both constants.
        let mut conv3_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv3.len() {
            let m = (self.multiplier_conv3[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv3_output_zeropoint_converted
                .push((self.conv3_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD to reduce the number of constraints
        let conv3_circuit = ConvCircuitOp3 {
            x: avg_pool2_output_fqvar.clone(),
            conv_kernel: conv3_weight_fqvar_input,
            y: conv3_output_fqvar.clone(),
            remainder: remainder_conv3.clone(),
            div: div_conv3.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: conv3_output_zeropoint_converted,

            multiplier: self.multiplier_conv3,
        };
        conv3_circuit.generate_constraints(cs.clone())?;

        //relu3 layer

        let relu3_cmp_res = relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let relu3_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv3_output.clone());
        let relu3_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu3_cmp_res.into_iter().flatten().collect();
        let relu3_cmp_res_2d: Vec<Vec<bool>> = relu3_cmp_res_3d.into_iter().flatten().collect();
        let relu3_cmp_res_1d: Vec<bool> = relu3_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu3_input3d: Vec<Vec<Vec<FqVar>>> =
            conv3_output_fqvar.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<FqVar>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<FqVar> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_relu3_output3d: Vec<Vec<Vec<FqVar>>> =
            relu3_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu3_output2d: Vec<Vec<FqVar>> =
            flattened_relu3_output3d.into_iter().flatten().collect();
        let flattened_relu3_output1d: Vec<FqVar> =
            flattened_relu3_output2d.into_iter().flatten().collect();

        let relu3_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu3_input1d.clone(),
            y_out: flattened_relu3_output1d.clone(),
            y_zeropoint: self.conv3_output_0,
            cmp_res: relu3_cmp_res_1d.clone(),
        };
        relu3_circuit.generate_constraints(cs.clone())?;

        //flatten to fit FC layers
        let mut transformed_conv3_output =
            vec![
                vec![
                    0u8;
                    conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
                ];
                conv3_output.len()
            ];
        let mut transformed_conv3_output_fqvar =
            vec![
                vec![
                    FpVar::<Fq>::Constant(Fq::zero());
                    conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
                ];
                conv3_output.len()
            ];
        for i in 0..conv3_output.len() {
            let mut counter = 0;
            for j in 0..conv3_output[0].len() {
                for p in 0..conv3_output[0][0].len() {
                    for q in 0..conv3_output[0][0][0].len() {
                        transformed_conv3_output[i][counter] = conv3_output[i][j][p][q];
                        transformed_conv3_output_fqvar[i][counter] =
                            relu3_output_fqvar[i][j][p][q].clone();
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        for i in 0..transformed_conv3_output.len() {
            let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
            transformed_conv3_output.len()]; //batch size
            let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

            //assume we only do inference on one image
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv3_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1,
            );

            let fc1_output_fqvar = generate_fqvar(cs.clone(), fc1_output[0].clone());
            let mut fc1_output0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_fc1.len() {
                let m = (self.multiplier_fc1[i] * (2u64.pow(M_EXP)) as f32) as u64;
                fc1_output0_converted.push((self.fc1_output_0 as u64 * 2u64.pow(M_EXP)) / m);
            }
            let fc1_weights_fqvar_input =
                generate_fqvar_witness2D(cs.clone(), self.fc1_weights.clone());

            let fc1_circuit = FCCircuitOp3 {
                x: transformed_conv3_output_fqvar[i].clone(),
                l1_mat: fc1_weights_fqvar_input.clone(),
                y: fc1_output_fqvar.clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv3_output_0,
                l1_mat_0: self.fc1_weights_0,
                y_0: fc1_output0_converted,

                multiplier: self.multiplier_fc1.clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;

            //relu4 layer
            //assume we only process one image
            let cmp_res = relu_u8(&mut fc1_output[0], self.fc1_output_0);
            let relu4_output_fqvar = generate_fqvar(cs.clone(), fc1_output[0].clone());
            let relu4_circuit = ReLUCircuitOp3 {
                y_in: fc1_output_fqvar.clone(),
                y_out: relu4_output_fqvar.clone(),
                y_zeropoint: self.fc1_output_0,
                cmp_res: cmp_res.clone(),
            };
            relu4_circuit.generate_constraints(cs.clone())?;

            //layer 5 :
            //FC2 -> output
            let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
            fc1_output.len()]; //batch size
            let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

            let (remainder_fc2, div_fc2) = vec_mat_mul_with_remainder_u8(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
                self.fc1_output_0,
                self.fc2_weights_0,
                self.fc2_output_0,
                &self.multiplier_fc2.clone(),
            );
            //println!("z within circuit {:?}", fc2_output.clone());

            let fc2_output_fqvar = generate_fqvar(cs.clone(), fc2_output[0].clone());
            let fc2_weights_fqvar_input =
                generate_fqvar_witness2D(cs.clone(), self.fc2_weights.clone());

            let mut fc2_output0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_fc2.len() {
                let m = (self.multiplier_fc2[i] * (2u64.pow(M_EXP)) as f32) as u64;
                fc2_output0_converted.push((self.fc2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
            }
            let fc2_circuit = FCCircuitOp3 {
                x: relu4_output_fqvar.clone(),
                l1_mat: fc2_weights_fqvar_input.clone(),
                y: fc2_output_fqvar.clone(),

                remainder: remainder_fc2.clone(),
                div: div_fc2.clone(),

                x_0: self.fc1_output_0,
                l1_mat_0: self.fc2_weights_0,
                y_0: fc2_output0_converted,

                multiplier: self.multiplier_fc2.clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;

            let classification_res = argmax_u8(fc2_output[i].clone()) as usize;
            //println!("z {:?}\n argmax {}", zz.clone(), classification_res);
            let classification_res_fq: Fq = (classification_res as u32).into();
            let classification_res_var =
                FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "classification res"), || {
                    Ok(classification_res_fq)
                })
                .unwrap();
            let classification_max: Fq = fc2_output[i][classification_res].into();
            let classification_max_var =
                FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "classification max"), || {
                    Ok(classification_max)
                })
                .unwrap();
            let argmax_circuit = ArgmaxCircuit {
                input: fc2_output_fqvar.clone(),
                argmax_res: classification_max_var.clone(),
            };

            argmax_circuit
                .clone()
                .generate_constraints(cs.clone())
                .unwrap();

            //add final constraints on the prediction correctness
            let is_prediction_correct = true_label_input[i].is_eq(&classification_res_var).unwrap();

            // println!("prediction_correct {:?}", prediction_correct.value().unwrap());
            if self.accuracy_result[i].clone() == 1u8 {
                is_prediction_correct
                    .enforce_equal(&correct_prediction)
                    .unwrap();
            } else {
                is_prediction_correct
                    .enforce_equal(&wrong_prediction)
                    .unwrap();
            }
        }

        println!(
            "Total number of LeNetCircuit Classification constraints {}",
            cs.num_constraints()
        );


        Ok(())
    }
}


#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv3PoseidonRecognitionAccuracy {
    pub params: SPNGParam,

    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_squeeze: SPNGOutput,

    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv1_squeeze: SPNGOutput,

    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_squeeze: SPNGOutput,

    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_squeeze: SPNGOutput,

    pub fc1_weights: Vec<Vec<u8>>,
    pub fc1_squeeze: SPNGOutput,

    pub fc2_weights: Vec<Vec<u8>>,
    pub fc2_squeeze: SPNGOutput,

    pub z: Vec<Vec<u8>>,
    pub z_squeeze: SPNGOutput,

    //zero points for quantization.
    pub x_0: u8,
    pub conv1_output_0: u8,
    pub conv2_output_0: u8,
    pub conv3_output_0: u8,
    pub fc1_output_0: u8,
    pub fc2_output_0: u8, // which is also lenet output(z) zero point

    pub conv1_weights_0: u8,
    pub conv2_weights_0: u8,
    pub conv3_weights_0: u8,
    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,

    //multiplier for quantization
    pub multiplier_conv1: Vec<f32>,
    pub multiplier_conv2: Vec<f32>,
    pub multiplier_conv3: Vec<f32>,
    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    //we do not need multiplier in relu and AvgPool layer


    pub person_feature_vector: Vec<u8>,
    pub threshold: u8,
    pub true_labels: Vec<u8>, // 1 or 0 to stands for whether the input image is the same person with the person_feature_vector.
    pub accuracy_result: Vec<u8>, //1 stands for prediction is correct, 0 is wrong prediction.
    pub accuracy_squeeze: Vec<Fq>
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv3PoseidonRecognitionAccuracy {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "LeNetCircuitU8OptimizedLv3PedersenRecognition is setup mode: {}",
            cs.is_in_setup_mode()
        );


        let mut _cir_number = cs.num_constraints();
        let correct_prediction = Boolean::<Fq>::Constant(true);
        let wrong_prediction = Boolean::<Fq>::Constant(false);


        let conv1_mat_1d = convert_4d_vector_into_1d(self.conv1_weights.clone());
        let conv1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv1_mat_1d.clone(),
            output: self.conv1_squeeze.clone()
        };
        conv1_com_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for conv1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv2_mat_1d = convert_4d_vector_into_1d(self.conv2_weights.clone());
        let conv2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv2_mat_1d.clone(),
            output: self.conv2_squeeze.clone()
        };
        conv2_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for conv2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv3_mat_1d = convert_4d_vector_into_1d(self.conv3_weights.clone());
        let conv3_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: conv3_mat_1d.clone(),
            output: self.conv3_squeeze.clone()
        };

        conv3_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for conv3 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let l1_mat_1d = convert_2d_vector_into_1d(self.fc1_weights.clone());
        let l1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l1_mat_1d.clone(),
            output: self.fc1_squeeze.clone()
        };
        l1_com_circuit.generate_constraints(cs.clone())?;
        
        println!(
            "Number of constraints for fc1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let l2_mat_1d = convert_2d_vector_into_1d(self.fc2_weights.clone());
        let l2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l2_mat_1d.clone(),
            output: self.fc2_squeeze.clone()
        };
        l2_com_circuit.generate_constraints(cs.clone())?;
        

        println!(
            "Number of constraints for fc2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let x_input = generate_fqvar_input4D(cs.clone(), self.x.clone());
        let true_label_input = generate_fqvar_input(cs.clone(), self.true_labels.clone());

        let person_feature_input =
        generate_fqvar_input(cs.clone(), self.person_feature_vector.clone());
        //layer 1
        //conv1
        let mut conv1_output = vec![vec![vec![vec![0u8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv1_weights.len()]; //number of conv kernels
                                            self.x.len()]; //input (image) batch size
        let (remainder_conv1, div_conv1) = vec_conv_with_remainder_u8(
            &self.x,
            &self.conv1_weights,
            &mut conv1_output,
            self.x_0,
            self.conv1_weights_0,
            self.conv1_output_0,
            &self.multiplier_conv1,
        );

        let conv1_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv1_output.clone());
        let conv1_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv1_weights.clone());
        // conv1_output_0 and multiplier_conv1 are both constants.
        let mut conv1_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv1.len() {
            let m = (self.multiplier_conv1[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv1_output_zeropoint_converted
                .push((self.conv1_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD for reducing constraints
        let conv1_circuit = ConvCircuitOp3 {
            x: x_input.clone(),
            conv_kernel: conv1_weight_fqvar_input.clone(),
            y: conv1_output_fqvar.clone(),
            remainder: remainder_conv1.clone(),
            div: div_conv1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: conv1_output_zeropoint_converted,

            multiplier: self.multiplier_conv1,
        };
        conv1_circuit.generate_constraints(cs.clone())?;

        //relu1
        let relu1_cmp_res = relu4d_u8(&mut conv1_output, self.conv1_output_0);
        let relu1_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv1_output.clone());

        let relu1_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu1_cmp_res.into_iter().flatten().collect();
        let relu1_cmp_res_2d: Vec<Vec<bool>> = relu1_cmp_res_3d.into_iter().flatten().collect();
        let relu1_cmp_res_1d: Vec<bool> = relu1_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu1_input3d: Vec<Vec<Vec<FqVar>>> =
            conv1_output_fqvar.into_iter().flatten().collect();
        let flattened_relu1_input2d: Vec<Vec<FqVar>> =
            flattened_relu1_input3d.into_iter().flatten().collect();
        let flattened_relu1_input1d: Vec<FqVar> =
            flattened_relu1_input2d.into_iter().flatten().collect();

        let flattened_relu1_output3d: Vec<Vec<Vec<FqVar>>> =
            relu1_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu1_output2d: Vec<Vec<FqVar>> =
            flattened_relu1_output3d.into_iter().flatten().collect();
        let flattened_relu1_output1d: Vec<FqVar> =
            flattened_relu1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu1_input1d.clone(),
            y_out: flattened_relu1_output1d.clone(),
            y_zeropoint: self.conv1_output_0,
            cmp_res: relu1_cmp_res_1d.clone(),
        };

        relu1_circuit.generate_constraints(cs.clone())?;

        //avg_pool1

        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_output_fqvar = generate_fqvar_witness4D(cs.clone(), avg_pool1_output.clone());
        let avg_pool1_circuit = AvgPoolCircuitLv3 {
            x: relu1_output_fqvar.clone(),
            y: avg_pool1_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;

        //layer 2 :
        //Conv2 -> relu -> AvgPool
        let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                        self.conv2_weights.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size

        let (remainder_conv2, div_conv2) = vec_conv_with_remainder_u8(
            &avg_pool1_output,
            &self.conv2_weights,
            &mut conv2_output,
            self.conv1_output_0,
            self.conv2_weights_0,
            self.conv2_output_0,
            &self.multiplier_conv2,
        );
        //println!("{:?}", self.conv2_weights.clone());
        let conv2_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv2_output.clone());
        let conv2_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv2_weights.clone());

        // y_0 and multiplier_l1 are both constants.
        let mut conv2_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv2.len() {
            let m = (self.multiplier_conv2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv2_output_zeropoint_converted
                .push((self.conv2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }
        // println!("conv2_output_zeropoint_converted {:?}", conv2_output_zeropoint_converted.clone());
        // println!("conv2 multiplier {:?}", self.multiplier_conv2.clone());
        //use SIMD to reduce the number of constraints
        let conv2_circuit = ConvCircuitOp3 {
            x: avg_pool1_output_fqvar.clone(),
            conv_kernel: conv2_weight_fqvar_input.clone(),
            y: conv2_output_fqvar.clone(),
            remainder: remainder_conv2.clone(),
            div: div_conv2.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: conv2_output_zeropoint_converted,

            multiplier: self.multiplier_conv2,
        };
        conv2_circuit.generate_constraints(cs.clone())?;

        //relu2 layer

        let relu2_cmp_res = relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let relu2_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv2_output.clone());
        let relu2_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu2_cmp_res.into_iter().flatten().collect();
        let relu2_cmp_res_2d: Vec<Vec<bool>> = relu2_cmp_res_3d.into_iter().flatten().collect();
        let relu2_cmp_res_1d: Vec<bool> = relu2_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu2_input3d: Vec<Vec<Vec<FqVar>>> =
            conv2_output_fqvar.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<FqVar>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<FqVar> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_relu2_output3d: Vec<Vec<Vec<FqVar>>> =
            relu2_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu2_output2d: Vec<Vec<FqVar>> =
            flattened_relu2_output3d.into_iter().flatten().collect();
        let flattened_relu2_output1d: Vec<FqVar> =
            flattened_relu2_output2d.into_iter().flatten().collect();

        let relu2_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu2_input1d.clone(),
            y_out: flattened_relu2_output1d.clone(),
            y_zeropoint: self.conv2_output_0,
            cmp_res: relu2_cmp_res_1d.clone(),
        };
        relu2_circuit.generate_constraints(cs.clone())?;

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_output_fqvar = generate_fqvar_witness4D(cs.clone(), avg_pool2_output.clone());
        let avg_pool2_circuit = AvgPoolCircuitLv3 {
            x: relu2_output_fqvar.clone(),
            y: avg_pool2_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;

        //layer 3 :
        //Conv3 -> relu -> reshape output for following FC layer
        let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                            avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                            self.conv3_weights.len()]; //number of conv kernels
                                                                            avg_pool2_output.len()]; //input (image) batch size
                                                                                                     //conv3 layer
        let (remainder_conv3, div_conv3) = vec_conv_with_remainder_u8(
            &avg_pool2_output,
            &self.conv3_weights,
            &mut conv3_output,
            self.conv2_output_0,
            self.conv3_weights_0,
            self.conv3_output_0,
            &self.multiplier_conv3,
        );

        let conv3_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv3_output.clone());
        let conv3_weight_fqvar_input =
            generate_fqvar_witness4D(cs.clone(), self.conv3_weights.clone());

        // y_0 and multiplier_l1 are both constants.
        let mut conv3_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv3.len() {
            let m = (self.multiplier_conv3[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv3_output_zeropoint_converted
                .push((self.conv3_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD to reduce the number of constraints
        let conv3_circuit = ConvCircuitOp3 {
            x: avg_pool2_output_fqvar.clone(),
            conv_kernel: conv3_weight_fqvar_input,
            y: conv3_output_fqvar.clone(),
            remainder: remainder_conv3.clone(),
            div: div_conv3.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: conv3_output_zeropoint_converted,

            multiplier: self.multiplier_conv3,
        };
        conv3_circuit.generate_constraints(cs.clone())?;

        //relu3 layer

        let relu3_cmp_res = relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let relu3_output_fqvar = generate_fqvar_witness4D(cs.clone(), conv3_output.clone());
        let relu3_cmp_res_3d: Vec<Vec<Vec<bool>>> = relu3_cmp_res.into_iter().flatten().collect();
        let relu3_cmp_res_2d: Vec<Vec<bool>> = relu3_cmp_res_3d.into_iter().flatten().collect();
        let relu3_cmp_res_1d: Vec<bool> = relu3_cmp_res_2d.into_iter().flatten().collect();

        let flattened_relu3_input3d: Vec<Vec<Vec<FqVar>>> =
            conv3_output_fqvar.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<FqVar>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<FqVar> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_relu3_output3d: Vec<Vec<Vec<FqVar>>> =
            relu3_output_fqvar.clone().into_iter().flatten().collect();
        let flattened_relu3_output2d: Vec<Vec<FqVar>> =
            flattened_relu3_output3d.into_iter().flatten().collect();
        let flattened_relu3_output1d: Vec<FqVar> =
            flattened_relu3_output2d.into_iter().flatten().collect();

        let relu3_circuit = ReLUCircuitOp3 {
            y_in: flattened_relu3_input1d.clone(),
            y_out: flattened_relu3_output1d.clone(),
            y_zeropoint: self.conv3_output_0,
            cmp_res: relu3_cmp_res_1d.clone(),
        };
        relu3_circuit.generate_constraints(cs.clone())?;

        //flatten to fit FC layers
        let mut transformed_conv3_output =
            vec![
                vec![
                    0u8;
                    conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
                ];
                conv3_output.len()
            ];
        let mut transformed_conv3_output_fqvar =
            vec![
                vec![
                    FpVar::<Fq>::Constant(Fq::zero());
                    conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
                ];
                conv3_output.len()
            ];
        for i in 0..conv3_output.len() {
            let mut counter = 0;
            for j in 0..conv3_output[0].len() {
                for p in 0..conv3_output[0][0].len() {
                    for q in 0..conv3_output[0][0][0].len() {
                        transformed_conv3_output[i][counter] = conv3_output[i][j][p][q];
                        transformed_conv3_output_fqvar[i][counter] =
                            relu3_output_fqvar[i][j][p][q].clone();
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        for i in 0..transformed_conv3_output.len() {
            let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
            transformed_conv3_output.len()]; //batch size
            let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

            //assume we only do inference on one image
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv3_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1,
            );

            let fc1_output_fqvar = generate_fqvar(cs.clone(), fc1_output[0].clone());
            let mut fc1_output0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_fc1.len() {
                let m = (self.multiplier_fc1[i] * (2u64.pow(M_EXP)) as f32) as u64;
                fc1_output0_converted.push((self.fc1_output_0 as u64 * 2u64.pow(M_EXP)) / m);
            }
            let fc1_weights_fqvar_input =
                generate_fqvar_witness2D(cs.clone(), self.fc1_weights.clone());

            let fc1_circuit = FCCircuitOp3 {
                x: transformed_conv3_output_fqvar[i].clone(),
                l1_mat: fc1_weights_fqvar_input.clone(),
                y: fc1_output_fqvar.clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv3_output_0,
                l1_mat_0: self.fc1_weights_0,
                y_0: fc1_output0_converted,

                multiplier: self.multiplier_fc1.clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;

            //relu4 layer
            //assume we only process one image
            let cmp_res = relu_u8(&mut fc1_output[0], self.fc1_output_0);
            let relu4_output_fqvar = generate_fqvar(cs.clone(), fc1_output[0].clone());
            let relu4_circuit = ReLUCircuitOp3 {
                y_in: fc1_output_fqvar.clone(),
                y_out: relu4_output_fqvar.clone(),
                y_zeropoint: self.fc1_output_0,
                cmp_res: cmp_res.clone(),
            };
            relu4_circuit.generate_constraints(cs.clone())?;

            //layer 5 :
            //FC2 -> output
            let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
            fc1_output.len()]; //batch size
            let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

            let (remainder_fc2, div_fc2) = vec_mat_mul_with_remainder_u8(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
                self.fc1_output_0,
                self.fc2_weights_0,
                self.fc2_output_0,
                &self.multiplier_fc2.clone(),
            );
            //println!("z within circuit {:?}", fc2_output.clone());

            let fc2_output_fqvar = generate_fqvar(cs.clone(), fc2_output[0].clone());
            let fc2_weights_fqvar_input =
                generate_fqvar_witness2D(cs.clone(), self.fc2_weights.clone());

            let mut fc2_output0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_fc2.len() {
                let m = (self.multiplier_fc2[i] * (2u64.pow(M_EXP)) as f32) as u64;
                fc2_output0_converted.push((self.fc2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
            }
            let fc2_circuit = FCCircuitOp3 {
                x: relu4_output_fqvar.clone(),
                l1_mat: fc2_weights_fqvar_input.clone(),
                y: fc2_output_fqvar.clone(),

                remainder: remainder_fc2.clone(),
                div: div_fc2.clone(),

                x_0: self.fc1_output_0,
                l1_mat_0: self.fc2_weights_0,
                y_0: fc2_output0_converted,

                multiplier: self.multiplier_fc2.clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;

            
            let is_the_same_person: bool = cosine_similarity(
                fc2_output[i].clone(),
                self.person_feature_vector.clone(),
                self.threshold as u32,
            );

            let similarity_circuit = CosineSimilarityCircuit {
                vec1: fc2_output_fqvar.clone(),
                vec2: person_feature_input.clone(),
                threshold: self.threshold,
                result: is_the_same_person,
            };

            let mut is_the_same_person_numeric = 0u32;
            if (is_the_same_person) {
                is_the_same_person_numeric = 1u32;
            }
            let is_the_same_person_fq: Fq = is_the_same_person_numeric.into();
            let is_the_same_person_var =
                FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "is the same person"), || {
                    Ok(is_the_same_person_fq)
                })
                .unwrap();
            let is_prediction_correct = true_label_input[i].is_eq(&is_the_same_person_var).unwrap();

            similarity_circuit
                .clone()
                .generate_constraints(cs.clone())
                .unwrap();


            // println!("prediction_correct {:?}", prediction_correct.value().unwrap());
            if self.accuracy_result[i].clone() == 1u8 {
                is_prediction_correct
                    .enforce_equal(&correct_prediction)
                    .unwrap();
            } else {
                is_prediction_correct
                    .enforce_equal(&wrong_prediction)
                    .unwrap();
            }
        }

        println!(
            "Total number of LeNetCircuit Classification constraints {}",
            cs.num_constraints()
        );

        Ok(())
    }
}

