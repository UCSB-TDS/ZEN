use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::mul_circuit::*;
use crate::relu_circuit::*;
use crate::vanilla::*;
use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use pedersen_commit::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;

fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
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
                        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
                    res[i][j][k][l] = tmp;
                }
            }
        }
    }
    res
}

#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv3Pedersen {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub fc1_weights: Vec<Vec<u8>>,
    pub fc2_weights: Vec<Vec<u8>>,

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
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv3Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());
        //println!("{:?}", self.x);

        // x commitment
        let flattened_x3d: Vec<Vec<Vec<u8>>> = self.x.clone().into_iter().flatten().collect();
        let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
        let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_x1d.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        let output: Vec<Vec<u8>> = lenet_circuit_forward_u8(
            self.x.clone(),
            self.conv1_weights.clone(),
            self.conv2_weights.clone(),
            self.conv3_weights.clone(),
            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
            self.x_0,
            self.conv1_output_0,
            self.conv2_output_0,
            self.conv3_output_0,
            self.fc1_output_0,
            self.fc2_output_0,
            self.conv1_weights_0,
            self.conv2_weights_0,
            self.conv3_weights_0,
            self.fc1_weights_0,
            self.fc2_weights_0,
            self.multiplier_conv1.clone(),
            self.multiplier_conv2.clone(),
            self.multiplier_conv3.clone(),
            self.multiplier_fc1.clone(),
            self.multiplier_fc2.clone(),
        );
        // z commitment
        let flattened_z1d: Vec<u8> = output.clone().into_iter().flatten().collect();
        //println!("x within circuit {:?}", self.x.clone());
        println!("z within circuit {:?}", flattened_z1d.clone());
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_z1d.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        let x_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());
        let conv1_output_fqvar = generate_fqvar4d(cs.clone(), conv1_output.clone());

        // conv1_output_0 and multipler_conv1 are both constants.
        let mut conv1_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv1.len() {
            let m = (self.multiplier_conv1[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv1_output_zeropoint_converted
                .push((self.conv1_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD for reducing constraints
        let conv1_circuit = ConvCircuitOp3 {
            x: x_fqvar.clone(),
            conv_kernel: self.conv1_weights.clone(),
            y: conv1_output_fqvar.clone(),
            remainder: remainder_conv1.clone(),
            div: div_conv1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: conv1_output_zeropoint_converted,

            multiplier: self.multiplier_conv1,
        };
        conv1_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv1 {:?}", conv1_output);

        //relu1
        let relu1_cmp_res = relu4d_u8(&mut conv1_output, self.conv1_output_0);
        let relu1_output_fqvar = generate_fqvar4d(cs.clone(), conv1_output.clone());

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
        println!(
            "Number of constraints for Relu1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg_pool1

        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_output_fqvar = generate_fqvar4d(cs.clone(), avg_pool1_output.clone());
        let avg_pool1_circuit = AvgPoolCircuitLv3 {
            x: relu1_output_fqvar.clone(),
            y: avg_pool1_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        let conv2_output_fqvar = generate_fqvar4d(cs.clone(), conv2_output.clone());

        // y_0 and multipler_l1 are both constants.
        let mut conv2_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv2.len() {
            let m = (self.multiplier_conv2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv2_output_zeropoint_converted
                .push((self.conv2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD to reduce the number of constraints
        let conv2_circuit = ConvCircuitOp3 {
            x: avg_pool1_output_fqvar.clone(),
            conv_kernel: self.conv2_weights.clone(),
            y: conv2_output_fqvar.clone(),
            remainder: remainder_conv2.clone(),
            div: div_conv2.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: conv2_output_zeropoint_converted,

            multiplier: self.multiplier_conv2,
        };
        conv2_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //println!("conv2 {:?}", conv2_output);

        //relu2 layer

        let relu2_cmp_res = relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let relu2_output_fqvar = generate_fqvar4d(cs.clone(), conv2_output.clone());
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

        println!(
            "Number of constraints for Relu2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_output_fqvar = generate_fqvar4d(cs.clone(), avg_pool2_output.clone());
        let avg_pool2_circuit = AvgPoolCircuitLv3 {
            x: relu2_output_fqvar.clone(),
            y: avg_pool2_output_fqvar.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        let conv3_output_fqvar = generate_fqvar4d(cs.clone(), conv3_output.clone());

        // y_0 and multipler_l1 are both constants.
        let mut conv3_output_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_conv3.len() {
            let m = (self.multiplier_conv3[i] * (2u64.pow(M_EXP)) as f32) as u64;
            conv3_output_zeropoint_converted
                .push((self.conv3_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD to reduce the number of constraints
        let conv3_circuit = ConvCircuitOp3 {
            x: avg_pool2_output_fqvar.clone(),
            conv_kernel: self.conv3_weights,
            y: conv3_output_fqvar.clone(),
            remainder: remainder_conv3.clone(),
            div: div_conv3.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: conv3_output_zeropoint_converted,

            multiplier: self.multiplier_conv3,
        };
        conv3_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //relu3 layer

        let relu3_cmp_res = relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let relu3_output_fqvar = generate_fqvar4d(cs.clone(), conv3_output.clone());
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

        println!(
            "Number of constraints for Relu3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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
        //assume we only do inference on one image(batch size = 1)
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
                                            transformed_conv3_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        //assume we only do inference on one image
        let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
            &transformed_conv3_output[0],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[0],
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

        let fc1_circuit = FCCircuitOp3 {
            x: transformed_conv3_output_fqvar[0].clone(),
            l1_mat: self.fc1_weights.clone(),
            y: fc1_output_fqvar.clone(),
            remainder: remainder_fc1.clone(),
            div: div_fc1.clone(),
            x_0: self.conv3_output_0,
            l1_mat_0: self.fc1_weights_0,
            y_0: fc1_output0_converted,

            multiplier: self.multiplier_fc1.clone(),
        };
        fc1_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints FC1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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
        println!(
            "Number of constraints for Relu4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 5 :
        //FC2 -> output
        let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
                                            fc1_output.len()]; //batch size
        let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        let (remainder_fc2, div_fc2) = vec_mat_mul_with_remainder_u8(
            &fc1_output[0],
            fc2_weight_ref[..].as_ref(),
            &mut fc2_output[0],
            self.fc1_output_0,
            self.fc2_weights_0,
            self.fc2_output_0,
            &self.multiplier_fc2.clone(),
        );
        println!("z within circuit {:?}", fc2_output.clone());

        let fc2_output_fqvar = generate_fqvar(cs.clone(), fc2_output[0].clone());
        let mut fc2_output0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_fc2.len() {
            let m = (self.multiplier_fc2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            fc2_output0_converted.push((self.fc2_output_0 as u64 * 2u64.pow(M_EXP)) / m);
        }
        let fc2_circuit = FCCircuitOp3 {
            x: relu4_output_fqvar.clone(),
            l1_mat: self.fc2_weights.clone(),
            y: fc2_output_fqvar.clone(),

            remainder: remainder_fc2.clone(),
            div: div_fc2.clone(),

            x_0: self.fc1_output_0,
            l1_mat_0: self.fc2_weights_0,
            y_0: fc2_output0_converted,

            multiplier: self.multiplier_fc2.clone(),
        };
        fc2_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints FC2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        Ok(())
    }
}

#[derive(Clone)]
pub struct LeNetCircuitNaivePedersen {
    pub x: Vec<Vec<Vec<Vec<i8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv1_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub conv2_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub conv3_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub fc1_weights: Vec<Vec<i8>>,
    pub fc2_weights: Vec<Vec<i8>>,

    pub z: Vec<Vec<i8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}
impl ConstraintSynthesizer<Fq> for LeNetCircuitNaivePedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());
        //println!("x : {:?}", self.x);

        // x commitment
        let flattened_x3d: Vec<Vec<Vec<i8>>> = self.x.clone().into_iter().flatten().collect();
        let flattened_x2d: Vec<Vec<i8>> = flattened_x3d.into_iter().flatten().collect();
        let flattened_x1d: Vec<i8> = flattened_x2d.into_iter().flatten().collect();
        let flattened_x1d_u8 = flattened_x1d
            .iter()
            .map(|x| (*x as i8) as u8)
            .collect::<Vec<u8>>();

        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_x1d_u8.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        let output: Vec<Vec<i8>> = lenet_circuit_forward(
            self.x.clone(),
            self.conv1_weights.clone(),
            self.conv2_weights.clone(),
            self.conv3_weights.clone(),
            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
        );
        // z commitment
        let flattened_z1d: Vec<i8> = output.clone().into_iter().flatten().collect();

        let flattened_z1d_u8 = flattened_z1d
            .iter()
            .map(|x| (*x as i8) as u8)
            .collect::<Vec<u8>>();
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_z1d_u8.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {}",
            cs.num_constraints() - _cir_number
        );
        _cir_number = cs.num_constraints();

        //layer 1
        //conv1
        let mut conv1_output = vec![vec![vec![vec![0i8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv1_weights.len()]; //number of conv kernels
                                            self.x.len()]; //input (image) batch size
        vec_conv(&self.x, &self.conv1_weights, &mut conv1_output);
        let conv1_circuit = ConvCircuit {
            x: self.x.clone(),
            conv_kernel: self.conv1_weights.clone(),
            y: conv1_output.clone(),
        };
        conv1_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv1 {:?}", conv1_output);

        //relu1
        let relu_input = conv1_output.clone(); //save the input for verification
        relu4d(&mut conv1_output);

        let flattened_relu_input3d: Vec<Vec<Vec<i8>>> = relu_input.into_iter().flatten().collect();
        let flattened_relu_input2d: Vec<Vec<i8>> =
            flattened_relu_input3d.into_iter().flatten().collect();
        let flattened_relu_input1d: Vec<i8> =
            flattened_relu_input2d.into_iter().flatten().collect();

        let flattened_conv1_output3d: Vec<Vec<Vec<i8>>> =
            conv1_output.clone().into_iter().flatten().collect();
        let flattened_conv1_output2d: Vec<Vec<i8>> =
            flattened_conv1_output3d.into_iter().flatten().collect();
        let flattened_conv1_output1d: Vec<i8> =
            flattened_conv1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuit {
            y_in: flattened_relu_input1d,
            y_out: flattened_conv1_output1d,
        };
        relu1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg_pool1
        //let avg_pool1_output = avg_pool_scala_u8(&conv1_output, self.conv1_weights.len());
        let avg_pool1_output = avg_pool_scala(&conv1_output, 2);
        let avg_pool1_circuit = AvgPoolCircuit {
            x: conv1_output.clone(),
            y: avg_pool1_output.clone(),
            kernel_size: 2,
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 2 :
        //Conv2 -> relu -> AvgPool
        let mut conv2_output = vec![vec![vec![vec![0i8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                        self.conv2_weights.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size
        vec_conv(&avg_pool1_output, &self.conv2_weights, &mut conv2_output);
        let conv2_circuit = ConvCircuit {
            x: avg_pool1_output.clone(),
            conv_kernel: self.conv2_weights.clone(),
            y: conv2_output.clone(),
        };
        conv2_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //println!("conv2 {:?}", conv2_output);

        //relu2 layer
        let relu2_input = conv2_output.clone(); //save the input for verification
        relu4d(&mut conv2_output);
        let flattened_relu2_input3d: Vec<Vec<Vec<i8>>> =
            relu2_input.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<i8>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<i8> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_conv2_output3d: Vec<Vec<Vec<i8>>> =
            conv2_output.clone().into_iter().flatten().collect();
        let flattened_conv2_output2d: Vec<Vec<i8>> =
            flattened_conv2_output3d.into_iter().flatten().collect();
        let flattened_conv2_output1d: Vec<i8> =
            flattened_conv2_output2d.into_iter().flatten().collect();
        let relu2_circuit = ReLUCircuit {
            y_in: flattened_relu2_input1d,
            y_out: flattened_conv2_output1d,
        };
        relu2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let avg_pool2_output = avg_pool_scala(&conv2_output, 2);
        let avg_pool2_circuit = AvgPoolCircuit {
            x: conv2_output.clone(),
            y: avg_pool2_output.clone(),
            kernel_size: 2,
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 3 :
        //Conv3 -> relu -> reshape output for following FC layer
        let mut conv3_output = vec![vec![vec![vec![0i8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                            avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                            self.conv3_weights.len()]; //number of conv kernels
                                                                            avg_pool2_output.len()]; //input (image) batch size
                                                                                                     //conv3 layer
        vec_conv(&avg_pool2_output, &self.conv3_weights, &mut conv3_output);
        let conv3_circuit = ConvCircuit {
            x: avg_pool2_output.clone(),
            conv_kernel: self.conv3_weights,
            y: conv3_output.clone(),
        };
        conv3_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv3 : {:?}", conv3_output);

        //relu3 layer
        let relu3_input = conv3_output.clone(); //save the input for verification
        relu4d(&mut conv3_output);
        let flattened_relu3_input3d: Vec<Vec<Vec<i8>>> =
            relu3_input.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<i8>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<i8> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_conv3_output3d: Vec<Vec<Vec<i8>>> =
            conv3_output.clone().into_iter().flatten().collect();
        let flattened_conv3_output2d: Vec<Vec<i8>> =
            flattened_conv3_output3d.into_iter().flatten().collect();
        let flattened_conv3_output1d: Vec<i8> =
            flattened_conv3_output2d.into_iter().flatten().collect();
        let relu3_circuit = ReLUCircuit {
            y_in: flattened_relu3_input1d,
            y_out: flattened_conv3_output1d,
        };
        relu3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let mut transformed_conv3_output =
            vec![
                vec![
                    0i8;
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
                        counter += 1;
                    }
                }
            }
        }
        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0i8; self.fc1_weights.len()];  // channels
                                            transformed_conv3_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[i8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..transformed_conv3_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            vec_mat_mul(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
            );
            let fc1_circuit = FCCircuit {
                x: transformed_conv3_output[i].clone(),
                l1_mat: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;
            println!(
                "Number of constraints FC1 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        //relu4 layer
        let relu4_input = fc1_output.clone();
        relu2d(&mut fc1_output);
        let flattened_relu4_input1d: Vec<i8> = relu4_input.into_iter().flatten().collect();
        let flattened_relu4_output1d: Vec<i8> = fc1_output.clone().into_iter().flatten().collect();
        let relu4_circuit = ReLUCircuit {
            y_in: flattened_relu4_input1d,
            y_out: flattened_relu4_output1d,
        };
        relu4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 5 :
        //FC2 -> output
        let mut fc2_output = vec![vec![0i8; self.fc2_weights.len()]; // channels
                                            fc1_output.len()]; //batch size
        let fc2_weight_ref: Vec<&[i8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..fc1_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            vec_mat_mul(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
            );
            let fc2_circuit = FCCircuit {
                x: fc1_output[i].clone(),
                l1_mat: self.fc2_weights.clone(),
                y: fc2_output[i].clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;
            println!(
                "Number of constraints for FC2 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv1Pedersen {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub fc1_weights: Vec<Vec<u8>>,
    pub fc2_weights: Vec<Vec<u8>>,

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
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}

//NOTE : This is the baseline where FC does not use SIMD. Conv layer is very naive.
impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv1Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());
        //println!("x : {:?}", self.x);

        // x commitment
        let flattened_x3d: Vec<Vec<Vec<u8>>> = self.x.clone().into_iter().flatten().collect();
        let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
        let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_x1d.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        let output: Vec<Vec<u8>> = lenet_circuit_forward_u8(
            self.x.clone(),
            self.conv1_weights.clone(),
            self.conv2_weights.clone(),
            self.conv3_weights.clone(),
            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
            self.x_0,
            self.conv1_output_0,
            self.conv2_output_0,
            self.conv3_output_0,
            self.fc1_output_0,
            self.fc2_output_0,
            self.conv1_weights_0,
            self.conv2_weights_0,
            self.conv3_weights_0,
            self.fc1_weights_0,
            self.fc2_weights_0,
            self.multiplier_conv1.clone(),
            self.multiplier_conv2.clone(),
            self.multiplier_conv3.clone(),
            self.multiplier_fc1.clone(),
            self.multiplier_fc2.clone(),
        );
        // z commitment
        let flattened_z1d: Vec<u8> = output.clone().into_iter().flatten().collect();
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_z1d.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {}",
            cs.num_constraints() - _cir_number
        );
        _cir_number = cs.num_constraints();

        //layer 1
        //conv1
        let mut conv1_output = vec![vec![vec![vec![0u8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv1_weights.len()]; //number of conv kernels
                                            self.x.len()]; //input (image) batch size
        vec_conv_with_remainder_u8(
            &self.x,
            &self.conv1_weights,
            &mut conv1_output,
            self.x_0,
            self.conv1_weights_0,
            self.conv1_output_0,
            &self.multiplier_conv1,
        );
        let conv1_circuit = ConvCircuitU8 {
            x: self.x.clone(),
            conv_kernel: self.conv1_weights.clone(),
            y: conv1_output.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: self.conv1_output_0,

            multiplier: self.multiplier_conv1,
        };
        conv1_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv1 {:?}", conv1_output);

        //relu1
        let relu_input = conv1_output.clone(); //save the input for verification
        relu4d_u8(&mut conv1_output, self.conv1_output_0);

        let flattened_relu_input3d: Vec<Vec<Vec<u8>>> = relu_input.into_iter().flatten().collect();
        let flattened_relu_input2d: Vec<Vec<u8>> =
            flattened_relu_input3d.into_iter().flatten().collect();
        let flattened_relu_input1d: Vec<u8> =
            flattened_relu_input2d.into_iter().flatten().collect();

        let flattened_conv1_output3d: Vec<Vec<Vec<u8>>> =
            conv1_output.clone().into_iter().flatten().collect();
        let flattened_conv1_output2d: Vec<Vec<u8>> =
            flattened_conv1_output3d.into_iter().flatten().collect();
        let flattened_conv1_output1d: Vec<u8> =
            flattened_conv1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuitU8 {
            y_in: flattened_relu_input1d,
            y_out: flattened_conv1_output1d,
            y_zeropoint: self.conv1_output_0,
        };
        relu1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg_pool1
        //let avg_pool1_output = avg_pool_scala_u8(&conv1_output, self.conv1_weights.len());
        let (avg_pool1_output, _avg1_remainder) =
            avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_circuit = AvgPoolCircuitU8 {
            x: conv1_output.clone(),
            y: avg_pool1_output.clone(),
            kernel_size: 2,
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 2 :
        //Conv2 -> relu -> AvgPool
        let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                        self.conv2_weights.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size
        vec_conv_with_remainder_u8(
            &avg_pool1_output,
            &self.conv2_weights,
            &mut conv2_output,
            self.conv1_output_0,
            self.conv2_weights_0,
            self.conv2_output_0,
            &self.multiplier_conv2,
        );
        let conv2_circuit = ConvCircuitU8 {
            x: avg_pool1_output.clone(),
            conv_kernel: self.conv2_weights.clone(),
            y: conv2_output.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: self.conv2_output_0,

            multiplier: self.multiplier_conv2,
        };
        conv2_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //println!("conv2 {:?}", conv2_output);

        //relu2 layer
        let relu2_input = conv2_output.clone(); //save the input for verification
        relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let flattened_relu2_input3d: Vec<Vec<Vec<u8>>> =
            relu2_input.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<u8>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<u8> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_conv2_output3d: Vec<Vec<Vec<u8>>> =
            conv2_output.clone().into_iter().flatten().collect();
        let flattened_conv2_output2d: Vec<Vec<u8>> =
            flattened_conv2_output3d.into_iter().flatten().collect();
        let flattened_conv2_output1d: Vec<u8> =
            flattened_conv2_output2d.into_iter().flatten().collect();
        let relu2_circuit = ReLUCircuitU8 {
            y_in: flattened_relu2_input1d,
            y_out: flattened_conv2_output1d,
            y_zeropoint: self.conv2_output_0,
        };
        relu2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, _avg2_remainder) =
            avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_circuit = AvgPoolCircuitU8 {
            x: conv2_output.clone(),
            y: avg_pool2_output.clone(),
            kernel_size: 2,
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 3 :
        //Conv3 -> relu -> reshape output for following FC layer
        let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                            avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                            self.conv3_weights.len()]; //number of conv kernels
                                                                            avg_pool2_output.len()]; //input (image) batch size
                                                                                                     //conv3 layer
        vec_conv_with_remainder_u8(
            &avg_pool2_output,
            &self.conv3_weights,
            &mut conv3_output,
            self.conv2_output_0,
            self.conv3_weights_0,
            self.conv3_output_0,
            &self.multiplier_conv3,
        );
        let conv3_circuit = ConvCircuitU8 {
            x: avg_pool2_output.clone(),
            conv_kernel: self.conv3_weights,
            y: conv3_output.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: self.conv3_output_0,

            multiplier: self.multiplier_conv3,
        };
        conv3_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv3 : {:?}", conv3_output);

        //relu3 layer
        let relu3_input = conv3_output.clone(); //save the input for verification
        relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let flattened_relu3_input3d: Vec<Vec<Vec<u8>>> =
            relu3_input.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<u8>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<u8> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_conv3_output3d: Vec<Vec<Vec<u8>>> =
            conv3_output.clone().into_iter().flatten().collect();
        let flattened_conv3_output2d: Vec<Vec<u8>> =
            flattened_conv3_output3d.into_iter().flatten().collect();
        let flattened_conv3_output1d: Vec<u8> =
            flattened_conv3_output2d.into_iter().flatten().collect();
        let relu3_circuit = ReLUCircuitU8 {
            y_in: flattened_relu3_input1d,
            y_out: flattened_conv3_output1d,
            y_zeropoint: self.conv3_output_0,
        };
        relu3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let mut transformed_conv3_output =
            vec![
                vec![
                    0u8;
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
                        counter += 1;
                    }
                }
            }
        }
        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
                                            transformed_conv3_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..transformed_conv3_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let m_fc1 = self.multiplier_fc1.clone();
            vec_mat_mul_u8(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv3_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &m_fc1,
            );
            let fc1_circuit = FCCircuitU8 {
                x: transformed_conv3_output[i].clone(),
                l1_mat: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),

                x_0: self.conv3_output_0,
                l1_mat_0: self.fc1_weights_0,
                y_0: self.fc1_output_0,

                multiplier: self.multiplier_fc1.clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;
            println!(
                "Number of constraints FC1 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        //relu4 layer
        let relu4_input = fc1_output.clone();
        relu2d_u8(&mut fc1_output, self.fc1_output_0);
        let flattened_relu4_input1d: Vec<u8> = relu4_input.into_iter().flatten().collect();
        let flattened_relu4_output1d: Vec<u8> = fc1_output.clone().into_iter().flatten().collect();
        let relu4_circuit = ReLUCircuitU8 {
            y_in: flattened_relu4_input1d,
            y_out: flattened_relu4_output1d,
            y_zeropoint: self.fc1_output_0,
        };
        relu4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 5 :
        //FC2 -> output
        let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
                                            fc1_output.len()]; //batch size
        let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..fc1_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            vec_mat_mul_u8(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
                self.fc1_output_0,
                self.fc2_weights_0,
                self.fc2_output_0,
                &self.multiplier_fc2,
            );
            let fc2_circuit = FCCircuitU8 {
                x: fc1_output[i].clone(),
                l1_mat: self.fc2_weights.clone(),
                y: fc2_output[i].clone(),

                x_0: self.fc1_output_0,
                l1_mat_0: self.fc2_weights_0,
                y_0: self.fc2_output_0,

                multiplier: self.multiplier_fc2.clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;
            println!(
                "Number of constraints for FC2 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv2Pedersen {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub fc1_weights: Vec<Vec<u8>>,
    pub fc2_weights: Vec<Vec<u8>>,

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
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv2Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("LeNet is setup mode: {}", cs.is_in_setup_mode());
        //println!("{:?}", self.x);

        // x commitment
        let flattened_x3d: Vec<Vec<Vec<u8>>> = self.x.clone().into_iter().flatten().collect();
        let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
        let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_x1d.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        let output: Vec<Vec<u8>> = lenet_circuit_forward_u8(
            self.x.clone(),
            self.conv1_weights.clone(),
            self.conv2_weights.clone(),
            self.conv3_weights.clone(),
            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
            self.x_0,
            self.conv1_output_0,
            self.conv2_output_0,
            self.conv3_output_0,
            self.fc1_output_0,
            self.fc2_output_0,
            self.conv1_weights_0,
            self.conv2_weights_0,
            self.conv3_weights_0,
            self.fc1_weights_0,
            self.fc2_weights_0,
            self.multiplier_conv1.clone(),
            self.multiplier_conv2.clone(),
            self.multiplier_conv3.clone(),
            self.multiplier_fc1.clone(),
            self.multiplier_fc2.clone(),
        );
        // z commitment
        let flattened_z1d: Vec<u8> = output.clone().into_iter().flatten().collect();
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_z1d.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv1_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: self.x.clone(),
            conv_kernel: self.conv1_weights.clone(),
            y: conv1_output.clone(),
            remainder: remainder_conv1.clone(),
            div: div_conv1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: self.conv1_output_0,

            multiplier: self.multiplier_conv1,
        };
        conv1_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv1 {:?}", conv1_output);

        //relu1
        let relu_input = conv1_output.clone(); //save the input for verification
        relu4d_u8(&mut conv1_output, self.conv1_output_0);

        let flattened_relu_input3d: Vec<Vec<Vec<u8>>> = relu_input.into_iter().flatten().collect();
        let flattened_relu_input2d: Vec<Vec<u8>> =
            flattened_relu_input3d.into_iter().flatten().collect();
        let flattened_relu_input1d: Vec<u8> =
            flattened_relu_input2d.into_iter().flatten().collect();

        let flattened_conv1_output3d: Vec<Vec<Vec<u8>>> =
            conv1_output.clone().into_iter().flatten().collect();
        let flattened_conv1_output2d: Vec<Vec<u8>> =
            flattened_conv1_output3d.into_iter().flatten().collect();
        let flattened_conv1_output1d: Vec<u8> =
            flattened_conv1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuitU8 {
            y_in: flattened_relu_input1d,
            y_out: flattened_conv1_output1d,
            y_zeropoint: self.conv1_output_0,
        };
        relu1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg_pool1

        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            x: conv1_output.clone(),
            y: avg_pool1_output.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv2_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: avg_pool1_output.clone(),
            conv_kernel: self.conv2_weights.clone(),
            y: conv2_output.clone(),
            remainder: remainder_conv2.clone(),
            div: div_conv2.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: self.conv2_output_0,

            multiplier: self.multiplier_conv2,
        };
        conv2_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //println!("conv2 {:?}", conv2_output);

        //relu2 layer
        let relu2_input = conv2_output.clone(); //save the input for verification
        relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let flattened_relu2_input3d: Vec<Vec<Vec<u8>>> =
            relu2_input.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<u8>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<u8> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_conv2_output3d: Vec<Vec<Vec<u8>>> =
            conv2_output.clone().into_iter().flatten().collect();
        let flattened_conv2_output2d: Vec<Vec<u8>> =
            flattened_conv2_output3d.into_iter().flatten().collect();
        let flattened_conv2_output1d: Vec<u8> =
            flattened_conv2_output2d.into_iter().flatten().collect();
        let relu2_circuit = ReLUCircuitU8 {
            y_in: flattened_relu2_input1d,
            y_out: flattened_conv2_output1d,
            y_zeropoint: self.conv2_output_0,
        };
        relu2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            x: conv2_output.clone(),
            y: avg_pool2_output.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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

        let conv3_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: avg_pool2_output.clone(),
            conv_kernel: self.conv3_weights,
            y: conv3_output.clone(),
            remainder: remainder_conv3.clone(),
            div: div_conv3.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: self.conv3_output_0,

            multiplier: self.multiplier_conv3,
        };
        conv3_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv3 {:?}", conv3_output);

        //relu3 layer
        let relu3_input = conv3_output.clone(); //save the input for verification
        relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let flattened_relu3_input3d: Vec<Vec<Vec<u8>>> =
            relu3_input.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<u8>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<u8> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_conv3_output3d: Vec<Vec<Vec<u8>>> =
            conv3_output.clone().into_iter().flatten().collect();
        let flattened_conv3_output2d: Vec<Vec<u8>> =
            flattened_conv3_output3d.into_iter().flatten().collect();
        let flattened_conv3_output1d: Vec<u8> =
            flattened_conv3_output2d.into_iter().flatten().collect();
        let relu3_circuit = ReLUCircuitU8 {
            y_in: flattened_relu3_input1d,
            y_out: flattened_conv3_output1d,
            y_zeropoint: self.conv3_output_0,
        };
        relu3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let mut transformed_conv3_output =
            vec![
                vec![
                    0u8;
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
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
                                            transformed_conv3_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        //println!("FC1 {} == {}?", self.fc1_weights[0].len(), transformed_conv3_output[0].len());
        for i in 0..transformed_conv3_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv3_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc1_circuit = FCCircuitU8BitDecomposeOptimized {
                x: transformed_conv3_output[i].clone(),
                l1_mat: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv3_output_0,
                l1_mat_0: self.fc1_weights_0,
                y_0: self.fc1_output_0,

                multiplier: self.multiplier_fc1.clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC1 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        //relu4 layer
        let relu4_input = fc1_output.clone();
        relu2d_u8(&mut fc1_output, self.fc1_output_0);
        let flattened_relu4_input1d: Vec<u8> = relu4_input.into_iter().flatten().collect();
        let flattened_relu4_output1d: Vec<u8> = fc1_output.clone().into_iter().flatten().collect();
        let relu4_circuit = ReLUCircuitU8 {
            y_in: flattened_relu4_input1d,
            y_out: flattened_relu4_output1d,
            y_zeropoint: self.fc1_output_0,
        };
        relu4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 5 :
        //FC2 -> output
        let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
                                            fc1_output.len()]; //batch size
        let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..fc2_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc2, div_fc2) = vec_mat_mul_with_remainder_u8(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
                self.fc1_output_0,
                self.fc2_weights_0,
                self.fc2_output_0,
                &self.multiplier_fc2.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc2_circuit = FCCircuitU8BitDecomposeOptimized {
                x: fc1_output[i].clone(),
                l1_mat: self.fc2_weights.clone(),
                y: fc2_output[i].clone(),

                x_0: self.fc1_output_0,
                l1_mat_0: self.fc2_weights_0,
                y_0: self.fc2_output_0,

                multiplier: self.multiplier_fc2.clone(),

                remainder: remainder_fc2.clone(),
                div: div_fc2.clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC2 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        Ok(())
    }
}
