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
use std::cmp::*;

pub fn convert_2d_vector_into_1d(vec: Vec<Vec<i8>>) -> Vec<i8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        res.extend(&vec[i]);
    }
    res
}

pub fn convert_4d_vector_into_1d(vec: Vec<Vec<Vec<Vec<i8>>>>) -> Vec<i8> {
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
pub struct LeNetCircuitNaivePedersen {
    pub x: Vec<Vec<Vec<Vec<i8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv1_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub conv1_open: PedersenRandomness,
    pub conv1_com_vec: Vec<PedersenCommitment>,
    pub conv2_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub conv2_open: PedersenRandomness,
    pub conv2_com_vec: Vec<PedersenCommitment>,
    pub conv3_weights: Vec<Vec<Vec<Vec<i8>>>>,
    pub conv3_open: PedersenRandomness,
    pub conv3_com_vec: Vec<PedersenCommitment>,
    pub fc1_weights: Vec<Vec<i8>>,
    pub fc1_open: PedersenRandomness,
    pub fc1_com_vec: Vec<PedersenCommitment>,
    pub fc2_weights: Vec<Vec<i8>>,
    pub fc2_open: PedersenRandomness,
    pub fc2_com_vec: Vec<PedersenCommitment>,

    pub z: Vec<Vec<i8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}
impl ConstraintSynthesizer<Fq> for LeNetCircuitNaivePedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        // // x commitment
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
        // // #[cfg(debug_assertion)]
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

        let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment

        let conv1_mat_1d = convert_4d_vector_into_1d(self.conv1_weights.clone());
        let num_of_commit_conv1 = conv1_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv1 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv1_mat_1d.len()) {
                tmp.push(conv1_mat_1d[j]);
            }
            let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

            let conv1_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp_u8.clone(),
                open: self.conv1_open.clone(),
                commit: self.conv1_com_vec[i],
            };
            conv1_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let conv2_mat_1d = convert_4d_vector_into_1d(self.conv2_weights.clone());
        let num_of_commit_conv2 = conv2_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv2 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv2_mat_1d.len()) {
                tmp.push(conv2_mat_1d[j]);
            }
            let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

            let conv2_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp_u8.clone(),
                open: self.conv2_open.clone(),
                commit: self.conv2_com_vec[i],
            };
            conv2_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let conv3_mat_1d = convert_4d_vector_into_1d(self.conv3_weights.clone());
        let num_of_commit_conv3 = conv3_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv3 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv3_mat_1d.len()) {
                tmp.push(conv3_mat_1d[j]);
            }
            let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

            let conv3_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp_u8.clone(),
                open: self.conv3_open.clone(),
                commit: self.conv3_com_vec[i],
            };
            conv3_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv3 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let l1_mat_1d = convert_2d_vector_into_1d(self.fc1_weights.clone());
        let num_of_commit_l1 = l1_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_l1 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, l1_mat_1d.len()) {
                tmp.push(l1_mat_1d[j]);
            }
            let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

            let l1_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp_u8.clone(),
                open: self.fc1_open.clone(),
                commit: self.fc1_com_vec[i],
            };
            l1_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for fc1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let l2_mat_1d = convert_2d_vector_into_1d(self.fc2_weights.clone());
        let num_of_commit_l2 = l2_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_l2 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, l2_mat_1d.len()) {
                tmp.push(l2_mat_1d[j]);
            }
            let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

            let l2_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp_u8.clone(),
                open: self.fc2_open.clone(),
                commit: self.fc2_com_vec[i],
            };
            l2_com_circuit.generate_constraints(cs.clone())?;
        }

        println!(
            "Number of constraints for fc2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        _cir_number = cs.num_constraints();

        //layer 1
        //conv1
        // let mut conv1_output = vec![vec![vec![vec![0i8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
        //                                     self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
        //                                     self.conv1_weights.len()]; //number of conv kernels
        //                                     self.x.len()]; //input (image) batch size
        // vec_conv(&self.x, &self.conv1_weights, &mut conv1_output);
        // let conv1_circuit = ConvCircuit {
        //     x: self.x.clone(),
        //     conv_kernel: self.conv1_weights.clone(),
        //     y: conv1_output.clone(),
        // };
        // conv1_circuit.generate_constraints(cs.clone())?;
        // // #[cfg(debug_assertion)]
        // println!(
        //     "Number of constraints for Conv1 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();
        // //println!("conv1 {:?}", conv1_output);

        // //relu1
        // let relu_input = conv1_output.clone(); //save the input for verification
        // relu4d(&mut conv1_output);

        // let flattened_relu_input3d: Vec<Vec<Vec<i8>>> = relu_input.into_iter().flatten().collect();
        // let flattened_relu_input2d: Vec<Vec<i8>> =
        //     flattened_relu_input3d.into_iter().flatten().collect();
        // let flattened_relu_input1d: Vec<i8> =
        //     flattened_relu_input2d.into_iter().flatten().collect();

        // let flattened_conv1_output3d: Vec<Vec<Vec<i8>>> =
        //     conv1_output.clone().into_iter().flatten().collect();
        // let flattened_conv1_output2d: Vec<Vec<i8>> =
        //     flattened_conv1_output3d.into_iter().flatten().collect();
        // let flattened_conv1_output1d: Vec<i8> =
        //     flattened_conv1_output2d.into_iter().flatten().collect();

        // let relu1_circuit = ReLUCircuit {
        //     y_in: flattened_relu_input1d,
        //     y_out: flattened_conv1_output1d,
        // };
        // relu1_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for Relu1 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //avg_pool1
        // //let avg_pool1_output = avg_pool_scala_u8(&conv1_output, self.conv1_weights.len());
        // let avg_pool1_output = avg_pool_scala(&conv1_output, 2);
        // let avg_pool1_circuit = AvgPoolCircuit {
        //     x: conv1_output.clone(),
        //     y: avg_pool1_output.clone(),
        //     kernel_size: 2,
        // };
        // avg_pool1_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for AvgPool1 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //layer 2 :
        // //Conv2 -> relu -> AvgPool
        // let mut conv2_output = vec![vec![vec![vec![0i8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        //                                                                 avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        //                                                                 self.conv2_weights.len()]; //number of conv kernels
        //                                                                 avg_pool1_output.len()]; //input (image) batch size
        // vec_conv(&avg_pool1_output, &self.conv2_weights, &mut conv2_output);
        // let conv2_circuit = ConvCircuit {
        //     x: avg_pool1_output.clone(),
        //     conv_kernel: self.conv2_weights.clone(),
        //     y: conv2_output.clone(),
        // };
        // conv2_circuit.generate_constraints(cs.clone())?;
        // // #[cfg(debug_assertion)]
        // println!(
        //     "Number of constraints for Conv2 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //println!("conv2 {:?}", conv2_output);

        // //relu2 layer
        // let relu2_input = conv2_output.clone(); //save the input for verification
        // relu4d(&mut conv2_output);
        // let flattened_relu2_input3d: Vec<Vec<Vec<i8>>> =
        //     relu2_input.into_iter().flatten().collect();
        // let flattened_relu2_input2d: Vec<Vec<i8>> =
        //     flattened_relu2_input3d.into_iter().flatten().collect();
        // let flattened_relu2_input1d: Vec<i8> =
        //     flattened_relu2_input2d.into_iter().flatten().collect();

        // let flattened_conv2_output3d: Vec<Vec<Vec<i8>>> =
        //     conv2_output.clone().into_iter().flatten().collect();
        // let flattened_conv2_output2d: Vec<Vec<i8>> =
        //     flattened_conv2_output3d.into_iter().flatten().collect();
        // let flattened_conv2_output1d: Vec<i8> =
        //     flattened_conv2_output2d.into_iter().flatten().collect();
        // let relu2_circuit = ReLUCircuit {
        //     y_in: flattened_relu2_input1d,
        //     y_out: flattened_conv2_output1d,
        // };
        // relu2_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for Relu2 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //avg pool2 layer
        // //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        // let avg_pool2_output = avg_pool_scala(&conv2_output, 2);
        // let avg_pool2_circuit = AvgPoolCircuit {
        //     x: conv2_output.clone(),
        //     y: avg_pool2_output.clone(),
        //     kernel_size: 2,
        // };
        // avg_pool2_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for AvgPool2 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //layer 3 :
        // //Conv3 -> relu -> reshape output for following FC layer
        // let mut conv3_output = vec![vec![vec![vec![0i8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        //                                                                     avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        //                                                                     self.conv3_weights.len()]; //number of conv kernels
        //                                                                     avg_pool2_output.len()]; //input (image) batch size
        //                                                                                              //conv3 layer
        // vec_conv(&avg_pool2_output, &self.conv3_weights, &mut conv3_output);
        // let conv3_circuit = ConvCircuit {
        //     x: avg_pool2_output.clone(),
        //     conv_kernel: self.conv3_weights,
        //     y: conv3_output.clone(),
        // };
        // conv3_circuit.generate_constraints(cs.clone())?;
        // // #[cfg(debug_assertion)]
        // println!(
        //     "Number of constraints for Conv3 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();
        // //println!("conv3 : {:?}", conv3_output);

        // //relu3 layer
        // let relu3_input = conv3_output.clone(); //save the input for verification
        // relu4d(&mut conv3_output);
        // let flattened_relu3_input3d: Vec<Vec<Vec<i8>>> =
        //     relu3_input.into_iter().flatten().collect();
        // let flattened_relu3_input2d: Vec<Vec<i8>> =
        //     flattened_relu3_input3d.into_iter().flatten().collect();
        // let flattened_relu3_input1d: Vec<i8> =
        //     flattened_relu3_input2d.into_iter().flatten().collect();

        // let flattened_conv3_output3d: Vec<Vec<Vec<i8>>> =
        //     conv3_output.clone().into_iter().flatten().collect();
        // let flattened_conv3_output2d: Vec<Vec<i8>> =
        //     flattened_conv3_output3d.into_iter().flatten().collect();
        // let flattened_conv3_output1d: Vec<i8> =
        //     flattened_conv3_output2d.into_iter().flatten().collect();
        // let relu3_circuit = ReLUCircuit {
        //     y_in: flattened_relu3_input1d,
        //     y_out: flattened_conv3_output1d,
        // };
        // relu3_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for Relu3 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // let mut transformed_conv3_output =
        //     vec![
        //         vec![
        //             0i8;
        //             conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
        //         ];
        //         conv3_output.len()
        //     ];
        // for i in 0..conv3_output.len() {
        //     let mut counter = 0;
        //     for j in 0..conv3_output[0].len() {
        //         for p in 0..conv3_output[0][0].len() {
        //             for q in 0..conv3_output[0][0][0].len() {
        //                 transformed_conv3_output[i][counter] = conv3_output[i][j][p][q];
        //                 counter += 1;
        //             }
        //         }
        //     }
        // }
        //layer 4 :
        //FC1 -> relu
        // let mut fc1_output = vec![vec![0i8; self.fc1_weights.len()];  // channels
        //                                     transformed_conv3_output.len()]; //batch size
        // let fc1_weight_ref: Vec<&[i8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        // for i in 0..transformed_conv3_output.len() {
        //     //iterate through each image in the batch
        //     //in the zkp nn system, we feed one image in each batch to reduce the overhead.
        //     vec_mat_mul(
        //         &transformed_conv3_output[i],
        //         fc1_weight_ref[..].as_ref(),
        //         &mut fc1_output[i],
        //     );
        //     let fc1_circuit = FCCircuit {
        //         x: transformed_conv3_output[i].clone(),
        //         l1_mat: self.fc1_weights.clone(),
        //         y: fc1_output[i].clone(),
        //     };
        //     fc1_circuit.generate_constraints(cs.clone())?;
        //     println!(
        //         "Number of constraints FC1 {} accumulated constraints {}",
        //         cs.num_constraints() - _cir_number,
        //         cs.num_constraints()
        //     );
        //     _cir_number = cs.num_constraints();
        // }

        // //relu4 layer
        // let relu4_input = fc1_output.clone();
        // relu2d(&mut fc1_output);
        // let flattened_relu4_input1d: Vec<i8> = relu4_input.into_iter().flatten().collect();
        // let flattened_relu4_output1d: Vec<i8> = fc1_output.clone().into_iter().flatten().collect();
        // let relu4_circuit = ReLUCircuit {
        //     y_in: flattened_relu4_input1d,
        //     y_out: flattened_relu4_output1d,
        // };
        // relu4_circuit.generate_constraints(cs.clone())?;
        // println!(
        //     "Number of constraints for Relu4 {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // _cir_number = cs.num_constraints();

        // //layer 5 :
        // //FC2 -> output
        // let mut fc2_output = vec![vec![0i8; self.fc2_weights.len()]; // channels
        //                                     fc1_output.len()]; //batch size
        // let fc2_weight_ref: Vec<&[i8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        // for i in 0..fc1_output.len() {
        //     //iterate through each image in the batch
        //     //in the zkp nn system, we feed one image in each batch to reduce the overhead.
        //     vec_mat_mul(
        //         &fc1_output[i],
        //         fc2_weight_ref[..].as_ref(),
        //         &mut fc2_output[i],
        //     );
        //     let fc2_circuit = FCCircuit {
        //         x: fc1_output[i].clone(),
        //         l1_mat: self.fc2_weights.clone(),
        //         y: fc2_output[i].clone(),
        //     };
        //     fc2_circuit.generate_constraints(cs.clone())?;
        //     println!(
        //         "Number of constraints for FC2 {} accumulated constraints {}",
        //         cs.num_constraints() - _cir_number,
        //         cs.num_constraints()
        //     );
        //     _cir_number = cs.num_constraints();
        // }

        Ok(())
    }
}
