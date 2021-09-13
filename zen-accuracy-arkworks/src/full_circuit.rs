use crate::argmax_circuit::*;
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
use ark_sponge::poseidon::PoseidonParameters;

use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::R1CSVar;
use ark_r1cs_std::ToBitsGadget;


pub fn convert_2d_vector_into_1d(vec: Vec<Vec<u8>>) -> Vec<u8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        res.extend(&vec[i]);
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

fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
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





fn generate_fqvar_input(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_input(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
}


fn generate_fqvar_input2D(cs: ConstraintSystemRef<Fq>, input: Vec<Vec<u8>>) -> Vec<Vec<FqVar>> {
    let zero_var = FpVar::<Fq>::Constant(Fq::zero());
    let mut res: Vec<Vec<FqVar>> = vec![vec![zero_var; input[0].len()]; input.len()];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            let fq: Fq = input[i][j].into();
            let tmp = FpVar::<Fq>::new_input(ark_relations::ns!(cs, "tmp"), || Ok(fq)).unwrap();
            res[i][j] = tmp;
        }
    }
    res
}

pub fn convert_1d_vector_into_fq(vec: Vec<u8>) -> Vec<Fq> {
    let mut res = vec![Fq::zero(); vec.len()];
    for i in 0..vec.len() {
        let tmp: Fq = vec[i].into();
        res[i] = tmp;
    }
    res
}



#[derive(Clone)]
pub struct FullCircuitClassificationAccuracy {
    pub params: SPNGParam,
    pub l1: Vec<Vec<u8>>,
    pub l1_squeeze: SPNGOutput,
    pub l2: Vec<Vec<u8>>,
    pub l2_squeeze: SPNGOutput,

    pub x: Vec<Vec<u8>>, // public dataset for inference accuracy calculation
    pub true_labels: Vec<u8>,
    pub accuracy_result: Vec<u8>, //1 stands for prediction is correct, 0 is wrong prediction.
    pub accuracy_squeeze: SPNGOutput,

    pub x_0: u8,
    pub y_0: u8,
    pub z_0: u8,
    pub l1_mat_0: u8,
    pub l2_mat_0: u8,
    pub multiplier_l1: Vec<f32>,
    pub multiplier_l2: Vec<f32>,
}


impl ConstraintSynthesizer<Fq> for FullCircuitClassificationAccuracy {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let mut _cir_number = cs.num_constraints();

        let l1_mat_1d = convert_2d_vector_into_1d(self.l1.clone());
        let l2_mat_1d = convert_2d_vector_into_1d(self.l2.clone());

        let l1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l1_mat_1d.clone(),
            output: self.l1_squeeze.clone()
        };
        l1_com_circuit.generate_constraints(cs.clone())?;
        println!("Number of constraints for l1 commitment {}", cs.num_constraints() - _cir_number);
        _cir_number = cs.num_constraints();

        let l2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l2_mat_1d.clone(),
            output: self.l2_squeeze.clone()
        };
        l2_com_circuit.generate_constraints(cs.clone())?;
        println!("Number of constraints for l2 commitment {}", cs.num_constraints() - _cir_number);
        _cir_number = cs.num_constraints();


        //commit prediction accuracy result vector. we will sum it together in the final circuit.
        //we do not reveal this vector of 1 or 0 which leak information about ZEN's inference results on each instance in the public dataset.
        let accuracy_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: self.accuracy_result.clone(),
            output: self.accuracy_squeeze.clone()
        };
        accuracy_com_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for accuracy result commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        // image dataset and inference accuracy are both public input

        let x_input = generate_fqvar_input2D(cs.clone(), self.x.clone());
        let true_label_input = generate_fqvar_input(cs.clone(), self.true_labels.clone());

        let len = self.x.clone().len();
        let correct_prediction = Boolean::<Fq>::Constant(true);
        let wrong_prediction = Boolean::<Fq>::Constant(false);
        for i in 0..len {
            //inference image one by one.
            let mut y = vec![0u8; self.l1.len()];
            let l1_mat_ref: Vec<&[u8]> = self.l1.iter().map(|x| x.as_ref()).collect();
            let x_fqvar_input = x_input[i].clone();

            let (remainder1, div1) = vec_mat_mul_with_remainder_u8(
                &self.x[i],
                l1_mat_ref[..].as_ref(),
                &mut y,
                self.x_0,
                self.l1_mat_0,
                self.y_0,
                &self.multiplier_l1.clone(),
            );

            let mut y_out = y.clone();
            let cmp_res = relu_u8(&mut y_out, self.y_0);

            let y_fqvar = generate_fqvar(cs.clone(), y.clone());
            let l1_fqvar_input = generate_fqvar_witness2D(cs.clone(), self.l1.clone());
            // y_0 and multiplier_l1 are both constants.
            let mut y0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_l1.len() {
                let m = (self.multiplier_l1[i] * (2u64.pow(M_EXP)) as f32) as u64;
                y0_converted.push((self.y_0 as u64 * 2u64.pow(M_EXP)) / m);
            }
            let l1_circuit = FCCircuitOp3 {
                x: x_fqvar_input,
                l1_mat: l1_fqvar_input,
                y: y_fqvar.clone(),
                remainder: remainder1.clone(),
                div: div1.clone(),

                x_0: self.x_0,
                l1_mat_0: self.l1_mat_0,
                y_0: y0_converted,

                multiplier: self.multiplier_l1.clone(),
            };

            l1_circuit.generate_constraints(cs.clone())?;

            _cir_number = cs.num_constraints();

            let relu1_output_var = generate_fqvar(cs.clone(), y_out.clone());
            let relu_circuit = ReLUCircuitOp3 {
                y_in: y_fqvar.clone(),
                y_out: relu1_output_var.clone(),
                y_zeropoint: self.y_0,
                cmp_res: cmp_res.clone(),
            };
            relu_circuit.generate_constraints(cs.clone())?;

            _cir_number = cs.num_constraints();
            let l2_mat_ref: Vec<&[u8]> = self.l2.iter().map(|x| x.as_ref()).collect();
            let mut zz = vec![0u8; self.l2.len()];
            let (remainder2, div2) = vec_mat_mul_with_remainder_u8(
                &y_out,
                l2_mat_ref[..].as_ref(),
                &mut zz,
                self.y_0,
                self.l2_mat_0,
                self.z_0,
                &self.multiplier_l2.clone(),
            );

            // z_0 and multiplier_l2 are both constants.
            let z_fqvar = generate_fqvar(cs.clone(), zz.clone());
            let l2_fqvar_input = generate_fqvar_witness2D(cs.clone(), self.l2.clone());

            let mut z0_converted: Vec<u64> = Vec::new();
            for i in 0..self.multiplier_l2.len() {
                let m = (self.multiplier_l2[i] * (2u64.pow(M_EXP)) as f32) as u64;
                z0_converted.push((self.z_0 as u64 * 2u64.pow(M_EXP)) / m);
            }

            let l2_circuit = FCCircuitOp3 {
                x: relu1_output_var.clone(),
                l1_mat: l2_fqvar_input,
                y: z_fqvar.clone(),
                remainder: remainder2.clone(),
                div: div2.clone(),

                x_0: self.y_0,
                l1_mat_0: self.l2_mat_0,
                y_0: z0_converted,

                multiplier: self.multiplier_l2.clone(),
            };
            l2_circuit.generate_constraints(cs.clone())?;

            _cir_number = cs.num_constraints();

            let classification_res = argmax_u8(zz.clone()) as usize;

            //println!("z {:?}\n argmax {}", zz.clone(), classification_res);
            //this is the index/label
            let classification_res_fq: Fq = (classification_res as u64).into();
            let classification_res_var =
                FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "classification"), || {
                    Ok(classification_res_fq)
                })
                .unwrap();

            //this is the value
            let classification_max: Fq = zz[classification_res].into();
            let classification_max_var =
                FpVar::<Fq>::new_witness(ark_relations::ns!(cs, "classification max"), || {
                    Ok(classification_max)
                })
                .unwrap();

            let argmax_circuit = ArgmaxCircuit {
                input: z_fqvar.clone(),
                argmax_res: classification_max_var.clone(),
            };

            argmax_circuit.generate_constraints(cs.clone())?;

            //add final constraints on the prediction correctness
            //println!("true label {:?}\n\n classification label {:?}", true_label_input[i].to_bits_le().unwrap().value().unwrap(), classification_res_var.to_bits_le().unwrap().value().unwrap());
            let is_prediction_correct = true_label_input[i].is_eq(&classification_res_var).unwrap();
            //println!("prediction_correct {:?}\n\ncorrect_prediction {:?}", is_prediction_correct.value().unwrap(), correct_prediction.value().unwrap());

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
            "Total number of FullCircuit inference constraints {}",
            cs.num_constraints()
        );

        Ok(())
    }
}


#[derive(Clone)]
pub struct FullCircuitOpLv3PoseidonClassification {
    pub params: SPNGParam,

    pub x: Vec<u8>,
    pub x_squeeze: SPNGOutput,
    pub l1: Vec<Vec<u8>>,
    pub l1_squeeze: SPNGOutput,
    pub l2: Vec<Vec<u8>>,
    pub l2_squeeze: SPNGOutput,
    pub z: Vec<u8>,
    pub z_squeeze: SPNGOutput,

    pub x_0: u8,
    pub y_0: u8,
    pub z_0: u8,
    pub l1_mat_0: u8,
    pub l2_mat_0: u8,
    pub multiplier_l1: Vec<f32>,
    pub multiplier_l2: Vec<f32>,

    pub argmax_res: usize
}

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv3PoseidonClassification {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let full_circuit = FullCircuitOpLv3Poseidon {
            params: self.params.clone(),
            x: self.x.clone(),
            x_squeeze: self.x_squeeze.clone(),
            l1: self.l1,
            l1_squeeze: self.l1_squeeze.clone(),
            l2: self.l2,
            l2_squeeze: self.l2_squeeze.clone(),
            z: self.z.clone(),
            z_squeeze: self.z_squeeze.clone(),
            

            x_0: self.x_0,
            y_0: self.y_0,
            z_0: self.z_0,
            l1_mat_0: self.l1_mat_0,
            l2_mat_0: self.l2_mat_0,
            multiplier_l1: self.multiplier_l1.clone(),
            multiplier_l2: self.multiplier_l2.clone(),
        };

        let argmax_circuit = ArgmaxCircuitU8 {
            input: self.z.clone(),
            argmax_res: self.argmax_res.clone(),
        };

        full_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();
        argmax_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();

        Ok(())
    }
}

#[derive(Clone)]
pub struct FullCircuitOpLv3Poseidon {
    pub params: SPNGParam,

    pub x: Vec<u8>,
    pub x_squeeze: SPNGOutput,
    pub l1: Vec<Vec<u8>>,
    pub l1_squeeze: SPNGOutput,
    pub l2: Vec<Vec<u8>>,
    pub l2_squeeze: SPNGOutput,
    pub z: Vec<u8>,
    pub z_squeeze: SPNGOutput,

    pub x_0: u8,
    pub y_0: u8,
    pub z_0: u8,
    pub l1_mat_0: u8,
    pub l2_mat_0: u8,
    pub multiplier_l1: Vec<f32>,
    pub multiplier_l2: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv3Poseidon {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        //x commitment
        let x_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: self.x.clone(),
            output: self.x_squeeze.clone()
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        println!("Number of constraints for x commitment {}", _cir_number);

        let l1_mat_1d = convert_2d_vector_into_1d(self.l1.clone());
        let l2_mat_1d = convert_2d_vector_into_1d(self.l2.clone());

        let l1_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l1_mat_1d.clone(),
            output: self.l1_squeeze.clone()
        };
        l1_com_circuit.generate_constraints(cs.clone())?;
        println!("Number of constraints for l1 commitment {}", cs.num_constraints() - _cir_number);
        _cir_number = cs.num_constraints();

        let l2_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: l2_mat_1d.clone(),
            output: self.l2_squeeze.clone()
        };
        l2_com_circuit.generate_constraints(cs.clone())?;
        println!("Number of constraints for l2 commitment {}", cs.num_constraints() - _cir_number);
        _cir_number = cs.num_constraints();

        let z_com_circuit = SPNGCircuit {
            param: self.params.clone(),
            input: self.z.clone(),
            output: self.z_squeeze.clone()
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        println!("Number of constraints for z commitment {}", cs.num_constraints() - _cir_number);
        _cir_number = cs.num_constraints();


        // layer 1
        let mut y = vec![0u8; self.l1.len()];
        let l1_mat_ref: Vec<&[u8]> = self.l1.iter().map(|x| x.as_ref()).collect();
        let x_fqvar = generate_fqvar(cs.clone(), self.x.clone());

        let (remainder1, div1) = vec_mat_mul_with_remainder_u8(
            &self.x,
            l1_mat_ref[..].as_ref(),
            &mut y,
            self.x_0,
            self.l1_mat_0,
            self.y_0,
            &self.multiplier_l1,
        );

        let mut y_out = y.clone();
        let cmp_res = relu_u8(&mut y_out, self.y_0);

        let y_fqvar = generate_fqvar(cs.clone(), y.clone());
        let l1_fqvar_input = generate_fqvar_witness2D(cs.clone(), self.l1.clone());
        // y_0 and multiplier_l1 are both constants.
        let mut y0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_l1.len() {
            let m = (self.multiplier_l1[i] * (2u64.pow(M_EXP)) as f32) as u64;
            y0_converted.push((self.y_0 as u64 * 2u64.pow(M_EXP)) / m);
        }
        let l1_circuit = FCCircuitOp3 {
            x: x_fqvar,
            l1_mat: l1_fqvar_input,
            y: y_fqvar.clone(),
            remainder: remainder1.clone(),
            div: div1.clone(),

            x_0: self.x_0,
            l1_mat_0: self.l1_mat_0,
            y_0: y0_converted,

            multiplier: self.multiplier_l1,
        };

        l1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let relu1_output_var = generate_fqvar(cs.clone(), y_out.clone());
        let relu_circuit = ReLUCircuitOp3 {
            y_in: y_fqvar.clone(),
            y_out: relu1_output_var.clone(),
            y_zeropoint: self.y_0,
            cmp_res: cmp_res.clone(),
        };
        relu_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for ReLU1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        _cir_number = cs.num_constraints();
        let l2_mat_ref: Vec<&[u8]> = self.l2.iter().map(|x| x.as_ref()).collect();
        let mut zz = vec![0u8; self.l2.len()];
        let (remainder2, div2) = vec_mat_mul_with_remainder_u8(
            &y_out,
            l2_mat_ref[..].as_ref(),
            &mut zz,
            self.y_0,
            self.l2_mat_0,
            self.z_0,
            &self.multiplier_l2,
        );

        // z_0 and multiplier_l2 are both constants.
        let z_fqvar = generate_fqvar(cs.clone(), zz.clone());
        let l2_fqvar_input = generate_fqvar_witness2D(cs.clone(), self.l2.clone());

        let mut z0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_l2.len() {
            let m = (self.multiplier_l2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            z0_converted.push((self.z_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        let l2_circuit = FCCircuitOp3 {
            x: relu1_output_var.clone(),
            l1_mat: l2_fqvar_input,
            y: z_fqvar.clone(),
            remainder: remainder2.clone(),
            div: div2.clone(),

            x_0: self.y_0,
            l1_mat_0: self.l2_mat_0,
            y_0: z0_converted,

            multiplier: self.multiplier_l2,
        };
        l2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        println!(
            "Total number of FullCircuit inference constraints {}",
            cs.num_constraints()
        );


        Ok(())
    }
}