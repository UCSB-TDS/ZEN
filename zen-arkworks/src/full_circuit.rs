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

#[derive(Clone)]
pub struct FullCircuitOpLv3PedersenClassification {
    pub x: Vec<u8>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub l1: Vec<Vec<u8>>,
    pub l1_open: PedersenRandomness,
    pub l1_com_vec: Vec<PedersenCommitment>,
    pub l2: Vec<Vec<u8>>,
    pub l2_open: PedersenRandomness,
    pub l2_com_vec: Vec<PedersenCommitment>,
    pub z: Vec<u8>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub argmax_res: usize,

    pub x_0: u8,
    pub y_0: u8,
    pub z_0: u8,
    pub l1_mat_0: u8,
    pub l2_mat_0: u8,
    pub multiplier_l1: Vec<f32>,
    pub multiplier_l2: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv3PedersenClassification {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let full_circuit = FullCircuitOpLv3Pedersen {
            params: self.params.clone(),
            x: self.x.clone(),
            x_com: self.x_com.clone(),
            x_open: self.x_open,
            l1: self.l1,
            l1_open: self.l1_open,
            l1_com_vec: self.l1_com_vec,
            l2: self.l2,
            l2_open: self.l2_open,
            l2_com_vec: self.l2_com_vec,
            z: self.z.clone(),
            z_com: self.z_com.clone(),
            z_open: self.z_open,

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
pub struct FullCircuitOpLv3Pedersen {
    pub x: Vec<u8>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub l1: Vec<Vec<u8>>,
    pub l1_open: PedersenRandomness,
    pub l1_com_vec: Vec<PedersenCommitment>,
    pub l2: Vec<Vec<u8>>,
    pub l2_open: PedersenRandomness,
    pub l2_com_vec: Vec<PedersenCommitment>,
    pub z: Vec<u8>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,

    pub x_0: u8,
    pub y_0: u8,
    pub z_0: u8,
    pub l1_mat_0: u8,
    pub l2_mat_0: u8,
    pub multiplier_l1: Vec<f32>,
    pub multiplier_l2: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv3Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        //x commitment
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: self.x.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        // z commitment
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: self.z.clone(),
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
        let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment

        let l1_mat_1d = convert_2d_vector_into_1d(self.l1.clone());
        let num_of_commit_l1 = l1_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_l1 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, l1_mat_1d.len()) {
                tmp.push(l1_mat_1d[j]);
            }
            let l1_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.l1_open.clone(),
                commit: self.l1_com_vec[i],
            };
            l1_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for l1 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let l2_mat_1d = convert_2d_vector_into_1d(self.l2.clone());
        let num_of_commit_l2 = l2_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_l2 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, l2_mat_1d.len()) {
                tmp.push(l2_mat_1d[j]);
            }
            let l2_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.l2_open.clone(),
                commit: self.l2_com_vec[i],
            };
            l2_com_circuit.generate_constraints(cs.clone())?;
        }

        println!(
            "Number of constraints for l2 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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