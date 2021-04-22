use crate::mul_circuit::*;
use crate::pedersen_commit::*;
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

#[derive(Clone)]
pub struct FullCircuitPedersen {
    pub x: X,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub l1: Vec<Vec<i8>>,
    pub l1_open: PedersenRandomness,
    pub l1_com_vec: Vec<PedersenCommitment>,
    pub l2: Vec<Vec<i8>>,
    pub l2_open: PedersenRandomness,
    pub l2_com_vec: Vec<PedersenCommitment>,
    pub z: Z,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
}

impl ConstraintSynthesizer<Fq> for FullCircuitPedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("is setup mode: {}", cs.is_in_setup_mode());
        let x_u8 = self.x.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

        let z_u8 = self.z.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

        // // x commitment
        // let x_com_circuit = PedersenComCircuit {
        //     param: self.params.clone(),
        //     input: x_u8.clone(),
        //     open: self.x_open,
        //     commit: self.x_com,
        // };
        // x_com_circuit.generate_constraints(cs.clone())?;
        let _cir_number = cs.num_constraints();
        // // #[cfg(debug_assertion)]
        // println!("Number of constraints for x commitment {}", _cir_number);

        // // z commitment
        // let z_com_circuit = PedersenComCircuit {
        //     param: self.params.clone(),
        //     input: z_u8.clone(),
        //     open: self.z_open,
        //     commit: self.z_com,
        // };
        // z_com_circuit.generate_constraints(cs.clone())?;
        // let _cir_number = cs.num_constraints() - _cir_number;
        // // #[cfg(debug_assertion)]
        // println!("Number of constraints for z commitment {}", _cir_number);
        // let mut _cir_number = cs.num_constraints();
        // let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment

        // let l1_mat_1d = convert_2d_vector_into_1d(self.l1.clone());
        // let num_of_commit_l1 = l1_mat_1d.len() / len_per_commit + 1;
        // for i in 0..num_of_commit_l1 {
        //     let mut tmp = Vec::new();
        //     for j in i * len_per_commit..min((i + 1) * len_per_commit, l1_mat_1d.len()) {
        //         tmp.push(l1_mat_1d[j]);
        //     }
        //     let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

        //     let l1_com_circuit = PedersenComCircuit {
        //         param: self.params.clone(),
        //         input: tmp_u8.clone(),
        //         open: self.l1_open.clone(),
        //         commit: self.l1_com_vec[i],
        //     };
        //     l1_com_circuit.generate_constraints(cs.clone())?;
        // }
        // println!(
        //     "Number of constraints for l1 layer commitment {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // let _cir_number = cs.num_constraints();

        // let l2_mat_1d = convert_2d_vector_into_1d(self.l2.clone());
        // let num_of_commit_l2 = l2_mat_1d.len() / len_per_commit + 1;
        // for i in 0..num_of_commit_l2 {
        //     let mut tmp = Vec::new();
        //     for j in i * len_per_commit..min((i + 1) * len_per_commit, l2_mat_1d.len()) {
        //         tmp.push(l2_mat_1d[j]);
        //     }
        //     let tmp_u8 = tmp.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

        //     let l2_com_circuit = PedersenComCircuit {
        //         param: self.params.clone(),
        //         input: tmp_u8.clone(),
        //         open: self.l2_open.clone(),
        //         commit: self.l2_com_vec[i],
        //     };
        //     l2_com_circuit.generate_constraints(cs.clone())?;
        // }

        // println!(
        //     "Number of constraints for l2 layer commitment {} accumulated constraints {}",
        //     cs.num_constraints() - _cir_number,
        //     cs.num_constraints()
        // );
        // layer 1
        let mut y = vec![0i8; M];
        let l1_mat_ref: Vec<&[i8]> = self.l1.iter().map(|x| x.as_ref()).collect();
        vec_mat_mul(&self.x, l1_mat_ref[..].as_ref(), &mut y);
        let mut y_out = y.clone();
        relu(&mut y_out);

        let l1_circuit = FCCircuit {
            x: self.x,
            l1_mat: self.l1,
            y: y.clone(),
        };
        l1_circuit.generate_constraints(cs.clone())?;
        let _cir_number = cs.num_constraints() - _cir_number;
        // #[cfg(debug_assertion)]
        println!("Number of constraints for L1 {}", _cir_number);
        let _cir_number = cs.num_constraints();

        let relu_circuit = ReLUCircuit {
            y_in: y,
            y_out: y_out.clone(),
        };
        relu_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Relu {}",
            cs.num_constraints() - _cir_number
        );
        let _cir_number = cs.num_constraints();
        let l2_circuit = FCCircuit {
            x: y_out,
            l1_mat: self.l2,
            y: self.z,
        };
        l2_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for L2 {}",
            cs.num_constraints() - _cir_number
        );
        println!(
            "Total number of FullCircuit inference constraints {}",
            cs.num_constraints()
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct FullCircuitOpLv1Pedersen {
    pub x: Vec<u8>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub l1: Vec<Vec<u8>>,
    pub l2: Vec<Vec<u8>>,
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

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv1Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());

        // x commitment
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

        // layer 1
        let mut y = vec![0u8; M];
        let l1_mat_ref: Vec<&[u8]> = self.l1.iter().map(|x| x.as_ref()).collect();
        // vec_mat_mul_u8(&self.x, l1_mat_ref[..].as_ref(), &mut y,
        //                 self.x_0, self.l1_mat_0, self.y_0, self.multiplier_l1);

        vec_mat_mul_with_remainder_u8(
            &self.x,
            l1_mat_ref[..].as_ref(),
            &mut y,
            self.x_0,
            self.l1_mat_0,
            self.y_0,
            &self.multiplier_l1,
        );
        let mut y_out = y.clone();
        relu_u8(&mut y_out, self.y_0);

        let l1_circuit = FCCircuitU8 {
            x: self.x,
            l1_mat: self.l1,
            y: y.clone(),

            x_0: self.x_0,
            l1_mat_0: self.l1_mat_0,
            y_0: self.y_0,

            multiplier: self.multiplier_l1.clone(),
        };
        l1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let relu_circuit = ReLUCircuitU8 {
            y_in: y,
            y_out: y_out.clone(),

            y_zeropoint: self.y_0,
        };
        relu_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for ReLU1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        let l2_mat_ref: Vec<&[u8]> = self.l2.iter().map(|x| x.as_ref()).collect();
        let mut zz = self.z.clone();
        vec_mat_mul_with_remainder_u8(
            &y_out,
            l2_mat_ref[..].as_ref(),
            &mut zz,
            self.y_0,
            self.l2_mat_0,
            self.z_0,
            &self.multiplier_l2,
        );
        let l2_circuit = FCCircuitU8 {
            x: y_out,
            l1_mat: self.l2,
            y: self.z,

            x_0: self.y_0,
            l1_mat_0: self.l2_mat_0,
            y_0: self.z_0,

            multiplier: self.multiplier_l2.clone(),
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
pub struct FullCircuitOpLv2Pedersen {
    pub x: Vec<u8>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub l1: Vec<Vec<u8>>,
    pub l2: Vec<Vec<u8>>,
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

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv2Pedersen {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FullCircuitU8 is setup mode: {}", cs.is_in_setup_mode());

        // x commitment
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

        // layer 1
        let mut y = vec![0u8; M];
        let l1_mat_ref: Vec<&[u8]> = self.l1.iter().map(|x| x.as_ref()).collect();
        // vec_mat_mul_u8(&self.x, l1_mat_ref[..].as_ref(), &mut y,
        //                 self.x_0, self.l1_mat_0, self.y_0, self.multiplier_l1);

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
        relu_u8(&mut y_out, self.y_0);

        let l1_circuit = FCCircuitU8BitDecomposeOptimized {
            x: self.x,
            l1_mat: self.l1,
            y: y.clone(),
            remainder: remainder1.clone(),
            div: div1.clone(),

            x_0: self.x_0,
            l1_mat_0: self.l1_mat_0,
            y_0: self.y_0,

            multiplier: self.multiplier_l1,
        };
        l1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for FC1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let relu_circuit = ReLUCircuitU8 {
            y_in: y,
            y_out: y_out.clone(),

            y_zeropoint: self.y_0,
        };
        relu_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for ReLU1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        let l2_mat_ref: Vec<&[u8]> = self.l2.iter().map(|x| x.as_ref()).collect();
        let mut zz = self.z.clone();
        let (remainder2, div2) = vec_mat_mul_with_remainder_u8(
            &y_out,
            l2_mat_ref[..].as_ref(),
            &mut zz,
            self.y_0,
            self.l2_mat_0,
            self.z_0,
            &self.multiplier_l2,
        );
        let l2_circuit = FCCircuitU8BitDecomposeOptimized {
            x: y_out,
            l1_mat: self.l2,
            y: self.z,
            remainder: remainder2.clone(),
            div: div2.clone(),

            x_0: self.y_0,
            l1_mat_0: self.l2_mat_0,
            y_0: self.z_0,

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
