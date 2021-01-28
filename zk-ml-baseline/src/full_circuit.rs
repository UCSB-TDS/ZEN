use crate::mul_circuit::*;
use crate::pedersen_commit::*;
use crate::relu_circuit::*;
use crate::vanilla::*;
use crate::*;
use algebra::ed_on_bls12_381::*;
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

#[derive(Clone)]
pub struct FullCircuitOpLv3Pedersen {
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

impl ConstraintSynthesizer<Fq> for FullCircuitOpLv3Pedersen {
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

        // y_0 and multipler_l1 are both constants.
        let mut y0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_l1.len() {
            let m = (self.multiplier_l1[i] * (2u64.pow(M_EXP)) as f32) as u64;
            y0_converted.push((self.y_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        let l1_circuit = FCCircuitOp3 {
            x: x_fqvar,
            l1_mat: self.l1,
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

        // z_0 and multipler_l2 are both constants.
        let z_fqvar = generate_fqvar(cs.clone(), zz.clone());
        let mut z0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier_l2.len() {
            let m = (self.multiplier_l2[i] * (2u64.pow(M_EXP)) as f32) as u64;
            z0_converted.push((self.z_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        let l2_circuit = FCCircuitOp3 {
            x: relu1_output_var.clone(),
            l1_mat: self.l2,
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
pub struct FullCircuitPedersen {
    pub x: X,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub l1: L1Mat,
    pub l2: L2Mat,
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

        // x commitment
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: x_u8.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        // z commitment
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: z_u8.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        let _cir_number = cs.num_constraints() - _cir_number;
        // #[cfg(debug_assertion)]
        println!("Number of constraints for z commitment {}", _cir_number);
        let _cir_number = cs.num_constraints();

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
