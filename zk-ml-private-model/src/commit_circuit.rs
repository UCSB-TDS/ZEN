use crate::pedersen_commit::*;
use crate::vanilla::*;
use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra::*;
use crypto_primitives::commitment::blake2s::constraints::*;
use crypto_primitives::commitment::blake2s::*;
use crypto_primitives::commitment::CommitmentGadget;
use r1cs_core::ConstraintSynthesizer;
use r1cs_core::ConstraintSystemRef;
use r1cs_core::SynthesisError;
use r1cs_std::alloc::AllocVar;
use r1cs_std::alloc::AllocationMode;
use r1cs_std::eq::EqGadget;
use r1cs_std::uint8::UInt8;

// ZK proved statements:
//  commit(data, open) = commitment
#[derive(Debug, Clone)]
pub(crate) struct MLCommitCircuitU8 {
    pub is_x: bool,
    pub data: Vec<u8>,
    pub open: Open,
    pub com: Commit,
}
// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for MLCommitCircuitU8 {
    // the following function generate the constraint system
    // for a
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        println!("is setup mode?: {}", cs.is_in_setup_mode());
        let _cs_no = cs.num_constraints();
        // step 1. Allocate Parameters for perdersen commitment
        let param = ();
        let param_var =
            <CommGadget as CommitmentGadget<Commitment, Fq>>::ParametersVar::new_witness(
                r1cs_core::ns!(cs, "gadget_parameters"),
                || Ok(&param),
            )
            .unwrap();
        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for parameters: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 2. Allocate inputs

        let input = if self.is_x {
            compress_x_u8(&self.data)
        } else {
            let tmp: Vec<[u8; 1]> = self.data.iter().map(|x| x.to_le_bytes()).collect();
            tmp[..].concat()
        };

        let mut input_var = vec![];
        for input_byte in input.iter() {
            input_var.push(UInt8::new_witness(cs.clone(), || Ok(*input_byte)).unwrap());
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for account: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 3. Allocate the opening
        let mut open_var = vec![];
        for r_byte in self.open.iter() {
            open_var.push(UInt8::new_witness(cs.clone(), || Ok(r_byte)).unwrap());
        }
        let open_var = RandomnessVar(open_var);

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for opening: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // step 4. Allocate the output
        let result_var = <CommGadget as CommitmentGadget<Commitment, Fq>>::commit(
            &param_var, &input_var, &open_var,
        )
        .unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for commitment: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // circuit to compare the commited value with supplied value
        for (i, com_byte) in self.com.iter().enumerate() {
            let tmp =
                UInt8::new_variable(cs.clone(), || Ok(com_byte), AllocationMode::Input).unwrap();
            tmp.enforce_equal(&result_var.0[i]).unwrap();
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for comparison: {}", _cs_no);

        #[cfg(debug_assertions)]
        println!("total cs for Commitment: {}", cs.num_constraints());
        Ok(())
    }
}

// =============================
// circuit
// =============================
//
// ZK proved statements:
//  commit(data, open) = commitment
#[derive(Debug, Clone)]
pub(crate) struct MLCommitCircuit {
    pub is_x: bool,
    pub data: Vec<i8>,
    pub open: Open,
    pub com: Commit,
}
// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for MLCommitCircuit {
    // the following function generate the constraint system
    // for a
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        println!("is setup mode?: {}", cs.is_in_setup_mode());
        let _cs_no = cs.num_constraints();
        // step 1. Allocate Parameters for perdersen commitment
        let param = ();
        let param_var =
            <CommGadget as CommitmentGadget<Commitment, Fq>>::ParametersVar::new_witness(
                r1cs_core::ns!(cs, "gadget_parameters"),
                || Ok(&param),
            )
            .unwrap();
        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for parameters: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 2. Allocate inputs

        let input = if self.is_x {
            compress_x(&self.data)
        } else {
            self.data
                .iter()
                .map(|x| (*x as i8) as u8)
                .collect::<Vec<u8>>()
        };

        let mut input_var = vec![];
        for input_byte in input.iter() {
            input_var.push(UInt8::new_witness(cs.clone(), || Ok(*input_byte)).unwrap());
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for account: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 3. Allocate the opening
        let mut open_var = vec![];
        for r_byte in self.open.iter() {
            open_var.push(UInt8::new_witness(cs.clone(), || Ok(r_byte)).unwrap());
        }
        let open_var = RandomnessVar(open_var);

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for opening: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // step 4. Allocate the output
        let result_var = <CommGadget as CommitmentGadget<Commitment, Fq>>::commit(
            &param_var, &input_var, &open_var,
        )
        .unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for commitment: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // circuit to compare the commited value with supplied value
        for (i, com_byte) in self.com.iter().enumerate() {
            let tmp =
                UInt8::new_variable(cs.clone(), || Ok(com_byte), AllocationMode::Input).unwrap();
            tmp.enforce_equal(&result_var.0[i]).unwrap();
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for comparison: {}", _cs_no);

        #[cfg(debug_assertions)]
        println!("total cs for Commitment: {}", cs.num_constraints());
        Ok(())
    }
}

// ZK proved statements:
//  commit(data, open) = commitment
#[derive(Clone)]
pub(crate) struct MLCommitCircuitU8Pedersen {
    pub data: Vec<u8>,
    pub param: PedersenParam,
    pub open: PedersenRandomness,
    pub com: PedersenCommitment,
}
// =============================
// constraints
// =============================
impl ConstraintSynthesizer<Fq> for MLCommitCircuitU8Pedersen {
    // the following function generate the constraint system
    // for a
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        println!("is setup mode?: {}", cs.is_in_setup_mode());
        let _cs_no = cs.num_constraints();
        // step 1. Allocate Parameters for perdersen commitment
        let param_var =
            PedersenParamVar::new_witness(r1cs_core::ns!(cs, "gadget_parameters"), || {
                Ok(&self.param)
            })
            .unwrap();
        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for parameters: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 2. Allocate inputs

        let mut input_var = vec![];
        for input_byte in self.data.iter() {
            input_var.push(UInt8::new_witness(cs.clone(), || Ok(*input_byte)).unwrap());
        }

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for account: {}", _cs_no);
        let _cs_no = cs.num_constraints();
        // step 3. Allocate the opening
        let open_var =
            PedersenRandomnessVar::new_witness(r1cs_core::ns!(cs, "gadget_randomness"), || {
                Ok(&self.open)
            })
            .unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for opening: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // step 4. Allocate the output
        let result_var = PedersenComSchemeVar::commit(&param_var, &input_var, &open_var).unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for commitment: {}", _cs_no);
        let _cs_no = cs.num_constraints();

        // circuit to compare the commited value with supplied value
        let commitment_var2 =
            PedersenCommitmentVar::new_input(r1cs_core::ns!(cs, "gadget_commitment"), || {
                Ok(self.com)
            })
            .unwrap();
        result_var.enforce_equal(&commitment_var2).unwrap();

        let _cs_no = cs.num_constraints() - _cs_no;
        #[cfg(debug_assertions)]
        println!("cs for comparison: {}", _cs_no);

        #[cfg(debug_assertions)]
        println!("total cs for Commitment: {}", cs.num_constraints());
        Ok(())
    }
}
