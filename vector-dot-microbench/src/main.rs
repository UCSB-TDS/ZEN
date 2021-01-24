use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use num_traits::Pow;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use std::cmp;
use std::ops::*;
use r1cs_core::*;
use std::time::Instant;
use groth16::*;

#[derive(Clone)]
pub struct ConstantConstantDotProductCircuit {
    pub input: Vec<u8>,
    pub weight: Vec<u8>,
}
impl ConstraintSynthesizer<Fq> for ConstantConstantDotProductCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _no_cs = cs.num_constraints();
            if self.input.len() != self.weight.len() {
                panic!("scala mul: length not equal");
            }

            let mut tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();

            for k in 0..self.input.len() {
                let w: Fq = self.weight[k].into();
                let w_const = FpVar::Constant(w);
                let i: Fq = self.input[k].into();
                let i_var = FpVar::Constant(i);
                tmp += i_var.mul(w_const.clone());
            }
            Ok(())
    }

}

#[derive(Clone)]
pub struct ConstantWireDotProductCircuit {
    pub input: Vec<u8>,
    pub weight: Vec<u8>,
}
impl ConstraintSynthesizer<Fq> for ConstantWireDotProductCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _no_cs = cs.num_constraints();
            if self.input.len() != self.weight.len() {
                panic!("scala mul: length not equal");
            }

            let mut tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();

            for k in 0..self.input.len() {
                let w: Fq = self.weight[k].into();
                let w_const = FpVar::Constant(w);
                let i: Fq = self.input[k].into();
                let i_var = FpVar::new_witness(r1cs_core::ns!(cs, "input"), || Ok(i)).unwrap();
                tmp += i_var * w_const;
            }
            Ok(())
    }

}

#[derive(Clone)]
pub struct WireWireDotProductCircuit {
    pub input: Vec<u8>,
    pub weight: Vec<u8>,
}
impl ConstraintSynthesizer<Fq> for WireWireDotProductCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _no_cs = cs.num_constraints();
            if self.input.len() != self.weight.len() {
                panic!("scala mul: length not equal");
            }

            let mut tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();

            for k in 0..self.input.len() {
                let w: Fq = self.weight[k].into();
                let w_const = FpVar::new_witness(r1cs_core::ns!(cs, format!("weight {}", k)), || Ok(w)).unwrap();
                let i: Fq = self.input[k].into();
                let i_var = FpVar::new_witness(r1cs_core::ns!(cs, format!("input {}", k)), || Ok(i)).unwrap();
                tmp += i_var.mul(w_const);
            }
            Ok(())
    }

}


#[derive(Clone)]
pub struct MixedWireDotProductCircuit {
    pub input: Vec<u8>,
    pub weight: Vec<u8>,
    pub constant_ratio : f32,
}
impl ConstraintSynthesizer<Fq> for MixedWireDotProductCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let _no_cs = cs.num_constraints();
            if self.input.len() != self.weight.len() {
                panic!("scala mul: length not equal");
            }

            let mut tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();

            for k in 0..self.input.len() {
                if((k as f32) < (self.input.len() as f32 * self.constant_ratio)){
                    let w: Fq = self.weight[k].into();
                    let w_const = FpVar::Constant(w);
                    let i: Fq = self.input[k].into();
                    let i_var = FpVar::new_witness(r1cs_core::ns!(cs, format!("input {}", k)), || Ok(i)).unwrap();
                    tmp += i_var.mul(w_const);
                }else{
                    let w: Fq = self.weight[k].into();
                    let w_const = FpVar::new_witness(r1cs_core::ns!(cs, format!("weight {}", k)), || Ok(w)).unwrap();
                    let i: Fq = self.input[k].into();
                    let i_var = FpVar::new_witness(r1cs_core::ns!(cs, format!("input {}", k)), || Ok(i)).unwrap();
                    tmp += i_var.mul(w_const);
                }

            }
            Ok(())
    }

}

fn main() {

    for i in (1..20).step_by(2) {

        let mut rng = rand::thread_rng();
        let len = 10000; 
        let x = vec![1u8; len];
        let y = vec![1u8; len];
        let ratio : f32 = (i as f32 / 20.0f32);
        let cs3 = ConstraintSystem::<Fq>::new_ref();
        println!("constant * wire ratio {}", ratio);
        let wire_wire_dot_product_circuit = MixedWireDotProductCircuit{
            input: x.clone(), 
            weight: y.clone(),
            constant_ratio: ratio,
        };
        let begin = Instant::now();

        // pre-computed parameters
        let param =
            generate_random_parameters::<algebra::Bls12_381, _, _>(wire_wire_dot_product_circuit.clone(), &mut rng)
                .unwrap();
        let end = Instant::now();
        println!("setup time {:?}", end.duration_since(begin));


        let pvk = prepare_verifying_key(&param.vk);

        // prover
        let begin = Instant::now();
        let proof = create_random_proof(wire_wire_dot_product_circuit.clone(), &param, &mut rng).unwrap();
        let end = Instant::now();
        println!("prove time {:?}", end.duration_since(begin));
        wire_wire_dot_product_circuit.clone()
        .generate_constraints(cs3.clone())
        .unwrap();
        println!("num of constraints {}\n\n\n\n", cs3.num_constraints());


    }
    // for i in (1..20).step_by(2) {
    //     let mut rng = rand::thread_rng();
    //     let len = 1000 * i ; 
    //     let x = vec![1u8; len];
    //     let y = vec![1u8; len];
    //     let cs1 = ConstraintSystem::<Fq>::new_ref();
    //     println!("constant * constant vector len {}", len);
    //     let constant_wire_dot_product_circuit = ConstantConstantDotProductCircuit{
    //         input: x.clone(), 
    //         weight: y.clone(),
    //     };
    //     let begin = Instant::now();

    //     // pre-computed parameters
    //     let param =
    //         generate_random_parameters::<algebra::Bls12_381, _, _>(constant_wire_dot_product_circuit.clone(), &mut rng)
    //             .unwrap();
    //     let end = Instant::now();
    //     println!("setup time {:?}", end.duration_since(begin));


    //     let pvk = prepare_verifying_key(&param.vk);

    //     // prover
    //     let begin = Instant::now();
    //     let proof = create_random_proof(constant_wire_dot_product_circuit.clone(), &param, &mut rng).unwrap();
    //     let end = Instant::now();
    //     println!("prove time {:?}", end.duration_since(begin));
    //     constant_wire_dot_product_circuit.clone()
    //                             .generate_constraints(cs1.clone())
    //                             .unwrap();
    //     println!("num of constraints {}\n\n\n\n", cs1.num_constraints());

    


    //     let mut rng = rand::thread_rng();
    //     let len = 1000 * i ; 
    //     let x = vec![1u8; len];
    //     let y = vec![1u8; len];
    //     let cs2 = ConstraintSystem::<Fq>::new_ref();
    //     println!("constant * wire vector len {}", len);
    //     let constant_wire_dot_product_circuit = ConstantWireDotProductCircuit{
    //         input: x.clone(), 
    //         weight: y.clone(),
    //     };
    //     let begin = Instant::now();

    //     // pre-computed parameters
    //     let param =
    //         generate_random_parameters::<algebra::Bls12_381, _, _>(constant_wire_dot_product_circuit.clone(), &mut rng)
    //             .unwrap();
    //     let end = Instant::now();
    //     println!("setup time {:?}", end.duration_since(begin));


    //     let pvk = prepare_verifying_key(&param.vk);

    //     // prover
    //     let begin = Instant::now();
    //     let proof = create_random_proof(constant_wire_dot_product_circuit.clone(), &param, &mut rng).unwrap();
    //     let end = Instant::now();
    //     println!("prove time {:?}", end.duration_since(begin));
    //     constant_wire_dot_product_circuit.clone()
    //     .generate_constraints(cs2.clone())
    //     .unwrap();
    //     println!("num of constraints {}\n\n\n\n", cs2.num_constraints());


    


    //     let mut rng = rand::thread_rng();
    //     let len = 1000 * i; 
    //     let x = vec![1u8; len];
    //     let y = vec![1u8; len];
    //     let cs3 = ConstraintSystem::<Fq>::new_ref();
    //     println!("wire * wire vector len {}", len);
    //     let wire_wire_dot_product_circuit = WireWireDotProductCircuit{
    //         input: x.clone(), 
    //         weight: y.clone(),
    //     };
    //     let begin = Instant::now();

    //     // pre-computed parameters
    //     let param =
    //         generate_random_parameters::<algebra::Bls12_381, _, _>(wire_wire_dot_product_circuit.clone(), &mut rng)
    //             .unwrap();
    //     let end = Instant::now();
    //     println!("setup time {:?}", end.duration_since(begin));


    //     let pvk = prepare_verifying_key(&param.vk);

    //     // prover
    //     let begin = Instant::now();
    //     let proof = create_random_proof(wire_wire_dot_product_circuit.clone(), &param, &mut rng).unwrap();
    //     let end = Instant::now();
    //     println!("prove time {:?}", end.duration_since(begin));
    //     wire_wire_dot_product_circuit.clone()
    //     .generate_constraints(cs3.clone())
    //     .unwrap();
    //     println!("num of constraints {}\n\n\n\n", cs3.num_constraints());


    // }
}
