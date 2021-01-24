use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;
use zk_ml_baseline::mul_circuit::*;

fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
}

fn main() {
    println!("start benchmarking\n\n\n");

    //just use random numbers for microbenchmark. verification correctness is ignored.
    let vec_256 = vec![vec![100u8; 256]; 100];
    let vec_1024 = vec![vec![100u8; 1024]; 100];
    let vec_4096 = vec![vec![100u8; 4096]; 100];
    let vec_16384 = vec![vec![100u8; 16384]; 100];
    let vec_256_i8 = vec![vec![100i8; 256]; 100];
    let vec_1024_i8 = vec![vec![100i8; 1024]; 100];
    let vec_4096_i8 = vec![vec![100i8; 4096]; 100];
    let vec_16384_i8 = vec![vec![100i8; 16384]; 100];

    let input_256 = vec![100u8; 256];
    let input_1024 = vec![100u8; 1024];
    let input_4096 = vec![100u8; 4096];
    let input_16384 = vec![100u8; 16384];
    let input_256_i8 = vec![100i8; 256];
    let input_1024_i8 = vec![100i8; 1024];
    let input_4096_i8 = vec![100i8; 4096];
    let input_16384_i8 = vec![100i8; 16384];

    //just random numbers for benchmarking. The verification can not pass.
    let remainder = vec![10u32; 100];
    let div = vec![2u32; 100];
    let output = vec![0u8; 100];
    let output_i8 = vec![0i8; 100];

    let zero_point: u8 = 80;
    let multiplier: f32 = 0.1;

    let cs11 = ConstraintSystem::<Fq>::new_ref();
    let cs12 = ConstraintSystem::<Fq>::new_ref();
    let cs13 = ConstraintSystem::<Fq>::new_ref();
    let cs14 = ConstraintSystem::<Fq>::new_ref();
    let cs21 = ConstraintSystem::<Fq>::new_ref();
    let cs22 = ConstraintSystem::<Fq>::new_ref();
    let cs23 = ConstraintSystem::<Fq>::new_ref();
    let cs24 = ConstraintSystem::<Fq>::new_ref();
    let cs31 = ConstraintSystem::<Fq>::new_ref();
    let cs32 = ConstraintSystem::<Fq>::new_ref();
    let cs33 = ConstraintSystem::<Fq>::new_ref();
    let cs34 = ConstraintSystem::<Fq>::new_ref();
    let cs41 = ConstraintSystem::<Fq>::new_ref();
    let cs42 = ConstraintSystem::<Fq>::new_ref();
    let cs43 = ConstraintSystem::<Fq>::new_ref();
    let cs44 = ConstraintSystem::<Fq>::new_ref();

    //vector dot product size is 256
    let fc_256_naive = FCCircuit {
        x: input_256_i8.clone(),
        l1_mat: vec_256_i8.clone(),
        y: output_i8.clone(),
    };

    let fc_256_op1 = FCCircuitU8 {
        x: input_256.clone(),
        l1_mat: vec_256.clone(),
        y: output.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };

    let fc_256_op2 = FCCircuitU8BitDecomposeOptimized {
        x: input_256.clone(),
        l1_mat: vec_256.clone(),
        y: output.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };

    let input_256_var = generate_fqvar(cs14.clone(), input_256.clone());
    let output_var = generate_fqvar(cs14.clone(), output.clone());
    let fc_256_op3 = FCCircuitOp3 {
        x: input_256_var.clone(),
        l1_mat: vec_256.clone(),
        y: output_var.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: vec![zero_point as u64; 100],
        multiplier: vec![multiplier; 100],
    };

    //vector dot product size is 1024
    let fc_1024_naive = FCCircuit {
        x: input_1024_i8.clone(),
        l1_mat: vec_1024_i8.clone(),
        y: output_i8.clone(),
    };

    let fc_1024_op1 = FCCircuitU8 {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };

    let fc_1024_op2 = FCCircuitU8BitDecomposeOptimized {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };
    let input_1024_var = generate_fqvar(cs24.clone(), input_1024.clone());

    let fc_1024_op3 = FCCircuitOp3 {
        x: input_1024_var.clone(),
        l1_mat: vec_1024.clone(),
        y: output_var.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: vec![zero_point as u64; 100],

        multiplier: vec![multiplier; 100],
    };

    //vector dot product size is 4096
    let fc_4096_naive = FCCircuit {
        x: input_4096_i8.clone(),
        l1_mat: vec_4096_i8.clone(),
        y: output_i8.clone(),
    };

    let fc_4096_op1 = FCCircuitU8 {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };

    let fc_4096_op2 = FCCircuitU8BitDecomposeOptimized {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };
    let input_4096_var = generate_fqvar(cs34.clone(), input_4096.clone());

    let fc_4096_op3 = FCCircuitOp3 {
        x: input_4096_var.clone(),
        l1_mat: vec_4096.clone(),
        y: output_var.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: vec![zero_point as u64; 100],

        multiplier: vec![multiplier; 100],
    };

    //vector dot product size is 16384
    let fc_16384_naive = FCCircuit {
        x: input_16384_i8.clone(),
        l1_mat: vec_16384_i8.clone(),
        y: output_i8.clone(),
    };

    let fc_16384_op1 = FCCircuitU8 {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };

    let fc_16384_op2 = FCCircuitU8BitDecomposeOptimized {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        multiplier: vec![multiplier; 100],
    };
    let input_16384_var = generate_fqvar(cs44.clone(), input_16384.clone());

    let fc_16384_op3 = FCCircuitOp3 {
        x: input_16384_var.clone(),
        l1_mat: vec_16384.clone(),
        y: output_var.clone(),
        remainder: remainder.clone(),
        div: div.clone(),
        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: vec![zero_point as u64; 100],

        multiplier: vec![multiplier; 100],
    };

    //vec len is 256
    // fc_256_naive
    //     .clone()
    //     .generate_constraints(cs11.clone())
    //     .unwrap();
    // println!("fc_256_naive {}", cs11.num_constraints());
    // cs11.inline_all_lcs();

    // fc_256_op1
    //     .clone()
    //     .generate_constraints(cs12.clone())
    //     .unwrap();
    // println!("fc_256_op1 {}", cs12.num_constraints());
    // cs12.inline_all_lcs();

    // fc_256_op2
    //     .clone()
    //     .generate_constraints(cs13.clone())
    //     .unwrap();
    // println!("fc_256_op2 {}", cs13.num_constraints());
    // cs13.inline_all_lcs();

    // fc_256_op3
    //     .clone()
    //     .generate_constraints(cs14.clone())
    //     .unwrap();
    // println!("fc_256_op3 {}", cs14.num_constraints());
    // cs14.inline_all_lcs();

    //vector len is 1024
    // fc_1024_naive
    //     .clone()
    //     .generate_constraints(cs21.clone())
    //     .unwrap();
    // println!("fc_1024_naive {}", cs21.num_constraints());
    // cs21.inline_all_lcs();

    fc_1024_op1
        .clone()
        .generate_constraints(cs22.clone())
        .unwrap();
    println!("fc_1024_op1 {}", cs22.num_constraints());
    cs22.inline_all_lcs();

    fc_1024_op2
        .clone()
        .generate_constraints(cs23.clone())
        .unwrap();
    println!("fc_1024_op2 {}", cs23.num_constraints());
    cs23.inline_all_lcs();

    fc_1024_op3
        .clone()
        .generate_constraints(cs24.clone())
        .unwrap();
    println!("fc_1024_op3 {}", cs24.num_constraints());
    cs24.inline_all_lcs();

    //vector len is 4096
    fc_4096_naive
        .clone()
        .generate_constraints(cs31.clone())
        .unwrap();
    println!("fc_4096_naive {}", cs31.num_constraints());
    cs31.inline_all_lcs();

    fc_4096_op1
        .clone()
        .generate_constraints(cs32.clone())
        .unwrap();
    println!("fc_4096_op1 {}", cs32.num_constraints());
    cs32.inline_all_lcs();

    fc_4096_op2
        .clone()
        .generate_constraints(cs33.clone())
        .unwrap();
    println!("fc_4096_op2 {}", cs33.num_constraints());
    cs33.inline_all_lcs();

    fc_4096_op3
        .clone()
        .generate_constraints(cs34.clone())
        .unwrap();
    println!("fc_4096_op3 {}", cs34.num_constraints());
    cs34.inline_all_lcs();

    //vector len is 16384
    fc_16384_naive
        .clone()
        .generate_constraints(cs41.clone())
        .unwrap();
    println!("fc_16384_naive {}", cs41.num_constraints());
    cs41.inline_all_lcs();

    fc_16384_op1
        .clone()
        .generate_constraints(cs42.clone())
        .unwrap();
    println!("fc_16384_op1 {}", cs42.num_constraints());
    cs42.inline_all_lcs();

    fc_16384_op2
        .clone()
        .generate_constraints(cs43.clone())
        .unwrap();
    println!("fc_16384_op2 {}", cs43.num_constraints());
    cs43.inline_all_lcs();

    fc_16384_op3
        .clone()
        .generate_constraints(cs44.clone())
        .unwrap();
    println!("fc_16384_op3 {}", cs44.num_constraints());
    cs44.inline_all_lcs();
}
