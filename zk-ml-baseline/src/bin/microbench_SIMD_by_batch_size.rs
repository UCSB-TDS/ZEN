use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use zk_ml_baseline::mul_circuit::FCCircuitU8BitDecomposeOptimized;
use zk_ml_baseline::mul_circuit_microbenchmark::FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark;
fn main() {
    // this file aims to provide microbenchmark on SIMD performance under different batch size and vector length.
    println!("start benchmarking\n\n\n");
    let cs = ConstraintSystem::<Fq>::new_ref();
    let mut _cir_number = cs.num_constraints();

    let vec_256 = vec![vec![10u8; 256]; 100];
    let vec_1024 = vec![vec![10u8; 1024]; 100];
    let vec_4096 = vec![vec![10u8; 4096]; 100];
    let vec_16384 = vec![vec![10u8; 16384]; 100];
    let input_256 = vec![10u8; 256];
    let input_1024 = vec![10u8; 1024];
    let input_4096 = vec![10u8; 4096];
    let input_16384 = vec![10u8; 16384];

    //just random numbers for benchmarking. The verification can not pass.
    let remainder = vec![10u32; 100];
    let div = vec![0u32; 100];
    let output = vec![110u8; 100];

    let zero_point: u8 = 1;
    let multiplier: f32 = 0.0001;

    _cir_number = cs.num_constraints();
    let fc_256_simd1 = FCCircuitU8BitDecomposeOptimized {
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

    let fc_256_simd2 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_256.clone(),
        l1_mat: vec_256.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 2,
    };

    let fc_256_simd3 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_256.clone(),
        l1_mat: vec_256.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 3,
    };

    let fc_256_simd4 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_256.clone(),
        l1_mat: vec_256.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 4,
    };

    let fc_1024_simd1 = FCCircuitU8BitDecomposeOptimized {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
    };

    let fc_1024_simd2 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 2,
    };

    let fc_1024_simd3 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 3,
    };

    let fc_1024_simd4 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_1024.clone(),
        l1_mat: vec_1024.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 4,
    };

    let fc_4096_simd1 = FCCircuitU8BitDecomposeOptimized {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
    };

    let fc_4096_simd2 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 2,
    };

    let fc_4096_simd3 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 3,
    };

    let fc_4096_simd4 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_4096.clone(),
        l1_mat: vec_4096.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 4,
    };

    let fc_16384_simd1 = FCCircuitU8BitDecomposeOptimized {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
    };

    let fc_16384_simd2 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 2,
    };

    let fc_16384_simd3 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 3,
    };

    let fc_16384_simd4 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
        x: input_16384.clone(),
        l1_mat: vec_16384.clone(),
        y: output.clone(),

        x_0: zero_point,
        l1_mat_0: zero_point,
        y_0: zero_point,

        remainder: remainder.clone(),
        div: div.clone(),
        multiplier: vec![multiplier; 100],
        batch_size: 4,
    };

    fc_256_simd1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_256_simd1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_simd2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_256_simd2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_simd3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_256_simd3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_simd4
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_256_simd4 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_simd1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_1024_simd1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_simd2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_1024_simd2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_simd3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_1024_simd3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_simd4
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_1024_simd4 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_simd1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_4096_simd1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_simd2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_4096_simd2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_simd3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_4096_simd3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_simd4
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_4096_simd4 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_simd1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_16384_simd1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_simd2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_16384_simd2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_simd3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_16384_simd3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_simd4
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_100_16384_simd4(4 vec SIMD on vector larger than 4096 could potentially bring error) {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    //below commented code is for testing the max capacity of FC dot product

    // let vec_50000 = vec![vec![100u8; 200000]; 100];
    // let input_50000 = vec![100u8; 200000];
    // let fc_50000_simd3 = FCCircuitU8BitDecomposeOptimizedSIMDMicrobenchmark {
    //     x: input_50000.clone(),
    //     l1_mat: vec_50000.clone(),
    //     y: output.clone(),

    //     x_0: zero_point,
    //     l1_mat_0: zero_point,
    //     y_0: zero_point,

    //     remainder: remainder.clone(),
    //     div: div.clone(),
    //     multiplier: multiplier.clone(),
    //     batch_size: 4,
    // };
    // fc_50000_simd3
    // .clone()
    // .generate_constraints(cs.clone())
    // .unwrap();
    // println!("fc_100_50000_simd3 {}", cs.num_constraints() - _cir_number);
    // let res = cs.is_satisfied().unwrap();
    // println!("are the constraints satisfied?: {}\n", res);
    // if !res {
    //     println!(
    //         "{:?} {} {:#?}",
    //         cs.constraint_names(),
    //         cs.num_constraints(),
    //         cs.which_is_unsatisfied().unwrap()
    //     );
    // }
}
