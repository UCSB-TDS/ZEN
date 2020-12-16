use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use zk_ml::mul_circuit::*;

fn main() {
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

    let fc_256_op3 = FCCircuitU8BitDecomposeOptimizedSIMD {
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

    let fc_1024_op3 = FCCircuitU8BitDecomposeOptimizedSIMD {
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

    let fc_4096_op3 = FCCircuitU8BitDecomposeOptimizedSIMD {
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

    let fc_16384_op3 = FCCircuitU8BitDecomposeOptimizedSIMD {
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

    println!("start benchmarking\n\n\n");
    let cs = ConstraintSystem::<Fq>::new_ref();
    let mut _cir_number = cs.num_constraints();

    //vec len is 256
    fc_256_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_256_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_op1.clone().generate_constraints(cs.clone()).unwrap();
    println!("fc_256_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_op2.clone().generate_constraints(cs.clone()).unwrap();
    println!("fc_256_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_256_op3.clone().generate_constraints(cs.clone()).unwrap();
    println!("fc_256_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    //vector len is 1024
    fc_1024_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_1024_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_1024_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_1024_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_1024_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_1024_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    //vector len is 4096
    fc_4096_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_4096_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_4096_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_4096_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_4096_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_4096_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    //vector len is 16384
    fc_16384_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_16384_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_16384_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_16384_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    fc_16384_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("fc_16384_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();
}
