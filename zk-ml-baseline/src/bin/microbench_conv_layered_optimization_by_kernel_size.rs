use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;
use zk_ml_baseline::conv_circuit::*;

use algebra_core::Zero;

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

fn main() {
    //conv_kernel(128*64*4*4)  input size (64*128*128), (64*256*256),(64*512*512)(64*1024*1024)

    //just use random numbers for microbenchmark. verification correctness is ignored.

    //same input size but different kernel size
    println!("start benchmarking\n\n\n");

    let kernel_size = 5;

    let channel1 = 1;
    let channel2 = 4;
    let channel3 = 16;
    let channel4 = 32;
    let out_channel1 = 8;
    let out_channel2 = 16;
    let out_channel3 = 32;
    let out_channel4 = 64;

    let height = 28;
    let width = 28;

    let conv_kernel1 =
        vec![vec![vec![vec![100u8; kernel_size]; kernel_size]; channel1]; out_channel1];
    let conv_kernel2 =
        vec![vec![vec![vec![100u8; kernel_size]; kernel_size]; channel2]; out_channel2];
    let conv_kernel3 =
        vec![vec![vec![vec![100u8; kernel_size]; kernel_size]; channel3]; out_channel3];
    let conv_kernel4 =
        vec![vec![vec![vec![100u8; kernel_size]; kernel_size]; channel4]; out_channel4];
    let conv_kernel1_i8 =
        vec![vec![vec![vec![100i8; kernel_size]; kernel_size]; channel1]; out_channel1];
    let conv_kernel2_i8 =
        vec![vec![vec![vec![100i8; kernel_size]; kernel_size]; channel2]; out_channel2];
    let conv_kernel3_i8 =
        vec![vec![vec![vec![100i8; kernel_size]; kernel_size]; channel3]; out_channel3];
    let conv_kernel4_i8 =
        vec![vec![vec![vec![100i8; kernel_size]; kernel_size]; channel4]; out_channel4];

    let input1 = vec![vec![vec![vec![100u8; width]; height]; channel1]; 1];
    let input2 = vec![vec![vec![vec![100u8; width]; height]; channel2]; 1];
    let input3 = vec![vec![vec![vec![100u8; width]; height]; channel3]; 1];
    let input4 = vec![vec![vec![vec![100u8; width]; height]; channel4]; 1];
    let input1_i8 = vec![vec![vec![vec![100i8; width]; height]; channel1]; 1];
    let input2_i8 = vec![vec![vec![vec![100i8; width]; height]; channel2]; 1];
    let input3_i8 = vec![vec![vec![vec![100i8; width]; height]; channel3]; 1];
    let input4_i8 = vec![vec![vec![vec![100i8; width]; height]; channel4]; 1];

    let output1 = vec![vec![vec![vec![100u8; width]; height]; out_channel1]; 1];

    let output2 = vec![vec![vec![vec![100u8; width]; height]; out_channel2]; 1];

    let output3 = vec![vec![vec![vec![100u8; width]; height]; out_channel3]; 1];

    let output4 = vec![vec![vec![vec![100u8; width]; height]; out_channel4]; 1];

    let output1_i8 = vec![vec![vec![vec![100i8; width]; height]; out_channel1]; 1];

    let output2_i8 = vec![vec![vec![vec![100i8; width]; height]; out_channel2]; 1];

    let output3_i8 = vec![vec![vec![vec![100i8; width]; height]; out_channel3]; 1];

    let output4_i8 = vec![vec![vec![vec![100i8; width]; height]; out_channel4]; 1];

    let remainder1 = vec![vec![vec![vec![100u32; width]; height]; out_channel1]; 1];

    let remainder2 = vec![vec![vec![vec![100u32; width]; height]; out_channel2]; 1];

    let remainder3 = vec![vec![vec![vec![100u32; width]; height]; out_channel3]; 1];

    let remainder4 = vec![vec![vec![vec![100u32; width]; height]; out_channel4]; 1];

    let div1 = vec![vec![vec![vec![100u32; width]; height]; out_channel1]; 1];

    let div2 = vec![vec![vec![vec![100u32; width]; height]; out_channel2]; 1];

    let div3 = vec![vec![vec![vec![100u32; width]; height]; out_channel3]; 1];

    let div4 = vec![vec![vec![vec![100u32; width]; height]; out_channel4]; 1];

    let zero_point: u8 = 80;
    let multiplier1 = vec![0.1f32; out_channel1];
    let multiplier2 = vec![0.1f32; out_channel2];
    let multiplier3 = vec![0.1f32; out_channel3];
    let multiplier4 = vec![0.1f32; out_channel4];

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

    //size 1
    let conv_size1_naive = ConvCircuit {
        x: input1_i8.clone(),
        conv_kernel: conv_kernel1_i8.clone(),
        y: output1_i8.clone(),
    };
    let conv_size1_op1 = ConvCircuitU8 {
        x: input1.clone(),
        conv_kernel: conv_kernel1.clone(),
        y: output1.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,

        multiplier: multiplier1.clone(),
    };
    let conv_size1_op2 = ConvCircuitU8BitDecomposeOptimization {
        x: input1.clone(),
        conv_kernel: conv_kernel1.clone(),
        y: output1.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,
        remainder: remainder1.clone(),
        div: div1.clone(),
        multiplier: multiplier1.clone(),
    };

    let input1_var = generate_fqvar4d(cs14.clone(), input1.clone());
    let output1_var = generate_fqvar4d(cs14.clone(), output1.clone());
    let conv_size1_op3 = ConvCircuitOp3 {
        x: input1_var.clone(),
        conv_kernel: conv_kernel1.clone(),
        y: output1_var.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: vec![zero_point as u64; conv_kernel1.len()],
        remainder: remainder1.clone(),
        div: div1.clone(),
        multiplier: multiplier1.clone(),
    };

    let conv_size2_naive = ConvCircuit {
        x: input2_i8.clone(),
        conv_kernel: conv_kernel2_i8.clone(),
        y: output2_i8.clone(),
    };
    let conv_size2_op1 = ConvCircuitU8 {
        x: input2.clone(),
        conv_kernel: conv_kernel2.clone(),
        y: output2.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,

        multiplier: multiplier2.clone(),
    };
    let conv_size2_op2 = ConvCircuitU8BitDecomposeOptimization {
        x: input2.clone(),
        conv_kernel: conv_kernel2.clone(),
        y: output2.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,
        remainder: remainder2.clone(),
        div: div2.clone(),
        multiplier: multiplier2.clone(),
    };

    let input2_var = generate_fqvar4d(cs24.clone(), input2.clone());
    let output2_var = generate_fqvar4d(cs24.clone(), output2.clone());
    let conv_size2_op3 = ConvCircuitOp3 {
        x: input2_var.clone(),
        conv_kernel: conv_kernel2.clone(),
        y: output2_var.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: vec![zero_point as u64; conv_kernel2.len()],
        remainder: remainder2.clone(),
        div: div2.clone(),
        multiplier: multiplier2.clone(),
    };

    //size 3
    let conv_size3_naive = ConvCircuit {
        x: input3_i8.clone(),
        conv_kernel: conv_kernel3_i8.clone(),
        y: output3_i8.clone(),
    };
    let conv_size3_op1 = ConvCircuitU8 {
        x: input3.clone(),
        conv_kernel: conv_kernel3.clone(),
        y: output3.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,

        multiplier: multiplier3.clone(),
    };
    let conv_size3_op2 = ConvCircuitU8BitDecomposeOptimization {
        x: input3.clone(),
        conv_kernel: conv_kernel3.clone(),
        y: output3.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,
        remainder: remainder3.clone(),
        div: div3.clone(),
        multiplier: multiplier3.clone(),
    };

    let input3_var = generate_fqvar4d(cs34.clone(), input3.clone());
    let output3_var = generate_fqvar4d(cs34.clone(), output3.clone());
    let conv_size3_op3 = ConvCircuitOp3 {
        x: input3_var.clone(),
        conv_kernel: conv_kernel3.clone(),
        y: output3_var.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: vec![zero_point as u64; conv_kernel3.len()],
        remainder: remainder3.clone(),
        div: div3.clone(),
        multiplier: multiplier3.clone(),
    };

    //size 4
    let conv_size4_naive = ConvCircuit {
        x: input4_i8.clone(),
        conv_kernel: conv_kernel4_i8.clone(),
        y: output4_i8.clone(),
    };
    let conv_size4_op1 = ConvCircuitU8 {
        x: input4.clone(),
        conv_kernel: conv_kernel4.clone(),
        y: output4.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,

        multiplier: multiplier4.clone(),
    };
    let conv_size4_op2 = ConvCircuitU8BitDecomposeOptimization {
        x: input4.clone(),
        conv_kernel: conv_kernel4.clone(),
        y: output4.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: zero_point,
        remainder: remainder4.clone(),
        div: div4.clone(),
        multiplier: multiplier4.clone(),
    };

    let input4_var = generate_fqvar4d(cs44.clone(), input4.clone());
    let output4_var = generate_fqvar4d(cs44.clone(), output4.clone());
    let conv_size4_op3 = ConvCircuitOp3 {
        x: input4_var.clone(),
        conv_kernel: conv_kernel4.clone(),
        y: output4_var.clone(),

        x_0: zero_point,
        conv_kernel_0: zero_point,
        y_0: vec![zero_point as u64; conv_kernel4.len()],
        remainder: remainder4.clone(),
        div: div4.clone(),
        multiplier: multiplier4.clone(),
    };

    conv_size1_naive
        .clone()
        .generate_constraints(cs11.clone())
        .unwrap();
    println!("conv_size1_naive {}", cs11.num_constraints());
    cs11.inline_all_lcs();

    conv_size1_op1
        .clone()
        .generate_constraints(cs12.clone())
        .unwrap();
    println!("\n\nconv_size1_op1 {}", cs12.num_constraints());
    cs12.inline_all_lcs();

    conv_size1_op2
        .clone()
        .generate_constraints(cs13.clone())
        .unwrap();
    println!("\n\nconv_size1_op2 {}", cs13.num_constraints());
    cs13.inline_all_lcs();

    conv_size1_op3
        .clone()
        .generate_constraints(cs14.clone())
        .unwrap();
    println!("\n\nconv_size1_op3 {}", cs14.num_constraints());
    cs14.inline_all_lcs();

    conv_size2_naive
        .clone()
        .generate_constraints(cs21.clone())
        .unwrap();
    println!("\n\nconv_size2_naive {}", cs21.num_constraints());
    cs21.inline_all_lcs();

    conv_size2_op1
        .clone()
        .generate_constraints(cs22.clone())
        .unwrap();
    println!("\n\nconv_size2_op1 {}", cs22.num_constraints());
    cs22.inline_all_lcs();

    conv_size2_op2
        .clone()
        .generate_constraints(cs23.clone())
        .unwrap();
    println!("\n\nconv_size2_op2 {}", cs23.num_constraints());
    cs23.inline_all_lcs();

    conv_size2_op3
        .clone()
        .generate_constraints(cs24.clone())
        .unwrap();
    println!("\n\nconv_size2_op3 {}", cs24.num_constraints());
    cs24.inline_all_lcs();

    conv_size3_naive
        .clone()
        .generate_constraints(cs31.clone())
        .unwrap();
    println!("\n\nconv_size3_naive {}", cs31.num_constraints());
    cs31.inline_all_lcs();

    conv_size3_op1
        .clone()
        .generate_constraints(cs32.clone())
        .unwrap();
    println!("\n\nconv_size3_op1 {}", cs32.num_constraints());
    cs32.inline_all_lcs();

    conv_size3_op2
        .clone()
        .generate_constraints(cs33.clone())
        .unwrap();
    println!("\n\nconv_size3_op2 {}", cs33.num_constraints());
    cs33.inline_all_lcs();

    conv_size3_op3
        .clone()
        .generate_constraints(cs34.clone())
        .unwrap();
    println!("\n\nconv_size3_op3 {}", cs34.num_constraints());
    cs34.inline_all_lcs();

    conv_size4_naive
        .clone()
        .generate_constraints(cs41.clone())
        .unwrap();
    println!("\n\nconv_size4_naive {}", cs41.num_constraints());
    cs41.inline_all_lcs();

    conv_size4_op1
        .clone()
        .generate_constraints(cs42.clone())
        .unwrap();
    println!("\n\nconv_size4_op1 {}", cs42.num_constraints());
    cs42.inline_all_lcs();

    conv_size4_op2
        .clone()
        .generate_constraints(cs43.clone())
        .unwrap();
    println!("\n\nconv_size4_op2 {}", cs43.num_constraints());
    cs43.inline_all_lcs();

    conv_size4_op3
        .clone()
        .generate_constraints(cs44.clone())
        .unwrap();
    println!("\n\nconv_size4_op3 {}", cs44.num_constraints());
    cs44.inline_all_lcs();
}
