use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use zk_ml::conv_circuit::*;

fn main() {
    //conv_kernel(128*64*4*4)  input size (64*128*128), (64*256*256),(64*512*512)(64*1024*1024)

    //just use random numbers for microbenchmark. verification correctness is ignored.

    //same input size but different kernel size

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

    let output1 = vec![
        vec![
            vec![vec![100u8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel1
        ];
        1
    ];

    let output2 = vec![
        vec![
            vec![vec![100u8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel2
        ];
        1
    ];

    let output3 = vec![
        vec![
            vec![vec![100u8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel3
        ];
        1
    ];

    let output4 = vec![
        vec![
            vec![vec![100u8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel4
        ];
        1
    ];

    let output1_i8 = vec![
        vec![
            vec![vec![100i8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel1
        ];
        1
    ];

    let output2_i8 = vec![
        vec![
            vec![vec![100i8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel2
        ];
        1
    ];

    let output3_i8 = vec![
        vec![
            vec![vec![100i8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel3
        ];
        1
    ];

    let output4_i8 = vec![
        vec![
            vec![vec![100i8; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel4
        ];
        1
    ];

    let remainder1 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel1
        ];
        1
    ];

    let remainder2 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel2
        ];
        1
    ];

    let remainder3 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel3
        ];
        1
    ];

    let remainder4 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel4
        ];
        1
    ];

    let div1 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel1
        ];
        1
    ];

    let div2 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel2
        ];
        1
    ];

    let div3 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel3
        ];
        1
    ];

    let div4 = vec![
        vec![
            vec![vec![100u32; width - kernel_size + 1]; height - kernel_size + 1];
            out_channel4
        ];
        1
    ];

    let zero_point: u8 = 80;
    let multiplier1 = vec![0.1f32; out_channel1];
    let multiplier2 = vec![0.1f32; out_channel2];
    let multiplier3 = vec![0.1f32; out_channel3];
    let multiplier4 = vec![0.1f32; out_channel4];

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
    let conv_size1_op3 = ConvCircuitU8BitDecomposeOptimizationSIMD {
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

    //si
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
    let conv_size2_op3 = ConvCircuitU8BitDecomposeOptimizationSIMD {
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
    let conv_size3_op3 = ConvCircuitU8BitDecomposeOptimizationSIMD {
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
    let conv_size4_op3 = ConvCircuitU8BitDecomposeOptimizationSIMD {
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

    println!("start benchmarking\n\n\n");
    let cs = ConstraintSystem::<Fq>::new_ref();
    let mut _cir_number = cs.num_constraints();

    conv_size1_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size1_naive {}", cs.num_constraints() - _cir_number);

    _cir_number = cs.num_constraints();
    conv_size1_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size1_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size1_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size1_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size1_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size1_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size2_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size2_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size2_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size2_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size2_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size2_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size2_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size2_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size3_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size3_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size3_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size3_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size3_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size3_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size3_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size3_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size4_op1
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size4_op1 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size4_op2
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size4_op2 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size4_op3
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size4_op3 {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();

    conv_size4_naive
        .clone()
        .generate_constraints(cs.clone())
        .unwrap();
    println!("conv_size4_naive {}", cs.num_constraints() - _cir_number);
    _cir_number = cs.num_constraints();
}
