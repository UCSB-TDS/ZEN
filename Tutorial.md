# Tutorial on ZEN

In this tutorial, we illustrate the end-to-end pipeline of converting a neural network (NN) from a pretrained PyTorch version to a ZK version. 
In this tutorial, we will use ShallowNet on MNIST dataset for illustration.

## Step 1: Floating-point NN with PyTorch
The first step is to develop a floating-point PyTorch NN.
If you already have a working knowledge of PyTorch NN or have a pretrained NN, then you may skip this step.

PyTorch is a handy tool to develope NNs and provides a set of training and testing functionalities.
For detailed tutorial, we suggest [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html). Here, we provide a simple example with our ShallowNet code at "numpyInferenceEngine/ShallowNet/Mnist_end_to_end.py".

For ShallowNet, we define a NN with two fully-connected layers (i.e., nn.Linear()) and one activation layer (i.e., nn.ReLU()):

    class ShallowNet(torch.nn.Module):
        def __init__(self):
            super(ShallowNet, self).__init__()
            self.l1 = nn.Linear(784, 128, bias=False)
            self.act = nn.ReLU()
            self.l2 = nn.Linear(128, 10, bias=False)

        def forward(self, x):
            x = self.l1(x)
            x = self.act(x)
            x = self.l2(x)
            return x

Note that this model is a floating-point model by default.

Then, we train the floating-point model with the nn.CrossEntropyLoss(). 

    model = ShallowNet()
    BS = 128
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001) # adam optimizer
    losses, accuracies = [], []

    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        optim.zero_grad()
        out = model(X)
        # compute accuracy
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        t.set_description("loss %.2f accuracy %.2f" % (loss.item(), accuracy.item()))

This code should provide an accuracy of 95.1%. Note that small variation on the accuracy may appear across runs, as widely reported in NN research.

Once we have trained the floating-point NN, we dump the pretrained weights to the disk:

    torch.save(model.state_dict(), "weights.pkl")

**We can use the dumped weights for quantization.** This is especially useful when training the floating-point model takes a long time.

To evaluate the model, we can use 

    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
    print((Y_test_preds == Y_test).mean())

## Step 2: Quantized NN with PyTorch


The second step is to generate a quantized NN from the floating-point NN.
To generate the quantization parameters, we use PyTorch's quantization functionality. Note that this quantization is not ZK-friendly as we discussed in the paper.
Once we have the quantization parameters, we use our zkSNARK friendly quantization for inference with equivalent accuracy, as detailed in the next section.

According to the [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) and [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf), the PyTorch static quantization follows the [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) paper.

The quantized ShallowNet can be defined similar to the floating-point NN. The main difference comes from the "QuantStub()" and "DeQuantStub()" for indicating the layers to be quantized.
We also add two helper functions to dump the quantization parameters and quantize the input data. 

    class ShallowNet_Quant(torch.nn.Module):
        def __init__(self):
            super(ShallowNet_Quant, self).__init__()
            self.l1 = nn.Linear(784, 128, bias=False)
            self.act = nn.ReLU()
            self.l2 = nn.Linear(128, 10, bias=False)
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.l1(x)
            x = self.act(x)
            x = self.l2(x)
            x = self.dequant(x)
            return x

        def dump_feat_param(self):
            dummy_image = torch.tensor(np.ones((1, 28*28))).float()
            x = self.quant(dummy_image)
            l1_output = self.l1(x)
            act_output = self.act(l1_output)
            l2_output = self.l2(act_output)
            # print("l2: ", x)
            output = self.dequant(l2_output)
            return x.q_scale(), x.q_zero_point(), l1_output.q_scale(), l1_output.q_zero_point(), act_output.q_scale(), act_output.q_zero_point(), l2_output.q_scale(), l2_output.q_zero_point()

        def quant_input(self, x):
            x = torch.tensor(x).float()
            x_quant = self.quant(x)
            return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()

Once we have define the model, we need to specify a few configurations as:

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

Finally, we can calibrate and quantize the model for quantization:

    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(128))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        out = model(X)

    torch.quantization.convert(model, inplace=True)

## Step 3: Numpy Inference Engine with zkSNARK Friendly Quantization

We adapt the PyTorch quantized model to our zkSNARK Friendly Quantization.
Indeed, we implement a numpy inference engine with a set of NN operators (i.e., Fully-connected layer, Convolution layer, and Average Pooling layer) following our zkSNARK Friendly Quantization.
We use this numpy inference engine as a quick check for the quantized NN and evaluating the accuracy. The numpy inference engine is semantically equivalent to our ZEN, while the numpy inference engine does not provide security properties.

The first step is to convert the int8 weight from PyTorch quantized NN to uint8 weights with the following code:

    def extract_Uint_Weight(weight):
        q = weight.int_repr().numpy().astype(np.int32)
        z = weight.q_per_channel_zero_points().numpy().astype(np.int32)
        s = weight.q_per_channel_scales().numpy()
        assert (z == np.zeros(z.shape)).all(), 'Warning: zero poing is not zero'
        z = 128
        q += 128
        q = q.astype(np.int8)
        return q, z, s

We also implement a function to dump these quantization parameters to the disk such that they can be used in zkSNARK.

    def dump_txt(q, z, s, prefix):
        np.savetxt(prefix+"_q.txt", q.flatten(), fmt='%u', delimiter=',')
        # print(z, s)
        f1 = open(prefix+"_z.txt", 'w+')
        if(str(z)[0] == '['):
            f1.write(str(z)[1:-1])
        else:
            f1.write(str(z))
        f1.close()
        f2 = open(prefix+"_s.txt", 'w+')
        if(str(s)[0]=='['):
            f2.write(str(s)[1:-1])
        else:
            f2.write(str(s))
        f2.close()


To evaluate the numpy inference engine on the testing dataset, we can use the following code:

    def forward(x):
        # First quant on input x.
        x_quant_int_repr, x_quant_scale, x_quant_zero_point = model.quant_input(x)

        # 1st layer 
        # weight
        q1, z1, s1 = extract_Uint_Weight(l1_weight)
        # input feature. 
        q2, z2, s2 = x_quant_int_repr, x_quant_zero_point, x_quant_scale
        # output feature. q3 needs to be computed. z3 and s3 is fixed.
        z3, s3 = 128, l1_qscale
        q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3) # Here, q3

        # Activation Function
        act = np.maximum(q3, z3)

        # 2nd layer.
        # weight
        q1, z1, s1 = extract_Uint_Weight(l2_weight)
        # input feature. 
        q2, z2, s2 = act, z3, s3
        # output feature. q3 needs to be computed. z3 and s3 is fixed.
        z3, s3 = 128, l2_qscale
        q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)

        return q3


## Step 4: ZK circuits for different layers

Under `zk-ml/src/` directory, we have implemented several NN layer gadget circuits like `ReLU`, `Conv`, `FC` and `AvgPool` for you to easily build complex neural networks. Please refer to the parameters and `generate_constraints` API implementation of each type of layer circuit before building your own ZK NN.

## Step 5: ZK NN

Firstly, we need to define a struct for shallownet containing necessary variables including input/output, input/output commitment, parameters of each layer and quantization parameters of each layer. In `zk-ml/src/full_circuit.rs`, we have the struct `FullCircuitOpLv3Pedersen`:

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

Then we need to implement the `generate_constraints` functionality for this struct. Within `generate_constraints` functions, we implement:

1. input and output commitment circuits and call their `generate_constraints` API.
>   
    let x_com_circuit = PedersenComCircuit {
        param: self.params.clone(),
        input: self.x.clone(),
        open: self.x_open,
        commit: self.x_com,
    };
    x_com_circuit.generate_constraints(cs.clone())?;

2. Calculate the inference output of next layer(including the remainder and div for circuit instantiation) and instantiate the next layer circuit with responding parameters. Then call `generate_constraints` API of the layer circuit. Please refer to `zk-ml/src/full_circuit.rs` for each layer implementation in detail.
>       
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
        let y_fqvar = generate_fqvar(cs.clone(), y.clone());
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



## Step 6: End to end running example.

### Prover 

The detailed code is located in `zk-ml/src/bin/tutorial_shallownet_prover.rs`. It does the following work:
1. Read in parameters of all layers of shallownet generated by step3, which are all saved in directory `zk-ml/pretrained_model/shallownet/`
>     
    let x: Vec<u8> = read_vector1d("pretrained_model/shallownet/X_q.txt".to_string(), 784); 
    let l1_mat: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/shallownet/l1_weight_q.txt".to_string(),
        128,
        784,
    );
    let l2_mat: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/shallownet/l2_weight_q.txt".to_string(),
        10,
        128,
    );
    ...
2. Obtain inference output `z`.
>    
    let z: Vec<u8> = full_circuit_forward_u8(
        x.clone(),
        l1_mat.clone(),
        l2_mat.clone(),
        x_0[0],
        l1_output_0[0],
        l2_output_0[0],
        l1_mat_0[0],
        l2_mat_0[0],
        l1_mat_multiplier.clone(),
        l2_mat_multiplier.clone(),
    );
3. Obtain the commitment of input `x` and output `z`.
4. Instantiate the Shallownet circuit with all necessary parameters.
>    
    let full_circuit = FullCircuitOpLv3PedersenClassification {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,
        l1: l1_mat,
        l2: l2_mat,
        z: z.clone(),
        z_com: z_com.clone(),
        z_open,
        argmax_res: classification_res,

        x_0: x_0[0],
        y_0: l1_output_0[0],
        z_0: l2_output_0[0],
        l1_mat_0: l1_mat_0[0],
        l2_mat_0: l2_mat_0[0],
        multiplier_l1: l1_mat_multiplier.clone(),
        multiplier_l2: l2_mat_multiplier.clone(),
    };
5. Generate CRS.
>      
    let param = generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng).unwrap();
    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    let mut f = File::create("crs.data").expect("Unable to create file");
    f.write_all(&buf).expect("Unable to write");
6. Generate proof.
>     
    let pvk = prepare_verifying_key(&param.vk);
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let mut proof_buf = vec![];
    proof.serialize(&mut proof_buf).unwrap();
    let mut f = File::create("proof.data").expect("Unable to create file");
    f.write_all((&proof_buf)).expect("Unable to write data");

### Verifier

In verifier code, we basically read in the commitment of input and output, CRS and proof. Then we check whether the verification of proof passes.

    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());










