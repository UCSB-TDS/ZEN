use std::time::Instant;
use ark_sponge::{ CryptographicSponge, FieldBasedCryptographicSponge, poseidon::PoseidonSponge};
use pedersen_example::*;



fn main() {
	let start = Instant::now();

    //let seed =  &[32u8; 32];
    //let mut rng = ChaCha20Rng::from_seed(*seed);
    let inp =[1u8; SIZEOFINPUT].to_vec();
   // let inp: Vec<_> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
	//output
    let  parameter = poseidon_parameters_for_test_s();
    //let spongparams= <PoseidonSponge<Fr> as CryptographicSponge>::new(&parameter);
    let mut native_sponge = PoseidonSponge::< >::new(&parameter);
    native_sponge.absorb(&inp);
	//let out = inp.to_sponge_field_elements_as_vec();
    let out=native_sponge.squeeze_native_field_elements(SIZEOFOUTPUT);

	//println!("out ={:?}",out);
    // build the circuit
    let circuit = SPNGCircuit {
        param: parameter.clone(),
        input: inp,
        output: out.clone(),
    };
    

    let elapse = start.elapsed();
    let start2 = Instant::now();

    // generate a zkp parameters
    let zk_param = groth_param_gen_s(parameter);

    let elapse2 = start2.elapsed();
    let start3 = Instant::now();
    
    let proof = groth_proof_gen_s(&zk_param, circuit, &[32u8; 32]);

    let elapse3 = start3.elapsed();

    let start4 = Instant::now();
    groth_verify_s(&zk_param, &proof, &out);
    let elapse4 = start4.elapsed();

    println!("time to prepare comm: {:?}", elapse);
    println!("time to gen groth param: {:?}", elapse2);
    println!("time to gen proof: {:?}", elapse3);
    println!("time to verify proof: {:?}", elapse4);
}
    
