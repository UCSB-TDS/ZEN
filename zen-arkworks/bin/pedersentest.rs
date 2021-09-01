use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_ff::UniformRand;
use ark_groth16::*;
use ark_serialize::CanonicalDeserialize;
//use manta_crypto::CommitmentParam;
//use manta_crypto::MantaSerDes;
use std::time::Instant;
use pedersen_example::*;

fn main() {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let len = 128;

    //let pedersen_param = CommitmentParam::deserialize(COMMIT_PARAM.data).unwrap();
    let pedersen_param =  pedersen_setup(&[0u8; 32]);

    let input = vec![2u8; len];
    let open = Randomness::<JubJub>(Fr::rand(&mut rng));
    let commit = pedersen_commit(&input, &pedersen_param, &open);

    let circuit = PedersenComCircuit {
        param: pedersen_param.clone(),
        input,
        open,
        commit,
    };

    //sanity_check();
    let elapse = start.elapsed();
    let start2 = Instant::now();

    // TODO: remove this and use pre-computed zk_param
    // let zk_param =
    //     <Groth16<CurveTypeG> as SNARK<Fq>>::ProvingKey::deserialize_unchecked(ZK_PARAM.data)
    //         .unwrap();
    let zk_param = groth_param_gen(pedersen_param);
    //let mut g_buf = vec! [];
    //zk_param.serialize_uncompressed(&mut g_buf).unwrap();
    //println!("check size {}:", g_buf.len());

    let elapse2 = start2.elapsed();
    let start3 = Instant::now();
    // This is the part that we want to benchmark:
    let proof = groth_proof_gen(&zk_param, circuit, &[0u8; 32]);
    let elapse3 = start3.elapsed();
    println!("time to prepare comm: {:?}", elapse);
    println!("time to gen groth param: {:?}", elapse2);
    println!("time to gen proof: {:?}", elapse3);

    assert!(groth_verify(&zk_param, &proof, &commit));
}
