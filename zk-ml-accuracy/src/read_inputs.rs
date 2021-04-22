use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_vector1d(filename: String, len: usize) -> Vec<u8> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<u8> = vec![0u8; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{} {:?}",filename, res);
    res
}

pub fn read_vector1d_f32(filename: String, len: usize) -> Vec<f32> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<f32> = vec![0.0f32; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;

    for line in buffered.lines() {
        //println!("{:?}", line.unwrap().split(" ").collect::<Vec<&str>>());
        let raw_vec: Vec<f32> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{:?}", res);
    res
}

pub fn read_vector2d(filename: String, rows: usize, cols: usize) -> Vec<Vec<u8>> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<u8>> = vec![vec![0u8; cols]; rows];
    let buffered = BufReader::new(f);

    let mut counter = 0usize;

    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        if counter < rows * cols {
            res[counter / cols][counter % cols] = raw_vec[0]; //flattened before writing to the file. each line only contains one number
        }
        counter += 1;
    }

    res
}

pub fn read_vector4d(
    filename: String,
    in_channel: usize,
    out_channel: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    //println!("{}\n\n", filename);

    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<Vec<Vec<u8>>>> =
        vec![vec![vec![vec![0u8; cols]; rows]; out_channel]; in_channel];
    let mut tmp: Vec<u8> = vec![0u8; cols * rows * out_channel * in_channel];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        if counter < cols * rows * out_channel * in_channel {
            tmp[counter] = raw_vec[0];
        }
        counter += 1;
    }

    let mut counter = 0;
    for i in 0..in_channel {
        for j in 0..out_channel {
            for k in 0..rows {
                for m in 0..cols {
                    res[i][j][k][m] = tmp[counter];
                    counter += 1;
                }
            }
        }
    }
    res
}
