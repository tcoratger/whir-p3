use p3_field::Field;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{
    FsParticipant, count_ending_zero_bits, deserialize_field, field_bytes_in_memory,
    generate_pseudo_random, hash_sha3, serialize_field,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsProver {
    state: [u8; 32],
    transcript: Vec<u8>,
}

impl FsProver {
    pub fn new() -> Self {
        FsProver { state: [0u8; 32], transcript: Vec::new() }
    }

    pub fn transcript_len(&self) -> usize {
        self.transcript.len()
    }

    fn update_state(&mut self, data: &[u8]) {
        self.state = hash_sha3(&[&self.state[..], data].concat());
    }

    pub fn add_bytes(&mut self, bytes: &[u8]) {
        self.transcript.extend_from_slice(bytes);
        self.update_state(bytes);
    }

    pub fn add_variable_bytes(&mut self, bytes: &[u8]) {
        self.add_bytes(&(bytes.len() as u32).to_le_bytes());
        self.add_bytes(bytes);
    }

    pub fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        let challenge = generate_pseudo_random(&self.state, len);
        self.update_state(len.to_be_bytes().as_ref());
        challenge
    }

    pub fn add_scalars<F: Field>(&mut self, scalars: &[F]) {
        for scalar in scalars {
            self.add_bytes(&serialize_field(*scalar));
        }
    }

    pub fn add_scalar_matrix<F: Field>(&mut self, scalars: &[Vec<F>], fixed_dims: bool) {
        let n = scalars.len();
        let m = scalars[0].len();
        assert!(scalars.iter().all(|v| v.len() == m));
        if !fixed_dims {
            self.add_bytes(&(n as u32).to_le_bytes());
            self.add_bytes(&(m as u32).to_le_bytes());
        }
        for row in scalars {
            for scalar in row {
                self.add_bytes(&serialize_field(*scalar));
            }
        }
    }

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        let mut res = Vec::new();
        for _ in 0..len {
            let bytes = self.challenge_bytes(field_bytes_in_memory::<F>());
            res.push(deserialize_field(&bytes).unwrap());
        }
        res
    }

    pub fn challenge_pow(&mut self, bits: usize) {
        let nonce = (0..u64::MAX)
            .into_par_iter()
            .find_any(|&nonce| {
                let hash = hash_sha3(&[&self.state[..], &nonce.to_le_bytes()].concat());
                count_ending_zero_bits(&hash) >= bits
            })
            .expect("Failed to find a nonce");
        self.add_bytes(&nonce.to_le_bytes())
    }

    pub fn transcript(self) -> Vec<u8> {
        self.transcript
    }
}

impl FsParticipant for FsProver {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        FsProver::challenge_bytes(self, len)
    }

    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        FsProver::challenge_scalars(self, len)
    }
}
