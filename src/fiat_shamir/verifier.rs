use super::{
    FsError, count_ending_zero_bits, deserialize_field, field_bytes_in_memory,
    generate_pseudo_random, hash_sha3,
};
use crate::fiat_shamir::FsParticipant;
use p3_field::Field;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsVerifier {
    state: [u8; 32],
    transcript: Vec<u8>,
    cursor: usize,
}

impl FsVerifier {
    pub fn new(transcript: Vec<u8>) -> Self {
        FsVerifier { state: [0u8; 32], transcript, cursor: 0 }
    }
    fn update_state(&mut self, data: &[u8]) {
        self.state = hash_sha3(&[&self.state[..], data].concat());
    }

    pub fn next_bytes(&mut self, len: usize) -> Result<Vec<u8>, FsError> {
        // take the len last bytes from the transcript
        if len + self.cursor > self.transcript.len() {
            return Err(FsError {});
        }
        let bytes = self.transcript[self.cursor..self.cursor + len].to_vec();
        self.cursor += len;
        self.update_state(&bytes);
        Ok(bytes)
    }

    pub fn next_variable_bytes(&mut self) -> Result<Vec<u8>, FsError> {
        let len = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
        self.next_bytes(len)
    }

    pub fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        let challenge = generate_pseudo_random(&self.state, len);
        self.update_state(len.to_be_bytes().as_ref());
        challenge
    }

    pub fn next_scalars<F: Field>(&mut self, len: usize) -> Result<Vec<F>, FsError> {
        let mut res = Vec::new();
        for _ in 0..len {
            let bytes = self.next_bytes(field_bytes_in_memory::<F>())?;
            res.push(deserialize_field(&bytes).ok_or(FsError {})?);
        }
        Ok(res)
    }

    pub fn next_scalar_matrix<F: Field>(
        &mut self,
        dims: Option<(usize, usize)>,
    ) -> Result<Vec<Vec<F>>, FsError> {
        let (n, m) = match dims {
            Some((n, m)) => (n, m),
            None => {
                let n = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
                let m = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
                (n, m)
            }
        };
        let mut res = Vec::new();
        for _ in 0..n {
            let mut row = Vec::new();
            for _ in 0..m {
                let bytes = self.next_bytes(field_bytes_in_memory::<F>())?;
                row.push(deserialize_field(&bytes).ok_or(FsError {})?);
            }
            res.push(row);
        }
        Ok(res)
    }

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        let mut res = Vec::new();
        for _ in 0..len {
            let bytes = self.challenge_bytes(field_bytes_in_memory::<F>());
            res.push(deserialize_field(&bytes).unwrap());
        }
        res
    }

    pub fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError> {
        let initial_state = self.state;
        let nonce = u64::from_le_bytes(self.next_bytes(8).unwrap().try_into().unwrap());
        if count_ending_zero_bits(&hash_sha3(&[&initial_state[..], &nonce.to_le_bytes()].concat())) >=
            bits
        {
            Ok(())
        } else {
            Err(FsError {})
        }
    }
}

impl FsParticipant for FsVerifier {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        FsVerifier::challenge_bytes(self, len)
    }

    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        FsVerifier::challenge_scalars(self, len)
    }
}
