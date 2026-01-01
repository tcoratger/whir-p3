use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_symmetric::Hash;

use crate::fiat_shamir::errors::FiatShamirError as Error;

pub trait SerializedField: p3_field::Field {
    fn from_bytes(bytes: &[u8]) -> Result<Self, Error>;
    fn to_bytes(&self) -> Result<Vec<u8>, Error>;
}

macro_rules! impl_from_uniform_bytes {
    ($field:ty) => {
        impl SerializedField for $field {
            fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
                bincode::deserialize(bytes).map_err(|_err| Error::ElementIO)
            }

            fn to_bytes(&self) -> Result<Vec<u8>, Error> {
                bincode::serialize(&self).map_err(|_err| Error::ElementIO)
            }
        }
    };
}

impl_from_uniform_bytes!(p3_goldilocks::Goldilocks);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>);
impl_from_uniform_bytes!(p3_baby_bear::BabyBear);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_baby_bear::BabyBear, 4>);
impl_from_uniform_bytes!(p3_koala_bear::KoalaBear);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_koala_bear::KoalaBear, 4>);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_koala_bear::KoalaBear, 8>);

pub trait ChallengeBits {
    fn sample(&mut self, bits: usize) -> usize;
    fn sample_many(&mut self, n: usize, bits: usize) -> Vec<usize> {
        (0..n).map(|_| self.sample(bits)).collect()
    }
}

pub trait Challenge<F> {
    fn sample(&mut self) -> F;
    fn sample_many(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.sample()).collect()
    }
}

pub trait Pow<F> {
    fn pow(&mut self, bits: usize) -> Result<(), Error>;
}

pub trait Writer<T> {
    // TOOD: rename as send?
    fn write(&mut self, el: T) -> Result<(), Error>;
    fn write_hint(&mut self, el: T) -> Result<(), Error>;
    fn write_many(&mut self, el: &[T]) -> Result<(), Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn write_hint_many(&mut self, el: &[T]) -> Result<(), Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write_hint(e))
    }
}

pub trait ProverTranscript<F, EF, const D: usize>:
    Writer<F> + Writer<EF> + Writer<[F; D]> + Challenge<EF> + Pow<F> + ChallengeBits
{
}

pub trait Reader<T> {
    fn read(&mut self) -> Result<T, Error>;
    fn read_hint(&mut self) -> Result<T, Error>;
    fn read_many(&mut self, n: usize) -> Result<Vec<T>, Error> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }
    fn read_hint_many(&mut self, n: usize) -> Result<Vec<T>, Error> {
        (0..n)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()
    }
}

pub trait VerifierTranscript<F, EF, const D: usize>:
    Reader<F> + Reader<EF> + Reader<[F; D]> + Challenge<EF> + ChallengeBits + Pow<F>
{
}

#[derive(Debug, Clone)]
pub struct FiatShamirWriter<F, Challenger> {
    challenger: Challenger,
    data: Vec<u8>,
    _marker: PhantomData<F>,
}

impl<Challenger, F: Field> FiatShamirWriter<F, Challenger> {
    pub const fn init(challenger: Challenger) -> Self {
        Self {
            data: vec![],
            challenger,
            _marker: PhantomData,
        }
    }

    pub fn finalize(self) -> Vec<u8> {
        self.data
    }
}

impl<F, EF, Challenger, const D: usize> ProverTranscript<F, EF, D>
    for FiatShamirWriter<F, Challenger>
where
    F: SerializedField,
    EF: ExtensionField<F> + SerializedField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
}

impl<F: Field, EF: ExtensionField<F> + SerializedField, Challenger> Writer<EF>
    for FiatShamirWriter<F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn write(&mut self, e: EF) -> Result<(), Error> {
        self.write_hint(e)?;
        self.challenger.observe_algebra_element(e);
        Ok(())
    }

    fn write_hint(&mut self, e: EF) -> Result<(), Error> {
        let bytes = e.to_bytes()?;
        assert_eq!(bytes.len(), EF::NUM_BYTES);
        self.data.extend_from_slice(&bytes);
        Ok(())
    }
}

impl<F: SerializedField, EF: ExtensionField<F>, Challenger> Challenge<EF>
    for FiatShamirWriter<F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element::<EF>()
    }
}

impl<F: Field, Challenger> ChallengeBits for FiatShamirWriter<F, Challenger>
where
    Challenger: GrindingChallenger,
{
    fn sample(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}

impl<F: SerializedField, Challenger, const D: usize> Writer<[F; D]>
    for FiatShamirWriter<F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn write(&mut self, e: [F; D]) -> Result<(), Error> {
        self.write_hint(e)?;
        self.challenger.observe_slice(&e);
        Ok(())
    }

    fn write_hint(&mut self, e: [F; D]) -> Result<(), Error> {
        e.into_iter()
            .try_for_each(|e| <Self as Writer<F>>::write_hint(self, e))
    }
}

impl<F: SerializedField, Challenger, const D: usize> Writer<Hash<F, F, D>>
    for FiatShamirWriter<F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn write(&mut self, e: Hash<F, F, D>) -> Result<(), Error> {
        self.write_hint(e)?;
        self.challenger.observe_slice(e.as_ref());
        Ok(())
    }

    fn write_hint(&mut self, e: Hash<F, F, D>) -> Result<(), Error> {
        e.into_iter()
            .try_for_each(|e| <Self as Writer<F>>::write_hint(self, e))
    }
}

impl<F: SerializedField, Challenger> Pow<F> for FiatShamirWriter<F, Challenger>
where
    Challenger: GrindingChallenger<Witness = F> + FieldChallenger<F>,
{
    fn pow(&mut self, bits: usize) -> Result<(), Error> {
        if bits > 0 {
            let witness = self.challenger.grind(bits);
            self.write_hint(witness)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Buffer {
    pos: usize,
    data: Vec<u8>,
}

impl Buffer {
    const fn new(data: Vec<u8>) -> Self {
        Self { pos: 0, data }
    }

    fn read_exact(&mut self, dst: &mut [u8]) -> Result<(), Error> {
        (self.pos + dst.len() <= self.data.len())
            .then_some(())
            .ok_or(Error::ElementIO)?;
        dst.copy_from_slice(&self.data[self.pos..self.pos + dst.len()]);
        self.pos += dst.len();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FiatShamirReader<F, Challenger> {
    challenger: Challenger,
    buffer: Buffer,
    _marker: PhantomData<F>,
}

impl<F: Field, Challenger> FiatShamirReader<F, Challenger> {
    pub const fn init(proof: Vec<u8>, challenger: Challenger) -> Self {
        let buffer = Buffer::new(proof);
        Self {
            buffer,
            challenger,
            _marker: PhantomData,
        }
    }
}

impl<F, EF, Challenger, const D: usize> VerifierTranscript<F, EF, D>
    for FiatShamirReader<F, Challenger>
where
    F: SerializedField,
    EF: ExtensionField<F> + SerializedField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
}

impl<F: SerializedField, EF: ExtensionField<F> + SerializedField, Challenger> Reader<EF>
    for FiatShamirReader<F, Challenger>
where
    Challenger: FieldChallenger<F> + Clone,
{
    fn read(&mut self) -> Result<EF, Error> {
        let e: EF = self.read_hint()?;
        self.challenger.observe_algebra_element(e);
        Ok(e)
    }

    fn read_hint(&mut self) -> Result<EF, Error> {
        let mut bytes = vec![0u8; EF::NUM_BYTES];
        self.buffer.read_exact(bytes.as_mut())?;
        EF::from_bytes(&bytes)
    }
}

impl<F: SerializedField, Challenger, const D: usize> Reader<[F; D]>
    for FiatShamirReader<F, Challenger>
where
    Challenger: FieldChallenger<F> + Clone,
{
    fn read(&mut self) -> Result<[F; D], Error> {
        let result: [F; D] = self.read_hint()?;
        self.challenger.observe_slice(&result);
        Ok(result)
    }

    fn read_hint(&mut self) -> Result<[F; D], Error> {
        Ok((0..D)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap())
    }
}

impl<F: SerializedField, EF: ExtensionField<F>, Challenger> Challenge<EF>
    for FiatShamirReader<F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element::<EF>()
    }
}

impl<F: Field, Challenger> ChallengeBits for FiatShamirReader<F, Challenger>
where
    Challenger: GrindingChallenger,
{
    fn sample(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}

impl<F: SerializedField, Challenger> Pow<F> for FiatShamirReader<F, Challenger>
where
    Challenger: GrindingChallenger<Witness = F> + FieldChallenger<F>,
{
    fn pow(&mut self, bits: usize) -> Result<(), Error> {
        if bits > 0 {
            let witness: F = self.read_hint()?;
            self.challenger
                .check_witness(bits, witness)
                .then_some(())
                .ok_or(Error::InvalidGrindingWitness)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
    use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
    use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, StandardUniform},
        rngs::SmallRng,
    };

    use super::*;

    enum TestFs<F, Challenger> {
        Writer(FiatShamirWriter<F, Challenger>),
        Reader(FiatShamirReader<F, Challenger>),
    }

    impl<F: SerializedField, EF: ExtensionField<F>, Challenger: FieldChallenger<F>> Challenge<EF>
        for TestFs<F, Challenger>
    {
        fn sample(&mut self) -> EF {
            match self {
                Self::Writer(w) => w.sample(),
                Self::Reader(r) => r.sample(),
            }
        }
    }

    fn writer<W, F: Field + SerializedField, Ext: ExtensionField<F> + SerializedField>(
        seed: u64,
        w: &mut W,
    ) -> (Vec<F>, Vec<Ext>, Vec<Ext>, Vec<usize>)
    where
        StandardUniform: Distribution<F> + Distribution<Ext>,
        W: Writer<F> + Writer<Ext> + Challenge<F> + Challenge<Ext> + ChallengeBits + Pow<F>,
    {
        let mut rng_range = SmallRng::seed_from_u64(seed);
        let mut rng_number = SmallRng::seed_from_u64(seed + 1);

        let mut els_w = vec![];
        let mut els_ext_w = vec![];
        let mut chs_w = vec![];
        let mut idx_w = vec![];

        (0..100).for_each(|_| {
            let n_el_draw = rng_range.random_range(0..10);
            let n_idx_draw = rng_range.random_range(0..10);
            let idx_bits = rng_range.random_range(0..4);
            let n_write = rng_range.random_range(0..10);
            let n_write_ext = rng_range.random_range(0..10);
            let pow_bits = rng_range.random_range(0..4);

            let chs: Vec<Ext> = <W as Challenge<Ext>>::sample_many(w, n_el_draw);
            let idx = <W as ChallengeBits>::sample_many(w, n_idx_draw, idx_bits);
            let els: Vec<F> = (0..n_write).map(|_| rng_number.random()).collect();
            let els_ext: Vec<Ext> = (0..n_write_ext).map(|_| rng_number.random()).collect();
            w.write_many(&els).unwrap();
            w.write_many(&els_ext).unwrap();

            chs_w.extend(chs);
            idx_w.extend(idx);
            els_w.extend(els);
            els_ext_w.extend(els_ext);

            w.pow(pow_bits).unwrap();

            let n_write = rng_range.random_range(0..10);
            let n_write_ext = rng_range.random_range(0..10);

            let els: Vec<F> = (0..n_write).map(|_| rng_number.random()).collect();
            let els_ext: Vec<Ext> = (0..n_write_ext).map(|_| rng_number.random()).collect();

            w.write_hint_many(&els).unwrap();
            w.write_hint_many(&els_ext).unwrap();

            els_w.extend(els);
            els_ext_w.extend(els_ext);
        });
        (els_w, els_ext_w, chs_w, idx_w)
    }

    #[cfg(test)]
    fn reader<R, F: Field + SerializedField, Ext: ExtensionField<F> + SerializedField>(
        seed: u64,
        r: &mut R,
    ) -> (Vec<F>, Vec<Ext>, Vec<Ext>, Vec<usize>)
    where
        StandardUniform: Distribution<F> + Distribution<Ext>,
        R: Reader<F> + Reader<Ext> + Challenge<Ext> + Challenge<F> + ChallengeBits + Pow<F>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut els_r = vec![];
        let mut els_ext_r = vec![];
        let mut chs_r = vec![];
        let mut idx_r = vec![];

        (0..100).for_each(|_| {
            let n_el_draw = rng.random_range(0..10);
            let n_idx_draw = rng.random_range(0..10);
            let idx_bits = rng.random_range(0..4);
            let n_read = rng.random_range(0..10);
            let n_read_ext = rng.random_range(0..10);
            let pow_bits = rng.random_range(0..4);

            let chs: Vec<Ext> = <R as Challenge<Ext>>::sample_many(r, n_el_draw);
            let idx: Vec<usize> = <R as ChallengeBits>::sample_many(r, n_idx_draw, idx_bits);
            let els: Vec<F> = r.read_many(n_read).unwrap();
            let els_ext: Vec<Ext> = r.read_many(n_read_ext).unwrap();

            els_r.extend(els);
            els_ext_r.extend(els_ext);
            chs_r.extend(chs);
            idx_r.extend(idx);

            r.pow(pow_bits).unwrap();

            let n_read = rng.random_range(0..10);
            let n_read_ext = rng.random_range(0..10);
            let els: Vec<F> = r.read_hint_many(n_read).unwrap();
            let els_ext: Vec<Ext> = r.read_hint_many(n_read_ext).unwrap();

            els_r.extend(els);
            els_ext_r.extend(els_ext);
        });

        (els_r, els_ext_r, chs_r, idx_r)
    }

    #[test]
    fn test_transcript() {
        fn run_test<
            F: SerializedField,
            EF: ExtensionField<F> + SerializedField,
            Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        >(
            challenger: &Challenger,
        ) where
            StandardUniform: Distribution<F> + Distribution<EF>,
        {
            for i in 0..10 {
                let mut w = FiatShamirWriter::<F, _>::init(challenger.clone());
                let (els0, els_ext0, chs0, idx0) = writer::<_, F, EF>(i, &mut w);
                let checkpoint0 = <FiatShamirWriter<_, _> as Challenge<EF>>::sample(&mut w);
                let proof = w.finalize();

                let mut r = FiatShamirReader::<F, _>::init(proof, challenger.clone());
                let (els1, els_ext1, chs1, idx1) = reader::<_, F, EF>(i, &mut r);
                let checkpoint1 = <FiatShamirReader<_, _> as Challenge<EF>>::sample(&mut r);

                assert_eq!(checkpoint0, checkpoint1);
                assert!(<FiatShamirReader<F, _> as Reader<F>>::read(&mut r).is_err());

                assert_eq!(els0, els1);
                assert_eq!(els_ext0, els_ext1);
                assert_eq!(chs0, chs1);
                assert_eq!(idx0, idx1);
            }
        }

        {
            type F = Goldilocks;
            type Ext = BinomialExtensionField<F, 2>;
            type Perm = Poseidon2Goldilocks<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = BabyBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2BabyBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2KoalaBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }
    }

    #[test]
    fn test_algebra_element() {
        fn run_test<
            F: SerializedField,
            EF: ExtensionField<F> + SerializedField,
            Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        >(
            challenger: &Challenger,
        ) where
            StandardUniform: Distribution<F> + Distribution<EF>,
        {
            let mut rng = SmallRng::seed_from_u64(1);

            let proof = {
                let mut w1 = FiatShamirWriter::<F, _>::init(challenger.clone());
                let mut w0 = FiatShamirWriter::<F, _>::init(challenger.clone());

                let e0 = <FiatShamirWriter<_, _> as Challenge<EF>>::sample(&mut w0);
                let e1 = (0..EF::DIMENSION)
                    .map(|_| <FiatShamirWriter<_, _> as Challenge<F>>::sample(&mut w1));
                let e1 = EF::from_basis_coefficients_iter(e1).unwrap();
                assert_eq!(e0, e1);

                let e0: EF = rng.random();
                w0.write(e0).unwrap();
                let e0 = e0.as_basis_coefficients_slice();
                w1.write_many(e0).unwrap();
                let checkpoint0 = <FiatShamirWriter<_, _> as Challenge<F>>::sample(&mut w0);
                let checkpoint1 = <FiatShamirWriter<_, _> as Challenge<F>>::sample(&mut w1);
                assert_eq!(checkpoint0, checkpoint1);
                let proof0 = w0.finalize();
                let proof1 = w1.finalize();
                assert_eq!(proof0, proof1);
                proof0
            };

            {
                let mut r1 = FiatShamirReader::<F, _>::init(proof.clone(), challenger.clone());
                let mut r0 = FiatShamirReader::<F, _>::init(proof, challenger.clone());

                let e0 = <FiatShamirReader<_, _> as Challenge<EF>>::sample(&mut r0);
                let e1 = (0..EF::DIMENSION)
                    .map(|_| <FiatShamirReader<_, _> as Challenge<F>>::sample(&mut r1));
                let e1 = EF::from_basis_coefficients_iter(e1).unwrap();
                assert_eq!(e0, e1);

                let e0: EF = r0.read().unwrap();
                let e1 = r1.read_many(EF::DIMENSION).unwrap();
                let e1 = EF::from_basis_coefficients_slice(&e1).unwrap();
                assert_eq!(e0, e1);

                let checkpoint0 = <FiatShamirReader<_, _> as Challenge<F>>::sample(&mut r0);
                let checkpoint1 = <FiatShamirReader<_, _> as Challenge<F>>::sample(&mut r1);
                assert_eq!(checkpoint0, checkpoint1);
            }
        }

        {
            type F = Goldilocks;
            type Ext = BinomialExtensionField<F, 2>;
            type Perm = Poseidon2Goldilocks<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = BabyBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2BabyBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2KoalaBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger =
                Challenger::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1000)));
            run_test::<F, Ext, _>(&challenger);
        }
    }
}
