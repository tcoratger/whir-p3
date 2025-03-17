use p3_baby_bear::Poseidon2BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_field::{Field, PackedValue};
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

type Perm = Poseidon2BabyBear<16>;
pub type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

pub type WhirP<F> = <F as Field>::Packing;
pub type WhirPw<F> = <F as Field>::Packing;

pub type WhirMerkleTreeMmcs<F> = MerkleTreeMmcs<WhirP<F>, WhirPw<F>, MyHash, MyCompress, 8>;

pub type WhirMerkleTree<F, M> = MerkleTree<
    <<F as Field>::Packing as PackedValue>::Value,
    <<F as Field>::Packing as PackedValue>::Value,
    M,
    8,
>;

type ByteHash = Keccak256Hash;
pub type WhirChallenger<F> = SerializingChallenger32<F, HashChallenger<u8, ByteHash, 32>>;

// Types related to using Poseidon2 in the Merkle tree.
pub(crate) type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;
pub(crate) type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
pub(crate) type Poseidon2MerkleMmcs<F, Perm16, Perm24> = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    Poseidon2Sponge<Perm24>,
    Poseidon2Compression<Perm16>,
    8,
>;
