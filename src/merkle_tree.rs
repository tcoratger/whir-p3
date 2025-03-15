use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

pub type WhirMerkleTree = MerkleTreeMmcs<
    <BabyBear as Field>::Packing,
    <BabyBear as Field>::Packing,
    MyHash,
    MyCompress,
    8,
>;
