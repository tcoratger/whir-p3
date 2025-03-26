use criterion::{Criterion, black_box, criterion_group, criterion_main};
use whir_p3::ntt::{matrix::MatrixMut, transpose::transpose_copy};

/// Creates an `M x N` matrix with elements `(row, col)` for benchmarking.
fn create_matrix(rows: usize, cols: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .collect()
}

/// Benchmark function for `transpose_copy`
fn benchmark_transpose_copy(c: &mut Criterion) {
    let rows = 1024;
    let cols = 512; // Rectangular matrix for non-square benchmark
    let mut src_matrix = create_matrix(rows, cols);
    let mut dst_matrix = vec![(0, 0); rows * cols]; // Empty transposed matrix

    let src_view = MatrixMut::from_mut_slice(&mut src_matrix, rows, cols);
    let dst_view = MatrixMut::from_mut_slice(&mut dst_matrix, cols, rows); // Transposed shape

    c.bench_function(&format!("transpose_copy {rows}x{cols}"), |b| {
        b.iter(|| {
            transpose_copy(black_box(src_view.clone()), black_box(dst_view.clone()));
        });
    });
}

// Register benchmark group
criterion_group!(benches, benchmark_transpose_copy);
criterion_main!(benches);
