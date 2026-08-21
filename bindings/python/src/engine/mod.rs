use std::{num::NonZeroUsize, ops::Range, sync::Arc};

use safetensors::tensor::Metadata;

// SAFETY: trivial
const CHUNK_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(16 * 1024 * 1024) };

#[derive(Debug, PartialEq)]
struct ChunkTensorSlice {
    tensor: usize,
    src: Range<usize>,
    dst_offset: usize,
}

pub struct LoadPlan {
    metadata: Arc<Metadata>,
    /// Flat list of tensor-slice-within-a-chunk ([`ChunkTensorSlice`])
    intersections: Box<[ChunkTensorSlice]>,
    /// index into [`Self::intersections`] such that `intersections[chunk_intersections[chunk_idx]..chunk_intersections[chunk_idx + 1]]` -> gives all the intersections that are contained in `chunk_idx`
    chunk_intersections: Box<[usize]>,
    in_file_offset: usize,
    n_chunks: usize,
}

impl LoadPlan {
    pub fn new(metadata: Arc<Metadata>, in_file_offset: usize, chunk_size: NonZeroUsize) -> Self {
        let chunk_size = chunk_size.get();
        let data_len = metadata.data_len();

        let n_chunks = data_len.div_ceil(chunk_size);

        let mut intersections = Vec::with_capacity(metadata.tensors().len() + n_chunks);
        let mut offsets = vec![0usize; n_chunks + 1];
        let mut cursor = 0;

        for (i, info) in metadata.tensor_infos().iter().enumerate() {
            let (start, end) = (info.data_offsets.0, info.data_offsets.1);
            if start == end {
                continue;
            }
            for c in (start / chunk_size)..=((end - 1) / chunk_size) {
                if cursor < c {
                    offsets[cursor + 1] = intersections.len();
                    cursor += 1;
                }
                debug_assert_eq!(c, cursor);
                let chunk_start = c * chunk_size;
                let overlap =
                    start.max(chunk_start)..end.min((chunk_start + chunk_size).min(data_len));
                intersections.push(ChunkTensorSlice {
                    tensor: i,
                    // convert to offsets within the chunk
                    src: (overlap.start - chunk_start)..(overlap.end - chunk_start),
                    // convert to offset within the destination tensor
                    dst_offset: overlap.start - start,
                });
            }
        }

        if cursor < n_chunks {
            offsets[cursor + 1] = intersections.len();
        }

        Self {
            metadata,
            intersections: intersections.into_boxed_slice(),
            chunk_intersections: offsets.into_boxed_slice(),
            in_file_offset,
            n_chunks,
        }
    }

    pub fn chunk_tensor_slices(&self, chunk_idx: usize) -> &[ChunkTensorSlice] {
        &self.intersections
            [self.chunk_intersections[chunk_idx]..self.chunk_intersections[chunk_idx + 1]]
    }
}

trait Sink {}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use safetensors::{
        tensor::{Metadata, TensorInfo},
        Dtype,
    };

    use crate::engine::{ChunkTensorSlice, LoadPlan};

    fn t(name: &str, start: usize, len: usize) -> (String, TensorInfo) {
        (
            name.into(),
            TensorInfo {
                dtype: Dtype::U8,
                shape: vec![len],
                data_offsets: (start, start + len),
            },
        )
    }

    fn check_plan(plan: &LoadPlan, metadata: Arc<Metadata>, chunk_size: NonZeroUsize) {
        let chunk_size = chunk_size.get();
        let infos = plan.metadata.tensor_infos();

        assert_eq!(plan.chunk_intersections[0], 0);
        assert_eq!(
            *plan.chunk_intersections.last().unwrap(),
            plan.intersections.len()
        );
        assert!(plan.chunk_intersections.windows(2).all(|w| w[0] <= w[1]));

        for c in 0..plan.n_chunks {
            let chunk_len = chunk_size.min(metadata.data_len() - c * chunk_size);
            let mut cursor = 0;
            for slice in plan.chunk_tensor_slices(c) {
                assert_eq!(slice.src.start, cursor, "gap/overlap inside chunk {c}");
                assert!(slice.src.end > slice.src.start);
                cursor = slice.src.end;
            }
            assert_eq!(
                cursor, chunk_len,
                "tensor chunk slices don't cover the full chunk"
            );
        }
        let mut tensor_slices = vec![vec![]; infos.len()];
        for slice in plan.intersections.iter() {
            tensor_slices[slice.tensor].push((slice.dst_offset, slice.src.end - slice.src.start))
        }
        for (i, info) in infos.iter().enumerate() {
            let size = info.data_offsets.1 - info.data_offsets.0;
            let mut cursor = 0;
            for (dst, len) in &tensor_slices[i] {
                assert_eq!(
                    *dst, cursor,
                    "tensor {i}: gap/overlap at in destination buffer offset: {dst}"
                );
                cursor += len;
            }
            assert_eq!(
                cursor, size,
                "tensor {i}: missing last slice ({cursor} / {size})"
            );
        }
    }

    #[test]
    fn test_plan_valid_intersect() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Arc::new(
            Metadata::new(
                None,
                vec![t("first", 0, 5), t("second", 5, 7), t("third", 12, 6)],
            )
            .unwrap(),
        );
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        assert_eq!(
            load_plan.intersections,
            vec![
                ChunkTensorSlice {
                    tensor: 0,
                    src: 0..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 1,
                    src: 0..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 1,
                    src: 0..2,
                    dst_offset: 5,
                },
                ChunkTensorSlice {
                    tensor: 2,
                    src: 2..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 2,
                    src: 0..3,
                    dst_offset: 3,
                }
            ]
            .into_boxed_slice(),
        );
        assert_eq!(
            load_plan.chunk_intersections,
            vec![0, 1, 2, 4, 5].into_boxed_slice()
        );
        check_plan(&load_plan, metadata, chunk_size);
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Arc::new(Metadata::new(None, vec![t("first", 0, 10)]).unwrap());
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        check_plan(&load_plan, metadata, chunk_size);
    }

    #[test]
    fn test_multi_chunk_span() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Arc::new(Metadata::new(None, vec![t("first", 0, 42)]).unwrap());
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        let offsets: Vec<_> = load_plan
            .intersections
            .iter()
            .map(|s| s.dst_offset)
            .collect();
        assert_eq!(offsets, vec![0, 5, 10, 15, 20, 25, 30, 35, 40]);
        check_plan(&load_plan, metadata, chunk_size);
    }

    #[test]
    fn test_many_tensors_in_single_chunk() {
        let chunk_size = NonZeroUsize::new(20).unwrap();
        let metadata = Arc::new(
            Metadata::new(
                None,
                vec![
                    t("a", 0, 1),
                    t("b", 1, 1),
                    t("c", 2, 1),
                    t("d", 3, 1),
                    t("e", 4, 1),
                    t("f", 5, 1),
                    t("g", 6, 1),
                    t("h", 7, 1),
                    t("i", 8, 3),
                    t("j", 11, 6),
                    t("k", 17, 3),
                ],
            )
            .unwrap(),
        );
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        check_plan(&load_plan, metadata, chunk_size);
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let chunk_size = NonZeroUsize::new(4242).unwrap();
        let metadata = Arc::new(Metadata::new(None, vec![t("a", 0, 5), t("b", 5, 3)]).unwrap());
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        check_plan(&load_plan, metadata, chunk_size);
    }

    #[test]
    fn test_empty_slices() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Arc::new(
            Metadata::new(
                None,
                vec![
                    t("z0", 0, 0),
                    t("a", 0, 5),
                    t("z1", 5, 0),
                    t("b", 5, 3),
                    t("z2", 8, 0),
                ],
            )
            .unwrap(),
        );
        let load_plan = LoadPlan::new(metadata.clone(), 0, chunk_size);
        check_plan(&load_plan, metadata, chunk_size);
    }
}
