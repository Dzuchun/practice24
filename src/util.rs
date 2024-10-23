use std::{borrow::Borrow, mem::MaybeUninit, ops::AddAssign};

use hdf5::{Dataset, H5Type};

use ndarray::{s, Dim};
use num::Zero;
use nutype::nutype;
/// # WARNING
/// This function is now created for prod. In fact, it can and WILL drop events that does not fit
/// into chunks.
pub fn read_chunks_2d<const N: usize, T: H5Type>(dataset: Dataset) -> impl Iterator<Item = [T; N]> {
    const CHUNK_LEN: usize = 1000;

    let shape = dataset.shape();
    assert!(
        shape.len() == 2,
        "Only supports 2d datasets, but {shape:?} was given"
    );
    assert!(
        shape[1] == N,
        "Dataset row length is incorrect: given {} expected {}",
        N,
        shape[1]
    );

    let length = shape[0];
    let iterations = length / CHUNK_LEN;
    let mut current_idx = 0;
    std::iter::from_fn(move || {
        let data = dataset
            .read_slice::<T, _, Dim<[usize; 2]>>(s![current_idx..(current_idx + CHUNK_LEN), ..N])
            .unwrap_or_else(|_| {
                panic!(
                    "For {} expected type {}",
                    dataset.name(),
                    dataset.dtype().unwrap().to_descriptor().unwrap()
                )
            });
        current_idx += CHUNK_LEN;
        Some(data)
    })
    .take(iterations)
    .flat_map(move |array| {
        let mut array = array.into_iter();
        std::iter::from_fn(move || {
            let mut arr: [_; N] = std::array::from_fn(|_| MaybeUninit::<T>::uninit());
            for arr_i in &mut arr {
                *arr_i = MaybeUninit::new(array.next().expect("Element"));
            }
            Some(arr.map(|v| unsafe { MaybeUninit::assume_init(v) }))
        })
        .take(CHUNK_LEN)
    })
}

#[macro_export]
macro_rules! linspace {
    ($start:expr, $end:expr, $step:expr) => {
        ::std::iter::successors(Some($start), |&(mut last)| {
            last += $step;
            (last <= $end).then_some(last)
        })
    };
}

#[macro_export]
macro_rules! merge {
    ($file:ident: $($size:literal of $tp:ty > $data:literal @ $path:literal), + $(,)?) => {
        ::itertools::izip!($($crate::util::read_chunks_2d::<$size, $tp>($file.group($path)?.dataset($data)?)), +)
    };
}

#[nutype(
    validate(finite),
    derive(Debug, Default, PartialEq, PartialOrd, Eq, Ord, Borrow, Clone, Into),
    default = 0.0
)]
pub struct OrdFloat(f64);

pub fn bins_1d<B, O: Ord, W: Zero + AddAssign, D>(bins: impl Borrow<[O]>, data: D) -> Vec<W>
where
    B: Borrow<O>,
    D: IntoIterator<Item = (B, W)>,
{
    let bins = bins.borrow();
    let mut res: Vec<W> = std::iter::repeat_with(W::zero)
        .take(bins.len() + 1)
        .collect();
    for (o, w) in data {
        let i = match bins.binary_search(o.borrow()) {
            Ok(i) => i,      // exact match for bin boundary
            Err(i) => i - 1, // sorted after (i-1)th bin boundary, so it belongs to (i-1)th bin.
        };
        res[i] += w;
    }
    res
}

pub fn bins_2d<B1, B2, O1: Ord, O2: Ord, W: Zero + AddAssign, D>(
    bins1: &[O1],
    bins2: &[O2],
    data: D,
) -> Vec<Vec<W>>
where
    B1: Borrow<O1>,
    B2: Borrow<O2>,
    D: IntoIterator<Item = (B1, B2, W)>,
{
    let mut res: Vec<Vec<W>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(W::zero)
            .take(bins2.len() + 1)
            .collect()
    })
    .take(bins1.len() + 1)
    .collect();
    for (o1, o2, w) in data {
        let (Ok(i1) | Err(i1)) = bins1.binary_search(o1.borrow());
        let (Ok(i2) | Err(i2)) = bins2.binary_search(o2.borrow());
        res[i1][i2] += w;
    }
    res
}
