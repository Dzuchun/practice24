use std::{borrow::Borrow, mem::MaybeUninit, ops::AddAssign};

use hdf5::{Dataset, H5Type};

use itertools::Itertools;
use ndarray::{s, Dim};
use num::Zero;
use nutype::nutype;
use plotly::{
    color::{NamedColor, Rgb},
    common::{Font, Marker, Mode, Title},
    layout::Axis,
    Configuration, Layout, Plot, Scatter,
};
/// # WARNING
/// This function is now created for prod. In fact, it can and WILL drop events that does not fit
/// into chunks.
pub fn read_chunks_2d<const N: usize, T: H5Type>(dataset: Dataset) -> impl Iterator<Item = [T; N]> {
    const CHUNK_LEN: usize = 2000;

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
    ($file:ident: $($var:ident $size:literal of $tp:ty > $data:literal @ $path:literal), + $(,)?) => {{
        fn inner($($var: impl Iterator<Item=[$tp; $size]>), +) -> impl Iterator<Item=($([$tp; $size]), +)> {
            ::itertools::izip!($($var), +)
        }
        inner($($crate::util::read_chunks_2d::<$size, $tp>($file.group($path)?.dataset($data)?)), +)
    }};
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
        let (Ok(i) | Err(i)) = bins.binary_search(o.borrow());
        res[i] += w;
    }
    res
}

pub fn bins_2d<B1, B2, O1: Ord, O2: Ord, W: Zero + AddAssign, D>(
    bins1: impl AsRef<[O1]>,
    bins2: impl AsRef<[O2]>,
    data: D,
) -> Vec<W>
where
    B1: Borrow<O1>,
    B2: Borrow<O2>,
    D: IntoIterator<Item = (B1, B2, W)>,
{
    let bins1 = bins1.as_ref();
    let bins2 = bins2.as_ref();
    let len = bins2.len() + 1;
    let mut res: Vec<W> = std::iter::repeat_with(W::zero)
        .take(len * (bins1.len() + 1))
        .collect();
    for (o1, o2, w) in data {
        let (Ok(i1) | Err(i1)) = bins1.binary_search(o1.borrow());
        let (Ok(i2) | Err(i2)) = bins2.binary_search(o2.borrow());
        res[i1 * len + i2] += w;
    }
    res
}

pub fn sane_plot(title: impl Into<Title>, x_axis: &str, y_axis: &str) -> Plot {
    let mut plot = Plot::new();
    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(Axis::new().title(x_axis))
            .y_axis(Axis::new().title(y_axis))
            .font(Font::new().size(14)),
    );
    plot.set_configuration(Configuration::new().show_tips(false).fill_frame(true));
    plot
}

pub fn scatter<const N: usize>(to_plot: &mut Plot, data: [(Vec<f64>, Vec<f64>, NamedColor); N]) {
    for (x, y, c) in data {
        let scatter = Scatter::new(x, y)
            .mode(Mode::Markers)
            .marker(Marker::new().color(c).size(1));
        to_plot.add_trace(scatter);
    }
}

pub fn colors<D, I, M>(data: D, mut map: M, color0: [f64; 3], color1: [f64; 3]) -> Vec<Rgb>
where
    for<'d> &'d D: IntoIterator<Item = &'d I>,
    for<'i> M: FnMut(&'i I) -> f64,
{
    // first, let's find min and max value
    let min = data
        .into_iter()
        .map(&mut map)
        .map(|v| OrdFloat::try_new(v).expect("Should be finite"))
        .min()
        .expect("Should have at least one value")
        .into_inner();
    let max = data
        .into_iter()
        .map(&mut map)
        .map(|v| OrdFloat::try_new(v).expect("Should be finite"))
        .max()
        .expect("Should have at least one value")
        .into_inner();
    let colormap = |v: &I| {
        let v = (map(v) - min) / (max - min); // 0..1
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        Rgb::new(
            (color0[0] + (color1[0] - color0[0]) * v) as u8,
            (color0[1] + (color1[1] - color0[1]) * v) as u8,
            (color0[2] + (color1[2] - color0[2]) * v) as u8,
        )
    };
    data.into_iter().map(colormap).collect_vec()
}
