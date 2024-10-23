use std::{borrow::Borrow, marker::PhantomData, ops::AddAssign};

use hdf5::{Dataset, H5Type};

use itertools::Itertools;
use ndarray::{s, Dim};
use num::Zero;
use nutype::nutype;
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, AxisType, BarMode},
    Configuration, Layout, Plot, Scatter,
};
use thiserror::Error;

const OP_SIZE: usize = 2000;
#[derive(Debug, derive_more::IsVariant)]
enum Chunk2DIterCache<const N: usize, T> {
    Uninint,
    Full([Option<[T; N]>; OP_SIZE], usize),
    Partial(Vec<[T; N]>),
    Finished,
}

impl<const N: usize, T> Chunk2DIterCache<N, T> {
    pub fn next(&mut self) -> Option<[T; N]> {
        match self {
            Chunk2DIterCache::Uninint
            | Chunk2DIterCache::Full(_, 0)
            | Chunk2DIterCache::Finished => None,
            Chunk2DIterCache::Full(arr, idx) => {
                *idx -= 1;
                Some(arr[*idx].take().expect("Must be present"))
            }
            Chunk2DIterCache::Partial(vec) => vec.pop(),
        }
    }
}

pub struct Chunk2DIter<const N: usize, T> {
    dataset: Dataset,
    length: usize,
    position: usize,
    cache: Chunk2DIterCache<N, T>,
}

#[derive(Debug, Error, derive_more::From)]
enum Chunk2DCreationErrror {
    #[error("Provided dataset is not 2-dimensional, but {:?} data was given", .0)]
    Non2DData(Vec<usize>),
    #[error("Provided dataset row length is wrong: {} expected, but {} found", .0, .1)]
    BadRowLength(usize, usize),
    #[error(transparent)]
    HDF5Error(hdf5::Error),
}

impl<const N: usize, T> Chunk2DIter<N, T> {
    fn new(dataset: Dataset) -> Result<Self, Chunk2DCreationErrror> {
        let shape = dataset.shape();
        if shape.len() != 2 {
            return Err(Chunk2DCreationErrror::Non2DData(shape));
        }
        if shape[1] != N {
            return Err(Chunk2DCreationErrror::BadRowLength(N, shape[1]));
        }
        Ok(Self {
            dataset,
            length: shape[0] / OP_SIZE,
            position: 0,
            cache: Chunk2DIterCache::Uninint,
        })
    }

    fn update_cache(&mut self)
    where
        T: H5Type,
    {
        self.cache = match self.cache {
            Chunk2DIterCache::Finished | Chunk2DIterCache::Partial(_) => Chunk2DIterCache::Finished,
            Chunk2DIterCache::Uninint | Chunk2DIterCache::Full(_, _)
                if self.length == self.position =>
            {
                // only partial will fit:
                let len = self.dataset.shape()[0] - self.position * OP_SIZE;
                let mut data = self
                    .dataset
                    .read_slice_2d(s![(OP_SIZE * self.position).., ..N])
                    .expect("HDF5 error")
                    .into_iter();
                let mut vec = Vec::with_capacity(len);
                for _ in 0..len {
                    vec.push(std::array::from_fn(|_| {
                        data.next().expect("Must have enough elements")
                    }));
                }
                vec.reverse();
                Chunk2DIterCache::Partial(vec)
            }
            Chunk2DIterCache::Uninint | Chunk2DIterCache::Full(_, _) => {
                // At least one more full iteration is possible
                let mut data = self
                    .dataset
                    .read_slice_2d(s![
                        (OP_SIZE * self.position)..(OP_SIZE * (self.position + 1)),
                        ..N
                    ])
                    .expect("HDF5 error")
                    .into_iter();
                self.position += 1;
                let mut arr = std::array::from_fn(|_| {
                    Some(std::array::from_fn(|_| {
                        data.next().expect("Should be enough data")
                    }))
                });
                arr.reverse();
                Chunk2DIterCache::Full(arr, OP_SIZE)
            }
        };
    }
}

impl<const N: usize, T: H5Type> Iterator for Chunk2DIter<N, T> {
    type Item = [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.cache.next() {
            return Some(next);
        }
        self.update_cache();
        if let Some(next) = self.cache.next() {
            return Some(next);
        }
        assert!(self.cache.is_finished(), "Must have finished state now");
        None
    }
}

pub fn read_chunks_2d<const N: usize, T: H5Type + Send>(
    dataset: Dataset,
) -> impl Iterator<Item = [T; N]> + Send {
    Chunk2DIter::new(dataset).expect("Bad dataset format")
}

/// # WARNING
/// This function is not created for prod. In fact, it can and WILL drop events that do not fit
/// into chunks.
pub fn read_chunks_2d_old<const N: usize, T: H5Type + Send + Sync + Clone>(
    dataset: Dataset,
) -> impl Iterator<Item = [T; N]> + Send {
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
    let chunk_len: usize = 2000 * N;
    let length = shape[0];
    let iterations = length / chunk_len;
    let mut current_idx = 0;
    std::iter::from_fn(move || {
        let data = dataset
            .read_slice::<T, _, Dim<[usize; 2]>>(s![current_idx..(current_idx + chunk_len), ..N])
            .unwrap_or_else(|_| {
                panic!(
                    "For {} expected type {}",
                    dataset.name(),
                    dataset.dtype().unwrap().to_descriptor().unwrap()
                )
            });
        current_idx += OP_SIZE;
        Some(data)
    })
    .take(iterations)
    .flat_map(|array| {
        array
            .axis_chunks_iter(ndarray::Axis(1), N)
            .map(|window| std::array::from_fn::<T, N, _>(|i| window[(0, i)].clone()))
            .collect_vec() // TODO: write own iterator that consumes the array and holds it during
                           // the iteration
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
macro_rules! impl_from_array {
    ($target:ty, $element:ty) => {
        impl From<[$element; 1]> for $target {
            fn from([val]: [$element; 1]) -> Self {
                Self(val)
            }
        }
    };
}

#[macro_export]
macro_rules! merge {
    ($file:ident @ $path:literal $(^ $max_events:expr)? =>
        $( $var:ident $size:literal of $tp:ty > $data:literal), + $(,)?) => {{
        $(
            let mut events_left: usize = $max_events;
            let mut last_event: Option<EventId> = None;
        )?
        #[allow(non_snake_case, clippy::items_after_statements)]
        fn inner(event_ids: impl Iterator<Item=$crate::EventId> + Send, $(::paste::paste!([<_ $var>]): impl Iterator<Item=$var> + Send), +) -> impl Iterator<Item=(EventId, $($var), +)> + Send {
            ::itertools::izip!(event_ids, $(::paste::paste!([<_ $var>])), +)
        }
        inner(
            $crate::util::read_chunks_2d::<3, i64>($file.group($path)?.dataset("event_id")?)
                .map(|arr| EventId::try_from(arr).expect(&format!("Input data of event_id is invalid")))
                $(
                    .take_while(move |event_id| {
                        let _ = $max_events;
                        // If it's a first event - return is regardless
                        let Some(last_event) = last_event.as_mut() else {
                            last_event = Some(*event_id);
                            events_left -= 1;
                            return true;
                        };
                        if last_event != event_id {
                            if events_left == 0 {
                                return false;
                            }
                            *last_event = *event_id;
                            events_left -= 1;
                        }
                        true
                    })
                )?,
            $($crate::util::read_chunks_2d::<$size, $tp>($file.group($path)?.dataset($data)?)
                .map(|arr| $var::try_from(arr).expect(&format!("Input data of {} is invalid", $data)))), +
        )
    }};
}

#[nutype(
    validate(finite),
    derive(
        Debug, Default, PartialEq, PartialOrd, Eq, Ord, Borrow, Clone, Copy, Into
    ),
    default = 0.0
)]
pub struct OrdFloat(f32);

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

pub fn sane_plot(
    title: &str,
    x_axis: &str,
    y_axis: &str,
    x_type: impl Into<Option<AxisType>>,
    y_type: impl Into<Option<AxisType>>,
) -> Plot {
    let mut plot = Plot::new();
    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(
                Axis::new()
                    .title(x_axis)
                    .type_(x_type.into().unwrap_or(AxisType::Linear)),
            )
            .y_axis(
                Axis::new()
                    .title(y_axis)
                    .type_(y_type.into().unwrap_or(AxisType::Linear)),
            )
            .bar_mode(BarMode::Overlay),
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

pub fn colors<D, I, M>(data: D, mut map: M, color0: [f32; 3], color1: [f32; 3]) -> Vec<Rgb>
where
    for<'d> &'d D: IntoIterator<Item = &'d I>,
    for<'i> M: FnMut(&'i I) -> f32,
{
    // first, let's find min and max value
    let min_color = data
        .into_iter()
        .map(&mut map)
        .map(|v| OrdFloat::try_new(v).expect("Should be finite"))
        .min()
        .expect("Should have at least one value")
        .into_inner();
    let max_color = data
        .into_iter()
        .map(&mut map)
        .map(|v| OrdFloat::try_new(v).expect("Should be finite"))
        .max()
        .expect("Should have at least one value")
        .into_inner();
    let colormap = |v: &I| {
        let v = (map(v) - min_color) / (max_color - min_color); // 0..1
        Rgb::new(
            (color0[0] + (color1[0] - color0[0]) * v) as u8,
            (color0[1] + (color1[1] - color0[1]) * v) as u8,
            (color0[2] + (color1[2] - color0[2]) * v) as u8,
        )
    };
    data.into_iter().map(colormap).collect_vec()
}

pub struct DedupMax<T, I, Id, K, Or, O> {
    inner: I,
    id: Id,
    or: Or,
    store: Option<T>,
    _phantom: PhantomData<(K, O)>,
}

pub trait DedupMaxExt<T>: Sized {
    fn dedup_max<Id, K, Or, O>(self, id: Id, or: Or) -> DedupMax<T, Self, Id, K, Or, O>
    where
        for<'id> Id: FnMut(&'id T) -> K + 'id,
        K: PartialEq,
        for<'or> Or: FnMut(&'or T) -> O + 'or,
        O: Ord,
    {
        DedupMax {
            inner: self,
            id,
            or,
            store: None,
            _phantom: PhantomData,
        }
    }
}

impl<T, I> DedupMaxExt<T> for I where I: Iterator<Item = T> {}

impl<T, I, Id, K, Or, O> Iterator for DedupMax<T, I, Id, K, Or, O>
where
    I: Iterator<Item = T>,
    for<'id> Id: FnMut(&'id T) -> K + 'id,
    K: PartialEq,
    for<'or> Or: FnMut(&'or T) -> O + 'or,
    O: Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Get previous element
            let previous = match self.store.take() {
                // If it's present - good
                Some(present) => present,
                // If it's not - try getting it, and return if it's the end
                None => self.inner.next()?,
            };
            // Then, get the next element
            let Some(next) = self.inner.next() else {
                // if there are no more elements - return the previous one
                return Some(previous);
            };
            // Compare their ids
            if (self.id)(&previous) != (self.id)(&next) {
                // ids are different, store the new one, return the old
                self.store = Some(next);
                return Some(previous);
            }
            // Ids are the same; nothing to return at this cycle;
            // Store the max one
            self.store = Some(std::cmp::max_by_key(previous, next, &mut self.or));
        }
    }
}
