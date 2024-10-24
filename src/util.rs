use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, RangeInclusive},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use dashmap::DashMap;
use hdf5::{Dataset, H5Type};

use itertools::Itertools;
use ndarray::s;
use num::Zero;
use nutype::nutype;
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, AxisType, BarMode, RangeMode},
    Configuration, Layout, Plot, Scatter,
};
use thiserror::Error;
use tokio::task::spawn_blocking;
use tokio_stream::{
    wrappers::{ReceiverStream, UnboundedReceiverStream},
    Stream,
};

const OP_SIZE: usize = 4000;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct EventId([i64; 3]);

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
        use $crate::util::EventId;
        $(
            let mut events_left: usize = $max_events;
            let mut last_event: Option<EventId> = None;
        )?
        #[allow(non_snake_case)]
        fn inner(event_ids: impl Iterator<Item=EventId> + Send, $(::paste::paste!([<_ $var>]): impl Iterator<Item=$var> + Send), +) -> impl Iterator<Item=(EventId, $($var), +)> + Send {
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
pub struct OrdFloat(f64);

impl OrdFloat {
    fn mean(self, other: OrdFloat) -> Self {
        Self::try_new((self.into_inner() + other.into_inner()) / 2.0)
            .expect("Adding finite numbers cannot result in infinite one")
    }
}

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
                    .type_(y_type.into().unwrap_or(AxisType::Linear))
                    .range_mode(RangeMode::ToZero),
            )
            .bar_mode(BarMode::Group),
    );
    plot.set_configuration(Configuration::new().show_tips(false).fill_frame(true));
    plot
}

pub fn scaller<const N: usize>(to_plot: &mut Plot, data: [(Vec<f64>, Vec<f64>, NamedColor); N]) {
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
            if (&mut self.id)(&previous) != (&mut self.id)(&next) {
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

pub struct Group<T, I, Id, O> {
    inner: I,
    id: Id,
    store: Option<T>,
    _phantom: PhantomData<O>,
}

impl<T, I, Id, O> Group<T, I, Id, O> {
    pub fn new<In>(inner: In, id: Id) -> Self
    where
        In: IntoIterator<Item = T, IntoIter = I>,
        for<'id> Id: FnMut(&'id T) -> O,
    {
        Self {
            inner: inner.into_iter(),
            id,
            store: None,
            _phantom: PhantomData,
        }
    }

    /// ### Panics
    /// If order of the ids is not assending
    pub fn fill(&mut self, mut vec: Vec<T>) -> Result<Vec<T>, Vec<T>>
    where
        I: Iterator<Item = T>,
        for<'id> Id: FnMut(&'id T) -> O,
        O: PartialEq,
    {
        // Get
        let previous = match self.store.take() {
            // There's something in store, let's use it
            Some(present) => present,
            // There's nothing in the store - try getting more or error out
            None => match self.inner.next() {
                // Inner iterator yielded something, yay!
                Some(new) => new,
                // Error out
                None => return Err(vec),
            },
        };
        let previous_id = (&mut self.id)(&previous);
        vec.push(previous);
        while let Some(next) = self.inner.next() {
            if (&mut self.id)(&next) == previous_id {
                // Continue
                vec.push(next);
            } else {
                // Break the cycle
                self.store = Some(next);
                return Ok(vec);
            }
        }
        // inner iterator was exhausted; return resulting vector
        Ok(vec)
    }
}

use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

/// Rewrite of a function defined in `plot_utils.py`
pub async fn eff<I>(
    data: impl Stream<Item = I>,
    binner: impl Fn(&I) -> OrdFloat + Sync,
    mut bin_dividers: impl BorrowMut<[OrdFloat]> + Send + 'static,
    tagger: impl Fn(&I) -> bool + Sync,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let mut_bin_dividers = bin_dividers.borrow_mut();
    mut_bin_dividers.sort_unstable();
    let len = mut_bin_dividers.len();
    let mut total = vec![0usize; len];
    let mut tagged = vec![0usize; len];
    let mut dropped = 0usize;
    tokio::pin!(data);
    while let Some(i) = data.next().await {
        let bin = binner(&i);
        let idx = match mut_bin_dividers.binary_search(&bin) {
            Err(0) => {
                // Matched BEFORE the first element - drop
                dropped += 1;
                continue;
            }
            Ok(idx) if idx == len - 1 => {
                // Matched AT the last element, return one less
                idx - 1
            }
            Err(idx) if idx == len => {
                // Matched AFTER last element - drop
                dropped += 1;
                continue;
            }
            // Exact element match
            Ok(idx) => idx,
            // Match after the element
            Err(idx) => idx - 1,
        };

        total[idx] += 1;
        if tagger(&i) {
            tagged[idx] += 1;
        }
    }
    assert_eq!(total[len - 1], 0);
    assert_eq!(tagged[len - 1], 0);

    spawn_blocking(move || {
        let bin_dividers = bin_dividers.borrow_mut();
        let mut x_vec = Vec::with_capacity(len);
        let mut y_vec = Vec::with_capacity(len);
        let mut yerr_vec = Vec::with_capacity(len);
        std::iter::zip(total, tagged)
            .map(|(total, tagged)| (total as f64, tagged as f64))
            .enumerate()
            .filter(|(_, (total, _))| *total != 0.0)
            .map(|(i, (total, tagged))| {
                let bin_center = bin_dividers[i].mean(bin_dividers[i + 1]).into_inner();
                let efficiency = (tagged as f64) / (total as f64);
                let error = (efficiency * (1.0 - efficiency) / total).sqrt();
                (bin_center, efficiency, error)
            })
            .for_each(|(x, y, yerr)| {
                x_vec.push(x);
                y_vec.push(y);
                yerr_vec.push(yerr);
            });
        (x_vec, y_vec, yerr_vec, dropped)
    })
    .await
    .expect("Inner task is non-panicking")
}

pub fn is_pos_in_active_vol(x: f32, y: f32, z: f32) -> bool {
    const X: RangeInclusive<f32> = 0.0..=255.0;
    const Y: RangeInclusive<f32> = -116.0..=116.0;
    const Z: RangeInclusive<f32> = 10.0..=1036.0;
    X.contains(&x) && Y.contains(&y) && Z.contains(&z)
}

pub fn pipeline_start<R: Send + 'static>(
    mut producer: impl FnMut() -> Option<R> + Send + 'static,
    cancel: CancellationToken,
    limit: usize,
) -> impl Stream<Item = R> + Send + 'static {
    let (destination, outer_destination) = tokio::sync::mpsc::channel::<R>(limit);
    // Spawn the syncronous operator
    spawn_blocking(move || {
        while !cancel.is_cancelled() {
            let Some(r) = producer() else { return };
            let Ok(()) = destination.blocking_send(r) else {
                return;
            };
        }
    });
    ReceiverStream::new(outer_destination)
}

#[derive(Debug, derive_more::IsVariant)]
pub enum JoinedEntry<D1, D2> {
    OnlyLeft(D1),
    OnlyRight(D2),
    Both(D1, D2),
}

pub fn joiner<
    S1: Send + 'static,
    S2: Send + 'static,
    D1: Send + Sync + 'static,
    D2: Send + Sync + 'static,
    K: Eq + std::hash::Hash + Clone + Send + Sync + std::fmt::Debug + 'static,
>(
    source1: impl Stream<Item = S1> + Send + 'static,
    mut key1: impl FnMut(&S1) -> K + Send + 'static,
    mut proj1: impl FnMut(S1) -> D1 + Send + 'static,
    source2: impl Stream<Item = S2> + Send + 'static,
    mut key2: impl FnMut(&S2) -> K + Send + 'static,
    mut proj2: impl FnMut(S2) -> D2 + Send + 'static,
    limit: usize,
) -> impl Stream<Item = JoinedEntry<D1, D2>> {
    // This joiner consists of three parts:
    // - First stream puller
    // - Second stream puller
    // - Matcher that checks for identical entries every now and then
    let (matcher_tx, matcher_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let store1 = Arc::new(DashMap::<K, D1>::new());
    let store2 = Arc::new(DashMap::<K, D2>::new());
    let store1cl = Arc::clone(&store1);
    let store2cl = Arc::clone(&store2);
    let (out, outer) = tokio::sync::mpsc::channel(limit);
    // Spawn matcher
    tokio::spawn(async move {
        const REQUIRED_CHANGES: usize = 100;
        const MAX_INTERVAL: Duration = Duration::from_secs(1);
        let change_stream = UnboundedReceiverStream::new(matcher_rx);
        let significant_change_stream =
            change_stream.chunks_timeout(REQUIRED_CHANGES, MAX_INTERVAL);
        tokio::pin!(significant_change_stream);
        let mut matched_keys = Vec::new();
        while let Some(_) = significant_change_stream.next().await {
            matched_keys.clear();
            for entry in store1cl.iter() {
                let key = entry.key();
                if store2cl.contains_key(key) {
                    matched_keys.push(key.clone());
                }
            }
            for key in matched_keys.drain(..) {
                let (_, d1) = store1cl.remove(&key).expect("Should be present");
                let (_, d2) = store2cl.remove(&key).expect("Should be present");
                let Ok(()) = out.send(JoinedEntry::Both(d1, d2)).await else {
                    return;
                };
            }
            store1cl.shrink_to_fit();
            store2cl.shrink_to_fit();
        }
        // Both streams were exhausted. Send currently held entries as incomplete ones
        for (_, d1) in Arc::into_inner(store1cl)
            .expect("Must be the only arc")
            .into_iter()
        {
            let Ok(()) = out.send(JoinedEntry::OnlyLeft(d1)).await else {
                return;
            };
        }
        for (_, d2) in Arc::into_inner(store2cl)
            .expect("Must be the only arc")
            .into_iter()
        {
            let Ok(()) = out.send(JoinedEntry::OnlyRight(d2)).await else {
                return;
            };
        }
    });
    // Spawn pullers
    let matcher_tx2 = matcher_tx.clone();
    tokio::spawn(async move {
        tokio::pin!(source1);
        while let Some(s1) = source1.next().await {
            let key = key1(&s1);
            if store1.contains_key(&key) {
                panic!("Key {:?} was encountered twice", key);
            }
            store1.insert(key, proj1(s1));
            let Ok(()) = matcher_tx2.send(()) else { return };
        }
        // Should drop them in this order specifically
        std::mem::drop(store1);
        std::sync::atomic::fence(Ordering::AcqRel);
        std::mem::drop(matcher_tx2);
    });
    tokio::spawn(async move {
        tokio::pin!(source2);
        while let Some(s2) = source2.next().await {
            let key = key2(&s2);
            if store2.contains_key(&key) {
                panic!("Key {:?} was encountered twice", key);
            }
            store2.insert(key, proj2(s2));
            let Ok(()) = matcher_tx.send(()) else { return };
        }
        // Should drop them in this order specifically
        std::mem::drop(store2);
        std::sync::atomic::fence(Ordering::AcqRel);
        std::mem::drop(matcher_tx);
    });
    ReceiverStream::new(outer)
}

#[derive(Debug)]
pub struct MeanVarBuilder<T, T2> {
    sum: T,
    sum_sq: T2,
    count: usize,
}

impl<T, T2> MeanVarBuilder<T, T2> {
    pub fn new() -> Self
    where
        T: Zero,
        T2: Zero,
    {
        Self {
            sum: T::zero(),
            sum_sq: T2::zero(),
            count: 0,
        }
    }

    pub fn add(&mut self, var: T)
    where
        T: AddAssign + Mul<Output = T2> + Clone,
        T2: AddAssign,
    {
        self.count += 1;
        self.sum += var.clone();
        self.sum_sq += var.clone() * var;
    }

    pub fn build(self) -> Option<(T, T2)>
    where
        T: Mul<Output = T2> + Div<usize, Output = T> + Clone,
        T2: Mul<f64, Output = T2>
            + Add<Output = T2>
            + Div<usize, Output = T2>
            + Mul<usize, Output = T2>,
    {
        if self.count == 0 {
            return None;
        }
        let mean = self.sum.clone() / self.count;
        let dev_sq = (mean.clone() * mean.clone() * self.count)
            + self.sum_sq
            + (self.sum * mean.clone() * (-2.0f64));
        let var = dev_sq / self.count;
        Some((mean, var))
    }
}
