use std::{error::Error, mem::MaybeUninit};

use hdf5::{Dataset, File, H5Type};

use ndarray::{s, Dim};
use plotly::{
    color::NamedColor,
    common::{Marker, Mode},
    layout::Axis,
    Configuration, Layout, Plot, Scatter,
};

fn read_chunks_2d<const N: usize, T: H5Type>(dataset: Dataset) -> impl Iterator<Item = [T; N]> {
    const CHUNK_LEN: usize = 1000;

    let shape = dataset.shape();
    assert!(
        shape.len() == 2,
        "Only supports 2d datasets, but {shape:?} was given"
    );
    assert!(
        shape[1] == N,
        "Dataset row length is incorrect: {}",
        shape[1]
    );
    let length = shape[0];
    let iterations = length / CHUNK_LEN;
    let mut current_idx = 0;
    std::iter::from_fn(move || {
        let data = dataset
            .read_slice::<T, _, Dim<[usize; 2]>>(s![current_idx..(current_idx + CHUNK_LEN), ..N])
            .expect("Read");
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

fn main() -> Result<(), Box<dyn Error>> {
    // construct vertex iterator
    let file = File::open("BNB_All_NoWire_00.h5")?;
    let iter = read_chunks_2d::<3, f32>(file.group("/event_table")?.dataset("nu_vtx")?);
    let (x, y, z) = iter.fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut x, mut y, mut z), [nx, ny, nz]| {
            x.push(nx);
            y.push(ny);
            z.push(nz);
            (x, y, z)
        },
    );

    let config = Configuration::new().fill_frame(true);
    let marker = Marker::new().size(2).color(NamedColor::DarkGrey);

    {
        let scatter = Scatter::new(x, y.clone())
            .mode(Mode::Markers)
            .marker(marker.clone());
        let mut plot = Plot::new();
        plot.set_layout(
            Layout::new()
                .title("XY plane")
                .x_axis(Axis::new().title("x [cm]"))
                .y_axis(Axis::new().title("y [cm]")),
        );
        plot.set_configuration(config.clone());
        plot.add_trace(scatter);
        plot.show();
    }

    {
        let scatter = Scatter::new(z, y).mode(Mode::Markers).marker(marker);
        let mut plot = Plot::new();
        plot.set_layout(
            Layout::new()
                .title("ZY plane")
                .x_axis(Axis::new().title("z [cm]"))
                .y_axis(Axis::new().title("y [cm]")),
        );
        plot.set_configuration(config);
        plot.add_trace(scatter);
        plot.show();
    }
    Ok(())
}
