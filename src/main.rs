use std::error::Error;

use hdf5::{types::FixedAscii, File};
use itertools::Itertools;
use plotly::{
    common::{Font, TickMode},
    histogram::{Bins, HistFunc},
    layout::{Axis, TicksDirection, TicksPosition},
    Configuration, Histogram, Layout, Plot,
};
use util::{bins_1d, OrdFloat};

mod util;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("BNB_All_NoWire_00.h5")?;
    let iter = merge![file:
        1 of FixedAscii<64> > "start_process"   @ "/particle_table",
        1 of i32            > "g4_pdg"          @ "/particle_table",
        1 of f32            > "momentum"        @ "/particle_table",
    ];
    // filter out primaries
    let primaries = iter.filter(|([process], _, _)| process.as_str() == "primary");
    // filter out protons (?)
    let primary_protons = primaries.filter(|(_, [pdg], _)| *pdg == 2212);
    // map for consumption
    let primary_protons = primary_protons
        .map(|(_, _, [momentum])| (OrdFloat::try_new(f64::from(momentum)).unwrap(), 1.0));
    let bins = linspace!(0.0, 2.1, 0.25).collect_vec();
    let binned = bins_1d(
        bins.iter()
            .map(|v| OrdFloat::try_new(*v).unwrap())
            .collect_vec(),
        primary_protons,
    );
    println!("{binned:?}");
    let mut plot = Plot::new();
    plot.set_configuration(Configuration::new().fill_frame(true));
    plot.set_layout(
        Layout::new()
            .title("Simulated primary proton momentum [GeV]")
            .font(Font::new().size(20))
            .x_axis(
                Axis::new()
                    .tick_mode(TickMode::Linear)
                    .ticks_on(TicksPosition::Labels)
                    .show_tick_labels(true)
                    .ticks(TicksDirection::Outside)
                    .tick_width(3)
                    .dtick(0.25),
            ),
    );
    let nbins = bins.len();
    let hist = Histogram::new_vertical(binned)
        .x(bins)
        .auto_bin_x(false)
        .x_bins(Bins::new(0.0, 2.1, 0.25))
        .n_bins_x(nbins)
        .hist_func(HistFunc::Sum);
    plot.add_trace(hist);
    plot.show();
    Ok(())
}
