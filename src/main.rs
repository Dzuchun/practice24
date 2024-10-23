use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    ops::{AddAssign, DerefMut},
    sync::{Arc, Mutex},
};

use hdf5::File;
use plotly::{
    color::NamedColor,
    common::Marker,
    histogram::{Bins, HistFunc},
    Histogram,
};
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use util::{sane_plot, DedupMaxExt, OrdFloat, OrdFloatError};

mod util;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct EventId([i64; 3]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct HitId(i32);
impl_from_array! {HitId, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct G4Id(i32);
impl_from_array! {G4Id, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct SliceId(i32);
impl_from_array! {SliceId, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct SlicePdg(i32);
impl_from_array! {SlicePdg, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EnergyFraction(OrdFloat);

impl TryFrom<[f32; 1]> for EnergyFraction {
    type Error = OrdFloatError;

    fn try_from([value]: [f32; 1]) -> Result<Self, Self::Error> {
        Ok(Self(OrdFloat::try_new(value)?))
    }
}

#[derive(Debug, Default)]
struct SliceInfo {
    hits: usize,
    nuhits: usize,
    pdg: Option<SlicePdg>,
}

enum PurityType {
    Neutrino(f32),
    Cosmic(f32),
}

impl SliceInfo {
    fn add_hit(&mut self, is_nu: bool) {
        self.hits += 1;
        if is_nu {
            self.nuhits += 1;
        }
    }

    fn assign_pdg(&mut self, pdg: SlicePdg) -> bool {
        if self.pdg.is_some() {
            return false;
        }
        self.pdg = Some(pdg);
        true
    }

    fn purity(&self) -> PurityType {
        let purity = (self.nuhits as f32) / (self.hits as f32);
        if self.pdg.is_some_and(|SlicePdg(id)| id == 12 || id == 14) {
            PurityType::Neutrino(purity)
        } else {
            PurityType::Cosmic(purity)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn Error>> {
    const MAX_HITS: usize = 3000;

    let file = File::open("BNB_All_NoWire_00.h5")?;
    let event_ids = Mutex::new(BTreeSet::<EventId>::new());

    // First, get unique hits
    let mut hits = merge![file @ "/hit_table" ^ MAX_HITS =>
        HitId 1 of i32 > "hit_id",
    ]
    .par_bridge()
    .inspect(|(event_id, _)| {
        event_ids.lock().unwrap().insert(*event_id);
    })
    .map(|key| (key, (None::<SliceId>, None::<G4Id>)))
    .collect::<BTreeMap<_, _>>();
    println!(
        "Found all hits, {} total ({} events)",
        hits.len(),
        event_ids.lock().unwrap().len()
    );

    // Merge into pandora hits
    merge![file @ "/pandoraHit_table" ^ MAX_HITS =>
        HitId 1 of i32 > "hit_id",
        SliceId 1 of i32 > "slice_id",
    ]
    .par_bridge()
    .filter(|(event_id, _, _)| event_ids.lock().unwrap().contains(event_id))
    .for_each_with(
        Arc::new(Mutex::new(&mut hits)),
        |&mut ref hits, (event_id, hit_id, slice_id)| {
            let mut lock = hits.lock().unwrap();
            let value = lock
                .deref_mut()
                .get_mut(&(event_id, hit_id))
                .expect("entry is present in pandoraHits, but not in hits");
            let present_slice_id = &mut value.0;
            assert!(
                present_slice_id.is_none(),
                "slice_id should be unique for a hit"
            );
            *present_slice_id = Some(slice_id);
        },
    );
    println!("Injected slice_id");

    // Merge into largest energy fraction from edep
    merge![file @ "/edep_table" ^ MAX_HITS =>
        HitId 1 of i32 > "hit_id",
        EnergyFraction 1 of f32 > "energy_fraction",
        G4Id 1 of i32 > "g4_id",
    ]
    .dedup_max(
        |(event_id, hit_id, _, _)| (*event_id, *hit_id),
        |(_, _, energy_fraction, _)| *energy_fraction,
    )
    .par_bridge()
    .filter(|(event_id, _, _, _)| event_ids.lock().unwrap().contains(event_id))
    .for_each_with(
        Arc::new(Mutex::new(&mut hits)),
        |&mut ref hits, (event_id, hit_id, _, g4_id)| {
            let mut lock = hits.lock().unwrap();
            let value = lock
                .get_mut(&(event_id, hit_id))
                .expect("entry is present in edep, but not in hits");
            let present_g4_id = &mut value.1;
            assert!(present_g4_id.is_none(), "g4_id should be unique for a hit");
            *present_g4_id = Some(g4_id);
        },
    );
    println!("Injected g4_id");

    let mut slice_info = BTreeMap::<(EventId, SliceId), SliceInfo>::new();
    let mut nuhitstot = BTreeMap::<EventId, usize>::new();
    hits.par_iter()
        .map(|((event_id, _), (slice_id, g4_id))| {
            (
                *event_id,
                slice_id.expect("All sliceids should've been assigned"),
                *g4_id,
            )
        })
        .for_each_with(
            (
                Arc::new(Mutex::new(&mut slice_info)),
                Arc::new(Mutex::new(&mut nuhitstot)),
            ),
            |&mut (ref slice_info, ref nuhitstot), (event_id, slice_id, g4_id)| {
                let is_nu = g4_id.is_some_and(|G4Id(id)| id >= 0);
                slice_info
                    .lock()
                    .unwrap()
                    .entry((event_id, slice_id))
                    .or_default()
                    .add_hit(is_nu);
                nuhitstot
                    .lock()
                    .unwrap()
                    .entry(event_id)
                    .or_default()
                    .add_assign(usize::from(is_nu));
            },
        );
    println!("Populated sliceinfo");

    // Pandora primaries
    merge![file @ "/pandoraPrimary_table" ^ MAX_HITS =>
        SliceId 1 of i32 > "slice_id",
        SlicePdg 1 of i32 > "slice_pdg",
    ]
    .dedup_max(|(event_id, slice_id, _)| (*event_id, *slice_id), |_| 0)
    .par_bridge()
    .filter(|(event_id, _, _)| event_ids.lock().unwrap().contains(event_id))
    .for_each_with(
        Arc::new(Mutex::new(&mut slice_info)),
        |&mut ref slice_info, (event_id, slice_id, slice_pdg)| {
            if !slice_info
                .lock()
                .unwrap()
                .get_mut(&(event_id, slice_id))
                .expect("entry is present in pandoraPrimary, but not in pandoraHit")
                .assign_pdg(slice_pdg)
            {
                println!(
                    "WARN: attempt to assign PDG for second time at ({event_id:?}, {slice_id:?})"
                );
            }
        },
    );
    println!("Injected slice_pdg");

    // Plot purity and completeness of neutrino slices
    let mut purity: Vec<f32> = Vec::with_capacity(slice_info.len());
    let mut completeness: Vec<f32> = Vec::with_capacity(slice_info.len());
    slice_info
        .par_iter()
        .filter_map(|((event_id, _), info)| {
            if let PurityType::Neutrino(purity) = info.purity() {
                Some((purity, (info.nuhits as f32) / (nuhitstot[event_id] as f32)))
            } else {
                None
            }
        })
        .for_each_with(
            (
                Arc::new(Mutex::new(&mut purity)),
                Arc::new(Mutex::new(&mut completeness)),
            ),
            |&mut (ref purity, ref completeness), (purity_entry, completeness_entry)| {
                purity.lock().unwrap().push(purity_entry);
                completeness.lock().unwrap().push(completeness_entry);
            },
        );

    let mut plot = sane_plot(
        "completeness and purity of Pandora neutrino slices",
        "metric value",
        "number of events", // WARN: not events, but slices actually!
        None,
        None,
    );
    plot.add_trace(
        Histogram::new(purity)
            .auto_bin_x(false)
            .x_bins(Bins::new(0.0, 1.0, 0.1))
            .hist_func(HistFunc::Sum)
            .marker(
                Marker::new()
                    .line(
                        plotly::common::Line::new()
                            .width(2.0)
                            .color(NamedColor::Orange),
                    )
                    .color(NamedColor::Transparent),
            )
            .name("purity"),
    );
    plot.add_trace(
        Histogram::new(completeness)
            .auto_bin_x(false)
            .x_bins(Bins::new(0.0, 1.0, 0.1))
            .hist_func(HistFunc::Sum)
            .marker(
                Marker::new()
                    .line(
                        plotly::common::Line::new()
                            .width(2.0)
                            .color(NamedColor::DarkBlue),
                    )
                    .color(NamedColor::Transparent),
            )
            .name("completeness"),
    );
    plot.show();

    Ok(())
}
