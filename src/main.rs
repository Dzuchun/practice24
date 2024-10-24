use std::{
    borrow::Borrow,
    collections::BTreeMap,
    error::Error,
    ops::Deref,
    sync::{Arc, LazyLock, RwLock},
};

use geometry3d::{
    distance::{DistanceFrom, DistanceTo},
    line_segment::LineSegment,
    point::Point,
    vector::Vector,
};
use itertools::Itertools;
use plotly::{
    color::NamedColor,
    common::{Line, Marker, Mode},
    Configuration, Plot, Scatter3D,
};
use serde::Serialize;
use util::OrdFloat;

//mod geometry2d;
mod geometry3d;
mod util;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
struct HitId(i32);
impl_from_array! {HitId, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum LocalPlane {
    Plus60 = 0,
    Minus60 = 1,
    Vertical = 2,
}

#[derive(Debug, thiserror::Error)]
#[error("Invalid local plane index: {}", .0)]
struct InvalidLocalPlane(i32);

impl TryFrom<[i32; 1]> for LocalPlane {
    type Error = InvalidLocalPlane;

    fn try_from([value]: [i32; 1]) -> Result<LocalPlane, Self::Error> {
        match value {
            0 => Ok(Self::Plus60),
            1 => Ok(Self::Minus60),
            2 => Ok(Self::Vertical),
            _ => Err(InvalidLocalPlane(value)),
        }
    }
}

impl Serialize for LocalPlane {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32(*self as i32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
struct LocalWire(i32);
impl_from_array! {LocalWire, i32}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
struct LocalTime(i32);
impl_from_array! {LocalTime, i32}
impl LocalTime {
    fn offset(self, offset: i32) -> Self {
        Self(self.0 + offset)
    }
}

macro_rules! impl_from_float_array {
    ($type:ident, [$float:ident; $count:expr]) => {
        impl From<[$float; $count]> for $type {
            fn from(value: [$float; $count]) -> Self {
                let ord_float =
                    value.map(|fl| OrdFloat::try_new(f64::from(fl)).expect("Should be finite"));
                Self(ord_float)
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct StartPos([OrdFloat; 3]);
impl_from_float_array! {StartPos, [f32; 3]}
impl_from_float_array! {StartPos, [f64; 3]}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
struct EndPos([OrdFloat; 3]);
impl_from_float_array! {EndPos, [f32; 3]}
impl_from_float_array! {EndPos, [f64; 3]}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
struct G4Id(i32);
impl_from_array! {G4Id, i32}

#[derive(Debug, derive_more::Display)]
enum Category {
    Pion = 0,
    Muon = 1,
    Kaon = 2,
    Proton = 3,
    Electron = 4,
    Michel = 5,
    Delta = 6,
    Other = 7,
    Photon = 8,
    Cosmic,
}
impl From<[i32; 1]> for Category {
    fn from([value]: [i32; 1]) -> Self {
        match value {
            0 => Category::Pion,
            1 => Category::Muon,
            2 => Category::Kaon,
            3 => Category::Proton,
            4 => Category::Electron,
            5 => Category::Michel,
            6 => Category::Delta,
            7 => Category::Other,
            8 => Category::Photon,
            _ => Category::Cosmic,
        }
    }
}

impl Category {
    fn color(&self) -> NamedColor {
        match self {
            Category::Pion => NamedColor::Yellow,
            Category::Muon => NamedColor::Green,
            Category::Kaon => NamedColor::Black,
            Category::Proton => NamedColor::Blue,
            Category::Electron => NamedColor::Red,
            Category::Michel => NamedColor::Purple,
            Category::Delta => NamedColor::Pink,
            Category::Other => NamedColor::Orange,
            Category::Photon => NamedColor::Cyan,
            Category::Cosmic => NamedColor::Gray,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
struct Rms(i32);
impl_from_array! {Rms, i32}

const PLANE1_X: f64 = -0.3;
const PLANE2_X: f64 = -0.6;

// From `microboone_utils.py` file
fn wire(plane: LocalPlane, wire: LocalWire) -> impl Borrow<LineSegment> {
    static WIRES: LazyLock<RwLock<BTreeMap<(LocalPlane, LocalWire), Arc<LineSegment>>>> =
        LazyLock::new(|| RwLock::default());
    if let Some(res) = WIRES.read().unwrap().get(&(plane, wire)).cloned() {
        return res;
    }

    let w = f64::from(wire.0);
    let segment = match plane {
        LocalPlane::Plus60 => LineSegment::between_points(
            Point::new(
                0.0,
                (117.153 - 0.34641 * w).max(-115.505),
                (-402.941 + 0.6 * w).max(0.0352608),
            ),
            Point::new(
                0.0,
                (715.8006 - 0.34640 * w).min(117.445),
                (0.541 + 0.6 * w).min(1036.96),
            ),
        ),
        LocalPlane::Minus60 => LineSegment::between_points(
            Point::new(
                -0.3,
                (-115.213 + 0.34641 * w).min(117.445),
                (-402.941 + 0.6 * w).max(0.0352608),
            ),
            Point::new(
                -0.3,
                (-713.8606 + 0.34640 * w).max(-115.505),
                (0.541 + 0.6 * w).min(1036.96),
            ),
        ),
        LocalPlane::Vertical => LineSegment::between_points(
            Point::new(-0.6, -115.53, 0.25 + 0.3 * w),
            Point::new(-0.6, 117.47, 0.25 + 0.3 * w),
        ),
    }
    .expect("Should not be the same point");
    let segment = Arc::new(segment);
    WIRES
        .write()
        .unwrap()
        .insert((plane, wire), Arc::clone(&segment));
    segment
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn Error>> {
    const EVENT_NO: usize = 10;

    const DRIFT_VELOCITY: f64 = 1.6 / 10.0; // cm/mus
    const CLOCK_SPEED: f64 = 64e6f64;
    const TICK_TIME: f64 = 1e6f64 / CLOCK_SPEED; // mus
    const _DISTANCE_PER_TIME: f64 = DRIFT_VELOCITY * TICK_TIME;
    const DISTANCE_PER_TIME: f64 = 0.03;

    const PLANE1_OFFSET: f64 = -PLANE1_X / DRIFT_VELOCITY;
    const PLANE2_OFFSET: f64 = -PLANE2_X / DRIFT_VELOCITY;
    const DIST_SUM_CUT: f64 = 1.0; // cm
    const TIME_TOLERANCE: f64 = 12.0;
    const RMS_FACTOR: f64 = 1.0;
    const MIN_WIRE: LocalWire = LocalWire(i32::MIN);
    const MAX_WIRE: LocalWire = LocalWire(i32::MAX);

    let file = hdf5::file::File::open("BNB_All_NoWire_00.h5")?;
    let mut plot = Plot::new();
    plot.set_configuration(Configuration::new().fill_frame(true));
    // first, draw true tracks
    let tracks = merge![file @ "/particle_table" =>
        StartPos 3 of f32 > "start_position_corr",
        EndPos 3 of f32 > "end_position_corr",
        G4Id 1 of i32 > "g4_id",
        Category 1 of i32 > "category",
    ]
    .chunk_by(|(event_id, _, _, _, _)| *event_id);
    let (selected_event, event_tracks) = tracks
        .into_iter()
        .nth(EVENT_NO)
        .expect("Need at least one event");
    let mut tracks = Vec::new();
    for (
        event_id,
        StartPos([start_x, start_y, start_z]),
        EndPos([end_x, end_y, end_z]),
        G4Id(g4_id),
        category,
    ) in event_tracks
    {
        assert_eq!(event_id, selected_event);
        plot.add_trace(
            Scatter3D::new(
                vec![start_x.into_inner(), end_x.into_inner()],
                vec![start_y.into_inner(), end_y.into_inner()],
                vec![start_z.into_inner(), end_z.into_inner()],
            )
            .marker(Marker::new())
            .mode(Mode::Lines)
            .line(Line::new().color(category.color()))
            .name(format!("{category} id {g4_id}")),
        );
        let Some(track) = LineSegment::between_points(
            Point {
                x: start_x.into_inner(),
                y: start_y.into_inner(),
                z: start_z.into_inner(),
            },
            Point {
                x: end_x.into_inner(),
                y: end_y.into_inner(),
                z: end_z.into_inner(),
            },
        ) else {
            continue;
        };
        tracks.push(track);
    }
    // then, find and draw draw hits
    let hits = merge![file @ "/hit_table" =>
        HitId 1 of i32 > "hit_id",
        LocalPlane 1 of i32 > "local_plane",
        LocalTime 1 of i32 > "local_time",
        LocalWire 1 of i32 > "local_wire",
        Rms 1 of i32 > "rms",
    ]
    .chunk_by(|(event_id, _, _, _, _, _)| *event_id);
    let (_, event_hits) = hits
        .into_iter()
        .find(|(event_id, _)| *event_id == selected_event)
        .expect("Need corresponding event");

    let mut event_hits = event_hits
        .map(|(_, hit_id, plane, time, wire, rms)| (hit_id, (plane, time, wire, rms)))
        .collect::<BTreeMap<HitId, _>>();
    // filter out hits that Pandora identified as real
    let edep = merge![file @ "/pandoraHit_table" =>
        HitId 1 of i32 > "hit_id",
    ]
    .chunk_by(|(event_id, _)| *event_id);
    let (_, real_hits) = edep
        .into_iter()
        .find(|(event_id, _)| *event_id == selected_event)
        .expect("Need hits for corresponding event");
    let mut plane0 = BTreeMap::new();
    let mut plane1 = BTreeMap::new();
    let mut plane2 = BTreeMap::new();
    real_hits.dedup().for_each(|(_, hit_id)| {
        let (plane, time, wire, rms) = event_hits
            .remove(&hit_id)
            .expect("Must have entry for this hit");
        let memo = match plane {
            LocalPlane::Plus60 => &mut plane0,
            LocalPlane::Minus60 => &mut plane1,
            LocalPlane::Vertical => &mut plane2,
        };
        memo.insert((time, wire), rms);
    });
    let mut data = Vec::new();
    for (&(time0, wire_no0), rms0) in &plane0 {
        let geom_wire0 = {
            let mut gw: LineSegment =
                LineSegment::clone(wire(LocalPlane::Plus60, wire_no0).borrow());
            gw.offset(Vector {
                vx: f64::from(time0.0) * DISTANCE_PER_TIME,
                vy: 0.0,
                vz: 0.0,
            });
            Arc::new(gw)
        };
        let min_time = LocalTime(
            (f64::from(time0.0) + PLANE1_OFFSET - TIME_TOLERANCE - f64::from(rms0.0)).ceil() as i32,
        );
        let max_time = LocalTime(
            (f64::from(time0.0) + PLANE1_OFFSET + TIME_TOLERANCE + f64::from(rms0.0)).ceil() as i32,
        );
        for (&(time1, wire_no1), rms1) in plane1.range((min_time, MIN_WIRE)..=(max_time, MAX_WIRE))
        {
            let time_diff = (f64::from(time1.0) - f64::from(time0.0) - PLANE1_OFFSET).abs();
            // drop, if this time difference cannot be explained by rmses
            if time_diff >= (f64::from(rms0.0) + f64::from(rms1.0)) * RMS_FACTOR {
                // println!("Drop: t0-1");
                continue;
            }
            let geom_wire1 = {
                let mut gw: LineSegment =
                    LineSegment::clone(wire(LocalPlane::Minus60, wire_no1).borrow());
                gw.offset(Vector {
                    vx: f64::from(time1.0) * DISTANCE_PER_TIME,
                    vy: 0.0,
                    vz: 0.0,
                });
                Arc::new(gw)
            };
            let dist01 = geom_wire0.distance_from(&*geom_wire1);
            if dist01 > DIST_SUM_CUT {
                // println!("Drop: d0-1");
                continue;
            }
            let min_time = LocalTime(
                (f64::from(time0.0) + PLANE2_OFFSET - TIME_TOLERANCE - f64::from(rms0.0)).ceil()
                    as i32,
            );
            let max_time = LocalTime(
                (f64::from(time0.0) + PLANE2_OFFSET + TIME_TOLERANCE + f64::from(rms0.0)).ceil()
                    as i32,
            );
            for (&(time2, wire_no2), rms2) in
                plane2.range((min_time, MIN_WIRE)..=(max_time, MAX_WIRE))
            {
                let time_diff = (f64::from(time2.0) - f64::from(time0.0) - PLANE2_OFFSET).abs();
                if time_diff >= (f64::from(rms0.0) + f64::from(rms2.0)) * RMS_FACTOR {
                    // println!("Drop: t0-2");
                    continue;
                }
                let geom_wire2 = {
                    let mut gw: LineSegment =
                        LineSegment::clone(wire(LocalPlane::Vertical, wire_no2).borrow());
                    gw.offset(Vector {
                        vx: f64::from(time2.0) * DISTANCE_PER_TIME,
                        vy: 0.0,
                        vz: 0.0,
                    });
                    Arc::new(gw)
                };
                let dist02 = geom_wire0.distance_from(&*geom_wire2);
                if dist02 > DIST_SUM_CUT {
                    // println!("Drop: d0-2");
                    continue;
                }
                let dist12 = geom_wire1.distance_from(&*geom_wire2);
                if dist12 > DIST_SUM_CUT {
                    // println!("Drop: d1-2");
                    continue;
                }
                let dist = dist01 + dist02 + dist12;
                if dist > DIST_SUM_CUT {
                    // println!("Drop: d0-1-2");
                    continue;
                }
                let l0: &geometry3d::line::Line = geom_wire0.deref().as_ref();
                let l1: &geometry3d::line::Line = geom_wire1.deref().as_ref();
                let l2: &geometry3d::line::Line = geom_wire2.deref().as_ref();
                let p01 = l0
                    .closest_to_other_line(l1)
                    .expect("Should not be parallel");
                let p12 = l1
                    .closest_to_other_line(l2)
                    .expect("Should not be parallel");
                let p20 = l2
                    .closest_to_other_line(l0)
                    .expect("Should not be parallel");
                let dist0 = p01.distance_to(&p20);
                let dist1 = p12.distance_to(&p01);
                let dist2 = p12.distance_to(&p20);
                let dist = dist0 + dist1 + dist2;
                if dist > DIST_SUM_CUT {
                    // println!("Drop: d0-1-2");
                    continue;
                }
                // println!("hit!");
                data.push((geom_wire0.clone(), geom_wire1.clone(), geom_wire2));
            }
        }
    }
    println!("{}", data.len());
    let mut reconstructed_points = Vec::new();
    for (gw0, gw1, gw2) in data {
        let gw0l: &geometry3d::line::Line = gw0.deref().as_ref();
        let gw1l: &geometry3d::line::Line = gw1.deref().as_ref();
        let gw2l: &geometry3d::line::Line = gw2.deref().as_ref();
        let p01 = gw0l
            .closest_to_other_line(gw1l)
            .expect("Should not be parallel");
        let p12 = gw1l
            .closest_to_other_line(gw2l)
            .expect("Should not be parallel");
        let p20 = gw2l
            .closest_to_other_line(gw0l)
            .expect("Should not be parallel");
        reconstructed_points.push(Point {
            x: (p01.x + p12.x + p20.x) / 3.0,
            y: (p01.y + p12.y + p20.y) / 3.0,
            z: (p01.z + p12.z + p20.z) / 3.0,
        });
        /*
        plot.add_trace(
            Scatter3D::new(
                vec![gw0.ends().0.x, gw0.ends().1.x],
                vec![gw0.ends().0.y, gw0.ends().1.y],
                vec![gw0.ends().0.z, gw0.ends().1.z],
            )
            .mode(Mode::Lines)
            .line(Line::new().color(NamedColor::Cyan)),
        );
        plot.add_trace(
            Scatter3D::new(
                vec![gw1.ends().0.x, gw1.ends().1.x],
                vec![gw1.ends().0.y, gw1.ends().1.y],
                vec![gw1.ends().0.z, gw1.ends().1.z],
            )
            .mode(Mode::Lines)
            .line(Line::new().color(NamedColor::Cyan)),
        );
        plot.add_trace(
            Scatter3D::new(
                vec![gw2.ends().0.x, gw2.ends().1.x],
                vec![gw2.ends().0.y, gw2.ends().1.y],
                vec![gw2.ends().0.z, gw2.ends().1.z],
            )
            .mode(Mode::Lines)
            .line(Line::new().color(NamedColor::Cyan)),
        );
        */
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for point in reconstructed_points {
        x.push(point.x);
        y.push(point.y);
        z.push(point.z);
    }
    plot.add_trace(
        Scatter3D::new(x, y, z)
            .mode(Mode::Markers)
            .marker(Marker::new().color(NamedColor::Black).size(1)),
    );
    for (plane, hits) in [
        (LocalPlane::Plus60, plane0),
        (LocalPlane::Minus60, plane1),
        (LocalPlane::Vertical, plane2),
    ] {
        for (time, wire_no) in hits.keys() {
            let geom_wire = {
                let mut gw: LineSegment = LineSegment::clone(wire(plane, *wire_no).borrow());
                gw.offset(Vector {
                    vx: f64::from(time.0) * DISTANCE_PER_TIME,
                    vy: 0.0,
                    vz: 0.0,
                });
                gw
            };
            /*
            plot.add_trace(
                Scatter3D::new(
                    vec![geom_wire.ends().0.x, geom_wire.ends().1.x],
                    vec![geom_wire.ends().0.y, geom_wire.ends().1.y],
                    vec![geom_wire.ends().0.z, geom_wire.ends().1.z],
                )
                .mode(Mode::Lines)
                .line(Line::new().color(NamedColor::Green)),
            );
            */
        }
    }
    plot.show();
    Ok(())
}
