use std::{error::Error, io::Write};

use hdf5::File;
use itertools::Itertools;
use plotly::color::NamedColor;
use util::{sane_plot, scatter};

mod util;

fn main() -> Result<(), Box<dyn Error>> {
    const EVENT_COUNT: usize = 10;
    const PLANE_ID: i32 = 0;

    let file = File::open("BNB_All_NoWire_00.h5")?;

    // First, let's find a couple of first pi0 entries in particles table
    let particles = merge![file:
        a 3 of i64    > "event_id"    @ "/particle_table",
        b 1 of i32    > "g4_pdg"      @ "/particle_table",
        c 1 of f32    > "momentum"    @ "/particle_table",
    ];
    let pi0_particles = particles.filter(|(_, [pdg], _)| *pdg == 111_i32);
    let first_p0_particles: Vec<([i64; 3], f32)> = pi0_particles
        .take(EVENT_COUNT)
        .map(|(evt_id, _, [momentum])| (evt_id, momentum))
        .collect_vec();
    println!("Here are first {EVENT_COUNT} events with pi0:");
    println!("i | event_id | momentum");
    for (i, (event_id, momentum)) in first_p0_particles.iter().enumerate() {
        println!("{i} | {event_id:?} | {momentum}");
    }
    print!("Please choose one: ");
    std::io::stdout().flush()?;
    let mut ans = String::new();
    std::io::stdin().read_line(&mut ans)?;
    let ans = ans.trim().trim_end_matches('\n');
    let selected_i: usize = ans.parse()?;
    let selected_event = first_p0_particles[selected_i].0;
    println!("Selected event {:?}", selected_event);

    // Now, sift through hits table and find all hits of this event
    let all_hits = merge![file:
        a 3 of i64    > "event_id"    @ "/hit_table",
        b 1 of i32    > "hit_id"      @ "/hit_table",
        c 1 of f32    > "local_time"  @ "/hit_table",
        d 1 of i32    > "local_plane" @ "/hit_table",
        e 1 of i32    > "local_wire"  @ "/hit_table",
    ];
    let event_hits = all_hits.filter(|(event_id, ..)| *event_id == selected_event);
    // collect them all into collection
    let mut event_hits: Vec<_> = event_hits
        .enumerate()
        .map(|(i, (_, [hit_id], [ltime], [lplane], [lwire]))| {
            assert_eq!(i, hit_id as usize, "{hit_id} does not come at {i} position");
            (ltime, lplane, lwire, false)
        })
        .collect();
    println!("Finished filtering, {} hits total", event_hits.len());
    // now, sift through edep table, and mark as non-bg event these with entry in it
    let edep = merge![file:
        a 3 of i64    > "event_id"    @ "/edep_table",
        b 1 of i32    > "hit_id"      @ "/edep_table",
    ];
    let event_edep =
        edep.filter_map(|(event_id, [hit_id])| (event_id == selected_event).then_some(hit_id));
    for hit_id in event_edep {
        event_hits[hit_id as usize].3 = true;
    }
    println!("Sifting done!");

    // Plot the thing
    let mut plot = sane_plot("Event hits", "wire no", "time [ticks]");
    scatter(
        &mut plot,
        [
            (
                event_hits
                    .iter()
                    .copied()
                    .filter_map(|(_, lplane, x, is_signal)| {
                        (is_signal && lplane == PLANE_ID).then_some(f64::from(x))
                    })
                    .collect_vec(),
                event_hits
                    .iter()
                    .copied()
                    .filter_map(|(y, lplane, _, is_signal)| {
                        (is_signal && lplane == PLANE_ID).then_some(f64::from(y))
                    })
                    .collect_vec(),
                "Neutrino-originated".into(),
                NamedColor::Red,
            ),
            (
                event_hits
                    .iter()
                    .copied()
                    .filter_map(|(_, lplane, x, is_signal)| {
                        (!is_signal && lplane == PLANE_ID).then_some(f64::from(x))
                    })
                    .collect_vec(),
                event_hits
                    .iter()
                    .copied()
                    .filter_map(|(y, lplane, _, is_signal)| {
                        (!is_signal && lplane == PLANE_ID).then_some(f64::from(y))
                    })
                    .collect_vec(),
                "Cosmic-originated".into(),
                NamedColor::Blue,
            ),
        ],
    );
    plot.show();
    Ok(())
}
