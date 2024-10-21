use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

const ENTRIES_INITIAL: usize = 28;
const ENTRIES_FILTERED: usize = 21;
fn read_data(path: PathBuf) -> Result<Vec<[f32; ENTRIES_FILTERED]>, std::io::Error> {
    let mut result = Vec::with_capacity(10_000);
    let file = File::options().read(true).write(false).open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 4 * ENTRIES_INITIAL];
    let mut valbuf = [0u8; 4];
    let mut entry_data = [0f32; ENTRIES_FILTERED];
    while let Ok(()) = reader.read_exact(&mut buf) {
        for i in 0..ENTRIES_FILTERED {
            valbuf.clone_from_slice(&buf[4 * i..4 * (i + 1)]);
            entry_data[i] = f32::from_ne_bytes(valbuf);
        }
        result.push(entry_data);
    }
    Ok(result)
}

fn save_data(path: PathBuf, data: Vec<[f32; ENTRIES_FILTERED]>) -> Result<(), std::io::Error> {
    let file = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);
    let mut val_buf;
    for entry_data in data {
        for i in 0..ENTRIES_FILTERED {
            val_buf = entry_data[i].to_ne_bytes();
            writer.write_all(val_buf.as_slice())?;
        }
    }
    Ok(())
}

const POSITIVE_PARAMS: [usize; 10] = [1, 4, 6, 9, 10, 13, 14, 17, 18, 21];
const NORMAL_PARAMS: [usize; 11] = [2, 3, 5, 7, 8, 11, 12, 15, 16, 19, 20];
fn main() -> Result<(), std::io::Error> {
    // load the data
    let mut data_signal = read_data(PathBuf::from("../higgs/signal"))?;
    let mut data_noise = read_data(PathBuf::from("../higgs/noise"))?;

    // transform the data
    for ind in POSITIVE_PARAMS {
        // for positive params, we'll need to find a mean value
        let (count, sum) = data_signal
            .iter()
            .chain(data_noise.iter())
            .fold((0usize, 0f32), |(count, sum), next| {
                (count + 1, sum + next[ind - 1])
            });
        let mean = sum / count as f32;
        // now, we divide by the average, to bring it to 1
        data_signal
            .iter_mut()
            .chain(data_noise.iter_mut())
            .for_each(|v| v[ind - 1] /= mean);
        println!("For param {ind} got mean = {mean}");
    }
    for ind in NORMAL_PARAMS {
        // for normal params, we first calculate the moments
        let (m0, m1, m2) = data_signal
            .iter()
            .chain(data_noise.iter())
            .map(|arr| arr[ind - 1])
            .fold((0usize, 0f32, 0f32), |(m0, m1, m2), next| {
                (m0 + 1, m1 + next, m2 + next.powi(2))
            });
        // then mean and variance
        let count = m0 as f32;
        let mean = m1 / count;
        let var = m2 / count - mean.powi(2);
        // and standard deviation
        let stdev = f32::sqrt(var);
        // now, we scale the data, so that it mimics normal distribution:
        data_signal
            .iter_mut()
            .chain(data_noise.iter_mut())
            .for_each(|v| v[ind - 1] = (v[ind - 1] - mean) / stdev);
        println!("For param {ind} got mean = {mean}, stdev = {stdev}");
    }

    // save the data
    save_data(PathBuf::from("../higgs/signal_reg"), data_signal)?;
    save_data(PathBuf::from("../higgs/noise_reg"), data_noise)?;
    Ok(())
}
