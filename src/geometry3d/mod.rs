const TOLERANCE: f64 = 1e-5;

fn hypot(x: f64, y: f64, z: f64) -> f64 {
    f64::sqrt(x * x + y * y + z * z)
}

macro_rules! det3 {
    ([$a11:expr, $a12:expr, $a13:expr], [$a21:expr, $a22:expr, $a23:expr], [$a31:expr, $a32:expr, $a33:expr]) => {
        $a11 * $a22 * $a33 + $a12 * $a23 * $a31 + $a13 * $a21 * $a32
            - $a13 * $a22 * $a31
            - $a12 * $a21 * $a33
            - $a11 * $a23 * $a32
    };
}

pub mod direction;
pub mod line;
pub mod line_segment;
pub mod plane;
pub mod point;
pub mod vector;

pub mod distance;
pub mod intersect;
