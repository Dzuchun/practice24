use std::ops::Sub;

use super::{distance::DistanceFrom, hypot, vector::Vector};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    pub fn new(x: impl Into<f64>, y: impl Into<f64>, z: impl Into<f64>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }
}

impl Sub for Point {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector {
            vx: self.x - rhs.x,
            vy: self.y - rhs.y,
            vz: self.z - rhs.z,
        }
    }
}

impl DistanceFrom<Point> for Point {
    fn distance_from(&self, other: &Point) -> f64 {
        (*self - *other).length()
    }
}
