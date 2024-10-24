use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::{direction::Direction, hypot, point::Point, TOLERANCE};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::Sub,
    derive_more::SubAssign,
    derive_more::Neg,
)]
pub struct Vector {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl Vector {
    pub fn length(&self) -> f64 {
        hypot(self.vx, self.vy, self.vz)
    }

    pub fn as_direction(&self) -> Option<Direction> {
        let len = self.length();
        if len < TOLERANCE {
            return None;
        }
        Some(Direction {
            dx: self.vx / len,
            dy: self.vy / len,
            dz: self.vz / len,
        })
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        self.vx * other.vx + self.vy * other.vy + self.vz * other.vz
    }

    pub fn mixed(&self, other1: Vector, other2: Vector) -> f64 {
        det3!(
            [self.vx, self.vy, self.vz],
            [other1.vx, other1.vy, other1.vz],
            [other2.vx, other2.vy, other2.vz]
        )
    }

    pub fn split(self, direction: Direction) -> (Vector, Vector) {
        direction.split(self)
    }
}

impl Mul<f64> for Vector {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.vx *= rhs;
        self.vy *= rhs;
        self.vz *= rhs;
        self
    }
}

impl MulAssign<f64> for Vector {
    fn mul_assign(&mut self, rhs: f64) {
        self.vx *= rhs;
        self.vy *= rhs;
        self.vz *= rhs;
    }
}

impl Div<f64> for Vector {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self::Output {
        self.vx /= rhs;
        self.vy /= rhs;
        self.vz /= rhs;
        self
    }
}

impl DivAssign<f64> for Vector {
    fn div_assign(&mut self, rhs: f64) {
        self.vx /= rhs;
        self.vy /= rhs;
        self.vz /= rhs;
    }
}

impl Add<Vector> for Point {
    type Output = Point;

    fn add(mut self, rhs: Vector) -> Self::Output {
        self.x += rhs.vx;
        self.y += rhs.vy;
        self.z += rhs.vz;
        self
    }
}

impl AddAssign<Vector> for Point {
    fn add_assign(&mut self, rhs: Vector) {
        self.x += rhs.vx;
        self.y += rhs.vy;
        self.z += rhs.vz;
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    fn sub(mut self, rhs: Vector) -> Self::Output {
        self.x -= rhs.vx;
        self.y -= rhs.vy;
        self.z -= rhs.vz;
        self
    }
}

impl SubAssign<Vector> for Point {
    fn sub_assign(&mut self, rhs: Vector) {
        self.x -= rhs.vx;
        self.y -= rhs.vy;
        self.z -= rhs.vz;
    }
}
