use std::ops::Mul;

use super::vector::Vector;

#[derive(Debug, Clone, Copy, PartialEq, derive_more::Neg)]
pub struct Direction {
    pub(super) dx: f64,
    pub(super) dy: f64,
    pub(super) dz: f64,
}

impl Direction {
    pub fn cosine_to(self, other: Direction) -> f64 {
        self.dx * other.dx + self.dy * other.dy + self.dz * other.dz
    }

    pub fn split(&self, vec: Vector) -> (Vector, Vector) {
        let dot = vec.vx * self.dx + vec.vy * self.dy + vec.vz * self.dz;
        let para = *self * dot;
        let trans = vec - para;
        (para, trans)
    }

    pub fn orth(&self, other: Direction) -> Option<Direction> {
        let (_, trans) = other.split(*self * 1.0);
        trans.as_direction()
    }
}

impl Mul<f64> for Direction {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector {
            vx: self.dx * rhs,
            vy: self.dy * rhs,
            vz: self.dz * rhs,
        }
    }
}
