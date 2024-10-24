use super::{
    direction::Direction, distance::DistanceFrom, line::Line, point::Point, vector::Vector,
    TOLERANCE,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub(super) base1: Direction,
    pub(super) base2: Direction,
    pub(super) point: Point,
}

impl Plane {
    pub fn trough_points(p1: Point, p2: Point, p3: Point) -> Option<Self> {
        Self::through_line_and_point(p3, Line::through_points(p1, p2)?)
    }

    pub fn through_line_and_point(p: Point, line: Line) -> Option<Self> {
        line.extend_to_point(p)
    }

    fn offset_components(&self, point: Point) -> (Vector, Vector, Vector) {
        let (para1, trans) = self.base1.split(point - self.point);
        let (para2, perp) = self.base2.split(trans);
        (para1, para2, perp)
    }

    pub fn perp(&self, point: Point) -> Vector {
        let (_, _, perp) = self.offset_components(point);
        perp
    }

    pub(super) fn at_coord(&self, coord1: f64, coord2: f64) -> Point {
        self.point + self.base1 * coord1 + self.base2 * coord2
    }

    pub(super) fn coord_on(&self, point: Point) -> Option<(f64, f64)> {
        let (para1, para2, perp) = self.offset_components(point);
        if perp.length() > TOLERANCE {
            return None;
        }
        Some((para1.length(), para2.length()))
    }
}

impl DistanceFrom<Point> for Plane {
    fn distance_from(&self, other: &Point) -> f64 {
        self.perp(*other).length()
    }
}
