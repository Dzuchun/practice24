use core::panic;
use std::ops::RangeBounds;

use super::{
    direction::Direction,
    distance::{DistanceFrom, DistanceTo},
    intersect::IntersectFrom,
    line_segment::LineSegment,
    plane::Plane,
    point::Point,
    vector::Vector,
    TOLERANCE,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    start: Point,
    direction: Direction,
}

impl Line {
    pub fn through_points(p1: Point, p2: Point) -> Option<Self> {
        Some(Self {
            start: p1,
            direction: (p2 - p1).as_direction()?,
        })
    }

    fn offset_components(&self, point: Point) -> (Vector, Vector) {
        self.direction.split(point - self.start)
    }

    pub(super) fn coord_on(&self, point: Point) -> Option<f64> {
        let (para, trans) = self.offset_components(point);
        if trans.length() > TOLERANCE {
            return None;
        }
        let para_len = para.length();
        let sign = match para.as_direction().map(|dir| dir.cosine_to(self.direction)) {
            None => 0.0,
            Some(v) if v >= 0.0 => 1.0,
            Some(v) if v < 0.0 => -1.0,
            Some(_) => unreachable!(),
        };
        Some(para_len * sign)
    }

    pub(super) fn at_coord(&self, coord: f64) -> Point {
        self.start + self.direction * coord
    }

    pub fn perp(&self, point: Point) -> Vector {
        let (_, perp) = self.offset_components(point);
        perp
    }

    pub fn offset(&mut self, offset: Vector) {
        self.start += offset;
    }

    pub fn extend_to_point(&self, point: Point) -> Option<Plane> {
        let perp = self.perp(point);
        let base2 = perp.as_direction()?;
        Some(Plane {
            base1: self.direction,
            base2,
            point: self.start,
        })
    }

    pub fn extend_to_direction(&self, direction: Direction) -> Option<Plane> {
        let base2 = direction.orth(self.direction)?;
        Some(Plane {
            base1: self.direction,
            base2,
            point: self.start,
        })
    }

    pub fn closest_to_other_line(&self, &(mut other): &Self) -> Option<Point> {
        // lines are parallel, there's no answer
        let plane = self.extend_to_direction(other.direction)?;
        let perp = plane.perp(other.start);
        other.offset(-perp);
        let IntLineLine::Point(res) = self.intersect_from(&other) else {
            panic!(
                "Should intersect at a point now; distance: {}",
                self.distance_from(&other)
            );
        };
        Some(res)
    }

    pub(super) fn closest_to_point(&self, point: Point) -> Point {
        point - self.perp(point)
    }
}

impl DistanceFrom<Point> for Line {
    fn distance_from(&self, other: &Point) -> f64 {
        self.perp(*other).length()
    }
}

impl DistanceFrom<Line> for Line {
    fn distance_from(&self, other: &Line) -> f64 {
        // the gist is to define a plane through one of the lines that's parallel to the other
        //
        // that's only possible if lines are not parallel
        //
        // if they ARE parallel, use any distance
        if let Some(plane) = self.extend_to_direction(other.direction) {
            // lines are NOT parallel;
            plane.distance_from(&other.start)
        } else {
            // lines are equal or parallel
            self.distance_from(&other.start)
        }
    }
}

pub enum IntLineLine {
    Point(Point),
    Line(Line),
    Parallel(Direction),
    None,
}

impl IntersectFrom<Line> for Line {
    type Intersection = IntLineLine;

    fn intersect_from(&self, other: &Line) -> Self::Intersection {
        if let Some(plane) = other.extend_to_direction(self.direction) {
            // lines are NOT parallel
            //
            // find the intersection point by predicting it's coordinate
            let mut perp = plane.perp(self.start);
            let mut perplen = perp.length();
            if perplen < TOLERANCE {
                // lines are in the same plane; they intersect for sure
                perp = other.perp(self.start);
                perplen = perp.length();
            }
            let coordinate_projection = (self.direction * 1.0).dot(&-perp) / perplen;
            let coordinate = perplen / coordinate_projection;
            let maybe_intersection = self.at_coord(coordinate);
            if other.distance_from(&maybe_intersection) < TOLERANCE {
                IntLineLine::Point(maybe_intersection)
            } else {
                IntLineLine::None
            }
        } else {
            // lines are equal or parallel
            if other.start.distance_to(self) < TOLERANCE {
                // lines are equal
                IntLineLine::Line(*self)
            } else {
                // lines are parallel
                IntLineLine::Parallel(self.direction)
            }
        }
    }
}
