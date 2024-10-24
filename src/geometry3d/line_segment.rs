use std::ops::RangeInclusive;

use super::{
    distance::{DistanceFrom, DistanceTo},
    intersect::IntersectFrom,
    line::{IntLineLine, Line},
    point::Point,
    vector::Vector,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LineSegment {
    pub(super) line: Line,
    pub(super) range: RangeInclusive<f64>,
}

impl LineSegment {
    pub fn between_points(point1: Point, point2: Point) -> Option<Self> {
        let line = Line::through_points(point1, point2)?;
        let mut p1 = line.coord_on(point1).expect("Must be on the line");
        let mut p2 = line.coord_on(point2).expect("Must be on the line");
        assert_ne!(p1, p2, "Must not be the same point");
        if p2 < p1 {
            std::mem::swap(&mut p1, &mut p2);
        }
        Some(Self {
            line,
            range: p1..=p2,
        })
    }

    pub fn ends(&self) -> (Point, Point) {
        (
            self.line.at_coord(*self.range.start()),
            self.line.at_coord(*self.range.end()),
        )
    }

    pub fn offset(&mut self, offset: Vector) {
        self.line.offset(offset)
    }
}

impl DistanceFrom<Line> for LineSegment {
    fn distance_from(&self, other: &Line) -> f64 {
        if let Some(closest) = self.line.closest_to_other_line(other) {
            // there is a closest point!
            let coord = self.line.coord_on(closest).expect("Must be on the line");
            if self.range.contains(&coord) {
                closest.distance_to(other)
            } else {
                let (end1, end2) = self.ends();
                end1.distance_to(other).min(end2.distance_to(other))
            }
        } else {
            // lines are parallel, any distance will do
            self.ends().0.distance_to(other)
        }
    }
}

impl DistanceFrom<Point> for LineSegment {
    fn distance_from(&self, other: &Point) -> f64 {
        let closest = self.line.closest_to_point(*other);
        if self
            .range
            .contains(&self.line.coord_on(closest).expect("Should be on the line"))
        {
            closest.distance_to(other)
        } else {
            let (end1, end2) = self.ends();
            end1.distance_to(other).min(end2.distance_to(other))
        }
    }
}

impl DistanceFrom<LineSegment> for LineSegment {
    fn distance_from(&self, other: &LineSegment) -> f64 {
        if let Some(self_closest) = self.line.closest_to_other_line(&other.line) {
            if self.range.contains(
                &self
                    .line
                    .coord_on(self_closest)
                    .expect("Should be on the line"),
            ) {
                let other_closest = other.line.closest_to_point(self_closest);
                if other.range.contains(
                    &other
                        .line
                        .coord_on(other_closest)
                        .expect("Must be on the line"),
                ) {
                    // common perpendicular is the answer
                    return self_closest.distance_to(&other_closest);
                }
            }
        }
        // segments are parallel, equal or not aligned
        //
        // best we can do - try to project their ends onto one another
        fn try_project(this: &LineSegment, point: Point) -> bool {
            let closest = this.line.closest_to_point(point);
            this.range
                .contains(&this.line.coord_on(closest).expect("Must be on the line"))
        }
        let (self1, self2) = self.ends();
        let (other1, other2) = other.ends();
        if try_project(self, other1)
            || try_project(self, other2)
            || try_project(other, self1)
            || try_project(other, self2)
        {
            // segments are aligned!
            self1
                .distance_to(other)
                .min(self1.distance_to(other))
                .min(other.distance_from(&self1))
                .min(other.distance_from(&self2))
        } else {
            self1
                .distance_to(&other1)
                .min(self1.distance_to(&other2))
                .min(self2.distance_to(&other1))
                .min(self2.distance_to(&other2))
        }
    }
}

pub enum IntLineLineSegment {
    Point(Point),
    LineSegment(LineSegment),
    None,
}

impl IntersectFrom<Line> for LineSegment {
    type Intersection = IntLineLineSegment;

    fn intersect_from(&self, other: &Line) -> Self::Intersection {
        match self.line.intersect_from(other) {
            IntLineLine::Point(p) => {
                let coord = self.line.coord_on(p).expect("Should be on the line");
                if self.range.contains(&coord) {
                    IntLineLineSegment::Point(p)
                } else {
                    IntLineLineSegment::None
                }
            }
            IntLineLine::Line(_) => IntLineLineSegment::LineSegment(self.clone()),
            IntLineLine::Parallel(_) | IntLineLine::None => IntLineLineSegment::None,
        }
    }
}

impl IntersectFrom<LineSegment> for LineSegment {
    type Intersection = IntLineLineSegment;

    fn intersect_from(&self, other: &LineSegment) -> Self::Intersection {
        match self.line.intersect_from(&other.line) {
            IntLineLine::Point(p) => {
                let coord1 = self.line.coord_on(p).expect("Must be on the line");
                let coord2 = other.line.coord_on(p).expect("Must be on the line");
                if self.range.contains(&coord1) && other.range.contains(&coord2) {
                    IntLineLineSegment::Point(p)
                } else {
                    IntLineLineSegment::None
                }
            }
            IntLineLine::Line(_) => {
                // segments are parallel
                let start = self.range.start().max(*other.range.start());
                let end = self.range.end().min(*other.range.end());
                if start < end {
                    IntLineLineSegment::LineSegment(LineSegment {
                        line: self.line,
                        range: start..=end,
                    })
                } else {
                    IntLineLineSegment::None
                }
            }
            IntLineLine::Parallel(_) | IntLineLine::None => IntLineLineSegment::None,
        }
    }
}

impl AsRef<Line> for LineSegment {
    fn as_ref(&self) -> &Line {
        &self.line
    }
}

impl AsRef<RangeInclusive<f64>> for LineSegment {
    fn as_ref(&self) -> &RangeInclusive<f64> {
        &self.range
    }
}
