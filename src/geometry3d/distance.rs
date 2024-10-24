pub trait DistanceTo<Other> {
    fn distance_to(&self, other: &Other) -> f64;
}

pub trait DistanceFrom<Other> {
    fn distance_from(&self, other: &Other) -> f64;
}

// Distance is commutative:
impl<A, B> DistanceTo<A> for B
where
    A: DistanceFrom<B>,
{
    fn distance_to(&self, other: &A) -> f64 {
        other.distance_from(self)
    }
}
