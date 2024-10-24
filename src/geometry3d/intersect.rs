pub trait IntersectFrom<Other> {
    type Intersection;

    fn intersect_from(&self, other: &Other) -> Self::Intersection;
}

pub trait IntersectTo<Other> {
    type Intersection;

    fn intersect_to(&self, other: &Other) -> Self::Intersection;
}

// Intersection is commutative:
impl<A, B> IntersectTo<A> for B
where
    A: IntersectFrom<B>,
{
    type Intersection = <A as IntersectFrom<B>>::Intersection;

    fn intersect_to(&self, other: &A) -> Self::Intersection {
        other.intersect_from(self)
    }
}
