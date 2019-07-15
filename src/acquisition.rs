pub trait AcquisitionStrategy<A> {
}

pub struct MutationAcquisition {
    pub breadth: usize,
}

impl<A> AcquisitionStrategy<A> for MutationAcquisition {
}
