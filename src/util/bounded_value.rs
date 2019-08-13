/// A value with some bounds, e.g. a parameter that can be tuned.
#[derive(Clone, PartialEq)]
pub struct BoundedValue<A: PartialOrd + Clone> {
    value: A,
    max: A,
    min: A,
}

impl<A: PartialOrd + Clone> BoundedValue<A>
where
    A: PartialEq + Copy,
{
    /// Create a new value with bounds (inclusive).
    pub fn new(value: A, min: A, max: A) -> Result<Self, BoundsError<A>> {
        if min <= value && value <= max {
            Ok(BoundedValue { value, min, max })
        } else {
            Err(BoundsError { value, min, max })
        }
    }

    /// Get the current value.
    pub fn value(&self) -> A {
        self.value
    }

    /// Get the lower bound.
    pub fn min(&self) -> A {
        self.min
    }

    /// Get the upper bound.
    pub fn max(&self) -> A {
        self.max
    }

    /// Set a new value.
    pub fn with_value(self, value: A) -> Result<Self, BoundsError<A>> {
        Self::new(value, self.min, self.max)
    }

    /// Set a new value. If bounds are violated, substitute the bound instead.
    pub fn with_clamped_value(self, value: A) -> Self {
        let value = if value < self.min {
            self.min
        } else if self.max < value {
            self.max
        } else {
            value
        };
        Self {
            value,
            min: self.min,
            max: self.max,
        }
    }
}

impl<A> std::fmt::Debug for BoundedValue<A>
where
    A: PartialOrd + Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "BoundedValue({:?} in {:?} .. {:?})",
            self.value, self.min, self.max,
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct BoundsError<A> {
    pub value: A,
    pub min: A,
    pub max: A,
}
