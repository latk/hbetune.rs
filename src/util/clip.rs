pub fn clip<T: PartialOrd>(value: T, min: Option<T>, max: Option<T>) -> T {
    if let Some(min) = min {
        if value < min {
            return min;
        }
    }

    if let Some(max) = max {
        if max < value {
            return max;
        }
    }

    value
}
