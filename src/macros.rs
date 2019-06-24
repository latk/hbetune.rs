#[macro_export]
macro_rules! assert_all_close {
    ($left:expr, $right:expr, $tol:expr) => (
        match ($left, $right, $tol) {
            (left, right, tol) =>
                assert!(left.all_close(&right, tol),
                        "expected left all close to right\n\
                         left: {}\n\
                         right: {}", left, right)

        }
    );
}
