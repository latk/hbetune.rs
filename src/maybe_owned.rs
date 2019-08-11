pub enum MaybeOwned<'life, T: ?Sized + 'life> {
    Borrowed(&'life mut T),
    Owned(Box<T>),
}

use MaybeOwned::*;

impl<'life, T: ?Sized> std::ops::Deref for MaybeOwned<'life, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Borrowed(x) => x,
            Owned(ref x) => x.deref(),
        }
    }
}

impl<'life, T: ?Sized> std::ops::DerefMut for MaybeOwned<'life, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Borrowed(x) => x,
            Owned(ref mut x) => x.deref_mut(),
        }
    }
}

impl<'life, T: ?Sized> AsRef<T> for MaybeOwned<'life, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'life, T: ?Sized> AsMut<T> for MaybeOwned<'life, T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}
