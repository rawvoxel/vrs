//! Viewing plain data as bytes, for push constants and buffer uploads.

use std::slice;

/// Only for plain data without implicit padding, such as `Vertex`, `Index` and `MeshInfo`.
#[inline]
pub fn slice_as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    // SAFETY: Callers only pass plain data without implicit padding, so every byte is initialized.
    unsafe { slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

/// See `slice_as_bytes`.
#[inline]
pub fn as_bytes<T: Copy>(value: &T) -> &[u8] {
    slice_as_bytes(slice::from_ref(value))
}

/// Only for types made of `u32`s and nothing else, such as `mesh::PackedVertex`.
#[inline]
pub fn slice_as_words<T: Copy>(data: &[T]) -> &[u32] {
    const {
        assert!(
            size_of::<T>().is_multiple_of(size_of::<u32>()) && align_of::<T>() >= align_of::<u32>()
        );
    }
    // SAFETY: `T` is whole, aligned words, every one of them initialized.
    unsafe { slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data) / size_of::<u32>()) }
}
