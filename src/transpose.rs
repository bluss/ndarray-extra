
use ndarray::prelude::*;
use std::cmp;
use std::mem;
use std::fmt::Debug;

pub type Ix2 = (Ix, Ix);

trait Order {
    fn inner_contig() -> bool;
}

enum COrder { }
enum MixedOrder { }

impl Order for COrder {
    fn inner_contig() -> bool { true }
}
impl Order for MixedOrder {
    fn inner_contig() -> bool { false }
}

pub fn transpose<'a, A, V>(view: V)
    where A: 'a + Debug + Send,
          V: Into<ArrayViewMut<'a, A, Ix2>>,
{
    // divide and conquer algo:
    // divide in quadrants
    // [ A B ]
    // [ C D ]
    //
    // A and D, transpose each
    // B and C, transpose each and swap B <-> C
    let v = view.into();
    let (s0, s1) = (v.strides()[0], v.strides()[1]);
    if s1 == 1 {
        trans_outer::<_, COrder>(v);
    } else if s0 == 1 {
        trans_outer::<_, COrder>(v.reversed_axes());
    } else {
        trans_outer::<_, MixedOrder>(v);
    }
}

#[inline(never)]
fn trans_outer<A, O>(mut v: ArrayViewMut<A, Ix2>)
    where A: Send + Debug,
          O: Order,
{
    //println!("transpose of shape {:?}", v.shape());
    //println!("{:.2?}", v);
    let (mut m, n) = v.dim();
    debug_assert_eq!(n, m);
    if m < 21 {
        trans_base_case::<_, O>(m, v);
    } else {
        // uneven case
        if m & 1 == 1 {
            m -= 1;
            // split off a single column & row
            let (top, bottom) = v.split_at(Axis(0), m);
            let (left, right) = top.split_at(Axis(1), m);
            swap_rows::<_, O>(bottom, right);
            v = left;
        }
        // split in quadrants
        let (left, right) = v.split_at(Axis(1), m / 2);
        let (a, mut c) = left.split_at(Axis(0), m / 2);
        let (mut b, d) = right.split_at(Axis(0), m / 2);

        /*
        rayon::join(|| trans_outer(a),
                    || rayon::join(
                        || trans_outer(d),
                        || {
                rayon::join(|| trans_outer(b.view_mut()),
                            || trans_outer(c.view_mut()));
                swap_views(b, c);
            }));
        */
        trans_outer::<_, O>(a);
        trans_outer::<_, O>(b.view_mut());
        trans_outer::<_, O>(d);
        trans_outer::<_, O>(c.view_mut());
        swap_views::<_, _, O>(b, c);
        /*
        */
    }
}

#[inline(never)]
fn trans_base_case<A, O>(m: Ix, mut v: ArrayViewMut<A, Ix2>)
    where O: Order,
{
    for i in 0..m - 1 {
        if O::inner_contig() {
            unsafe {
                let mut ptr1: *mut _ = v.uget_mut((i, i + 1));
                for j in i + 1..m {
                    let ptr2: *mut _ = v.uget_mut((j, i));
                    mem::swap(&mut *ptr1, &mut *ptr2);
                    ptr1 = ptr1.offset(1);
                }
            }
        } else {
            unsafe {
                for j in i + 1..m {
                    let ptr1: *mut _ = v.uget_mut((i, j));
                    let ptr2: *mut _ = v.uget_mut((j, i));
                    mem::swap(&mut *ptr1, &mut *ptr2);
                }
            }
        }
    }
}

#[inline(never)]
/// arrays must be same size
fn swap_views<A, D, O>(mut a: ArrayViewMut<A, D>, mut b: ArrayViewMut<A, D>)
    where D: Dimension,
          O: Order,
{
    debug_assert_eq!(a.shape(), b.shape());
    let rows = a.inner_iter_mut().zip(b.inner_iter_mut());
    for (mut r1, mut r2) in rows {
        if O::inner_contig() {
            if let Some(rs1) = r1.as_slice_mut() {
                if let Some(rs2) = r2.as_slice_mut() {
                    swap_slices(rs1, rs2);
                }
            }
        } else {
            for i in 0..r1.len() {
                unsafe {
                    mem::swap(r1.uget_mut(i), r2.uget_mut(i));
                }
            }
        }
    }
}

// b must be the shorter of the two rows
#[inline(never)]
fn swap_rows<A, O>(mut a: ArrayViewMut<A, Ix2>, mut b: ArrayViewMut<A, Ix2>)
    where O: Order
{
    debug_assert_eq!(a.len(), b.len() + 1);
    let len = b.len();
    if O::inner_contig() {
        let mut aptr = a.as_mut_ptr();
        for i in 0..len {
            unsafe {
                mem::swap(&mut *aptr, b.uget_mut((i, 0)));
                aptr = aptr.offset(1);
            }
        }
    } else {
        for i in 0..len {
            unsafe {
                mem::swap(a.uget_mut((0, i)), b.uget_mut((i, 0)));
            }
        }
    }
}

#[inline(never)]
fn swap_slices<A>(a: &mut [A], b: &mut [A]) {
    let len = cmp::min(a.len(), b.len());
    let a = &mut a[..len];
    let b = &mut b[..len];
    for i in 0..len {
        mem::swap(&mut a[i], &mut b[i]);
    }
}

#[test]
fn test_small_transpose() {
    let (m, n) = (8, 8);
    let mut data = Array::linspace(0., 1., m * n).into_shape((m, n)).unwrap();

    println!("{:.2}", data);
    transpose(&mut data);
    println!("{:.2}", data);

    let mut data = data.reversed_axes();
    println!("{:.2}", data);
    transpose(&mut data);
    println!("{:.2}", data);
}

#[test]
fn test_uneven_transpose() {
    let (m, n) = (61, 61);
    let mut data = Array::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    let old = data.to_owned();
    transpose(&mut data);
    transpose(&mut data);
    assert_eq!(data, old);

    let (m, n) = (127, 127);
    let mut data = Array::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    let old = data.to_owned();
    transpose(&mut data);
    transpose(&mut data);
    assert_eq!(data, old);

}
#[test]
fn test_stride_transpose() {
    let (m, n) = (8, 12);
    let mut data = Array::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    {
        let mut v = data.slice_mut(s![..;2, ..;3]);
        transpose(&mut v);
    }
    println!("{:6.4?}", data);

    let (m, n) = (127 * 2, 127 * 3);
    let mut data = Array::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    let old1 = data.to_owned();
    let old2 = data.slice_mut(s![..;2, ..;3]).to_owned();
    {
        let mut v = data.slice_mut(s![..;2, ..;3]);
        transpose(&mut v);
    }
    // check that the other regions of the array are unchanged
    assert_eq!(old1.slice(s![1..;2, 1..;3]), data.slice(s![1..;2, 1..;3]));
    assert_eq!(old1.slice(s![1..;2, 2..;3]), data.slice(s![1..;2, 2..;3]));
    {
        let mut v = data.slice_mut(s![..;2, ..;3]);
        transpose(&mut v);
        assert_eq!(old2, v);
    }
    assert_eq!(old1, data);
}

#[test]
fn test_large_transpose() {
    let (m, n) = (1024 << 2, 1024 << 2);
    let data = OwnedArray::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    let mut data = data.reversed_axes();

    //println!("{:.2}", data);
    transpose(&mut data);
    //println!("{:.2}", data);
}

macro_rules! transpose_bench {
    ($name:ident, $ty:ty, $n:expr) => {
#[cfg(test)]
mod $name {
    use ndarray::prelude::*;
    use test::Bencher;
    use super::*;
    use std::mem::size_of;
    const N: usize = $n;
    #[bench]
    fn transpose_c(b: &mut Bencher) {
        let mut input = OwnedArray::linspace(0. as $ty, 1., N * N).into_shape((N, N)).unwrap();
        b.iter(|| {
            transpose(&mut input);
        });
        b.bytes = (size_of::<$ty>() * input.len()) as u64;
    }
    /*
    #[bench]
    fn transpose_f(b: &mut Bencher) {
        let mut input = OwnedArray::linspace(0. as $ty, 1., N * N).into_shape((N, N)).unwrap();
        input = input.reversed_axes();
        b.iter(|| {
            transpose(&mut input);
        });
        b.bytes = (size_of::<$ty>() * input.len()) as u64;
    }

    #[bench]
    fn transpose_quarter(b: &mut Bencher) {
        let mut input = OwnedArray::linspace(0. as $ty, 1., N * N).into_shape((N, N)).unwrap();
        let mut v = input.slice_mut(s![..N as isize/2, ..N as isize/2]);
        b.iter(|| {
            transpose(&mut v);
        });
        b.bytes = (size_of::<$ty>() * v.len()) as u64;
    }
    */
    #[bench]
    fn transpose_stride2(b: &mut Bencher) {
        let mut input = OwnedArray::linspace(0. as $ty, 1., N * N).into_shape((N, N)).unwrap();
        let mut v = input.slice_mut(s![..;2, ..;2]);
        b.iter(|| {
            transpose(&mut v);
        });
        b.bytes = (size_of::<$ty>() * v.len()) as u64;
    }
}
    }
}

transpose_bench!{f32_020, f32, 20}
transpose_bench!{f32_063, f32, 63}
transpose_bench!{f32_127, f32, 127}
transpose_bench!{f32_128, f32, 128}
transpose_bench!{f32_400, f32, 400}
transpose_bench!{f32_999, f32, 999}
/*
transpose_bench!{f64_020, f64, 20}
transpose_bench!{f64_063, f64, 63}
transpose_bench!{f64_128, f64, 128}
transpose_bench!{f64_400, f64, 400}
transpose_bench!{f64_999, f64, 999}
*/
