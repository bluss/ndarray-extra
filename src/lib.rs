#![cfg_attr(test, feature(test))]

#[macro_use(s)]
extern crate ndarray;

#[cfg(test)]
extern crate test;

pub mod transpose;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
