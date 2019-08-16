//use std::ffi::CString;
//use std::os::raw::c_char;
//
//extern "C" {
//    fn Rprintf(fmt: *const c_char);
//}
//
//pub fn rprintf(x: &String) {
//    let x2 = CString::new(&x[..]).expect("Invalid string.");
//    unsafe {
//        Rprintf(x2.as_ptr());
//    }
//}

pub fn rprintf(x: &String) {
    println!("{}", x);
}
