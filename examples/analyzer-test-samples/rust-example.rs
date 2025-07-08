// Example Rust code demonstrating various patterns for analyzer testing

use std::mem;
use std::rc::Rc;
use std::thread;
use rand::random;

// Hardcoded secrets - security issue
const API_KEY: &str = "sk_live_4242424242424242";
const SECRET_TOKEN: &str = "super-secret-token-123";

// Static mutable - not thread-safe
static mut GLOBAL_COUNTER: i32 = 0;

// Missing documentation for public function
pub fn public_function_without_docs() {
    println!("This should have documentation");
}

// Unsafe code examples
fn dangerous_operations() {
    unsafe {
        // Raw pointer manipulation
        let address = 0x012345usize;
        let raw_ptr = address as *const i32;
        let value = *raw_ptr; // Dereferencing arbitrary memory
        
        // Dangerous transmute
        let float_bits: u32 = mem::transmute(1.0f32);
        
        // Unchecked array access
        let vec = vec![1, 2, 3];
        let elem = vec.get_unchecked(10); // Out of bounds
    }
}

// Unsafe function without clear safety requirements
unsafe fn unsafe_function(ptr: *mut i32) {
    *ptr = 42; // No null check or validity guarantee
}

// Use after free potential
fn memory_safety_issue() {
    let data = vec![1, 2, 3];
    let reference = &data[0];
    
    drop(data); // Explicitly drop data
    
    // This would be use-after-free if it compiled
    // println!("{}", reference);
}

// Memory leak via Box::into_raw
fn potential_memory_leak() {
    let boxed_value = Box::new(vec![1; 1000]);
    let raw_ptr = Box::into_raw(boxed_value);
    // Missing Box::from_raw to reclaim memory
}

// Multiple mutable borrows (wouldn't compile, but pattern to detect)
fn borrowing_issues() {
    let mut vec = vec![1, 2, 3];
    let borrow1 = &mut vec;
    let borrow2 = &mut vec; // Second mutable borrow
}

// Excessive cloning
fn clone_heavy_code() {
    let large_vec = vec![0u8; 1_000_000];
    let clone1 = large_vec.clone();
    let clone2 = large_vec.clone();
    let clone3 = large_vec.clone();
    let clone4 = large_vec.clone();
    let clone5 = large_vec.clone();
    let clone6 = large_vec.clone();
    // Too many clones - consider Rc/Arc
}

// Thread safety issues
fn concurrency_problems() {
    // Static mutable access without synchronization
    unsafe {
        GLOBAL_COUNTER += 1;
    }
    
    // Rc in multi-threaded context (wouldn't compile)
    let data = Rc::new(vec![1, 2, 3]);
    // thread::spawn(move || {
    //     println!("{:?}", data); // Rc is not Send
    // });
}

// Custom type that might be used across threads
struct DataContainer {
    data: Vec<i32>,
}
// Missing explicit Send/Sync implementation documentation

// Weak cryptography
fn insecure_random() {
    // Using non-cryptographic RNG for security-sensitive operations
    let key: u64 = random(); // Not cryptographically secure
    let token = rand::thread_rng().gen::<u32>();
    
    // Hardcoded password
    let password = "admin123";
    authenticate(password);
}

// Error handling issues
fn poor_error_handling() -> i32 {
    let result = some_operation();
    let value = result.unwrap(); // Can panic
    
    let file_content = std::fs::read_to_string("config.toml")
        .expect("Failed to read config"); // Can panic
    
    value
}

// Unchecked type conversions
fn unsafe_conversions() {
    let large_num: u64 = std::u64::MAX;
    let truncated = large_num as u32; // Loses data
    let might_overflow = large_num as usize; // Platform-dependent
}

// Panic in library code
pub fn library_function_that_panics(input: &str) -> Result<i32, String> {
    if input.is_empty() {
        panic!("Input cannot be empty!"); // Should return Err instead
    }
    
    unimplemented!("This function is not implemented"); // In production code
}

// Performance issues
fn allocation_heavy() {
    // Unnecessary String allocation
    let s = "hello".to_string().as_str();
    
    // Collect just to get length
    let count = (0..100).collect::<Vec<_>>().len();
    
    // Vec without pre-allocation
    let mut vec = Vec::new();
    for i in 0..10000 {
        vec.push(i); // Many reallocations
    }
}

// String operations in loop
fn inefficient_string_building() -> String {
    let mut result = String::new();
    
    for i in 0..1000 {
        result.push_str(&format!("Item {}\n", i)); // Reallocation on each iteration
    }
    
    result
}

// Inefficient iterator usage
fn bad_iterator_patterns(data: &[i32]) -> bool {
    // Using filter().count() == 0
    if data.iter().filter(|&&x| x > 100).count() == 0 {
        return true;
    }
    
    // Unnecessary collect and into_iter
    let _processed = data.iter()
        .map(|&x| x * 2)
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|&x| x > 10);
    
    false
}

// HashMap without capacity
fn create_large_map() -> std::collections::HashMap<String, i32> {
    let mut map = std::collections::HashMap::new(); // No capacity hint
    
    for i in 0..10000 {
        map.insert(format!("key_{}", i), i);
    }
    
    map
}

// Blocking operations in async context
async fn blocking_in_async() {
    // Blocking sleep in async function
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // Blocking file I/O
    let _contents = std::fs::read_to_string("file.txt").unwrap();
}

// Missing tests
fn complex_business_logic(input: Vec<i32>) -> i32 {
    // Complex implementation without tests
    input.iter()
        .filter(|&&x| x > 0)
        .map(|&x| x * x)
        .fold(0, |acc, x| acc + x)
}

// Naming convention issues
const not_screaming_case: i32 = 42; // Should be NOT_SCREAMING_CASE

// Empty match arms
fn process_option(opt: Option<i32>) {
    match opt {
        Some(value) => println!("{}", value),
        None => {} // Empty arm - could use if let
    }
}

// Redundant code patterns
fn redundant_patterns() {
    // Redundant clone
    let s = String::from("hello");
    let _s2 = s.clone().clone();
    
    // Unnecessary type annotations
    let x: i32 = 42i32;
    
    // Verbose option handling
    let opt = Some(5);
    let _value = match opt {
        Some(v) => v,
        None => 0,
    }; // Could use unwrap_or
}

fn main() {
    println!("Rust analyzer test file");
}