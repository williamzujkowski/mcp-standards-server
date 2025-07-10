"""Tests for Rust language analyzer."""

import pytest

from src.analyzers.base import IssueType
from src.analyzers.rust_analyzer import RustAnalyzer


class TestRustAnalyzer:
    """Test Rust analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create Rust analyzer instance."""
        return RustAnalyzer()

    def test_language_properties(self, analyzer):
        """Test language and file extension properties."""
        assert analyzer.language == "rust"
        assert analyzer.file_extensions == [".rs"]

    def test_unsafe_code_detection(self, analyzer, tmp_path):
        """Test unsafe code block detection."""
        test_file = tmp_path / "unsafe_code.rs"
        test_file.write_text(
            """
fn dangerous_function() {
    unsafe {
        let raw_ptr = 0x1234 as *const i32;
        let value = *raw_ptr;  // Dereferencing raw pointer
    }

    unsafe {
        let x: i32 = std::mem::transmute(1.0f32);  // transmute
    }

    let vec = vec![1, 2, 3];
    unsafe {
        let elem = vec.get_unchecked(10);  // Unchecked access
    }
}

unsafe fn unsafe_function(ptr: *const u8) -> u8 {
    *ptr
}
"""
        )

        result = analyzer.analyze_file(test_file)

        unsafe_issues = [
            i
            for i in result.issues
            if i.type == IssueType.MEMORY_SAFETY
            or (hasattr(i, "cwe_id") and i.cwe_id in ["CWE-824", "CWE-843", "CWE-125"])
        ]
        assert len(unsafe_issues) >= 3

        # Check for specific unsafe patterns
        assert any("raw pointer" in i.message.lower() for i in unsafe_issues)
        assert any("transmute" in i.message.lower() for i in unsafe_issues)
        assert any("unchecked" in i.message.lower() for i in unsafe_issues)

    def test_use_after_free_detection(self, analyzer, tmp_path):
        """Test use-after-free detection."""
        test_file = tmp_path / "memory_safety.rs"
        test_file.write_text(
            """
fn use_after_drop() {
    let mut data = vec![1, 2, 3];
    let ptr = &data[0];

    drop(data);

    println!("{}", data.len());  // Use after drop
}

fn box_leak() {
    let boxed = Box::new(42);
    let raw = Box::into_raw(boxed);
    // Missing Box::from_raw to reclaim memory
}
"""
        )

        result = analyzer.analyze_file(test_file)

        memory_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-416"
        ]
        assert len(memory_issues) >= 1

        leak_issues = [i for i in result.issues if "leak" in i.message.lower()]
        assert len(leak_issues) >= 1

    def test_ownership_and_borrowing(self, analyzer, tmp_path):
        """Test ownership and borrowing issue detection."""
        test_file = tmp_path / "ownership.rs"
        test_file.write_text(
            """
fn multiple_mutable_borrows() {
    let mut data = vec![1, 2, 3];
    let ref1 = &mut data;
    let ref2 = &mut data;  // Second mutable borrow

    ref1.push(4);
    ref2.push(5);
}

fn excessive_cloning() {
    let data = vec![1; 1000];
    let copy1 = data.clone();
    let copy2 = data.clone();
    let copy3 = data.clone();
    let copy4 = data.clone();
    let copy5 = data.clone();
    let copy6 = data.clone();
}
"""
        )

        result = analyzer.analyze_file(test_file)

        borrow_issues = [
            i for i in result.issues if "mutable borrow" in i.message.lower()
        ]
        assert len(borrow_issues) >= 1

        clone_issues = [i for i in result.issues if "clone" in i.message.lower()]
        assert len(clone_issues) >= 1

    def test_concurrency_safety(self, analyzer, tmp_path):
        """Test concurrency safety detection."""
        test_file = tmp_path / "concurrency.rs"
        test_file.write_text(
            """
use std::thread;
use std::rc::Rc;

static mut COUNTER: i32 = 0;

fn unsafe_static() {
    unsafe {
        COUNTER += 1;  // Static mutable is not thread-safe
    }
}

fn wrong_smart_pointer() {
    let data = Rc::new(vec![1, 2, 3]);  // Rc in multi-threaded context

    thread::spawn(move || {
        println!("{:?}", data);
    });
}

struct CustomType {
    data: Vec<i32>,
}

// Missing Send/Sync implementation for type used across threads
"""
        )

        result = analyzer.analyze_file(test_file)

        static_mut_issues = [
            i for i in result.issues if "static mutable" in i.message.lower()
        ]
        assert len(static_mut_issues) >= 1

        rc_issues = [
            i
            for i in result.issues
            if "Rc" in i.message and "thread" in i.message.lower()
        ]
        assert len(rc_issues) >= 1

    def test_crypto_and_secrets(self, analyzer, tmp_path):
        """Test cryptographic and secret handling issues."""
        test_file = tmp_path / "crypto.rs"
        test_file.write_text(
            """
use rand::random;

const API_KEY: &str = "sk_live_1234567890abcdef";
const SECRET_TOKEN: &str = "super-secret-token-12345";

fn weak_random_for_crypto() {
    let key: u64 = random();  // Not crypto-secure
    let token: u32 = rand::thread_rng().gen();  // Also not crypto-secure
}

fn hardcoded_password() {
    let password = "admin123";
    authenticate(password);
}
"""
        )

        result = analyzer.analyze_file(test_file)

        # Check for hardcoded secrets
        secret_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-798"
        ]
        assert len(secret_issues) >= 2

        # Check for weak RNG
        rng_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-338"
        ]
        assert len(rng_issues) >= 1

    def test_error_handling(self, analyzer, tmp_path):
        """Test error handling patterns."""
        test_file = tmp_path / "errors.rs"
        test_file.write_text(
            """
fn uses_unwrap() {
    let result = some_operation();
    let value = result.unwrap();  // Can panic
}

fn uses_expect() {
    let file = std::fs::read_to_string("file.txt").expect("Failed to read");
}

fn unchecked_conversion() {
    let big_num: u64 = 1_000_000;
    let small_num = big_num as usize;  // Unchecked conversion
}

// Library code should not panic
pub fn library_function() {
    panic!("This should return Result");
}

pub fn unfinished_function() {
    unimplemented!("TODO: implement this");
}
"""
        )

        result = analyzer.analyze_file(test_file)

        unwrap_issues = [i for i in result.issues if "unwrap" in i.message.lower()]
        assert len(unwrap_issues) >= 1

        panic_issues = [i for i in result.issues if "panic" in i.message.lower()]
        assert len(panic_issues) >= 1

        conversion_issues = [
            i for i in result.issues if "conversion" in i.message.lower()
        ]
        assert len(conversion_issues) >= 1

    def test_performance_allocations(self, analyzer, tmp_path):
        """Test allocation performance issues."""
        test_file = tmp_path / "performance.rs"
        test_file.write_text(
            """
fn unnecessary_allocations() {
    let s = "hello".to_string().as_str();  // Unnecessary String allocation

    let count = vec![1, 2, 3, 4, 5].into_iter().collect::<Vec<_>>().len();
}

fn vec_push_without_capacity() {
    let mut vec = Vec::new();
    for i in 0..1000 {
        vec.push(i);  // Multiple reallocations
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        alloc_issues = [i for i in result.issues if "allocation" in i.message.lower()]
        assert len(alloc_issues) >= 2

    def test_iterator_performance(self, analyzer, tmp_path):
        """Test iterator usage patterns."""
        test_file = tmp_path / "iterators.rs"
        test_file.write_text(
            """
fn inefficient_iterator_usage() {
    let data = vec![1, 2, 3, 4, 5];

    // Inefficient filter().count() == 0
    if data.iter().filter(|&x| x > 10).count() == 0 {
        println!("No items > 10");
    }

    // Unnecessary collect and into_iter
    let processed = data.iter()
        .map(|x| x * 2)
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|x| x > 5);
}
"""
        )

        result = analyzer.analyze_file(test_file)

        iter_issues = [i for i in result.issues if i.type == IssueType.PERFORMANCE]
        assert len(iter_issues) >= 2

    def test_async_patterns(self, analyzer, tmp_path):
        """Test async/await patterns."""
        test_file = tmp_path / "async_code.rs"
        test_file.write_text(
            """
async fn blocking_in_async() {
    std::thread::sleep(std::time::Duration::from_secs(1));  // Blocking in async

    let contents = std::fs::read_to_string("file.txt").unwrap();  // Blocking I/O
}

async fn missing_await() {
    let future = async_operation();
    // Missing .await
}
"""
        )

        result = analyzer.analyze_file(test_file)

        blocking_issues = [i for i in result.issues if "blocking" in i.message.lower()]
        assert len(blocking_issues) >= 2

    def test_best_practices(self, analyzer, tmp_path):
        """Test Rust best practices."""
        test_file = tmp_path / "best_practices.rs"
        test_file.write_text(
            """
const lowercase_const: i32 = 42;  // Should be UPPERCASE

pub fn public_function() {
    // Missing documentation
}

mod large_module {
    // This is a mod.rs with lots of code (simulated)
    // ' + 'x' * 1100 + '
}

#[cfg(not(test))]
fn production_code() {
    panic!("Should return Result");
    todo!("Implement this");
}
"""
        )

        result = analyzer.analyze_file(test_file)

        naming_issues = [
            i for i in result.issues if "SCREAMING_SNAKE_CASE" in i.message
        ]
        assert len(naming_issues) >= 1

        doc_issues = [i for i in result.issues if "documentation" in i.message.lower()]
        assert len(doc_issues) >= 1

    def test_testing_recommendations(self, analyzer, tmp_path):
        """Test testing pattern recommendations."""
        test_file = tmp_path / "no_tests.rs"
        test_file.write_text(
            """
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

pub fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

pub fn complex_logic() {
    // Complex implementation
}

pub fn another_function() {
    // Another implementation
}

pub fn yet_another_function() {
    // More code
}
"""
        )

        result = analyzer.analyze_file(test_file)

        test_issues = [i for i in result.issues if "test" in i.message.lower()]
        assert len(test_issues) >= 1
