// Package main demonstrates various Go code patterns for analyzer testing
package main

import (
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"sync"
)

// Hardcoded credentials - security issue
const (
	APIKey   = "sk_live_4242424242424242"
	Password = "admin123"
)

// Global mutable state - concurrency issue
var (
	counter int
	users   []User
	mutex   sync.Mutex
)

// User represents a user in the system
type User struct {
	ID       int
	Username string
	Password string // Storing plain text password
}

// VulnerableQuery demonstrates SQL injection vulnerability
func VulnerableQuery(userInput string) (*User, error) {
	// SQL Injection vulnerability
	query := fmt.Sprintf("SELECT * FROM users WHERE username = '%s'", userInput)
	row := db.QueryRow(query)
	
	var user User
	err := row.Scan(&user.ID, &user.Username, &user.Password)
	return &user, err
}

// CommandInjection demonstrates command injection vulnerability
func CommandInjection(filename string) error {
	// Command injection vulnerability
	cmd := exec.Command("sh", "-c", "cat "+filename)
	_, err := cmd.Output()
	return err
}

// RaceCondition demonstrates a race condition
func RaceCondition() {
	// Race condition - multiple goroutines accessing shared state
	for i := 0; i < 100; i++ {
		go func() {
			counter++ // Not thread-safe
		}()
	}
}

// IgnoredError demonstrates poor error handling
func IgnoredError() {
	file, _ := os.Open("important.txt") // Error ignored
	defer file.Close()
	
	_, _ = file.Read(make([]byte, 100)) // Error ignored
}

// MemoryLeak demonstrates potential memory leak
func MemoryLeak() {
	for {
		conn, err := net.Dial("tcp", "example.com:80")
		if err != nil {
			continue
		}
		// Missing conn.Close() - resource leak
		go handleConnection(conn)
	}
}

// InefficientStringConcat demonstrates performance issue
func InefficientStringConcat(items []string) string {
	result := ""
	for _, item := range items {
		result += item // Inefficient string concatenation
	}
	return result
}

// DeferInLoop demonstrates defer in loop issue
func DeferInLoop(files []string) error {
	for _, file := range files {
		f, err := os.Open(file)
		if err != nil {
			return err
		}
		defer f.Close() // Defer in loop - accumulates until function returns
		
		// Process file...
	}
	return nil
}

// ComplexInterface demonstrates overly complex interface
type ComplexInterface interface {
	Method1()
	Method2()
	Method3()
	Method4()
	Method5()
	Method6()
	Method7()
	Method8()
	// Too many methods - violates interface segregation
}

// publicFunction should start with uppercase
func publicFunction() {
	// Naming convention issue
}

// HTTPHandler missing context parameter
func HTTPHandler(w http.ResponseWriter, r *http.Request) {
	// Should accept context.Context as first parameter
	data, err := fetchDataFromDB(r.URL.Query().Get("id"))
	if err != nil {
		panic(err) // Panic without recover
	}
	
	fmt.Fprintf(w, "Data: %v", data)
}

// WeakCrypto uses weak cryptographic algorithms
func WeakCrypto(data []byte) {
	// Using MD5 - weak algorithm
	h := md5.New()
	h.Write(data)
	
	// Using SHA1 - weak algorithm
	sha1.Sum(data)
}

// init function without clear purpose
func init() {
	// Complex initialization in init can be problematic
	loadConfiguration()
	connectDatabase()
	startBackgroundJobs()
}

// SliceAppendWithoutPreallocation demonstrates performance issue
func SliceAppendWithoutPreallocation(n int) []int {
	var result []int // Should use make([]int, 0, n)
	for i := 0; i < n; i++ {
		result = append(result, i)
	}
	return result
}

// UnbufferedChannel demonstrates performance consideration
func UnbufferedChannel() {
	ch := make(chan int) // Consider buffered channel for better performance
	
	go func() {
		for i := 0; i < 1000; i++ {
			ch <- i
		}
		close(ch)
	}()
	
	for range ch {
		// Process...
	}
}

// MapWithoutSizeHint demonstrates map initialization issue
func MapWithoutSizeHint() map[string]int {
	m := make(map[string]int) // Should provide size hint if known
	
	for i := 0; i < 1000; i++ {
		m[fmt.Sprintf("key%d", i)] = i
	}
	
	return m
}

func main() {
	// Example usage
	log.Println("Go analyzer test file")
}