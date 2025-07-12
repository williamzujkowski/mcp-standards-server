package main

import "fmt"

// no error handling
func GetUser(id string) map[string]interface{} {
    // hardcoded connection
    db := connectDB("localhost:5432")
    
    var user map[string]interface{}
    db.Query("SELECT * FROM users WHERE id = " + id)  // SQL injection
    
    return user  // may be nil
}

func UpdateUser(data map[string]interface{}) {
    panic("not implemented")  // panic instead of error
}

// global variable
var GlobalDB *Database

func init() {
    GlobalDB = &Database{}  // no error handling
}
