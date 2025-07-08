package com.example.vulnerable;

import java.io.*;
import java.sql.*;
import java.util.*;
import java.security.MessageDigest;
import javax.crypto.Cipher;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

@RestController
@CrossOrigin(origins = "*") // Security misconfiguration
public class VulnerableController {
    
    // Hardcoded secrets - security issue
    private static final String PASSWORD = "admin123";
    private static final String API_KEY = "sk_test_4242424242424242";
    private static final String DB_URL = "jdbc:mysql://localhost:3306/mydb?user=root&password=root123";
    
    // Field injection - bad practice
    @Autowired
    private UserService userService;
    
    @Autowired
    private DatabaseService databaseService;
    
    // SimpleDateFormat is not thread-safe
    private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
    
    // Missing authorization annotation
    @GetMapping("/admin/users")
    public List<User> getAllUsers() {
        return userService.findAll();
    }
    
    // SQL Injection vulnerability
    @GetMapping("/user")
    public User getUser(@RequestParam String username) {
        String query = "SELECT * FROM users WHERE username = '" + username + "'";
        return jdbcTemplate.queryForObject(query, User.class);
    }
    
    // Command injection vulnerability
    @PostMapping("/execute")
    public String executeCommand(@RequestParam String cmd) {
        try {
            Process p = Runtime.getRuntime().exec("sh -c " + cmd);
            return "Command executed";
        } catch (IOException e) {
            return "Error: " + e.getMessage();
        }
    }
    
    // Unsafe deserialization
    @PostMapping("/deserialize")
    public Object deserialize(@RequestBody byte[] data) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data));
        return ois.readObject(); // Unsafe deserialization
    }
    
    // SSRF vulnerability
    @GetMapping("/fetch")
    public String fetchUrl(@RequestParam String url) throws Exception {
        URL targetUrl = new URL(url); // User-controlled URL
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(targetUrl.openStream())
        );
        return reader.readLine();
    }
    
    // Weak cryptography
    public String hashPassword(String password) throws Exception {
        MessageDigest md = MessageDigest.getInstance("MD5"); // Weak algorithm
        byte[] digest = md.digest(password.getBytes());
        return Base64.getEncoder().encodeToString(digest);
    }
    
    // Resource leak
    public String readFile(String filename) throws IOException {
        FileInputStream fis = new FileInputStream(filename);
        // Missing fis.close() - resource leak
        byte[] data = new byte[1024];
        fis.read(data);
        return new String(data);
    }
    
    // Logging sensitive data
    public void authenticate(String username, String password) {
        log.info("Authenticating user: " + username + " with password: " + password);
        
        if (checkCredentials(username, password)) {
            String token = generateToken();
            log.debug("Generated token: " + token); // Logging sensitive token
        }
    }
    
    // Empty catch block
    public void processData() {
        try {
            riskyOperation();
        } catch (Exception e) {
            // Empty catch block - bad practice
        }
    }
    
    // String concatenation in loop - performance issue
    public String buildReport(List<String> items) {
        String result = "";
        for (String item : items) {
            result += item + "\n"; // Inefficient
        }
        return result;
    }
    
    // N+1 query problem
    public List<OrderDTO> getOrders() {
        List<Order> orders = orderRepository.findAll();
        List<OrderDTO> dtos = new ArrayList<>();
        
        for (Order order : orders) {
            // N+1 query problem
            List<Item> items = itemRepository.findByOrderId(order.getId());
            OrderDTO dto = new OrderDTO(order, items);
            dtos.add(dto);
        }
        
        return dtos;
    }
    
    // LinkedList random access - performance issue
    public void processLinkedList() {
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < 1000; i++) {
            list.add(i);
        }
        
        // O(n) access on LinkedList
        for (int i = 0; i < list.size(); i++) {
            Integer value = list.get(i); // Inefficient
        }
    }
    
    // HashMap without initial capacity
    public Map<String, String> createLargeMap() {
        Map<String, String> map = new HashMap<>(); // No initial capacity
        
        for (int i = 0; i < 10000; i++) {
            map.put("key" + i, "value" + i);
        }
        
        return map;
    }
    
    // Catching generic Exception
    public void handleRequest() {
        try {
            processRequest();
        } catch (Exception e) { // Too generic
            log.error("Error occurred", e);
        }
    }
    
    // Singleton without thread safety
    public class UnsafeSingleton {
        private static UnsafeSingleton instance;
        
        public static UnsafeSingleton getInstance() {
            if (instance == null) {
                instance = new UnsafeSingleton(); // Not thread-safe
            }
            return instance;
        }
    }
    
    // Class name should start with uppercase
    class userModel {
        private String name;
        // Naming convention issue
    }
    
    // Constant should be UPPER_CASE
    private static final String defaultTimeout = "30s";
    
    // Large allocation in loop
    public void memoryIntensive() {
        for (int i = 0; i < 100; i++) {
            byte[] largeArray = new byte[10000000]; // 10MB allocation in loop
            processArray(largeArray);
        }
    }
    
    // Input validation missing
    @PostMapping("/create")
    public User createUser(@RequestParam String name, @RequestParam int age) {
        // No validation on parameters
        return userService.create(name, age);
    }
    
    // Weak password validation
    public boolean isValidPassword(String password) {
        return password.length() >= 4; // Too weak
    }
}

// Support classes
class User {
    private Long id;
    private String username;
    private String password;
    // getters and setters
}

class Order {
    private Long id;
    private String status;
    // getters and setters
}

class OrderDTO {
    private Order order;
    private List<Item> items;
    
    public OrderDTO(Order order, List<Item> items) {
        this.order = order;
        this.items = items;
    }
}