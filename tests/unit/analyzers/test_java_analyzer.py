"""Tests for Java language analyzer."""

import pytest

from src.analyzers.base import IssueType, Severity
from src.analyzers.java_analyzer import JavaAnalyzer


class TestJavaAnalyzer:
    """Test Java analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create Java analyzer instance."""
        return JavaAnalyzer()

    def test_language_properties(self, analyzer):
        """Test language and file extension properties."""
        assert analyzer.language == "java"
        assert analyzer.file_extensions == [".java"]

    def test_access_control_detection(self, analyzer, tmp_path):
        """Test broken access control detection."""
        test_file = tmp_path / "Controller.java"
        test_file.write_text(
            """
@RestController
public class UserController {

    @GetMapping("/admin/users")
    public List<User> getAllUsers() {
        // Missing @PreAuthorize or @Secured
        return userService.findAll();
    }

    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping("/admin/delete")
    public void deleteUser(Long id) {
        // Properly secured
        userService.delete(id);
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        access_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-862"
        ]
        assert len(access_issues) >= 1
        assert any("authorization" in issue.message.lower() for issue in access_issues)

    def test_cryptographic_failures(self, analyzer, tmp_path):
        """Test cryptographic failure detection."""
        test_file = tmp_path / "Crypto.java"
        test_file.write_text(
            """
public class CryptoUtil {
    private static final String PASSWORD = "hardcoded-password";
    private static final String API_KEY = "sk_test_1234567890";

    public String hashPassword(String input) {
        MessageDigest md = MessageDigest.getInstance("MD5");
        return Base64.encode(md.digest(input.getBytes()));
    }

    public void encrypt(String data) {
        Cipher cipher = Cipher.getInstance("DES/ECB/PKCS5Padding");
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        # Check weak algorithms
        crypto_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-327"
        ]
        assert len(crypto_issues) >= 2  # MD5 and DES

        # Check hardcoded secrets
        secret_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-798"
        ]
        assert len(secret_issues) >= 2  # password and API key

    def test_sql_injection_detection(self, analyzer, tmp_path):
        """Test SQL injection detection."""
        test_file = tmp_path / "UserDao.java"
        test_file.write_text(
            """
public class UserDao {
    public User findUser(String username) {
        String sql = "SELECT * FROM users WHERE username = '" + username + "'";
        return jdbcTemplate.queryForObject(sql, User.class);
    }

    public List<User> searchUsers(String name) {
        String query = "SELECT * FROM users WHERE name LIKE '%" + request.getParameter("search") + "%'";
        return em.createQuery(query).getResultList();
    }

    public User findById(Long id) {
        return jdbcTemplate.queryForObject(
            "SELECT * FROM users WHERE id = ?",
            new Object[]{id},
            User.class
        );
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        sql_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-89"
        ]
        assert len(sql_issues) >= 2
        assert all(i.severity == Severity.CRITICAL for i in sql_issues)

    def test_insecure_deserialization(self, analyzer, tmp_path):
        """Test unsafe deserialization detection."""
        test_file = tmp_path / "Deserialize.java"
        test_file.write_text(
            """
public class DataProcessor {
    public Object deserialize(byte[] data) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data));
        return ois.readObject();
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, defaultImpl = Object.class)
    public class UnsafeData {
        private Object data;
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        deserialize_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-502"
        ]
        assert len(deserialize_issues) >= 2

    def test_logging_sensitive_data(self, analyzer, tmp_path):
        """Test detection of sensitive data in logs."""
        test_file = tmp_path / "AuthService.java"
        test_file.write_text(
            """
@Service
public class AuthService {
    private static final Logger log = LoggerFactory.getLogger(AuthService.class);

    public void authenticate(String username, String password) {
        log.info("Authenticating user: " + username + " with password: " + password);

        String token = generateToken(username);
        log.debug("Generated token: " + token);
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        log_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-532"
        ]
        assert len(log_issues) >= 2

    def test_resource_leak_detection(self, analyzer, tmp_path):
        """Test resource leak detection."""
        test_file = tmp_path / "FileHandler.java"
        test_file.write_text(
            """
public class FileHandler {
    public String readFile(String path) throws IOException {
        FileInputStream fis = new FileInputStream(path);
        // Missing close()
        return readContent(fis);
    }

    public void processFiles(List<String> files) {
        for (String file : files) {
            Connection conn = getConnection();
            // Connection not closed in loop
            processWithConnection(conn, file);
        }
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        leak_issues = [i for i in result.issues if "leak" in i.message.lower()]
        assert len(leak_issues) >= 2

    def test_performance_issues(self, analyzer, tmp_path):
        """Test performance issue detection."""
        test_file = tmp_path / "Performance.java"
        test_file.write_text(
            """
public class DataProcessor {
    public String concatenate(List<String> items) {
        String result = "";
        for (String item : items) {
            result += item;  // String concatenation in loop
        }
        return result;
    }

    public void processLinkedList() {
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < 1000; i++) {
            int value = list.get(i);  // O(n) access
        }
    }

    public void createMaps() {
        Map<String, String> map = new HashMap<>();  // No initial capacity
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        perf_issues = [i for i in result.issues if i.type == IssueType.PERFORMANCE]
        assert len(perf_issues) >= 3

    def test_thread_safety_issues(self, analyzer, tmp_path):
        """Test thread safety issue detection."""
        test_file = tmp_path / "DateUtil.java"
        test_file.write_text(
            """
public class DateUtil {
    private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");

    public String formatDate(Date date) {
        return dateFormat.format(date);  // Not thread-safe
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        thread_issues = [i for i in result.issues if "thread-safe" in i.message.lower()]
        assert len(thread_issues) >= 1

    def test_n_plus_one_query(self, analyzer, tmp_path):
        """Test N+1 query problem detection."""
        test_file = tmp_path / "OrderService.java"
        test_file.write_text(
            """
@Service
public class OrderService {
    public List<OrderDTO> getOrders() {
        List<Order> orders = orderRepository.findAll();
        List<OrderDTO> dtos = new ArrayList<>();

        for (Order order : orders) {
            OrderDTO dto = new OrderDTO();
            dto.setItems(itemRepository.findByOrderId(order.getId()));
            dtos.add(dto);
        }

        return dtos;
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        n_plus_one = [i for i in result.issues if "n+1" in i.message.lower()]
        assert len(n_plus_one) >= 1

    def test_spring_best_practices(self, analyzer, tmp_path):
        """Test Spring framework best practices."""
        test_file = tmp_path / "UserService.java"
        test_file.write_text(
            """
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;  // Field injection

    @Autowired
    private EmailService emailService;  // Field injection

    public User getUser(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        spring_issues = [
            i
            for i in result.issues
            if "injection" in i.message and "field" in i.message.lower()
        ]
        assert len(spring_issues) >= 2

    def test_exception_handling(self, analyzer, tmp_path):
        """Test exception handling issues."""
        test_file = tmp_path / "ErrorHandler.java"
        test_file.write_text(
            """
public class ErrorHandler {
    public void process() {
        try {
            doSomething();
        } catch (Exception e) {
            // Empty catch block
        }

        try {
            riskyOperation();
        } catch (Exception e) {  // Catching generic Exception
            log.error("Error occurred");
        }
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        exception_issues = [
            i for i in result.issues if i.type == IssueType.ERROR_HANDLING
        ]
        assert len(exception_issues) >= 2

    def test_owasp_compliance(self, analyzer, tmp_path):
        """Test OWASP category detection."""
        test_file = tmp_path / "VulnerableApp.java"
        test_file.write_text(
            """
@RestController
public class VulnerableController {
    @CrossOrigin(origins = "*")  // A05:2021
    @GetMapping("/data")
    public Data getData() {
        return data;
    }

    @PostMapping("/ssrf")
    public String makeRequest(@RequestParam String url) {
        URL targetUrl = new URL(request.getParameter("url"));  // A10:2021
        return targetUrl.openConnection().getContent();
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        # Should have issues from different OWASP categories
        owasp_issues = [
            i
            for i in result.issues
            if hasattr(i, "owasp_category") and i.owasp_category
        ]
        assert len(owasp_issues) >= 2

        categories = {i.owasp_category for i in owasp_issues}
        assert "A05:2021" in str(categories)
        assert "A10:2021" in str(categories)
