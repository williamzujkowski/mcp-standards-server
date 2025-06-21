"""
Tests for Java analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Java analyzer testing
"""


import pytest

from src.analyzers.java_analyzer import JavaAnalyzer


class TestJavaAnalyzer:
    """Test Java code analysis capabilities"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return JavaAnalyzer()

    def test_detect_spring_security_config(self, analyzer, tmp_path):
        """Test detection of Spring Security configuration"""
        test_file = tmp_path / "SecurityConfig.java"
        code = '''
package com.example.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

/**
 * @nist-controls: AC-3, IA-2, SC-8, SC-13
 * @evidence: Comprehensive Spring Security configuration
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // CSRF protection
            .csrf(csrf -> csrf
                .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
            )
            // Session management
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                .maximumSessions(1)
                .maxSessionsPreventsLogin(true)
            )
            // Authorization rules
            .authorizeHttpRequests(authz -> authz
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers("/api/user/**").hasAnyRole("USER", "ADMIN")
                .anyRequest().authenticated()
            )
            // JWT filter
            .addFilterBefore(jwtAuthenticationFilter(), UsernamePasswordAuthenticationFilter.class)
            // Security headers
            .headers(headers -> headers
                .frameOptions().deny()
                .xssProtection().and()
                .contentSecurityPolicy("default-src 'self'")
            )
            // HTTPS only
            .requiresChannel(channel -> channel
                .anyRequest().requiresSecure()
            );

        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }

    @Bean
    public JwtAuthenticationFilter jwtAuthenticationFilter() {
        return new JwtAuthenticationFilter();
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-3" in controls  # Authorization
        assert "IA-2" in controls  # Authentication
        assert "SC-8" in controls or "SC-13" in controls  # HTTPS/Encryption
        assert "AC-12" in controls or "SC-23" in controls  # Session management

        # Should identify Spring Security patterns
        assert any("spring security" in ann.evidence.lower() for ann in results)
        assert any("bcrypt" in ann.evidence.lower() for ann in results)

    def test_detect_authentication_service(self, analyzer, tmp_path):
        """Test detection of authentication patterns"""
        test_file = tmp_path / "AuthenticationService.java"
        code = '''
package com.example.auth;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Service
public class AuthenticationService {
    private static final Logger auditLogger = LoggerFactory.getLogger("AUDIT");

    @Value("${jwt.secret}")
    private String jwtSecret;

    @Value("${jwt.expiration}")
    private Long jwtExpiration;

    private final AuthenticationManager authenticationManager;
    private final PasswordEncoder passwordEncoder;
    private final UserRepository userRepository;

    /**
     * Authenticate user and generate JWT token
     * @nist-controls: IA-2, IA-5
     * @evidence: JWT authentication with BCrypt password verification
     */
    public AuthResponse authenticate(String username, String password) {
        try {
            // Authenticate with Spring Security
            Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(username, password)
            );

            // Generate JWT token
            UserDetails userDetails = (UserDetails) authentication.getPrincipal();
            String token = generateToken(userDetails);

            // Audit successful login
            auditLogger.info("Successful login - User: {}, IP: {}",
                username, getClientIP());

            return new AuthResponse(token, userDetails);

        } catch (Exception e) {
            // Audit failed login
            auditLogger.warn("Failed login attempt - User: {}, IP: {}, Reason: {}",
                username, getClientIP(), e.getMessage());
            throw new AuthenticationException("Invalid credentials");
        }
    }

    /**
     * Generate JWT token with claims
     */
    private String generateToken(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("roles", userDetails.getAuthorities());
        claims.put("username", userDetails.getUsername());

        return Jwts.builder()
            .setClaims(claims)
            .setSubject(userDetails.getUsername())
            .setIssuedAt(new Date())
            .setExpiration(new Date(System.currentTimeMillis() + jwtExpiration))
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }

    /**
     * Validate JWT token
     */
    public Claims validateToken(String token) {
        try {
            return Jwts.parser()
                .setSigningKey(jwtSecret)
                .parseClaimsJws(token)
                .getBody();
        } catch (Exception e) {
            auditLogger.warn("Invalid JWT token - IP: {}, Reason: {}",
                getClientIP(), e.getMessage());
            throw new AuthenticationException("Invalid token");
        }
    }

    /**
     * Register new user with secure password
     * @nist-controls: IA-5
     * @evidence: BCrypt password hashing with strength 12
     */
    public User registerUser(UserRegistrationDto dto) {
        // Validate password strength
        if (!isPasswordStrong(dto.getPassword())) {
            throw new ValidationException("Password does not meet security requirements");
        }

        // Hash password
        String hashedPassword = passwordEncoder.encode(dto.getPassword());

        // Create user
        User user = new User();
        user.setUsername(dto.getUsername());
        user.setEmail(dto.getEmail());
        user.setPasswordHash(hashedPassword);
        user.setMfaEnabled(true); // Enable MFA by default

        return userRepository.save(user);
    }

    private boolean isPasswordStrong(String password) {
        // At least 12 characters, uppercase, lowercase, digit, special char
        String pattern = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=])(?=\\S+$).{12,}$";
        return password.matches(pattern);
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect authentication controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-2" in controls  # Authentication
        assert "IA-5" in controls  # Authenticator management
        assert "AU-2" in controls or "AU-3" in controls  # Audit logging

        # Should identify JWT and BCrypt
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("jwt" in ev for ev in evidence_texts)
        assert any("bcrypt" in ev or "password" in ev for ev in evidence_texts)

    def test_detect_input_validation(self, analyzer, tmp_path):
        """Test detection of input validation patterns"""
        test_file = tmp_path / "ValidationService.java"
        code = '''
package com.example.validation;

import javax.validation.constraints.*;
import org.hibernate.validator.constraints.SafeHtml;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.owasp.encoder.Encode;
import org.apache.commons.text.StringEscapeUtils;

import java.util.regex.Pattern;

@RestController
@Validated
public class UserController {

    private static final Pattern SQL_INJECTION_PATTERN = Pattern.compile(
        "('.+--)|(\\\\\\')|(\\\\\\\")|(;.*(drop|delete|truncate|update|insert))|(union.+select)",
        Pattern.CASE_INSENSITIVE
    );

    /**
     * @nist-controls: SI-10
     * @evidence: Comprehensive input validation using Bean Validation
     */
    @PostMapping("/api/users")
    public User createUser(@Valid @RequestBody UserDto userDto) {
        // Additional validation
        validateInput(userDto);

        // Sanitize inputs
        userDto.setName(sanitizeHtml(userDto.getName()));
        userDto.setBio(sanitizeHtml(userDto.getBio()));

        return userService.create(userDto);
    }

    /**
     * User DTO with validation constraints
     */
    public static class UserDto {
        @NotBlank(message = "Username is required")
        @Size(min = 3, max = 20)
        @Pattern(regexp = "^[a-zA-Z0-9_]+$", message = "Username must be alphanumeric")
        private String username;

        @NotBlank
        @Email(message = "Valid email is required")
        private String email;

        @NotNull
        @Min(13)
        @Max(120)
        private Integer age;

        @SafeHtml(whitelistType = SafeHtml.WhiteListType.NONE)
        private String bio;

        @NotBlank
        @Size(min = 12, message = "Password must be at least 12 characters")
        @Pattern(regexp = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=]).*$",
                message = "Password must contain uppercase, lowercase, digit and special character")
        private String password;

        // Getters and setters...
    }

    /**
     * Custom validation logic
     */
    private void validateInput(UserDto dto) {
        // Check for SQL injection patterns
        if (containsSqlInjection(dto.getUsername()) ||
            containsSqlInjection(dto.getBio())) {
            throw new ValidationException("Invalid input detected");
        }

        // Validate against blocklist
        if (isBlocklisted(dto.getEmail())) {
            throw new ValidationException("Email domain not allowed");
        }
    }

    private boolean containsSqlInjection(String input) {
        if (input == null) return false;
        return SQL_INJECTION_PATTERN.matcher(input).find();
    }

    /**
     * Sanitize HTML to prevent XSS
     */
    private String sanitizeHtml(String input) {
        if (input == null) return null;

        // Use OWASP encoder
        String encoded = Encode.forHtml(input);

        // Additional escaping
        return StringEscapeUtils.escapeHtml4(encoded);
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ErrorResponse> handleValidationException(ConstraintViolationException e) {
        // Log validation failures
        auditLogger.warn("Input validation failed: {}", e.getMessage());

        return ResponseEntity.badRequest()
            .body(new ErrorResponse("Validation failed", extractViolations(e)));
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect input validation controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Information input validation

        # Should identify various validation patterns
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("valid" in ev for ev in evidence_texts)
        assert any("sanitize" in ev or "escape" in ev for ev in evidence_texts)
        assert any("sql" in ev for ev in evidence_texts)  # SQL injection prevention

    def test_detect_jpa_security(self, analyzer, tmp_path):
        """Test detection of JPA/Hibernate security patterns"""
        test_file = tmp_path / "UserRepository.java"
        code = '''
package com.example.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.repository.query.Param;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.transaction.annotation.Transactional;

import javax.persistence.*;
import java.util.Optional;
import java.util.List;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    /**
     * Safe parameterized query to prevent SQL injection
     * @nist-controls: SI-10
     * @evidence: Parameterized JPA query
     */
    @Query("SELECT u FROM User u WHERE u.username = :username AND u.active = true")
    Optional<User> findActiveByUsername(@Param("username") String username);

    /**
     * Native query with parameter binding
     */
    @Query(value = "SELECT * FROM users WHERE email = ?1 AND deleted_at IS NULL",
           nativeQuery = true)
    Optional<User> findByEmailNative(String email);

    /**
     * Secured method with role check
     * @nist-controls: AC-3
     * @evidence: Method-level authorization
     */
    @PreAuthorize("hasRole('ADMIN')")
    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.active = false WHERE u.id = :userId")
    void deactivateUser(@Param("userId") Long userId);

    /**
     * Criteria query for complex searches
     */
    default List<User> searchUsers(String searchTerm) {
        return findAll((root, query, criteriaBuilder) -> {
            // Safe criteria building
            String pattern = "%" + searchTerm.toLowerCase() + "%";
            return criteriaBuilder.or(
                criteriaBuilder.like(
                    criteriaBuilder.lower(root.get("username")),
                    pattern
                ),
                criteriaBuilder.like(
                    criteriaBuilder.lower(root.get("email")),
                    pattern
                )
            );
        });
    }
}

@Entity
@Table(name = "users")
@EntityListeners(AuditingEntityListener.class)
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    @NotBlank
    private String username;

    @Column(unique = true, nullable = false)
    @Email
    private String email;

    @Column(name = "password_hash", nullable = false)
    @JsonIgnore  // Never serialize password
    private String passwordHash;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "user_roles")
    @Enumerated(EnumType.STRING)
    private Set<Role> roles = new HashSet<>();

    @Column(name = "mfa_secret")
    @JsonIgnore  // Sensitive data
    private String mfaSecret;

    @CreatedDate
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Version  // Optimistic locking
    private Long version;

    // Audit fields
    @CreatedBy
    @Column(name = "created_by")
    private String createdBy;

    @LastModifiedBy
    @Column(name = "modified_by")
    private String modifiedBy;
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect JPA security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # SQL injection prevention
        assert "AC-3" in controls  # Access control
        assert "AU-2" in controls or "AU-3" in controls  # Auditing

        # Should identify JPA security patterns
        assert any("param" in ann.evidence.lower() or "jpa" in ann.evidence.lower() for ann in results)
        assert any("preauthorize" in ann.evidence.lower() for ann in results)

    def test_detect_crypto_patterns(self, analyzer, tmp_path):
        """Test detection of cryptographic patterns"""
        test_file = tmp_path / "CryptoService.java"
        code = '''
package com.example.crypto;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.*;
import java.security.spec.KeySpec;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import java.util.Base64;

/**
 * @nist-controls: SC-13, SC-28
 * @evidence: AES-256-GCM encryption with secure key management
 */
@Service
public class CryptoService {

    private static final String AES_ALGORITHM = "AES/GCM/NoPadding";
    private static final int GCM_TAG_LENGTH = 128;
    private static final int GCM_IV_LENGTH = 12;
    private static final int AES_KEY_SIZE = 256;
    private static final int SALT_LENGTH = 32;
    private static final int PBKDF2_ITERATIONS = 100000;

    private final SecureRandom secureRandom = new SecureRandom();

    /**
     * Encrypt data using AES-256-GCM
     */
    public EncryptedData encrypt(String plaintext, String password) throws Exception {
        // Generate salt
        byte[] salt = new byte[SALT_LENGTH];
        secureRandom.nextBytes(salt);

        // Derive key from password
        SecretKey key = deriveKey(password, salt);

        // Generate IV
        byte[] iv = new byte[GCM_IV_LENGTH];
        secureRandom.nextBytes(iv);

        // Encrypt
        Cipher cipher = Cipher.getInstance(AES_ALGORITHM);
        GCMParameterSpec parameterSpec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
        cipher.init(Cipher.ENCRYPT_MODE, key, parameterSpec);

        byte[] ciphertext = cipher.doFinal(plaintext.getBytes("UTF-8"));

        return new EncryptedData(
            Base64.getEncoder().encodeToString(ciphertext),
            Base64.getEncoder().encodeToString(salt),
            Base64.getEncoder().encodeToString(iv)
        );
    }

    /**
     * Derive key using PBKDF2
     */
    private SecretKey deriveKey(String password, byte[] salt) throws Exception {
        KeySpec spec = new PBEKeySpec(
            password.toCharArray(),
            salt,
            PBKDF2_ITERATIONS,
            AES_KEY_SIZE
        );

        SecretKeyFactory factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
        SecretKey tmp = factory.generateSecret(spec);

        return new SecretKeySpec(tmp.getEncoded(), "AES");
    }

    /**
     * Generate RSA key pair
     * @nist-controls: SC-12
     * @evidence: 4096-bit RSA key generation
     */
    public KeyPair generateKeyPair() throws Exception {
        KeyPairGenerator keyGen = KeyPairGenerator.getInstance("RSA");
        keyGen.initialize(4096, secureRandom);
        return keyGen.generateKeyPair();
    }

    /**
     * Compute secure hash
     */
    public String computeHash(String data) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(data.getBytes("UTF-8"));
        return Base64.getEncoder().encodeToString(hash);
    }

    /**
     * Generate secure random token
     */
    public String generateSecureToken(int length) {
        byte[] token = new byte[length];
        secureRandom.nextBytes(token);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(token);
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect cryptographic controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-13" in controls  # Cryptographic protection
        assert "SC-28" in controls  # Protection at rest
        assert "SC-12" in controls  # Key establishment

        # Should identify strong crypto
        assert any("aes-256" in ann.evidence.lower() or "aes" in ann.evidence.lower() for ann in results)
        assert any("pbkdf2" in ann.evidence.lower() for ann in results)

    def test_detect_audit_logging(self, analyzer, tmp_path):
        """Test detection of audit logging patterns"""
        test_file = tmp_path / "AuditService.java"
        code = '''
package com.example.audit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.event.EventListener;
import org.springframework.security.authentication.event.*;
import org.springframework.security.access.event.AuthorizationFailureEvent;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * @nist-controls: AU-2, AU-3, AU-4, AU-9
 * @evidence: Comprehensive security event auditing
 */
@Component
public class AuditService {

    private static final Logger auditLogger = LoggerFactory.getLogger("AUDIT");
    private static final Logger securityLogger = LoggerFactory.getLogger("SECURITY");

    @EventListener
    public void handleAuthenticationSuccess(AuthenticationSuccessEvent event) {
        String username = event.getAuthentication().getName();
        String details = event.getAuthentication().getDetails().toString();

        AuditEvent audit = AuditEvent.builder()
            .eventType("AUTH_SUCCESS")
            .username(username)
            .timestamp(LocalDateTime.now())
            .ipAddress(extractIP(details))
            .userAgent(extractUserAgent(details))
            .outcome("SUCCESS")
            .build();

        auditLogger.info(formatAuditLog(audit));
    }

    @EventListener
    public void handleAuthenticationFailure(AbstractAuthenticationFailureEvent event) {
        String username = event.getAuthentication().getName();
        String reason = event.getException().getMessage();

        AuditEvent audit = AuditEvent.builder()
            .eventType("AUTH_FAILURE")
            .username(username)
            .timestamp(LocalDateTime.now())
            .reason(reason)
            .outcome("FAILURE")
            .build();

        securityLogger.warn(formatAuditLog(audit));

        // Alert on multiple failures
        if (getRecentFailureCount(username) > 5) {
            securityLogger.error("Multiple authentication failures for user: {}", username);
            alertSecurityTeam(username);
        }
    }

    @EventListener
    public void handleAuthorizationFailure(AuthorizationFailureEvent event) {
        Authentication auth = event.getAuthentication();
        String username = auth != null ? auth.getName() : "anonymous";

        AuditEvent audit = AuditEvent.builder()
            .eventType("AUTHZ_FAILURE")
            .username(username)
            .resource(event.getSource().toString())
            .timestamp(LocalDateTime.now())
            .outcome("DENIED")
            .build();

        securityLogger.warn(formatAuditLog(audit));
    }

    /**
     * Log data access events
     */
    public void logDataAccess(String username, String resource, String action, boolean success) {
        Map<String, Object> auditData = Map.of(
            "event_type", "DATA_ACCESS",
            "username", username,
            "resource", resource,
            "action", action,
            "timestamp", LocalDateTime.now(),
            "outcome", success ? "SUCCESS" : "FAILURE"
        );

        if (isSensitiveResource(resource)) {
            securityLogger.info("Sensitive data access: {}", auditData);
        } else {
            auditLogger.info("Data access: {}", auditData);
        }
    }

    /**
     * Log configuration changes
     */
    public void logConfigurationChange(String username, String setting, String oldValue, String newValue) {
        Map<String, Object> auditData = Map.of(
            "event_type", "CONFIG_CHANGE",
            "username", username,
            "setting", setting,
            "old_value", maskSensitive(setting, oldValue),
            "new_value", maskSensitive(setting, newValue),
            "timestamp", LocalDateTime.now()
        );

        auditLogger.warn("Configuration change: {}", auditData);
    }

    private String formatAuditLog(AuditEvent event) {
        // Structured format for SIEM integration
        return String.format(
            "AUDIT|%s|%s|%s|%s|%s|%s",
            event.getTimestamp(),
            event.getEventType(),
            event.getUsername(),
            event.getIpAddress(),
            event.getOutcome(),
            event.getAdditionalData()
        );
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect audit controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AU-2" in controls  # Audit events
        assert "AU-3" in controls  # Content of audit records
        assert "AU-4" in controls or "AU-9" in controls  # Audit storage/protection

        # Should identify comprehensive auditing
        assert any("audit" in ann.evidence.lower() for ann in results)
        assert any("security event" in ann.evidence.lower() or "authentication" in ann.evidence.lower()
                  for ann in results)

    def test_pom_xml_analysis(self, analyzer, tmp_path):
        """Test pom.xml security analysis"""
        pom_file = tmp_path / "pom.xml"
        pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>secure-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <!-- Spring Security -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>

        <!-- JWT -->
        <dependency>
            <groupId>io.jsonwebtoken</groupId>
            <artifactId>jjwt-api</artifactId>
            <version>0.11.5</version>
        </dependency>

        <!-- Encryption -->
        <dependency>
            <groupId>org.bouncycastle</groupId>
            <artifactId>bcprov-jdk15on</artifactId>
            <version>1.70</version>
        </dependency>

        <!-- Validation -->
        <dependency>
            <groupId>org.hibernate.validator</groupId>
            <artifactId>hibernate-validator</artifactId>
        </dependency>

        <!-- OWASP Security -->
        <dependency>
            <groupId>org.owasp.encoder</groupId>
            <artifactId>encoder</artifactId>
            <version>1.2.3</version>
        </dependency>

        <!-- Security Testing -->
        <dependency>
            <groupId>org.owasp</groupId>
            <artifactId>dependency-check-maven</artifactId>
            <version>8.4.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.owasp</groupId>
                <artifactId>dependency-check-maven</artifactId>
                <version>8.4.0</version>
            </plugin>

            <plugin>
                <groupId>com.github.spotbugs</groupId>
                <artifactId>spotbugs-maven-plugin</artifactId>
                <version>4.7.3.0</version>
            </plugin>
        </plugins>
    </build>
</project>'''
        pom_file.write_text(pom_content)

        results = analyzer._analyze_pom_xml(pom_file)

        # Should detect security dependencies
        assert len(results) >= 6

        # Check specific security packages
        packages = [ann.evidence for ann in results]
        assert any('spring-boot-starter-security' in pkg or 'spring security' in pkg.lower() for pkg in packages)
        assert any('jwt' in pkg.lower() for pkg in packages)
        assert any('bouncycastle' in pkg or 'bcprov' in pkg for pkg in packages)
        assert any('owasp' in pkg.lower() or 'encoder' in pkg.lower() or 'dependency-check' in pkg.lower() for pkg in packages)

    def test_detect_rest_controller_security(self, analyzer, tmp_path):
        """Test detection of REST controller security patterns"""
        test_file = tmp_path / "UserController.java"
        code = '''
package com.example.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.security.access.annotation.Secured;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.access.prepost.PostAuthorize;
import org.springframework.security.access.prepost.PreFilter;
import org.springframework.security.access.prepost.PostFilter;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.validation.annotation.Validated;
import org.springframework.http.ResponseEntity;

import javax.validation.Valid;
import javax.validation.constraints.Pattern;
import java.util.List;

@RestController
@RequestMapping("/api/users")
@Validated
@Secured("ROLE_USER")
public class UserController {

    /**
     * Get user by ID with authorization check
     * @nist-controls: AC-3, AC-4
     * @evidence: Post-authorization ensures users can only access their own data
     */
    @GetMapping("/{id}")
    @PostAuthorize("returnObject.username == authentication.name or hasRole('ADMIN')")
    public User getUser(@PathVariable @Pattern(regexp = "^[0-9]+$") String id) {
        return userService.findById(Long.parseLong(id));
    }

    /**
     * Create new user - admin only
     */
    @PostMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<User> createUser(@Valid @RequestBody UserDto userDto) {
        User created = userService.create(userDto);

        // Audit log
        auditService.logUserCreation(getCurrentUser(), created);

        return ResponseEntity.ok(created);
    }

    /**
     * Update user - owner or admin
     */
    @PutMapping("/{id}")
    @PreAuthorize("#id == authentication.principal.id or hasRole('ADMIN')")
    public User updateUser(
            @PathVariable Long id,
            @Valid @RequestBody UserUpdateDto dto,
            @AuthenticationPrincipal UserPrincipal principal) {

        return userService.update(id, dto, principal);
    }

    /**
     * Delete users - admin only with filtering
     */
    @DeleteMapping
    @PreAuthorize("hasRole('ADMIN')")
    @PreFilter("filterObject.deletable == true")
    public void deleteUsers(@RequestBody List<User> users) {
        userService.deleteAll(users);
    }

    /**
     * Search users with result filtering
     */
    @GetMapping("/search")
    @PostFilter("filterObject.active == true or hasRole('ADMIN')")
    public List<User> searchUsers(
            @RequestParam @Pattern(regexp = "^[a-zA-Z0-9\\s]+$") String query) {

        return userService.search(query);
    }

    /**
     * Get current user profile
     */
    @GetMapping("/me")
    public User getCurrentUserProfile(@AuthenticationPrincipal UserPrincipal principal) {
        return userService.findById(principal.getId());
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ErrorResponse> handleAccessDenied(AccessDeniedException e) {
        // Log authorization failure
        securityLogger.warn("Access denied: {}", e.getMessage());

        return ResponseEntity.status(403)
            .body(new ErrorResponse("Access denied", "FORBIDDEN"));
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect authorization controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-3" in controls  # Access enforcement
        assert "AC-4" in controls  # Information flow enforcement

        # Should identify Spring Security annotations
        assert any("preauthorize" in ann.evidence.lower() or "postauthorize" in ann.evidence.lower()
                  for ann in results)
        assert any("secured" in ann.evidence.lower() for ann in results)
