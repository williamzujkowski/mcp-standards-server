"""
Tests for Go analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Go analyzer testing
"""


import pytest

from src.analyzers.go_analyzer import GoAnalyzer


class TestGoAnalyzer:
    """Test Go code analysis capabilities"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return GoAnalyzer()

    def test_detect_authentication_controls(self, analyzer, tmp_path):
        """Test detection of authentication controls"""
        test_file = tmp_path / "auth.go"
        code = '''
package auth

import (
    "crypto/subtle"
    "time"

    "github.com/dgrijalva/jwt-go"
    "golang.org/x/crypto/bcrypt"
)

// @nist-controls: IA-2, IA-5
// @evidence: JWT authentication with bcrypt password hashing
type AuthService struct {
    jwtSecret []byte
}

// AuthenticateUser validates credentials and returns JWT token
func (s *AuthService) AuthenticateUser(username, password string) (*AuthToken, error) {
    user, err := getUserByUsername(username)
    if err != nil {
        return nil, ErrInvalidCredentials
    }

    // Constant-time comparison to prevent timing attacks
    if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
        // Log failed attempt
        logFailedAuth(username)
        return nil, ErrInvalidCredentials
    }

    // Generate JWT token
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
        "user_id": user.ID,
        "roles":   user.Roles,
        "exp":     time.Now().Add(24 * time.Hour).Unix(),
    })

    tokenString, err := token.SignedString(s.jwtSecret)
    if err != nil {
        return nil, err
    }

    return &AuthToken{
        Token:     tokenString,
        ExpiresAt: time.Now().Add(24 * time.Hour),
    }, nil
}

// VerifyToken validates JWT token
func (s *AuthService) VerifyToken(tokenString string) (*Claims, error) {
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, ErrInvalidToken
        }
        return s.jwtSecret, nil
    })

    if err != nil || !token.Valid {
        return nil, ErrInvalidToken
    }

    claims, ok := token.Claims.(jwt.MapClaims)
    if !ok {
        return nil, ErrInvalidToken
    }

    return &Claims{
        UserID: claims["user_id"].(string),
        Roles:  claims["roles"].([]string),
    }, nil
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect authentication controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-2" in controls
        assert "IA-5" in controls

        # Should identify JWT and bcrypt
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("jwt" in ev for ev in evidence_texts)
        assert any("bcrypt" in ev for ev in evidence_texts)

    def test_detect_gin_security_middleware(self, analyzer, tmp_path):
        """Test detection of Gin framework security middleware"""
        test_file = tmp_path / "middleware.go"
        code = '''
package middleware

import (
    "net/http"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/ulule/limiter/v3"
    "github.com/ulule/limiter/v3/drivers/store/memory"
    "github.com/gin-contrib/cors"
    "github.com/gin-contrib/secure"
)

// SetupSecurityMiddleware configures security middleware
func SetupSecurityMiddleware(router *gin.Engine) {
    // Security headers
    router.Use(secure.New(secure.Config{
        AllowedHosts:          []string{"example.com", "www.example.com"},
        SSLRedirect:           true,
        STSSeconds:            31536000,
        STSIncludeSubdomains:  true,
        FrameDeny:             true,
        ContentTypeNosniff:    true,
        BrowserXssFilter:      true,
        ContentSecurityPolicy: "default-src 'self'",
    }))

    // CORS configuration
    router.Use(cors.New(cors.Config{
        AllowOrigins:     []string{"https://app.example.com"},
        AllowMethods:     []string{"GET", "POST", "PUT", "DELETE"},
        AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
        ExposeHeaders:    []string{"Content-Length"},
        AllowCredentials: true,
        MaxAge:          12 * time.Hour,
    }))

    // Rate limiting
    rate := limiter.Rate{
        Period: 1 * time.Minute,
        Limit:  60,
    }
    store := memory.NewStore()
    rateLimiter := limiter.New(store, rate)

    router.Use(func(c *gin.Context) {
        limiterCtx, err := rateLimiter.Get(c, c.ClientIP())
        if err != nil {
            c.AbortWithStatus(http.StatusInternalServerError)
            return
        }

        if limiterCtx.Reached {
            c.AbortWithStatus(http.StatusTooManyRequests)
            return
        }

        c.Next()
    })

    // Request logging
    router.Use(gin.LoggerWithConfig(gin.LoggerConfig{
        SkipPaths: []string{"/health"},
        Output:    getAuditLogger(),
    }))
}

// AuthRequired middleware for protected routes
func AuthRequired() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "Authorization required",
            })
            return
        }

        // Verify token
        claims, err := verifyToken(token)
        if err != nil {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "Invalid token",
            })
            return
        }

        c.Set("user_id", claims.UserID)
        c.Set("roles", claims.Roles)
        c.Next()
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-8" in controls or "SC-13" in controls  # HTTPS/TLS
        assert "SC-5" in controls  # DoS protection (rate limiting)
        assert "AU-2" in controls or "AU-3" in controls  # Audit logging
        assert "IA-2" in controls  # Authentication

        # Should identify Gin security middleware
        assert any("gin" in ann.evidence.lower() for ann in results)
        assert any("rate" in ann.evidence.lower() and "limit" in ann.evidence.lower() for ann in results)

    def test_detect_crypto_patterns(self, analyzer, tmp_path):
        """Test detection of cryptographic patterns"""
        test_file = tmp_path / "crypto.go"
        code = '''
package crypto

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/base64"
    "io"

    "golang.org/x/crypto/scrypt"
)

// @nist-controls: SC-13, SC-28
// @evidence: AES-256-GCM encryption with secure key derivation
type CryptoService struct {
    privateKey *rsa.PrivateKey
}

// EncryptData encrypts data using AES-256-GCM
func (s *CryptoService) EncryptData(plaintext []byte, password string) (string, error) {
    // Generate salt
    salt := make([]byte, 32)
    if _, err := io.ReadFull(rand.Reader, salt); err != nil {
        return "", err
    }

    // Derive key using scrypt
    key, err := scrypt.Key([]byte(password), salt, 32768, 8, 1, 32)
    if err != nil {
        return "", err
    }

    // Create cipher
    block, err := aes.NewCipher(key)
    if err != nil {
        return "", err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }

    // Generate nonce
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }

    // Encrypt
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

    // Combine salt and ciphertext
    combined := append(salt, ciphertext...)

    return base64.StdEncoding.EncodeToString(combined), nil
}

// GenerateKeyPair generates RSA key pair for asymmetric encryption
func GenerateKeyPair(bits int) (*rsa.PrivateKey, error) {
    if bits < 2048 {
        bits = 2048 // Minimum key size for security
    }

    privateKey, err := rsa.GenerateKey(rand.Reader, bits)
    if err != nil {
        return nil, err
    }

    return privateKey, nil
}

// HashPassword creates secure password hash
func HashPassword(password string) string {
    hash := sha256.Sum256([]byte(password))
    return base64.StdEncoding.EncodeToString(hash[:])
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect encryption controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-13" in controls  # Cryptographic protection
        assert "SC-28" in controls  # Protection at rest

        # Should identify strong crypto algorithms
        assert any("aes-256" in ann.evidence.lower() or "aes" in ann.evidence.lower() for ann in results)
        assert any("scrypt" in ann.evidence.lower() for ann in results)

    def test_detect_grpc_security(self, analyzer, tmp_path):
        """Test detection of gRPC security patterns"""
        test_file = tmp_path / "grpc_server.go"
        code = '''
package server

import (
    "context"
    "crypto/tls"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/credentials"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/status"
    "go.uber.org/zap"
)

// NewSecureGRPCServer creates a secure gRPC server
func NewSecureGRPCServer(certFile, keyFile string) (*grpc.Server, error) {
    // Load TLS credentials
    creds, err := credentials.NewServerTLSFromFile(certFile, keyFile)
    if err != nil {
        return nil, err
    }

    // Create server with security options
    opts := []grpc.ServerOption{
        grpc.Creds(creds),
        grpc.UnaryInterceptor(authInterceptor),
        grpc.StreamInterceptor(streamAuthInterceptor),
    }

    return grpc.NewServer(opts...), nil
}

// authInterceptor validates authentication for unary RPCs
func authInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    // Skip auth for health check
    if info.FullMethod == "/grpc.health.v1.Health/Check" {
        return handler(ctx, req)
    }

    // Extract metadata
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    // Verify auth token
    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing auth token")
    }

    claims, err := verifyToken(tokens[0])
    if err != nil {
        // Log authentication failure
        logger.Warn("authentication failed",
            zap.String("method", info.FullMethod),
            zap.Error(err),
        )
        return nil, status.Error(codes.Unauthenticated, "invalid token")
    }

    // Add user context
    ctx = context.WithValue(ctx, "user_id", claims.UserID)
    ctx = context.WithValue(ctx, "roles", claims.Roles)

    // Log successful authentication
    logger.Info("authenticated request",
        zap.String("user_id", claims.UserID),
        zap.String("method", info.FullMethod),
    )

    return handler(ctx, req)
}

// RequireRole checks if user has required role
func RequireRole(ctx context.Context, role string) error {
    roles, ok := ctx.Value("roles").([]string)
    if !ok {
        return status.Error(codes.PermissionDenied, "no roles found")
    }

    for _, r := range roles {
        if r == role {
            return nil
        }
    }

    return status.Error(codes.PermissionDenied, "insufficient permissions")
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect gRPC security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-8" in controls or "SC-13" in controls  # TLS
        assert "IA-2" in controls  # Authentication
        assert "AC-3" in controls  # Access control
        assert "AU-2" in controls  # Audit logging

        # Should identify gRPC patterns
        assert any("grpc" in ann.evidence.lower() for ann in results)
        assert any("tls" in ann.evidence.lower() for ann in results)

    def test_detect_input_validation(self, analyzer, tmp_path):
        """Test detection of input validation patterns"""
        test_file = tmp_path / "validation.go"
        code = '''
package validation

import (
    "fmt"
    "net"
    "net/url"
    "regexp"
    "strings"

    "github.com/go-playground/validator/v10"
)

var (
    emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$`)
    phoneRegex = regexp.MustCompile(`^\\+?[1-9]\\d{1,14}$`)
)

// Validator handles input validation
type Validator struct {
    v *validator.Validate
}

// NewValidator creates a new validator instance
func NewValidator() *Validator {
    v := validator.New()

    // Register custom validators
    v.RegisterValidation("safe_string", validateSafeString)
    v.RegisterValidation("strong_password", validateStrongPassword)

    return &Validator{v: v}
}

// ValidateStruct validates a struct
func (val *Validator) ValidateStruct(s interface{}) error {
    return val.v.Struct(s)
}

// ValidateEmail validates email format
func ValidateEmail(email string) bool {
    return emailRegex.MatchString(email)
}

// ValidateURL validates and sanitizes URLs
func ValidateURL(rawURL string) (*url.URL, error) {
    // Parse URL
    u, err := url.Parse(rawURL)
    if err != nil {
        return nil, fmt.Errorf("invalid URL format: %w", err)
    }

    // Check scheme
    if u.Scheme != "http" && u.Scheme != "https" {
        return nil, fmt.Errorf("invalid scheme: %s", u.Scheme)
    }

    // Validate host
    if u.Host == "" {
        return nil, fmt.Errorf("missing host")
    }

    // Check for local addresses
    if isLocalAddress(u.Host) {
        return nil, fmt.Errorf("local addresses not allowed")
    }

    return u, nil
}

// SanitizeString removes dangerous characters
func SanitizeString(input string) string {
    // Remove null bytes
    input = strings.ReplaceAll(input, "\x00", "")

    // Remove control characters
    input = regexp.MustCompile(`[\x00-\x1F\x7F-\x9F]`).ReplaceAllString(input, "")

    // Trim whitespace
    return strings.TrimSpace(input)
}

// validateSafeString custom validator for safe strings
func validateSafeString(fl validator.FieldLevel) bool {
    str := fl.Field().String()

    // Check for SQL injection patterns
    sqlPatterns := []string{
        "';",
        "--",
        "/*",
        "*/",
        "xp_",
        "sp_",
        "OR 1=1",
        "' OR '",
    }

    lower := strings.ToLower(str)
    for _, pattern := range sqlPatterns {
        if strings.Contains(lower, strings.ToLower(pattern)) {
            return false
        }
    }

    return true
}

// validateStrongPassword validates password strength
func validateStrongPassword(fl validator.FieldLevel) bool {
    password := fl.Field().String()

    if len(password) < 12 {
        return false
    }

    var hasUpper, hasLower, hasDigit, hasSpecial bool

    for _, char := range password {
        switch {
        case 'A' <= char && char <= 'Z':
            hasUpper = true
        case 'a' <= char && char <= 'z':
            hasLower = true
        case '0' <= char && char <= '9':
            hasDigit = true
        case strings.ContainsRune("!@#$%^&*()_+-=[]{}|;:,.<>?", char):
            hasSpecial = true
        }
    }

    return hasUpper && hasLower && hasDigit && hasSpecial
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect input validation controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Input validation

        # Should identify various validation patterns
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("sanitize" in ev for ev in evidence_texts)
        assert any("validate" in ev for ev in evidence_texts)
        assert any("sql" in ev for ev in evidence_texts)  # SQL injection prevention

    def test_detect_audit_logging(self, analyzer, tmp_path):
        """Test detection of audit logging patterns"""
        test_file = tmp_path / "audit.go"
        code = '''
package audit

import (
    "context"
    "time"

    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

// @nist-controls: AU-2, AU-3, AU-4
// @evidence: Comprehensive security audit logging
type AuditLogger struct {
    logger *zap.Logger
}

// Event types for audit logging
const (
    EventLogin          = "user.login"
    EventLogout         = "user.logout"
    EventLoginFailed    = "user.login.failed"
    EventAccessDenied   = "access.denied"
    EventDataAccess     = "data.access"
    EventDataModify     = "data.modify"
    EventConfigChange   = "config.change"
    EventSecurityAlert  = "security.alert"
)

// LogSecurityEvent logs security-relevant events
func (a *AuditLogger) LogSecurityEvent(ctx context.Context, event string, fields ...zap.Field) {
    // Extract request context
    reqID := ctx.Value("request_id").(string)
    userID := ctx.Value("user_id").(string)
    clientIP := ctx.Value("client_ip").(string)

    // Build audit fields
    auditFields := []zap.Field{
        zap.String("event_type", event),
        zap.String("request_id", reqID),
        zap.String("user_id", userID),
        zap.String("client_ip", clientIP),
        zap.Time("timestamp", time.Now().UTC()),
        zap.String("audit_version", "1.0"),
    }

    // Add custom fields
    auditFields = append(auditFields, fields...)

    // Log with appropriate level
    switch event {
    case EventSecurityAlert:
        a.logger.Error("Security alert", auditFields...)
    case EventLoginFailed, EventAccessDenied:
        a.logger.Warn("Security event", auditFields...)
    default:
        a.logger.Info("Audit event", auditFields...)
    }
}

// LogDataAccess logs data access events
func (a *AuditLogger) LogDataAccess(userID, resource string, action string, success bool) {
    fields := []zap.Field{
        zap.String("user_id", userID),
        zap.String("resource", resource),
        zap.String("action", action),
        zap.Bool("success", success),
        zap.Time("timestamp", time.Now().UTC()),
    }

    if !success {
        a.logger.Warn("Data access denied", fields...)
    } else {
        a.logger.Info("Data access granted", fields...)
    }
}

// SetupAuditLogger creates audit logger with security requirements
func SetupAuditLogger(logPath string) (*AuditLogger, error) {
    config := zap.NewProductionConfig()

    // Configure for audit requirements
    config.OutputPaths = []string{logPath, "stdout"}
    config.ErrorOutputPaths = []string{logPath, "stderr"}

    // Ensure all required fields are logged
    config.EncoderConfig.TimeKey = "timestamp"
    config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

    // Set sampling to ensure all security events are logged
    config.Sampling = nil

    logger, err := config.Build()
    if err != nil {
        return nil, err
    }

    return &AuditLogger{logger: logger}, nil
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
        assert "AU-4" in controls  # Audit storage capacity

        # Should identify comprehensive logging
        assert any("timestamp" in ann.evidence.lower() for ann in results)
        assert any("security event" in ann.evidence.lower() or "audit" in ann.evidence.lower()
                  for ann in results)

    def test_detect_database_security(self, analyzer, tmp_path):
        """Test detection of database security patterns"""
        test_file = tmp_path / "database.go"
        code = '''
package database

import (
    "database/sql"
    "fmt"

    _ "github.com/lib/pq"
    "github.com/jmoiron/sqlx"
)

// SecureDB wraps database with security features
type SecureDB struct {
    db *sqlx.DB
}

// GetUser safely retrieves user by ID (prevents SQL injection)
func (s *SecureDB) GetUser(userID string) (*User, error) {
    var user User

    // Use parameterized query
    query := `SELECT id, username, email, created_at
              FROM users
              WHERE id = $1 AND deleted_at IS NULL`

    err := s.db.Get(&user, query, userID)
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrUserNotFound
        }
        return nil, fmt.Errorf("failed to get user: %w", err)
    }

    return &user, nil
}

// SearchUsers safely searches with prepared statement
func (s *SecureDB) SearchUsers(searchTerm string) ([]User, error) {
    var users []User

    // Prepare statement for reuse
    stmt, err := s.db.Preparex(`
        SELECT id, username, email, created_at
        FROM users
        WHERE (username ILIKE $1 OR email ILIKE $1)
        AND deleted_at IS NULL
        ORDER BY created_at DESC
        LIMIT 100
    `)
    if err != nil {
        return nil, err
    }
    defer stmt.Close()

    // Execute with bound parameter
    searchPattern := fmt.Sprintf("%%%s%%", searchTerm)
    err = stmt.Select(&users, searchPattern)
    if err != nil {
        return nil, err
    }

    return users, nil
}

// CreateUser safely inserts user with transaction
func (s *SecureDB) CreateUser(user *User) error {
    tx, err := s.db.Beginx()
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // Insert with named parameters
    query := `
        INSERT INTO users (id, username, email, password_hash, created_at)
        VALUES (:id, :username, :email, :password_hash, :created_at)
    `

    _, err = tx.NamedExec(query, user)
    if err != nil {
        return fmt.Errorf("failed to insert user: %w", err)
    }

    // Audit log
    auditQuery := `
        INSERT INTO audit_log (user_id, action, resource, timestamp)
        VALUES ($1, $2, $3, NOW())
    `
    _, err = tx.Exec(auditQuery, user.ID, "create", "user")
    if err != nil {
        return fmt.Errorf("failed to create audit log: %w", err)
    }

    return tx.Commit()
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect database security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Input validation (SQL injection prevention)
        assert "AU-2" in controls or "AU-3" in controls  # Audit logging

        # Should identify parameterized queries
        assert any("parameter" in ann.evidence.lower() or "prepared" in ann.evidence.lower()
                  for ann in results)

    def test_go_mod_analysis(self, analyzer, tmp_path):
        """Test go.mod security analysis"""
        mod_file = tmp_path / "go.mod"
        mod_content = '''module github.com/example/secure-app

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/dgrijalva/jwt-go v3.2.0+incompatible
    golang.org/x/crypto v0.14.0
    github.com/go-playground/validator/v10 v10.15.5
    go.uber.org/zap v1.26.0
    github.com/ulule/limiter/v3 v3.11.2
    github.com/gin-contrib/secure v1.0.1
    github.com/gin-contrib/cors v1.4.0
)
'''
        mod_file.write_text(mod_content)

        results = analyzer._analyze_go_mod(mod_file)

        # Should detect security packages
        assert len(results) >= 6

        # Check specific security packages
        packages = [ann.evidence for ann in results]
        assert any('jwt' in pkg for pkg in packages)
        assert any('crypto' in pkg for pkg in packages)
        assert any('validator' in pkg for pkg in packages)
        assert any('limiter' in pkg for pkg in packages)

    def test_detect_fiber_security(self, analyzer, tmp_path):
        """Test detection of Fiber framework security"""
        test_file = tmp_path / "fiber_app.go"
        code = '''
package main

import (
    "time"

    "github.com/gofiber/fiber/v2"
    "github.com/gofiber/fiber/v2/middleware/cors"
    "github.com/gofiber/fiber/v2/middleware/csrf"
    "github.com/gofiber/fiber/v2/middleware/helmet"
    "github.com/gofiber/fiber/v2/middleware/limiter"
    "github.com/gofiber/fiber/v2/middleware/logger"
)

func setupFiberApp() *fiber.App {
    app := fiber.New(fiber.Config{
        // Security configurations
        ProxyHeader:             "X-Real-IP",
        DisableStartupMessage:   true,
        EnableTrustedProxyCheck: true,
        TrustedProxies:         []string{"10.0.0.0/8"},
    })

    // Security headers
    app.Use(helmet.New(helmet.Config{
        XSSProtection:         "1; mode=block",
        ContentTypeNosniff:    "nosniff",
        XFrameOptions:         "SAMEORIGIN",
        HSTSMaxAge:           31536000,
        HSTSIncludeSubdomains: true,
    }))

    // CORS
    app.Use(cors.New(cors.Config{
        AllowOrigins:     "https://app.example.com",
        AllowCredentials: true,
        AllowMethods:     "GET,POST,PUT,DELETE",
        MaxAge:          86400,
    }))

    // CSRF protection
    app.Use(csrf.New(csrf.Config{
        KeyLookup:      "header:X-CSRF-Token",
        CookieName:     "csrf_token",
        CookieSecure:   true,
        CookieHTTPOnly: true,
        Expiration:     1 * time.Hour,
    }))

    // Rate limiting
    app.Use(limiter.New(limiter.Config{
        Max:        100,
        Expiration: 1 * time.Minute,
        KeyGenerator: func(c *fiber.Ctx) string {
            return c.IP()
        },
        LimitReached: func(c *fiber.Ctx) error {
            return c.Status(429).JSON(fiber.Map{
                "error": "Too many requests",
            })
        },
    }))

    // Logging
    app.Use(logger.New(logger.Config{
        Format:     "${time} ${status} ${method} ${path} ${ip} ${latency}\\n",
        TimeFormat: "2006-01-02 15:04:05",
        Output:     getAuditLogWriter(),
    }))

    return app
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect Fiber security middleware
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-8" in controls or "SC-13" in controls  # HTTPS/Security headers
        assert "SC-5" in controls  # Rate limiting
        assert "AU-2" in controls or "AU-3" in controls  # Logging

        # Should identify Fiber patterns
        assert any("fiber" in ann.evidence.lower() for ann in results)
        assert any("csrf" in ann.evidence.lower() for ann in results)
