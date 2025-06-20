/**
 * NIST-Compliant JavaScript Frontend Example
 * @nist-controls AC-3, IA-2, SC-8, SI-10, SI-11
 * @evidence Secure frontend with authentication and validation
 */

// @nist-controls: SC-8
// @evidence: HTTPS enforcement
if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
    location.replace('https:' + window.location.href.substring(window.location.protocol.length));
}

class SecureApp {
    constructor() {
        this.apiBase = 'https://api.example.com';
        this.token = localStorage.getItem('auth_token');
        this.currentUser = null;
        
        // @nist-controls: SI-10
        // @evidence: Input validation patterns
        this.validators = {
            email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            password: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
            username: /^[a-zA-Z0-9_]{3,20}$/
        };
        
        this.init();
    }
    
    init() {
        this.setupCSP();
        this.setupEventListeners();
        this.checkAuthentication();
    }
    
    // @nist-controls: SI-10
    // @evidence: Content Security Policy setup
    setupCSP() {
        const meta = document.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
        document.head.appendChild(meta);
    }
    
    setupEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.bindFormEvents();
            this.setupLogoutTimer();
        });
    }
    
    // @nist-controls: IA-2
    // @evidence: User authentication
    async login(username, password) {
        try {
            // Validate inputs
            if (!this.validateInput('username', username)) {
                throw new Error('Invalid username format');
            }
            
            if (!this.validateInput('password', password)) {
                throw new Error('Password does not meet security requirements');
            }
            
            const response = await this.secureRequest('/api/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });
            
            if (response.access_token) {
                this.token = response.access_token;
                localStorage.setItem('auth_token', this.token);
                
                // @nist-controls: AU-2
                // @evidence: Authentication event logging
                this.logSecurityEvent('LOGIN_SUCCESS', { username });
                
                await this.loadUserProfile();
                this.showDashboard();
            }
            
        } catch (error) {
            // @nist-controls: SI-11
            // @evidence: Secure error handling
            this.logSecurityEvent('LOGIN_FAILED', { username, error: error.message });
            this.showError('Login failed. Please check your credentials.');
        }
    }
    
    // @nist-controls: AC-3
    // @evidence: Session management and logout
    logout() {
        this.logSecurityEvent('LOGOUT', { username: this.currentUser?.username });
        
        localStorage.removeItem('auth_token');
        this.token = null;
        this.currentUser = null;
        
        // Clear sensitive data
        this.clearSensitiveData();
        
        this.showLogin();
    }
    
    // @nist-controls: SI-10
    // @evidence: Input validation
    validateInput(type, value) {
        if (!value || typeof value !== 'string') {
            return false;
        }
        
        // Length check
        if (value.length > 1000) {
            return false;
        }
        
        // Type-specific validation
        if (this.validators[type]) {
            return this.validators[type].test(value);
        }
        
        // Basic sanitization check
        const dangerousPatterns = [
            /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
            /javascript:/gi,
            /on\w+\s*=/gi
        ];
        
        return !dangerousPatterns.some(pattern => pattern.test(value));
    }
    
    // @nist-controls: SC-8
    // @evidence: Secure API communication
    async secureRequest(endpoint, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        };
        
        // Add authentication header if token exists
        if (this.token) {
            defaultOptions.headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        const finalOptions = {
            ...defaultOptions,
            ...options,
            headers: { ...defaultOptions.headers, ...options.headers }
        };
        
        try {
            const response = await fetch(this.apiBase + endpoint, finalOptions);
            
            if (response.status === 401) {
                // Token expired or invalid
                this.logout();
                throw new Error('Authentication required');
            }
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
            
        } catch (error) {
            this.logSecurityEvent('API_ERROR', { 
                endpoint, 
                error: error.message,
                status: error.status 
            });
            throw error;
        }
    }
    
    // @nist-controls: AU-2
    // @evidence: Security event logging
    logSecurityEvent(event, details = {}) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            event,
            userAgent: navigator.userAgent,
            url: location.href,
            ...details
        };
        
        // In production, send to logging service
        console.info('Security Event:', logEntry);
        
        // Store locally for audit trail
        const logs = JSON.parse(localStorage.getItem('security_logs') || '[]');
        logs.push(logEntry);
        
        // Keep only last 100 entries
        if (logs.length > 100) {
            logs.splice(0, logs.length - 100);
        }
        
        localStorage.setItem('security_logs', JSON.stringify(logs));
    }
    
    // @nist-controls: AC-3
    // @evidence: Session timeout
    setupLogoutTimer() {
        let timeout;
        const TIMEOUT_DURATION = 30 * 60 * 1000; // 30 minutes
        
        const resetTimer = () => {
            clearTimeout(timeout);
            if (this.token) {
                timeout = setTimeout(() => {
                    this.showError('Session expired due to inactivity');
                    this.logout();
                }, TIMEOUT_DURATION);
            }
        };
        
        // Reset timer on user activity
        ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetTimer, true);
        });
        
        resetTimer();
    }
    
    // @nist-controls: SC-8
    // @evidence: Clear sensitive data
    clearSensitiveData() {
        // Clear forms
        document.querySelectorAll('input[type="password"]').forEach(input => {
            input.value = '';
        });
        
        // Clear any cached sensitive data
        const sensitiveKeys = ['temp_data', 'user_session'];
        sensitiveKeys.forEach(key => {
            localStorage.removeItem(key);
            sessionStorage.removeItem(key);
        });
    }
    
    async checkAuthentication() {
        if (this.token) {
            try {
                await this.loadUserProfile();
                this.showDashboard();
            } catch (error) {
                this.logout();
            }
        } else {
            this.showLogin();
        }
    }
    
    async loadUserProfile() {
        const profile = await this.secureRequest('/api/profile');
        this.currentUser = profile;
    }
    
    bindFormEvents() {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                this.login(username, password);
            });
        }
        
        // Logout button
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }
    }
    
    showLogin() {
        document.body.innerHTML = `
            <div class="login-container">
                <h2>Secure Login</h2>
                <form id="loginForm">
                    <div>
                        <label for="username">Username:</label>
                        <input type="text" id="username" required maxlength="20">
                    </div>
                    <div>
                        <label for="password">Password:</label>
                        <input type="password" id="password" required maxlength="100">
                    </div>
                    <button type="submit">Login</button>
                </form>
                <div id="error-message"></div>
            </div>
        `;
        this.bindFormEvents();
    }
    
    showDashboard() {
        document.body.innerHTML = `
            <div class="dashboard">
                <h2>Dashboard</h2>
                <p>Welcome, ${this.currentUser.username}!</p>
                <p>Roles: ${this.currentUser.roles.join(', ')}</p>
                <button id="logoutBtn">Logout</button>
            </div>
        `;
        this.bindFormEvents();
    }
    
    // @nist-controls: SI-11
    // @evidence: Secure error display
    showError(message) {
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.color = 'red';
            
            // Clear error after 5 seconds
            setTimeout(() => {
                errorDiv.textContent = '';
            }, 5000);
        }
    }
}

// Initialize the application
const app = new SecureApp();