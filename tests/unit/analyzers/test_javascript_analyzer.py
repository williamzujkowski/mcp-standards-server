"""
Tests for JavaScript/TypeScript analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive JavaScript analyzer testing
"""


import pytest

from src.analyzers.javascript_analyzer import JavaScriptAnalyzer


class TestJavaScriptAnalyzer:
    """Test JavaScript/TypeScript code analysis capabilities"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return JavaScriptAnalyzer()

    def test_detect_authentication_controls(self, analyzer, tmp_path):
        """Test detection of authentication controls"""
        test_file = tmp_path / "auth.js"
        code = '''
// Authentication middleware
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

/**
 * @nist-controls: IA-2, IA-5
 * @evidence: JWT authentication with bcrypt password hashing
 */
async function authenticateUser(username, password) {
    const user = await User.findOne({ username });

    if (!user || !bcrypt.compareSync(password, user.passwordHash)) {
        throw new AuthenticationError('Invalid credentials');
    }

    // Generate JWT token
    const token = jwt.sign(
        { userId: user.id, roles: user.roles },
        process.env.JWT_SECRET,
        { expiresIn: '24h' }
    );

    return { user, token };
}

// Express middleware for authentication
function requireAuth(req, res, next) {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
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

        # Should have high confidence for explicit annotations
        ia2_results = [r for r in results if "IA-2" in r.control_ids]
        assert any(r.confidence >= 0.9 for r in ia2_results)

    def test_detect_react_security_patterns(self, analyzer, tmp_path):
        """Test detection of React security patterns"""
        test_file = tmp_path / "UserForm.tsx"
        code = '''
import React, { useState } from 'react';
import DOMPurify from 'dompurify';
import { useAuth } from './auth';

interface UserFormProps {
    onSubmit: (data: UserData) => void;
}

export const UserForm: React.FC<UserFormProps> = ({ onSubmit }) => {
    const { user, hasPermission } = useAuth();
    const [formData, setFormData] = useState<UserData>({});

    // Input validation
    const validateEmail = (email: string): boolean => {
        const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
        return emailRegex.test(email);
    };

    // Sanitize user input to prevent XSS
    const sanitizeInput = (input: string): string => {
        return DOMPurify.sanitize(input, { ALLOWED_TAGS: [] });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        // Check permissions
        if (!hasPermission('user.create')) {
            alert('Insufficient permissions');
            return;
        }

        // Validate and sanitize data
        const sanitizedData = {
            ...formData,
            name: sanitizeInput(formData.name),
            email: validateEmail(formData.email) ? formData.email : ''
        };

        onSubmit(sanitizedData);
    };

    return (
        <form onSubmit={handleSubmit}>
            {/* Form fields */}
        </form>
    );
};
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect various security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Input validation
        assert "AC-3" in controls or "AC-6" in controls  # Permission checking

        # Should identify DOMPurify for XSS prevention
        assert any("dompur" in ann.evidence.lower() or "xss" in ann.evidence.lower()
                  for ann in results)

    def test_detect_express_security(self, analyzer, tmp_path):
        """Test detection of Express.js security middleware"""
        test_file = tmp_path / "server.js"
        code = '''
const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const cors = require('cors');
const morgan = require('morgan');

const app = express();

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
        },
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
}));

// CORS configuration
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP'
});

app.use('/api/', limiter);

// Audit logging
app.use(morgan('combined', {
    stream: {
        write: (message) => {
            logger.info(message.trim(), {
                type: 'http',
                timestamp: new Date().toISOString()
            });
        }
    }
}));
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-8" in controls or "SC-13" in controls  # HTTPS/TLS (helmet HSTS)
        assert "SC-5" in controls  # DoS protection (rate limiting)
        assert "AU-2" in controls or "AU-3" in controls  # Audit logging (morgan)

        # Should identify specific middleware
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("helmet" in ev for ev in evidence_texts)
        assert any("rate" in ev and "limit" in ev for ev in evidence_texts)

    def test_detect_crypto_patterns(self, analyzer, tmp_path):
        """Test detection of cryptographic patterns"""
        test_file = tmp_path / "crypto.ts"
        code = '''
import crypto from 'crypto';
import { promisify } from 'util';

const scrypt = promisify(crypto.scrypt);

export class CryptoService {
    private algorithm = 'aes-256-gcm';

    /**
     * Encrypt sensitive data
     * @nist-controls: SC-28
     * @evidence: AES-256-GCM encryption for data at rest
     */
    async encrypt(text: string, password: string): Promise<EncryptedData> {
        const salt = crypto.randomBytes(32);
        const key = await scrypt(password, salt, 32) as Buffer;
        const iv = crypto.randomBytes(16);

        const cipher = crypto.createCipheriv(this.algorithm, key, iv);
        const encrypted = Buffer.concat([
            cipher.update(text, 'utf8'),
            cipher.final()
        ]);

        const tag = cipher.getAuthTag();

        return {
            encrypted: encrypted.toString('base64'),
            salt: salt.toString('base64'),
            iv: iv.toString('base64'),
            tag: tag.toString('base64')
        };
    }

    // Generate secure random tokens
    generateSecureToken(length: number = 32): string {
        return crypto.randomBytes(length).toString('hex');
    }

    // Hash passwords with salt
    async hashPassword(password: string): Promise<string> {
        const salt = crypto.randomBytes(16).toString('hex');
        const hash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
        return `${salt}:${hash}`;
    }
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
        assert "IA-5" in controls  # Password hashing

        # Should identify strong algorithms
        assert any("aes-256" in ann.evidence.lower() for ann in results)

    def test_detect_angular_security(self, analyzer, tmp_path):
        """Test detection of Angular security patterns"""
        test_file = tmp_path / "user.component.ts"
        code = '''
import { Component, OnInit } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { AuthService } from './auth.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
    selector: 'app-user',
    template: `
        <div [innerHTML]="sanitizedContent"></div>
        <form [formGroup]="userForm" (ngSubmit)="onSubmit()">
            <input formControlName="email" type="email">
            <button [disabled]="!canEdit">Submit</button>
        </form>
    `
})
export class UserComponent implements OnInit {
    userForm: FormGroup;
    sanitizedContent: SafeHtml;

    constructor(
        private sanitizer: DomSanitizer,
        private auth: AuthService,
        private fb: FormBuilder
    ) {}

    ngOnInit() {
        // Form validation
        this.userForm = this.fb.group({
            email: ['', [Validators.required, Validators.email]],
            password: ['', [Validators.required, Validators.minLength(8)]],
            username: ['', [Validators.required, Validators.pattern(/^[a-zA-Z0-9_]+$/)]
        });

        // Sanitize user content
        const userContent = this.getUserContent();
        this.sanitizedContent = this.sanitizer.sanitize(SecurityContext.HTML, userContent);
    }

    get canEdit(): boolean {
        return this.auth.hasRole('editor') || this.auth.hasRole('admin');
    }

    onSubmit() {
        if (!this.auth.isAuthenticated()) {
            this.router.navigate(['/login']);
            return;
        }

        if (this.userForm.valid) {
            // Process form
        }
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect Angular security patterns
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Input validation
        assert "AC-3" in controls  # Access control (role checking)

        # Should identify Angular sanitization
        assert any("sanitiz" in ann.evidence.lower() for ann in results)

    def test_detect_vue_security(self, analyzer, tmp_path):
        """Test detection of Vue.js security patterns"""
        test_file = tmp_path / "UserProfile.vue"
        code = '''
<template>
    <div>
        <!-- Automatic XSS protection with v-text -->
        <p v-text="userBio"></p>

        <!-- Manual sanitization for v-html -->
        <div v-html="sanitizedContent"></div>

        <form @submit.prevent="updateProfile">
            <input
                v-model="form.email"
                type="email"
                :pattern="emailPattern"
                required
            >
            <button :disabled="!canEdit">Update</button>
        </form>
    </div>
</template>

<script>
import DOMPurify from 'dompurify';
import { mapGetters } from 'vuex';

export default {
    name: 'UserProfile',

    data() {
        return {
            form: {
                email: '',
                bio: ''
            },
            emailPattern: '^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$'
        };
    },

    computed: {
        ...mapGetters(['currentUser', 'hasPermission']),

        canEdit() {
            return this.hasPermission('profile.edit') && this.isOwner;
        },

        sanitizedContent() {
            return DOMPurify.sanitize(this.form.bio, {
                ALLOWED_TAGS: ['p', 'br', 'strong', 'em']
            });
        }
    },

    methods: {
        async updateProfile() {
            // CSRF token included automatically by axios interceptor
            if (!this.validateForm()) {
                return;
            }

            try {
                await this.$http.put('/api/profile', this.form);
                this.$notify.success('Profile updated');
            } catch (error) {
                this.$notify.error('Update failed');
            }
        },

        validateForm() {
            // Additional validation
            const emailRegex = new RegExp(this.emailPattern);
            return emailRegex.test(this.form.email);
        }
    }
};
</script>
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect Vue security patterns
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SI-10" in controls  # Input validation
        assert "AC-3" in controls or "AC-6" in controls  # Permission checking

        # Should identify Vue-specific patterns
        assert any("v-text" in ann.evidence or "v-html" in ann.evidence for ann in results)

    def test_detect_websocket_security(self, analyzer, tmp_path):
        """Test detection of WebSocket security"""
        test_file = tmp_path / "websocket.js"
        code = '''
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');

class SecureWebSocketServer {
    constructor(server) {
        this.wss = new WebSocket.Server({
            server,
            verifyClient: this.verifyClient.bind(this)
        });

        this.clients = new Map();
        this.setupHandlers();
    }

    // Authenticate WebSocket connections
    verifyClient(info, cb) {
        const token = this.extractToken(info.req);

        if (!token) {
            cb(false, 401, 'Unauthorized');
            return;
        }

        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            info.req.user = decoded;
            cb(true);
        } catch (err) {
            cb(false, 401, 'Invalid token');
        }
    }

    setupHandlers() {
        this.wss.on('connection', (ws, req) => {
            const userId = req.user.id;

            // Rate limiting per client
            const limiter = {
                messages: 0,
                resetTime: Date.now() + 60000
            };

            this.clients.set(userId, { ws, limiter });

            ws.on('message', (data) => {
                // Rate limit check
                if (Date.now() > limiter.resetTime) {
                    limiter.messages = 0;
                    limiter.resetTime = Date.now() + 60000;
                }

                if (limiter.messages++ > 100) {
                    ws.close(1008, 'Rate limit exceeded');
                    return;
                }

                // Validate message format
                try {
                    const message = JSON.parse(data);
                    this.handleMessage(userId, message);
                } catch (err) {
                    ws.send(JSON.stringify({ error: 'Invalid message format' }));
                }
            });
        });
    }
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect WebSocket security controls
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-2" in controls  # Authentication
        assert "SC-5" in controls  # DoS protection (rate limiting)
        assert "SI-10" in controls  # Input validation

    def test_package_json_analysis(self, analyzer, tmp_path):
        """Test package.json security analysis"""
        pkg_file = tmp_path / "package.json"
        pkg_content = '''{
    "name": "secure-app",
    "version": "1.0.0",
    "dependencies": {
        "express": "^4.18.2",
        "helmet": "^7.0.0",
        "bcryptjs": "^2.4.3",
        "jsonwebtoken": "^9.0.0",
        "express-rate-limit": "^6.7.0",
        "express-validator": "^7.0.1",
        "cors": "^2.8.5",
        "dotenv": "^16.0.3"
    },
    "devDependencies": {
        "eslint": "^8.45.0",
        "eslint-plugin-security": "^1.7.1",
        "jest": "^29.5.0",
        "snyk": "^1.1100.0"
    },
    "scripts": {
        "test": "jest",
        "lint": "eslint .",
        "security": "snyk test"
    }
}'''
        pkg_file.write_text(pkg_content)

        results = analyzer._analyze_config_file(pkg_file)

        # Should detect security packages
        assert len(results) >= 7

        # Check specific security packages
        packages = [ann.evidence for ann in results]
        assert any('helmet' in pkg for pkg in packages)
        assert any('bcrypt' in pkg for pkg in packages)
        assert any('jsonwebtoken' in pkg or 'jwt' in pkg for pkg in packages)
        assert any('snyk' in pkg for pkg in packages)

    def test_typescript_interfaces(self, analyzer, tmp_path):
        """Test TypeScript interface analysis"""
        test_file = tmp_path / "types.ts"
        code = '''
// Security-related interfaces
export interface User {
    id: string;
    username: string;
    passwordHash: string;  // Never store plain text
    roles: Role[];
    permissions: Permission[];
    mfaEnabled: boolean;
    lastLogin: Date;
}

export interface AuthToken {
    token: string;
    expiresAt: Date;
    refreshToken?: string;
    csrfToken: string;
}

export interface Permission {
    resource: string;
    action: 'read' | 'write' | 'delete' | 'admin';
    conditions?: Record<string, any>;
}

export interface AuditLog {
    userId: string;
    action: string;
    resource: string;
    timestamp: Date;
    ipAddress: string;
    userAgent: string;
    result: 'success' | 'failure';
}
'''
        test_file.write_text(code)

        results = analyzer.analyze_file(test_file)

        # Should detect security-related types
        assert len(results) >= 4

        # Should identify security concepts in interfaces
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-2" in controls or "IA-5" in controls  # Auth interfaces
        assert "AC-3" in controls or "AC-6" in controls  # Permission interface
        assert "AU-2" in controls or "AU-3" in controls  # Audit log interface
