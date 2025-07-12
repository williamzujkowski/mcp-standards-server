"""Tests for TypeScript language analyzer."""

import pytest

from src.analyzers.base import IssueType
from src.analyzers.typescript_analyzer import TypeScriptAnalyzer


class TestTypeScriptAnalyzer:
    """Test TypeScript analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create TypeScript analyzer instance."""
        return TypeScriptAnalyzer()

    def test_language_properties(self, analyzer):
        """Test language and file extension properties."""
        assert analyzer.language == "typescript"
        assert analyzer.file_extensions == [".ts", ".tsx", ".js", ".jsx"]

    def test_xss_vulnerabilities(self, analyzer, tmp_path):
        """Test XSS vulnerability detection."""
        test_file = tmp_path / "xss.tsx"
        test_file.write_text(
            """
import React from 'react';

const DangerousComponent = ({ userInput }) => {
    return (
        <div>
            <div dangerouslySetInnerHTML={{ __html: userInput }} />
            <div id="output"></div>
        </div>
    );
};

function unsafeDOM() {
    document.getElementById('output').innerHTML = getUserInput();
    document.write('<script>alert("XSS")</script>');
    eval(getUserCode());
    new Function(getUserCode())();
}

function unsafeRedirect() {
    window.location.href = getUserUrl();
    window.open(request.params.url);
}
"""
        )

        result = analyzer.analyze_file(test_file)

        xss_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-79"
        ]
        assert len(xss_issues) >= 4

        redirect_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-601"
        ]
        assert len(redirect_issues) >= 2

    def test_injection_vulnerabilities(self, analyzer, tmp_path):
        """Test injection vulnerability detection."""
        test_file = tmp_path / "injection.ts"
        test_file.write_text(
            """
import { exec, spawn } from 'child_process';

async function sqlInjection(userId: string) {
    const query = `SELECT * FROM users WHERE id = ${userId}`;
    await db.query(query);

    const deleteQuery = "DELETE FROM users WHERE name = '" + userName + "'";
    await db.execute(deleteQuery);
}

function commandInjection(userInput: string) {
    exec(`ls -la ${userInput}`);
    spawn('rm', [`-rf ${userInput}`]);
}
"""
        )

        result = analyzer.analyze_file(test_file)

        sql_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-89"
        ]
        assert len(sql_issues) >= 2

        cmd_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-78"
        ]
        assert len(cmd_issues) >= 2

    def test_type_safety_issues(self, analyzer, tmp_path):
        """Test TypeScript type safety issues."""
        test_file = tmp_path / "types.ts"
        test_file.write_text(
            """
// Excessive use of any
function processData(input: any): any {
    const result: any = {};
    const items: any[] = input.items;
    return items.map((item: any) => item.value);
}

// Type assertions without validation
function unsafeAssertions(data: unknown) {
    const user = data as any;
    const id = (data as { id: number }).id;
    const name = user!.name;  // Non-null assertion
}

// @ts-ignore usage
// @ts-ignore
const invalidCode = undefined.property;

// @ts-ignore
function brokenFunction() {
    return notDefined;
}
"""
        )

        result = analyzer.analyze_file(test_file)

        any_issues = [
            i
            for i in result.issues
            if i.type == IssueType.TYPE_SAFETY and "'any'" in i.message
        ]
        assert len(any_issues) >= 1

        assertion_issues = [
            i for i in result.issues if "assertion" in i.message.lower()
        ]
        assert len(assertion_issues) >= 2

        tsignore_issues = [i for i in result.issues if "@ts-ignore" in i.message]
        assert len(tsignore_issues) >= 1

    def test_auth_patterns(self, analyzer, tmp_path):
        """Test authentication pattern issues."""
        test_file = tmp_path / "auth.ts"
        test_file.write_text(
            """
// JWT in localStorage
function saveToken(token: string) {
    localStorage.setItem('jwt', token);
    localStorage.setItem('token', token);
    sessionStorage.setItem('jwt', token);
}

// Routes without auth
router.get('/api/users', (req, res) => {
    // No auth check
    res.json(getAllUsers());
});

router.post('/api/admin', (req, res) => {
    // Missing authentication
    performAdminAction();
});

// Route with auth
router.get('/api/profile', requireAuth, (req, res) => {
    res.json(req.user);
});
"""
        )

        result = analyzer.analyze_file(test_file)

        storage_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-522"
        ]
        assert len(storage_issues) >= 3

        auth_issues = [
            i for i in result.issues if "authentication" in i.message.lower()
        ]
        assert len(auth_issues) >= 2

    def test_sensitive_data_exposure(self, analyzer, tmp_path):
        """Test sensitive data exposure detection."""
        test_file = tmp_path / "secrets.ts"
        test_file.write_text(
            """
const apiKey = "sk_live_4242424242424242";
const password = "admin123";
const privateKey = "-----BEGIN PRIVATE KEY-----MIIEvQ...";
const dbUrl = "mongodb://admin:password123@localhost:27017/mydb";

console.log("User password:", password);
console.error("API token:", token);
console.debug("Secret key:", secretKey);
"""
        )

        result = analyzer.analyze_file(test_file)

        secret_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-798"
        ]
        assert len(secret_issues) >= 4

        log_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-532"
        ]
        assert len(log_issues) >= 3

    def test_react_performance(self, analyzer, tmp_path):
        """Test React performance issues."""
        test_file = tmp_path / "react-perf.tsx"
        test_file.write_text(
            """
import React, { useEffect, useState, useMemo, useCallback } from 'react';

const ExpensiveComponent = ({ data }) => {
    // Component without React.memo
    return <div>{data.map(item => <Item key={item.id} {...item} />)}</div>;
};

const BadComponent = () => {
    const [count, setCount] = useState(0);

    // Inline function in JSX
    return (
        <button onClick={() => setCount(count + 1)}>
            Count: {count}
        </button>
    );
};

const MissingDeps = () => {
    // Missing dependency array
    useEffect(() => {
        fetchData();
    });

    // Missing dependency array
    const computed = useMemo(() => expensiveComputation());

    // Missing dependency array
    const handler = useCallback((e) => console.log(e));

    return <div />;
};

const LargeList = ({ items }) => {
    // Large list without virtualization
    return (
        <ul>
            {items.map(item => (
                <li key={item.id}>{item.name}</li>
            ))}
        </ul>
    );
};
"""
        )

        result = analyzer.analyze_file(test_file)

        memo_issues = [i for i in result.issues if "React.memo" in i.message]
        assert len(memo_issues) >= 1

        inline_issues = [
            i for i in result.issues if "inline function" in i.message.lower()
        ]
        assert len(inline_issues) >= 1

        dep_issues = [i for i in result.issues if "dependency array" in i.message]
        assert len(dep_issues) >= 3

        virtual_issues = [
            i for i in result.issues if "virtualization" in i.message.lower()
        ]
        assert len(virtual_issues) >= 1

    def test_javascript_performance(self, analyzer, tmp_path):
        """Test general JavaScript performance issues."""
        test_file = tmp_path / "js-perf.ts"
        test_file.write_text(
            """
function inefficientArrayOps(items: number[]) {
    // Multiple iterations
    const filtered = items.filter(x => x > 0).map(x => x * 2);

    // Inefficient checks
    const hasItems = items.find(x => x > 100).length;
    const isEmpty = items.filter(x => x > 0).length === 0;

    // Inefficient clone
    const clone = JSON.parse(JSON.stringify(items));
}
"""
        )

        result = analyzer.analyze_file(test_file)

        perf_issues = [i for i in result.issues if i.type == IssueType.PERFORMANCE]
        assert len(perf_issues) >= 4

    def test_async_patterns(self, analyzer, tmp_path):
        """Test async/await patterns."""
        test_file = tmp_path / "async.ts"
        test_file.write_text(
            """
async function sequentialAwaits() {
    const user = await fetchUser();
    const profile = await fetchProfile();
    const settings = await fetchSettings();
    return { user, profile, settings };
}

async function noErrorHandling() {
    const data = await fetchData();
    const processed = await processData(data);
    return processed;
}

async function properErrorHandling() {
    try {
        const data = await fetchData();
        return data;
    } catch (error) {
        console.error(error);
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        parallel_issues = [i for i in result.issues if "parallel" in i.message.lower()]
        assert len(parallel_issues) >= 1

        error_issues = [i for i in result.issues if i.type == IssueType.ERROR_HANDLING]
        assert len(error_issues) >= 1

    def test_bundle_size_issues(self, analyzer, tmp_path):
        """Test bundle size optimization issues."""
        test_file = tmp_path / "imports.ts"
        test_file.write_text(
            """
import * as _ from 'lodash';
import moment from 'moment';
import * as rxjs from 'rxjs';

// Better imports
import debounce from 'lodash/debounce';
import { format } from 'date-fns';
"""
        )

        result = analyzer.analyze_file(test_file)

        bundle_issues = [i for i in result.issues if "bundle size" in i.message.lower()]
        assert len(bundle_issues) >= 3

    def test_error_handling(self, analyzer, tmp_path):
        """Test error handling patterns."""
        test_file = tmp_path / "errors.ts"
        test_file.write_text(
            """
function badErrorHandling() {
    try {
        riskyOperation();
    } catch (e) {
        // Empty catch block
    }

    somePromise()
        .then(data => processData(data));
        // Missing .catch()

    new Promise((resolve, reject) => {
        doAsyncWork();
    }); // No error handling
}
"""
        )

        result = analyzer.analyze_file(test_file)

        empty_catch = [i for i in result.issues if "empty catch" in i.message.lower()]
        assert len(empty_catch) >= 1

        promise_issues = [i for i in result.issues if "promise" in i.message.lower()]
        assert len(promise_issues) >= 2

    def test_modern_features(self, analyzer, tmp_path):
        """Test modern JavaScript feature recommendations."""
        test_file = tmp_path / "old-patterns.ts"
        test_file.write_text(
            """
var oldVariable = 42;

function oldCallback() {
    return function() {
        console.log("callback");
    };
}

const hasItem = array.indexOf(item) !== -1;

const merged = Object.assign({}, obj1, obj2);

const args = Array.prototype.slice.call(arguments);
"""
        )

        result = analyzer.analyze_file(test_file)

        modern_issues = [
            i
            for i in result.issues
            if i.type == IssueType.BEST_PRACTICE
            and "modern" in i.recommendation.lower()
        ]
        assert len(modern_issues) >= 5

    def test_testing_patterns(self, analyzer, tmp_path):
        """Test testing pattern detection."""
        # Test file without proper structure
        test_file = tmp_path / "bad.test.ts"
        test_file.write_text(
            """
// Test file but no tests
console.log("This is supposed to be a test file");
"""
        )

        result = analyzer.analyze_file(test_file)

        test_structure_issues = [
            i for i in result.issues if "test structure" in i.message.lower()
        ]
        assert len(test_structure_issues) >= 1

        assertion_issues = [
            i for i in result.issues if "assertion" in i.message.lower()
        ]
        assert len(assertion_issues) >= 1

        # Non-test file
        module_file = tmp_path / "module.ts"
        module_file.write_text(
            """
export function add(a: number, b: number): number {
    return a + b;
}

export function multiply(a: number, b: number): number {
    return a * b;
}

export class Calculator {
    // Complex logic
}

export interface Config {
    // Configuration
}

export type Result = {
    // Result type
}

export const utils = {
    // Utilities
}
"""
        )

        result2 = analyzer.analyze_file(module_file)

        test_recommendation = [
            i
            for i in result2.issues
            if "test" in i.message.lower() and "adding" in i.message.lower()
        ]
        assert len(test_recommendation) >= 1
