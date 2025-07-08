// TypeScript/React example demonstrating various patterns for analyzer testing

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { exec } from 'child_process';
import * as _ from 'lodash'; // Large import
import moment from 'moment'; // Deprecated library

// Hardcoded secrets - security issue
const API_KEY = "sk_live_4242424242424242";
const password = "admin123";
const mongoUrl = "mongodb://admin:password@localhost:27017/mydb";

// Use of 'any' type - type safety issue
function processData(input: any): any {
    const result: any = {};
    const items: any[] = input.items;
    
    return items.map((item: any) => {
        // Type assertions without validation
        const id = (item as { id: number }).id;
        const user = item as any;
        const name = user!.name; // Non-null assertion
        
        return { id, name };
    });
}

// @ts-ignore usage
// @ts-ignore
const invalidCode = undefined.property.access;

// @ts-ignore
function brokenFunction(): string {
    return notDefinedVariable;
}

// XSS vulnerabilities
const DangerousComponent: React.FC<{ html: string }> = ({ html }) => {
    return (
        <>
            {/* XSS via dangerouslySetInnerHTML */}
            <div dangerouslySetInnerHTML={{ __html: html }} />
            
            {/* Direct DOM manipulation */}
            <div id="output" />
        </>
    );
};

function unsafeDOM(userInput: string) {
    // Direct innerHTML assignment
    document.getElementById('output')!.innerHTML = userInput;
    
    // document.write usage
    document.write('<script>' + userInput + '</script>');
    
    // eval usage
    eval(userInput);
    
    // Dynamic function creation
    const func = new Function('input', userInput);
    
    // Unsafe URL redirect
    window.location.href = userInput;
}

// SQL Injection vulnerabilities
async function vulnerableQueries(userId: string, name: string) {
    // Template literal SQL injection
    await db.query(`SELECT * FROM users WHERE id = ${userId}`);
    
    // String concatenation SQL injection
    await db.execute("DELETE FROM users WHERE name = '" + name + "'");
}

// Command injection
function commandInjection(filename: string) {
    // Command injection via template literal
    exec(`cat ${filename}`, (err, stdout) => {
        console.log(stdout);
    });
}

// JWT in localStorage - security issue
function insecureTokenStorage(token: string) {
    localStorage.setItem('jwt', token);
    localStorage.setItem('authToken', token);
    sessionStorage.setItem('token', token);
}

// Console logging sensitive data
function logSensitiveData(user: any) {
    console.log("User password:", user.password);
    console.debug("API key:", API_KEY);
    console.error("Token:", user.token);
}

// React performance issues
const ExpensiveComponent: React.FC<{ data: any[] }> = ({ data }) => {
    // Missing React.memo
    return (
        <div>
            {data.map(item => (
                <div key={item.id}>
                    {/* Inline function in JSX */}
                    <button onClick={() => console.log(item)}>
                        Click
                    </button>
                </div>
            ))}
        </div>
    );
};

const BadHooksComponent: React.FC = () => {
    const [count, setCount] = useState(0);
    
    // useEffect without dependency array
    useEffect(() => {
        fetchData();
    });
    
    // useMemo without dependency array
    const computed = useMemo(() => {
        return expensiveComputation(count);
    });
    
    // useCallback without dependency array
    const handler = useCallback((e: any) => {
        console.log(e);
    });
    
    return <div>{count}</div>;
};

// Large list without virtualization
const LargeListComponent: React.FC<{ items: any[] }> = ({ items }) => {
    return (
        <ul>
            {items.map(item => (
                <li key={item.id}>{item.name}</li>
            ))}
        </ul>
    );
};

// Sequential awaits that could be parallel
async function inefficientAsync() {
    const user = await fetchUser();
    const profile = await fetchProfile();
    const settings = await fetchSettings();
    
    return { user, profile, settings };
}

// Async function without error handling
async function noErrorHandling() {
    const data = await fetchData();
    const processed = await processAsync(data);
    return processed;
}

// Promise without error handling
function promiseWithoutCatch() {
    fetchData()
        .then(data => processData(data))
        .then(result => saveResult(result));
    // Missing .catch()
    
    new Promise((resolve, reject) => {
        doAsyncWork();
    }); // No error handling
}

// Inefficient array operations
function inefficientArrays(items: number[]) {
    // Multiple iterations
    const result = items
        .filter(x => x > 0)
        .map(x => x * 2);
    
    // Inefficient find().length check
    const hasLarge = items.find(x => x > 100).length;
    
    // Filter().length === 0
    const isEmpty = items.filter(x => x > 0).length === 0;
    
    // Inefficient deep clone
    const clone = JSON.parse(JSON.stringify(items));
}

// Old JavaScript patterns
var oldVariable = 42; // Should use const/let

function oldStyleCallback() {
    return function() { // Should use arrow function
        console.log("callback");
    };
}

const hasItem = array.indexOf(item) !== -1; // Should use includes()

const merged = Object.assign({}, obj1, obj2); // Should use spread

// Empty error handling
function badErrorHandling() {
    try {
        riskyOperation();
    } catch (e) {
        // Empty catch block
    }
}

// Routes without authentication checks
router.get('/api/admin/users', (req, res) => {
    // No auth check
    res.json(getAllUsers());
});

router.post('/api/admin/delete', (req, res) => {
    // Missing authentication
    deleteUser(req.params.id);
});

// File too long (simulated with comments)
// ... imagine 300+ more lines of code here ...

// Component with API calls (mixed concerns)
const DataComponent: React.FC = () => {
    useEffect(() => {
        // API call directly in component
        fetch('/api/data')
            .then(res => res.json())
            .then(data => setData(data));
    }, []);
    
    return <div>Data Component</div>;
};

// Test file pattern (in .test.tsx file)
// Missing proper test structure
console.log("This should be a proper test file");

// Type definitions
interface UserData {
    id: number;
    name: string;
    email: string;
}

type Status = 'active' | 'inactive' | 'pending';

// Excessive type assertions
function unsafeTypeUsage(data: unknown) {
    const user = data as any;
    const typed = <any>data;
    return typed;
}

// Missing return type annotations
function noReturnType(a: number, b: number) {
    return a + b;
}

// Large imports affecting bundle size
import { debounce, throttle, merge, clone, isEqual } from 'lodash';

export default function AnalyzerTestComponent() {
    return <div>TypeScript Analyzer Test</div>;
}