/* Extra JavaScript for MCP Standards Server documentation */

document.addEventListener('DOMContentLoaded', function() {
    // Add any custom JavaScript functionality here
    
    // Example: Add copy button functionality to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(function(block) {
        // Additional code block enhancements can be added here
    });
    
    // Example: Add scroll-to-top functionality
    const scrollToTop = function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    
    // Console log for debugging
    console.log('MCP Standards Server documentation scripts loaded');
});