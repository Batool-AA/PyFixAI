import React, { useState, useEffect } from "react";
import "./App.css";

// Code protection patterns to check for problematic content
const problematicPatterns = [
  {
    pattern: /\b(gender|race|ethnicity|religion|disability|nationality)\b/i,
    category: "Personal identifiers",
    description: "Code references potentially sensitive demographic identifiers"
  },
  {
    pattern: /\b(blacklist|whitelist|master|slave|illegal alien)\b/i,
    category: "Problematic terminology",
    description: "Code uses terms that may perpetuate harmful stereotypes"
  },
  {
    pattern: /if\s*\(\s*(?:gender|race|ethnicity|religion)\s*==?\s*['"](.+?)['"]\)/i,
    category: "Conditional logic based on protected attributes",
    description: "Code contains conditional logic based on protected attributes"
  },
  {
    pattern: /\b(discrimination|prejudice|unfair|bias)\b/i,
    category: "Algorithmic bias indicators",
    description: "Code contains terms that may indicate algorithmic bias"
  }
];

function App() {
  const [code, setCode] = useState("");
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState("fix"); // "fix" or "explain"
  const [theme, setTheme] = useState("light");
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [validationError, setValidationError] = useState("");
  
  // New state variables for code protection
  const [codeIssues, setCodeIssues] = useState([]);
  const [showGuardrailWarning, setShowGuardrailWarning] = useState(false);
  const [bypassGuardrails, setBypassGuardrails] = useState(false);

  // Check if input text is likely Python code
  const isPythonCode = (text) => {
    // Basic validation to check if the text looks like Python code
    if (!text || text.trim() === '') return false;
    
    const pythonIndicators = [
      // Check for Python keywords
      /\b(def|class|import|from|if|elif|else|for|while|try|except|return|yield|with)\b/,
      // Check for Python-style indentation
      /^( {4}|\t)+\S+/m,
      // Check for Python function definitions
      /def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(/,
      // Check for Python-style comments
      /^\s*#.*$/m,
      // Check for Python assignments with =
      /[a-zA-Z_][a-zA-Z0-9_]*\s*=/,
      // Check for Python-style string literals
      /(['"])(?:(?=(\\?))\2.)*?\1/,
      // Check for Python list or dictionary
      /[\[\{].*[\]\}]/
    ];
    
    // Count how many Python indicators are present
    const indicatorsFound = pythonIndicators.filter(regex => regex.test(text)).length;
    
    // If we found at least 2 indicators, it's likely Python code
    return indicatorsFound >= 2;
  };

  // New function to detect problematic code
  const detectProblematicCode = (code) => {
    const issues = [];
    
    problematicPatterns.forEach(item => {
      const matches = code.match(item.pattern);
      if (matches) {
        issues.push({
          category: item.category,
          description: item.description,
          matches: matches.map(match => match.trim()).filter(Boolean)
        });
      }
    });
    
    return issues;
  };

  const handleCodeChange = (event) => {
    const newCode = event.target.value;
    setCode(newCode);
    setValidationError(""); // Clear validation error when code changes
    
    // Reset guardrail-related states
    setCodeIssues([]);
    setShowGuardrailWarning(false);
    setBypassGuardrails(false);
  };

  const checkCodeGuardrails = () => {
    const issues = detectProblematicCode(code);
    setCodeIssues(issues);
    
    if (issues.length > 0) {
      setShowGuardrailWarning(true);
      return false;
    }
    
    return true;
  };

  const handleSubmit = async () => {
    if (!code.trim()) return;
    
    // Check if input is valid Python code
    if (!isPythonCode(code)) {
      setValidationError("The input doesn't appear to be valid Python code. Please check your code and try again.");
      setResponse(null);
      return;
    }
    
    setValidationError("");
    
    // Check guardrails if not bypassed
    if (!bypassGuardrails && !checkCodeGuardrails()) {
      return;
    }
    
    setIsLoading(true);
    try {
      let endpoint = "suggest_fix";
      let requestBody = { buggy_code: code };
      
      // Different endpoints based on mode
      if (mode === "explain") {
        endpoint = "explain_code";
        requestBody = { code: code };
      }
      
      const res = await fetch(`http://localhost:5000/${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
    
      const data = await res.json();
      setResponse(data.fixed_code || data.explanation || "No response from server");
      
      // Add to history
      const newHistoryItem = {
        id: Date.now(),
        code,
        response: data.fixed_code || data.explanation,
        mode,
        timestamp: new Date().toLocaleString(),
        hadGuardrailIssues: codeIssues.length > 0
      };
      
      setHistory(prevHistory => [newHistoryItem, ...prevHistory.slice(0, 9)]); // Keep last 10 items
      
    } catch (error) {
      console.error("Error submitting code:", error);
      setResponse("Failed to connect to the server. Please check if the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBypassGuardrails = () => {
    setBypassGuardrails(true);
    setShowGuardrailWarning(false);
  };

  const loadFromHistory = (item) => {
    setCode(item.code);
    setResponse(item.response);
    setMode(item.mode);
    setShowHistory(false);
  };

  const clearHistory = () => {
    setHistory([]);
    setShowHistory(false);
  };

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === "light" ? "dark" : "light");
  };

  return (
    <div className={`app-container ${theme}`}>
      <header className="app-header">
        <div className="logo-container">
          <h1 className="app-title">PyFixAI</h1>
          <span className="app-subtitle">Python Code Assistant</span>
        </div>
        <div className="header-controls">
          <button 
            className="theme-toggle" 
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>
          <button 
            className="history-button"
            onClick={() => setShowHistory(!showHistory)}
          >
            History
          </button>
        </div>
      </header>

      <main className="main-content">
        <section className="control-panel">
          <div className="mode-selector">
            <button 
              className={`mode-button ${mode === "fix" ? "active" : ""}`}
              onClick={() => setMode("fix")}
            >
              Fix Code
            </button>
            <button 
              className={`mode-button ${mode === "explain" ? "active" : ""}`}
              onClick={() => setMode("explain")}
            >
              Explain Code
            </button>
          </div>
        </section>

        <section className="code-section">
          <div className="code-editor-container">
            <div className="editor-header">
              <span>Input Python Code</span>
            </div>
            <textarea
              className="code-input"
              value={code}
              onChange={handleCodeChange}
              placeholder="Paste your Python code here..."
              rows="12"
              spellCheck="false"
            />
            {validationError && (
              <div className="validation-error">
                {validationError}
              </div>
            )}
          </div>

          <div className="action-container">
            <button
              className="submit-button"
              onClick={handleSubmit}
              disabled={isLoading || !code.trim()}
            >
              {isLoading ? "Processing..." : mode === "fix" ? "Debug Code" : "Explain Code"}
            </button>
          </div>

          {/* Guardrail Warning Modal */}
          {showGuardrailWarning && (
            <div className="guardrail-warning">
              <div className="guardrail-warning-content">
                <h3>⚠️ Code Protection Alert</h3>
                <p>We've detected potentially problematic code that may contain bias or discriminatory elements:</p>
                <ul>
                  {codeIssues.map((issue, index) => (
                    <li key={index}>
                      <strong>{issue.category}:</strong> {issue.description}
                      <ul>
                        {issue.matches.map((match, i) => (
                          <li key={i} className="code-match-item">"{match}"</li>
                        ))}
                      </ul>
                    </li>
                  ))}
                </ul>
                <p>We recommend reviewing your code to ensure it follows ethical coding practices.</p>
                <div className="guardrail-buttons">
                  <button 
                    className="cancel-button"
                    onClick={() => setShowGuardrailWarning(false)}
                  >
                    Cancel
                  </button>
                  <button 
                    className="bypass-button"
                    onClick={handleBypassGuardrails}
                  >
                    Proceed Anyway
                  </button>
                </div>
              </div>
            </div>
          )}

          {response && (
            <div className="results-container">
              <div className="editor-header">
                <span>
                  {mode === "fix" ? "Suggested Fix" : "Code Explanation"}
                </span>
                <button 
                  className="copy-button"
                  onClick={() => navigator.clipboard.writeText(response)}
                  title="Copy to clipboard"
                >
                  Copy
                </button>
              </div>
              <pre className="response-content">{response}</pre>
            </div>
          )}
        </section>
      </main>

      {showHistory && (
        <div className="history-panel">
          <div className="history-header">
            <h3>Code History</h3>
            <button 
              className="close-history" 
              onClick={() => setShowHistory(false)}
            >
              ✕
            </button>
          </div>
          {history.length > 0 ? (
            <>
              <ul className="history-list">
                {history.map(item => (
                  <li key={item.id} className="history-item" onClick={() => loadFromHistory(item)}>
                    <div className="history-item-header">
                      <span className="history-mode">{item.mode === "fix" ? "FIX" : "EXPLAIN"}</span>
                      {item.hadGuardrailIssues && <span className="history-guardrail-flag">⚠️</span>}
                      <span className="history-time">{item.timestamp}</span>
                    </div>
                    <div className="history-preview">
                      {item.code.substring(0, 100)}...
                    </div>
                  </li>
                ))}
              </ul>
              <button className="clear-history" onClick={clearHistory}>
                Clear History
              </button>
            </>
          ) : (
            <p className="no-history">No history yet</p>
          )}
        </div>
      )}

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} PyFixAI - Python Code Assistant</p>
      </footer>
    </div>
  );
}

export default App;