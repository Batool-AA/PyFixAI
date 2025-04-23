import React, { useState, useEffect } from "react";
import "./App.css";

// Code protection patterns to check for problematic content
const safetyPatterns = [
  // Bias 
  {
    pattern: /\b(gender|race|ethnicity|religion|disability|nationality|caste|tribe)\b/i,
    category: "Bias or Discrimination",
    description: "Mentions of sensitive demographic categories that may indicate bias"
  },
  {
    pattern: /if\s*\((.*\b(gender|race|religion|ethnicity)\b.*?)\)/i,
    category: "Bias or Discrimination",
    description: "Conditional logic based on protected or sensitive attributes"
  },
  // Hate Content
  {
    pattern: /\b(hate|violence|kill|terror|abuse|slur|nazi|racist|sexist|homophobic|die)\b/i,
    category: "Hate or Inappropriate Content",
    description: "Terms indicating hate speech or inappropriate content"
  },
  {
    pattern: /\b(fuck|shit|bitch|asshole|bastard)\b/i,
    category: "Hate or Inappropriate Content",
    description: "Profanity detected in the code"
  },

  //Private / Sensitive Info
  {
    pattern: /['"]?(apikey|api_key|token|access_token)['"]?\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"]/i,
    category: "Private or Sensitive Info",
    description: "Possible hardcoded API key or token"
  },
  {
    pattern: /['"]?(password|passwd|secret)['"]?\s*[:=]\s*['"][^'"]{4,}['"]/i,
    category: "Private or Sensitive Info",
    description: "Potential hardcoded password or secret key"
  },
  {
    pattern: /[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/,
    category: "Private or Sensitive Info",
    description: "Email address detected"
  },
  {
    pattern: /-----BEGIN (RSA|EC|DSA)? PRIVATE KEY-----/,
    category: "Private or Sensitive Info",
    description: "Private key found in the code"
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
    if (!text || text.trim() === '') return false;

  const loosePythonIndicators = [
    // Keywords often found in code
    /\b(def|class|import|from|if|elif|else|for|while|try|except|return|with|as|pass|break|continue|lambda|print)\b/,
    // Possible function or class structure even if malformed
    /\b(def|class)\s+[a-zA-Z_][a-zA-Z0-9_]*\b/,
    // Use of common Python symbols (e.g., :, (), [], {}, =, etc.)
    /[:\[\]{}()=]/,
    // Use of indentation (even if inconsistent)
    /^\s{2,}\S+/m,
    // Comment-like lines
    /^\s*#.*$/m,
    // String literals, even if unmatched
    /['"]/,
    // Presence of known built-in functions or keywords
    /\b(print|len|range|open|input|int|str|list|dict)\b/,
  ];

  // Count how many indicators are present
  const indicatorsFound = loosePythonIndicators.filter(regex => regex.test(text)).length;
  return indicatorsFound >= 2;  };

// function to check valid python code
const isValidPythonCode = (text) => {
  if (!text || text.trim() === '') return false;

  const strictIndicators = [
    /\b(def|class|import|from|if|elif|else|for|while|try|except|with|return|yield|lambda|print)\b/,
    /^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\):/m,
    /^class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(\(.*\))?:/m,
    /^( {4}|\t)+\S+/m,
    /^\s*(if|elif|else|for|while|try|except|with)\s+.*:\s*$/m,
    /^\s*#.*$/m,
    /\b(print|len|range|input|str|int|float|list|dict|set)\s*\(/,
  ];

  const indicatorsFound = strictIndicators.filter(regex => regex.test(text)).length;
  if (/^\s*print\s*\(.*\)\s*$/.test(text)) return true;
  return indicatorsFound >= 3;
};



 // New function to detect problematic code
  const detectProblematicCode = (code) => {
    const issues = [];
    
    safetyPatterns.forEach(item => {
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

    const isSafe = checkCodeGuardrails();
    if (!isSafe) {
      // setValidationError("The code contains unsafe content. Please correct it before proceeding.");
      return; // Halt here if unsafe
    }

    if (mode === "fix")
    {
      if (!isPythonCode(code)) {
        setValidationError("The input doesn't appear to be valid Python code. Please check your code and try again.");
        setResponse(null);
        return;
      }
      
      setValidationError("");
    }
    else if (mode === "explain"){
      if (!isValidPythonCode(code)) {
        setValidationError("The input doesn't appear to be valid Python code. Please check your code and try again.");
        setResponse(null);
        return;
      }
      
      setValidationError("");
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

          {/* Content Safety */}
          {showGuardrailWarning && (
          <div className="warning">
          <strong>⚠️ Warning:</strong> The code contains unsafe or sensitive content. Please remove or modify it before continuing.
          </div>
          )}

          {response && (
            <div className="results-container">
              <div className="warning-message">
                <span>
                  {mode === "fix" ? "Suggested Fix" : "Explanation"}
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