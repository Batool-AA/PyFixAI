import React, { useState } from "react";
import "./App.css";

function App() {
  const [code, setCode] = useState("");
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleCodeChange = (event) => {
    setCode(event.target.value);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
    //   const res = await fetch("http://localhost:5000/predict", {
    //     method: "POST",
    //     headers: {
    //       "Content-Type": "application/json",
    //     },
    //     body: JSON.stringify({ code: code }),
    //   });
      
    //   const data = await res.json();
    //   console.log(data);
    //   setResponse(data.bug_type);
      const res = await fetch("http://localhost:5000/suggest_fix", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ buggy_code: code }),
      });
    
      const data = await res.json();
      // console.log(data);
      setResponse(data.fixed_code);
    } catch (error) {
      console.error("Error submitting code:", error);
      setResponse({ errors: [{ message: "Failed to connect to the server." }] });
    } finally {
      setIsLoading(false);
    }
    
  };

  return (
    <div className="app-container">
      <h1 className="app-title">PyFixAI</h1>

      <textarea
        className="code-input"
        value={code}
        onChange={handleCodeChange}
        placeholder="Paste your Python code here..."
        rows="12"
      />

      <button
        className="submit-button"
        onClick={handleSubmit}
        disabled={isLoading}
      >
        {isLoading ? "Analyzing..." : "Debug Code"}
      </button>

      {response && (
        <div className="results-container">
          <h2 className="results-title">Suggested Fix</h2>
          <p className="no-errors">{response}</p>
          {/* {response.errors.length === 0 ? (
            <p className="no-errors">✅ No issues found!</p>
          ) : (
            <ul className="error-list">
              {response.errors.map((err, index) => (
                <li key={index} className="error-item">
                  {err.message}
                </li>
              ))}
            </ul>
          )} */}
        </div>
      )}
    </div>
  );
}

export default App;
