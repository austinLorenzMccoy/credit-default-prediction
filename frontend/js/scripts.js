const apiBaseUrl = "http://127.0.0.1:8000"; // Update with your API's base URL

// Default Prediction Form
document.getElementById("defaultPredictionForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  
  // Collect features in the order expected by the model
  const features = [
    Number(document.getElementById("limitBal").value),
    Number(document.getElementById("sex").value),
    Number(document.getElementById("education").value),
    Number(document.getElementById("marriage").value),
    Number(document.getElementById("age").value),
    Number(document.getElementById("billAmtAvg").value),
    Number(document.getElementById("payAmtAvg").value),
    Number(document.getElementById("paymentStatus").value)
  ];

  try {
    const response = await fetch(`${apiBaseUrl}/predict/default`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const data = await response.json();
    
    // Interpret the prediction
    const predictionText = data.prediction > 0.5 
      ? "High Risk of Default (Probability: " + (data.prediction * 100).toFixed(2) + "%)"
      : "Low Risk of Default (Probability: " + (data.prediction * 100).toFixed(2) + "%)";
    
    document.getElementById("defaultResult").innerHTML = `
      <div class="alert ${data.prediction > 0.5 ? 'alert-danger' : 'alert-success'}">
        ${predictionText}
      </div>
    `;
  } catch (error) {
    document.getElementById("defaultResult").innerHTML = `
      <div class="alert alert-danger">Error: ${error.message}</div>
    `;
  }
});

// Credit Limit Prediction Form
document.getElementById("creditLimitForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  
  // Collect features in the order expected by the model
  const features = [
    Number(document.getElementById("climitBal").value),
    Number(document.getElementById("csex").value),
    Number(document.getElementById("ceducation").value),
    Number(document.getElementById("cmarriage").value),
    Number(document.getElementById("cage").value),
    Number(document.getElementById("cbillAmtAvg").value),
    Number(document.getElementById("cpayAmtAvg").value),
    Number(document.getElementById("cpaymentStatus").value)
  ];

  try {
    const response = await fetch(`${apiBaseUrl}/predict/credit-limit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const data = await response.json();
    
    document.getElementById("creditLimitResult").innerHTML = `
      <div class="alert alert-success">
        Predicted Credit Limit: $${data.predicted_credit_limit.toFixed(2)}
      </div>
    `;
  } catch (error) {
    document.getElementById("creditLimitResult").innerHTML = `
      <div class="alert alert-danger">Error: ${error.message}</div>
    `;
  }
});