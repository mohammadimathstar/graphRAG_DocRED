let currentTraceId = null;
let currentThumbsUp = null; // Tracks if they clicked up or down

function handleKeyPress(event) {
    if (event.key === "Enter") askQuestion();
}

async function askQuestion() {
    const question = document.getElementById("questionInput").value;
    if (!question) return;

    // Update UI
    document.getElementById("chatBox").innerHTML += `<div class="message user-msg">${question}</div>`;
    document.getElementById("questionInput").value = "";
    document.getElementById("askBtn").disabled = true;
    document.getElementById("feedbackBox").style.display = "none";
    
    // Add loading message
    document.getElementById("chatBox").innerHTML += `<div class="message bot-msg" id="loadingMsg">Searching knowledge graph...</div>`;
    document.getElementById("chatBox").scrollTop = document.getElementById("chatBox").scrollHeight;

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        
        // Remove loading message
        document.getElementById("loadingMsg").remove();
        
        // Add answer
        document.getElementById("chatBox").innerHTML += `<div class="message bot-msg">${data.answer}</div>`;
        document.getElementById("chatBox").scrollTop = document.getElementById("chatBox").scrollHeight;
        
        // Save trace ID and show feedback box
        currentTraceId = data.trace_id;
        currentThumbsUp = null; // Reset thumbs up state
        
        if(currentTraceId) {
            // Reset button styles
            document.getElementById("thumbsUpBtn").style.opacity = "0.5";
            document.getElementById("thumbsDownBtn").style.opacity = "0.5";
            document.getElementById("feedbackText").value = "";
            
            // Show the feedback box
            document.getElementById("feedbackBox").style.display = "block";
        }
    } catch (error) {
        document.getElementById("loadingMsg").innerText = "Error: Could not connect to server.";
    }
    
    document.getElementById("askBtn").disabled = false;
}

function setThumbsUp(isThumbsUp) {
    currentThumbsUp = isThumbsUp;
    
    // Highlight the clicked button and dim the other
    document.getElementById("thumbsUpBtn").style.opacity = isThumbsUp ? "1.0" : "0.5";
    document.getElementById("thumbsDownBtn").style.opacity = isThumbsUp ? "0.5" : "1.0";
}

async function submitCombinedFeedback() {
    if (!currentTraceId) return;
    
    // If they didn't click a thumb, default to false (or you can choose to block submission)
    if (currentThumbsUp === null) {
        alert("Please select 👍 or 👎 before submitting.");
        return;
    }

    const text = document.getElementById("feedbackText").value;
    
    try {
        await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                trace_id: currentTraceId, 
                thumbs_up: currentThumbsUp, 
                user_feedback: text 
            })
        });
        
        // Hide the box after submission
        document.getElementById("feedbackBox").style.display = "none";
        
        // Optional: Show a small thank you message
        alert("Thank you for your feedback!");
    } catch (e) {
        console.error("Failed to send feedback");
    }
}
