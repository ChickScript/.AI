function sendQuestion() {
    const question = document.getElementById('userQuestion').value.trim();
    const chatbox = document.getElementById('chatbox');
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = `You: ${question}`;
    
    // Check if the question is empty before proceeding
    if (!question) {
        alert("Please enter a question.");
        return;
    }

    document.getElementById('userQuestion').value = ''; 

    // Display the user's question in the chatbox
    chatbox.appendChild(userMessage);

    // Show a loading indicator for the bot's response
    const loadingMessage = document.createElement('div');
    loadingMessage.className = 'bot-message';
    loadingMessage.textContent = "Bot: Thinking...";
    chatbox.appendChild(loadingMessage);
    chatbox.scrollTop = chatbox.scrollHeight;

    fetch('http://127.0.0.1:5000/api/get_mental_health_support', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        // Remove the loading indicator
        loadingMessage.remove();

        // Display the bot's response in the chatbox
        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message';
        botMessage.textContent = `Bot: ${data.answer}`;
        chatbox.appendChild(botMessage);
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        loadingMessage.remove();

        // Show an error message in the chatbox
        const errorMessage = document.createElement('div');
        errorMessage.className = 'bot-message error';
        errorMessage.textContent = "Bot: Sorry, something went wrong. Please try again later.";
        chatbox.appendChild(errorMessage);
        chatbox.scrollTop = chatbox.scrollHeight;
    });
}
