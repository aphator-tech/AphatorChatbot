document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chatBox');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.querySelector('.typing-indicator');
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Add initial bot greeting
    setTimeout(() => {
        addBotMessage("Hello! Welcome to Aphator Tech. How can I assist you with our crypto and tech services today?");
    }, 500);
    
    // Send message when button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Also send message when Enter key is pressed
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Theme toggle functionality
    themeToggle.addEventListener('click', function() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-bs-theme');
        
        // Toggle theme
        if (currentTheme === 'dark') {
            html.setAttribute('data-bs-theme', 'light');
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        } else {
            html.setAttribute('data-bs-theme', 'dark');
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
        }
        
        // Save preference to localStorage
        localStorage.setItem('theme', html.getAttribute('data-bs-theme'));
    });
    
    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-bs-theme', savedTheme);
        if (savedTheme === 'light') {
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addUserMessage(message);
        
        // Clear input field
        messageInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response after a slight delay to simulate typing
            setTimeout(() => {
                if (data.error) {
                    addBotMessage("Sorry, I encountered an error. Please try again.");
                } else {
                    addBotMessage(data.response);
                }
            }, 500);
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addBotMessage("Sorry, I'm having trouble connecting to the server. Please try again later.");
        });
    }
    
    function addUserMessage(message) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const messageElement = createMessageElement('user-message', message, time);
        chatBox.appendChild(messageElement);
        scrollToBottom();
    }
    
    function addBotMessage(message) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const messageElement = createMessageElement('bot-message', message, time);
        chatBox.appendChild(messageElement);
        scrollToBottom();
    }
    
    function createMessageElement(className, message, time) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${className}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = time;
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeSpan);
        
        return messageDiv;
    }
    
    function showTypingIndicator() {
        typingIndicator.classList.add('active');
    }
    
    function hideTypingIndicator() {
        typingIndicator.classList.remove('active');
    }
    
    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
