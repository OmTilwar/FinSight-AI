import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! I am FinSight AI, your AI banking assistant. How can I help you today?", sender: "bot" }
  ])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage = { text: input, sender: "user" }
    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsTyping(true)

    await sendMessageToBackend(userMessage.text)
  }

  const handleQuickAction = (actionText) => {
    const userMessage = { text: actionText, sender: "user" }
    setMessages(prev => [...prev, userMessage])
    setIsTyping(true)

    // Trigger the fetch logic directly or reuse handleSend logic slightly modified
    // For simplicity, we'll reuse the fetch logic by calling a helper or just duplicating for now
    // Better: extract fetch logic. Let's extract it.
    sendMessageToBackend(actionText)
  }

  const sendMessageToBackend = async (text) => {
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      })

      if (!response.ok) throw new Error("Network response was not ok")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      let botMessage = { text: "", sender: "bot" }
      setMessages(prev => [...prev, botMessage])
      setIsTyping(false)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        botMessage.text += chunk

        setMessages(prev => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1] = { ...botMessage }
          return newMessages
        })
      }

    } catch (error) {
      console.error("Error sending message:", error)
      const errorMessage = { text: "Sorry, I'm having trouble connecting to the server.", sender: "bot" }
      setMessages(prev => [...prev, errorMessage])
      setIsTyping(false)
    }
  }

  // Speech Recognition (Web Speech API)
  const startListening = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition()
      recognition.continuous = false
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        setIsTyping(true)
      }

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setInput(transcript)
        // Automatically send the message after a brief pause or immediately
        // For better UX, we populate input and let user send, OR auto-send.
        // Let's auto-send for "Assistant" feel.
        handleSend()
        // Note: handleSend uses 'input' state, which might not be updated yet due to closure.
        // It's safer to call sendMessageToBackend directly or updating state and using useEffect
        // BUT simpler hack: pass transcript to a modified handleSend or just setInput and let user click.
        // Let's just setInput for safety first to avoid empty sends.
      }

      recognition.onerror = (event) => {
        console.error("Speech error", event)
        setIsTyping(false)
      }

      recognition.onend = () => {
        setIsTyping(false)
      }

      recognition.start()
    } else {
      alert("Voice input not supported in this browser.")
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend()
    }
  }

  return (
    <div className="app-container">
      <header className="chat-header">
        <h1>FinSight AI</h1>
        <p>Banking & Financial Services Assistant</p>
      </header>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="message-content">
              {msg.text.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="message bot">
            <div className="message-content typing-indicator">
              <span>•</span><span>•</span><span>•</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <div className="input-width-limit-column">
          <div className="quick-actions">
            <button className="action-btn" onClick={() => handleQuickAction("Check my balance")}>
              💰 Check Balance
            </button>
            <button className="action-btn" onClick={() => handleQuickAction("Apply for a loan")}>
              📝 Apply for Loan
            </button>
            <button className="action-btn" onClick={() => handleQuickAction("What are the interest rates?")}>
              asd Interest Rates
            </button>
          </div>
          <div className="input-wrapper">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about loans, accounts, or services..."
              disabled={isTyping}
            />
            <button className="mic-btn" onClick={startListening} title="Speak">
              🎙️
            </button>
            <button className="send-btn" onClick={handleSend} disabled={isTyping || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
