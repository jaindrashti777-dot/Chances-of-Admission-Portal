'use client';

import React, { useState } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './ChatWidget.module.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

export function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hi. I can explain how this synthetic model responds to fields like CGPA, backlogs, internships, and research papers.',
      sender: 'bot',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputValue.trim()) return;

    const userMsg: Message = { id: Date.now().toString(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.text }),
      });

      if (!response.ok) {
        throw new Error('Advisor request failed');
      }

      const data: { response?: string } = await response.json();

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response || 'The advisor did not return a response.',
        sender: 'bot',
      };
      setMessages(prev => [...prev, botMsg]);
    } catch {
      const errorMsg: Message = { id: (Date.now() + 1).toString(), text: "I couldn't connect to the server. Please try again later.", sender: 'bot' };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button className={styles.toggleButton} onClick={toggleChat} aria-label="Toggle Chat">
        <MessageCircle size={24} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={styles.chatWindow}
          >
            <div className={styles.header}>
              <div className={styles.headerInfo}>
                <div className={styles.botIconWrapper}>
                  <Bot size={18} />
                </div>
                <span className={styles.headerTitle}>Model Advisor</span>
              </div>
              <button className={styles.closeButton} onClick={toggleChat}>
                <X size={18} />
              </button>
            </div>

            <div className={styles.messagesContainer}>
              {messages.map((msg) => (
                <div key={msg.id} className={`${styles.messageWrapper} ${styles[msg.sender]}`}>
                  {msg.sender === 'bot' && (
                    <div className={styles.messageAvatar}>
                      <Bot size={14} />
                    </div>
                  )}
                  <div className={`${styles.messageBubble} ${styles[msg.sender + 'Bubble']}`}>
                    {msg.text}
                  </div>
                  {msg.sender === 'user' && (
                    <div className={styles.messageAvatar}>
                      <User size={14} />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className={`${styles.messageWrapper} ${styles.bot}`}>
                  <div className={styles.messageAvatar}>
                    <Bot size={14} />
                  </div>
                  <div className={`${styles.messageBubble} ${styles.botBubble} ${styles.typing}`}>
                    <span className={styles.dot}></span>
                    <span className={styles.dot}></span>
                    <span className={styles.dot}></span>
                  </div>
                </div>
              )}
            </div>

            <div className={styles.disclaimerText}>
              Rule-based helper for this demo model. Not admissions advice.
            </div>

            <form onSubmit={handleSend} className={styles.inputArea}>
              <input
                type="text"
                placeholder="Ask about CGPA, backlogs, or internships"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                className={styles.input}
              />
              <button type="submit" className={styles.sendButton} disabled={!inputValue.trim() || isLoading}>
                <Send size={18} />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
