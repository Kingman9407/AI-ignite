'use client';
import { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface PatientInfo {
  age: number;
  gender: string;
}

type ServerStatus = 'checking' | 'connected' | 'disconnected' | 'error';

export default function ClinicalChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [patientInfo, setPatientInfo] = useState<PatientInfo | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>('checking');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkServerAndFetchInfo();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkServerAndFetchInfo = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(`${API_BASE}/api/patient-info`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setPatientInfo(data.patient_info);
          setServerStatus('connected');
        }
      } else {
        setServerStatus('error');
      }
    } catch (err) {
      console.error('Server check failed:', err);
      setServerStatus('disconnected');
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: userText,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userText }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();

      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.reply || 'Processed.',
          timestamp: new Date(),
        },
      ]);

      setServerStatus('connected');
    } catch (err) {
      console.error('Chat error:', err);

      const msg =
        err instanceof Error && err.name === 'AbortError'
          ? 'Request timed out. Server may still be processing.'
          : 'Server unavailable. Please start the backend.';

      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: msg,
          timestamp: new Date(),
        },
      ]);

      setServerStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getStatusColor = () => {
    switch (serverStatus) {
      case 'connected':
        return 'bg-green-500';
      case 'checking':
        return 'bg-yellow-500';
      case 'disconnected':
        return 'bg-red-500';
      case 'error':
        return 'bg-orange-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (serverStatus) {
      case 'connected':
        return 'Connected';
      case 'checking':
        return 'Checking...';
      case 'disconnected':
        return 'Disconnected';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-gray-800">
            Clinical Documentation
          </h1>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
            <span className="text-xs text-gray-500">{getStatusText()}</span>
          </div>
        </div>

        {patientInfo && (
          <div className="text-sm text-gray-600">
            {patientInfo.gender}, {patientInfo.age} yrs
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-10">
            <div className="text-lg font-medium mb-2">
              🏥 Clinical Documentation Assistant
            </div>
            <div className="text-sm">
              Try: <span className="italic">Patient has headache after breakfast</span>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xl px-4 py-3 rounded-xl text-sm ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-800 border'
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">
                {msg.content}
              </pre>
              <div className="text-xs opacity-60 mt-1">
                {msg.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-xl px-4 py-2 text-sm text-gray-500">
              Processing…
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t p-4 flex justify-center">
        <div className="flex gap-3 w-full max-w-4xl items-center">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            rows={2}
            disabled={loading}
            placeholder="Describe symptom or medication…"
            className="
              flex-1 resize-none border rounded-xl
              px-4 py-3 text-sm leading-relaxed
              focus:outline-none focus:ring-2 focus:ring-blue-500
              disabled:bg-gray-100
            "
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="
              px-6 py-3 bg-blue-600 text-white rounded-xl font-medium
              hover:bg-blue-700 transition-colors
              disabled:bg-gray-400
            "
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
