import { useEffect, useRef, useState } from 'react';
import {
  ArrowUp,
  BookOpen,
  Bot,
  BrainCircuit,
  ChevronDown,
  Code2,
  Copy,
  FileText,
  Menu,
  MoreHorizontal,
  PanelLeftClose,
  Plus,
  RotateCcw,
  Sparkles,
  UserRound,
  X,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || '/api';

const modes = [
  { label: 'Technical interview', icon: Code2, caption: 'Concepts, systems, trade-offs' },
  { label: 'DSA practice', icon: BrainCircuit, caption: 'Problems, hints, complexity' },
  { label: 'HR interview', icon: UserRound, caption: 'Clear, structured answers' },
  { label: 'Resume checker', icon: FileText, caption: 'Extract skills and projects' },
];

const starterPrompts = [
  'Explain binary search in simple English.',
  'Start a technical interview about hash maps.',
  'Give me a DSA problem about sliding windows.',
];

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Welcome to Interview Lab. Pick a mode, then send a question or start a practice round.',
      time: 'Now',
    },
  ]);
  const [message, setMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [activeMode, setActiveMode] = useState(modes[0].label);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const composerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const resumeInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  async function sendMessage(event) {
    event?.preventDefault();
    const trimmed = message.trim();
    if (!trimmed || isSending) return;

    setMessages((current) => [...current, { role: 'user', content: trimmed, time: 'Now' }]);
    setMessage('');
    setError('');
    setIsSending(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `[${activeMode}] ${trimmed}`,
          session_id: sessionId,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'The interview assistant is unavailable.');
      setSessionId(payload.session_id);
      setMessages((current) => [...current, { role: 'assistant', content: payload.response, time: 'Now' }]);
    } catch (requestError) {
      setError(requestError instanceof TypeError
        ? 'Cannot reach the backend. Start FastAPI on port 8000 and refresh this page.'
        : requestError.message);
    } finally {
      setIsSending(false);
    }
  }

  function startNewSession() {
    setSessionId(null);
    setError('');
    setMessages([
      {
        role: 'assistant',
        content: 'New practice session ready. What would you like to work on?',
        time: 'Now',
      },
    ]);
    setSidebarOpen(false);
    composerRef.current?.focus();
  }

  function useStarterPrompt(prompt) {
    setMessage(prompt);
    setSidebarOpen(false);
    composerRef.current?.focus();
  }

  async function analyzeResume(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setError('');
    setIsSending(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${API_URL}/resume/analyze`, { method: 'POST', body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Resume analysis failed.');
      const profile = payload.profile;
      const summary = [
        `Resume analyzed: ${payload.filename}`,
        profile.skills.length ? `Skills: ${profile.skills.join(', ')}` : 'Skills: none detected',
        profile.projects.length ? `Projects: ${profile.projects.join(' | ')}` : 'Projects: none detected',
        profile.experience.length ? `Experience: ${profile.experience.join(' | ')}` : 'Experience: none detected',
        profile.contact.emails.length ? `Email: ${profile.contact.emails.join(', ')}` : 'Email: none detected',
      ].join('\n');
      setActiveMode('Resume checker');
      setMessages((current) => [...current, { role: 'assistant', content: summary, time: 'Now' }]);
    } catch (requestError) {
      setError(requestError instanceof TypeError ? 'Cannot reach the backend. Start FastAPI on port 8000.' : requestError.message);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={17} strokeWidth={2.5} /></div>
          <div>
            <p className="brand-name">Interview Lab</p>
            <p className="brand-subtitle">Your private practice room</p>
          </div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation">
            <X size={18} />
          </button>
        </div>

        <button className="new-session" onClick={startNewSession}><Plus size={17} /> New session</button>
        <input ref={resumeInputRef} className="hidden-file-input" type="file" accept=".pdf,.docx,.txt" onChange={analyzeResume} />
        <button className="resume-upload" onClick={() => resumeInputRef.current?.click()}><FileText size={16} /> Analyze a resume</button>

        <section className="sidebar-section">
          <div className="section-label">Practice mode</div>
          <div className="mode-list">
            {modes.map(({ label, icon: Icon, caption }) => (
              <button
                className={`mode-item ${activeMode === label ? 'mode-active' : ''}`}
                key={label}
                onClick={() => setActiveMode(label)}
              >
                <span className="mode-icon"><Icon size={17} /></span>
                <span className="mode-copy"><strong>{label}</strong><small>{caption}</small></span>
                {activeMode === label && <span className="active-dot" />}
              </button>
            ))}
          </div>
        </section>

        <section className="sidebar-section starter-section">
          <div className="section-label">Try a prompt</div>
          {starterPrompts.map((prompt) => (
            <button className="starter-prompt" key={prompt} onClick={() => useStarterPrompt(prompt)}>
              <BookOpen size={14} /> <span>{prompt}</span>
            </button>
          ))}
        </section>

        <div className="sidebar-footer">
          <div className="local-status"><span className="status-dot" /> Local model ready</div>
          <p>Built for deliberate practice.<br />Your sessions stay in this workspace.</p>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="breadcrumb"><span>Practice</span><ChevronDown size={15} /><strong>{activeMode}</strong></div>
          <div className="topbar-actions">
            <span className="session-chip"><span className="status-dot" /> Session active</span>
            <button className="icon-button" aria-label="More options"><MoreHorizontal size={20} /></button>
          </div>
        </header>

        <div className="conversation-wrap">
          <div className="conversation-header">
            <div>
              <p className="eyebrow">{activeMode}</p>
              <h1>Let's sharpen your thinking.</h1>
              <p className="header-copy">Work through the reasoning. The assistant will challenge, clarify, and coach.</p>
            </div>
            <button className="outline-button" onClick={startNewSession}><RotateCcw size={15} /> Reset</button>
          </div>

          <div className="message-list" aria-live="polite">
            {messages.map((item, index) => <MessageBubble key={`${item.role}-${index}`} message={item} />)}
            {isSending && <div className="message-row assistant-row"><div className="avatar assistant-avatar"><Bot size={16} /></div><div className="typing-bubble"><span /><span /><span /></div></div>}
            {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}
            <div ref={messagesEndRef} />
          </div>

          <div className="composer-area">
            <form className="composer" onSubmit={sendMessage}>
              <textarea
                ref={composerRef}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(event); }
                }}
                placeholder="Ask a question or describe your approach..."
                rows="1"
                aria-label="Your interview response"
              />
              <button className="send-button" type="submit" disabled={!message.trim() || isSending} aria-label="Send message"><ArrowUp size={18} /></button>
            </form>
            <div className="composer-meta"><span>Shift + Enter for a new line</span><span>Custom model · Educational use</span></div>
          </div>
        </div>
      </main>
    </div>
  );
}

function MessageBubble({ message }) {
  const isAssistant = message.role === 'assistant';
  return (
    <div className={`message-row ${isAssistant ? 'assistant-row' : 'user-row'}`}>
      <div className={`avatar ${isAssistant ? 'assistant-avatar' : 'user-avatar'}`}>{isAssistant ? <Bot size={16} /> : <UserRound size={16} />}</div>
      <div className="message-content">
        <div className="message-meta"><strong>{isAssistant ? 'Interview Lab' : 'You'}</strong><span>{message.time}</span></div>
        <div className={`message-bubble ${isAssistant ? 'assistant-bubble' : 'user-bubble'}`}>{message.content}</div>
        {isAssistant && <button className="copy-button" onClick={() => navigator.clipboard?.writeText(message.content)}><Copy size={13} /> Copy</button>}
      </div>
    </div>
  );
}

export default App;
