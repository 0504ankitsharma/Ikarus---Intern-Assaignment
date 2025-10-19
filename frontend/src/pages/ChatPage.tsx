import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, RecommendedProduct } from '@/types';
import { chatRecommendations, health } from '@/lib/api';
import ProductCard from '@/components/ProductCard';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendOK, setBackendOK] = useState<boolean | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendedProduct[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  const quickAsks = [
    "Show me some affordable storage racks for my room.",
    "Recommend some modern dining chairs for my home.",
    "I need waterproof gardening mats or tools.",
    "Suggest decorative outdoor doormats for my patio.",
    "Show me foldable TV trays or compact furniture.",
    "Find me metal organizers or racks under $30.",
    "Recommend products made of iron or metal for home use.",
    "Show me items available in white or grey color.",
    "Suggest home and kitchen furniture within a reasonable price range.",
    "I’m looking for multi-purpose furniture for small apartments.",
    "Find eco-friendly or polyethylene material products.",
    "Show me top brands like GOYMFK, Subrtex, or MUYETOL.",
    "Recommend outdoor décor items for my garden.",
    "I want compact furniture for dining or living rooms.",
    "Find rubber-based doormats or floor covers.",
    "Show me trending furniture items from China manufacturers.",
    "Suggest space-saving storage items for organizing shoes and clothes.",
    "Find foldable or portable home accessories.",
    "Recommend stylish dining furniture for 2–4 people.",
    "Show me garden accessories under $10.",
  ];

  const [startIndex, setStartIndex] = useState(0);
  const visibleCount = 4;

  const visibleQuickAsks = quickAsks.slice(startIndex, startIndex + visibleCount);

  const next = () =>
    setStartIndex((prev) => Math.min(prev + visibleCount, quickAsks.length - visibleCount));
  const prev = () => setStartIndex((prev) => Math.max(prev - visibleCount, 0));

  useEffect(() => {
    health().then(
      () => setBackendOK(true),
      () => setBackendOK(false)
    );
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const send = async (customInput?: string) => {
    const query = (customInput ?? input).trim();
    if (query.length === 0 || loading) return;

    // Add user message with timestamp
    const userMessage: ChatMessage = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Send to API with conversation history (excluding the waiting message)
      const resp = await chatRecommendations(query, messages, 8);
      
      // Update messages with the response
      setMessages(resp.conversation_history);
      setRecommendations(resp.recommendations ?? []);
    } catch (e: any) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${e.message}`,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="chat">
        <div className="chat-header">
          <h2>Product Recommendations</h2>
          <span
            className={`status ${
              backendOK ? 'ok' : backendOK === false ? 'bad' : 'unknown'
            }`}
          >
            {backendOK === null
              ? 'Checking...'
              : backendOK
              ? 'API online'
              : 'API offline'}
          </span>
        </div>

        {/* Quick Ask Carousel */}
        <div className="quick-ask">
          <h4>Try asking:</h4>
          <div className="carousel">
            <button className="nav-btn" onClick={prev} disabled={startIndex === 0}>
              ‹
            </button>
            <div className="carousel-items">
              {visibleQuickAsks.map((q, i) => (
                <button
                  key={i}
                  className="quick-btn"
                  onClick={() => send(q)}
                  disabled={loading}
                >
                  {q}
                </button>
              ))}
            </div>
            <button
              className="nav-btn"
              onClick={next}
              disabled={startIndex + visibleCount >= quickAsks.length}
            >
              ›
            </button>
          </div>
        </div>

        {/* Chat Window */}
        <div className="chat-window">
          {messages.map((m, idx) => (
            <div key={idx} className={`bubble ${m.role}`}>
              <div className="role">
                {m.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="content">{m.content}</div>
            </div>
          ))}

          {/* Loading animation while waiting */}
          {loading && (
            <div className="bubble assistant loading-bubble">
              <div className="role">Assistant</div>
              <div className="dots">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>

        <div className="chat-input">
          <input
            placeholder="Describe what you need (e.g., modern wooden dining table under ₹3000)..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => (e.key === 'Enter' ? send() : undefined)}
          />
          <button onClick={() => send()} disabled={!canSend}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>

      <div className="section-header">
        <h3>Recommendations</h3>
        <span className="muted">{recommendations.length} items</span>
      </div>

      {recommendations.length === 0 && !loading ? (
        <div className="empty-state">
          No recommendations yet. Try asking for a style, material, or budget.
        </div>
      ) : (
        <div className="grid">
          {recommendations.map((rec) => (
            <ProductCard key={rec.product.uniq_id} item={rec} />
          ))}
        </div>
      )}
    </div>
  );
}
