import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import toast from 'react-hot-toast';

interface WebSocketContextType {
  isConnected: boolean;
  sendMessage: (message: any) => void;
  subscribe: (topic: string, callback: (data: any) => void) => void;
  unsubscribe: (topic: string) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const subscribersRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      setIsConnected(true);
      toast.success('Connected to server');
    };

    ws.onclose = () => {
      setIsConnected(false);
      toast.error('Disconnected from server');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('WebSocket connection error');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const topic = message.type || 'default';
        const callbacks = subscribersRef.current.get(topic);
        if (callbacks) {
          callbacks.forEach(callback => callback(message));
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    setSocket(ws);

    // Ping-pong to keep connection alive
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      ws.close();
    };
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      toast.error('WebSocket is not connected');
    }
  }, [socket]);

  const subscribe = useCallback((topic: string, callback: (data: any) => void) => {
    if (!subscribersRef.current.has(topic)) {
      subscribersRef.current.set(topic, new Set());
    }
    subscribersRef.current.get(topic)!.add(callback);
  }, []);

  const unsubscribe = useCallback((topic: string) => {
    subscribersRef.current.delete(topic);
  }, []);

  return (
    <WebSocketContext.Provider value={{ isConnected, sendMessage, subscribe, unsubscribe }}>
      {children}
    </WebSocketContext.Provider>
  );
};