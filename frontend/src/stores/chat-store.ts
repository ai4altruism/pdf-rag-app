import { create } from "zustand";
import { devtools } from "zustand/middleware";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  /** ISO timestamp */
  createdAt: string;
  /** Source document chunks returned by the RAG pipeline */
  sources?: SourceChunk[];
}

export interface SourceChunk {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  score: number;
  text: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

// ---------------------------------------------------------------------------
// Store state & actions
// ---------------------------------------------------------------------------

interface ChatState {
  /** All loaded conversations keyed by ID */
  conversations: Record<string, Conversation>;
  /** The currently active conversation ID */
  activeConversationId: string | null;
  /** Whether a streaming response is in progress */
  isStreaming: boolean;

  // Actions
  setActiveConversation: (id: string | null) => void;
  addConversation: (conversation: Conversation) => void;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  updateLastMessage: (conversationId: string, delta: string) => void;
  setIsStreaming: (value: boolean) => void;
  clearConversations: () => void;
}

// ---------------------------------------------------------------------------
// Store implementation
// ---------------------------------------------------------------------------

export const useChatStore = create<ChatState>()(
  devtools(
    (set) => ({
      conversations: {},
      activeConversationId: null,
      isStreaming: false,

      setActiveConversation: (id) =>
        set({ activeConversationId: id }, false, "setActiveConversation"),

      addConversation: (conversation) =>
        set(
          (state) => ({
            conversations: {
              ...state.conversations,
              [conversation.id]: conversation,
            },
          }),
          false,
          "addConversation"
        ),

      addMessage: (conversationId, message) =>
        set(
          (state) => {
            const conversation = state.conversations[conversationId];
            if (!conversation) return state;
            return {
              conversations: {
                ...state.conversations,
                [conversationId]: {
                  ...conversation,
                  messages: [...conversation.messages, message],
                  updatedAt: new Date().toISOString(),
                },
              },
            };
          },
          false,
          "addMessage"
        ),

      /** Append streamed token delta to the last assistant message */
      updateLastMessage: (conversationId, delta) =>
        set(
          (state) => {
            const conversation = state.conversations[conversationId];
            if (!conversation) return state;
            const messages = [...conversation.messages];
            const last = messages[messages.length - 1];
            if (!last || last.role !== "assistant") return state;
            messages[messages.length - 1] = {
              ...last,
              content: last.content + delta,
            };
            return {
              conversations: {
                ...state.conversations,
                [conversationId]: { ...conversation, messages },
              },
            };
          },
          false,
          "updateLastMessage"
        ),

      setIsStreaming: (value) =>
        set({ isStreaming: value }, false, "setIsStreaming"),

      clearConversations: () =>
        set(
          { conversations: {}, activeConversationId: null },
          false,
          "clearConversations"
        ),
    }),
    { name: "ChatStore" }
  )
);
