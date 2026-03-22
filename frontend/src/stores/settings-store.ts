import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type EmbeddingProvider = "openai" | "sentence-transformers";
export type LLMProvider = "openai" | "anthropic" | "local";
export type Theme = "light" | "dark" | "system";

export interface SettingsState {
  // API connection
  backendUrl: string;

  // Model preferences
  embeddingProvider: EmbeddingProvider;
  embeddingModel: string;
  llmProvider: LLMProvider;
  llmModel: string;

  // RAG pipeline settings
  topK: number;
  scoreThreshold: number;
  maxTokens: number;

  // UI preferences
  theme: Theme;
  streamResponses: boolean;

  // Actions
  setBackendUrl: (url: string) => void;
  setEmbeddingProvider: (provider: EmbeddingProvider) => void;
  setEmbeddingModel: (model: string) => void;
  setLLMProvider: (provider: LLMProvider) => void;
  setLLMModel: (model: string) => void;
  setTopK: (k: number) => void;
  setScoreThreshold: (threshold: number) => void;
  setMaxTokens: (tokens: number) => void;
  setTheme: (theme: Theme) => void;
  setStreamResponses: (value: boolean) => void;
  resetToDefaults: () => void;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_SETTINGS = {
  backendUrl: "http://localhost:8000",
  embeddingProvider: "openai" as EmbeddingProvider,
  embeddingModel: "text-embedding-3-small",
  llmProvider: "openai" as LLMProvider,
  llmModel: "gpt-4o-mini",
  topK: 5,
  scoreThreshold: 0.7,
  maxTokens: 2048,
  theme: "system" as Theme,
  streamResponses: true,
};

// ---------------------------------------------------------------------------
// Store implementation
// ---------------------------------------------------------------------------

export const useSettingsStore = create<SettingsState>()(
  devtools(
    persist(
      (set) => ({
        ...DEFAULT_SETTINGS,

        setBackendUrl: (url) =>
          set({ backendUrl: url }, false, "setBackendUrl"),

        setEmbeddingProvider: (provider) =>
          set({ embeddingProvider: provider }, false, "setEmbeddingProvider"),

        setEmbeddingModel: (model) =>
          set({ embeddingModel: model }, false, "setEmbeddingModel"),

        setLLMProvider: (provider) =>
          set({ llmProvider: provider }, false, "setLLMProvider"),

        setLLMModel: (model) =>
          set({ llmModel: model }, false, "setLLMModel"),

        setTopK: (k) => set({ topK: k }, false, "setTopK"),

        setScoreThreshold: (threshold) =>
          set({ scoreThreshold: threshold }, false, "setScoreThreshold"),

        setMaxTokens: (tokens) =>
          set({ maxTokens: tokens }, false, "setMaxTokens"),

        setTheme: (theme) => set({ theme }, false, "setTheme"),

        setStreamResponses: (value) =>
          set({ streamResponses: value }, false, "setStreamResponses"),

        resetToDefaults: () =>
          set({ ...DEFAULT_SETTINGS }, false, "resetToDefaults"),
      }),
      {
        name: "pdf-rag-settings",
        // Only persist non-sensitive preferences; API keys should never be stored
        partialize: (state) => ({
          backendUrl: state.backendUrl,
          embeddingProvider: state.embeddingProvider,
          embeddingModel: state.embeddingModel,
          llmProvider: state.llmProvider,
          llmModel: state.llmModel,
          topK: state.topK,
          scoreThreshold: state.scoreThreshold,
          maxTokens: state.maxTokens,
          theme: state.theme,
          streamResponses: state.streamResponses,
        }),
      }
    ),
    { name: "SettingsStore" }
  )
);
