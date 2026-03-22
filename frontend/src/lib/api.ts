/**
 * Typed API client for the PDF RAG backend.
 *
 * All endpoints are prefixed with /api — the Next.js dev server proxies
 * these to the FastAPI backend via next.config.ts rewrites.
 */

import type {
  CollectionCreate,
  CollectionResponse,
  CollectionUpdate,
  ConversationCreate,
  ConversationResponse,
  DocumentResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  ReingestRequest,
  SettingsResponse,
} from "./types";

const BASE_URL = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body?: unknown,
  ) {
    super(`API Error ${status}: ${statusText}`);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, res.statusText, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// --- Health ---
export function getHealth(): Promise<HealthResponse> {
  return request("/health");
}

// --- Collections ---
export function listCollections(): Promise<CollectionResponse[]> {
  return request("/collections");
}

export function getCollection(id: string): Promise<CollectionResponse> {
  return request(`/collections/${id}`);
}

export function createCollection(data: CollectionCreate): Promise<CollectionResponse> {
  return request("/collections", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateCollection(
  id: string,
  data: CollectionUpdate,
): Promise<CollectionResponse> {
  return request(`/collections/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export function deleteCollection(id: string): Promise<void> {
  return request(`/collections/${id}`, { method: "DELETE" });
}

// --- Documents ---
export function listDocuments(collectionId?: string): Promise<DocumentResponse[]> {
  const query = collectionId ? `?collection_id=${collectionId}` : "";
  return request(`/documents${query}`);
}

export function getDocument(id: string): Promise<DocumentResponse> {
  return request(`/documents/${id}`);
}

export async function uploadDocument(
  collectionId: string,
  file: File,
): Promise<DocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection_id", collectionId);

  const res = await fetch(`${BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
    // Don't set Content-Type — browser sets it with boundary
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, res.statusText, body);
  }
  return res.json();
}

export function reingestDocument(
  id: string,
  data?: ReingestRequest,
): Promise<DocumentResponse> {
  return request(`/documents/${id}/reingest`, {
    method: "POST",
    body: JSON.stringify(data ?? {}),
  });
}

export function deleteDocument(id: string): Promise<void> {
  return request(`/documents/${id}`, { method: "DELETE" });
}

// --- Query ---
export function submitQuery(data: QueryRequest): Promise<QueryResponse> {
  return request("/query", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// --- Conversations ---
export function listConversations(): Promise<ConversationResponse[]> {
  return request("/conversations");
}

export function getConversation(id: string): Promise<ConversationResponse> {
  return request(`/conversations/${id}`);
}

export function createConversation(
  data: ConversationCreate,
): Promise<ConversationResponse> {
  return request("/conversations", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function deleteConversation(id: string): Promise<void> {
  return request(`/conversations/${id}`, { method: "DELETE" });
}

// --- Settings ---
export function getSettings(): Promise<SettingsResponse> {
  return request("/settings");
}

export { ApiError };
