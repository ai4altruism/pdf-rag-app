/**
 * WebSocket client for streaming query responses.
 *
 * Handles connection lifecycle, automatic reconnection,
 * and typed message parsing.
 */

import type { QueryMetadata, QueryOptions, SourceResponse, WSMessage } from "./types";

export type WSEventHandler = {
  onToken?: (content: string) => void;
  onSources?: (sources: SourceResponse[]) => void;
  onMetadata?: (metadata: QueryMetadata) => void;
  onDone?: () => void;
  onError?: (error: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
};

const DEFAULT_WS_URL = "ws://localhost:8000/ws/query";
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY = 1000;

export class QueryWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WSEventHandler;
  private reconnectAttempts = 0;
  private shouldReconnect = true;

  constructor(handlers: WSEventHandler, url?: string) {
    this.handlers = handlers;
    this.url = url ?? DEFAULT_WS_URL;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.handlers.onConnect?.();
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch {
        this.handlers.onError?.("Failed to parse WebSocket message");
      }
    };

    this.ws.onclose = () => {
      this.handlers.onDisconnect?.();
      if (this.shouldReconnect) {
        this.attemptReconnect();
      }
    };

    this.ws.onerror = () => {
      this.handlers.onError?.("WebSocket connection error");
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
    this.ws?.close();
    this.ws = null;
  }

  sendQuery(
    query: string,
    collectionIds: string[],
    options?: QueryOptions,
  ): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      this.handlers.onError?.("WebSocket not connected");
      return;
    }

    this.ws.send(
      JSON.stringify({
        query,
        collection_ids: collectionIds,
        options,
      }),
    );
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private handleMessage(msg: WSMessage): void {
    switch (msg.type) {
      case "token":
        this.handlers.onToken?.(msg.content as string);
        break;
      case "sources":
        this.handlers.onSources?.(msg.content as SourceResponse[]);
        break;
      case "metadata":
        this.handlers.onMetadata?.(msg.content as QueryMetadata);
        break;
      case "done":
        this.handlers.onDone?.();
        break;
      case "error":
        this.handlers.onError?.(msg.content as string);
        break;
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.handlers.onError?.(
        `Failed to reconnect after ${MAX_RECONNECT_ATTEMPTS} attempts`,
      );
      return;
    }

    const delay = BASE_RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    setTimeout(() => {
      this.connect();
    }, delay);
  }
}
