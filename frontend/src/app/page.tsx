export default function ChatPage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
      <div className="max-w-md text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Chat</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Ask questions about your documents. The RAG pipeline will retrieve
          relevant context and generate an answer.
        </p>
      </div>

      {/* Chat UI will be built out in Sprint 6 */}
      <div className="w-full max-w-2xl rounded-lg border border-dashed border-border p-10 text-center">
        <p className="text-xs text-muted-foreground">
          Chat interface — coming soon
        </p>
      </div>
    </div>
  );
}
