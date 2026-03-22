export default function SettingsPage() {
  return (
    <div className="flex h-full flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure the backend URL, embedding provider, LLM model, and
          retrieval parameters.
        </p>
      </div>

      {/* Settings form will be built out in Sprint 6 */}
      <div className="flex-1 rounded-lg border border-dashed border-border p-10 text-center">
        <p className="text-xs text-muted-foreground">
          Settings form — coming soon
        </p>
      </div>
    </div>
  );
}
