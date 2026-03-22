export default function DocumentsPage() {
  return (
    <div className="flex h-full flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload and manage the PDF documents indexed in your knowledge base.
        </p>
      </div>

      {/* Document list and upload UI will be built out in Sprint 6 */}
      <div className="flex-1 rounded-lg border border-dashed border-border p-10 text-center">
        <p className="text-xs text-muted-foreground">
          Document management — coming soon
        </p>
      </div>
    </div>
  );
}
