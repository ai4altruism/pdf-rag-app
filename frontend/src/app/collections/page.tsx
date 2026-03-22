export default function CollectionsPage() {
  return (
    <div className="flex h-full flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Collections</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Organise documents into named collections to scope your queries.
        </p>
      </div>

      {/* Collections UI will be built out in Sprint 6 */}
      <div className="flex-1 rounded-lg border border-dashed border-border p-10 text-center">
        <p className="text-xs text-muted-foreground">
          Collections management — coming soon
        </p>
      </div>
    </div>
  );
}
