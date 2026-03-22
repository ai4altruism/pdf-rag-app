"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { useState, type ReactNode } from "react";

interface QueryProviderProps {
  children: ReactNode;
}

/**
 * TanStack Query provider. Creates a stable QueryClient per browser session.
 * Must be a Client Component so it can hold the QueryClient instance in state.
 */
export function QueryProvider({ children }: QueryProviderProps) {
  // Use useState so each request gets its own QueryClient in SSR,
  // while the browser retains a single instance across renders.
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Cache results for 60 seconds before considering them stale
            staleTime: 60 * 1000,
            // Retry once on failure
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* DevTools only loads in development */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
