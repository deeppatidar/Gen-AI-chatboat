"use client";

/* eslint-disable react-hooks/set-state-in-effect -- load document list on mount */

import { useCallback, useEffect, useRef, useState } from "react";

type Source = {
  documentId: string;
  documentName: string;
  pageNumber?: number | null;
  chunkId: string;
  relevanceScore: number;
  excerpt: string;
};

type DocumentItem = {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: string;
  status: string;
  chunkCount: number;
  error?: string | null;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  error?: boolean;
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ChatApp() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [input, setInput] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openSource, setOpenSource] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const indexed = documents.filter((doc) => doc.status === "indexed");

  const loadDocuments = useCallback(async () => {
    try {
      const response = await fetch("/api/documents");
      const data = (await response.json()) as {
        documents?: DocumentItem[];
        error?: string;
      };
      if (!response.ok) {
        setError(data.error || "Unable to load documents.");
        return;
      }
      setDocuments(data.documents ?? []);
    } catch {
      setError("Unable to load documents. Is the Python backend running?");
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleUpload(file: File) {
    setUploading(true);
    setNotice(null);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as {
        message?: string;
        error?: string;
        name?: string;
        chunkCount?: number;
      };
      if (response.status === 409) {
        setNotice(data.message || "This document has already been indexed.");
      } else if (!response.ok) {
        setError(data.error || "Unable to process the document.");
      } else {
        setNotice(`Indexed ${data.name ?? file.name} (${data.chunkCount ?? 0} chunks).`);
      }
      await loadDocuments();
    } catch {
      setError("Unable to process the document.");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id: string) {
    const response = await fetch(`/api/documents/${id}`, { method: "DELETE" });
    if (!response.ok) {
      const data = (await response.json()) as { error?: string };
      setError(data.error || "Unable to delete the document.");
      return;
    }
    await loadDocuments();
  }

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    if (indexed.length === 0) {
      setError("Upload a PDF or TXT document before asking a question.");
      return;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    const next = [...messages, userMessage];
    setMessages(next);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history: messages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
      });
      const data = (await response.json()) as {
        answer?: string;
        sources?: Source[];
        error?: string;
      };
      setMessages([
        ...next,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            data.answer ||
            data.error ||
            "Unable to generate a response right now. Please try again.",
          sources: data.sources,
          error: !response.ok,
        },
      ]);
    } catch {
      setMessages([
        ...next,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Unable to generate a response right now. Please try again.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100 text-slate-900">
      <div className={`${sidebarOpen ? "fixed inset-0 z-20 flex" : "hidden"} md:static md:flex`}>
        <div className="absolute inset-0 bg-slate-900/40 md:hidden" onClick={() => setSidebarOpen(false)} />
        <aside className="relative z-30 flex h-full w-80 max-w-[85vw] flex-col border-r border-slate-200 bg-slate-50">
          <div className="border-b border-slate-200 px-5 py-5">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-600">
              Trusted knowledge
            </p>
            <h1 className="mt-1 text-xl font-semibold">RAG Assistant</h1>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              {indexed.length > 0
                ? `Answering from ${indexed.length} uploaded document${indexed.length === 1 ? "" : "s"}.`
                : "Upload a PDF or TXT file first. There is no built-in document."}
            </p>
          </div>
          <div className="flex items-center justify-between px-5 py-4">
            <h2 className="text-sm font-semibold">Documents</h2>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-60"
            >
              {uploading ? "Processing…" : "+ Upload"}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.txt,application/pdf,text/plain"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleUpload(file);
                event.target.value = "";
              }}
            />
          </div>
          {notice ? (
            <p className="mx-5 mb-3 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">{notice}</p>
          ) : null}
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-5">
            {documents.length === 0 ? (
              <p className="px-2 py-6 text-sm leading-6 text-slate-500">
                Upload your own PDF or TXT. On upload the file is extracted, chunked,
                embedded, and stored. Questions only search those stored vectors.
              </p>
            ) : (
              <ul className="space-y-2">
                {documents.map((doc) => (
                  <li key={doc.id} className="rounded-lg border border-slate-200 bg-white p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">{doc.name}</p>
                        <p className="mt-1 text-xs text-slate-500">
                          {doc.type.toUpperCase()} · {formatSize(doc.size)}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => void handleDelete(doc.id)}
                        className="text-xs font-medium text-slate-500 hover:text-red-600"
                      >
                        Delete
                      </button>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs">
                      <span className={doc.status === "indexed" ? "font-medium text-emerald-700" : "font-medium text-red-700"}>
                        {doc.status === "indexed" ? "✓ Indexed" : doc.status}
                      </span>
                      <span className="text-slate-500">
                        {doc.status === "indexed" ? `${doc.chunkCount} chunks` : doc.error || ""}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </div>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3">
          <button
            type="button"
            className="rounded-md border border-slate-200 px-2 py-1 text-sm md:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            Documents
          </button>
          <div>
            <p className="text-sm font-semibold">
              {indexed.length > 0 ? "Document Q&A" : "Upload a document to begin"}
            </p>
            <p className="text-xs text-slate-500">
              Chunking and embeddings run at upload time. Questions retrieve stored chunks.
            </p>
          </div>
        </header>
        {error ? (
          <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>
        ) : null}

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8">
          {messages.length === 0 && !loading ? (
            <div className="flex h-full items-center justify-center">
              <p className="max-w-md text-center text-sm leading-6 text-slate-600">
                Upload your PDF or TXT on the left, wait until it is indexed, then ask a
                question. Answers come only from your uploaded files.
              </p>
            </div>
          ) : null}
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-6 ${
                  message.role === "user"
                    ? "bg-indigo-600 text-white"
                    : message.error
                      ? "bg-red-50 text-red-800"
                      : "bg-white text-slate-800 shadow-sm ring-1 ring-slate-200"
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.sources && message.sources.length > 0 ? (
                  <div className="mt-3 border-t border-slate-200 pt-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p>
                    <ul className="mt-2 space-y-2">
                      {message.sources.map((source, index) => (
                        <li key={source.chunkId}>
                          <button
                            type="button"
                            className="text-left text-sm text-indigo-700 hover:underline"
                            onClick={() =>
                              setOpenSource(openSource === source.chunkId ? null : source.chunkId)
                            }
                          >
                            [{index + 1}] {source.documentName}
                            {source.pageNumber ? ` — Page ${source.pageNumber}` : ""}
                          </button>
                          {openSource === source.chunkId ? (
                            <div className="mt-2 rounded-md bg-slate-50 p-3 text-xs leading-5 text-slate-600">
                              <p><span className="font-semibold">Document:</span> {source.documentName}</p>
                              {source.pageNumber ? (
                                <p><span className="font-semibold">Page:</span> {source.pageNumber}</p>
                              ) : null}
                              <p className="mt-1">
                                <span className="font-semibold">Relevant excerpt:</span> “{source.excerpt}”
                              </p>
                            </div>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
          {loading ? (
            <div className="flex justify-start">
              <div className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-500 shadow-sm ring-1 ring-slate-200">
                Thinking…
              </div>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>

        <form
          className="border-t border-slate-200 bg-white p-4"
          onSubmit={(event) => {
            event.preventDefault();
            void handleSend();
          }}
        >
          <div className="mx-auto flex max-w-3xl gap-3">
            <textarea
              value={input}
              rows={1}
              placeholder={indexed.length ? "Ask a question about your documents..." : "Upload a document first..."}
              className="min-h-12 flex-1 resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none ring-indigo-500 focus:ring-2"
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || indexed.length === 0}
              className="rounded-xl bg-indigo-600 px-4 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
