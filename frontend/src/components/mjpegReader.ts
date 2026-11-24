/**
 * [view modes] Reads the server's multipart MJPEG (overlay=client) with fetch, so the browser gets each frame AND the
 * boxes of that same frame (an X-Lifeguard-Boxes part header) and draws the boxes itself: one stream serves every
 * view mode, and switching mode never reconnects. Aborting the signal ends the request (the server stops encoding).
 */

const enc = new TextEncoder();
const dec = new TextDecoder();
const CRLF2 = enc.encode("\r\n\r\n");

function indexOf(buf: Uint8Array, pat: Uint8Array, from = 0): number {
  outer: for (let i = from; i <= buf.length - pat.length; i++) {
    for (let j = 0; j < pat.length; j++) if (buf[i + j] !== pat[j]) continue outer;
    return i;
  }
  return -1;
}

export interface Part {
  jpeg: Uint8Array;
  headers: Record<string, string>;
}

/** Pull every complete part out of `buf`; returns the parts and the unread rest. Pure (tested). */
export function takeParts(buf: Uint8Array): { parts: Part[]; rest: Uint8Array } {
  const parts: Part[] = [];
  let pos = 0;
  for (;;) {
    const end = indexOf(buf, CRLF2, pos);
    if (end < 0) break;
    const headers: Record<string, string> = {};
    for (const line of dec.decode(buf.subarray(pos, end)).split("\r\n")) {
      const k = line.indexOf(":");
      if (k > 0) headers[line.slice(0, k).trim().toLowerCase()] = line.slice(k + 1).trim();
    }
    const len = Number(headers["content-length"]);
    const start = end + 4;
    if (!Number.isFinite(len) || len <= 0) {
      pos = start; // a boundary line without a body: skip it
      continue;
    }
    if (buf.length < start + len) break;
    parts.push({ jpeg: buf.slice(start, start + len), headers });
    pos = start + len;
  }
  return { parts, rest: buf.slice(pos) };
}

export async function readMjpeg(url: string, signal: AbortSignal, onPart: (p: Part) => void): Promise<void> {
  const r = await fetch(url, { signal, credentials: "same-origin", cache: "no-store" });
  if (!r.ok || !r.body) throw new Error(`stream ${String(r.status)}`);
  const reader = r.body.getReader();
  let buf: Uint8Array = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    const next = new Uint8Array(buf.length + value.length);
    next.set(buf);
    next.set(value, buf.length);
    const { parts, rest } = takeParts(next);
    buf = rest;
    const last = parts[parts.length - 1]; // only the newest frame matters: never fall behind live
    if (last) onPart(last);
  }
}
