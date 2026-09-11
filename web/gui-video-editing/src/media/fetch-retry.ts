/// One retry policy for every URL read the media layer makes.
///
/// mediabunny's default `getRetryDelay` retries indefinitely *except* when it
/// suspects CORS — `fetch()` rejected, the browser is online, and the URL's
/// origin differs from the page's — where it gives up on the first failure,
/// since an opaque CORS rejection can never succeed on a retry. For this editor
/// that heuristic is backwards: the video is **normally** cross-origin
/// (`?video=<url>` names a media server, and `serve-documents.ts` answers on its
/// own port), so one transient rejection ended the load with
/// `TypeError: Failed to fetch` and an empty timeline that only a manual reload
/// cleared. Our own `urlByteSource` had the same shape: it threw on the first
/// rejection, asserting a missing `Access-Control-Allow-Origin` that was not
/// actually the cause.
///
/// Retrying a bounded number of times restores the distinction the heuristic
/// throws away: a blip recovers by itself, and a genuine CORS
/// misconfiguration still surfaces — under two seconds later instead of
/// instantly. The policy lives here, not in each reader, so the mediabunny path
/// and the `moov` byte source cannot drift apart on it (#180).

/// Retries after the first failure, then the first backoff step. The delays are
/// `0.25s, 0.5s, 1s`, so a load that never recovers reports after ~1.75s.
const MAX_FETCH_RETRIES = 3;
const FIRST_RETRY_SECONDS = 0.25;

/// The shape mediabunny's `UrlSourceOptions.getRetryDelay` expects: seconds to
/// wait, or `null` once the retries are spent so a real failure still reports.
export function retryDelaySeconds(previousAttempts: number): number | null {
  return previousAttempts > MAX_FETCH_RETRIES
    ? null
    : FIRST_RETRY_SECONDS * 2 ** (previousAttempts - 1);
}

/// `fetch` with that same policy, for the reads mediabunny does not make.
///
/// Only a **rejection** is retried — the browser refusing to make the request.
/// An answered request is the server's verdict and is returned as-is, however
/// unwelcome its status: retrying a `404` or a `416` cannot change it.
export async function fetchWithRetry(url: string, init?: RequestInit): Promise<Response> {
  for (let attempts = 1; ; attempts++) {
    try {
      return await fetch(url, init);
    } catch (error) {
      const delay = retryDelaySeconds(attempts);
      if (delay === null) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, delay * 1000));
    }
  }
}
