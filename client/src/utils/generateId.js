/**
 * client/src/utils/generateId.js
 *
 * Three-tier, crash-safe unique identifier generator.
 *
 * This module must be used for ALL optimistic UI entry IDs, task tracking
 * handles, and ephemeral state keys throughout the MindScan frontend.
 * It guarantees that headless CI/CD automation runners (GitHub Actions,
 * Jest / Vitest in a Node.js JSDOM context) never encounter a fatal
 * runtime crash caused by a missing or restricted `crypto` global.
 *
 * ─── Tier Resolution Order ───────────────────────────────────────────────
 *
 *  TIER 1 — window.crypto.randomUUID()
 *    Standard, cryptographically secure UUID v4 via the Web Crypto API.
 *    Available in: Chrome 92+, Firefox 95+, Safari 15.4+, Node 19+,
 *                  and all modern production browser targets.
 *
 *  TIER 2 — window.crypto.getRandomValues() + bitwise UUID assembly
 *    Assembles a RFC-4122-compliant UUID v4 string by filling a 16-byte
 *    Uint8Array with high-entropy randomness and mapping it manually.
 *    Available in: Chrome 11+, Firefox 26+, IE 11+, Safari 7+, Node 15+.
 *    Fallback for environments where randomUUID exists on the crypto object
 *    but is blocked by security policy (e.g., non-secure origin contexts).
 *
 *  TIER 3 — Math.random() pseudo-random string
 *    Generates a 36-character pseudo-UUID-shaped string using Math.random().
 *    Not cryptographically secure. Intended exclusively as a final safety
 *    net for headless JSDOM test runners, Node environments without the
 *    Crypto module, and legacy CI automation contexts.
 *    Collision probability across a single test execution session is
 *    negligible given the ~5e-13 per-call chance.
 *
 * ─────────────────────────────────────────────────────────────────────────
 */

/**
 * Tier 2 implementation: assembles a UUID v4 string using
 * crypto.getRandomValues() for each byte.
 *
 * Template: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
 * The "4" pins version, and the "y" nibble is masked to the variant bits 10xx.
 *
 * @returns {string} RFC-4122 UUID v4 string
 */
function uuidFromGetRandomValues() {
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);

  // Set version bits: version 4 → 0100xxxx
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  // Set variant bits: variant 1 → 10xxxxxx
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  // Convert to hex string and insert hyphens at RFC-4122 positions
  const hex = Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');

  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join('-');
}

/**
 * Tier 3 implementation: pseudo-random UUID-shaped string built from
 * Math.random(). Safe for headless JSDOM and Node test runners.
 * Not cryptographically secure — for ephemeral UI IDs only.
 *
 * @returns {string} UUID-shaped string (not RFC-4122 guaranteed)
 */
function uuidFromMathRandom() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * generateId()
 *
 * Resolves and returns a unique identifier string using the highest-entropy
 * method available in the current JavaScript runtime environment.
 *
 * Resolution priority:
 *   Tier 1 → Tier 2 → Tier 3
 *
 * This function is synchronous, throws no exceptions, and always returns
 * a non-empty string. It is safe to call in any browser, Node.js, or
 * headless JSDOM test runner environment.
 *
 * @returns {string} Unique identifier string
 */
export function generateId() {
  try {
    // ── Tier 1: window.crypto.randomUUID() ──────────────────────────────
    if (
      typeof window !== 'undefined' &&
      window.crypto != null &&
      typeof window.crypto.randomUUID === 'function'
    ) {
      return window.crypto.randomUUID();
    }

    // ── Tier 2: window.crypto.getRandomValues() ──────────────────────────
    if (
      typeof window !== 'undefined' &&
      window.crypto != null &&
      typeof window.crypto.getRandomValues === 'function'
    ) {
      return uuidFromGetRandomValues();
    }

    // ── Tier 2b: Node.js / non-browser crypto module ─────────────────────
    // Handles Node 19+ environments (e.g., Vitest running in Node mode)
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.crypto != null &&
      typeof globalThis.crypto.randomUUID === 'function'
    ) {
      return globalThis.crypto.randomUUID();
    }

    // ── Tier 3: Math.random() pseudo-random fallback ─────────────────────
    // Final safety net for headless JSDOM runners and legacy CI contexts.
    return uuidFromMathRandom();

  } catch (_err) {
    // Absolute last resort: if every tier above throws for any reason
    // (e.g., security policy violation), fall back to Math.random().
    return uuidFromMathRandom();
  }
}
