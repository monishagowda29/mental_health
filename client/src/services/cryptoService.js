/**
 * client/src/services/cryptoService.js
 * Client-side cryptographic helper utilizing the native Web Crypto API.
 * Implements PBKDF2 for key derivation and AES-GCM (256-bit) for zero-knowledge encryption/decryption.
 */

// Helper to convert string to array buffer
const strToBuffer = (str) => new TextEncoder().encode(str);

// Helper to convert array buffer to string
const bufferToStr = (buf) => new TextDecoder().decode(buf);

// Convert buffer to hex string for easy storage
const bufToHex = (buf) => {
  return Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
};

// Convert hex string to buffer
const hexToBuf = (hex) => {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes.buffer;
};

/**
 * Derives an AES-GCM key from a passphrase and a salt using PBKDF2.
 */
async function deriveKey(passphrase, saltBuffer) {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    strToBuffer(passphrase),
    { name: "PBKDF2" },
    false,
    ["deriveKey"]
  );

  return crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: saltBuffer,
      iterations: 100000,
      hash: "SHA-256"
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

/**
 * Encrypts cleartext using AES-GCM-256 with a derived key.
 * Returns a serialized hex string containing: salt (16 bytes) + IV (12 bytes) + ciphertext.
 * 
 * @param {string} text - Plain text to encrypt.
 * @param {string} passphrase - User passphrase.
 * @returns {Promise<string>} Hex-encoded cipher bundle.
 */
export async function encryptText(text, passphrase) {
  if (!text) return "";
  if (!passphrase) throw new Error("Encryption passphrase required.");

  try {
    // 1. Generate random 16-byte salt and 12-byte IV
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));

    // 2. Derive key using PBKDF2
    const key = await deriveKey(passphrase, salt);

    // 3. Encrypt data
    const ciphertext = await crypto.subtle.encrypt(
      {
        name: "AES-GCM",
        iv: iv
      },
      key,
      strToBuffer(text)
    );

    // 4. Serialize the bundle: salt (32 hex chars) + iv (24 hex chars) + ciphertext (hex)
    const saltHex = bufToHex(salt.buffer);
    const ivHex = bufToHex(iv.buffer);
    const cipherHex = bufToHex(ciphertext);

    return `${saltHex}:${ivHex}:${cipherHex}`;
  } catch (error) {
    console.error("Encryption failed:", error);
    throw new Error("Client-side encryption failed.");
  }
}

/**
 * Decrypts a cipher bundle hex string using a user passphrase.
 * 
 * @param {string} cipherBundle - Serialized hex string (salt:iv:ciphertext).
 * @param {string} passphrase - User passphrase.
 * @returns {Promise<string>} Decrypted plain text.
 */
export async function decryptText(cipherBundle, passphrase) {
  if (!cipherBundle) return "";
  if (!passphrase) throw new Error("Decryption passphrase required.");

  try {
    const parts = cipherBundle.split(":");
    if (parts.length !== 3) {
      throw new Error("Invalid cipher bundle format.");
    }

    const [saltHex, ivHex, cipherHex] = parts;
    const salt = hexToBuf(saltHex);
    const iv = hexToBuf(ivHex);
    const ciphertext = hexToBuf(cipherHex);

    // 1. Derive the key using same salt and passphrase
    const key = await deriveKey(passphrase, salt);

    // 2. Decrypt
    const decryptedBuffer = await crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: new Uint8Array(iv)
      },
      key,
      ciphertext
    );

    return bufferToStr(decryptedBuffer);
  } catch (error) {
    console.error("Decryption failed:", error);
    throw new Error("Decryption failed. Invalid passphrase or corrupted data.");
  }
}
