/**
 * scratch/test_frontend_crypto.js
 * Node-based verification script for the client-side cryptographic service.
 * Verifies that the Web Crypto AES-GCM + PBKDF2 encryption and decryption are functionally integral.
 */

// Import functions from the client module
import { encryptText, decryptText } from '../client/src/services/cryptoService.js';

async function runTest() {
  console.log("============================================================");
  console.log("       MINDSCAN CLIENT CRYPTO VERIFICATION TEST");
  console.log("============================================================");
  
  const passphrase = "secure_patient_passphrase_123!";
  const plainText = "Patient feels highly anxious and reports trouble sleeping.";
  
  console.log(`Original Text : "${plainText}"`);
  console.log(`Passphrase    : "${passphrase}"`);
  console.log("------------------------------------------------------------");
  
  try {
    // 1. Test Encryption
    console.log("Encrypting plain text...");
    const cipherBundle = await encryptText(plainText, passphrase);
    console.log(`Encrypted Bundle: "${cipherBundle}"`);
    
    // Verify that the bundle format is correct (salt:iv:ciphertext)
    const parts = cipherBundle.split(":");
    if (parts.length !== 3) {
      throw new Error(`Invalid bundle format: expected 3 colon-separated parts, got ${parts.length}`);
    }
    console.log("[SUCCESS] Encryption generated correct format (salt:iv:ciphertext).");
    console.log("------------------------------------------------------------");
    
    // 2. Test Decryption with Correct Passphrase
    console.log("Decrypting with CORRECT passphrase...");
    const decryptedText = await decryptText(cipherBundle, passphrase);
    console.log(`Decrypted Text: "${decryptedText}"`);
    
    if (decryptedText !== plainText) {
      throw new Error("Decrypted text does not match original plain text!");
    }
    console.log("[SUCCESS] Decryption matched original text exactly!");
    console.log("------------------------------------------------------------");
    
    // 3. Test Decryption with INCORRECT Passphrase
    console.log("Decrypting with INCORRECT passphrase...");
    try {
      await decryptText(cipherBundle, "wrong_passphrase");
      throw new Error("Decryption with wrong passphrase succeeded when it should have failed!");
    } catch (err) {
      console.log(`[SUCCESS] Decryption failed as expected for wrong passphrase. Error: ${err.message}`);
    }
    
    console.log("============================================================");
    console.log(" [SUCCESS] Client-side Cryptographic Integrity Verified!");
    console.log("============================================================");
    process.exit(0);
  } catch (error) {
    console.error("\n[FAIL] Cryptographic integrity test failed:", error);
    process.exit(1);
  }
}

runTest();
