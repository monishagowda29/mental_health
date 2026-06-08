/**
 * client/src/store/useJournalStore.js
 * Zustand state store for local mental health screening histories.
 * Handles state transitions, LocalStorage isolation, optimistic UI updates, and backend polling.
 *
 * ID Generation: Uses the three-tier crash-safe generateId() helper instead of
 * raw crypto.randomUUID() calls, ensuring the store operates correctly in
 * headless CI/CD runners, JSDOM test environments, and older browsers.
 */
import { create } from 'zustand';
import { encryptText, decryptText } from '../services/cryptoService';
import { generateId } from '../utils/generateId';

const BACKEND_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useJournalStore = create((set, get) => ({
  patientId: 'Guest Patient',
  passphrase: '',
  journals: [], // Contains objects: { id, timestamp, encryptedText, decryptedText, prediction, status, scores: { depression, anxiety, normal } }
  loading: false,
  error: null,

  setPatientId: (patientId) => {
    const cleanedId = patientId.trim() || 'Guest Patient';
    set({ patientId: cleanedId, error: null });
    get().loadHistory();
  },

  setPassphrase: (passphrase) => {
    set({ passphrase });
    // Re-trigger history decryption with the new passphrase
    get().decryptAllJournals();
  },

  // Load encrypted history from local storage scoped by patientId
  loadHistory: () => {
    const { patientId } = get();
    const storageKey = `mindscan_history_${patientId}`;
    try {
      const stored = localStorage.getItem(storageKey);
      const journals = stored ? JSON.parse(stored) : [];
      set({ journals });
      // Attempt decryption
      get().decryptAllJournals();
    } catch (err) {
      console.error("Failed to load local journal logs:", err);
      set({ error: "Failed to load cached local logs.", journals: [] });
    }
  },

  // Saves the encrypted journal records list to scoped localStorage
  saveHistory: () => {
    const { patientId, journals } = get();
    const storageKey = `mindscan_history_${patientId}`;
    try {
      // Strip out the decrypted text before saving to disk to maintain absolute privacy
      const encryptedOnly = journals.map(({ decryptedText, ...rest }) => rest);
      localStorage.setItem(storageKey, JSON.stringify(encryptedOnly));
    } catch (err) {
      console.error("Failed to persist journal logs locally:", err);
      set({ error: "Failed to persist journal logs locally." });
    }
  },

  // Decrypts all loaded journal entries with the current passphrase
  decryptAllJournals: async () => {
    const { journals, passphrase } = get();
    if (!passphrase) {
      // Clear any decrypted text if passphrase is removed
      const cleared = journals.map(j => ({ ...j, decryptedText: undefined }));
      set({ journals: cleared });
      return;
    }

    const decrypted = await Promise.all(
      journals.map(async (j) => {
        try {
          const decryptedText = await decryptText(j.encryptedText, passphrase);
          return { ...j, decryptedText, decryptError: false };
        } catch (err) {
          return { ...j, decryptedText: "[Decryption Failed — Invalid Passphrase]", decryptError: true };
        }
      })
    );
    set({ journals: decrypted });
  },

  // Submits a new journal entry with Optimistic UI updates
  submitJournalEntry: async (rawText, languageHint = 'auto') => {
    const { patientId, passphrase, saveHistory } = get();
    if (!rawText.trim()) return;
    if (!passphrase) {
      set({ error: "Passphrase required to encrypt journal entries." });
      return;
    }

    // Three-tier crash-safe ID (window.crypto.randomUUID → getRandomValues → Math.random).
    // Safe for headless GitHub Actions runners, JSDOM test environments, and all browsers.
    const tempId = generateId();
    const timestamp = new Date().toISOString();

    // 1. Client-Side Encryption
    let encryptedText = '';
    try {
      encryptedText = await encryptText(rawText, passphrase);
    } catch (err) {
      set({ error: "Local encryption failed. Record not sent." });
      return;
    }

    // 2. Optimistic UI Update: append to the list immediately
    const optimisticEntry = {
      id: tempId,
      timestamp,
      encryptedText,
      decryptedText: rawText, // Keep raw in-memory only
      prediction: 'analyzing...',
      status: 'PENDING',
      scores: { depression: 0, anxiety: 0, normal: 0 }
    };

    set((state) => ({
      journals: [optimisticEntry, ...state.journals],
      error: null
    }));
    
    // Save state (decrypted text stripped automatically in saveHistory)
    saveHistory();

    // 3. Dispatch Background analysis Task
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          raw_text: rawText,
          language_hint: languageHint
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned code ${response.status}`);
      }

      const { task_id } = await response.json();
      
      // Start polling for this background task
      get().pollTaskStatus(tempId, task_id);
    } catch (err) {
      console.error("Failed to queue analysis task:", err);
      // Update optimistic entry to failure state
      set((state) => ({
        journals: state.journals.map((j) =>
          j.id === tempId
            ? { ...j, status: 'FAILURE', prediction: 'failed', error: "Failed to dispatch analysis job." }
            : j
        )
      }));
      saveHistory();
    }
  },

  // Polls task status from FastAPI backend until finished
  pollTaskStatus: async (entryTempId, taskId) => {
    const maxRetries = 30; // 30 seconds
    let retries = 0;

    const interval = setInterval(async () => {
      retries++;
      if (retries > maxRetries) {
        clearInterval(interval);
        set((state) => ({
          journals: state.journals.map((j) =>
            j.id === entryTempId
              ? { ...j, status: 'FAILURE', prediction: 'timeout', error: 'Analysis task timed out.' }
              : j
          )
        }));
        get().saveHistory();
        return;
      }

      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/tasks/${taskId}`);
        if (!res.ok) throw new Error("Broker poll request failed");
        
        const data = await res.json();
        
        if (data.status === 'SUCCESS') {
          clearInterval(interval);
          const result = data.result;
          
          set((state) => ({
            journals: state.journals.map((j) =>
              j.id === entryTempId
                ? {
                    ...j,
                    status: 'SUCCESS',
                    prediction: result.prediction,
                    scores: {
                      depression: result.depression_score,
                      anxiety: result.anxiety_score,
                      normal: result.normal_score
                    }
                  }
                : j
            )
          }));
          get().saveHistory();
        } else if (data.status === 'FAILURE') {
          clearInterval(interval);
          set((state) => ({
            journals: state.journals.map((j) =>
              j.id === entryTempId
                ? { ...j, status: 'FAILURE', prediction: 'failed', error: data.result?.error || 'Task execution failed' }
                : j
            )
          }));
          get().saveHistory();
        }
      } catch (err) {
        console.warn("Polling error (retrying):", err);
      }
    }, 1000);
  },

  // Securely erase all history for this patient
  clearHistory: () => {
    const { patientId } = get();
    const storageKey = `mindscan_history_${patientId}`;
    localStorage.removeItem(storageKey);
    set({ journals: [] });
  }
}));
