import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useJournalStore } from './store/useJournalStore';
import {
  Brain, Shield, Lock, Upload, Activity,
  Sparkles, Trash2, Globe, RefreshCw,
  Download, User, X, AlertTriangle, Info,
  TrendingUp, TrendingDown, Minus, RotateCcw,
  Menu, CheckCircle2
} from 'lucide-react';

const BACKEND_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const MAX_CHARS = 2500;

// ─────────────────────────────────────────────────────────────────────────────
// Toast Notification System
// ─────────────────────────────────────────────────────────────────────────────
function ToastContainer({ toasts, removeToast }) {
  return (
    <div className="fixed top-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none max-w-xs w-full">
      {toasts.map(t => (
        <div
          key={t.id}
          className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl border shadow-2xl text-sm animate-slide-in
            ${t.type === 'error'   ? 'bg-rose-950/95 border-rose-500/40 text-rose-200 backdrop-blur-md' :
              t.type === 'success' ? 'bg-emerald-950/95 border-emerald-500/40 text-emerald-200 backdrop-blur-md' :
                                     'bg-slate-900/95 border-white/10 text-gray-200 backdrop-blur-md'}`}
        >
          {t.type === 'error'   && <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />}
          {t.type === 'success' && <CheckCircle2  className="h-4 w-4 shrink-0 mt-0.5 text-emerald-400" />}
          {t.type === 'info'    && <Info          className="h-4 w-4 shrink-0 mt-0.5 text-indigo-400" />}
          <span className="flex-1 text-xs leading-relaxed">{t.message}</span>
          <button onClick={() => removeToast(t.id)} className="shrink-0 opacity-50 hover:opacity-100 transition-opacity">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Passphrase Strength Meter
// ─────────────────────────────────────────────────────────────────────────────
function getPassphraseStrength(pass) {
  if (!pass) return { level: 0, label: '', color: '' };
  let score = 0;
  if (pass.length >= 8) score++;
  if (pass.length >= 12) score++;
  if (/[A-Z]/.test(pass) && /[a-z]/.test(pass)) score++;
  if (/[0-9]/.test(pass)) score++;
  if (/[^A-Za-z0-9]/.test(pass)) score++;
  if (score <= 1) return { level: 1, label: 'Weak',      color: 'bg-rose-500',    text: 'text-rose-400' };
  if (score <= 2) return { level: 2, label: 'Fair',      color: 'bg-amber-500',   text: 'text-amber-400' };
  if (score <= 3) return { level: 3, label: 'Strong',    color: 'bg-blue-500',    text: 'text-blue-400' };
  return            { level: 4, label: 'Excellent', color: 'bg-emerald-500', text: 'text-emerald-400' };
}

// ─────────────────────────────────────────────────────────────────────────────
// Elapsed Timer Hook for PENDING journal entries
// ─────────────────────────────────────────────────────────────────────────────
function useElapsedSeconds(timestamp, active) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) { setElapsed(0); return; }
    const start = new Date(timestamp).getTime();
    const tick  = () => setElapsed(Math.floor((Date.now() - start) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [timestamp, active]);
  return elapsed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Journal Entry Card  (extracted to use the elapsed-timer hook per card)
// ─────────────────────────────────────────────────────────────────────────────
function JournalCard({ j }) {
  const elapsed = useElapsedSeconds(j.timestamp, j.status === 'PENDING');

  return (
    <div className="glass-panel p-5 rounded-2xl space-y-3 transition-all duration-200 hover:border-white/10">
      {/* Header row */}
      <div className="flex justify-between items-start gap-2">
        <div className="text-[10px] text-gray-500 font-mono shrink-0">
          {new Date(j.timestamp).toLocaleString()}
        </div>
        <div>
          {j.status === 'PENDING' && (
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-indigo-400 bg-indigo-950/40 border border-indigo-500/20 px-2 py-0.5 rounded-md animate-pulse">
              <RefreshCw className="h-2.5 w-2.5 animate-spin" />
              <span>Analyzing… {elapsed}s</span>
            </span>
          )}
          {j.status === 'FAILURE' && (
            <span className="text-[10px] font-bold text-rose-400 bg-rose-950/40 border border-rose-500/20 px-2 py-0.5 rounded-md">
              Failed
            </span>
          )}
          {j.status === 'SUCCESS' && (
            <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-md ${
              j.prediction === 'normal'   ? 'text-emerald-400 bg-emerald-950/40 border border-emerald-500/20' :
              j.prediction === 'anxiety'  ? 'text-amber-400  bg-amber-950/40  border border-amber-500/20'  :
                                            'text-rose-400   bg-rose-950/40   border border-rose-500/20'
            }`}>
              {j.prediction === 'normal' ? 'No significant risk detected' : `${j.prediction} patterns detected`}
            </span>
          )}
        </div>
      </div>

      {/* Decrypted text or locked state */}
      <p className="text-gray-300 text-xs leading-relaxed font-light">
        {j.decryptedText ? j.decryptedText : (
          <span className="text-gray-500 italic">🔒 Zero-Knowledge Encrypted (passphrase required to view)</span>
        )}
      </p>

      {/* Score breakdown – animated progress bar pills */}
      {j.status === 'SUCCESS' && j.scores && (
        <div className="pt-2 border-t border-white/5 space-y-1.5">
          {[
            { key: 'normal',     label: 'Wellness',   color: 'bg-emerald-500', val: j.scores.normal },
            { key: 'anxiety',    label: 'Anxiety',    color: 'bg-amber-500',   val: j.scores.anxiety },
            { key: 'depression', label: 'Depression', color: 'bg-rose-500',    val: j.scores.depression },
          ].map(({ key, label, color, val }) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-[9px] text-gray-500 w-16 shrink-0">{label}</span>
              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${color}`}
                  style={{ width: `${(val * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="text-[9px] text-gray-400 font-mono w-8 text-right shrink-0">
                {(val * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {j.error && <p className="text-rose-400 text-[10px] font-mono">{j.error}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend Arrow Component
// ─────────────────────────────────────────────────────────────────────────────
function TrendIcon({ current, prev, invertGood }) {
  if (prev == null) return null;
  const diff = current - prev;
  if (Math.abs(diff) < 5) return <Minus className="h-3 w-3 text-gray-500" />;
  const improving = invertGood ? diff < 0 : diff > 0;
  return improving
    ? <TrendingUp   className="h-3 w-3 text-emerald-400" />
    : <TrendingDown className="h-3 w-3 text-rose-400" />;
}

// ─────────────────────────────────────────────────────────────────────────────
// Screener Progress Bar
// ─────────────────────────────────────────────────────────────────────────────
function ScreenerProgress({ answered, total }) {
  const pct = Math.round((answered / total) * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-[10px] text-gray-500">
        <span>{answered} of {total} questions answered</span>
        <span>{pct}% complete</span>
      </div>
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const { patientId, passphrase, journals, setPatientId, setPassphrase, submitJournalEntry, clearHistory } = useJournalStore();

  // ── Navigation ────────────────────────────────────────────────────────────
  const [activeTab,   setActiveTab]   = useState('journal');
  const [tabKey,      setTabKey]      = useState('journal');       // forces re-mount for fade animation
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => { setTabKey(activeTab); }, [activeTab]);

  // ── Toast System ──────────────────────────────────────────────────────────
  const [toasts, setToasts] = useState([]);
  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4500);
  }, []);
  const removeToast = useCallback(id => setToasts(prev => prev.filter(t => t.id !== id)), []);

  // ── Journal Tab State ─────────────────────────────────────────────────────
  const [journalInput,  setJournalInput]  = useState('');
  const [langHint,      setLangHint]      = useState('auto');

  // ── Modals ────────────────────────────────────────────────────────────────
  const [showCrisisBanner,  setShowCrisisBanner]  = useState(true);
  const [showWipeConfirm,   setShowWipeConfirm]   = useState(false);

  // ── Scan Tab State ────────────────────────────────────────────────────────
  const [scanMode,       setScanMode]       = useState('general');
  const [scanFile,       setScanFile]       = useState(null);
  const [scanPreview,    setScanPreview]    = useState(null);
  const [scanTaskStatus, setScanTaskStatus] = useState(null);
  const [scanResult,     setScanResult]     = useState('');
  const [scanLoading,    setScanLoading]    = useState(false);

  // ── Screener State  (-1 = unanswered so 0-score is distinguishable) ───────
  const [activeScreener,    setActiveScreener]    = useState('phq9');
  const [phq9Answers,       setPhq9Answers]       = useState(Array(9).fill(-1));
  const [gad7Answers,       setGad7Answers]       = useState(Array(7).fill(-1));
  const [screenerSubmitted, setScreenerSubmitted] = useState({ phq9: false, gad7: false });

  // ── SVG Tooltip ───────────────────────────────────────────────────────────
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // ── Derived Values ────────────────────────────────────────────────────────
  const phq9Score    = phq9Answers.map(a => Math.max(0, a)).reduce((a, b) => a + b, 0);
  const gad7Score    = gad7Answers.map(a => Math.max(0, a)).reduce((a, b) => a + b, 0);
  const phq9Answered = phq9Answers.filter(a => a >= 0).length;
  const gad7Answered = gad7Answers.filter(a => a >= 0).length;

  const passphraseStrength = useMemo(() => getPassphraseStrength(passphrase), [passphrase]);

  // ── Crisis Trigger ────────────────────────────────────────────────────────
  const triggerCrisisOverlay = useMemo(() => {
    const keywords = ['suicide', 'self-harm', 'kill myself', 'die', 'end my life', 'cutting', 'hanging'];
    if (keywords.some(k => journalInput.toLowerCase().includes(k))) return true;
    if (journals.length > 0) {
      const latest = journals[0];
      if (latest.status === 'SUCCESS' && latest.prediction === 'depression' && latest.scores?.depression >= 0.5) return true;
    }
    if (phq9Answers[8] > 0) return true;
    if (screenerSubmitted.phq9 && phq9Score >= 10) return true;
    return false;
  }, [journalInput, journals, phq9Answers, phq9Score, screenerSubmitted.phq9]);

  // ── Questionnaire Definitions ──────────────────────────────────────────────
  const phq9Questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
  ];
  const gad7Questions = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
  ];
  const answerOptions = [
    { label: "Not at all", value: 0 },
    { label: "Several days", value: 1 },
    { label: "More than half the days", value: 2 },
    { label: "Nearly every day", value: 3 }
  ];

  const getSeverity = (type, score) => {
    if (type === 'phq9') {
      if (score >= 20) return { label: 'Severe Depression',             color: 'text-rose-400   bg-rose-950/40   border-rose-500/30'   };
      if (score >= 15) return { label: 'Moderately Severe Depression',  color: 'text-orange-400 bg-orange-950/40 border-orange-500/30' };
      if (score >= 10) return { label: 'Moderate Depression',           color: 'text-amber-400  bg-amber-950/40  border-amber-500/30'  };
      if (score >= 5)  return { label: 'Mild Depression',               color: 'text-blue-400   bg-blue-950/40   border-blue-500/30'   };
      return                   { label: 'Minimal or No Depression Indicators', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-500/30' };
    } else {
      if (score >= 15) return { label: 'Severe Anxiety',   color: 'text-rose-400  bg-rose-950/40  border-rose-500/30'  };
      if (score >= 10) return { label: 'Moderate Anxiety', color: 'text-amber-400 bg-amber-950/40 border-amber-500/30' };
      if (score >= 5)  return { label: 'Mild Anxiety',     color: 'text-blue-400  bg-blue-950/40  border-blue-500/30'  };
      return                   { label: 'Minimal or No Anxiety Indicators', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-500/30' };
    }
  };

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleJournalSubmit = async (e) => {
    e?.preventDefault();
    if (!journalInput.trim()) return;
    if (!passphrase) {
      addToast("Please set a zero-knowledge passphrase in the sidebar before submitting.", "error");
      return;
    }
    if (passphraseStrength.level < 2) {
      addToast("Your passphrase is too weak. Please use a stronger passphrase to protect your health data.", "error");
      return;
    }
    await submitJournalEntry(journalInput, langHint);
    setJournalInput('');
  };

  const handleTextareaKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleJournalSubmit();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setScanFile(file);
      setScanPreview(URL.createObjectURL(file));
      setScanResult('');
      setScanTaskStatus(null);
    }
  };

  const handleScanSubmit = async () => {
    if (!scanFile) return;
    setScanLoading(true);
    setScanResult('');
    setScanTaskStatus('PENDING');
    const formData = new FormData();
    formData.append('patient_id', patientId);
    formData.append('mode', scanMode);
    formData.append('file', scanFile);
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/api/scan`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error("Failed to queue scan task.");
      const data = await response.json();
      pollScanTask(data.task_id);
    } catch (err) {
      console.error(err);
      setScanTaskStatus('FAILURE');
      setScanLoading(false);
      addToast("Failed to dispatch document scan task. Ensure backend services are running.", "error");
    }
  };

  const pollScanTask = (taskId) => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch(`${BACKEND_BASE_URL}/api/tasks/${taskId}`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (data.status === 'SUCCESS') {
          clearInterval(interval);
          setScanTaskStatus('SUCCESS');
          setScanResult(data.result.analysis);
          setScanLoading(false);
          addToast("Document scan completed successfully!", "success");
        } else if (data.status === 'FAILURE') {
          clearInterval(interval);
          setScanTaskStatus('FAILURE');
          setScanResult(`Analysis failed: ${data.result?.error || 'Unknown error'}`);
          setScanLoading(false);
          addToast("Document scan failed. Please try again.", "error");
        }
      } catch (err) { console.warn("Retrying scan poll…", err); }
    }, 1000);
  };

  const downloadReport = async (type = 'journal') => {
    try {
      let results = {};
      if (type === 'journal') {
        const lj = journals.find(j => j.status === 'SUCCESS');
        results = {
          prediction:       lj ? lj.prediction          : 'No text assessed',
          depression_score: lj ? lj.scores.depression   : 0.0,
          anxiety_score:    lj ? lj.scores.anxiety      : 0.0,
          normal_score:     lj ? lj.scores.normal       : 0.0,
          processed_text:   lj ? lj.decryptedText       : 'No entries submitted yet.',
          assessment_type:  "Comprehensive Clinical Screener Sync",
          language:         "English / Local Translate"
        };
      } else if (type === 'screener') {
        const phqSeverity = getSeverity('phq9', phq9Score).label;
        const gadSeverity = getSeverity('gad7', gad7Score).label;
        let txt = `CLINICAL SCREENERS SUMMARY REPORT\n\n1. PHQ-9 Depression Screener Score: ${phq9Score}/27\n   Severity: ${phqSeverity}\n\n   PHQ-9 Question Details:\n`;
        phq9Questions.forEach((q, i) => {
          const v = Math.max(0, phq9Answers[i]);
          txt += `   - Q${i+1}: ${q} -> [${["Not at all","Several days","More than half the days","Nearly every day"][v] || 'Not answered'}]\n`;
        });
        txt += `\n2. GAD-7 Anxiety Screener Score: ${gad7Score}/21\n   Severity: ${gadSeverity}\n\n   GAD-7 Question Details:\n`;
        gad7Questions.forEach((q, i) => {
          const v = Math.max(0, gad7Answers[i]);
          txt += `   - Q${i+1}: ${q} -> [${["Not at all","Several days","More than half the days","Nearly every day"][v] || 'Not answered'}]\n`;
        });
        results = {
          prediction:       `PHQ-9: ${phqSeverity} | GAD-7: ${gadSeverity}`,
          depression_score: phq9Score / 27,
          anxiety_score:    gad7Score / 21,
          normal_score:     Math.max(0.0, 1.0 - (phq9Score / 27 + gad7Score / 21) / 2),
          processed_text:   txt,
          assessment_type:  "Interactive Clinical Screeners (PHQ-9 & GAD-7)",
          language:         "Clinical Standard Questionnaire"
        };
      } else {
        results = {
          prediction: 'scan_assessment', depression_score: 0.0, anxiety_score: 0.0, normal_score: 0.0,
          processed_text:  scanResult || 'No scan results available.',
          assessment_type: `OCR Document Scan: ${scanMode.toUpperCase()}`,
          language:        "Image Input"
        };
      }
      const res = await fetch(`${BACKEND_BASE_URL}/api/report`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: patientId, results })
      });
      if (!res.ok) throw new Error("PDF generation failed");
      const blob = await res.blob();
      const url  = window.URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = `mindscan_${type}_report_${patientId}.pdf`;
      document.body.appendChild(a); a.click(); a.remove();
      window.URL.revokeObjectURL(url);
      addToast("Encrypted PDF report downloaded successfully.", "success");
    } catch (err) {
      console.error(err);
      addToast("Failed to download PDF report. Ensure backend services are running.", "error");
    }
  };

  // ── Chart Data ─────────────────────────────────────────────────────────────
  const chartData = useMemo(() => {
    const rows = [...journals].filter(j => j.status === 'SUCCESS').reverse();
    if (!rows.length) return { points: '', data: [] };
    const W = 500, H = 120, P = 20, C = rows.length;
    const map = { normal: 0, anxiety: 1, depression: 2 };
    const data = rows.map((j, i) => {
      const v = map[j.prediction] ?? 0;
      return { x: P + (i * (W - 2*P)) / Math.max(1, C - 1), y: H - P - (v * (H - 2*P)) / 2, j };
    });
    return { points: data.map(d => `${d.x},${d.y}`).join(' '), data };
  }, [journals]);

  // ── Distribution & Trend ──────────────────────────────────────────────────
  const labelDistribution = useMemo(() => {
    const ok = journals.filter(j => j.status === 'SUCCESS');
    if (!ok.length) return { normal: 0, anxiety: 0, depression: 0 };
    const c = { normal: 0, anxiety: 0, depression: 0 };
    ok.forEach(j => { if (c[j.prediction] !== undefined) c[j.prediction]++; });
    return { normal: Math.round(c.normal/ok.length*100), anxiety: Math.round(c.anxiety/ok.length*100), depression: Math.round(c.depression/ok.length*100) };
  }, [journals]);

  const prevDistribution = useMemo(() => {
    const ok = journals.filter(j => j.status === 'SUCCESS');
    if (ok.length <= 2) return null;
    const half = Math.floor(ok.length / 2);
    const older = ok.slice(half);
    const c = { normal: 0, anxiety: 0, depression: 0 };
    older.forEach(j => { if (c[j.prediction] !== undefined) c[j.prediction]++; });
    return { normal: Math.round(c.normal/older.length*100), anxiety: Math.round(c.anxiety/older.length*100), depression: Math.round(c.depression/older.length*100) };
  }, [journals]);

  // ─────────────────────────────────────────────────────────────────────────
  // Screener Question Block (shared by PHQ-9 and GAD-7)
  // ─────────────────────────────────────────────────────────────────────────
  const ScreenerQuestionBlock = ({ questions, answers, setAnswers, screenerKey }) => (
    <div className="space-y-5">
      {questions.map((q, qIdx) => (
        <div key={qIdx} className="space-y-2.5">
          <p className={`text-xs font-semibold flex items-start gap-2 ${answers[qIdx] >= 0 ? 'text-gray-200' : 'text-gray-400'}`}>
            <span className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold transition-colors ${answers[qIdx] >= 0 ? 'bg-indigo-600/50 text-indigo-300' : 'bg-slate-800 text-gray-500'}`}>
              {qIdx + 1}
            </span>
            {q}
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {answerOptions.map(opt => (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  const next = [...answers];
                  next[qIdx] = opt.value;
                  setAnswers(next);
                  setScreenerSubmitted(prev => ({ ...prev, [screenerKey]: false }));
                }}
                className={`text-[11px] p-2.5 rounded-xl border text-center transition-all ${
                  answers[qIdx] === opt.value
                    ? 'bg-indigo-600/30 border-indigo-500 text-white font-bold'
                    : 'bg-slate-900/30 border-white/5 text-gray-400 hover:bg-slate-900/60'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Screener Result Panel (shared by PHQ-9 and GAD-7)
  // ─────────────────────────────────────────────────────────────────────────
  const ScreenerResult = ({ type, score }) => {
    const { label, color } = getSeverity(type, score);
    const maxScore = type === 'phq9' ? 27 : 21;
    const phqActions = [
      [20, "Immediate psychological evaluation and specialist clinical correlation is strongly recommended."],
      [15, "Active therapy / counseling coordination and clinical correlation is recommended."],
      [10, "Professional well-being screening/consultation is suggested."],
      [5,  "Routine well-being assessment and general self-care practices."],
      [0,  "Maintain healthy wellness routines and check progress periodically."],
    ];
    const gadActions = [
      [15, "Immediate professional stress evaluation and counseling sync is recommended."],
      [10, "Therapeutic assessment is suggested to help manage anxiety triggers."],
      [5,  "General stress reduction, mindfulness, and routine monitoring."],
      [0,  "Keep maintaining healthy stress-relieving practices."],
    ];
    const actions = type === 'phq9' ? phqActions : gadActions;
    const action  = actions.find(([threshold]) => score >= threshold)?.[1] || actions[actions.length-1][1];
    const latestAI = journals.find(j => j.status === 'SUCCESS');
    return (
      <div className="space-y-4 pt-4 border-t border-white/5">
        <div className="flex justify-between items-center">
          <span className="text-sm font-semibold text-gray-300">Self-Report Summary</span>
          <span className="text-xs font-mono font-bold">Sum Score: {score} / {maxScore}</span>
        </div>
        <div className={`p-4 rounded-xl border ${color}`}>
          <h4 className="text-sm font-bold font-outfit uppercase tracking-wider">Outcome: {label}</h4>
          <p className="text-[11px] text-gray-300 mt-2 leading-relaxed">Recommended Action: {action}</p>
        </div>
        {latestAI && (
          <div className="bg-slate-900/40 border border-white/5 rounded-xl p-4 space-y-2">
            <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="h-4 w-4 text-indigo-400" />
              AI vs. Self-Report Contrast Analysis
            </h4>
            <p className="text-[11px] text-gray-400 leading-relaxed">
              Your self-reported {type === 'phq9' ? 'PHQ-9' : 'GAD-7'} screening indicates <b>{label}</b>,
              while your latest AI text assessment indicates <b>{latestAI.prediction.toUpperCase()}</b> patterns.
              Spontaneous expression and standardized questionnaires explore distinct cognitive dimensions.
            </p>
          </div>
        )}
        <button
          onClick={() => downloadReport('screener')}
          className="w-full bg-slate-900 border border-white/10 hover:bg-slate-800 text-white font-bold text-xs py-2.5 rounded-xl transition-all flex items-center justify-center gap-1.5"
        >
          <Download className="h-3.5 w-3.5" />
          Export Screener Report to Encrypted PDF
        </button>
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════
  return (
    <div className="min-h-screen flex flex-col font-sans">

      {/* ── Toast Layer ──────────────────────────────────────────────────── */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* ── Wipe History Confirmation Modal ──────────────────────────────── */}
      {showWipeConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm modal-backdrop">
          <div className="glass-panel rounded-2xl p-6 max-w-sm w-full mx-4 space-y-4 border border-rose-500/30 modal-panel">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-6 w-6 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <h3 className="font-outfit font-bold text-white text-sm">Erase All Journal History?</h3>
                <p className="text-xs text-gray-400 mt-1.5 leading-relaxed">
                  This will permanently delete all encrypted journal logs for{' '}
                  <span className="text-white font-semibold">{patientId}</span>.
                  This action <span className="text-rose-400 font-bold">cannot be undone</span>.
                </p>
              </div>
            </div>
            <div className="flex gap-3 pt-1">
              <button
                onClick={() => setShowWipeConfirm(false)}
                className="flex-1 bg-slate-900 border border-white/10 hover:bg-slate-800 text-white text-xs font-semibold py-2.5 rounded-xl transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  clearHistory();
                  setShowWipeConfirm(false);
                  addToast(`Journal history for "${patientId}" has been permanently wiped.`, "info");
                }}
                className="flex-1 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold py-2.5 rounded-xl transition-all"
              >
                Yes, Wipe History
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Mobile Sidebar Scrim ──────────────────────────────────────────── */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="glass-panel sticky top-0 z-40 border-b border-white/5 py-4 px-6 md:px-12 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3 w-full md:w-auto">
          <button
            className="lg:hidden p-2 rounded-lg bg-slate-900/50 border border-white/5 text-gray-400 hover:text-white transition-colors mr-1"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="font-outfit text-xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-violet-300 to-rose-300 bg-clip-text text-transparent">
              MindScan
            </h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest font-semibold font-outfit">AI mental health screening</p>
          </div>
        </div>

        <nav className="flex bg-slate-950/50 p-1 rounded-xl border border-white/5">
          {[
            { id: 'journal',   label: '🧠 Journal Analytics'  },
            { id: 'screeners', label: '📋 Clinical Screeners' },
            { id: 'scan',      label: '📷 Document Scan'      },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all ${
                activeTab === tab.id ? 'bg-indigo-600/90 text-white shadow-md' : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── Main Grid ─────────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-4 gap-8">

        {/* Sidebar */}
        <aside className={`
          lg:col-span-1 flex flex-col gap-6
          fixed lg:relative inset-y-0 left-0 z-40 w-72 lg:w-auto
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          bg-[#0b0f19] lg:bg-transparent overflow-y-auto lg:overflow-visible
          p-4 lg:p-0 pt-20 lg:pt-0`}
        >
          {/* Profile + ZK Panel */}
          <div className="glass-panel p-6 rounded-2xl flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-indigo-400 font-outfit font-semibold uppercase text-xs leading-normal tracking-wider">
                <User className="h-4 w-4" />
                <span>Patient Profile Workspace</span>
              </div>
              <button className="lg:hidden text-gray-500 hover:text-white transition-colors" onClick={() => setSidebarOpen(false)}>
                <X className="h-4 w-4" />
              </button>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Active Patient Record Name / ID</label>
              <input
                type="text"
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
                className="w-full glass-input px-3.5 py-2 rounded-xl text-sm text-white focus:outline-none"
                placeholder="e.g. Guest Patient"
              />
            </div>

            <div className="border-t border-white/5 pt-4 space-y-3">
              <div className="flex items-center gap-2 text-rose-400 font-outfit font-semibold uppercase text-xs leading-normal tracking-wider">
                <Lock className="h-4 w-4" />
                <span>Zero-Knowledge Encryption</span>
              </div>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Enter a secret passphrase. All journal logs are encrypted client-side using <b>AES-GCM-256</b> before saving to local storage.
              </p>
              <div className="space-y-2">
                <label className="block text-xs font-medium text-gray-400">Passphrase Key</label>
                <input
                  type="password"
                  value={passphrase}
                  onChange={e => setPassphrase(e.target.value)}
                  className="w-full glass-input px-3.5 py-2 rounded-xl text-sm text-white focus:outline-none focus:border-rose-500/50"
                  placeholder="Enter secret passphrase"
                />
                {/* Strength Meter */}
                {passphrase && (
                  <div className="space-y-1">
                    <div className="flex gap-1">
                      {[1, 2, 3, 4].map(lvl => (
                        <div
                          key={lvl}
                          className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                            lvl <= passphraseStrength.level ? passphraseStrength.color : 'bg-slate-700'
                          }`}
                        />
                      ))}
                    </div>
                    <p className={`text-[10px] font-semibold ${passphraseStrength.text}`}>
                      {passphraseStrength.label} passphrase
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Privacy Trust Card */}
          <div className="glass-panel p-5 rounded-2xl border-l-4 border-indigo-500 bg-indigo-950/10">
            <div className="flex gap-3">
              <Shield className="h-5 w-5 text-indigo-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-gray-200 uppercase tracking-wide">Privacy Sovereignty</h4>
                <p className="text-[11px] text-gray-400 mt-1 leading-relaxed">
                  MindScan processes model predictions in-memory. Raw images and text strings are never physically stored on server disk drives.
                </p>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <section className="lg:col-span-3 flex flex-col gap-6">

          {/* Crisis Banner */}
          {triggerCrisisOverlay && showCrisisBanner && (
            <div className="pulse-emergency bg-rose-950/80 border border-rose-500/40 rounded-2xl p-6 relative">
              <button onClick={() => setShowCrisisBanner(false)} className="absolute top-4 right-4 text-rose-400 hover:text-white transition-colors">
                <X className="h-4 w-4" />
              </button>
              <div className="flex gap-4">
                <AlertTriangle className="h-8 w-8 text-rose-400 shrink-0" />
                <div className="space-y-3">
                  <h3 className="font-outfit font-bold text-rose-300 text-sm tracking-wide uppercase">🚨 Clinical Emergency Support & Crisis Resources</h3>
                  <p className="text-rose-100/90 text-xs leading-relaxed">
                    If you or someone you know is experiencing severe distress or thoughts of self-harm, please reach out immediately. Support is free, confidential, and available 24/7.
                  </p>
                  <div className="flex flex-wrap gap-3 pt-2">
                    <a href="tel:14416"      className="bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs py-2 px-4 rounded-xl shadow-lg transition-all text-center">📞 Tele-MANAS: 14416</a>
                    <a href="tel:9152987821" className="bg-rose-950/60 border border-rose-500/20 hover:bg-rose-900 text-rose-300 font-semibold text-xs py-2 px-4 rounded-xl transition-all text-center">📞 TISS iCall: 9152987821</a>
                    <a href="tel:9820466726" className="bg-rose-950/60 border border-rose-500/20 hover:bg-rose-900 text-rose-300 font-semibold text-xs py-2 px-4 rounded-xl transition-all text-center">📞 AASRA: 9820466726</a>
                    <a href="tel:988"        className="bg-rose-950/60 border border-rose-500/20 hover:bg-rose-900 text-rose-300 font-semibold text-xs py-2 px-4 rounded-xl transition-all text-center">📞 988 Lifeline (US)</a>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab Content — key forces remount for fade-in */}
          <div key={tabKey} className="tab-content flex flex-col gap-6">

            {/* ══════════════════════════════════════════════════════════════
                TAB 1 · Journal Analytics
            ══════════════════════════════════════════════════════════════ */}
            {activeTab === 'journal' && (
              <div className="space-y-6">

                {/* Journal Input Form */}
                <div className="glass-panel p-6 rounded-2xl space-y-4">
                  <div className="flex justify-between items-center">
                    <h2 className="font-outfit font-bold text-lg text-white">Daily Wellbeing Journal Entry</h2>
                    <div className="flex items-center gap-2">
                      <Globe className="h-4 w-4 text-gray-500" />
                      <select
                        value={langHint}
                        onChange={e => setLangHint(e.target.value)}
                        className="bg-slate-950/50 border border-white/10 rounded-lg text-xs py-1 pl-2.5 pr-8 text-gray-300 focus:outline-none custom-select"
                      >
                        <option value="auto">🌐 Auto-Detect Language</option>
                        <option value="en">English</option>
                        <option value="hi">Hindi (हिन्दी)</option>
                        <option value="kn">Kannada (ಕನ್ನಡ)</option>
                        <option value="ta">Tamil (தமிழ்)</option>
                        <option value="te">Telugu (తెలుగు)</option>
                        <option value="ml">Malayalam (മലയാളം)</option>
                        <option value="mr">Marathi (मराठी)</option>
                        <option value="bn">Bengali (বাংলা)</option>
                      </select>
                    </div>
                  </div>

                  <form onSubmit={handleJournalSubmit} className="space-y-3">
                    <textarea
                      value={journalInput}
                      onChange={e => setJournalInput(e.target.value.slice(0, MAX_CHARS))}
                      onKeyDown={handleTextareaKeyDown}
                      className="w-full glass-input p-4 rounded-2xl text-sm text-white focus:outline-none min-h-[120px] placeholder-gray-500 leading-relaxed resize-none"
                      placeholder="Describe how you are feeling today… (Press Ctrl+Enter to submit)"
                    />
                    <div className="flex justify-between items-center">
                      {/* Live character counter */}
                      <p className={`text-[11px] font-mono transition-colors ${
                        journalInput.length >= 2400 ? 'text-rose-400' :
                        journalInput.length >= 2000 ? 'text-amber-400' :
                        'text-gray-500'
                      }`}>
                        {journalInput.length} / {MAX_CHARS}
                      </p>
                      <button
                        type="submit"
                        disabled={!journalInput.trim() || !passphrase || passphraseStrength.level < 2}
                        className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-semibold text-xs px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/10 flex items-center gap-1.5"
                      >
                        <Sparkles className="h-3.5 w-3.5" />
                        <span>Analyze Wellbeing Patterns</span>
                      </button>
                    </div>
                  </form>
                </div>

                {/* SVG Timeline Chart */}
                {journals.filter(j => j.status === 'SUCCESS').length > 0 && (
                  <div className="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 className="font-outfit font-bold text-sm text-gray-300 uppercase tracking-wider">Wellbeing Risk Severity Timeline</h3>
                    <div className="w-full bg-slate-950/50 rounded-xl p-4 border border-white/5 relative">
                      <svg viewBox="0 0 500 120" className="w-full h-auto overflow-visible">
                        {/* Grid */}
                        {[20, 60, 100].map(y => <line key={y} x1="20" y1={y} x2="480" y2={y} stroke="rgba(255,255,255,0.03)" strokeWidth="1" />)}
                        {/* Y-axis labels */}
                        <text x="5" y="24"  fill="#ef4444" fontSize="7" fontWeight="bold">Depression</text>
                        <text x="5" y="64"  fill="#f59e0b" fontSize="7" fontWeight="bold">Anxiety</text>
                        <text x="5" y="104" fill="#10b981" fontSize="7" fontWeight="bold">Normal</text>
                        {/* Line */}
                        {chartData.points && <polyline fill="none" stroke="url(#chartGrad)" strokeWidth="2.5" points={chartData.points} />}
                        {/* Dots with hover */}
                        {chartData.data.map(({ x, y, j }) => {
                          const col = { normal: '#10b981', anxiety: '#f59e0b', depression: '#ef4444' };
                          return (
                            <circle
                              key={j.id}
                              cx={x} cy={y} r="5"
                              fill={col[j.prediction] || '#fff'}
                              stroke="rgba(15,23,42,0.9)"
                              strokeWidth="1.5"
                              className="cursor-pointer transition-all duration-150 hover:r-7"
                              onMouseEnter={() => setHoveredPoint({ x, y, j })}
                              onMouseLeave={() => setHoveredPoint(null)}
                            />
                          );
                        })}
                        {/* SVG Tooltip */}
                        {hoveredPoint && (() => {
                          const { x, y, j } = hoveredPoint;
                          const bw = 140, bh = 50;
                          const tx = Math.min(x, 500 - bw - 4);
                          const ty = y > 65 ? y - bh - 10 : y + 12;
                          const col = { normal: '#10b981', anxiety: '#f59e0b', depression: '#ef4444' };
                          return (
                            <g style={{ pointerEvents: 'none' }}>
                              <rect x={tx} y={ty} width={bw} height={bh} rx="6" ry="6"
                                fill="rgba(15,23,42,0.97)" stroke="rgba(255,255,255,0.12)" strokeWidth="0.5" />
                              <text x={tx+8} y={ty+15} fill="#d1d5db" fontSize="8" fontWeight="600">
                                {new Date(j.timestamp).toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'2-digit' })}
                              </text>
                              <text x={tx+8} y={ty+28} fill={col[j.prediction] || '#fff'} fontSize="9" fontWeight="bold">
                                {j.prediction?.toUpperCase()}
                              </text>
                              <text x={tx+8} y={ty+41} fill="#6b7280" fontSize="7">
                                {j.scores ? `D:${(j.scores.depression*100).toFixed(0)}%  A:${(j.scores.anxiety*100).toFixed(0)}%  N:${(j.scores.normal*100).toFixed(0)}%` : ''}
                              </text>
                            </g>
                          );
                        })()}
                        <defs>
                          <linearGradient id="chartGrad" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%"   stopColor="#4f46e5" />
                            <stop offset="50%"  stopColor="#a855f7" />
                            <stop offset="100%" stopColor="#f43f5e" />
                          </linearGradient>
                        </defs>
                      </svg>
                    </div>

                    {/* Distribution stats with trend arrows */}
                    <div className="grid grid-cols-3 gap-4 text-center">
                      {[
                        { key: 'normal',     label: 'Wellness',   color: 'text-emerald-400', invertGood: false },
                        { key: 'anxiety',    label: 'Anxious',    color: 'text-amber-400',   invertGood: true  },
                        { key: 'depression', label: 'Depressive', color: 'text-rose-400',    invertGood: true  },
                      ].map(({ key, label, color, invertGood }) => (
                        <div key={key} className="bg-slate-900/40 p-3.5 rounded-xl border border-white/5">
                          <div className={`text-xl font-extrabold ${color} flex items-center justify-center gap-1.5`}>
                            {labelDistribution[key]}%
                            <TrendIcon current={labelDistribution[key]} prev={prevDistribution?.[key]} invertGood={invertGood} />
                          </div>
                          <div className="text-[10px] text-gray-500 uppercase tracking-wider leading-normal mt-0.5">{label} Patterns</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Journal History */}
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h3 className="font-outfit font-bold text-sm text-gray-400 uppercase tracking-wider">Journal Logs & Assessments</h3>
                    {journals.length > 0 && (
                      <div className="flex gap-2">
                        <button
                          onClick={() => downloadReport('journal')}
                          className="text-gray-400 hover:text-white text-xs font-semibold flex items-center gap-1 bg-slate-900/50 px-3 py-1.5 rounded-lg border border-white/5 transition-all"
                        >
                          <Download className="h-3 w-3" /><span>Export Encrypted PDF</span>
                        </button>
                        <button
                          onClick={() => setShowWipeConfirm(true)}
                          className="text-rose-400 hover:text-rose-300 text-xs font-semibold flex items-center gap-1 bg-rose-950/25 px-3 py-1.5 rounded-lg border border-rose-500/10 transition-all"
                        >
                          <Trash2 className="h-3 w-3" /><span>Wipe History</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {journals.length === 0 ? (
                    <div className="glass-panel p-8 rounded-2xl text-center space-y-2">
                      <Info className="h-8 w-8 text-gray-600 mx-auto" />
                      <h4 className="text-gray-400 text-sm font-semibold">No journal history logs found.</h4>
                      <p className="text-xs text-gray-500 max-w-sm mx-auto">Set a passphrase in the sidebar and enter your first journal entry above to initialize wellbeing diagnostics.</p>
                    </div>
                  ) : (
                    <div className="space-y-3.5">
                      {journals.map(j => <JournalCard key={j.id} j={j} />)}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════
                TAB 2 · Clinical Screeners
            ══════════════════════════════════════════════════════════════ */}
            {activeTab === 'screeners' && (
              <div className="glass-panel p-6 rounded-2xl space-y-6">
                {/* Screener type tabs */}
                <div className="flex border-b border-white/5 pb-4 gap-4">
                  {[
                    { id: 'phq9', label: '📋 PHQ-9 (Depression Screener)' },
                    { id: 'gad7', label: '📋 GAD-7 (Anxiety Screener)'    },
                  ].map(s => (
                    <button
                      key={s.id}
                      onClick={() => setActiveScreener(s.id)}
                      className={`pb-2 text-sm font-bold transition-all ${
                        activeScreener === s.id ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>

                {activeScreener === 'phq9' ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="font-outfit font-bold text-white text-base">Patient Health Questionnaire (PHQ-9)</h3>
                      <p className="text-gray-400 text-xs mt-1 leading-relaxed">Over the last 2 weeks, how often have you been bothered by the following?</p>
                    </div>
                    <ScreenerProgress answered={phq9Answered} total={phq9Questions.length} />
                    <ScreenerQuestionBlock
                      questions={phq9Questions}
                      answers={phq9Answers}
                      setAnswers={setPhq9Answers}
                      screenerKey="phq9"
                    />
                    <div className="flex gap-3">
                      <button
                        onClick={() => { setPhq9Answers(Array(9).fill(-1)); setScreenerSubmitted(p => ({ ...p, phq9: false })); }}
                        className="flex items-center gap-1.5 bg-slate-900 border border-white/10 hover:bg-slate-800 text-gray-400 hover:text-white text-xs font-semibold px-4 py-2.5 rounded-xl transition-all"
                      >
                        <RotateCcw className="h-3.5 w-3.5" /> Reset
                      </button>
                      <button
                        onClick={() => setScreenerSubmitted(p => ({ ...p, phq9: true }))}
                        disabled={phq9Answered < phq9Questions.length}
                        className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-bold text-xs py-3 rounded-xl transition-all"
                      >
                        {phq9Answered < phq9Questions.length
                          ? `Answer ${phq9Questions.length - phq9Answered} more question${phq9Questions.length - phq9Answered !== 1 ? 's' : ''} to continue`
                          : 'Calculate PHQ-9 Severity'}
                      </button>
                    </div>
                    {screenerSubmitted.phq9 && <ScreenerResult type="phq9" score={phq9Score} />}
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div>
                      <h3 className="font-outfit font-bold text-white text-base">Generalized Anxiety Disorder (GAD-7)</h3>
                      <p className="text-gray-400 text-xs mt-1 leading-relaxed">Over the last 2 weeks, how often have you been bothered by the following?</p>
                    </div>
                    <ScreenerProgress answered={gad7Answered} total={gad7Questions.length} />
                    <ScreenerQuestionBlock
                      questions={gad7Questions}
                      answers={gad7Answers}
                      setAnswers={setGad7Answers}
                      screenerKey="gad7"
                    />
                    <div className="flex gap-3">
                      <button
                        onClick={() => { setGad7Answers(Array(7).fill(-1)); setScreenerSubmitted(p => ({ ...p, gad7: false })); }}
                        className="flex items-center gap-1.5 bg-slate-900 border border-white/10 hover:bg-slate-800 text-gray-400 hover:text-white text-xs font-semibold px-4 py-2.5 rounded-xl transition-all"
                      >
                        <RotateCcw className="h-3.5 w-3.5" /> Reset
                      </button>
                      <button
                        onClick={() => setScreenerSubmitted(p => ({ ...p, gad7: true }))}
                        disabled={gad7Answered < gad7Questions.length}
                        className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-bold text-xs py-3 rounded-xl transition-all"
                      >
                        {gad7Answered < gad7Questions.length
                          ? `Answer ${gad7Questions.length - gad7Answered} more question${gad7Questions.length - gad7Answered !== 1 ? 's' : ''} to continue`
                          : 'Calculate GAD-7 Severity'}
                      </button>
                    </div>
                    {screenerSubmitted.gad7 && <ScreenerResult type="gad7" score={gad7Score} />}
                  </div>
                )}
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════
                TAB 3 · Document Scanner
            ══════════════════════════════════════════════════════════════ */}
            {activeTab === 'scan' && (
              <div className="glass-panel p-6 rounded-2xl space-y-6">
                <div>
                  <h2 className="font-outfit font-bold text-lg text-white">Scanned Sheet Document Normalization</h2>
                  <p className="text-gray-400 text-xs mt-1 leading-relaxed">
                    Upload a scanned paper screener or journal page. OpenCV corrects document perspective and Meta Llama 4 Scout extracts key clinical insights.
                  </p>
                </div>

                {/* Mode buttons */}
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-gray-400">Analysis Mode</label>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { id: 'general',      icon: '🌐', label: 'General Visual Tone'      },
                      { id: 'social_media', icon: '📱', label: 'Social Media Sentiment'   },
                      { id: 'chart',        icon: '📊', label: 'Scientific Chart Analysis' },
                    ].map(m => (
                      <button
                        key={m.id}
                        onClick={() => setScanMode(m.id)}
                        className={`text-xs px-4 py-2 rounded-xl border transition-all flex items-center gap-1.5 ${
                          scanMode === m.id
                            ? 'bg-indigo-600/30 border-indigo-500 text-white font-bold'
                            : 'bg-slate-900/30 border-white/5 text-gray-400 hover:bg-slate-900/60'
                        }`}
                      >
                        <span>{m.icon}</span><span>{m.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Upload area or preview */}
                {!scanPreview ? (
                  /* ── Empty / Drop State ─────────────────────────────── */
                  <label
                    htmlFor="scan-uploader"
                    className="block border-2 border-dashed border-white/10 rounded-2xl p-10 text-center space-y-5 bg-slate-950/20 hover:border-indigo-500/40 hover:bg-indigo-950/10 transition-all cursor-pointer group"
                  >
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-slate-900/60 border border-white/5 flex items-center justify-center group-hover:scale-105 group-hover:border-indigo-500/20 transition-all duration-300">
                      <Upload className="h-8 w-8 text-gray-500 group-hover:text-indigo-400 transition-colors" />
                    </div>
                    <div className="space-y-1.5">
                      <p className="text-sm text-gray-300 font-semibold">Drop your scanned document here</p>
                      <p className="text-xs text-gray-500">Supports JPG, PNG, WEBP — up to 5 MB</p>
                      <p className="text-[10px] text-gray-600 leading-relaxed max-w-xs mx-auto">
                        OpenCV auto-corrects perspective and applies adaptive binarization before extracting clinical insights via vision AI.
                      </p>
                    </div>
                    <span className="inline-block bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/10">
                      Choose Document Image
                    </span>
                    <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" id="scan-uploader" />
                  </label>
                ) : (
                  /* ── Preview + Submit ─────────────────────────────────── */
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/5">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-400">Uploaded Document</span>
                        <button
                          onClick={() => { setScanPreview(null); setScanFile(null); setScanResult(''); setScanTaskStatus(null); }}
                          className="text-gray-500 hover:text-white text-[10px] flex items-center gap-1 transition-all"
                        >
                          <X className="h-3 w-3" /> Remove
                        </button>
                      </div>
                      <img src={scanPreview} alt="Scan preview" className="w-full h-auto max-h-[300px] object-contain rounded-xl border border-white/5" />
                    </div>
                    <div className="flex flex-col justify-center gap-4">
                      <div className="bg-slate-900/40 p-4 rounded-xl border border-white/5 space-y-2 text-xs">
                        <div className="text-gray-400 uppercase text-[10px] font-bold tracking-wider leading-normal">File Metadata</div>
                        <div className="text-gray-300">Name: {scanFile.name}</div>
                        <div className="text-gray-300">Size: {(scanFile.size / 1024).toFixed(1)} KB</div>
                        <div className="text-gray-300">Mode: {scanMode.replace('_', ' ').toUpperCase()}</div>
                      </div>
                      <button
                        onClick={handleScanSubmit}
                        disabled={scanLoading}
                        className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-bold text-xs py-3 rounded-xl transition-all flex items-center justify-center gap-1.5 shadow-lg shadow-indigo-600/10"
                      >
                        {scanLoading
                          ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" /><span>Processing Document…</span></>
                          : <><Brain className="h-3.5 w-3.5" /><span>🔍 Analyze Uploaded Document</span></>
                        }
                      </button>
                      <p className="text-[10px] text-gray-600 text-center leading-relaxed">
                        Applies perspective correction and extracts clinical insights from your document.
                      </p>
                    </div>
                  </div>
                )}

                {/* Scan Results */}
                {scanTaskStatus && (
                  <div className="space-y-4 pt-4 border-t border-white/5">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold text-gray-300">Scan Assessment Report</span>
                      {scanTaskStatus === 'PENDING'  && <span className="flex items-center gap-1 text-[10px] font-bold text-indigo-400 bg-indigo-950/40 border border-indigo-500/20 px-2.5 py-1 rounded-md animate-pulse"><RefreshCw className="h-3 w-3 animate-spin" />Processing via Celery…</span>}
                      {scanTaskStatus === 'FAILURE'  && <span className="text-[10px] font-bold text-rose-400   bg-rose-950/40   border border-rose-500/20   px-2.5 py-1 rounded-md">Execution Failed</span>}
                      {scanTaskStatus === 'SUCCESS'  && <span className="text-[10px] font-bold text-emerald-400 bg-emerald-950/40 border border-emerald-500/20 px-2.5 py-1 rounded-md">Successfully Processed</span>}
                    </div>
                    {scanResult && (
                      <div className="bg-slate-950/50 p-5 rounded-2xl border border-white/5 max-h-[400px] overflow-y-auto space-y-4">
                        <pre className="text-gray-300 text-xs font-sans whitespace-pre-wrap leading-relaxed">{scanResult}</pre>
                        {scanTaskStatus === 'SUCCESS' && (
                          <button
                            onClick={() => downloadReport('scan')}
                            className="w-full bg-slate-900 border border-white/10 hover:bg-slate-800 text-white font-bold text-xs py-2 rounded-xl transition-all flex items-center justify-center gap-1.5"
                          >
                            <Download className="h-3.5 w-3.5" /> Export Results to Encrypted PDF
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

          </div> {/* end tab-content */}
        </section>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="glass-panel border-t border-white/5 py-8 text-center mt-12 space-y-1.5">
        <p className="text-xs text-gray-500">Built with Fine-tuned BERT · Meta Llama 4 Scout · Helsinki-NLP Translation · React & Zustand</p>
        <p className="text-[10px] text-gray-500">Mysore University School of Engineering · Department of AI & Data Science</p>
        <p className="text-[9px] text-gray-600 max-w-md mx-auto leading-relaxed mt-1">
          ⚠️ MindScan is an AI-assisted screening tool only. It does not provide medical advice, diagnoses, or professional psychological consultation.
        </p>
      </footer>
    </div>
  );
}

export default App;
