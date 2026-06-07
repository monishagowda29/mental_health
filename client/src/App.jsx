import React, { useState, useEffect, useMemo } from 'react';
import { useJournalStore } from './store/useJournalStore';
import { 
  Brain, Shield, Lock, AlertCircle, Upload, Activity, 
  FileText, Sparkles, Plus, Trash2, Globe, RefreshCw, 
  Download, User, ArrowRight, CheckCircle2, X, AlertTriangle, Info
} from 'lucide-react';

const BACKEND_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const {
    patientId,
    passphrase,
    journals,
    setPatientId,
    setPassphrase,
    submitJournalEntry,
    clearHistory
  } = useJournalStore();

  const [activeTab, setActiveTab] = useState('journal'); // 'journal' | 'screeners' | 'scan'
  const [journalInput, setJournalInput] = useState('');
  const [langHint, setLangHint] = useState('auto');
  
  // Crisis Alert State
  const [showCrisisBanner, setShowCrisisBanner] = useState(true);
  
  // Scanned Image Tab States
  const [scanMode, setScanMode] = useState('general');
  const [scanFile, setScanFile] = useState(null);
  const [scanPreview, setScanPreview] = useState(null);
  const [scanTaskId, setScanTaskId] = useState(null);
  const [scanTaskStatus, setScanTaskStatus] = useState(null);
  const [scanResult, setScanResult] = useState('');
  const [scanLoading, setScanLoading] = useState(false);

  // PHQ-9 & GAD-7 Screener States
  const [activeScreener, setActiveScreener] = useState('phq9'); // 'phq9' | 'gad7'
  const [phq9Answers, setPhq9Answers] = useState(Array(9).fill(0));
  const [gad7Answers, setGad7Answers] = useState(Array(7).fill(0));
  const [screenerSubmitted, setScreenerSubmitted] = useState({ phq9: false, gad7: false });

  // Self-Harm / Crisis Keyword Check
  const triggerCrisisOverlay = useMemo(() => {
    // 1. Check current journal input for keywords
    const keywords = ['suicide', 'self-harm', 'kill myself', 'die', 'end my life', 'cutting', 'hanging'];
    const lowerText = journalInput.toLowerCase();
    const hasKeyword = keywords.some(k => lowerText.includes(k));
    
    if (hasKeyword) return true;

    // 2. Check last processed journal results (depression > 80%)
    if (journals.length > 0) {
      const latest = journals[0];
      if (latest.status === 'SUCCESS' && latest.prediction === 'depression' && latest.scores?.depression >= 0.8) {
        return true;
      }
    }

    // 3. Check PHQ-9 Screener results (depression / self-harm)
    if (phq9Answers[8] > 0) {
      return true;
    }
    if (screenerSubmitted.phq9 && phq9Score >= 15) {
      return true;
    }

    return false;
  }, [journalInput, journals, phq9Answers, phq9Score, screenerSubmitted.phq9]);

  // PHQ-9 Questions
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

  // GAD-7 Questions
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

  // Screener Calculation Helpers
  const phq9Score = phq9Answers.reduce((a, b) => a + b, 0);
  const gad7Score = gad7Answers.reduce((a, b) => a + b, 0);

  const getSeverity = (type, score) => {
    if (type === 'phq9') {
      if (score >= 20) return { label: 'Severe Depression', color: 'text-rose-400 bg-rose-950/40 border-rose-500/30' };
      if (score >= 15) return { label: 'Moderately Severe Depression', color: 'text-orange-400 bg-orange-950/40 border-orange-500/30' };
      if (score >= 10) return { label: 'Moderate Depression', color: 'text-amber-400 bg-amber-950/40 border-amber-500/30' };
      if (score >= 5) return { label: 'Mild Depression', color: 'text-blue-400 bg-blue-950/40 border-blue-500/30' };
      return { label: 'Minimal or No Depression Indicators', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-500/30' };
    } else {
      if (score >= 15) return { label: 'Severe Anxiety', color: 'text-rose-400 bg-rose-950/40 border-rose-500/30' };
      if (score >= 10) return { label: 'Moderate Anxiety', color: 'text-amber-400 bg-amber-950/40 border-amber-500/30' };
      if (score >= 5) return { label: 'Mild Anxiety', color: 'text-blue-400 bg-blue-950/40 border-blue-500/30' };
      return { label: 'Minimal or No Anxiety Indicators', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-500/30' };
    }
  };

  // Submit Journal Handler
  const handleJournalSubmit = async (e) => {
    e.preventDefault();
    if (!journalInput.trim()) return;
    if (!passphrase) {
      alert("Please enter a zero-knowledge decryption passphrase in the sidebar to secure your health journals.");
      return;
    }
    await submitJournalEntry(journalInput, langHint);
    setJournalInput('');
  };

  // Handle OCR Document file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setScanFile(file);
      setScanPreview(URL.createObjectURL(file));
      setScanResult('');
      setScanTaskStatus(null);
    }
  };

  // Submit OCR Scan to FastAPI
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
      const response = await fetch(`${BACKEND_BASE_URL}/api/scan`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error("Failed to queue scan task.");
      const data = await response.json();
      setScanTaskId(data.task_id);
      pollScanTask(data.task_id);
    } catch (err) {
      console.error(err);
      setScanTaskStatus('FAILURE');
      setScanLoading(false);
      setScanResult('Error occurred dispatching document scan task.');
    }
  };

  // Poll scan task status
  const pollScanTask = (taskId) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/tasks/${taskId}`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        
        if (data.status === 'SUCCESS') {
          clearInterval(interval);
          setScanTaskStatus('SUCCESS');
          setScanResult(data.result.analysis);
          setScanLoading(false);
        } else if (data.status === 'FAILURE') {
          clearInterval(interval);
          setScanTaskStatus('FAILURE');
          setScanResult(`Analysis failed: ${data.result?.error || 'Unknown error'}`);
          setScanLoading(false);
        }
      } catch (err) {
        console.warn("Retrying scan task poll...", err);
      }
    }, 1000);
  };

  // Download Encrypted PDF Report
  const downloadReport = async (type = 'journal') => {
    try {
      let results = {};
      if (type === 'journal') {
        const latestJournal = journals.find(j => j.status === 'SUCCESS');
        results = {
          prediction: latestJournal ? latestJournal.prediction : 'No text assessed',
          depression_score: latestJournal ? latestJournal.scores.depression : 0.0,
          anxiety_score: latestJournal ? latestJournal.scores.anxiety : 0.0,
          normal_score: latestJournal ? latestJournal.scores.normal : 0.0,
          processed_text: latestJournal ? latestJournal.decryptedText : 'No entries submitted yet.',
          assessment_type: "Comprehensive Clinical Screener Sync",
          language: "English / Local Translate"
        };
      } else if (type === 'screener') {
        // Clinical Screener results
        const phqSeverity = getSeverity('phq9', phq9Score).label;
        const gadSeverity = getSeverity('gad7', gad7Score).label;
        
        let textSummary = `CLINICAL SCREENERS SUMMARY REPORT\n\n`;
        textSummary += `1. PHQ-9 Depression Screener Score: ${phq9Score}/27\n`;
        textSummary += `   Severity: ${phqSeverity}\n\n`;
        textSummary += `   PHQ-9 Question Details:\n`;
        phq9Questions.forEach((q, idx) => {
          const val = phq9Answers[idx];
          const labels = ["Not at all", "Several days", "More than half the days", "Nearly every day"];
          textSummary += `   - Q${idx+1}: ${q} -> [${labels[val] || 'Not answered'}]\n`;
        });
        
        textSummary += `\n2. GAD-7 Anxiety Screener Score: ${gad7Score}/21\n`;
        textSummary += `   Severity: ${gadSeverity}\n\n`;
        textSummary += `   GAD-7 Question Details:\n`;
        gad7Questions.forEach((q, idx) => {
          const val = gad7Answers[idx];
          const labels = ["Not at all", "Several days", "More than half the days", "Nearly every day"];
          textSummary += `   - Q${idx+1}: ${q} -> [${labels[val] || 'Not answered'}]\n`;
        });

        results = {
          prediction: `PHQ-9: ${phqSeverity.split(' ')[0]} | GAD-7: ${gadSeverity.split(' ')[0]}`,
          depression_score: phq9Score / 27,
          anxiety_score: gad7Score / 21,
          normal_score: Math.max(0.0, 1.0 - (phq9Score / 27 + gad7Score / 21) / 2),
          processed_text: textSummary,
          assessment_type: "Interactive Clinical Screeners (PHQ-9 & GAD-7)",
          language: "Clinical Standard Questionnaire"
        };
      } else {
        // Document Scan results
        results = {
          prediction: 'scan_assessment',
          depression_score: 0.0,
          anxiety_score: 0.0,
          normal_score: 0.0,
          processed_text: scanResult || 'No scan results available.',
          assessment_type: `OCR Document Scan: ${scanMode.toUpperCase()}`,
          language: "Image Input"
        };
      }

      const res = await fetch(`${BACKEND_BASE_URL}/api/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          results
        })
      });

      if (!res.ok) throw new Error("PDF generation failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mindscan_${type}_report_${patientId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("Failed to download PDF report. Ensure backend services are running.");
    }
  };

  // SVG Line Chart coordinates calculation for Wellbeing Trend Tracker
  const chartPoints = useMemo(() => {
    const successfulJournals = [...journals]
      .filter(j => j.status === 'SUCCESS')
      .reverse();
      
    if (successfulJournals.length === 0) return '';
    
    const width = 500;
    const height = 120;
    const padding = 20;
    const count = successfulJournals.length;
    
    // Map predictions to severity index: normal = 0, anxiety = 1, depression = 2
    const mapping = { normal: 0, anxiety: 1, depression: 2 };
    
    return successfulJournals.map((j, idx) => {
      const val = mapping[j.prediction] !== undefined ? mapping[j.prediction] : 0;
      const x = padding + (idx * (width - 2 * padding)) / Math.max(1, count - 1);
      const y = height - padding - (val * (height - 2 * padding)) / 2;
      return `${x},${y}`;
    }).join(' ');
  }, [journals]);

  // Check language distribution statistics
  const labelDistribution = useMemo(() => {
    const successful = journals.filter(j => j.status === 'SUCCESS');
    if (successful.length === 0) return { normal: 0, anxiety: 0, depression: 0 };
    const counts = { normal: 0, anxiety: 0, depression: 0 };
    successful.forEach(j => {
      if (counts[j.prediction] !== undefined) counts[j.prediction]++;
    });
    const total = successful.length;
    return {
      normal: Math.round((counts.normal / total) * 100),
      anxiety: Math.round((counts.anxiety / total) * 100),
      depression: Math.round((counts.depression / total) * 100)
    };
  }, [journals]);

  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Top Calming Header / Branding */}
      <header className="glass-panel sticky top-0 z-40 border-b border-white/5 py-4 px-6 md:px-12 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="font-outfit text-xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-violet-300 to-rose-300 bg-clip-text text-transparent">
              MindScan
            </h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest font-semibold font-outfit">AI mental health screening</p>
          </div>
        </div>

        {/* Global Nav Tabs */}
        <nav className="flex bg-slate-950/50 p-1 rounded-xl border border-white/5">
          <button 
            onClick={() => setActiveTab('journal')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all ${activeTab === 'journal' ? 'bg-indigo-600/90 text-white shadow-md' : 'text-gray-400 hover:text-white'}`}
          >
            🧠 Journal Analytics
          </button>
          <button 
            onClick={() => setActiveTab('screeners')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all ${activeTab === 'screeners' ? 'bg-indigo-600/90 text-white shadow-md' : 'text-gray-400 hover:text-white'}`}
          >
            📋 Clinical Screeners
          </button>
          <button 
            onClick={() => setActiveTab('scan')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all ${activeTab === 'scan' ? 'bg-indigo-600/90 text-white shadow-md' : 'text-gray-400 hover:text-white'}`}
          >
            📷 Document Scan
          </button>
        </nav>
      </header>

      {/* Main Layout Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
        
        {/* Left Sidebar: Workspace & ZK Security Settings */}
        <aside className="lg:col-span-1 flex flex-col gap-6">
          <div className="glass-panel p-6 rounded-2xl flex flex-col gap-4">
            <div className="flex items-center gap-2 text-indigo-400 font-outfit font-semibold uppercase text-xs leading-normal tracking-wider">
              <User className="h-4 w-4" />
              <span>Patient Profile Workspace</span>
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Active Patient Record Name / ID</label>
              <input 
                type="text" 
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full glass-input px-3.5 py-2 rounded-xl text-sm text-white focus:outline-none"
                placeholder="e.g. Guest Patient"
              />
            </div>

            <div className="border-t border-white/5 pt-4">
              <div className="flex items-center gap-2 text-rose-400 font-outfit font-semibold uppercase text-xs leading-normal tracking-wider mb-2">
                <Lock className="h-4 w-4" />
                <span>Zero-Knowledge Encryption</span>
              </div>
              <p className="text-[11px] text-gray-400 mb-3 leading-relaxed">
                Enter a secret passphrase. All journal logs are encrypted client-side using <b>AES-GCM-256</b> before saving to local storage. Only your passphrase can decrypt them.
              </p>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1.5">Passphrase Key</label>
                <input 
                  type="password" 
                  value={passphrase}
                  onChange={(e) => setPassphrase(e.target.value)}
                  className="w-full glass-input px-3.5 py-2 rounded-xl text-sm text-white focus:outline-none focus:border-rose-500/50"
                  placeholder="Enter secret passphrase"
                />
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

        {/* Center / Right Content Panel */}
        <section className="lg:col-span-3 flex flex-col gap-6">

          {/* Safety Crisis Emergency Support Banner */}
          {triggerCrisisOverlay && showCrisisBanner && (
            <div className="pulse-emergency bg-rose-950/80 border border-rose-500/40 rounded-2xl p-6 relative">
              <button 
                onClick={() => setShowCrisisBanner(false)}
                className="absolute top-4 right-4 text-rose-400 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
              <div className="flex gap-4">
                <AlertTriangle className="h-8 w-8 text-rose-400 shrink-0" />
                <div className="space-y-3">
                  <h3 className="font-outfit font-bold text-rose-300 text-sm tracking-wide uppercase">
                    🚨 Clinical Emergency Support & Crisis Resources
                  </h3>
                  <p className="text-rose-100/90 text-xs leading-relaxed">
                    If you or someone you know is going through a difficult time, experiencing severe distress, or having thoughts of self-harm, please reach out immediately. Compassionate support is free, confidential, and available 24/7.
                  </p>
                  <div className="flex flex-wrap gap-3 pt-2">
                    <a href="tel:14416" className="bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs py-2 px-4 rounded-xl shadow-lg transition-all text-center">
                      📞 Call Tele-MANAS: 14416
                    </a>
                    <a href="tel:9152987821" className="bg-rose-950/60 border border-rose-500/20 hover:bg-rose-900 text-rose-300 font-semibold text-xs py-2 px-4 rounded-xl transition-all text-center">
                      📞 Call TISS iCall: 9152987821
                    </a>
                    <a href="tel:9820466726" className="bg-rose-950/60 border border-rose-500/20 hover:bg-rose-900 text-rose-300 font-semibold text-xs py-2 px-4 rounded-xl transition-all text-center">
                      📞 Call AASRA: 9820466726
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 1: Journal Analytics Workspace */}
          {activeTab === 'journal' && (
            <div className="space-y-6">
              
              {/* Journal Form Box */}
              <div className="glass-panel p-6 rounded-2xl space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="font-outfit font-bold text-lg text-white">Daily Wellbeing Journal Entry</h2>
                  <div className="flex items-center gap-2">
                    <Globe className="h-4 w-4 text-gray-500" />
                    <select 
                      value={langHint} 
                      onChange={(e) => setLangHint(e.target.value)}
                      className="bg-slate-950/50 border border-white/10 rounded-lg text-xs py-1 pl-2.5 pr-8 text-gray-300 focus:outline-none custom-select"
                    >
                      <option value="auto">🌐 Auto-Detect Language</option>
                      <option value="en">English</option>
                      <option value="hi">Hindi (हिन्दी)</option>
                      <option value="kn">Kannada (ಕನ್ನಡ)</option>
                      <option value="ta">Tamil (தமிழ்)</option>
                      <option value="te">Telugu (తెలుగు)</option>
                      <option value="ml">Malayalam (മലയാളம்)</option>
                      <option value="mr">Marathi (मराठी)</option>
                      <option value="bn">Bengali (বাংলা)</option>
                    </select>
                  </div>
                </div>

                <form onSubmit={handleJournalSubmit} className="space-y-4">
                  <textarea 
                    value={journalInput}
                    onChange={(e) => setJournalInput(e.target.value)}
                    className="w-full glass-input p-4 rounded-2xl text-sm text-white focus:outline-none min-h-[120px] placeholder-gray-500 leading-relaxed"
                    placeholder="Describe how you are feeling today... (Input native languages or English. Your content remains encrypted client-side.)"
                  />
                  <div className="flex justify-between items-center">
                    <p className="text-[11px] text-gray-500">Max length: 2500 characters</p>
                    <button 
                      type="submit"
                      disabled={!journalInput.trim() || !passphrase}
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-semibold text-xs px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/10 flex items-center gap-1.5"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      <span>Analyze Wellbeing Patterns</span>
                    </button>
                  </div>
                </form>
              </div>

              {/* Wellbeing Trend Timeline */}
              {journals.filter(j => j.status === 'SUCCESS').length > 0 && (
                <div className="glass-panel p-6 rounded-2xl space-y-4">
                  <h3 className="font-outfit font-bold text-sm text-gray-300 uppercase tracking-wider">Wellbeing Risk Severity Timeline</h3>
                  <div className="w-full bg-slate-950/50 rounded-xl p-4 border border-white/5 relative">
                    <svg viewBox="0 0 500 120" className="w-full h-auto overflow-visible">
                      {/* Grid Lines */}
                      <line x1="20" y1="20" x2="480" y2="20" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="60" x2="480" y2="60" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="100" x2="480" y2="100" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      
                      {/* Y Axis Labels */}
                      <text x="5" y="24" fill="#ef4444" className="text-[8px] font-bold">Depression</text>
                      <text x="5" y="64" fill="#f59e0b" className="text-[8px] font-bold">Anxiety</text>
                      <text x="5" y="104" fill="#10b981" className="text-[8px] font-bold">Normal</text>

                      {/* Connect Line */}
                      {chartPoints && (
                        <polyline
                          fill="none"
                          stroke="url(#chartGrad)"
                          strokeWidth="2.5"
                          points={chartPoints}
                        />
                      )}

                      {/* Point Markers */}
                      {journals.filter(j => j.status === 'SUCCESS').reverse().map((j, idx) => {
                        const width = 500;
                        const height = 120;
                        const padding = 20;
                        const count = journals.filter(j => j.status === 'SUCCESS').length;
                        const mapping = { normal: 0, anxiety: 1, depression: 2 };
                        const val = mapping[j.prediction] !== undefined ? mapping[j.prediction] : 0;
                        const cx = padding + (idx * (width - 2 * padding)) / Math.max(1, count - 1);
                        const cy = height - padding - (val * (height - 2 * padding)) / 2;
                        
                        const colorMap = { normal: '#10b981', anxiety: '#f59e0b', depression: '#ef4444' };
                        
                        return (
                          <circle
                            key={j.id}
                            cx={cx}
                            cy={cy}
                            r="4"
                            fill={colorMap[j.prediction] || '#fff'}
                            stroke="rgba(15, 23, 42, 0.9)"
                            strokeWidth="1.5"
                          />
                        );
                      })}

                      {/* Gradient definition */}
                      <defs>
                        <linearGradient id="chartGrad" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="#4f46e5" />
                          <stop offset="50%" stopColor="#a855f7" />
                          <stop offset="100%" stopColor="#f43f5e" />
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  
                  {/* Distribution Statistics */}
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-slate-900/40 p-3.5 rounded-xl border border-white/5">
                      <div className="text-xl font-extrabold text-emerald-400">{labelDistribution.normal}%</div>
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider leading-normal mt-0.5">Wellness Patterns</div>
                    </div>
                    <div className="bg-slate-900/40 p-3.5 rounded-xl border border-white/5">
                      <div className="text-xl font-extrabold text-amber-400">{labelDistribution.anxiety}%</div>
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider leading-normal mt-0.5">Anxious Patterns</div>
                    </div>
                    <div className="bg-slate-900/40 p-3.5 rounded-xl border border-white/5">
                      <div className="text-xl font-extrabold text-rose-400">{labelDistribution.depression}%</div>
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider leading-normal mt-0.5">Depressive Patterns</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Journal History Log List */}
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="font-outfit font-bold text-sm text-gray-400 uppercase tracking-wider">Journal Logs & Assessments</h3>
                  {journals.length > 0 && (
                    <div className="flex gap-2">
                      <button 
                        onClick={() => downloadReport('journal')}
                        className="text-gray-400 hover:text-white text-xs font-semibold flex items-center gap-1 bg-slate-900/50 px-3 py-1.5 rounded-lg border border-white/5"
                      >
                        <Download className="h-3 w-3" />
                        <span>Export Encrypted PDF</span>
                      </button>
                      <button 
                        onClick={clearHistory}
                        className="text-rose-400 hover:text-rose-300 text-xs font-semibold flex items-center gap-1 bg-rose-950/25 px-3 py-1.5 rounded-lg border border-rose-500/10"
                      >
                        <Trash2 className="h-3 w-3" />
                        <span>Wipe History</span>
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
                    {journals.map((j) => (
                      <div key={j.id} className="glass-panel p-5 rounded-2xl space-y-3 transition-all hover:border-white/10">
                        <div className="flex justify-between items-start">
                          <div className="text-[10px] text-gray-500 font-mono">
                            {new Date(j.timestamp).toLocaleString()}
                          </div>
                          
                          {/* Status / Label Badges */}
                          <div>
                            {j.status === 'PENDING' && (
                              <span className="flex items-center gap-1 text-[10px] font-bold text-indigo-400 bg-indigo-950/40 border border-indigo-500/20 px-2 py-0.5 rounded-md animate-pulse">
                                <RefreshCw className="h-2.5 w-2.5 animate-spin" />
                                <span>Analyzing...</span>
                              </span>
                            )}
                            {j.status === 'FAILURE' && (
                              <span className="text-[10px] font-bold text-rose-400 bg-rose-950/40 border border-rose-500/20 px-2 py-0.5 rounded-md">
                                Failed
                              </span>
                            )}
                            {j.status === 'SUCCESS' && (
                              <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-md ${
                                j.prediction === 'normal' ? 'text-emerald-400 bg-emerald-950/40 border border-emerald-500/20' :
                                j.prediction === 'anxiety' ? 'text-amber-400 bg-amber-950/40 border border-amber-500/20' :
                                'text-rose-400 bg-rose-950/40 border border-rose-500/20'
                              }`}>
                                {j.prediction === 'normal' ? 'No significant risk detected' : `${j.prediction} patterns detected`}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Journal Content */}
                        <p className="text-gray-300 text-xs leading-relaxed font-light">
                          {j.decryptedText ? j.decryptedText : (
                            <span className="text-gray-500 italic">
                              🔒 Zero-Knowledge Encrypted (passphrase required to view)
                            </span>
                          )}
                        </p>

                        {/* Score breakdown if successful */}
                        {j.status === 'SUCCESS' && j.scores && (
                          <div className="flex gap-4 pt-1.5 border-t border-white/5 text-[9px] text-gray-500 font-mono">
                            <div>Normal: {(j.scores.normal * 100).toFixed(0)}%</div>
                            <div>Anxiety: {(j.scores.anxiety * 100).toFixed(0)}%</div>
                            <div>Depression: {(j.scores.depression * 100).toFixed(0)}%</div>
                          </div>
                        )}
                        
                        {j.error && (
                          <p className="text-rose-400 text-[10px] font-mono">{j.error}</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>
          )}

          {/* TAB 2: PHQ-9 & GAD-7 Clinical Screeners */}
          {activeTab === 'screeners' && (
            <div className="glass-panel p-6 rounded-2xl space-y-6">
              
              {/* Questionnaire Selector */}
              <div className="flex border-b border-white/5 pb-4 gap-4">
                <button 
                  onClick={() => { setActiveScreener('phq9'); }}
                  className={`pb-2 text-sm font-bold transition-all relative ${activeScreener === 'phq9' ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-gray-400 hover:text-white'}`}
                >
                  📋 PHQ-9 (Depression Screener)
                </button>
                <button 
                  onClick={() => { setActiveScreener('gad7'); }}
                  className={`pb-2 text-sm font-bold transition-all relative ${activeScreener === 'gad7' ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-gray-400 hover:text-white'}`}
                >
                  📋 GAD-7 (Anxiety Screener)
                </button>
              </div>

              {activeScreener === 'phq9' ? (
                <div className="space-y-6">
                  <div>
                    <h3 className="font-outfit font-bold text-white text-base">Patient Health Questionnaire (PHQ-9)</h3>
                    <p className="text-gray-400 text-xs mt-1 leading-relaxed">
                      Over the last 2 weeks, how often have you been bothered by any of the following problems?
                    </p>
                  </div>

                  <div className="space-y-6">
                    {phq9Questions.map((q, qIdx) => (
                      <div key={qIdx} className="space-y-2.5">
                        <p className="text-gray-200 text-xs font-semibold">{qIdx + 1}. {q}</p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          {answerOptions.map((opt) => (
                            <button
                              key={opt.value}
                              type="button"
                              onClick={() => {
                                const newAnswers = [...phq9Answers];
                                newAnswers[qIdx] = opt.value;
                                setPhq9Answers(newAnswers);
                                setScreenerSubmitted(prev => ({ ...prev, phq9: false }));
                              }}
                              className={`text-[11px] p-2.5 rounded-xl border text-center transition-all ${
                                phq9Answers[qIdx] === opt.value 
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

                  <button 
                    onClick={() => setScreenerSubmitted(prev => ({ ...prev, phq9: true }))}
                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-3 rounded-xl transition-all"
                  >
                    Calculate PHQ-9 Severity
                  </button>

                  {screenerSubmitted.phq9 && (
                    <div className="space-y-4 pt-4 border-t border-white/5">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300">Self-Report Summary</span>
                        <span className="text-xs font-mono font-bold">Sum Score: {phq9Score} / 27</span>
                      </div>
                      
                      {/* Severity outcome */}
                      <div className={`p-4 rounded-xl border ${getSeverity('phq9', phq9Score).color}`}>
                        <h4 className="text-sm font-bold font-outfit uppercase tracking-wider">
                          Outcome: {getSeverity('phq9', phq9Score).label}
                        </h4>
                        <p className="text-[11px] text-gray-300 mt-2 leading-relaxed">
                          Recommended Action: {
                            phq9Score >= 20 ? "Immediate psychological evaluation and specialist clinical correlation is strongly recommended." :
                            phq9Score >= 15 ? "Active therapy / counseling coordination and clinical correlation is recommended." :
                            phq9Score >= 10 ? "Professional well-being screening/consultation is suggested." :
                            phq9Score >= 5 ? "Routine well-being assessment and general self-care practices." :
                            "Maintain healthy wellness routines and check progress periodically."
                          }
                        </p>
                      </div>

                      {/* AI vs Self-Report Contrast Card */}
                      {journals.length > 0 && journals.find(j => j.status === 'SUCCESS') && (
                        <div className="bg-slate-900/40 border border-white/5 rounded-xl p-4 space-y-2">
                          <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                            <Activity className="h-4 w-4 text-indigo-400" />
                            <span>AI vs. Self-Report Contrast Analysis</span>
                          </h4>
                          <p className="text-[11px] text-gray-400 leading-relaxed">
                            Your self-reported GAD-7/PHQ-9 screening indicates <b>{getSeverity('phq9', phq9Score).label}</b>, while your latest spontaneous text assessment indicates <b>{journals.find(j => j.status === 'SUCCESS').prediction.toUpperCase()}</b> patterns. Spontaneous expression classification and standardized questionnaires explore distinct cognitive dimensions.
                          </p>
                        </div>
                      )}
                      
                      <button 
                        onClick={() => downloadReport('screener')}
                        className="w-full bg-slate-900 border border-white/10 hover:bg-slate-800 text-white font-bold text-xs py-2.5 rounded-xl transition-all flex items-center justify-center gap-1.5 mt-2"
                      >
                        <Download className="h-3.5 w-3.5" />
                        <span>Export Screener Report to Encrypted PDF</span>
                      </button>
                    </div>
                  )}

                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h3 className="font-outfit font-bold text-white text-base">Generalized Anxiety Disorder (GAD-7)</h3>
                    <p className="text-gray-400 text-xs mt-1 leading-relaxed">
                      Over the last 2 weeks, how often have you been bothered by any of the following problems?
                    </p>
                  </div>

                  <div className="space-y-6">
                    {gad7Questions.map((q, qIdx) => (
                      <div key={qIdx} className="space-y-2.5">
                        <p className="text-gray-200 text-xs font-semibold">{qIdx + 1}. {q}</p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          {answerOptions.map((opt) => (
                            <button
                              key={opt.value}
                              type="button"
                              onClick={() => {
                                const newAnswers = [...gad7Answers];
                                newAnswers[qIdx] = opt.value;
                                setGad7Answers(newAnswers);
                                setScreenerSubmitted(prev => ({ ...prev, gad7: false }));
                              }}
                              className={`text-[11px] p-2.5 rounded-xl border text-center transition-all ${
                                gad7Answers[qIdx] === opt.value 
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

                  <button 
                    onClick={() => setScreenerSubmitted(prev => ({ ...prev, gad7: true }))}
                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-3 rounded-xl transition-all"
                  >
                    Calculate GAD-7 Severity
                  </button>

                  {screenerSubmitted.gad7 && (
                    <div className="space-y-4 pt-4 border-t border-white/5">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300">Self-Report Summary</span>
                        <span className="text-xs font-mono font-bold">Sum Score: {gad7Score} / 21</span>
                      </div>
                      
                      {/* Severity outcome */}
                      <div className={`p-4 rounded-xl border ${getSeverity('gad7', gad7Score).color}`}>
                        <h4 className="text-sm font-bold font-outfit uppercase tracking-wider">
                          Outcome: {getSeverity('gad7', gad7Score).label}
                        </h4>
                        <p className="text-[11px] text-gray-300 mt-2 leading-relaxed">
                          Recommended Action: {
                            gad7Score >= 15 ? "Immediate professional stress evaluation and counseling sync is recommended." :
                            gad7Score >= 10 ? "Therapeutic assessment is suggested to help manage anxiety triggers." :
                            gad7Score >= 5 ? "General stress reduction, mindfulness, and routine monitoring." :
                            "Keep maintaining healthy stress-relieving practices."
                          }
                        </p>
                      </div>

                      {/* AI vs Self-Report Contrast Card */}
                      {journals.length > 0 && journals.find(j => j.status === 'SUCCESS') && (
                        <div className="bg-slate-900/40 border border-white/5 rounded-xl p-4 space-y-2">
                          <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                            <Activity className="h-4 w-4 text-indigo-400" />
                            <span>AI vs. Self-Report Contrast Analysis</span>
                          </h4>
                          <p className="text-[11px] text-gray-400 leading-relaxed">
                            Your self-reported GAD-7/PHQ-9 screening indicates <b>{getSeverity('gad7', gad7Score).label}</b>, while your latest spontaneous text assessment indicates <b>{journals.find(j => j.status === 'SUCCESS').prediction.toUpperCase()}</b> patterns. Spontaneous expression classification and standardized questionnaires explore distinct cognitive dimensions.
                          </p>
                        </div>
                      )}
                      
                      <button 
                        onClick={() => downloadReport('screener')}
                        className="w-full bg-slate-900 border border-white/10 hover:bg-slate-800 text-white font-bold text-xs py-2.5 rounded-xl transition-all flex items-center justify-center gap-1.5 mt-2"
                      >
                        <Download className="h-3.5 w-3.5" />
                        <span>Export Screener Report to Encrypted PDF</span>
                      </button>
                    </div>
                  )}

                </div>
              )}

            </div>
          )}

          {/* TAB 3: Document Scanner (OCR & Groq Vision) */}
          {activeTab === 'scan' && (
            <div className="glass-panel p-6 rounded-2xl space-y-6">
              <div>
                <h2 className="font-outfit font-bold text-lg text-white">Scanned Sheet Document Normalization</h2>
                <p className="text-gray-400 text-xs mt-1 leading-relaxed">
                  Upload a scanned paper screener or journal page. Our OpenCV pipeline corrects document alignment boundaries, runs a binarization filter, and utilizes Meta Llama 4 Scout to extract key clinical insights.
                </p>
              </div>

              {/* Mode Selection */}
              <div className="space-y-2">
                <label className="block text-xs font-medium text-gray-400">Analysis Mode</label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { id: 'general', icon: '🌐', label: 'General Visual Tone' },
                    { id: 'social_media', icon: '📱', label: 'Social Media Sentiment' },
                    { id: 'chart', icon: '📊', label: 'Scientific Chart Analysis' }
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setScanMode(m.id)}
                      className={`text-xs px-4 py-2 rounded-xl border transition-all flex items-center gap-1.5 ${
                        scanMode === m.id 
                          ? 'bg-indigo-600/30 border-indigo-500 text-white font-bold' 
                          : 'bg-slate-900/30 border-white/5 text-gray-400 hover:bg-slate-900/60'
                      }`}
                    >
                      <span>{m.icon}</span>
                      <span>{m.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* File Uploader */}
              <div className="border border-dashed border-white/10 rounded-2xl p-6 text-center space-y-4 bg-slate-950/20">
                <Upload className="h-8 w-8 text-gray-500 mx-auto" />
                <div className="space-y-1">
                  <p className="text-xs text-gray-300 font-semibold">Click to upload or drag & drop</p>
                  <p className="text-[10px] text-gray-500">Supports JPG, PNG, WEBP up to 5MB</p>
                </div>
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden" 
                  id="scan-uploader" 
                />
                <label 
                  htmlFor="scan-uploader"
                  className="inline-block bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-4 py-2 rounded-xl transition-all cursor-pointer shadow-lg shadow-indigo-600/10"
                >
                  Choose Document Image
                </label>
              </div>

              {/* Uploaded Image Preview & Run Button */}
              {scanPreview && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/5">
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-gray-400">Scanned Document Image</span>
                    <img 
                      src={scanPreview} 
                      alt="Uploaded scan preview" 
                      className="w-full h-auto max-h-[300px] object-contain rounded-xl border border-white/5" 
                    />
                  </div>

                  <div className="flex flex-col justify-center gap-4">
                    <div className="bg-slate-900/40 p-4 rounded-xl border border-white/5 space-y-2 text-xs">
                      <div className="text-gray-400 uppercase text-[10px] font-bold tracking-wider leading-normal">File Metadata</div>
                      <div className="text-gray-300">Name: {scanFile.name}</div>
                      <div className="text-gray-300">Size: {(scanFile.size / 1024).toFixed(1)} KB</div>
                    </div>
                    
                    <button
                      onClick={handleScanSubmit}
                      disabled={scanLoading}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-bold text-xs py-3 rounded-xl transition-all flex items-center justify-center gap-1.5 shadow-lg shadow-indigo-600/10"
                    >
                      {scanLoading ? (
                        <>
                          <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                          <span>Processing Document...</span>
                        </>
                      ) : (
                        <>
                          <Brain className="h-3.5 w-3.5" />
                          <span>Warp Perspective & Run OCR Scan</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* OCR Scan Status & Results */}
              {scanTaskStatus && (
                <div className="space-y-4 pt-4 border-t border-white/5">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-semibold text-gray-300">Scan Assessment Report</span>
                    <div>
                      {scanTaskStatus === 'PENDING' && (
                        <span className="flex items-center gap-1 text-[10px] font-bold text-indigo-400 bg-indigo-950/40 border border-indigo-500/20 px-2.5 py-1 rounded-md animate-pulse">
                          <RefreshCw className="h-3 w-3 animate-spin" />
                          <span>Polling Celery Worker...</span>
                        </span>
                      )}
                      {scanTaskStatus === 'FAILURE' && (
                        <span className="text-[10px] font-bold text-rose-400 bg-rose-950/40 border border-rose-500/20 px-2.5 py-1 rounded-md">
                          Execution Failed
                        </span>
                      )}
                      {scanTaskStatus === 'SUCCESS' && (
                        <span className="text-[10px] font-bold text-emerald-400 bg-emerald-950/40 border border-emerald-500/20 px-2.5 py-1 rounded-md">
                          Successfully Processed
                        </span>
                      )}
                    </div>
                  </div>

                  {scanResult && (
                    <div className="bg-slate-950/50 p-5 rounded-2xl border border-white/5 max-h-[400px] overflow-y-auto space-y-4">
                      <pre className="text-gray-300 text-xs font-sans whitespace-pre-wrap leading-relaxed">
                        {scanResult}
                      </pre>
                      
                      {scanTaskStatus === 'SUCCESS' && (
                        <button 
                          onClick={() => downloadReport('scan')}
                          className="w-full bg-slate-900 border border-white/10 hover:bg-slate-800 text-white font-bold text-xs py-2 rounded-xl transition-all flex items-center justify-center gap-1.5"
                        >
                          <Download className="h-3.5 w-3.5" />
                          <span>Export Results to Encrypted PDF</span>
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}

            </div>
          )}

        </section>

      </main>

      {/* Premium calming footer */}
      <footer className="glass-panel border-t border-white/5 py-8 text-center text-xs text-gray-600 mt-12 space-y-1">
        <p>Built with Fine-tuned BERT · Meta Llama 4 Scout · Helsinki-NLP Translation · React & Zustand</p>
        <p className="text-[10px] text-gray-700">Mysore University School of Engineering · Department of AI & Data Science</p>
      </footer>
    </div>
  );
}

export default App;
