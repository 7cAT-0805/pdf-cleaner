import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileText, Loader2, Play, Zap, CheckCircle2, ShieldCheck, XCircle } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | processing | success | error
  const [progress, setProgress] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' });

  const onDrop = useCallback((files) => {
    if (files?.[0]) {
      setFile(files[0]);
      setStatus('idle');
      setProgress(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
  });

  // Poll backend status while processing
  useEffect(() => {
    if (status === 'processing') {
      const interval = setInterval(async () => {
        try {
          const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:5000';
          const res = await fetch(`${apiBase}/status`);
          const data = await res.json();
          if (data.is_processing) {
            setCurrentPage(data.current_page);
            setTotalPages(data.total_pages);
            setProgress(Math.round((data.current_page / data.total_pages) * 100));
          }
        } catch (e) {
          console.error('Polling error:', e);
        }
      }, 500);
      return () => clearInterval(interval);
    }
  }, [status]);

  const runFullConversion = async () => {
    if (!file) return;
    setStatus('processing');
    setProgress(0);
    setCurrentPage(0);
    setTotalPages(0);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiBase}/convert`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || '後端辨識失敗');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `EDITABLE_${file.name.replace('.pdf', '')}.pptx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus('success');
      setProgress(100);
      setToast({ show: true, message: '✅ 轉換完成！', type: 'success' });
    } catch (e) {
      console.error(e);
      setStatus('error');
      setToast({ show: true, message: `❌ ${e.message}`, type: 'error' });
    } finally {
      setTimeout(() => setToast({ show: false, message: '', type: 'success' }), 3000);
    }
  };

  // Reveal animation observer
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add('active');
      });
    }, { threshold: 0.1 });
    document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-slate-300 antialiased selection:bg-blue-500/30 selection:text-blue-200 overflow-x-hidden relative">
      {/* Loading overlay */}
      {status === 'processing' && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <Loader2 className="animate-spin text-blue-400 w-12 h-12" />
          <span className="ml-4 text-white text-lg">處理中 {currentPage}/{totalPages}</span>
        </div>
      )}

      {/* Toast */}
      {toast.show && (
        <div className={clsx(
          'fixed top-4 left-1/2 -translate-x-1/2 z-50 px-6 py-3 rounded-xl shadow-lg transition-opacity',
          toast.type === 'success' ? 'bg-blue-600 text-white' : 'bg-red-600 text-white'
        )}>
          {toast.message}
        </div>
      )}

      {/* Ambient glow */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-blue-600/10 blur-[150px] rounded-full pointer-events-none opacity-40 mix-blend-screen"></div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-40 border-b border-white/[0.08] bg-[#0a0f1a]/90 backdrop-blur-xl supports-[backdrop-filter]:bg-[#0a0f1a]/70">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 group">
            <div className="relative">
              <Zap className="w-5 h-5 text-blue-400 group-hover:text-blue-300 transition-colors duration-500" strokeWidth={1.5} />
              <div className="absolute inset-0 bg-blue-500/50 blur-[10px] opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            </div>
            <span className="text-sm font-bold tracking-widest text-white uppercase">PDF CLEANER</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-blue-400">v2.3.1</span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="relative pt-32 pb-16 md:pt-40 md:pb-24">
        <div className="max-w-4xl mx-auto px-6 relative z-10 text-center">
          <div className="reveal delay-100 inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-xs font-medium text-blue-200 mb-8">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-blue-500"></span>
            </span>
            AI 文字漂白引擎
          </div>
          <h1 className="reveal delay-200 text-5xl md:text-7xl font-semibold tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white via-white to-white/50 mb-6 leading-[1.05]">
            讓 PDF 文字<br />變得可以編輯
          </h1>
          <p className="reveal delay-300 text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
            使用深度學習自動偵測、抹除、重建。支援繁體中文與自動旋轉辨識。
          </p>
        </div>
      </header>

      {/* Control Panel */}
      <section className="max-w-2xl mx-auto px-6 pb-20">
        <div className="reveal delay-100 p-10 rounded-2xl bg-[#0f1420] border border-blue-500/10 hover:border-blue-500/20 transition-all">
          {/* Dropzone */}
          <div {...getRootProps()} className={clsx(
            "group border-2 border-dashed rounded-xl p-16 text-center transition-all cursor-pointer mb-8",
            isDragActive ? "border-blue-500/50 bg-blue-500/10 scale-[1.01]" : "border-blue-500/20 hover:border-blue-500/30 hover:bg-blue-500/5"
          )}>
            <input {...getInputProps()} />
            <FileText className="w-16 h-16 mx-auto text-slate-700 group-hover:text-blue-400 transition-colors mb-4" strokeWidth={1.5} />
            <p className="text-sm font-medium text-slate-400 group-hover:text-slate-200 transition-colors mb-1">
              {file ? file.name : "拖曳 PDF 檔案到此處"}
            </p>
            <p className="text-xs text-slate-600">或點擊選擇檔案</p>
          </div>

          {/* Action button */}
          <button
            onClick={runFullConversion}
            disabled={!file || status === 'processing'}
            className="w-full py-5 bg-blue-600 hover:bg-blue-500 disabled:bg-[#1A1A1A] disabled:text-slate-700 disabled:border disabled:border-white/5 rounded-xl font-semibold text-base transition-all active:scale-[0.98] flex items-center justify-center gap-3 shadow-[0_0_30px_rgba(37,99,235,0.3)] relative overflow-hidden group"
          >
            {status === 'success' ? (
              <>
                <CheckCircle2 className="w-5 h-5" strokeWidth={2} />
                <span>轉換完成</span>
              </>
            ) : status === 'error' ? (
              <>
                <XCircle className="w-5 h-5" strokeWidth={2} />
                <span>轉換失敗</span>
              </>
            ) : (
              <>
                <Play className="fill-current w-4 h-4 group-hover:translate-x-0.5 transition-transform" strokeWidth={0} />
                <span>開始智能轉換</span>
              </>
            )}
            {status === 'processing' && (
              <div className="absolute inset-x-0 bottom-0 h-1 bg-white/10">
                <div className="h-full bg-blue-400 transition-all duration-300" style={{ width: `${progress}%` }} />
              </div>
            )}
          </button>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="max-w-5xl mx-auto px-6 pb-32">
        <div className="grid md:grid-cols-3 gap-6">
          <div className="reveal delay-100 group p-8 rounded-2xl bg-[#0f1420] border border-blue-500/10 hover:border-blue-500/20 transition-all relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-white/5 rounded-bl-[100px] -mr-6 -mt-6 transition-transform group-hover:scale-110"></div>
            <div className="relative z-10">
              <div className="h-10 w-10 flex items-center justify-center text-blue-400 mb-6 bg-blue-500/10 rounded-lg">
                <ShieldCheck className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <h3 className="text-base font-semibold text-white mb-2 tracking-tight">100% 本機運算</h3>
              <p className="text-sm text-slate-500 leading-relaxed">資料不上傳雲端，隱私絕對安全。</p>
            </div>
          </div>
          <div className="reveal delay-200 group p-8 rounded-2xl bg-[#0f1420] border border-blue-500/10 hover:border-blue-500/20 transition-all relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-white/5 rounded-bl-[100px] -mr-6 -mt-6 transition-transform group-hover:scale-110"></div>
            <div className="relative z-10">
              <div className="h-10 w-10 flex items-center justify-center text-blue-400 mb-6 bg-blue-500/10 rounded-lg">
                <Zap className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <h3 className="text-base font-semibold text-white mb-2 tracking-tight">GPU 硬體加速</h3>
              <p className="text-sm text-slate-500 leading-relaxed">自動偵測並啟用 CUDA 運算。</p>
            </div>
          </div>
          <div className="reveal delay-300 group p-8 rounded-2xl bg-[#0f1420] border border-blue-500/10 hover:border-blue-500/20 transition-all relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-white/5 rounded-bl-[100px] -mr-6 -mt-6 transition-transform group-hover:scale-110"></div>
            <div className="relative z-10">
              <div className="h-10 w-10 flex items-center justify-center text-blue-400 mb-6 bg-blue-500/10 rounded-lg">
                <CheckCircle2 className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <h3 className="text-base font-semibold text-white mb-2 tracking-tight">自動旋轉辨識</h3>
              <p className="text-sm text-slate-500 leading-relaxed">傾斜文字自動對齊原始角度。</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.08] bg-[#0a0f1a] py-8">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider">© 2026 PDF CLEANER · v2.3.1</p>
        </div>
      </footer>

      {/* Inline styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; font-feature-settings: "cv02", "cv03", "cv04", "cv11"; }
        .reveal { opacity: 0; transform: translateY(30px) scale(0.98); transition: all 0.8s cubic-bezier(0.16,1,0.3,1); filter: blur(10px); }
        .reveal.active { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
        .delay-100 { transition-delay: 100ms; }
        .delay-200 { transition-delay: 200ms; }
        .delay-300 { transition-delay: 300ms; }
      `}</style>
    </div>
  );
}

export default App;
