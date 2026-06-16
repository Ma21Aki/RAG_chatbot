// Alpine.js コンポーネント。/api/ask を呼んで回答・出典を表示する。

// HTML特殊文字をエスケープ（先にエスケープしてからタグ化するのでXSSを防ぐ）
function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

// AIが使う範囲（太字・箇条書き・見出し・段落）だけを安全に描画する軽量Markdownレンダラ
function renderMarkdown(md) {
  const inline = (t) => t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  const lines = escapeHtml(md || '').split('\n');
  let html = '';
  let inList = false;
  const closeList = () => {
    if (inList) {
      html += '</ul>';
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[*-]\s+(.*)$/);
    const heading = line.match(/^\s*(#{1,6})\s+(.*)$/);

    if (bullet) {
      if (!inList) {
        html += '<ul>';
        inList = true;
      }
      html += '<li>' + inline(bullet[1]) + '</li>';
      continue;
    }
    closeList();

    if (heading) {
      const lvl = Math.min(heading[1].length + 2, 6);
      html += `<h${lvl}>` + inline(heading[2]) + `</h${lvl}>`;
    } else if (line.trim()) {
      html += '<p>' + inline(line) + '</p>';
    }
  }
  closeList();
  return html;
}

const HISTORY_KEY = 'nec_chatbot_history';
const HISTORY_MAX = 10;
const SEARCH_MODE_KEY = 'nec_chatbot_search_mode';

function loadSearchMode() {
  const v = localStorage.getItem(SEARCH_MODE_KEY);
  return v === 'detailed' || v === 'broad' ? v : 'broad'; // 既定は「ざっくり検索」
}

function loadHistory() {
  try {
    const v = JSON.parse(localStorage.getItem(HISTORY_KEY));
    return Array.isArray(v) ? v.slice(0, HISTORY_MAX) : [];
  } catch {
    return [];
  }
}

function chat() {
  return {
    query: '',
    answer: '',
    lastQuery: '', // 報告に添える、回答時点の質問
    sources: [],
    metrics: null,
    loading: false,
    error: '',

    // ① 検索履歴
    history: loadHistory(),
    showHistory: false,

    // ② 設定（タブ・検索設定・利用統計）
    settingsOpen: false,
    tab: 'search',
    searchMode: loadSearchMode(),
    stats: null,
    statsError: '',

    // ③ 不具合報告
    feedbackOpen: false,
    feedbackType: 'wrong',
    feedbackDetail: '',
    feedbackSending: false,
    feedbackDone: false,
    feedbackError: '',

    get answerHtml() {
      return renderMarkdown(this.answer);
    },

    get metricsText() {
      if (!this.metrics) return '';
      const m = this.metrics;
      return `検索 ${m.searchTime}s / 生成 ${m.generationTime}s / 合計 ${m.totalTime}s / トークン ${m.tokens.total}`;
    },

    async ask() {
      const q = this.query.trim();
      if (!q) return;
      this.showHistory = false;
      this.loading = true;
      this.error = '';
      this.answer = '';
      this.sources = [];
      this.metrics = null;
      // 新しい質問なので不具合フォームを初期化
      this.feedbackOpen = false;
      this.feedbackDone = false;
      this.feedbackDetail = '';
      this.feedbackError = '';
      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, mode: this.searchMode }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `エラー (${res.status})`);
        this.answer = data.answer;
        this.sources = data.sources || [];
        this.metrics = data.metrics;
        this.lastQuery = q;
        this.pushHistory(q);
      } catch (e) {
        this.error = e.message || '失敗しました';
      } finally {
        this.loading = false;
      }
    },

    // ① 検索履歴 ----------------------------------------
    pushHistory(q) {
      const next = [q, ...this.history.filter((h) => h !== q)].slice(0, HISTORY_MAX);
      this.history = next;
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      } catch {}
    },
    useHistory(q) {
      this.query = q;
      this.showHistory = false;
      this.ask();
    },
    clearHistory() {
      this.history = [];
      this.showHistory = false;
      try {
        localStorage.removeItem(HISTORY_KEY);
      } catch {}
    },

    // ② 設定（検索設定・利用統計） ----------------------
    openSettings() {
      this.settingsOpen = true;
      this.tab = 'search';
      this.loadStats();
    },
    saveSearchMode() {
      try {
        localStorage.setItem(SEARCH_MODE_KEY, this.searchMode);
      } catch {}
    },
    async loadStats() {
      this.stats = null;
      this.statsError = '';
      try {
        const res = await fetch('/api/stats');
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `エラー (${res.status})`);
        this.stats = data;
      } catch (e) {
        this.statsError = e.message || '統計の取得に失敗しました';
      }
    },

    // ③ 不具合報告 --------------------------------------
    async submitFeedback() {
      this.feedbackSending = true;
      this.feedbackError = '';
      try {
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: this.feedbackType,
            detail: this.feedbackDetail.trim(),
            query: this.lastQuery,
            answer: this.answer,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `エラー (${res.status})`);
        this.feedbackDone = true;
        this.feedbackOpen = false;
      } catch (e) {
        this.feedbackError = e.message || '送信に失敗しました';
      } finally {
        this.feedbackSending = false;
      }
    },
  };
}
