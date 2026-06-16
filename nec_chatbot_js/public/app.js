// Alpine.js コンポーネント。/api/ask を呼んで回答・出典を表示する。
function chat() {
  return {
    query: '',
    answer: '',
    sources: [],
    metrics: null,
    loading: false,
    error: '',

    get metricsText() {
      if (!this.metrics) return '';
      const m = this.metrics;
      return `検索 ${m.searchTime}s / 生成 ${m.generationTime}s / 合計 ${m.totalTime}s / トークン ${m.tokens.total}`;
    },

    async ask() {
      const q = this.query.trim();
      if (!q) return;
      this.loading = true;
      this.error = '';
      this.answer = '';
      this.sources = [];
      this.metrics = null;
      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `エラー (${res.status})`);
        this.answer = data.answer;
        this.sources = data.sources || [];
        this.metrics = data.metrics;
      } catch (e) {
        this.error = e.message || '失敗しました';
      } finally {
        this.loading = false;
      }
    },
  };
}
