// レート制限（IP単位の固定ウィンドウ）。乱用・コスト暴発の防止。
export function rateLimiter({ windowMs, max }) {
  const hits = new Map(); // ip -> { count, reset }

  return (req, res, next) => {
    const ip = req.ip || 'unknown';
    const now = Date.now();
    let e = hits.get(ip);
    if (!e || now > e.reset) {
      e = { count: 0, reset: now + windowMs };
      hits.set(ip, e);
    }
    e.count++;
    if (e.count > max) {
      res.status(429).json({ error: 'リクエストが多すぎます。しばらく待ってください' });
      return;
    }
    next();
  };
}
