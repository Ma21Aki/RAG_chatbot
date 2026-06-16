// 認証ミドルウェア。Authorization: Bearer <token> または X-Auth-Token を検証する。
// authToken が未設定（空）の場合は認証を無効化（ローカル/開発用）。
export function authMiddleware(authToken) {
  return (req, res, next) => {
    if (!authToken) return next(); // 未設定なら素通し
    const header = req.get('authorization') || '';
    const bearer = header.startsWith('Bearer ') ? header.slice(7) : '';
    const token = bearer || req.get('x-auth-token') || '';
    if (token === authToken) return next();
    res.status(401).json({ error: 'unauthorized' });
  };
}
