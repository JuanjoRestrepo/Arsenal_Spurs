# Application Security Reference

Production security has two mandatory layers: **infrastructure** (covered in SKILL.md) and
**application** (this file). This reference covers authentication, authorization, data protection,
API hardening, and operational security patterns across all stacks in this skill.

---

## 1. Password Security

Never store plaintext passwords. Never use MD5 or SHA-1 for password hashing — they are
cryptographically broken for this purpose.

**Hashing — use Argon2id (preferred) or bcrypt:**

```typescript
// Node.js — argon2 (preferred, winner of Password Hashing Competition)
import argon2 from 'argon2';

const hash = await argon2.hash(password, {
  type: argon2.argon2id,
  memoryCost: 65536, // 64 MB
  timeCost: 3,
  parallelism: 4,
});
const valid = await argon2.verify(hash, candidatePassword);

// Alternatively: bcrypt (widely used, still acceptable)
import bcrypt from 'bcryptjs';
const hash = await bcrypt.hash(password, 12); // cost factor ≥ 12
const valid = await bcrypt.compare(candidatePassword, hash);
```

```python
# Python — passlib with argon2
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
hashed = pwd_context.hash(password)
valid = pwd_context.verify(candidate, hashed)
```

**Password strength validation — enforce on the server, not just the client:**

```typescript
// Use zxcvbn for realistic strength estimation (not just regex rules)
import { zxcvbn } from '@zxcvbn-ts/core';

const result = zxcvbn(password);
if (result.score < 3) {
  throw new Error('Password is too weak');
}

// Minimum baseline rules (apply alongside zxcvbn):
const MIN_LENGTH = 12;
const hasUpper = /[A-Z]/.test(password);
const hasLower = /[a-z]/.test(password);
const hasDigit = /\d/.test(password);
const hasSpecial = /[^A-Za-z0-9]/.test(password);
```

**Breach checking — optional but recommended for high-security apps:**
Use the HaveIBeenPwned Passwords API (k-anonymity model — only sends first 5 chars of SHA-1 hash).

---

## 2. Session Management & Cookie Security

**Never store session tokens in `localStorage` or `sessionStorage`** — they are accessible to
JavaScript and vulnerable to XSS. Always use cookies with the correct flags.

**Secure cookie configuration:**

```typescript
// Express
res.cookie('session_id', token, {
  httpOnly: true, // not accessible via document.cookie — XSS mitigation
  secure: true, // only sent over HTTPS
  sameSite: 'lax', // CSRF mitigation; use "strict" for high-security apps
  maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days in ms
  path: '/',
});

// Next.js (App Router) — via cookies() from next/headers
import { cookies } from 'next/headers';
cookies().set('session_id', token, {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax',
  maxAge: 60 * 60 * 24 * 7,
});
```

**Session invalidation on logout — mandatory:**

```typescript
// Server-side: delete or blacklist the session token from DB/Redis
await db.session.delete({ where: { token } });
// or in Redis:
await redis.del(`session:${token}`);

// Client-side: clear the cookie
res.clearCookie('session_id');
// Clearing the cookie alone is NOT sufficient — always invalidate server-side too
```

**Session fixation prevention:** Regenerate session ID after login — never reuse the pre-auth session token.

---

## 3. JWT Best Practices

JWTs are stateless by design — understand the tradeoffs before choosing them over sessions.

**Do:**

- Use short expiry for access tokens: `15m` to `1h`
- Use longer expiry for refresh tokens: `7d` to `30d`, stored server-side (DB or Redis)
- Sign with asymmetric keys (RS256 / ES256) for multi-service architectures
- Sign with HS256 only for single-service apps where secret rotation is controlled
- Verify signature, expiry, issuer (`iss`), and audience (`aud`) on every request

**Don't:**

- Never store JWTs in `localStorage` — store access token in memory, refresh token in httpOnly cookie
- Never put sensitive data in the payload — it is base64-encoded, not encrypted
- Never use `alg: none`
- Never accept tokens without verifying the signature

**Refresh token rotation pattern:**

```typescript
// On token refresh:
// 1. Validate the incoming refresh token against DB
// 2. Issue a new access token AND a new refresh token
// 3. Invalidate the old refresh token immediately (rotation)
// 4. If an already-used refresh token is presented → revoke the entire family (reuse detection)

async function refreshTokens(incomingRefreshToken: string) {
  const stored = await db.refreshToken.findUnique({
    where: { token: incomingRefreshToken },
  });

  if (!stored || stored.used) {
    // Reuse detected — revoke entire family
    await db.refreshToken.deleteMany({ where: { userId: stored?.userId } });
    throw new UnauthorizedException('Refresh token reuse detected');
  }

  await db.refreshToken.update({
    where: { id: stored.id },
    data: { used: true },
  });

  const newAccessToken = signAccessToken(stored.userId);
  const newRefreshToken = await createRefreshToken(stored.userId);

  return { accessToken: newAccessToken, refreshToken: newRefreshToken };
}
```

---

## 4. Role-Based Access Control (RBAC)

Define roles and permissions explicitly — never rely on frontend-only guards.

**Simple RBAC pattern (DB-backed):**

```typescript
// Prisma schema
model User {
  id    String @id @default(cuid())
  role  Role   @default(USER)
}

enum Role {
  USER
  MODERATOR
  ADMIN
}

// Middleware guard
function requireRole(...roles: Role[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: "Forbidden" });
    }
    next();
  };
}

// Route usage
router.delete("/posts/:id", authenticate, requireRole("ADMIN", "MODERATOR"), deletePost);
```

**tRPC (T3 Stack) — role-aware procedures:**

```typescript
const adminProcedure = protectedProcedure.use(({ ctx, next }) => {
  if (ctx.session.user.role !== "ADMIN") {
    throw new TRPCError({ code: "FORBIDDEN" });
  }
  return next({ ctx });
});

export const adminRouter = createTRPCRouter({
  deleteUser: adminProcedure.input(z.object({ id: z.string() })).mutation(...),
});
```

**FastAPI — dependency-based RBAC:**

```python
from fastapi import Depends, HTTPException, status

def require_role(*roles: str):
    def checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return current_user
    return checker

@router.delete("/users/{user_id}")
async def delete_user(user_id: str, _: User = Depends(require_role("admin"))):
    ...
```

---

## 5. API Rate Limiting

Apply rate limiting at multiple layers: reverse proxy/WAF (preferred) + application level (defense in depth).

**Node.js / Express — `express-rate-limit` + Redis store:**

```typescript
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import { redis } from './redis';

// General API limit
export const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  store: new RedisStore({ sendCommand: (...args) => redis.sendCommand(args) }),
});

// Strict limit for auth endpoints
export const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10, // max 10 login attempts per 15 min
  skipSuccessfulRequests: true, // only count failures
  store: new RedisStore({ sendCommand: (...args) => redis.sendCommand(args) }),
});

app.use('/api/', apiLimiter);
app.use('/api/auth/', authLimiter);
```

**FastAPI — `slowapi`:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/data")
@limiter.limit("100/minute")
async def get_data(request: Request):
    ...

@app.post("/auth/login")
@limiter.limit("10/15minute")
async def login(request: Request):
    ...
```

**WebSocket rate limiting:**

```typescript
// Track message frequency per connection
const messageCount = new Map<string, number>();

wss.on('connection', (ws, req) => {
  const ip = req.socket.remoteAddress!;
  messageCount.set(ip, 0);

  ws.on('message', (data) => {
    const count = (messageCount.get(ip) ?? 0) + 1;
    messageCount.set(ip, count);

    if (count > 60) {
      // max 60 messages/min
      ws.close(1008, 'Rate limit exceeded');
      return;
    }
    // process message
  });

  // Reset counter every minute
  const interval = setInterval(() => messageCount.set(ip, 0), 60_000);
  ws.on('close', () => clearInterval(interval));
});
```

---

## 6. IP Controls (Banning & Whitelisting)

**IP whitelisting — for internal/admin routes:**

```typescript
const ADMIN_WHITELIST = (process.env.ADMIN_IP_WHITELIST ?? '').split(',');

function ipWhitelist(req: Request, res: Response, next: NextFunction) {
  const clientIp = req.ip ?? req.socket.remoteAddress;
  if (!ADMIN_WHITELIST.includes(clientIp!)) {
    return res.status(403).json({ error: 'Access denied' });
  }
  next();
}

app.use('/admin', ipWhitelist);
```

**Dynamic IP banning — Redis-backed:**

```typescript
async function checkIpBan(req: Request, res: Response, next: NextFunction) {
  const ip = req.ip!;
  const banned = await redis.get(`ban:${ip}`);
  if (banned) return res.status(403).json({ error: 'Forbidden' });
  next();
}

// Ban an IP for 24 hours
async function banIp(ip: string, reason: string) {
  await redis.setex(`ban:${ip}`, 86400, reason);
  logger.warn({ ip, reason }, 'IP banned');
}

// Auto-ban after N failed auth attempts (pair with rate limiter)
async function recordFailedAttempt(ip: string) {
  const key = `fail:${ip}`;
  const count = await redis.incr(key);
  await redis.expire(key, 3600);
  if (count >= 20) await banIp(ip, 'Excessive failed login attempts');
}
```

**Production recommendation:** prefer WAF-level IP controls (CloudFlare, AWS WAF, GCP Cloud Armor)
over application-level banning — they block traffic before it reaches your server.

---

## 7. WAF (Web Application Firewall)

A WAF is a mandatory layer for any internet-facing production application. It blocks OWASP Top 10
attacks (SQLi, XSS, RFI, path traversal) at the network edge, before traffic reaches your app.

| Provider                 | Best for          | Notes                                                      |
| ------------------------ | ----------------- | ---------------------------------------------------------- |
| **Cloudflare WAF**       | Most apps         | Free tier available; DDoS + bot protection included        |
| **AWS WAF**              | AWS-hosted apps   | Pair with ALB or CloudFront; managed rule groups available |
| **GCP Cloud Armor**      | GCP-hosted apps   | Adaptive protection with ML-based anomaly detection        |
| **Azure Front Door WAF** | Azure-hosted apps | Integrated with Azure CDN and App Gateway                  |

**Minimum WAF ruleset to enable:**

- OWASP Core Rule Set (CRS)
- Rate limiting rules
- Bot management / challenge pages
- Geo-blocking if your app has no international audience

---

## 8. Generic Error Responses

Never leak internal implementation details in API error responses. Stack traces, ORM error messages,
SQL queries, file paths, and library versions are all exploitable intelligence for an attacker.

```typescript
// ❌ WRONG — leaks Prisma internals
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  res.status(500).json({ error: err.message, stack: err.stack });
});

// ✅ CORRECT — structured generic response, full detail in server logs only
import { logger } from './logger';

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  const requestId = req.headers['x-request-id'] ?? crypto.randomUUID();

  // Log full detail server-side — never send to client
  logger.error({ err, requestId, path: req.path }, 'Unhandled error');

  // Send generic response with correlation ID for debugging
  res.status(500).json({
    error: 'An unexpected error occurred',
    requestId, // lets you correlate client reports with server logs
  });
});
```

**Distinguish error types — don't treat everything as 500:**

```typescript
// Use a typed error class hierarchy
class AppError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public isOperational = true,
  ) {
    super(message);
  }
}

class ValidationError extends AppError {
  constructor(message: string) {
    super(400, message);
  }
}
class UnauthorizedError extends AppError {
  constructor() {
    super(401, 'Unauthorized');
  }
}
class ForbiddenError extends AppError {
  constructor() {
    super(403, 'Forbidden');
  }
}
class NotFoundError extends AppError {
  constructor(resource: string) {
    super(404, `${resource} not found`);
  }
}

// In error handler:
if (err instanceof AppError) {
  return res.status(err.statusCode).json({ error: err.message });
}
// Anything else is unexpected — log fully, respond generically
```

**FastAPI:**

```python
from fastapi import Request
from fastapi.responses import JSONResponse
import logging, uuid

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.exception("Unhandled error", extra={"request_id": request_id, "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "request_id": request_id},
    )
```

---

## 9. Security Backups

Backups are a security control — ransomware, accidental deletion, and supply chain attacks all require
a working backup strategy to recover from.

**Backup strategy — follow the 3-2-1 rule:**

- **3** copies of the data
- **2** different storage media/services
- **1** copy offsite (different cloud region or provider)

**Automated DB backup (Postgres example — GitHub Actions):**

```yaml
name: Database Backup
on:
  schedule:
    - cron: '0 2 * * *' # daily at 02:00 UTC

jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - name: Dump and upload to S3
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          AWS_ACCESS_KEY_ID: ${{ secrets.BACKUP_AWS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.BACKUP_AWS_SECRET }}
        run: |
          DATE=$(date +%Y-%m-%d)
          pg_dump "$DATABASE_URL" | gzip > backup-$DATE.sql.gz
          aws s3 cp backup-$DATE.sql.gz s3://my-backups/db/$DATE.sql.gz \
            --sse aws:kms
          rm backup-$DATE.sql.gz

      - name: Verify backup exists
        run: aws s3 ls s3://my-backups/db/$(date +%Y-%m-%d).sql.gz
```

**Key backup rules:**

- Encrypt backups at rest (AES-256 minimum; KMS-managed keys preferred)
- Test restores on a schedule — an untested backup is not a backup
- Set retention policy: daily for 7 days, weekly for 4 weeks, monthly for 12 months
- Use a dedicated IAM role with write-only access to the backup bucket — compromise of the app cannot delete backups
- For managed DBs (Neon, Supabase, PlanetScale): verify their built-in backup retention matches your RPO

---

## 10. Supply Chain Security (Dependency Attacks)

npm's lifecycle hooks (`preinstall`, `postinstall`) execute arbitrary code from the internet
with full developer privileges the moment you run `npm install`. This is the primary attack
vector for modern JavaScript supply chain attacks — and it requires explicit, proactive defense.

### The Threat Landscape

This is not a theoretical risk. Since September 2025, the npm ecosystem has experienced a
documented wave of escalating attacks, all confirmed by Wiz, Trend Micro, Splunk, Palo Alto
Unit 42, Snyk, and StepSecurity:

| Incident                  | Date         | Scope                                           | Vector                                               |
| ------------------------- | ------------ | ----------------------------------------------- | ---------------------------------------------------- |
| **Shai-Hulud**            | Sep 2025     | 500+ packages; first self-propagating npm worm  | `postinstall` script; phished maintainer credentials |
| **Shai-Hulud 2.0**        | Nov 2025     | 796 packages; 132M monthly downloads affected   | `preinstall` scripts; credential theft + backdoors   |
| **Axios / Chalk / Debug** | Mar 2026     | High-impact individual packages                 | Compromised maintainer accounts                      |
| **Mini Shai-Hulud**       | Apr–May 2026 | `@tanstack/*`, `@mistralai/*`, `@bitwarden/cli` | GitHub Actions Pwn Request + OIDC token extraction   |

**What happens on a compromised install:** the malicious `postinstall` script runs silently,
harvests npm tokens, GitHub tokens, AWS/GCP/Azure credentials, and SSH keys from the local
environment; clones private repositories and makes them public; injects malicious GitHub
Actions workflows; and uses any npm tokens found to publish poisoned versions of every
accessible package — propagating automatically without further attacker involvement.

### Defense Layer 1 — Block Lifecycle Scripts

**pnpm (recommended) — disabled by default since v10:**

pnpm v10+ disables `postinstall` script execution for all dependencies by default. Rather than
re-enabling them globally, maintain an explicit allowlist of packages whose build scripts you
trust:

```yaml
# pnpm-workspace.yaml
allowBuilds:
  - esbuild # required to compile its Go binary
  - sharp # requires native image processing compilation
  - bcrypt # native Node.js addon
  - '@parcel/watcher'
  - better-sqlite3
```

Never use `dangerouslyAllowAllBuilds: true` — this fully disables the protection.

**npm — add to `.npmrc` at project root:**

```ini
# .npmrc
ignore-scripts=true
```

Or for a one-off install without disabling globally:

```bash
npm install some-package --ignore-scripts
```

**Trade-off:** some legitimate packages require their build scripts to compile native addons
(e.g., `esbuild`, `sharp`, `bcrypt`, `better-sqlite3`). These will fail silently or with
build errors when scripts are blocked. The correct fix is the `allowBuilds` allowlist
(pnpm) or a per-package exception — never disabling the protection globally.

### Defense Layer 2 — Delay New Versions (Cooldown Period)

Most malicious packages are detected and removed from the npm registry within hours. A
version age requirement means you never install a package version fresh off the registry
during the highest-risk window.

**pnpm v11+ (default: 1 day):**

```yaml
# pnpm-workspace.yaml
minimumReleaseAge: '1440' # minutes — 1 day default in pnpm v11
# set to "10080" for 1 week on high-security projects
# set to "0" to disable (not recommended)
```

**Dependabot (since July 2025) — cooldown on automated PRs:**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: 'npm'
    directory: '/'
    schedule:
      interval: 'weekly'
    cooldown:
      default-days: 7 # wait 7 days before opening a PR for any new version
      semver-patch-days: 3 # shorter wait for patch versions
```

### Defense Layer 3 — Block Exotic Dependency Sources

Prevent transitive dependencies from resolving from git repositories, direct tarball URLs,
or other non-registry sources. These bypass registry-level security scanning entirely.

```yaml
# pnpm-workspace.yaml
blockExoticSubdeps: true
```

### Defense Layer 4 — Trust Policy Enforcement (pnpm v11+)

Prevent installation of a package whose trust level has decreased compared to previous
releases — catching cases where a previously verified publisher's account is compromised:

```yaml
# pnpm-workspace.yaml
trustPolicy: 'no-downgrade'
```

### Defense Layer 5 — Active Scanning

**Always run before merging dependency PRs:**

```bash
npm audit                    # detects known malicious/vulnerable packages
pnpm audit                   # pnpm equivalent
```

**Integrate into CI — block PRs with critical findings:**

```yaml
# .github/workflows/audit.yml
- name: Security audit
  run: pnpm audit --audit-level=high
```

**Third-party scanners (deeper detection — recommended for teams):**

| Tool       | Strength                                                | Integration      |
| ---------- | ------------------------------------------------------- | ---------------- |
| **Socket** | Detects new/changed scripts in PRs before install       | GitHub App       |
| **Snyk**   | Broad vulnerability DB + supply chain monitoring        | GitHub App + CLI |
| **Aikido** | Developer-focused; detects compromised packages quickly | GitHub App       |

Socket is particularly effective: it analyzes package diffs at PR review time and flags
new `postinstall` scripts that weren't present in previous versions — catching attacks before
`npm install` is ever run.

### Defense Layer 6 — Always Commit the Lockfile

A committed lockfile (`pnpm-lock.yaml`, `package-lock.json`, `yarn.lock`) pins every
transitive dependency to an exact version and hash. Without it, `npm install` resolves to
"latest compatible" — which may be a freshly published malicious version.

```bash
# CI should always install from the lockfile — never resolve fresh
pnpm install --frozen-lockfile
npm ci                         # npm equivalent of --frozen-lockfile
```

### Minimum Required Configuration (apply to every project)

**`pnpm-workspace.yaml` (pnpm projects):**

```yaml
allowBuilds:
  - esbuild
  - sharp
  # add others only as needed, with justification

blockExoticSubdeps: true
minimumReleaseAge: '1440'
trustPolicy: 'no-downgrade'
```

**`.npmrc` (npm projects):**

```ini
ignore-scripts=true
audit=true
```

**`.github/dependabot.yml`:**

```yaml
version: 2
updates:
  - package-ecosystem: 'npm'
    directory: '/'
    schedule:
      interval: 'weekly'
    cooldown:
      default-days: 7
```

### If You Suspect Compromise

```bash
# 1. Immediately rotate all credentials accessible from your dev environment:
#    npm token, GitHub token/SSH keys, AWS/GCP/Azure credentials, database passwords

# 2. Audit recently installed packages
npm audit
pnpm audit

# 3. Check for unexpected files dropped during install
find . -name "*.sh" -newer package.json -not -path "*/node_modules/*"
find /tmp -newer /tmp -maxdepth 1 2>/dev/null

# 4. Check for unexpected GitHub Actions workflows added to your repos
git log --all --oneline -- .github/workflows/

# 5. Check for repositories unexpectedly made public
#    GitHub → Settings → Repositories → sort by "recently updated"

# 6. If a GitHub token was exposed: revoke all tokens, audit all org repo access logs
```

---

## 11. Security Checklist (Pre-launch)

Use this before going to production on any project:

**Supply Chain**

- [ ] `ignore-scripts=true` in `.npmrc` (npm) or `allowBuilds` allowlist configured (pnpm)
- [ ] `minimumReleaseAge` / Dependabot cooldown configured
- [ ] `blockExoticSubdeps: true` set (pnpm)
- [ ] Lockfile committed and CI uses `--frozen-lockfile` / `npm ci`
- [ ] Socket, Snyk, or Aikido integrated in GitHub for PR-level scanning
- [ ] `pnpm audit` / `npm audit` runs in CI and blocks on high severity

**Authentication & Authorization**

- [ ] Passwords hashed with Argon2id or bcrypt (cost ≥ 12)
- [ ] Password strength enforced server-side
- [ ] Session tokens in httpOnly + Secure + SameSite cookies
- [ ] Session invalidated server-side on logout
- [ ] JWT expiry ≤ 1h; refresh tokens rotated on use
- [ ] RBAC enforced server-side on every sensitive route

**API & Transport**

- [ ] Rate limiting on all endpoints; stricter on auth routes
- [ ] Rate limiting on WebSocket message frequency
- [ ] HTTPS enforced; HTTP redirects to HTTPS
- [ ] Security headers set (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)
- [ ] WAF enabled (Cloudflare / AWS WAF / GCP Cloud Armor)
- [ ] CORS configured to allowed origins only — never `*` in production

**Data & Error Handling**

- [ ] All user input validated server-side (Zod / Pydantic / Joi)
- [ ] Parameterized queries / ORM used — no raw string SQL interpolation
- [ ] Error responses are generic — no stack traces or internal messages sent to client
- [ ] Sensitive fields excluded from API responses (passwords, tokens, internal IDs)
- [ ] PII minimized — don't store what you don't need

**Infrastructure**

- [ ] Secrets in secret manager — not in `.env` files on servers
- [ ] Dependency audit passing (`npm audit`, `pip-audit`)
- [ ] Backup strategy implemented and restore tested
- [ ] Least-privilege IAM roles — no wildcard permissions in production
- [ ] IP whitelisting on admin/internal routes
- [ ] Logging in place — errors logged with correlation IDs, no sensitive data in logs
