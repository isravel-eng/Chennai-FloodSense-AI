import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { createClient } from '@supabase/supabase-js';
import { MapContainer, Marker, TileLayer, Tooltip as LeafletTooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
const CFS_LOGO = "/cfs-logo.webp";

// Supabase client (anon key only — safe in browser)
const supabase = SUPABASE_URL && SUPABASE_ANON_KEY
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
  : null;

const chennai = [13.0827, 80.2707];
const markerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41],
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function formatRainfall(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const num = Number(value);
  if (Number.isInteger(num)) return `${num} mm`;
  const rounded = Math.round(num * 100) / 100;
  const trimmed = rounded.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  return `${trimmed} mm`;
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw new Error(payload?.detail || payload?.message || `Request failed (${response.status})`);
  return payload;
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ---------------------------------------------------------------------------
// Auth Context
// ---------------------------------------------------------------------------
const AuthContext = createContext(null);

function AuthProvider({ children }) {
  const [session, setSession] = useState(undefined); // undefined = loading
  const [profile, setProfile] = useState(null);

  // Restore session from Supabase
  useEffect(() => {
    if (!supabase) { setSession(null); return; }
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      setSession(s || null);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s || null);
      if (!s) setProfile(null);
    });
    return () => subscription.unsubscribe();
  }, []);

  // Fetch user profile whenever session token changes
  useEffect(() => {
    if (!session?.access_token) { setProfile(null); return; }
    getJson(`${API}/users/me`, { headers: authHeaders(session.access_token) })
      .then(setProfile)
      .catch(() => setProfile(null));
  }, [session?.access_token]);

  const register = useCallback(async ({ name, email, password, native_locality }) => {
    const data = await getJson(`${API}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, native_locality }),
    });

    if (data.requires_verification) {
      // Email confirmation required — do NOT attempt login.
      // Return a sentinel so the caller can show the verification message.
      return { requiresVerification: true, email };
    }

    // Auto-confirmed path: backend returned a session immediately.
    if (supabase && data.session?.access_token) {
      await supabase.auth.setSession({
        access_token: data.session.access_token,
        refresh_token: data.session.refresh_token,
      });
    } else if (data.session?.access_token) {
      // Fallback when Supabase client-side is not configured.
      setSession({ access_token: data.session.access_token, ...data.session });
      setProfile({ id: data.id, name: data.name, email: data.email, native_locality: data.native_locality });
    }
    // If no session was returned and no verification is needed, the backend
    // had an unexpected state — surface that to the caller.
    return data;
  }, []);

  const login = useCallback(async ({ email, password }) => {
    const data = await getJson(`${API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (supabase && data.session?.access_token) {
      await supabase.auth.setSession({
        access_token: data.session.access_token,
        refresh_token: data.session.refresh_token,
      });
    } else {
      // Manual session (no Supabase client-side)
      setSession({ access_token: data.session.access_token, ...data.session });
      setProfile(data.user);
    }
    return data;
  }, []);

  const logout = useCallback(async () => {
    if (supabase) {
      await supabase.auth.signOut();
    } else {
      setSession(null);
      setProfile(null);
    }
    try {
      if (session?.access_token) {
        await fetch(`${API}/auth/logout`, {
          method: 'POST',
          headers: authHeaders(session.access_token),
        });
      }
    } catch { /* best-effort */ }
  }, [session]);

  return (
    <AuthContext.Provider value={{ session, profile, register, login, logout, loading: session === undefined }}>
      {children}
    </AuthContext.Provider>
  );
}

function useAuth() { return useContext(AuthContext); }

// ---------------------------------------------------------------------------
// Auth page — Login / Register
// ---------------------------------------------------------------------------
function AuthPage({ localities }) {
  const [tab, setTab] = useState('login');
  const [verificationPending, setVerificationPending] = useState(false);
  const [pendingEmail, setPendingEmail] = useState('');
  const { login, register } = useAuth();

  function handleVerificationRequired(email) {
    setPendingEmail(email);
    setVerificationPending(true);
  }

  return (
    <div className="authPage">
      <div className="authLayout">
        <div className="authIdentity">
          <img className="authHeroLogo" src={CFS_LOGO} alt="Chennai FloodSense AI logo" />
          <div className="authHeroName">Chennai <b>FloodSense AI</b></div>
          <p>AI-powered locality-level flood monitoring for Chennai.</p>
        </div>

        <div className="authCard">
          <div className="authCardHeading">
            <span>{verificationPending ? 'Verify your email' : tab === 'login' ? 'Welcome back' : 'Create your account'}</span>
            <small>
              {verificationPending
                ? 'Finish account verification before signing in.'
                : tab === 'login'
                  ? 'Sign in to access your flood monitoring dashboard.'
                  : 'Set up your profile to personalize locality monitoring.'}
            </small>
          </div>

          {verificationPending ? (
            <div className="verifyBox">
              <div className="verifyIcon">✉️</div>
              <h2>Check your inbox</h2>
              <p>Account created for <b>{pendingEmail}</b>.<br />Click the verification link in your email, then return here to sign in.</p>
              <button className="primary fullButton" onClick={() => { setVerificationPending(false); setPendingEmail(''); setTab('login'); }}>
                Go to Sign In
              </button>
            </div>
          ) : (
            <>
              <div className="authTabs">
                <button type="button" className={tab === 'login' ? 'active' : ''} onClick={() => setTab('login')}>Sign In</button>
                <button type="button" className={tab === 'register' ? 'active' : ''} onClick={() => setTab('register')}>Create Account</button>
              </div>
              {tab === 'login'
                ? <LoginForm onLogin={login} />
                : <RegisterForm onRegister={register} localities={localities} onVerificationRequired={handleVerificationRequired} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function LoginForm({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    setError(''); setLoading(true);
    try { await onLogin({ email: email.trim(), password }); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <form className="authForm" onSubmit={handleSubmit} id="login-form">
      <label htmlFor="login-email">Email</label>
      <input id="login-email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required autoComplete="email" />
      <label htmlFor="login-password">Password</label>
      <input id="login-password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Your password" required autoComplete="current-password" />
      {error && <div className="errorBox">{error}</div>}
      <button className="primary fullButton" type="submit" disabled={loading} id="login-submit">
        {loading ? 'Signing in…' : 'Sign In'}
      </button>
    </form>
  );
}

function RegisterForm({ onRegister, localities, onVerificationRequired }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [locality, setLocality] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    if (password !== confirm) { setError('Passwords do not match.'); return; }
    if (!locality) { setError('Please select your native locality.'); return; }
    setLoading(true);
    try {
      const result = await onRegister({ name: name.trim(), email: email.trim(), password, native_locality: locality });
      // Case B: email verification required — hand off to parent to show message.
      if (result?.requiresVerification && onVerificationRequired) {
        onVerificationRequired(result.email || email.trim());
      }
      // Case A: session was set by onRegister — AuthProvider handles the transition.
    }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <form className="authForm" onSubmit={handleSubmit} id="register-form">
      <label htmlFor="reg-name">Full Name</label>
      <input id="reg-name" type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Your name" required autoComplete="name" />
      <label htmlFor="reg-email">Email</label>
      <input id="reg-email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required autoComplete="email" />
      <label htmlFor="reg-locality">Native Locality</label>
      <select id="reg-locality" value={locality} onChange={e => setLocality(e.target.value)} required>
        <option value="">Select your locality…</option>
        {localities.map(x => <option key={x.name} value={x.name}>{x.name}</option>)}
      </select>
      <label htmlFor="reg-password">Password</label>
      <input id="reg-password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Min 6 characters" required autoComplete="new-password" minLength={6} />
      <label htmlFor="reg-confirm">Confirm Password</label>
      <input id="reg-confirm" type="password" value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Repeat password" required autoComplete="new-password" />
      {error && <div className="errorBox">{error}</div>}
      <button className="primary fullButton" type="submit" disabled={loading} id="register-submit">
        {loading ? 'Creating account…' : 'Create Account'}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Explainability panel
// ---------------------------------------------------------------------------
function ExplainPanel({ risk, locality }) {
  if (!risk) return null;
  const { next_24h, current, context } = risk;
  const forecastMm = next_24h?.forecast_rainfall_mm ?? 0;
  const prob = next_24h?.probability ?? 0;
  const band = String(next_24h?.risk_band || 'LOW').toUpperCase();
  const sevenDay = context?.rainfall_last_7d_mm ?? 0;
  const thirtyDay = context?.rainfall_last_30d_mm ?? 0;
  const currentMm = current?.rainfall_input_mm ?? 0;
  const isNEM = context?.is_northeast_monsoon;
  const histSource = context?.rainfall_history_source || '';
  const change1d = forecastMm - (current?.rainfall_input_mm ?? 0);
  const trendLabel = change1d > 2 ? 'increasing' : change1d < -2 ? 'decreasing' : 'stable';
  const forecastLevel = forecastMm > 30 ? 'elevated (>30 mm)' : forecastMm > 10 ? 'moderate (10–30 mm)' : `low (${forecastMm.toFixed(1)} mm)`;

  const bullets = [
    `Forecast precipitation for the next 24 hours: ${formatRainfall(forecastMm)} — ${forecastLevel}.`,
    `Recent 7-day accumulated rainfall: ${formatRainfall(sevenDay)}.`,
    `Recent 30-day accumulated rainfall: ${formatRainfall(thirtyDay)}.`,
    `Rainfall trend: ${trendLabel} relative to current observation (${formatRainfall(currentMm)}).`,
    isNEM ? 'The Northeast Monsoon season (Oct–Dec) is currently active — this is Chennai\'s primary rainfall period.' : 'The Northeast Monsoon season (Oct–Dec) is not currently active.',
  ];

  return (
    <div className="explainPanel">
      <div className="explainTitle">Why this risk?</div>
      <ul className="explainBullets">
        {bullets.map((b, i) => <li key={i}>{b}</li>)}
      </ul>
      <div className="explainMeta">
        <span><b>Flood model:</b> XGBoost classifier</span>
        <span><b>Data source:</b> Live weather (WeatherAPI) + {histSource === 'live_log' ? 'persisted rainfall history' : 'historical rainfall data'} + ML model</span>
        <span className="explainNote">Risk probability ({Math.round(prob * 100)}%) is a model score, not a guaranteed real-world flood probability.</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Map page
// ---------------------------------------------------------------------------
function MapPage({ localities, selected, setSelected, risk, loading, error, onRefresh }) {
  const [panelOpen, setPanelOpen] = useState(false);
  useEffect(() => { if (selected) setPanelOpen(true); }, [selected]);
  const chart7 = useMemo(() => (risk?.next_7_days || []).map(d => ({ ...d, shortDate: String(d.date).slice(5) })), [risk]);
  const riskClass = String(risk?.next_24h?.risk_band || 'low').toLowerCase();

  return (
    <main className="mapPage">
      <MapContainer center={chennai} zoom={11} scrollWheelZoom className="fullMap">
        <MapResize />
        <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {localities.map(x => (
          <Marker key={x.name} position={[x.latitude, x.longitude]} icon={markerIcon} eventHandlers={{ click: () => setSelected(x.name) }}>
            <LeafletTooltip permanent direction="top" offset={[0, -38]} className="localityLabel">{x.name}</LeafletTooltip>
          </Marker>
        ))}
      </MapContainer>

      <div className="mapOverlay mapTitle">
        <div className="eyebrow">LIVE FLOOD MONITORING</div>
        <h1>Chennai FloodSense AI</h1>
        <p>Click a locality marker to inspect its current risk and forecast.</p>
      </div>

      <div className={`riskDrawer ${panelOpen ? 'open' : ''}`}>
        <button className="drawerClose" onClick={() => setPanelOpen(false)} aria-label="Close locality details">×</button>
        <div className="panelLabel">SELECTED LOCALITY</div>
        <h2>{selected || 'Select a locality on the map'}</h2>
        {loading && <div className="loadingBox">Loading prediction…</div>}
        {!loading && error && <div className="errorBox">{error}</div>}
        {!loading && !error && risk && <>
          <div className={`riskHero ${riskClass}`}>
            <span className="riskDot">●</span>
            <div>
              <small>Next 24-hour flood risk</small>
              <strong>{risk.next_24h?.risk_band || 'Unknown'}</strong>
            </div>
            <b>{Math.round((risk.next_24h?.probability || 0) * 100)}%</b>
          </div>
          <div className="stats">
            <Stat label="Current risk" value={risk.current?.risk_band || '—'} />
            <Stat label="Forecast rainfall" value={formatRainfall(risk.next_24h?.forecast_rainfall_mm ?? 0)} />
            <Stat label="Last 7 days" value={formatRainfall(risk.context?.rainfall_last_7d_mm ?? 0)} />
            <Stat label="Last 30 days" value={formatRainfall(risk.context?.rainfall_last_30d_mm ?? 0)} />
          </div>

          <div className="chartCard compactChart">
            <div className="cardTitle">Next 7 Days · Rainfall</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chart7}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="shortDate" />
                <YAxis />
                <Tooltip formatter={(value) => formatRainfall(value)} />
                <Bar dataKey="rainfall_mm" name="Rainfall (mm)" fill="#1976d2" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="forecastRows">
              {chart7.map(d => (
                <div className="forecastRow" key={d.date}>
                  <span>{d.date}</span>
                  <b>{formatRainfall(d.rainfall_mm)}</b>
                  <em className={String(d.risk_band).toLowerCase()}>{d.risk_band}</em>
                </div>
              ))}
            </div>
          </div>
          <button className="primary fullButton" onClick={onRefresh}>{loading ? 'Updating…' : 'Refresh prediction'}</button>
        </>}
      </div>
    </main>
  );
}

function MapResize() {
  const map = useMap();
  useEffect(() => { const t = setTimeout(() => map.invalidateSize(), 50); return () => clearTimeout(t); }, [map]);
  return null;
}
function Stat({ label, value }) { return <div className="stat"><small>{label}</small><b>{value}</b></div>; }

// ---------------------------------------------------------------------------
// Localities page
// ---------------------------------------------------------------------------
function LocalitiesPage({ localities, onOpen }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('name');
  const [direction, setDirection] = useState('asc');
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    getJson(`${API}/flood-risk-all`)
      .then(data => {
        if (!active) return;
        const next = {};
        (data?.results || []).forEach(item => { if (item.ok) next[item.name] = item.data; });
        setPredictions(next);
      })
      .catch(err => active && setMessage(err.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [localities]);

  const rows = useMemo(() => {
    const filtered = localities.filter(x => x.name.toLowerCase().includes(query.toLowerCase()));
    return [...filtered].sort((a, b) => {
      const av = sort === 'name' ? a.name : sort === 'risk' ? (predictions[a.name]?.next_24h?.probability || 0) : (predictions[a.name]?.next_24h?.forecast_rainfall_mm || 0);
      const bv = sort === 'name' ? b.name : sort === 'risk' ? (predictions[b.name]?.next_24h?.probability || 0) : (predictions[b.name]?.next_24h?.forecast_rainfall_mm || 0);
      const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
      return direction === 'asc' ? cmp : -cmp;
    });
  }, [localities, query, sort, direction, predictions]);

  function changeSort(value) {
    if (value === sort) setDirection(x => x === 'asc' ? 'desc' : 'asc');
    else { setSort(value); setDirection('asc'); }
  }

  return (
    <main className="page">
      <div className="eyebrow">LOCALITY ANALYSIS</div>
      <h1>All Project Localities</h1>
      <p className="subtitle">Live flood-risk and rainfall prediction for every configured locality.</p>
      <div className="toolbar localityToolbar">
        <input value={query} placeholder="Search locality…" onChange={e => setQuery(e.target.value)} id="locality-search" />
        <select value={sort} onChange={e => changeSort(e.target.value)} id="locality-sort">
          <option value="name">Sort: Locality</option>
          <option value="risk">Sort: Risk probability</option>
          <option value="rainfall">Sort: Forecast rainfall</option>
        </select>
        <span className="sortHint">Choose the same sort again to reverse order</span>
      </div>
      {message && <div className="notice">{message}</div>}
      <div className="table">
        <div className="thead localityGrid">
          <span>LOCALITY</span><span>24H RISK</span><span>PROBABILITY</span><span>RAINFALL</span><span>ACTION</span>
        </div>
        {loading && <div className="tableMessage">Loading all locality predictions…</div>}
        {!loading && rows.map(x => {
          const p = predictions[x.name];
          return (
            <button className="tr localityGrid" key={x.name} onClick={() => onOpen(x.name)}>
              <span><b>{x.name}</b><small>{x.latitude.toFixed(4)}, {x.longitude.toFixed(4)}</small></span>
              <span className={String(p?.next_24h?.risk_band || '—').toLowerCase()}>{p?.next_24h?.risk_band || '—'}</span>
              <span>{p ? `${Math.round((p.next_24h.probability || 0) * 100)}%` : '—'}</span>
              <span>{p ? formatRainfall(p.next_24h.forecast_rainfall_mm) : '—'}</span>
              <span>View →</span>
            </button>
          );
        })}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// About page
// ---------------------------------------------------------------------------
function AboutPage({ risk, selected }) {
  return (
    <main className="page infoPage">
      <div className="infoHero">
        <img className="infoHeroLogo" src={CFS_LOGO} alt="Chennai FloodSense AI logo" />
        <div>
          <div className="eyebrow">ABOUT THE APPLICATION</div>
          <h1>Chennai FloodSense AI</h1>
          <p className="subtitle">Locality-level rainfall monitoring, flood-risk assessment, and forecasting for Chennai.</p>
        </div>
      </div>

      <div className="infoCard">
        <h2>How to use the app</h2>
        <p>Use <b>Map</b> to view Chennai localities and select a marker to inspect the current 24-hour flood-risk assessment and the next 7 days of rainfall.</p>
        <p>Use <b>Localities</b> to compare configured locations and open any locality directly on the map.</p>
        <p>Use <b>Rainfall Prediction</b> to select a locality and forecast horizon and generate the seasonal SARIMA rainfall forecast.</p>
        <p>Use <b>your name</b> in the navigation bar to open your user profile.</p>

        <h2>What the risk colours mean</h2>
        <div className="riskLegend">
          <div><span className="riskLegendDot low"></span><span><b>LOW</b> — lower model-assessed flood risk.</span></div>
          <div><span className="riskLegendDot medium"></span><span><b>MEDIUM</b> — elevated model-assessed flood risk.</span></div>
          <div><span className="riskLegendDot high"></span><span><b>HIGH</b> — high model-assessed flood risk.</span></div>
          <div><span className="riskLegendDot veryHigh"></span><span><b>VERY HIGH</b> — highest model-assessed flood-risk band.</span></div>
        </div>

        <h2>Why this risk?</h2>
        {risk ? <ExplainPanel risk={risk} locality={selected} /> : (
          <p>Select a locality on the Map first. This explanation uses forecast rainfall, recent rainfall history, seasonal context, and the ML risk score.</p>
        )}

        <h2>Forecast and risk note</h2>
        <p>The rainfall forecast and flood-risk score are model outputs intended to support monitoring. They are not guaranteed real-world flood probabilities or official warnings.</p>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// User profile page
// ---------------------------------------------------------------------------
function ProfilePage({ profile, session }) {
  const email = profile?.email || session?.user?.email || '—';
  const name = profile?.name || session?.user?.user_metadata?.name || (email.includes('@') ? email.split('@')[0] : 'User');
  const nativeLocality = profile?.native_locality || 'Not set';

  return (
    <main className="page profilePage">
      <div className="profileHero">
        <img className="profileHeroLogo" src={CFS_LOGO} alt="Chennai FloodSense AI logo" />
        <h1>{name}</h1>
        <p>Chennai FloodSense AI user profile</p>
      </div>

      <div className="profileCard">
        <div className="profileRow"><span>NAME</span><strong>{name}</strong></div>
        <div className="profileRow"><span>EMAIL</span><strong>{email}</strong></div>
        <div className="profileRow"><span>NATIVE LOCALITY</span><strong>{nativeLocality}</strong></div>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Rainfall forecast page
// ---------------------------------------------------------------------------
function RainfallPage({ localities, selected, setSelected }) {
  const [months, setMonths] = useState(12);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function generate() {
    if (!selected) return;
    setLoading(true); setError('');
    try { setData(await getJson(`${API}/rainfall-forecast/locality/${encodeURIComponent(selected)}?months=${months}`)); }
    catch (err) { setData(null); setError(err.message); }
    finally { setLoading(false); }
  }

  useEffect(() => { if (selected) generate(); }, [selected]);
  const chart = useMemo(() => data?.forecast || [], [data]);

  return (
    <main className="page">
      <div className="eyebrow">RAINFALL FORECASTING</div>
      <h1>Rainfall Prediction</h1>
      <p className="subtitle">Locality-specific rainfall forecasting with a clear trend graph.</p>
      <div className="controlCard">
        <div>
          <label>LOCALITY</label>
          <select value={selected} onChange={e => setSelected(e.target.value)} id="rf-locality-select">
            {localities.map(x => <option key={x.name} value={x.name}>{x.name}</option>)}
          </select>
        </div>
        <div>
          <label>HORIZON</label>
          <select value={months} onChange={e => setMonths(Number(e.target.value))} id="rf-horizon-select">
            <option value={12}>12 months</option>
            <option value={24}>24 months</option>
            <option value={36}>36 months</option>
          </select>
        </div>
        <button className="primary" onClick={generate} disabled={loading} id="rf-generate-btn">
          {loading ? 'Generating…' : 'Generate Forecast'}
        </button>
      </div>
      {error && <div className="errorBox">{error}</div>}
      {data?.status === 'fallback_forecast' && (
        <div className="notice">{data.message}</div>
      )}
      {data?.status === 'model_fit_failed' && <div className="notice">{data.message}</div>}
      {chart.length > 0 && <>
        <div className="chartCard">
          <div className="cardTitle">{data?.model?.name || 'Rainfall'} Forecast · {selected} · {months} months</div>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" interval={months > 18 ? 2 : 0} />
              <YAxis />
              <Tooltip formatter={(value) => formatRainfall(value)} />
              <Legend />
              <Line type="monotone" dataKey="forecast_mm" name="Predicted rainfall" stroke="#1976d2" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="forecastNote">
            Predicted rainfall is shown as the primary forecast trend. Confidence bounds are retained in the table and are not plotted.
          </div>
        </div>
        <div className="table">
          <div className="thead forecastGrid">
            <span>MONTH</span><span>PREDICTED RAINFALL</span><span>LOWER BOUND</span><span>UPPER BOUND</span>
          </div>
          {chart.map(x => (
            <div className="tr static forecastGrid" key={x.month}>
              <span>{x.month}</span>
              <span>{formatRainfall(x.forecast_mm)}</span>
              <span>{formatRainfall(x.lower_95_mm)}</span>
              <span>{formatRainfall(x.upper_95_mm)}</span>
            </div>
          ))}
        </div>
      </>}
    </main>
  );
}

// ---------------------------------------------------------------------------
// Main app shell (authenticated)
// ---------------------------------------------------------------------------
function AppShell({ localities }) {
  const { profile, session, logout } = useAuth();
  const [page, setPage] = useState('map');
  const [selected, setSelected] = useState('');
  const [risk, setRisk] = useState(null);
  const [loadingRisk, setLoadingRisk] = useState(false);
  const [riskError, setRiskError] = useState('');

  useEffect(() => {
    if (profile?.native_locality && !selected) setSelected(profile.native_locality);
  }, [profile, selected]);

  useEffect(() => {
    if (page === 'map' && selected) loadRisk(selected);
  }, [selected, page]);

  async function loadRisk(name) {
    setLoadingRisk(true);
    setRiskError('');
    try {
      setRisk(await getJson(`${API}/flood-risk/${encodeURIComponent(name)}`));
    } catch (err) {
      setRisk(null);
      setRiskError(err.message);
    } finally {
      setLoadingRisk(false);
    }
  }

  function openLocality(name) {
    setSelected(name);
    setPage('map');
  }

  const profileLabel =
    profile?.name ||
    profile?.email?.split('@')[0] ||
    session?.user?.email?.split('@')[0] ||
    'Profile';

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img className="headerLogo" src={CFS_LOGO} alt="Chennai FloodSense AI logo" />
          <span>Chennai <b>FloodSense AI</b></span>
        </div>

        <nav className="mainNav">
          {[
            ['map', 'Map'],
            ['localities', 'Localities'],
            ['rainfall', 'Rainfall Prediction'],
            ['about', 'About'],
            ['profile', profileLabel],
          ].map(([id, label]) => (
            <button type="button" key={id} className={page === id ? 'active' : ''} onClick={() => setPage(id)} id={`nav-${id}`}>
              {label}
            </button>
          ))}
        </nav>

        <div className="userChip">
          <button className="logoutBtn" onClick={logout} id="logout-btn" title="Sign out">Sign out</button>
        </div>
      </header>

      {page === 'map' && <MapPage localities={localities} selected={selected} setSelected={setSelected} risk={risk} loading={loadingRisk} error={riskError} onRefresh={() => selected && loadRisk(selected)} />}
      {page === 'localities' && <LocalitiesPage localities={localities} onOpen={openLocality} />}
      {page === 'rainfall' && <RainfallPage localities={localities} selected={selected} setSelected={setSelected} />}
      {page === 'about' && <AboutPage risk={risk} selected={selected} />}
      {page === 'profile' && <ProfilePage profile={profile} session={session} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root — authentication gate
// ---------------------------------------------------------------------------
export default function App() {
  const { session, loading } = useAuth();
  const [localities, setLocalities] = useState([]);
  const [localitiesError, setLocalitiesError] = useState('');

  useEffect(() => {
    getJson(`${API}/localities`)
      .then(data => setLocalities(data.localities || []))
      .catch(err => setLocalitiesError(err.message));
  }, []);

  if (loading) {
    return (
      <div className="authPage">
        <div className="authLayout authLayoutCompact">
          <div className="authIdentity">
            <img className="authHeroLogo" src={CFS_LOGO} alt="Chennai FloodSense AI logo" />
            <div className="authHeroName">Chennai <b>FloodSense AI</b></div>
            <p>AI-powered locality-level flood monitoring for Chennai.</p>
          </div>
          <div className="authCard">
            <div className="loadingBox">Checking session…</div>
          </div>
        </div>
      </div>
    );
  }

  if (!session) return <AuthPage localities={localities} />;
  return <AppShell localities={localities} />;
}

// Wrap the export with AuthProvider at module level
const WrappedApp = () => <AuthProvider><App /></AuthProvider>;
export { WrappedApp };
