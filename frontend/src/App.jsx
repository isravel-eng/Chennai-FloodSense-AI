import React, { useEffect, useMemo, useState } from 'react';
import { MapContainer, Marker, TileLayer, Tooltip as LeafletTooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
const chennai = [13.0827, 80.2707];
const markerIcon = new L.Icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png', iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png', shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png', iconSize: [25, 41], iconAnchor: [12, 41] });

async function getJson(url) {
  const response = await fetch(url);
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw new Error(payload?.detail || payload?.message || `Request failed (${response.status})`);
  return payload;
}

function MapPage({ localities, selected, setSelected, risk, loading, error, onRefresh }) {
  const [panelOpen, setPanelOpen] = useState(false);
  useEffect(() => { if (selected) setPanelOpen(true); }, [selected]);
  const chart7 = useMemo(() => (risk?.next_7_days || []).map(d => ({ ...d, shortDate: String(d.date).slice(5) })), [risk]);
  const riskClass = String(risk?.next_24h?.risk_band || 'low').toLowerCase();

  return <main className="mapPage">
    <MapContainer center={chennai} zoom={11} scrollWheelZoom className="fullMap">
      <MapResize />
      <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {localities.map(x => <Marker key={x.name} position={[x.latitude, x.longitude]} icon={markerIcon} eventHandlers={{ click: () => setSelected(x.name) }}>
        <LeafletTooltip permanent direction="top" offset={[0, -38]} className="localityLabel">{x.name}</LeafletTooltip>
      </Marker>)}
    </MapContainer>

    <div className="mapOverlay mapTitle"><div className="eyebrow">LIVE FLOOD MONITORING</div><h1>Chennai FloodSense AI</h1><p>Click a locality marker to inspect its current risk and forecast.</p></div>

    <div className={`riskDrawer ${panelOpen ? 'open' : ''}`}>
      <button className="drawerClose" onClick={() => setPanelOpen(false)} aria-label="Close locality details">×</button>
      <div className="panelLabel">SELECTED LOCALITY</div><h2>{selected || 'Select a locality on the map'}</h2>
      {loading && <div className="loadingBox">Loading prediction…</div>}
      {!loading && error && <div className="errorBox">{error}</div>}
      {!loading && !error && risk && <>
        <div className={`riskHero ${riskClass}`}><span className="riskDot">●</span><div><small>Next 24-hour flood risk</small><strong>{risk.next_24h?.risk_band || 'Unknown'}</strong></div><b>{Math.round((risk.next_24h?.probability || 0) * 100)}%</b></div>
        <div className="stats"><Stat label="Current risk" value={risk.current?.risk_band || '—'} /><Stat label="Forecast rainfall" value={`${risk.next_24h?.forecast_rainfall_mm ?? 0} mm`} /><Stat label="Last 7 days" value={`${risk.context?.rainfall_last_7d_mm ?? 0} mm`} /><Stat label="Last 30 days" value={`${risk.context?.rainfall_last_30d_mm ?? 0} mm`} /></div>
        <div className="chartCard compactChart"><div className="cardTitle">Next 7 Days · Rainfall</div><ResponsiveContainer width="100%" height={230}><BarChart data={chart7}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="shortDate" /><YAxis /><Tooltip /><Bar dataKey="rainfall_mm" name="Rainfall (mm)" fill="#1976d2" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer><div className="forecastRows">{chart7.map(d => <div className="forecastRow" key={d.date}><span>{d.date}</span><b>{d.rainfall_mm} mm</b><em className={String(d.risk_band).toLowerCase()}>{d.risk_band}</em></div>)}</div></div>
        <button className="primary fullButton" onClick={onRefresh}>{loading ? 'Updating…' : 'Refresh prediction'}</button>
      </>}
    </div>
  </main>;
}

function MapResize() { const map = useMap(); useEffect(() => { const timer = setTimeout(() => map.invalidateSize(), 50); return () => clearTimeout(timer); }, [map]); return null; }
function Stat({ label, value }) { return <div className="stat"><small>{label}</small><b>{value}</b></div>; }

function LocalitiesPage({ localities, onOpen }) {
  const [query, setQuery] = useState(''); const [sort, setSort] = useState('name'); const [direction, setDirection] = useState('asc'); const [predictions, setPredictions] = useState({}); const [loading, setLoading] = useState(true); const [message, setMessage] = useState('');
  useEffect(() => { let active = true; setLoading(true); getJson(`${API}/flood-risk-all`).then(data => { if (!active) return; const next = {}; (data?.results || []).forEach(item => { if (item.ok) next[item.name] = item.data; }); setPredictions(next); }).catch(err => active && setMessage(err.message)).finally(() => active && setLoading(false)); return () => { active = false; }; }, [localities]);
  const rows = useMemo(() => { const filtered = localities.filter(x => x.name.toLowerCase().includes(query.toLowerCase())); return [...filtered].sort((a,b) => { const av = sort === 'name' ? a.name : sort === 'risk' ? (predictions[a.name]?.next_24h?.probability || 0) : (predictions[a.name]?.next_24h?.forecast_rainfall_mm || 0); const bv = sort === 'name' ? b.name : sort === 'risk' ? (predictions[b.name]?.next_24h?.probability || 0) : (predictions[b.name]?.next_24h?.forecast_rainfall_mm || 0); const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv; return direction === 'asc' ? cmp : -cmp; }); }, [localities, query, sort, direction, predictions]);
  function changeSort(value) { if (value === sort) setDirection(x => x === 'asc' ? 'desc' : 'asc'); else { setSort(value); setDirection('asc'); } }
  return <main className="page"><div className="eyebrow">LOCALITY ANALYSIS</div><h1>All Project Localities</h1><p className="subtitle">Live flood-risk and rainfall prediction for every configured locality.</p><div className="toolbar localityToolbar"><input value={query} placeholder="Search locality…" onChange={e => setQuery(e.target.value)} /><select value={sort} onChange={e => changeSort(e.target.value)}><option value="name">Sort: Locality</option><option value="risk">Sort: Risk probability</option><option value="rainfall">Sort: Forecast rainfall</option></select><span className="sortHint">Choose the same sort again to reverse order</span></div>{message && <div className="notice">{message}</div>}<div className="table"><div className="thead localityGrid"><span>LOCALITY</span><span>24H RISK</span><span>PROBABILITY</span><span>RAINFALL</span><span>ACTION</span></div>{loading && <div className="tableMessage">Loading all locality predictions…</div>}{!loading && rows.map(x => { const p = predictions[x.name]; return <button className="tr localityGrid" key={x.name} onClick={() => onOpen(x.name)}><span><b>{x.name}</b><small>{x.latitude.toFixed(4)}, {x.longitude.toFixed(4)}</small></span><span className={String(p?.next_24h?.risk_band || '—').toLowerCase()}>{p?.next_24h?.risk_band || '—'}</span><span>{p ? `${Math.round((p.next_24h.probability || 0) * 100)}%` : '—'}</span><span>{p ? `${p.next_24h.forecast_rainfall_mm} mm` : '—'}</span><span>View →</span></button>; })}</div></main>;
}

function RainfallPage({ localities, selected, setSelected }) {
  const [months, setMonths] = useState(12); const [data, setData] = useState(null); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  async function generate() { if (!selected) return; setLoading(true); setError(''); try { setData(await getJson(`${API}/rainfall-forecast/locality/${encodeURIComponent(selected)}?months=${months}`)); } catch (err) { setData(null); setError(err.message); } finally { setLoading(false); } }
  useEffect(() => { if (selected) generate(); }, [selected]);
  const chart = useMemo(() => data?.forecast || [], [data]);
  return <main className="page"><div className="eyebrow">RAINFALL FORECASTING</div><h1>Rainfall Prediction</h1><p className="subtitle">Locality-specific rainfall forecasting with a clear trend graph.</p><div className="controlCard"><div><label>LOCALITY</label><select value={selected} onChange={e => setSelected(e.target.value)}>{localities.map(x => <option key={x.name} value={x.name}>{x.name}</option>)}</select></div><div><label>HORIZON</label><select value={months} onChange={e => setMonths(Number(e.target.value))}><option value={12}>12 months</option><option value={24}>24 months</option><option value={36}>36 months</option></select></div><button className="primary" onClick={generate} disabled={loading}>{loading ? 'Generating…' : 'Generate Forecast'}</button></div>{error && <div className="errorBox">{error}</div>}{data?.status === 'fallback_forecast' && <div className="notice">Limited history: {data.observed_months} months. A non-seasonal rainfall forecast is being shown instead of a 24-month seasonal SARIMA model.</div>}{data?.status === 'model_fit_failed' && <div className="notice">{data.message}</div>}{chart.length > 0 && <><div className="chartCard"><div className="cardTitle">{data?.model?.name || 'Rainfall'} Forecast · {selected} · {months} months</div><ResponsiveContainer width="100%" height={380}><LineChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="month" interval={months > 18 ? 2 : 0} /><YAxis /><Tooltip /><Legend /><Line type="monotone" dataKey="forecast_mm" name="Forecast (mm)" stroke="#1976d2" strokeWidth={3} dot={false} /><Line type="monotone" dataKey="lower_95_mm" name="Lower 95%" stroke="#90a4ae" dot={false} /><Line type="monotone" dataKey="upper_95_mm" name="Upper 95%" stroke="#90a4ae" dot={false} /></LineChart></ResponsiveContainer></div><div className="table"><div className="thead"><span>MONTH</span><span>FORECAST</span><span>LOWER 95%</span><span>UPPER 95%</span></div>{chart.map(x => <div className="tr static" key={x.month}><span>{x.month}</span><span>{x.forecast_mm} mm</span><span>{x.lower_95_mm} mm</span><span>{x.upper_95_mm} mm</span></div>)}</div></>}</main>;
}

export default function App() {
  const [page, setPage] = useState('map'); const [localities, setLocalities] = useState([]); const [selected, setSelected] = useState(''); const [risk, setRisk] = useState(null); const [loadingRisk, setLoadingRisk] = useState(false); const [riskError, setRiskError] = useState('');
  useEffect(() => { getJson(`${API}/localities`).then(data => setLocalities(data.localities || [])).catch(err => setRiskError(err.message)); }, []);
  useEffect(() => { if (page === 'map' && selected) loadRisk(selected); }, [selected, page]);
  async function loadRisk(name) { setLoadingRisk(true); setRiskError(''); try { setRisk(await getJson(`${API}/flood-risk/${encodeURIComponent(name)}`)); } catch (err) { setRisk(null); setRiskError(err.message); } finally { setLoadingRisk(false); } }
  return <div className="app"><header className="topbar"><div className="brand"><span className="brandMark">⌁</span><span>Chennai <b>FloodSense AI</b></span></div><nav>{[['map','Map'],['localities','Localities'],['rainfall','Rainfall Prediction']].map(([id,label]) => <button key={id} className={page===id?'active':''} onClick={() => setPage(id)}>{label}</button>)}</nav></header>{page==='map' && <MapPage localities={localities} selected={selected} setSelected={setSelected} risk={risk} loading={loadingRisk} error={riskError} onRefresh={() => selected && loadRisk(selected)} />}{page==='localities' && <LocalitiesPage localities={localities} onOpen={name => { setSelected(name); setPage('map'); }} />}{page==='rainfall' && <RainfallPage localities={localities} selected={selected} setSelected={setSelected} />}</div>;
}
