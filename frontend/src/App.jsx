import { useEffect, useMemo, useState } from 'react';
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
const chennai = [13.0827, 80.2707];
const markerIcon = new L.Icon({ iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png', iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png', shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png', iconSize: [25,41], iconAnchor: [12,41] });

function App() {
  const [page, setPage] = useState('map');
  const [localities, setLocalities] = useState([]);
  const [selected, setSelected] = useState('');
  const [risk, setRisk] = useState(null);
  const [longTerm, setLongTerm] = useState(null);
  const [loading, setLoading] = useState(false);
  const [months, setMonths] = useState(12);

  useEffect(() => { fetch(`${API}/localities`).then(r => r.json()).then(d => { setLocalities(d.localities || []); if (!selected && d.localities?.length) setSelected(d.localities[0].name); }).catch(() => {}); }, []);
  useEffect(() => { if (selected && page === 'map') loadRisk(selected); }, [selected, page]);

  async function loadRisk(name) {
    setLoading(true); try { const r = await fetch(`${API}/flood-risk/${encodeURIComponent(name)}`); setRisk(await r.json()); } catch { setRisk({ error: 'Backend unavailable' }); } finally { setLoading(false); }
  }
  async function loadLongTerm() {
    if (!selected) return; setLoading(true); try { const r = await fetch(`${API}/rainfall-forecast/locality/${encodeURIComponent(selected)}?months=${months}`); setLongTerm(await r.json()); } catch { setLongTerm({ status: 'error', message: 'Backend unavailable' }); } finally { setLoading(false); }
  }

  return <div className="app">
    <header className="topbar"><div className="brand"><span className="brandMark">⌁</span><span>Chennai <b>FloodSense AI</b></span></div><nav>{[['map','Map'],['localities','Localities'],['rainfall','Rainfall Prediction']].map(([id,label]) => <button key={id} className={page===id?'active':''} onClick={()=>setPage(id)}>{label}</button>)}</nav></header>
    {page==='map' && <MapPage localities={localities} selected={selected} setSelected={setSelected} risk={risk} loading={loading} onRefresh={()=>loadRisk(selected)} />}
    {page==='localities' && <LocalitiesPage localities={localities} selected={selected} setSelected={setSelected} onOpen={(n)=>{setSelected(n);setPage('map')}} />}
    {page==='rainfall' && <RainfallPage localities={localities} selected={selected} setSelected={setSelected} months={months} setMonths={setMonths} data={longTerm} loading={loading} onGenerate={loadLongTerm} />}
  </div>
}

function MapPage({ localities, selected, setSelected, risk, loading, onRefresh }) {
  return <main className="page"><div className="eyebrow">PROJECT LOCALITIES MAP</div><h1>Chennai FloodSense AI</h1><p className="subtitle">Interactive locality-level flood-risk monitoring</p><div className="mapLayout"><div className="mapCard"><MapContainer center={chennai} zoom={11} scrollWheelZoom className="map"><TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{localities.map(x=><Marker key={x.name} position={[x.latitude,x.longitude]} icon={markerIcon} eventHandlers={{click:()=>setSelected(x.name)}}><Popup>{x.name}</Popup></Marker>)}</MapContainer></div><aside className="riskPanel"><div className="panelLabel">SELECTED LOCALITY</div><h2>{selected || 'Select a locality'}</h2>{risk?.error ? <div className="error">{risk.error}</div> : risk && <><div className={`riskHero ${risk.current.risk_band.toLowerCase()}`}><span>●</span><div><small>Next 24-hour flood risk</small><strong>{risk.next_24h.risk_band}</strong></div></div><div className="stats"><Stat label="Current risk" value={risk.current.risk_band}/><Stat label="Probability" value={`${(risk.next_24h.probability*100).toFixed(0)}%`}/><Stat label="Forecast rainfall" value={`${risk.next_24h.forecast_rainfall_mm} mm`}/><Stat label="Last 7 days" value={`${risk.context.rainfall_last_7d_mm} mm`}/></div><div className="dailyMini"><h3>Next 7 days</h3>{risk.next_7_days?.map(d=><div className="dayRow" key={d.date}><span>{d.date}</span><b>{d.rainfall_mm} mm</b><em className={d.risk_band.toLowerCase()}>{d.risk_band}</em></div>)}</div><button className="primary" onClick={onRefresh}>{loading?'Updating…':'Refresh prediction'}</button></>}</aside></div></main>
}
function Stat({label,value}){return <div className="stat"><small>{label}</small><b>{value}</b></div>}
function LocalitiesPage({localities,selected,setSelected,onOpen}){return <main className="page"><div className="eyebrow">LOCALITY ANALYSIS</div><h1>All Project Localities</h1><p className="subtitle">Search and inspect configured Chennai localities.</p><div className="toolbar"><input placeholder="Search locality…" onChange={e=>setSelected(e.target.value)}/></div><div className="table"><div className="thead"><span>LOCALITY</span><span>LATITUDE</span><span>LONGITUDE</span><span>ELEVATION</span></div>{localities.filter(x=>x.name.toLowerCase().includes((selected||'').toLowerCase())).map(x=><button className="tr" key={x.name} onClick={()=>onOpen(x.name)}><span>{x.name}</span><span>{x.latitude}</span><span>{x.longitude}</span><span>{x.elevation_m_approx} m</span></button>)}</div></main>}
function RainfallPage({localities,selected,setSelected,months,setMonths,data,loading,onGenerate}){const chart=useMemo(()=>data?.forecast||[],[data]);return <main className="page"><div className="eyebrow">RAINFALL FORECASTING</div><h1>Rainfall Prediction</h1><p className="subtitle">Daily live forecasting and locality-wise long-term SARIMA forecasting.</p><div className="controlCard"><div><label>LOCALITY</label><select value={selected} onChange={e=>setSelected(e.target.value)}>{localities.map(x=><option key={x.name}>{x.name}</option>)}</select></div><div><label>LONG-TERM HORIZON</label><select value={months} onChange={e=>setMonths(Number(e.target.value))}><option value={12}>12 months</option><option value={24}>24 months</option><option value={36}>36 months</option></select></div><button className="primary" onClick={onGenerate}>{loading?'Generating…':'Generate Forecast'}</button></div>{data?.status==='insufficient_historical_data'?<div className="notice">{data.message}</div>:chart.length>0&&<><div className="chartCard"><div className="cardTitle">Model 1 · SARIMA — {selected}</div><ResponsiveContainer width="100%" height={360}><LineChart data={chart}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="month"/><YAxis/><Tooltip/><Line type="monotone" dataKey="forecast_mm" stroke="#1976d2" strokeWidth={3} dot={false}/><Line type="monotone" dataKey="lower_95_mm" stroke="#90a4ae" dot={false}/><Line type="monotone" dataKey="upper_95_mm" stroke="#90a4ae" dot={false}/></LineChart></ResponsiveContainer></div><div className="table"><div className="thead"><span>MONTH</span><span>FORECAST</span><span>LOWER 95%</span><span>UPPER 95%</span></div>{chart.slice(0,12).map(x=><div className="tr static" key={x.month}><span>{x.month}</span><span>{x.forecast_mm} mm</span><span>{x.lower_95_mm} mm</span><span>{x.upper_95_mm} mm</span></div>)}</div></>}</main>}

export default App;
