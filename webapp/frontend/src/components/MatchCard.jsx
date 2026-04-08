export default function MatchCard({ match, active, onSelect }) {
  return (
    <article className={`match-card ${active ? "active" : ""}`} onClick={onSelect}>
      <div className="match-thumb" style={{ background: match.poster }}>
        <strong>{match.league}</strong>
      </div>
      <div className="match-body">
        <h4>{match.title}</h4>
        <div className="meta-row"><span>{match.time}</span><span>{match.venue}</span></div>
        <div className="meta-row"><span>القنوات</span><span>{match.channels.join(" • ")}</span></div>
      </div>
    </article>
  );
}
