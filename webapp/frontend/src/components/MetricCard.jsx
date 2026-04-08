export default function MetricCard({ title, value, note, emphasis = false }) {
  return (
    <article className={`metric-card ${emphasis ? "emphasis" : ""}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}
