import { useEffect, useState } from "react";

export type QoEControlValues = {
  rebufPenalty: number;
  smoothPenalty: number;
};

type QoEControlsProps = {
  onChange: (values: QoEControlValues) => void;
};

const DEFAULT_VALUES: QoEControlValues = {
  rebufPenalty: 5.5,
  smoothPenalty: 0.8,
};

export default function QoEControls({ onChange }: QoEControlsProps) {
  const [rebufPenalty, setRebufPenalty] = useState(DEFAULT_VALUES.rebufPenalty);
  const [smoothPenalty, setSmoothPenalty] = useState(DEFAULT_VALUES.smoothPenalty);

  useEffect(() => {
    onChange({ rebufPenalty, smoothPenalty });
  }, [onChange, rebufPenalty, smoothPenalty]);

  const handleReset = () => {
    setRebufPenalty(DEFAULT_VALUES.rebufPenalty);
    setSmoothPenalty(DEFAULT_VALUES.smoothPenalty);
  };

  return (
    <div className="qoe-controls">
      <div className="qoe-controls__header">
        <span className="qoe-controls__title">QoE CONTROLS</span>
      </div>

      <div className="qoe-controls__body">
        <div className="qoe-controls__group">
          <div className="qoe-controls__label-row">
            <label htmlFor="rebuf-penalty">Rebuffering Penalty (β_r)</label>
            <strong>{`β_r: ${rebufPenalty.toFixed(1)}`}</strong>
          </div>
          <input
            id="rebuf-penalty"
            className="qoe-controls__slider"
            type="range"
            min="3"
            max="10"
            step="0.1"
            value={rebufPenalty}
            onChange={(event) => setRebufPenalty(Number(event.target.value))}
          />
          <p>Rebuffering Penalty (β_r) - Higher = less rebuffering</p>
        </div>

        <div className="qoe-controls__group">
          <div className="qoe-controls__label-row">
            <label htmlFor="smooth-penalty">Smoothness Penalty (β_s)</label>
            <strong>{`β_s: ${smoothPenalty.toFixed(1)}`}</strong>
          </div>
          <input
            id="smooth-penalty"
            className="qoe-controls__slider"
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={smoothPenalty}
            onChange={(event) => setSmoothPenalty(Number(event.target.value))}
          />
          <p>Smoothness Penalty (β_s) - Higher = more stable quality</p>
        </div>

        <button className="qoe-controls__reset" type="button" onClick={handleReset}>
          Reset
        </button>
      </div>
    </div>
  );
}
