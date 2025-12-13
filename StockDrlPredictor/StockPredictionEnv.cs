namespace StockDrlPredictor;

/// <summary>
/// Environment that turns a time-indexed table into an RL-like stream.
/// State: lookback window of numeric features (optionally including the target column).
/// Action: predicted next-step return (percentage change).
/// Reward is computed outside the env (based on prediction error).
/// </summary>
internal sealed class StockPredictionEnv
{
    private readonly CsvNumericTable _data;
    private readonly int _targetCol;
    private readonly int _lookback;
    private readonly bool _includeTargetInState;
    private readonly FeatureScaler _scaler;

    // Derived
    private readonly int[] _featureCols;

    private int _t; // points to current row index

    public int StateSize { get; }

    public bool Done { get; private set; }

    /// <summary>Index within the original (split) table rows.</summary>
    public int GlobalIndex => _t;

    public StockPredictionEnv(CsvNumericTable data, int targetCol, int lookback, bool includeTargetInState, FeatureScaler scaler)
    {
        _data = data;
        _targetCol = targetCol;
        _lookback = Math.Clamp(lookback, 1, 512);
        _includeTargetInState = includeTargetInState;
        _scaler = scaler;

        var cols = new List<int>();
        for (int c = 0; c < data.NumericColumnCount; c++)
        {
            if (c == targetCol && !includeTargetInState) continue;
            cols.Add(c);
        }

        _featureCols = cols.ToArray();
        StateSize = _featureCols.Length * _lookback;

        Reset();
    }

    public void Reset()
    {
        _t = _lookback - 1;
        Done = _data.RowCount < _lookback + 1;
    }

    public (float[] State, float CurrentClose, float NextClose) Observe()
    {
        if (Done) throw new InvalidOperationException("Environment is done.");
        if (_t < _lookback - 1) throw new InvalidOperationException("Invalid internal state.");
        if (_t >= _data.RowCount - 1) throw new InvalidOperationException("No next step available.");

        var state = new float[StateSize];
        int k = 0;

        for (int i = _t - (_lookback - 1); i <= _t; i++)
        {
            var row = _data.Rows[i];
            for (int j = 0; j < _featureCols.Length; j++)
            {
                var v = row[_featureCols[j]];
                if (float.IsNaN(v) || float.IsInfinity(v)) v = 0;
                state[k++] = _scaler.Transform(_featureCols[j], v);
            }
        }

        var cur = _data.Rows[_t][_targetCol];
        var nxt = _data.Rows[_t + 1][_targetCol];
        if (float.IsNaN(cur) || float.IsNaN(nxt))
            throw new InvalidOperationException("Target column has NaN values near current timestep.");

        return (state, cur, nxt);
    }

    public void Step()
    {
        if (Done) return;
        _t++;
        if (_t >= _data.RowCount - 1) Done = true;
    }
}
