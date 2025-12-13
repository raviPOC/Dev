namespace StockDrlPredictor;

internal sealed class FeatureScaler
{
    private readonly float[] _mean;
    private readonly float[] _invStd;

    private FeatureScaler(float[] mean, float[] invStd)
    {
        _mean = mean;
        _invStd = invStd;
    }

    public static FeatureScaler Fit(CsvNumericTable train, int targetCol, bool includeTargetInState)
    {
        var use = new bool[train.NumericColumnCount];
        for (int c = 0; c < train.NumericColumnCount; c++)
        {
            if (c == targetCol && !includeTargetInState) continue;
            use[c] = true;
        }

        var mean = new float[train.NumericColumnCount];
        var m2 = new float[train.NumericColumnCount];
        var n = new int[train.NumericColumnCount];

        // Welford
        foreach (var row in train.Rows)
        {
            for (int c = 0; c < train.NumericColumnCount; c++)
            {
                if (!use[c]) continue;
                var x = row[c];
                if (float.IsNaN(x) || float.IsInfinity(x)) continue;

                n[c]++;
                var delta = x - mean[c];
                mean[c] += delta / n[c];
                var delta2 = x - mean[c];
                m2[c] += delta * delta2;
            }
        }

        var invStd = new float[train.NumericColumnCount];
        for (int c = 0; c < train.NumericColumnCount; c++)
        {
            if (!use[c])
            {
                mean[c] = 0;
                invStd[c] = 1;
                continue;
            }

            var var = (n[c] > 1) ? (m2[c] / (n[c] - 1)) : 0;
            var std = MathF.Sqrt(MathF.Max(1e-12f, var));
            invStd[c] = 1.0f / std;
        }

        return new FeatureScaler(mean, invStd);
    }

    public float Transform(int numericColumnIndex, float value)
    {
        return (value - _mean[numericColumnIndex]) * _invStd[numericColumnIndex];
    }
}
