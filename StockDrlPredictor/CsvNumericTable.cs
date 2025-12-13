using System.Globalization;

namespace StockDrlPredictor;

internal sealed class CsvNumericTable
{
    public required string[] Headers { get; init; }

    // Only numeric columns are kept here (including the target if it's numeric)
    public required int[] NumericColumnIndices { get; init; }
    public required string[] NumericColumnNames { get; init; }

    // Rows: [row][numericCol]
    public required float[][] Rows { get; init; }

    public int RowCount => Rows.Length;
    public int NumericColumnCount => NumericColumnIndices.Length;

    public static CsvNumericTable Load(string path)
    {
        using var sr = new StreamReader(path);
        var headerLine = sr.ReadLine();
        if (headerLine is null)
            throw new InvalidOperationException("CSV is empty.");

        var headers = ParseCsvLine(headerLine).ToArray();
        if (headers.Length == 0)
            throw new InvalidOperationException("CSV has no headers.");

        // First pass: collect raw rows and infer numeric columns.
        var raw = new List<string[]>();
        while (!sr.EndOfStream)
        {
            var line = sr.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;
            var fields = ParseCsvLine(line).ToArray();
            if (fields.Length == 0) continue;

            // pad / trim
            if (fields.Length != headers.Length)
            {
                Array.Resize(ref fields, headers.Length);
            }

            raw.Add(fields);
        }

        if (raw.Count < 3)
            throw new InvalidOperationException("CSV has too few rows to train.");

        var isNumeric = new bool[headers.Length];
        Array.Fill(isNumeric, true);

        for (int c = 0; c < headers.Length; c++)
        {
            int checkedRows = 0;
            int numericRows = 0;
            for (int r = 0; r < raw.Count && checkedRows < 50; r++)
            {
                var s = raw[r][c];
                if (string.IsNullOrWhiteSpace(s)) continue;
                checkedRows++;
                if (float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out _)) numericRows++;
            }

            // consider numeric if we saw at least a few parsable numbers
            isNumeric[c] = numericRows >= Math.Max(2, checkedRows / 2);
        }

        var numericIdx = new List<int>();
        var numericNames = new List<string>();
        for (int c = 0; c < headers.Length; c++)
        {
            if (!isNumeric[c]) continue;
            numericIdx.Add(c);
            numericNames.Add(headers[c]);
        }

        if (numericIdx.Count == 0)
            throw new InvalidOperationException("No numeric columns found in CSV.");

        var rows = new float[raw.Count][];
        for (int r = 0; r < raw.Count; r++)
        {
            var vec = new float[numericIdx.Count];
            for (int j = 0; j < numericIdx.Count; j++)
            {
                var s = raw[r][numericIdx[j]];
                if (!float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    v = float.NaN;
                vec[j] = v;
            }
            rows[r] = vec;
        }

        return new CsvNumericTable
        {
            Headers = headers,
            NumericColumnIndices = numericIdx.ToArray(),
            NumericColumnNames = numericNames.ToArray(),
            Rows = rows,
        };
    }

    public int ResolveTargetColumn(string targetName)
    {
        for (int i = 0; i < NumericColumnNames.Length; i++)
        {
            if (string.Equals(NumericColumnNames[i], targetName, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        // fallback: last numeric column
        return NumericColumnCount - 1;
    }

    public (CsvNumericTable Train, CsvNumericTable Test) SequentialTrainTestSplit(float trainFraction)
    {
        trainFraction = Math.Clamp(trainFraction, 0.5f, 0.95f);
        var cut = (int)MathF.Floor(RowCount * trainFraction);
        cut = Math.Clamp(cut, 2, RowCount - 2);

        CsvNumericTable Slice(int start, int count)
        {
            var sliced = new float[count][];
            Array.Copy(Rows, start, sliced, 0, count);
            return new CsvNumericTable
            {
                Headers = Headers,
                NumericColumnIndices = NumericColumnIndices,
                NumericColumnNames = NumericColumnNames,
                Rows = sliced
            };
        }

        return (Slice(0, cut), Slice(cut, RowCount - cut));
    }

    private static IEnumerable<string> ParseCsvLine(string line)
    {
        // Minimal CSV parser that supports quoted fields.
        if (line.Length == 0) yield break;

        var sb = new System.Text.StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            var ch = line[i];
            if (inQuotes)
            {
                if (ch == '"')
                {
                    // escaped quote
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    sb.Append(ch);
                }
            }
            else
            {
                if (ch == ',')
                {
                    yield return sb.ToString();
                    sb.Clear();
                }
                else if (ch == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    sb.Append(ch);
                }
            }
        }

        yield return sb.ToString();
    }
}
