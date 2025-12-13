namespace StockDrlPredictor;

internal sealed class Mlp
{
    private readonly List<ILayer> _layers = new();
    private readonly Adam _adam;

    public Mlp(int inputSize, int[] hiddenSizes, int outputSize, int seed)
    {
        var rng = new Random(seed);

        int inSize = inputSize;
        foreach (var h in hiddenSizes)
        {
            _layers.Add(new Dense(inSize, h, rng));
            _layers.Add(new Relu());
            inSize = h;
        }

        _layers.Add(new Dense(inSize, outputSize, rng));

        _adam = new Adam(1e-8f, 0.9f, 0.999f);
    }

    public float[] Forward(float[] input)
    {
        float[] x = input;
        foreach (var layer in _layers)
            x = layer.Forward(x);
        return x;
    }

    public void Backward(float[] gradOutput)
    {
        float[] g = gradOutput;
        for (int i = _layers.Count - 1; i >= 0; i--)
            g = _layers[i].Backward(g);
    }

    public void ZeroGrad()
    {
        foreach (var layer in _layers)
            layer.ZeroGrad();
    }

    public void AdamStep(float lr)
    {
        foreach (var layer in _layers)
            layer.Step(_adam, lr);
    }

    private interface ILayer
    {
        float[] Forward(float[] input);
        float[] Backward(float[] gradOutput);
        void ZeroGrad();
        void Step(Adam adam, float lr);
    }

    private sealed class Dense : ILayer
    {
        private readonly int _in;
        private readonly int _out;

        private float[] _w; // [out * in]
        private float[] _b; // [out]

        private float[] _gw;
        private float[] _gb;

        private float[] _mw;
        private float[] _vw;
        private float[] _mb;
        private float[] _vb;

        private int _t;

        private float[]? _lastX;

        public Dense(int inputSize, int outputSize, Random rng)
        {
            _in = inputSize;
            _out = outputSize;

            _w = new float[_out * _in];
            _b = new float[_out];

            _gw = new float[_w.Length];
            _gb = new float[_b.Length];

            _mw = new float[_w.Length];
            _vw = new float[_w.Length];
            _mb = new float[_b.Length];
            _vb = new float[_b.Length];

            // He init for ReLU
            var scale = MathF.Sqrt(2.0f / Math.Max(1, _in));
            for (int i = 0; i < _w.Length; i++)
                _w[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        }

        public float[] Forward(float[] input)
        {
            _lastX = input;
            var y = new float[_out];

            for (int o = 0; o < _out; o++)
            {
                float sum = _b[o];
                int wOff = o * _in;
                for (int i = 0; i < _in; i++)
                    sum += _w[wOff + i] * input[i];
                y[o] = sum;
            }

            return y;
        }

        public float[] Backward(float[] gradOutput)
        {
            if (_lastX is null) throw new InvalidOperationException("Forward must be called before Backward.");

            var gradIn = new float[_in];

            for (int o = 0; o < _out; o++)
            {
                var go = gradOutput[o];
                _gb[o] += go;

                int wOff = o * _in;
                for (int i = 0; i < _in; i++)
                {
                    _gw[wOff + i] += go * _lastX[i];
                    gradIn[i] += _w[wOff + i] * go;
                }
            }

            return gradIn;
        }

        public void ZeroGrad()
        {
            Array.Clear(_gw);
            Array.Clear(_gb);
        }

        public void Step(Adam adam, float lr)
        {
            _t++;
            adam.UpdateInPlace(_w, _gw, _mw, _vw, lr, _t);
            adam.UpdateInPlace(_b, _gb, _mb, _vb, lr, _t);
        }
    }

    private sealed class Relu : ILayer
    {
        private float[]? _lastX;

        public float[] Forward(float[] input)
        {
            _lastX = input;
            var y = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
                y[i] = input[i] > 0 ? input[i] : 0;
            return y;
        }

        public float[] Backward(float[] gradOutput)
        {
            if (_lastX is null) throw new InvalidOperationException("Forward must be called before Backward.");
            var g = new float[gradOutput.Length];
            for (int i = 0; i < gradOutput.Length; i++)
                g[i] = _lastX[i] > 0 ? gradOutput[i] : 0;
            return g;
        }

        public void ZeroGrad() { }
        public void Step(Adam adam, float lr) { }
    }

    private sealed class Adam
    {
        private readonly float _eps;
        private readonly float _b1;
        private readonly float _b2;

        public Adam(float eps, float beta1, float beta2)
        {
            _eps = eps;
            _b1 = beta1;
            _b2 = beta2;
        }

        public void UpdateInPlace(float[] param, float[] grad, float[] m, float[] v, float lr, int t)
        {
            var b1t = Pow(_b1, t);
            var b2t = Pow(_b2, t);

            for (int i = 0; i < param.Length; i++)
            {
                var g = grad[i];
                m[i] = _b1 * m[i] + (1 - _b1) * g;
                v[i] = _b2 * v[i] + (1 - _b2) * g * g;

                var mHat = m[i] / (1 - b1t);
                var vHat = v[i] / (1 - b2t);

                param[i] -= lr * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }

        private static float Pow(float x, int n)
        {
            // fast-ish pow for integers
            float r = 1;
            float a = x;
            int k = n;
            while (k > 0)
            {
                if ((k & 1) == 1) r *= a;
                a *= a;
                k >>= 1;
            }
            return r;
        }
    }
}
