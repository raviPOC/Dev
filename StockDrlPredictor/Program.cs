using System.Globalization;

namespace StockDrlPredictor;

internal static class Program
{
    private static int Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

        var opt = Options.Parse(args);

        if (!File.Exists(opt.CsvPath))
        {
            Console.Error.WriteLine($"CSV not found: '{opt.CsvPath}'");
            Console.Error.WriteLine("Tip: pass --csv /full/path/to/ten.csv (or place it in ~/Downloads/ten.csv)");
            return 2;
        }

        var data = CsvNumericTable.Load(opt.CsvPath);
        var targetCol = data.ResolveTargetColumn(opt.TargetColumn);

        var split = data.SequentialTrainTestSplit(opt.TrainFraction);
        var train = split.Train;
        var test = split.Test;

        var includeTargetInState = true;
        var scaler = FeatureScaler.Fit(train, targetCol, includeTargetInState);

        var envTrain = new StockPredictionEnv(train, targetCol, opt.Lookback, includeTargetInState, scaler);
        var envTest = new StockPredictionEnv(test, targetCol, opt.Lookback, includeTargetInState, scaler);

        var stateSize = envTrain.StateSize;

        var actor = new Mlp(
            inputSize: stateSize,
            hiddenSizes: new[] { opt.Hidden, opt.Hidden },
            outputSize: 2,
            seed: opt.Seed);

        var critic = new Mlp(
            inputSize: stateSize,
            hiddenSizes: new[] { opt.Hidden, opt.Hidden },
            outputSize: 1,
            seed: opt.Seed + 1);

        var rng = new Random(opt.Seed);

        for (var epoch = 1; epoch <= opt.Epochs; epoch++)
        {
            envTrain.Reset();
            actor.ZeroGrad();
            critic.ZeroGrad();

            double sumReward = 0;
            double sumAbsPctErr = 0;
            int steps = 0;

            while (!envTrain.Done)
            {
                var (state, currentClose, nextClose) = envTrain.Observe();

                // Actor: output mean & logStd of return (percentage change)
                var aOut = actor.Forward(state);
                var mean = aOut[0];
                var logStd = Math.Clamp(aOut[1], opt.MinLogStd, opt.MaxLogStd);
                var std = MathF.Exp(logStd);

                var action = mean + std * (float)Randn(rng);
                action = Math.Clamp(action, -opt.MaxAbsReturn, opt.MaxAbsReturn);

                var predicted = currentClose * (1.0f + action);

                var absPctErr = Math.Abs((predicted - nextClose) / Math.Max(1e-6f, currentClose));
                var reward = -(float)absPctErr;

                // Critic target
                var v = critic.Forward(state)[0];

                envTrain.Step();
                float vNext = 0;
                if (!envTrain.Done)
                {
                    var (nextState, _, _) = envTrain.Observe();
                    vNext = critic.Forward(nextState)[0];
                }

                var target = reward + opt.Gamma * vNext;
                var advantage = target - v;

                // --- Critic update: minimize 0.5*(v-target)^2
                critic.ZeroGrad();
                var gradV = (v - target); // dLoss/dv
                critic.Backward(new[] { gradV });
                critic.AdamStep(opt.LrCritic);

                // --- Actor update: maximize logProb * advantage + entropyBeta * entropy
                actor.ZeroGrad();

                var invVar = 1.0f / (std * std + 1e-8f);
                var diff = (action - mean);

                // d logProb / d mean = (a-mean)/std^2
                var dLogP_dMean = diff * invVar;

                // d logProb / d logStd = ((a-mean)^2)/std^2 - 1
                var dLogP_dLogStd = (diff * diff) * invVar - 1.0f;

                // We do gradient descent on actorLoss = -(logP*adv + beta*entropy).
                // entropy w.r.t logStd derivative is +1.
                var gradMean = -advantage * dLogP_dMean;
                var gradLogStd = -advantage * dLogP_dLogStd - opt.EntropyBeta;

                actor.Backward(new[] { gradMean, gradLogStd });
                actor.AdamStep(opt.LrActor);

                sumReward += reward;
                sumAbsPctErr += absPctErr;
                steps++;
            }

            Console.WriteLine($"epoch {epoch}/{opt.Epochs} | steps={steps} | avgReward={(sumReward / Math.Max(1, steps)):F6} | avgAbsPctErr={(sumAbsPctErr / Math.Max(1, steps)):F6}");
        }

        // Evaluate (deterministic: use mean)
        var predPath = Path.Combine(AppContext.BaseDirectory, "predictions.csv");
        using var sw = new StreamWriter(predPath);
        sw.WriteLine("Index,CurrentClose,NextClose,PredictedClose,AbsPctErr");

        envTest.Reset();
        double testMae = 0;
        double testMape = 0;
        int n = 0;

        while (!envTest.Done)
        {
            var (state, currentClose, nextClose) = envTest.Observe();
            var aOut = actor.Forward(state);
            var mean = aOut[0];
            mean = Math.Clamp(mean, -opt.MaxAbsReturn, opt.MaxAbsReturn);

            var predicted = currentClose * (1.0f + mean);
            var absErr = Math.Abs(predicted - nextClose);
            var absPctErr = Math.Abs((predicted - nextClose) / Math.Max(1e-6f, currentClose));

            sw.WriteLine($"{envTest.GlobalIndex},{currentClose.ToString(CultureInfo.InvariantCulture)},{nextClose.ToString(CultureInfo.InvariantCulture)},{predicted.ToString(CultureInfo.InvariantCulture)},{absPctErr.ToString(CultureInfo.InvariantCulture)}");

            testMae += absErr;
            testMape += absPctErr;
            n++;

            envTest.Step();
        }

        Console.WriteLine($"test | n={n} | MAE={(testMae / Math.Max(1, n)):F6} | MAPE={(testMape / Math.Max(1, n)):F6}");
        Console.WriteLine($"wrote: {predPath}");

        return 0;
    }

    private static double Randn(Random rng)
    {
        // Box-Muller
        var u1 = Math.Max(1e-12, rng.NextDouble());
        var u2 = Math.Max(1e-12, rng.NextDouble());
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private sealed record Options(
        string CsvPath,
        string TargetColumn,
        int Epochs,
        int Hidden,
        int Lookback,
        float Gamma,
        float LrActor,
        float LrCritic,
        float EntropyBeta,
        float MaxAbsReturn,
        float MinLogStd,
        float MaxLogStd,
        float TrainFraction,
        int Seed)
    {
        public static Options Parse(string[] args)
        {
            string? csv = null;
            string target = "Close";
            int epochs = 25;
            int hidden = 64;
            int lookback = 10;
            float gamma = 0.99f;
            float lrActor = 2e-4f;
            float lrCritic = 1e-3f;
            float entropy = 1e-3f;
            float maxAbsReturn = 0.10f;
            float minLogStd = -4.0f;
            float maxLogStd = 1.0f;
            float trainFraction = 0.8f;
            int seed = 123;

            for (int i = 0; i < args.Length; i++)
            {
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"Missing value for {args[i]}");

                switch (args[i])
                {
                    case "--csv": csv = Next(); break;
                    case "--target": target = Next(); break;
                    case "--epochs": epochs = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--hidden": hidden = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--lookback": lookback = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--gamma": gamma = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--lr-actor": lrActor = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--lr-critic": lrCritic = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--entropy": entropy = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-abs-return": maxAbsReturn = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--min-log-std": minLogStd = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-log-std": maxLogStd = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--train-frac": trainFraction = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--seed": seed = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                }
            }

            csv ??= GuessDefaultCsvPath();

            return new Options(
                CsvPath: csv,
                TargetColumn: target,
                Epochs: epochs,
                Hidden: hidden,
                Lookback: lookback,
                Gamma: gamma,
                LrActor: lrActor,
                LrCritic: lrCritic,
                EntropyBeta: entropy,
                MaxAbsReturn: maxAbsReturn,
                MinLogStd: minLogStd,
                MaxLogStd: maxLogStd,
                TrainFraction: trainFraction,
                Seed: seed);
        }

        private static string GuessDefaultCsvPath()
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var candidates = new[]
            {
                Path.Combine(Directory.GetCurrentDirectory(), "ten.csv"),
                Path.Combine(home, "Downloads", "ten.csv"),
                Path.Combine(home, "downloads", "ten.csv"),
                Path.Combine(home, "Download", "ten.csv"),
                Path.Combine(home, "download", "ten.csv"),
            };

            foreach (var p in candidates)
                if (File.Exists(p)) return p;

            // default to cwd (helpful error message will show)
            return candidates[0];
        }
    }
}
