// Voyag V8 CLI for the .NET binding.
//
// Usage:
//   dotnet run --project Voyag.csproj -- <model-dir> "打开车窗，同时把空调调到24度"
using System.Text.Json;
using Voyag;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: dotnet run -- <model-dir> <utterance>");
    return 2;
}

var modelDir = Path.GetFullPath(args[0]);
var utterance = args[1];

using var predictor = VoyagPredictor.Create(modelDir);
Console.WriteLine($"device: {predictor.Device}");
var result = predictor.Predict(utterance);
Console.WriteLine(JsonSerializer.Serialize(result,
    new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
return 0;
