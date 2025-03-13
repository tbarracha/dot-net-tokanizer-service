using Microsoft.ML.Tokenizers;
using System.Collections.Concurrent;

public class TokenizerService
{
    private readonly ILogger<TokenizerService> _logger;
    private readonly ConcurrentDictionary<string, Tokenizer> _tokenizerCache;

    private const string DataFolder = "Data/Tokenizer";

    private static readonly Dictionary<string, TokenizerConfig> TokenizerMap = new()
    {
        {
            "gpt4", new TokenizerConfig(
                name: "gpt4",
                extension: "tiktoken",
                url: "https://huggingface.co/microsoft/Phi-3-small-8k-instruct/resolve/main/cl100k_base.tiktoken"
            )
        },
        {
            "llama", new TokenizerConfig(
                name: "llama",
                extension: "model",
                url: "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model"
            )
        }
    };

    public TokenizerService(ILogger<TokenizerService> logger)
    {
        _logger = logger;
        _tokenizerCache = new ConcurrentDictionary<string, Tokenizer>();

        _logger.LogInformation("Created tokenizers at {Path}", DataFolder);

        // Ensure the Data/Tokenizer directory exists
        if (!Directory.Exists(DataFolder))
        {
            Directory.CreateDirectory(DataFolder);
            _logger.LogInformation("Created directory for tokenizers at {Path}", DataFolder);
        }
    }

    private async Task<Tokenizer> LoadTokenizerAsync(string tokenizerName)
    {
        tokenizerName = tokenizerName.ToLowerInvariant();

        if (!TokenizerMap.ContainsKey(tokenizerName))
        {
            _logger.LogWarning("Unsupported tokenizer requested: {TokenizerName}. Defaulting to 'gpt4'.", tokenizerName);
            tokenizerName = "gpt4";
        }

        if (_tokenizerCache.TryGetValue(tokenizerName, out var cachedTokenizer))
        {
            return cachedTokenizer;
        }

        var tokenizerConfig = TokenizerMap[tokenizerName];
        string cachePath = FindTokenizerFile(tokenizerName) ?? Path.Combine(DataFolder, $"{tokenizerConfig.Name}.{tokenizerConfig.Extension}");

        Tokenizer tokenizer = await LoadTokenizerFromUrl(tokenizerConfig.Url, cachePath, tokenizerConfig.Name);
        _tokenizerCache[tokenizerName] = tokenizer;
        return tokenizer;
    }

    private async Task<Tokenizer> LoadTokenizerFromUrl(string modelUrl, string cachePath, string tokenizerName)
    {
        if (!File.Exists(cachePath))
        {
            _logger.LogInformation("Downloading tokenizer '{TokenizerName}' from {ModelUrl}", tokenizerName, modelUrl);
            using HttpClient httpClient = new();
            using Stream remoteStream = await httpClient.GetStreamAsync(modelUrl);
            using FileStream localStream = File.Create(cachePath);
            await remoteStream.CopyToAsync(localStream);
            _logger.LogInformation("Tokenizer '{TokenizerName}' downloaded and saved to {CachePath}", tokenizerName, cachePath);
        }

        _logger.LogInformation("Loading tokenizer '{TokenizerName}' from {CachePath}", tokenizerName, cachePath);
        using FileStream tokenizerStream = File.OpenRead(cachePath);

        return tokenizerName == "llama"
            ? LlamaTokenizer.Create(tokenizerStream)
            : await TiktokenTokenizer.CreateAsync(tokenizerStream, null, null);
    }

    private string? FindTokenizerFile(string tokenizerName)
    {
        var files = Directory.GetFiles(DataFolder, $"{tokenizerName}.*");
        return files.Length > 0 ? files[0] : null;
    }

    public async Task<int> CountTokensAsync(string text, string tokenizerName = "gpt4")
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        var tokenizer = await LoadTokenizerAsync(tokenizerName);
        return tokenizer.CountTokens(text);
    }

    public List<string> ListAvailableTokenizers()
    {
        return TokenizerMap.Keys.ToList();
    }
}

public class TokenizerConfig
{
    public string Name { get; }
    public string Extension { get; }
    public string Url { get; }

    public TokenizerConfig(string name, string extension, string url)
    {
        Name = name;
        Extension = extension;
        Url = url;
    }
}
