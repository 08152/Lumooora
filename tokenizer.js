```js
// tokenizer.js
// Eigener Tokenizer für Lumora

function normalize(text) {
    return String(text || "")
        .normalize("NFKC")
        .toLowerCase()
        .trim();
}

function tokenize(text) {
    const input = normalize(text);

    return input.match(
        /https?:\/\/[^\s]+|www\.[^\s]+|[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu
    ) || [];
}

function words(text) {
    return tokenize(text).filter(token =>
        /[\p{L}\p{N}]/u.test(token)
    );
}

function containsWord(text, word) {
    return words(text).includes(normalize(word));
}

module.exports = {
    normalize,
    tokenize,
    words,
    containsWord
};
```
