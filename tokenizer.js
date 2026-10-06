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
    const tokenList = words(text);
    const wanted = normalize(word);

    return tokenList.includes(wanted);
}

function countTokens(text) {
    return tokenize(text).length;
}

function countWords(text) {
    return words(text).length;
}

module.exports = {
    normalize,
    tokenize,
    words,
    containsWord,
    countTokens,
    countWords
};
