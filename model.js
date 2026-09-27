// model.js
// Einfaches, kostenloses "KI"-Modell im Browser, das aus DATEN/ lernt.

window.myModel = (function () {
  // Interner Speicher
  let nGram = {};      // z.B. { "ich bin": { "müde": 2, "glücklich": 1 } }
  const N = 2;         // N-Gramm-Größe (2 = Bigram-Kontext)
  const MAX_TOKENS = 30;

  // --- Tokenizer ---
  function tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-zäöüß0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);
  }

  // --- Training aus Text ---
  function trainOnText(text) {
    const tokens = tokenize(text);
    if (tokens.length <= N) return;

    for (let i = 0; i <= tokens.length - N - 1; i++) {
      const context = tokens.slice(i, i + N).join(' ');
      const next = tokens[i + N];

      if (!nGram[context]) {
        nGram[context] = {};
      }
      if (!nGram[context][next]) {
        nGram[context][next] = 0;
      }
      nGram[context][next] += 1;
    }
  }

  // --- Hilfsfunktion: zufälliges Wort aus Verteilung wählen ---
  function sampleNext(context) {
    const options = nGram[context];
    if (!options) return null;

    const entries = Object.entries(options); // [ [word, count], ... ]
    const total = entries.reduce((sum, [, c]) => sum + c, 0);
    let r = Math.random() * total;

    for (const [word, count] of entries) {
      if (r < count) return word;
      r -= count;
    }
    return entries[entries.length - 1][0];
  }

  // --- Antwort generieren ---
  function generateReply(prompt) {
    const promptTokens = tokenize(prompt);
    let contextTokens;

    if (promptTokens.length >= N) {
      contextTokens = promptTokens.slice(-N);
    } else {
      // Falls der Prompt zu kurz ist, nimm irgendeinen Kontext aus dem Modell
      const keys = Object.keys(nGram);
      if (keys.length === 0) {
        return "Ich habe noch keine Daten gelernt.";
      }
      const randomKey = keys[Math.floor(Math.random() * keys.length)];
      contextTokens = randomKey.split(' ');
    }

    let resultTokens = [...contextTokens];

    for (let i = 0; i < MAX_TOKENS; i++) {
      const context = resultTokens.slice(-N).join(' ');
      const next = sampleNext(context);
      if (!next) break;
      resultTokens.push(next);
    }

    // Entferne die Kontext-Wörter am Anfang, damit die Antwort nicht nur den Prompt wiederholt
    const answerTokens = resultTokens.slice(N);
    if (answerTokens.length === 0) {
      return "Ich weiß darauf gerade nichts zu sagen.";
    }

    return answerTokens.join(' ');
  }

  // --- Daten aus DATEN/ laden ---
  async function loadData() {
    // Hier kannst du beliebig viele Dateien eintragen
    const files = [
      'DATEN/corpus1.txt',
      'DATEN/corpus2.txt'
      // weitere Dateien möglich
    ];

    for (const file of files) {
      try {
        const res = await fetch(file);
        if (!res.ok) {
          console.warn('Konnte Datei nicht laden:', file);
          continue;
        }
        const text = await res.text();
        trainOnText(text);
      } catch (e) {
        console.error('Fehler beim Laden von', file, e);
      }
    }
  }

  // --- Öffentliche API ---
  return {
    async init() {
      await loadData();
      console.log('Modell geladen. Anzahl Kontexte:', Object.keys(nGram).length);
    },
    async generate(prompt) {
      return generateReply(prompt);
    }
  };
})();
