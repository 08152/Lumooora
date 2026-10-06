const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 10000;

app.use(express.json({ limit: "2mb" }));

// index.html liegt direkt im selben Ordner
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"));
});

/* =========================================================
   TOKENIZER
========================================================= */

function normalizeText(text) {
    return String(text || "")
        .normalize("NFKC")
        .toLowerCase();
}

function tokenize(text) {
    const clean = normalizeText(text);

    return clean.match(
        /https?:\/\/[^\s]+|www\.[^\s]+|[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu
    ) || [];
}

/* =========================================================
   DEUTSCH -> ENGLISCH
========================================================= */

const deToEn = {
    ich: "i",
    du: "you",
    er: "he",
    sie: "she",
    es: "it",
    wir: "we",
    ihr: "you",

    hallo: "hello",
    tschüss: "bye",
    danke: "thank you",
    bitte: "please",

    ja: "yes",
    nein: "no",

    der: "the",
    die: "the",
    das: "the",

    ein: "a",
    eine: "a",

    ist: "is",
    sind: "are",
    war: "was",

    bin: "am",
    bist: "are",

    haben: "have",
    habe: "have",
    hast: "have",
    hat: "has",

    können: "can",
    kann: "can",

    wollen: "want",
    will: "want",

    machen: "make",
    macht: "makes",

    spielen: "play",
    spiel: "game",

    schule: "school",
    freund: "friend",
    freunde: "friends",

    welt: "world",
    haus: "house",
    auto: "car",

    wasser: "water",
    essen: "food",
    musik: "music",

    heute: "today",
    morgen: "tomorrow",
    gestern: "yesterday",

    gut: "good",
    schlecht: "bad",
    groß: "big",
    klein: "small",
    neu: "new",
    alt: "old",

    sehr: "very",
    nicht: "not",

    was: "what",
    wer: "who",
    wo: "where",
    wann: "when",
    warum: "why",
    wie: "how"
};

/* =========================================================
   ENGLISCH -> DEUTSCH
========================================================= */

const enToDe = {
    i: "ich",
    you: "du",
    he: "er",
    she: "sie",
    it: "es",
    we: "wir",
    they: "sie",

    hello: "hallo",
    bye: "tschüss",
    thanks: "danke",
    "thank": "danke",
    please: "bitte",

    yes: "ja",
    no: "nein",

    the: "der",
    a: "ein",

    is: "ist",
    are: "sind",
    was: "war",

    am: "bin",

    have: "haben",
    has: "hat",

    can: "kann",

    want: "wollen",

    make: "machen",
    makes: "macht",

    play: "spielen",
    game: "spiel",

    school: "schule",
    friend: "freund",
    friends: "freunde",

    world: "welt",
    house: "haus",
    car: "auto",

    water: "wasser",
    food: "essen",
    music: "musik",

    today: "heute",
    tomorrow: "morgen",
    yesterday: "gestern",

    good: "gut",
    bad: "schlecht",
    big: "groß",
    small: "klein",
    new: "neu",
    old: "alt",

    very: "sehr",
    not: "nicht",

    what: "was",
    who: "wer",
    where: "wo",
    when: "wann",
    why: "warum",
    how: "wie"
};

/* =========================================================
   PHRASEN
========================================================= */

const phrasesDeToEn = {
    "guten morgen": "good morning",
    "guten tag": "good afternoon",
    "gute nacht": "good night",
    "wie geht es dir": "how are you",
    "wie geht's dir": "how are you",
    "was machst du": "what are you doing",
    "wer bist du": "who are you",
    "wie heißt du": "what is your name",
    "ich verstehe": "i understand",
    "ich verstehe das nicht": "i do not understand",
    "bis später": "see you later",
    "auf wiedersehen": "goodbye"
};

const phrasesEnToDe = {
    "good morning": "guten morgen",
    "good afternoon": "guten tag",
    "good night": "gute nacht",
    "how are you": "wie geht es dir",
    "what are you doing": "was machst du",
    "who are you": "wer bist du",
    "what is your name": "wie heißt du",
    "i understand": "ich verstehe",
    "i do not understand": "ich verstehe das nicht",
    "see you later": "bis später",
    "goodbye": "auf wiedersehen"
};

function translate(text, from, to) {
    const original = String(text || "").trim();

    if (!original) {
        return "";
    }

    if (from === to) {
        return original;
    }

    const normalized = original.toLowerCase();

    if (from === "de" && to === "en") {
        if (phrasesDeToEn[normalized]) {
            return phrasesDeToEn[normalized];
        }
    }

    if (from === "en" && to === "de") {
        if (phrasesEnToDe[normalized]) {
            return phrasesEnToDe[normalized];
        }
    }

    const dictionary =
        from === "de" && to === "en"
            ? deToEn
            : enToDe;

    const parts = tokenize(original);

    return parts
        .map((part, index) => {
            const key = part.toLowerCase();
            const result = dictionary[key];

            if (!result) {
                return part;
            }

            // Großschreibung erhalten
            if (part[0] === part[0].toUpperCase()) {
                return result.charAt(0).toUpperCase() + result.slice(1);
            }

            return result;
        })
        .reduce((output, current, index) => {
            if (index === 0) {
                return current;
            }

            if (/^[,.!?;:%)]$/.test(current)) {
                return output + current;
            }

            return output + " " + current;
        }, "");
}

/* =========================================================
   MINI-KI
========================================================= */

const intents = [
    {
        name: "greeting",
        examples: [
            "hallo",
            "hi",
            "hey",
            "guten morgen",
            "guten tag",
            "servus"
        ],
        answers: [
            "Hallo! Ich bin Lumora.",
            "Hey! Lumora ist bereit.",
            "Hallo 👋 Was möchtest du wissen?"
        ]
    },

    {
        name: "identity",
        examples: [
            "wer bist du",
            "was bist du",
            "wie heißt du",
            "bist du eine ki",
            "was ist lumora"
        ],
        answers: [
            "Ich bin Lumora, eine kleine eigene KI in JavaScript.",
            "Ich bin Lumora. Meine Logik läuft auf deinem eigenen Server.",
            "Ich bin eine selbst gebaute JavaScript-KI."
        ]
    },

    {
        name: "tokenizer",
        examples: [
            "was ist ein tokenizer",
            "erkläre tokenizer",
            "wie funktioniert tokenisierung",
            "tokenizer"
        ],
        answers: [
            "Ein Tokenizer zerlegt Text in einzelne Tokens wie Wörter, Zahlen und Satzzeichen.",
            "Ich besitze einen eigenen JavaScript-Tokenizer."
        ]
    },

    {
        name: "translator",
        examples: [
            "kannst du übersetzen",
            "übersetzer",
            "übersetzung",
            "kannst du englisch",
            "kannst du deutsch"
        ],
        answers: [
            "Ja. Ich habe einen lokalen Deutsch-Englisch-Übersetzer.",
            "Mein Übersetzer funktioniert ohne externe KI-API."
        ]
    },

    {
        name: "javascript",
        examples: [
            "was ist javascript",
            "was ist js",
            "erkläre javascript",
            "was kann javascript"
        ],
        answers: [
            "JavaScript ist eine Programmiersprache für Webseiten und Server.",
            "Mit JavaScript kannst du Webseiten, Apps und Server programmieren."
        ]
    },

    {
        name: "render",
        examples: [
            "was ist render",
            "kann ich dich auf render hosten",
            "render deployment",
            "funktioniert das auf render"
        ],
        answers: [
            "Ja. Dieses Projekt ist für Render vorbereitet.",
            "Du kannst Lumora als Node.js Web Service auf Render starten."
        ]
    },

    {
        name: "thanks",
        examples: [
            "danke",
            "vielen dank",
            "dankeschön"
        ],
        answers: [
            "Gerne!",
            "Kein Problem!"
        ]
    }
];

/* =========================================================
   EINFACHE VECTORIZATION
========================================================= */

const vocabulary = [
    ...new Set(
        intents.flatMap(intent =>
            intent.examples.flatMap(example =>
                tokenize(example).filter(x => /\p{L}|\p{N}/u.test(x))
            )
        )
    )
];

function vectorize(text) {
    const tokens = tokenize(text)
        .filter(x => /\p{L}|\p{N}/u.test(x));

    const counts = {};

    for (const token of tokens) {
        counts[token] = (counts[token] || 0) + 1;
    }

    return vocabulary.map(word => counts[word] || 0);
}

function cosineSimilarity(a, b) {
    let dot = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }

    if (magA === 0 || magB === 0) {
        return 0;
    }

    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/* =========================================================
   KI ANTWORT
========================================================= */

function generateAnswer(message) {
    const inputVector = vectorize(message);

    let bestIntent = null;
    let bestScore = 0;

    for (const intent of intents) {
        for (const example of intent.examples) {
            const exampleVector = vectorize(example);
            const score = cosineSimilarity(
                inputVector,
                exampleVector
            );

            if (score > bestScore) {
                bestScore = score;
                bestIntent = intent;
            }
        }
    }

    if (!bestIntent || bestScore < 0.28) {
        return {
            answer:
                "Das weiß ich momentan noch nicht. Ich bin eine kleine selbst gebaute KI und kann später mit mehr Trainingsdaten erweitert werden.",
            confidence: bestScore
        };
    }

    const randomAnswer =
        bestIntent.answers[
            Math.floor(Math.random() * bestIntent.answers.length)
        ];

    return {
        answer: randomAnswer,
        confidence: bestScore,
        intent: bestIntent.name
    };
}

/* =========================================================
   API
========================================================= */

app.get("/api/health", (req, res) => {
    res.json({
        online: true,
        ai: "Lumora",
        tokenizer: true,
        translator: true,
        vocabulary: vocabulary.length,
        intents: intents.length
    });
});

app.post("/api/chat", (req, res) => {
    const message = req.body?.message || "";

    res.json(
        generateAnswer(message)
    );
});

app.post("/api/tokenize", (req, res) => {
    const text = req.body?.text || "";

    res.json({
        text,
        tokens: tokenize(text)
    });
});

app.post("/api/translate", (req, res) => {
    const text = req.body?.text || "";
    const from = req.body?.from || "de";
    const to = req.body?.to || "en";

    res.json({
        text,
        from,
        to,
        translation: translate(text, from, to)
    });
});

app.listen(PORT, "0.0.0.0", () => {
    console.log("=================================");
    console.log("       LUMORA JS KI");
    console.log("=================================");
    console.log("Server läuft auf Port:", PORT);
    console.log("Tokenizer: AKTIV");
    console.log("Übersetzer: AKTIV");
    console.log("KI: AKTIV");
});
