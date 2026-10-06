const fs = require("fs");
const path = require("path");

/*
    LUMORA TRANSFORMER
    Zielgröße: ca. 750 Mio. Parameter

    Architektur:
    vocab size   = 32000
    hidden size  = 1536
    layers       = 26
    heads        = 24
    head dim     = 64
    FFN          = 3840
    context      = 2048

    WICHTIG:
    Dieser Code baut die Modellstruktur und lädt DATEN/*.json.
    Er erzeugt NICHT automatisch trainierte 750M-Gewichte.
*/

const CONFIG = {
    vocabSize: 32000,
    hiddenSize: 1536,
    layers: 26,
    heads: 24,
    headDim: 64,
    ffnSize: 3840,
    contextLength: 2048
};


/* =========================================
   750M PARAMETER BERECHNUNG
========================================= */

function calculateParameters(config) {

    const {
        vocabSize,
        hiddenSize,
        layers,
        ffnSize
    } = config;

    // Token Embedding
    const embedding = vocabSize * hiddenSize;

    // Attention:
    // Q + K + V + Output
    const attentionPerLayer =
        4 * hiddenSize * hiddenSize;

    // SwiGLU:
    // gate + up + down
    const ffnPerLayer =
        3 * hiddenSize * ffnSize;

    // 2 LayerNorms pro Block
    const layerNormPerLayer =
        4 * hiddenSize;

    const transformerLayers =
        layers *
        (
            attentionPerLayer +
            ffnPerLayer +
            layerNormPerLayer
        );

    // Final LayerNorm
    const finalNorm = hiddenSize * 2;

    // Output-Layer wird mit Embedding geteilt
    const output = 0;

    return (
        embedding +
        transformerLayers +
        finalNorm +
        output
    );
}


/* =========================================
   DATEN LADEN
========================================= */

function findJsonFiles(directory) {

    const result = [];

    if (!fs.existsSync(directory)) {
        return result;
    }

    const entries = fs.readdirSync(
        directory,
        {
            withFileTypes: true
        }
    );

    for (const entry of entries) {

        const fullPath =
            path.join(
                directory,
                entry.name
            );

        if (entry.isDirectory()) {

            result.push(
                ...findJsonFiles(fullPath)
            );

        } else if (
            entry.isFile() &&
            entry.name.toLowerCase().endsWith(".json")
        ) {

            result.push(fullPath);
        }
    }

    return result;
}


/* =========================================
   JSON NORMALISIEREN
========================================= */

function extractText(value, output) {

    if (typeof value === "string") {

        if (value.trim()) {
            output.push(value.trim());
        }

        return;
    }


    if (Array.isArray(value)) {

        for (const item of value) {
            extractText(item, output);
        }

        return;
    }


    if (
        value &&
        typeof value === "object"
    ) {

        /*
            Typische Trainingsdaten:

            {
                "text": "Hallo",
                "prompt": "...",
                "response": "..."
            }
        */

        const preferredKeys = [
            "text",
            "content",
            "prompt",
            "response",
            "answer",
            "question",
            "instruction",
            "output",
            "input"
        ];

        let foundPreferred = false;

        for (const key of preferredKeys) {

            if (
                Object.prototype.hasOwnProperty.call(
                    value,
                    key
                )
            ) {

                foundPreferred = true;

                extractText(
                    value[key],
                    output
                );
            }
        }


        /*
            Falls keine bekannten Felder vorhanden
            sind: alle Werte durchsuchen.
        */

        if (!foundPreferred) {

            for (const key of Object.keys(value)) {

                extractText(
                    value[key],
                    output
                );
            }
        }
    }
}


/* =========================================
   GESAMTEN DATENSATZ LADEN
========================================= */

function loadDataset() {

    const dataDirectory =
        path.join(
            __dirname,
            "DATEN"
        );

    const files =
        findJsonFiles(
            dataDirectory
        );

    const texts = [];

    const errors = [];

    for (const file of files) {

        try {

            const raw =
                fs.readFileSync(
                    file,
                    "utf8"
                );

            const json =
                JSON.parse(raw);

            extractText(
                json,
                texts
            );

        } catch (error) {

            errors.push({
                file,
                error: error.message
            });

        }
    }


    return {
        directory: dataDirectory,
        files,
        texts,
        errors
    };
}


/* =========================================
   TRAINING-BEISPIELE ERSTELLEN
========================================= */

function createTrainingExamples(texts) {

    const examples = [];

    for (const text of texts) {

        if (!text || text.length < 2) {
            continue;
        }

        examples.push({
            text,
            length: text.length
        });
    }

    return examples;
}


/* =========================================
   TOKEN-STATISTIK
========================================= */

function buildTokenStatistics(
    texts,
    tokenizer
) {

    const frequencies =
        new Map();

    let totalTokens = 0;

    for (const text of texts) {

        const tokens =
            tokenizer.tokenize(text);

        totalTokens += tokens.length;

        for (const token of tokens) {

            const old =
                frequencies.get(token) || 0;

            frequencies.set(
                token,
                old + 1
            );
        }
    }


    const vocabulary =
        [...frequencies.entries()]
            .sort(
                (a, b) => b[1] - a[1]
            )
            .map(
                ([token]) => token
            );


    return {
        totalTokens,
        uniqueTokens: vocabulary.length,
        vocabulary
    };
}


/* =========================================
   MODEL-INFORMATION
========================================= */

function getModelInfo() {

    const parameters =
        calculateParameters(
            CONFIG
        );

    return {
        name: "Lumora Transformer 750M",
        type: "decoder-only transformer",
        config: CONFIG,
        parameters,
        parametersMillions:
            parameters / 1_000_000
    };
}


/* =========================================
   INITIALISIERUNG
========================================= */

function initialize(tokenizer) {

    const dataset =
        loadDataset();

    const trainingExamples =
        createTrainingExamples(
            dataset.texts
        );

    const statistics =
        buildTokenStatistics(
            dataset.texts,
            tokenizer
        );

    const model =
        getModelInfo();


    console.log("");
    console.log("====================================");
    console.log("       LUMORA TRANSFORMER");
    console.log("====================================");
    console.log(
        "Modell:",
        model.name
    );
    console.log(
        "Parameter:",
        model.parametersMillions.toFixed(1),
        "M"
    );
    console.log(
        "JSON-Dateien:",
        dataset.files.length
    );
    console.log(
        "Texte:",
        dataset.texts.length
    );
    console.log(
        "Tokens:",
        statistics.totalTokens
    );
    console.log(
        "Einzigartige Tokens:",
        statistics.uniqueTokens
    );
    console.log("====================================");
    console.log("");

    if (dataset.errors.length > 0) {

        console.log(
            "JSON-Fehler:",
            dataset.errors
        );
    }


    return {
        model,
        dataset,
        trainingExamples,
        statistics
    };
}


module.exports = {
    CONFIG,
    calculateParameters,
    findJsonFiles,
    loadDataset,
    createTrainingExamples,
    buildTokenStatistics,
    getModelInfo,
    initialize
};
